"""
Long-context text extraction branch for text-native PDFs.

This path is for documents that are too long or too table-heavy for page-image
VLM prompting, such as bank statements and tax computations. It extracts text
with PyMuPDF, sends the compacted text to a long-context text model, and then
normalizes the returned JSON through the same parser used by the VLM pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Optional

try:
    from .crypto import generate_deletion_cert, zero_bytes
    from .extractor import _best_confidence
    from .models import (
        ConfidenceWrapper,
        DocumentExtractionResult,
        PageExtractionResult,
        PageType,
        SourceQuality,
    )
    from .parser import ExtractionParser
except ImportError:
    from crypto import generate_deletion_cert, zero_bytes
    from extractor import _best_confidence
    from models import (
        ConfidenceWrapper,
        DocumentExtractionResult,
        PageExtractionResult,
        PageType,
        SourceQuality,
    )
    from parser import ExtractionParser


TEXT_SYSTEM_PROMPT = """You are Uplan's long-context financial document extraction engine.
Return only one valid JSON object. Do not include markdown fences, comments,
or explanations. Do not infer values that are not in the text. Preserve names,
account numbers, dates, and monetary values exactly as printed, normalizing
monetary amounts to numbers."""


TEXT_PROMPTS: dict[PageType, str] = {
    PageType.BANK_STATEMENT: """Extract this bank statement into JSON:
{
  "account_holder_name": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "statement_period_start": "<YYYY-MM or null>",
  "statement_period_end": "<YYYY-MM or null>",
  "opening_balance": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "closing_balance": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "monthly_closing_balances": [
    {"month": "<YYYY-MM>", "closing_balance": <number>, "confidence": "<high|medium|low>"}
  ],
  "deposits": [
    {"date": "<YYYY-MM-DD or null>", "amount": <number>, "description": "<string or null>", "confidence": "<high|medium|low>"}
  ],
  "source_quality": "digital"
}

Rules:
- Use the running balance/closing balance column, not withdrawal/deposit columns,
  for monthly_closing_balances.
- Return one closing balance per statement month when visible.
- Deposits should include meaningful credits, salary, cash deposits, transfers,
  and unusually large credits. Do not include interest or tiny bank adjustments
  unless they are the only deposits present.
- If the PDF contains many pages, use all supplied text and summarize into the
  schema above instead of copying transaction tables.
""",
    PageType.TAX_RETURN: """Extract this tax return or income tax computation into JSON:
{
  "taxpayer_name": "<string or null>",
  "tax_year": <integer or null>,
  "currency_code": "<ISO 4217 code or null>",
  "gross_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "taxable_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "tax_paid": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "income_sources": ["<string>", "..."],
  "document_issuer": "<string or null>",
  "source_quality": "digital"
}

Rules:
- Prefer total income / gross total income / taxable income from the computation
  summary, not incidental figures from schedules.
- For Indian ITR/computation documents, currency_code is INR unless another
  currency is explicitly printed.
- tax_year should be the assessment or filing year most clearly tied to the
  return. If both assessment year and financial year appear, use the assessment
  year as an integer.
""",
    PageType.BANK_BALANCE_CERTIFICATE: """Extract this bank balance certificate into JSON:
{
  "account_holder_name": "<string or null>",
  "institution_name": "<string or null>",
  "account_number": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "certificate_date": "<YYYY-MM-DD or null>",
  "balances": [
    {
      "account_type": "<savings|fixed_deposit|term_deposit|current|other>",
      "amount": {"value": <number or null>, "confidence": "<high|medium|low>"},
      "description": "<string or null>"
    }
  ],
  "available_balance": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "exchange_rate": {"value": <number or null>, "currency_pair": "<string or null>", "confidence": "<high|medium|low>"},
  "held_minimum_days": <integer or null>,
  "source_quality": "digital"
}

Rules:
- Extract every separately listed balance. If text says there is one amount in
  savings and another in fixed deposits, return both under balances.
- Do not put fixed-deposit balances into available_balance unless the printed
  document explicitly gives a total available balance.
- Do not extract account numbers or exchange rates as balances.
""",
}


class HFTextLLMBackend:
    """Minimal Hugging Face text-generation backend for long-context models."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **kwargs,
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HFTextLLMBackend.load() must be called first.")

        messages = [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{TEXT_SYSTEM_PROMPT}\n\n{prompt}\n\nJSON:"

        inputs = self._tokenizer([text], return_tensors="pt")
        target_device = next(self._model.parameters()).device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
        output = self._model.generate(**generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class LongContextTextExtractor:
    """Extract a whole text-native PDF into a DocumentExtractionResult."""

    def __init__(
        self,
        llm: HFTextLLMBackend,
        doc_type: PageType,
        max_input_chars: int = 160_000,
        page_start: int = 1,
        page_end: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> None:
        if doc_type not in TEXT_PROMPTS:
            raise ValueError(f"Text extraction is not configured for {doc_type.value}.")
        self.llm = llm
        self.doc_type = doc_type
        self.max_input_chars = max_input_chars
        self.page_start = page_start
        self.page_end = page_end
        self.max_pages = max_pages
        self.parser = ExtractionParser()
        self.debug_events: list[dict[str, Any]] = []

    def extract(self, pdf_path: str | Path, session_id: str) -> DocumentExtractionResult:
        import fitz

        start_ms = int(time.time() * 1000)
        path = Path(pdf_path)
        raw_bytes = bytearray(path.read_bytes())
        doc_hash = hashlib.sha256(raw_bytes).hexdigest()
        nonce = secrets.token_hex(8)

        pdf_doc = fitz.open(stream=bytes(raw_bytes), filetype="pdf")
        total_pages = pdf_doc.page_count
        selected_pages = self._selected_page_indexes(total_pages)

        chunks: list[str] = []
        for page_index in selected_pages:
            page = pdf_doc[page_index]
            text = page.get_text("text", sort=True)
            if text.strip():
                chunks.append(f"\n--- PAGE {page_index + 1} ---\n{text.strip()}")
        pdf_doc.close()

        zero_bytes(raw_bytes)
        del raw_bytes
        gc.collect()

        extracted_text = "\n".join(chunks).strip()
        if self.max_input_chars and len(extracted_text) > self.max_input_chars:
            extracted_text = extracted_text[: self.max_input_chars]

        raw_response = "{}"
        parsed: dict[str, Any] = {}
        if extracted_text:
            prompt = (
                f"{TEXT_PROMPTS[self.doc_type]}\n\n"
                f"Document text follows. Extract only from this text:\n"
                f"{extracted_text}"
            )
            raw_response = self.llm.generate(prompt)
            parsed = _extract_json_object(raw_response)

        self.debug_events.append({
            "stage": "text_extract",
            "page_type": self.doc_type.value,
            "selected_pages": [idx + 1 for idx in selected_pages],
            "text_chars": len(extracted_text),
            "raw_response": raw_response,
            "parsed_response": parsed,
        })

        page_result = self.parser.parse_page(
            data=parsed,
            page_number=selected_pages[0] + 1 if selected_pages else 1,
            page_type=self.doc_type,
            mean_logprob=-0.05,
        )
        merged = _merge_page_results(
            page_results=[page_result],
            doc_hash=doc_hash,
            session_id=session_id,
            filename=path.name,
            total_pages=total_pages,
            source_quality=SourceQuality.DIGITAL,
            start_ms=start_ms,
        )
        merged.deletion_cert = generate_deletion_cert(
            doc_hash=doc_hash,
            nonce=nonce,
            session_id=session_id,
        )
        merged.raw_purge_confirmed = True
        del extracted_text, chunks
        gc.collect()
        return merged

    def _selected_page_indexes(self, total_pages: int) -> list[int]:
        start = max(1, self.page_start)
        end = min(total_pages, self.page_end or total_pages)
        indexes = list(range(start - 1, end))
        if self.max_pages is not None:
            indexes = indexes[: self.max_pages]
        return indexes


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _merge_page_results(
    page_results: list[PageExtractionResult],
    doc_hash: str,
    session_id: str,
    filename: str,
    total_pages: int,
    source_quality: SourceQuality,
    start_ms: int,
) -> DocumentExtractionResult:
    elapsed_ms = int(time.time() * 1000) - start_ms
    merged = DocumentExtractionResult(
        document_hash=doc_hash,
        session_id=session_id,
        deletion_cert="",
        filename=filename,
        total_pages=total_pages,
        source_quality=source_quality,
        extraction_ms=elapsed_ms,
        pages=page_results,
    )

    for page in page_results:
        merged.balance_series.extend(page.balance_series)
        merged.deposit_entries.extend(page.deposit_entries)

        if merged.currency_code is None and page.currency_code:
            merged.currency_code = page.currency_code

        merged.i_tax = _best_confidence(merged.i_tax, page.i_tax)
        merged.i_form = _best_confidence(merged.i_form, page.i_form)
        merged.i_aff = _best_confidence(merged.i_aff, page.i_aff)
        merged.i_spon = _best_confidence(merged.i_spon, page.i_spon)

        if merged.spon_relationship is None:
            merged.spon_relationship = page.spon_relationship
        if merged.tax_year is None:
            merged.tax_year = page.tax_year
        if page.name_string:
            merged.name_variants[page.page_type.value] = page.name_string

        merged.financial_accounts.extend(page.financial_accounts)

        if merged.declarant_address is None:
            merged.declarant_address = page.declarant_address
        if merged.beneficiary_name is None:
            merged.beneficiary_name = page.beneficiary_name

        merged.family_members.extend(page.family_members)
        merged.income_sources.extend(page.income_sources)
        merged.movable_assets.extend(page.movable_assets)
        merged.properties.extend(page.properties)

    return merged
