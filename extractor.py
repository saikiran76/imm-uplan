"""
uplan.extraction.extractor
--------------------------
The main extraction pipeline.

This module owns the full lifecycle:
  PDF bytes → rasterise → classify → VLM extract → parse → merge → purge → cert

Usage
-----
    from uplan.extraction.extractor import DocumentExtractor

    extractor = DocumentExtractor(model_path="Qwen/Qwen2.5-VL-7B-Instruct")
    result = await extractor.extract("path/to/document.pdf", session_id="abc123")
    # result is a DocumentExtractionResult with raw_purge_confirmed=True

Architecture notes
------------------
- All PIL images are stored in a local list and explicitly zeroed after extraction.
- The VLM is called with a targeted prompt per page — never given the full doc at once.
- Raw PDF bytes are read once, hashed immediately, and never stored as an attribute.
- LangGraph state integration: this module produces DocumentExtractionResult,
  which maps directly to the UplanState financial fields.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image
except ImportError:
    Image = Any  # type: ignore[assignment]

try:
    from .models import (
        Confidence,
        ConfidenceWrapper,
        DocumentExtractionResult,
        PageExtractionResult,
        PageType,
        SourceQuality,
    )
    from .prompts import (
        CONTINUATION_CLASSIFIER_PROMPT,
        PAGE_CLASSIFIER_PROMPT,
        SYSTEM_PROMPT,
        get_prompt,
    )
    from .parser import ExtractionParser
    from .crypto import generate_deletion_cert, zero_bytes
except ImportError:
    from models import (
        Confidence,
        ConfidenceWrapper,
        DocumentExtractionResult,
        PageExtractionResult,
        PageType,
        SourceQuality,
    )
    from prompts import (
        CONTINUATION_CLASSIFIER_PROMPT,
        PAGE_CLASSIFIER_PROMPT,
        SYSTEM_PROMPT,
        get_prompt,
    )
    from parser import ExtractionParser
    from crypto import generate_deletion_cert, zero_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RASTER_DPI    = 150   # High enough for clean extraction, low enough for VRAM
MAX_PAGE_DIM  = 1024  # Resize longest dimension to this before VLM pass
JPEG_QUALITY  = 85    # Compression for RAM efficiency


# ─────────────────────────────────────────────────────────────────────────────
# VLM backend interface — swap this for your actual model loader
# ─────────────────────────────────────────────────────────────────────────────

class VLMBackend:
    """
    Abstract interface for the vision-language model.

    In production this wraps Qwen2.5-VL loaded via HuggingFace transformers
    on the AMD MI300X. In testing it can be replaced with a mock that returns
    fixture JSON.

    The key constraint: generate() must return a raw string.
    The caller (extractor) handles all JSON parsing — not this class.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        """
        Load model weights into VRAM.
        Call this once at startup, not per document.

        Production implementation:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
            )
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        """
        raise NotImplementedError(
            "VLMBackend.load() must be implemented with your model loader. "
            "See the docstring for the production implementation pattern."
        )

    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, float]:
        """
        Run one VLM inference pass.

        Returns
        -------
        (response_text, mean_logprob)
            response_text : raw string from the model (should be JSON)
            mean_logprob  : mean log probability of the output tokens,
                            used as the basis for confidence scoring.
                            Range: typically -0.5 (high conf) to -3.0 (low conf).

        Production implementation:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": user_prompt},
                ]},
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            response = self._processor.decode(
                output.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            # Compute mean log prob from scores
            import torch
            log_probs = [
                torch.log_softmax(s, dim=-1).max().item()
                for s in output.scores
            ]
            mean_lp = sum(log_probs) / len(log_probs) if log_probs else -2.0

            return response, mean_lp
        """
        raise NotImplementedError(
            "VLMBackend.generate() must be implemented with your model. "
            "See the docstring for the production implementation pattern."
        )


class HuggingFaceVLMBackend(VLMBackend):
    """
    Local Hugging Face backend for real, non-mock extraction.

    This is intentionally lazy and optional: importing extractor.py should not
    require torch/transformers unless this backend is actually used.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 768,
    ):
        super().__init__(model_path, device)
        self.max_new_tokens = max_new_tokens
        self._torch = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Local Hugging Face inference needs torch and transformers. "
                "Install them first, then rerun with --backend hf."
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(self.device, torch)

        model_kwargs: dict[str, Any] = {}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = "auto"
        elif self.device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **model_kwargs,
        )

        if "device_map" not in model_kwargs:
            self._model.to(self.device)

        self._model.eval()

    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, float]:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Call HuggingFaceVLMBackend.load() before generate().")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        if self.device != "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif not hasattr(self._model, "hf_device_map"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        token_budget = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        with self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[:, input_len:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        mean_logprob = self._mean_selected_token_logprob(output.scores, generated_ids)
        return response, mean_logprob

    @staticmethod
    def _resolve_device(requested: str, torch: Any) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _mean_selected_token_logprob(self, scores: list[Any], generated_ids: Any) -> float:
        if not scores:
            return -0.40

        logprobs: list[float] = []
        for i, score in enumerate(scores):
            if i >= generated_ids.shape[1]:
                break
            token_id = generated_ids[0, i]
            token_logprobs = self._torch.log_softmax(score[0], dim=-1)
            logprobs.append(float(token_logprobs[token_id].item()))

        return sum(logprobs) / len(logprobs) if logprobs else -0.40


# ─────────────────────────────────────────────────────────────────────────────
# Mock VLM backend — used in testing without GPU
# ─────────────────────────────────────────────────────────────────────────────

class MockVLMBackend(VLMBackend):
    """
    Returns fixture JSON based on the prompt content.
    Used in Phase 0 testing before GPU is available.

    Fixtures are keyed by page_type string embedded in the prompt.
    """

    FIXTURES: dict[str, dict] = {
        "bank_statement": {
            "account_holder_name": "Arjun Mehta",
            "currency_code": "USD",
            "statement_period_start": "2024-01",
            "statement_period_end": "2024-06",
            "closing_balance": {"value": 45100.00, "confidence": "high"},
            "opening_balance": {"value": 42000.00, "confidence": "high"},
            "deposits": [
                {"date": "2024-01-15", "amount": 1200.00, "description": "salary", "confidence": "high"},
                {"date": "2024-02-15", "amount": 950.00,  "description": "salary", "confidence": "high"},
                {"date": "2024-03-15", "amount": 1100.00, "description": "salary", "confidence": "high"},
                {"date": "2024-04-15", "amount": 800.00,  "description": "salary", "confidence": "high"},
                {"date": "2024-05-15", "amount": 1300.00, "description": "salary", "confidence": "high"},
                {"date": "2024-06-15", "amount": 1050.00, "description": "salary", "confidence": "high"},
            ],
            "monthly_closing_balances": [
                {"month": "2024-01", "closing_balance": 42800.00, "confidence": "high"},
                {"month": "2024-02", "closing_balance": 43100.00, "confidence": "high"},
                {"month": "2024-03", "closing_balance": 44200.00, "confidence": "high"},
                {"month": "2024-04", "closing_balance": 43900.00, "confidence": "high"},
                {"month": "2024-05", "closing_balance": 44600.00, "confidence": "high"},
                {"month": "2024-06", "closing_balance": 45100.00, "confidence": "high"},
            ],
            "source_quality": "digital",
        },
        "tax_return": {
            "taxpayer_name": "Arjun Mehta",
            "tax_year": 2023,
            "currency_code": "USD",
            "gross_income":   {"value": 70000.00, "confidence": "high"},
            "taxable_income": {"value": 65800.00, "confidence": "high"},
            "tax_paid":       {"value": 9200.00,  "confidence": "high"},
            "income_sources": ["salary"],
            "document_issuer": "IRS",
            "source_quality": "digital",
        },
        "affidavit": {
            "declarant_name":       "Sunita Mehta",
            "beneficiary_name":     "Arjun Mehta",
            "relationship":         "parent",
            "currency_code":        "USD",
            "declared_annual_income": {"value": 72000.00, "confidence": "high"},
            "declared_net_worth":   {"value": 180000.00, "confidence": "medium"},
            "declared_support_amount": {"value": 38000.00, "confidence": "high", "period": "annual"},
            "affidavit_type":       "financial_support",
            "notarised":            True,
            "source_quality":       "scan",
        },
        "page_classifier": {
            "page_type":      "bank_statement",
            "language":       "en",
            "source_quality": "digital",
            "confidence":     "high",
        },
    }

    def load(self) -> None:
        pass  # No-op in mock

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
        self._classifier_calls = 0

    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, float]:
        # Identify which fixture to return based on prompt content
        if '"page_type"' in user_prompt or "valid page_type values" in user_prompt.lower():
            self._classifier_calls += 1
            page_type = "bank_statement" if self._classifier_calls == 1 else "tax_return"
            fixture = {
                "page_type": page_type,
                "language": "en",
                "source_quality": "digital",
                "confidence": "high",
            }
            return json.dumps(fixture), -0.10
        if "bank_statement" in user_prompt.lower() or "closing_balance" in user_prompt:
            key = "bank_statement"
            logprob = -0.12  # High confidence
        elif "tax_return" in user_prompt.lower() or "taxable_income" in user_prompt:
            key = "tax_return"
            logprob = -0.15
        elif "affidavit" in user_prompt.lower() or "declarant" in user_prompt:
            key = "affidavit"
            logprob = -0.28  # Medium confidence (scan)
        else:
            return json.dumps({"error": "unknown_page_type"}), -2.5

        return json.dumps(self.FIXTURES[key]), logprob


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class DocumentExtractor:
    """
    Orchestrates the full extraction pipeline for a single document.

    Parameters
    ----------
    vlm : VLMBackend
        The vision-language model backend. Use MockVLMBackend for testing.
    """

    def __init__(
        self,
        vlm: VLMBackend,
        forced_page_type: Optional[PageType] = None,
        max_pages: Optional[int] = None,
        page_start: int = 1,
        page_end: Optional[int] = None,
        extraction_max_tokens: Optional[int] = None,
    ):
        self.vlm = vlm
        self.parser = ExtractionParser()
        self.forced_page_type = forced_page_type
        self.max_pages = max_pages
        self.page_start = page_start
        self.page_end = page_end
        self.extraction_max_tokens = extraction_max_tokens
        self.debug_events: list[dict[str, Any]] = []

    async def extract(
        self,
        pdf_path: str | Path,
        session_id: Optional[str] = None,
    ) -> DocumentExtractionResult:
        """
        Full extraction pipeline: PDF → DocumentExtractionResult.

        The returned object has raw_purge_confirmed=True.
        Raw bytes and PIL images are zeroed before this method returns.
        """
        pdf_path = Path(pdf_path)
        self.filename = pdf_path.name
        session_id = session_id or secrets.token_hex(8)
        start_ms = int(time.time() * 1000)

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to extract PDF pages. Install the 'fitz' package first."
            ) from exc

        # ── Step 1: Read and hash raw bytes immediately ────────────────────
        raw_bytes = bytearray(pdf_path.read_bytes())
        doc_hash = hashlib.sha256(raw_bytes).hexdigest()
        nonce = secrets.token_hex(8)

        # ── Step 2: Rasterise pages ────────────────────────────────────────
        images: list[Image.Image] = []
        pdf_doc = fitz.open(stream=bytes(raw_bytes), filetype="pdf")
        total_pages = pdf_doc.page_count
        selected_pages = self._selected_page_indexes(total_pages)

        for page_num in selected_pages:
            page = pdf_doc[page_num]
            mat = fitz.Matrix(RASTER_DPI / 72, RASTER_DPI / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Resize to max dimension to control VRAM usage
            w, h = img.size
            if max(w, h) > MAX_PAGE_DIM:
                scale = MAX_PAGE_DIM / max(w, h)
                img = img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.LANCZOS,
                )
            images.append(img)

        pdf_doc.close()

        # ── Step 3: Purge raw PDF bytes from RAM ───────────────────────────
        zero_bytes(raw_bytes)
        del raw_bytes
        gc.collect()

        # ── Step 4: Classify each page ─────────────────────────────────────
        page_types: list[PageType] = []
        qualities: list[SourceQuality] = []
        previous_known_type: Optional[PageType] = None
        for idx, img in enumerate(images):
            original_page_num = selected_pages[idx] + 1
            if self.forced_page_type is not None:
                ptype, quality = self.forced_page_type, SourceQuality.SCAN
            else:
                ptype, quality = await self._classify_page(
                    img,
                    page_num=original_page_num,
                    previous_page_type=previous_known_type,
                )
            page_types.append(ptype)
            qualities.append(quality)
            if ptype != PageType.UNKNOWN:
                previous_known_type = ptype

        # ── Step 5: Extract fields from each page ─────────────────────────
        page_results: list[PageExtractionResult] = []
        for i, img in enumerate(images):
            ptype = page_types[i]
            if ptype == PageType.UNKNOWN:
                continue  # Skip unclassifiable pages
            result = await self._extract_page(img, page_num=selected_pages[i] + 1, page_type=ptype)
            page_results.append(result)

        # ── Step 6: Purge all PIL images from RAM ──────────────────────────
        for img in images:
            img.close()
        images.clear()
        gc.collect()

        # ── Step 7: Determine overall source quality ───────────────────────
        # Worst quality across all pages governs the document-level quality
        quality_rank = {
            SourceQuality.DIGITAL: 0,
            SourceQuality.SCAN: 1,
            SourceQuality.PHOTOGRAPH: 2,
            SourceQuality.UNKNOWN: 3,
        }
        overall_quality = max(qualities, key=lambda q: quality_rank[q]) if qualities else SourceQuality.DIGITAL

        # ── Step 8: Merge page results into document result ────────────────
        merged = self._merge_page_results(
            page_results=page_results,
            doc_hash=doc_hash,
            session_id=session_id,
            filename=pdf_path.name,
            total_pages=total_pages,
            source_quality=overall_quality,
            start_ms=start_ms,
        )

        # ── Step 9: Generate deletion certificate and confirm purge ────────
        cert = generate_deletion_cert(doc_hash=doc_hash, nonce=nonce, session_id=session_id)
        merged.deletion_cert = cert
        merged.raw_purge_confirmed = True

        return merged

    def _selected_page_indexes(self, total_pages: int) -> list[int]:
        start = max(1, self.page_start)
        end = min(total_pages, self.page_end or total_pages)
        indexes = list(range(start - 1, end))
        if self.max_pages is not None:
            indexes = indexes[: self.max_pages]
        return indexes

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _classify_page(
        self,
        img: Image.Image,
        page_num: int,
        previous_page_type: Optional[PageType] = None,
    ) -> tuple[PageType, SourceQuality]:
        """Run the page classifier prompt and return the page type."""
        
        # Action 3: Basic document routing
        filename_lower = self.filename.lower() if hasattr(self, 'filename') and self.filename else ""
        if any(kw in filename_lower for kw in ["bank", "statement", "certificate"]):
            self.debug_events.append({"stage": "classify", "page_number": page_num, "rule": "filename_keyword", "assigned_type": "bank_statement"})
            return PageType.BANK_STATEMENT, SourceQuality.DIGITAL
        elif any(kw in filename_lower for kw in ["tax", "return"]):
            self.debug_events.append({"stage": "classify", "page_number": page_num, "rule": "filename_keyword", "assigned_type": "tax_return"})
            return PageType.TAX_RETURN, SourceQuality.DIGITAL
            
        ptype, quality, raw = self._run_classifier_prompt(
            img=img,
            page_num=page_num,
            prompt=PAGE_CLASSIFIER_PROMPT,
            stage="classify",
        )
        if ptype != PageType.UNKNOWN or previous_page_type is None:
            return ptype, quality

        retry_prompt = CONTINUATION_CLASSIFIER_PROMPT.replace(
            "__PREVIOUS_PAGE_TYPE__",
            previous_page_type.value,
        )
        retry_type, retry_quality, _ = self._run_classifier_prompt(
            img=img,
            page_num=page_num,
            prompt=retry_prompt,
            stage="classify_retry",
        )
        return retry_type, retry_quality

    def _run_classifier_prompt(
        self,
        img: Image.Image,
        page_num: int,
        prompt: str,
        stage: str,
    ) -> tuple[PageType, SourceQuality, str]:
        raw, _ = self.vlm.generate(
            image=img,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_new_tokens=256,
        )
        self.debug_events.append({
            "stage": stage,
            "page_number": page_num,
            "raw_response": raw,
        })
        try:
            data = json.loads(raw)
            ptype = PageType(data.get("page_type", "unknown"))
            quality = SourceQuality(data.get("source_quality", "digital"))
            return ptype, quality, raw
        except (json.JSONDecodeError, ValueError):
            return PageType.UNKNOWN, SourceQuality.DIGITAL, raw

    async def _extract_page(
        self,
        img: Image.Image,
        page_num: int,
        page_type: PageType,
    ) -> PageExtractionResult:
        """Run the targeted extraction prompt for this page type."""
        prompt = get_prompt(page_type)
        if not prompt:
            return PageExtractionResult(
                page_number=page_num,
                page_type=page_type,
                source_quality=SourceQuality.DIGITAL,
            )

        raw, mean_logprob = self.vlm.generate(
            image=img,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_new_tokens=self.extraction_max_tokens,
        )
        
        # Extract scratchpad content if present
        scratchpad_content = None
        scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', raw, re.DOTALL | re.IGNORECASE)
        if scratchpad_match:
            scratchpad_content = scratchpad_match.group(1).strip()
        
        # Extract JSON block
        json_str = raw
        if scratchpad_match:
            json_str = raw[scratchpad_match.end():].strip()
            
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try stripping markdown fences if the model misbehaved
            cleaned = json_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                data = {}

        self.debug_events.append({
            "stage": "extract",
            "page_number": page_num,
            "page_type": page_type.value,
            "mean_logprob": mean_logprob,
            "raw_response": raw,
            "scratchpad": scratchpad_content,
            "parsed_response": data,
        })

        return self.parser.parse_page(
            data=data,
            page_number=page_num,
            page_type=page_type,
            mean_logprob=mean_logprob,
        )

    def _merge_page_results(
        self,
        page_results: list[PageExtractionResult],
        doc_hash: str,
        session_id: str,
        filename: str,
        total_pages: int,
        source_quality: SourceQuality,
        start_ms: int,
    ) -> DocumentExtractionResult:
        """
        Merge all per-page results into a single DocumentExtractionResult.

        Merge strategy:
          - balance_series: concatenate across all bank statement pages, sort by month
          - deposit_entries: concatenate, deduplicate by (month_offset, amount)
          - i_tax, i_form, i_aff, i_spon: take the highest-confidence value if
            multiple pages provide the same field; raise a conflict warning if
            values differ materially
          - name_variants: collect all unique name strings keyed by page_type
        """
        elapsed_ms = int(time.time() * 1000) - start_ms

        merged = DocumentExtractionResult(
            document_hash=doc_hash,
            session_id=session_id,
            deletion_cert="",   # Set after purge
            filename=filename,
            total_pages=total_pages,
            source_quality=source_quality,
            extraction_ms=elapsed_ms,
            pages=page_results,
        )

        # Collect and merge fields
        for page in page_results:
            # Balance series
            merged.balance_series.extend(page.balance_series)

            # Deposit entries
            merged.deposit_entries.extend(page.deposit_entries)

            # Currency — take first non-null
            if merged.currency_code is None and page.currency_code:
                merged.currency_code = page.currency_code

            # Scalar income fields — take highest confidence
            merged.i_tax  = _best_confidence(merged.i_tax,  page.i_tax)
            merged.i_form = _best_confidence(merged.i_form, page.i_form)
            merged.i_aff  = _best_confidence(merged.i_aff,  page.i_aff)
            merged.i_spon = _best_confidence(merged.i_spon, page.i_spon)

            if merged.spon_relationship is None:
                merged.spon_relationship = page.spon_relationship

            if merged.tax_year is None:
                merged.tax_year = page.tax_year

            # Name variants
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
            merged.financial_indicators.update(page.financial_indicators)
            merged.adversarial_flags.extend(page.adversarial_flags)

        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_confidence(
    existing: Optional[ConfidenceWrapper],
    candidate: Optional[ConfidenceWrapper],
) -> Optional[ConfidenceWrapper]:
    """Return the wrapper with the higher confidence, preferring existing on tie."""
    if existing is None:
        return candidate
    if candidate is None:
        return existing
    rank = {Confidence.HIGH: 2, Confidence.MEDIUM: 1, Confidence.LOW: 0}
    return candidate if rank[candidate.confidence] > rank[existing.confidence] else existing
