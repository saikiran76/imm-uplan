"""
Run the long-context text extraction branch against a text-native PDF.

Use this for long bank statements and ITR/computation PDFs where PyMuPDF can
extract meaningful text. It emits the same JSON envelope as run_real_extraction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from models import PageType
from text_extractor import HFTextLLMBackend, LongContextTextExtractor


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured JSON from a text-native PDF using a long-context model.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF to process.")
    parser.add_argument(
        "--doc-type",
        required=True,
        choices=[
            PageType.BANK_STATEMENT.value,
            PageType.BANK_BALANCE_CERTIFICATE.value,
            PageType.TAX_RETURN.value,
        ],
        help="Document type to extract with the text branch.",
    )
    parser.add_argument("--out", type=Path, help="Optional path to write JSON output.")
    parser.add_argument("--session-id", default="manual-text-doc-test")
    parser.add_argument("--model-path", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or an accelerate device_map value.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-chars", type=int, default=160000)
    parser.add_argument("--max-pages", type=int, help="Only read this many selected pages.")
    parser.add_argument("--page-start", type=int, default=1, help="1-based first page to read.")
    parser.add_argument("--page-end", type=int, help="1-based last page to read.")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    backend = HFTextLLMBackend(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    backend.load()

    extractor = LongContextTextExtractor(
        llm=backend,
        doc_type=PageType(args.doc_type),
        max_input_chars=args.max_input_chars,
        page_start=args.page_start,
        page_end=args.page_end,
        max_pages=args.max_pages,
    )
    result = extractor.extract(args.pdf, session_id=args.session_id)
    payload = {
        "summary": result.extraction_summary(),
        "reliable_fields": result.reliable_fields(),
        "full_result": result,
    }
    if args.include_debug:
        payload["text_debug"] = extractor.debug_events

    text = json.dumps(to_jsonable(payload), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote text extraction JSON to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
