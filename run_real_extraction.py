"""
Run the extraction pipeline against a real local PDF and emit JSON.

Example:
    python run_real_extraction.py "C:\\path\\to\\document.pdf" --out result.json

By default this uses MockVLMBackend, which proves the PDF/rasterise/parse/merge
flow but returns fixture values. Use --backend hf for real local inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from extractor import DocumentExtractor, HuggingFaceVLMBackend, MockVLMBackend
from models import PageType


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


async def run(args: argparse.Namespace) -> dict[str, Any]:
    use_hf = args.backend == "hf" or args.real_backend
    if use_hf:
        backend = HuggingFaceVLMBackend(
            args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        backend.load()
    else:
        backend = MockVLMBackend("mock")

    forced_page_type = PageType(args.force_page_type) if args.force_page_type else None
    extractor = DocumentExtractor(vlm=backend, forced_page_type=forced_page_type)
    result = await extractor.extract(args.pdf, session_id=args.session_id)

    payload = {
        "summary": result.extraction_summary(),
        "reliable_fields": result.reliable_fields(),
        "full_result": result,
    }
    if args.include_vlm_debug:
        payload["vlm_debug"] = extractor.debug_events
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a local PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF to process.")
    parser.add_argument("--out", type=Path, help="Optional path to write JSON output.")
    parser.add_argument("--session-id", default="manual-real-doc-test")
    parser.add_argument("--backend", choices=["mock", "hf"], default="mock")
    parser.add_argument("--real-backend", action="store_true", help="Alias for --backend hf.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--force-page-type",
        choices=[p.value for p in PageType if p != PageType.UNKNOWN],
        help="Skip page classification and extract every page as this type.",
    )
    parser.add_argument(
        "--include-vlm-debug",
        action="store_true",
        help="Include raw classifier/extractor model responses in the output JSON.",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    payload = to_jsonable(asyncio.run(run(args)))
    text = json.dumps(payload, indent=2, sort_keys=True)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote extraction JSON to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
