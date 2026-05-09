from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from extractor import DocumentExtractor, HuggingFaceVLMBackend, MockVLMBackend
from graph import build_graph
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


def money_items(rows: list[dict], amount_key: str = "amount") -> list[dict]:
    items: list[dict] = []
    for row in rows:
        amount = row.get(amount_key)
        if amount is None:
            continue
        items.append({
            "source": row.get("source") or row.get("institution_name") or row.get("description") or "unknown",
            "amount": float(amount),
            "description": row.get("description") or row.get("account_type") or "",
        })
    return items


def build_state_from_payload(
    payload: dict[str, Any],
    t_req: float,
    visa_type: str,
    destination_jurisdiction: str,
    applicant_income_percentile: float | None,
) -> dict[str, Any]:
    full = payload.get("full_result", {})
    fields = payload.get("reliable_fields", {})

    balance_series = [
        float(item["value"]) for item in full.get("balance_series", []) if item.get("value") is not None
    ]
    deposit_entries = [
        (float(offset), float(wrapper["value"]))
        for offset, wrapper in full.get("deposit_entries", [])
        if wrapper.get("value") is not None
    ]

    return {
        "visa_type": visa_type,
        "destination_jurisdiction": destination_jurisdiction,
        "applicant_income_percentile": applicant_income_percentile,
        "currency_normalized": False,
        "balance_series": balance_series,
        "deposit_entries": deposit_entries,
        "t_req": t_req,
        "i_form": fields.get("i_form"),
        "i_tax": fields.get("i_tax"),
        "i_aff": fields.get("i_aff"),
        "i_spon": fields.get("i_spon"),
        "spon_relationship": fields.get("spon_relationship"),
        "currency_code": fields.get("currency_code"),
        "name_variants": fields.get("name_variants", {}),
        "financial_accounts": money_items(fields.get("financial_accounts", [])),
        "income_sources": money_items(fields.get("income_sources", []), amount_key="annual_amount"),
        "movable_assets": money_items(fields.get("movable_assets", [])),
        "properties": money_items(fields.get("properties", []), amount_key="value"),
        "alpha": 0.0,
        "epsilon": 0.0,
        "delta_warn": 0.0,
        "delta_crit": 0.0,
        "w_late": 0.0,
        "kappa": 0.0,
        "threshold_trace": {},
        "policy_sources": [],
        "findings": [],
        "completed_agents": [],
        "narrative_score": None,
        "human_review_required": False,
        "synthesis_trace": None,
        "rejection_case": None,
        "rebuttal_case": None,
        "raw_purge_confirmed": bool(full.get("raw_purge_confirmed") or payload.get("summary", {}).get("raw_purge_confirmed")),
    }


async def extract_pdf(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "hf":
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
    return to_jsonable(payload)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction plus Layer 1 UPlan agents.")
    parser.add_argument("--pdf", type=Path, help="PDF to extract before running agents.")
    parser.add_argument("--extraction-json", type=Path, help="Existing run_real_extraction JSON output.")
    parser.add_argument("--out", type=Path, help="Optional path to write full pipeline JSON.")
    parser.add_argument("--t-req", type=float, default=800000.0)
    parser.add_argument("--visa-type", default="student")
    parser.add_argument("--destination-jurisdiction", default="JP")
    parser.add_argument("--applicant-income-percentile", type=float)
    parser.add_argument("--session-id", default="manual-layer1-test")
    parser.add_argument("--backend", choices=["mock", "hf"], default="hf")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--force-page-type",
        choices=[page.value for page in PageType if page != PageType.UNKNOWN],
    )
    parser.add_argument("--include-vlm-debug", action="store_true")
    args = parser.parse_args()

    if args.extraction_json:
        payload = json.loads(args.extraction_json.read_text(encoding="utf-8"))
    elif args.pdf:
        payload = await extract_pdf(args)
    else:
        raise SystemExit("Provide either --extraction-json or --pdf.")

    state = build_state_from_payload(
        payload,
        t_req=args.t_req,
        visa_type=args.visa_type,
        destination_jurisdiction=args.destination_jurisdiction,
        applicant_income_percentile=args.applicant_income_percentile,
    )
    output = build_graph().invoke(state)
    result = {
        "summary": payload.get("summary", {}),
        "reliable_fields": payload.get("reliable_fields", {}),
        "agent_output": output,
        "findings": output["findings"],
    }

    text = json.dumps(to_jsonable(result), indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote pipeline JSON to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    asyncio.run(main())
