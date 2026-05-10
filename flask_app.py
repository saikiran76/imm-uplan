"""
flask_app.py — Uplan Extraction API Server
-------------------------------------------
Runs on the AMD MI300X GPU server. Loads Qwen3.6-27B into VRAM at startup
and exposes HTTP endpoints for the Gradio frontend on HuggingFace Spaces.

Usage
-----
    # Start with real VLM (GPU required):
    python flask_app.py --model-path Qwen/Qwen3.6-27B --port 8000

    # Start with mock backend (no GPU, for testing):
    python flask_app.py --backend mock --port 8000

Endpoints
---------
    POST /extract      — Upload PDF files, run full pipeline, return JSON
    GET  /health       — Health check with model & GPU status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from extractor import DocumentExtractor, HuggingFaceVLMBackend, MockVLMBackend
from graph import build_graph
from models import PageType

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("uplan.flask")


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# Global state — populated at startup
_vlm_backend = None
_model_path = ""
_backend_type = ""
_start_time = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Auth middleware
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("UPLAN_API_KEY", "")


@app.before_request
def check_api_key():
    """Optional API key auth. If UPLAN_API_KEY is set, require it."""
    if not API_KEY:
        return  # No key configured — allow all requests
    if request.endpoint == "health":
        return  # Health check is always public
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        return jsonify({"error": "Invalid or missing API key"}), 401


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers (from run_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/enums to JSON-safe dicts."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline logic
# ─────────────────────────────────────────────────────────────────────────────

def build_state_from_payload(
    payload: dict[str, Any],
    t_req: float = 800000.0,
    visa_type: str = "student",
    destination_jurisdiction: str = "JP",
    applicant_income_percentile: float | None = None,
) -> dict[str, Any]:
    """Convert extraction payload into LangGraph agent state."""
    full = payload.get("full_result", {})
    fields = payload.get("reliable_fields", {})

    balance_series = [
        float(item["value"]) for item in full.get("balance_series", [])
        if item.get("value") is not None
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
        "raw_purge_confirmed": bool(
            full.get("raw_purge_confirmed")
            or payload.get("summary", {}).get("raw_purge_confirmed")
        ),
    }


async def run_extraction(pdf_path: Path, session_id: str) -> dict[str, Any]:
    """Run VLM extraction on a single PDF file."""
    extractor = DocumentExtractor(vlm=_vlm_backend)
    result = await extractor.extract(pdf_path, session_id=session_id)

    payload = {
        "summary": result.extraction_summary(),
        "reliable_fields": result.reliable_fields(),
        "full_result": result,
    }
    return to_jsonable(payload)


def merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple single-document extraction payloads."""
    if len(payloads) == 1:
        return payloads[0]

    merged_full: dict[str, Any] = {
        "balance_series": [],
        "deposit_entries": [],
        "financial_accounts": [],
        "income_sources": [],
        "movable_assets": [],
        "properties": [],
        "pages": [],
        "name_variants": {},
        "raw_purge_confirmed": True,
        "total_pages": 0,
        "deletion_cert": "",
    }
    merged_fields: dict[str, Any] = {
        "financial_accounts": [],
        "income_sources": [],
        "movable_assets": [],
        "properties": [],
        "family_members": [],
        "name_variants": {},
    }
    summaries = []

    for payload in payloads:
        full = payload.get("full_result", {})
        fields = payload.get("reliable_fields", {})
        summaries.append(payload.get("summary", {}))

        for key in ("balance_series", "deposit_entries", "financial_accounts",
                     "income_sources", "movable_assets", "properties", "pages"):
            merged_full[key].extend(full.get(key, []))
        for key in ("financial_accounts", "income_sources", "movable_assets",
                     "properties", "family_members"):
            merged_fields[key].extend(fields.get(key, []))

        merged_full["name_variants"].update(full.get("name_variants", {}))
        merged_fields["name_variants"].update(fields.get("name_variants", {}))
        merged_full["raw_purge_confirmed"] = (
            merged_full["raw_purge_confirmed"] and bool(full.get("raw_purge_confirmed"))
        )
        merged_full["total_pages"] += int(full.get("total_pages", 0) or 0)
        if full.get("deletion_cert"):
            sep = "\n---\n" if merged_full["deletion_cert"] else ""
            merged_full["deletion_cert"] += sep + full["deletion_cert"]

        for key in ("currency_code", "i_tax", "i_form", "i_aff", "i_spon",
                     "spon_relationship", "tax_year", "beneficiary_name",
                     "declarant_address"):
            if merged_fields.get(key) is None and fields.get(key) is not None:
                merged_fields[key] = fields[key]
            if merged_full.get(key) is None and full.get(key) is not None:
                merged_full[key] = full[key]

    return {
        "summary": {
            "filename": "combined-packet",
            "total_pages": merged_full["total_pages"],
            "raw_purge_confirmed": merged_full["raw_purge_confirmed"],
            "source_quality": "mixed",
            "documents": summaries,
        },
        "reliable_fields": merged_fields,
        "full_result": merged_full,
    }


def run_agent_pipeline(payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Run the LangGraph agent pipeline on an extraction payload."""
    state = build_state_from_payload(
        payload,
        t_req=params.get("t_req", 800000.0),
        visa_type=params.get("visa_type", "student"),
        destination_jurisdiction=params.get("destination_jurisdiction", "JP"),
        applicant_income_percentile=params.get("applicant_income_percentile"),
    )
    output = build_graph().invoke(state)

    return {
        "summary": payload.get("summary", {}),
        "reliable_fields": payload.get("reliable_fields", {}),
        "full_result": payload.get("full_result", {}),
        "agent_output": output,
        "findings": output["findings"],
        "narrative_synthesis": {
            "narrative_score": output.get("narrative_score"),
            "human_review_required": output.get("human_review_required"),
            "synthesis_trace": output.get("synthesis_trace"),
        },
        "adversarial_audit": {
            "rejection_case": output.get("rejection_case"),
            "rebuttal_case": output.get("rebuttal_case"),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns model info and uptime."""
    gpu_info = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return jsonify({
        "status": "ok",
        "model": _model_path,
        "backend": _backend_type,
        "gpu": gpu_info,
        "uptime_seconds": int(time.time() - _start_time),
    })


@app.route("/extract", methods=["POST"])
def extract():
    """
    Main extraction endpoint.

    Accepts:
        - multipart/form-data with one or more PDF files under key "files"
        - Optional JSON params in form field "params":
          {"t_req": 800000, "visa_type": "student", ...}

    Returns:
        Full pipeline JSON (extraction + agent findings)
    """
    if _vlm_backend is None:
        return jsonify({"error": "Model not loaded"}), 503

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No files uploaded. Send PDFs under key 'files'."}), 400

    # Parse optional pipeline parameters
    params_raw = request.form.get("params", "{}")
    try:
        params = json.loads(params_raw)
    except json.JSONDecodeError:
        params = {}

    session_id = secrets.token_hex(8)
    tmp_dir = Path(tempfile.mkdtemp(prefix="uplan_"))

    try:
        # Save uploaded files to temp directory
        pdf_paths: list[Path] = []
        for f in uploaded_files:
            if not f.filename:
                continue
            safe_name = f.filename.replace("/", "_").replace("\\", "_")
            dest = tmp_dir / safe_name
            f.save(str(dest))
            pdf_paths.append(dest)
            logger.info("Saved upload: %s (%d bytes)", safe_name, dest.stat().st_size)

        if not pdf_paths:
            return jsonify({"error": "No valid PDF files received."}), 400

        # Run extraction on each PDF
        logger.info("Starting extraction of %d file(s) with session=%s", len(pdf_paths), session_id)
        start = time.time()

        payloads: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            payload = asyncio.run(run_extraction(pdf_path, session_id))
            payloads.append(payload)

        # Merge if multiple documents
        merged = merge_payloads(payloads) if len(payloads) > 1 else payloads[0]

        extraction_time = time.time() - start
        logger.info("Extraction complete in %.1fs", extraction_time)

        # Run agent pipeline
        logger.info("Running agent pipeline...")
        agent_start = time.time()
        result = run_agent_pipeline(merged, params)
        agent_time = time.time() - agent_start
        logger.info("Agent pipeline complete in %.1fs", agent_time)

        # Add timing metadata
        result["_meta"] = {
            "session_id": session_id,
            "extraction_seconds": round(extraction_time, 2),
            "agent_seconds": round(agent_time, 2),
            "total_seconds": round(extraction_time + agent_time, 2),
            "model": _model_path,
            "files_processed": len(pdf_paths),
        }

        return Response(
            json.dumps(to_jsonable(result), indent=2),
            mimetype="application/json",
        )

    except Exception as exc:
        logger.exception("Extraction failed")
        return jsonify({
            "error": str(exc),
            "session_id": session_id,
        }), 500

    finally:
        # Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/audit", methods=["POST"])
def audit():
    """
    Agent-only endpoint.

    Accepts existing extraction JSON and returns deterministic findings,
    narrative synthesis, and adversarial audit without rerunning extraction.
    """
    body = request.get_json(silent=True) or {}
    payloads = body.get("extraction_payloads") or body.get("payloads") or []
    if isinstance(body.get("extraction_payload"), dict):
        payloads = [body["extraction_payload"]]
    if isinstance(payloads, dict):
        payloads = [payloads]
    if not payloads:
        return jsonify({"error": "Provide extraction_payloads or extraction_payload JSON."}), 400

    params = {
        "t_req": body.get("t_req", 800000.0),
        "visa_type": body.get("visa_type", "student"),
        "destination_jurisdiction": body.get("destination_jurisdiction", "JP"),
        "applicant_income_percentile": body.get("applicant_income_percentile"),
    }
    merged = merge_payloads(payloads) if len(payloads) > 1 else payloads[0]
    result = run_agent_pipeline(merged, params)
    return Response(
        json.dumps(to_jsonable(result), indent=2),
        mimetype="application/json",
    )


def init_backend(model_path: str, backend_type: str, device: str, max_new_tokens: int):
    """Initialize the VLM backend (called once at startup)."""
    global _vlm_backend, _model_path, _backend_type, _start_time

    _model_path = model_path
    _backend_type = backend_type
    _start_time = time.time()

    if backend_type == "hf":
        logger.info("Loading HuggingFace VLM: %s (device=%s)", model_path, device)
        backend = HuggingFaceVLMBackend(
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        backend.load()
        logger.info("Model loaded successfully!")
    else:
        logger.info("Using MockVLMBackend (no GPU required)")
        backend = MockVLMBackend("mock")

    _vlm_backend = backend


def main():
    parser = argparse.ArgumentParser(description="Uplan Flask Extraction Server")
    parser.add_argument("--model-path", default="Qwen/Qwen3.6-27B",
                        help="HuggingFace model ID (default: Qwen/Qwen3.6-27B)")
    parser.add_argument("--backend", choices=["hf", "mock"], default="hf",
                        help="VLM backend: 'hf' for real GPU, 'mock' for testing")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cpu, mps")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens for VLM generation")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port (default: 8000)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode")
    args = parser.parse_args()

    # Load model into VRAM
    init_backend(
        model_path=args.model_path,
        backend_type=args.backend,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    # Start Flask server
    logger.info("Starting Uplan API server on %s:%d", args.host, args.port)
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,  # Don't reload — model is already loaded
    )


if __name__ == "__main__":
    main()
