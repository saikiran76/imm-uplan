from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from run_pipeline import build_state_from_payload, merge_payloads, to_jsonable
from graph import build_graph


SAMPLE_PATH = Path(__file__).parent / "sample_outputs" / "demo_result.json"
DEMO_RESULT = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else {}


class AnalyzeJsonRequest(BaseModel):
    extraction_payloads: list[dict[str, Any]]
    t_req: float = 800000.0
    visa_type: str = "student"
    destination_jurisdiction: str = "JP"
    applicant_income_percentile: Optional[float] = None


app = FastAPI(title="Uplan Analysis API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-json")
def analyze_json(request: AnalyzeJsonRequest) -> dict[str, Any]:
    return run_agent_pipeline(
        payloads=request.extraction_payloads,
        t_req=request.t_req,
        visa_type=request.visa_type,
        destination_jurisdiction=request.destination_jurisdiction,
        applicant_income_percentile=request.applicant_income_percentile,
    )


@app.post("/analyze")
async def analyze(
    files: list[UploadFile] = File(default=[]),
    t_req: float = Form(default=800000.0),
    visa_type: str = Form(default="student"),
    destination_jurisdiction: str = Form(default="JP"),
    applicant_income_percentile: Optional[float] = Form(default=None),
) -> dict[str, Any]:
    """
    Demo/live bridge endpoint.

    If JSON extraction outputs are uploaded, run the deterministic agent graph
    over them. If PDFs are uploaded, return the current demo result instead of
    blocking the UI on unfinished extraction research.
    """
    json_payloads: list[dict[str, Any]] = []
    uploaded_pdf = False

    for upload in files:
        name = upload.filename or ""
        data = await upload.read()
        if name.lower().endswith(".json"):
            json_payloads.append(json.loads(data.decode("utf-8")))
        elif name.lower().endswith(".pdf"):
            uploaded_pdf = True

    if json_payloads:
        return run_agent_pipeline(
            payloads=json_payloads,
            t_req=t_req,
            visa_type=visa_type,
            destination_jurisdiction=destination_jurisdiction,
            applicant_income_percentile=applicant_income_percentile,
        )

    result = dict(DEMO_RESULT)
    result["backend_status"] = (
        "pdf_demo_fallback" if uploaded_pdf else "demo_fallback"
    )
    return result


def run_agent_pipeline(
    payloads: list[dict[str, Any]],
    t_req: float,
    visa_type: str,
    destination_jurisdiction: str,
    applicant_income_percentile: Optional[float],
) -> dict[str, Any]:
    if not payloads:
        return DEMO_RESULT

    payload = merge_payloads(payloads)
    state = build_state_from_payload(
        payload,
        t_req=t_req,
        visa_type=visa_type,
        destination_jurisdiction=destination_jurisdiction,
        applicant_income_percentile=applicant_income_percentile,
    )
    output = build_graph().invoke(state)
    result = {
        "summary": payload.get("summary", {}),
        "reliable_fields": payload.get("reliable_fields", {}),
        "full_result": payload.get("full_result", {}),
        "agent_output": output,
        "findings": output.get("findings", []),
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
    return to_jsonable(result)
