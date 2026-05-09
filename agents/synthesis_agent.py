from __future__ import annotations

from agents.state import UplanState


def run_synthesis_agent(state: UplanState) -> dict:
    findings = state.get("findings", [])
    critical = [f for f in findings if f.get("severity") == "critical"]
    warnings = [f for f in findings if f.get("severity") == "warning"]

    score = max(0.05, 0.90 - len(critical) * 0.25 - len(warnings) * 0.08)
    trace_parts = [
        f"{len(critical)} critical finding(s)",
        f"{len(warnings)} warning finding(s)",
        f"policy sources: {len(state.get('policy_sources', []))}",
    ]
    if state.get("threshold_trace"):
        trace_parts.append("thresholds conditioned by visa/jurisdiction/income context")

    return {
        "narrative_score": score,
        "human_review_required": bool(critical) or any(f.get("requires_human_review") for f in findings),
        "synthesis_trace": "; ".join(trace_parts),
    }
