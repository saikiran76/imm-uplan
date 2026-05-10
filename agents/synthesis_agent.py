from __future__ import annotations

import logging

from agents.state import UplanState

logger = logging.getLogger("uplan.synthesis_agent")


def run_synthesis_agent(state: UplanState) -> dict:
    findings = state.get("findings", [])
    critical = [f for f in findings if f.get("severity") == "critical"]
    warnings = [f for f in findings if f.get("severity") == "warning"]

    logger.info(
        "SYNTHESIS AGENT: %d findings (%d critical, %d warnings), "
        "balance_series=%d, financial_accounts=%d, i_tax=%s, i_aff=%s",
        len(findings), len(critical), len(warnings),
        len(state.get("balance_series", [])),
        len(state.get("financial_accounts", [])),
        state.get("i_tax"), state.get("i_aff"),
    )

    score = max(0.05, 0.90 - len(critical) * 0.25 - len(warnings) * 0.08)
    trace_parts = [
        f"{len(critical)} critical finding(s)",
        f"{len(warnings)} warning finding(s)",
        f"policy sources: {len(state.get('policy_sources', []))}",
    ]
    if state.get("threshold_trace"):
        trace_parts.append("thresholds conditioned by visa/jurisdiction/income context")

    evidence_parts = []
    if state.get("financial_accounts"):
        total = sum(float(item.get("amount") or 0.0) for item in state["financial_accounts"])
        evidence_parts.append(f"financial evidence total {total:,.0f} from bank/fixed-deposit records")
    if state.get("spon_relationship"):
        evidence_parts.append(f"sponsor relationship stated as {state['spon_relationship']}")
    if state.get("i_tax"):
        evidence_parts.append(f"tax baseline {float(state['i_tax']):,.0f}")
    else:
        evidence_parts.append("tax baseline missing")
    if not state.get("balance_series"):
        evidence_parts.append("bank statement time series missing")

    logger.info("SYNTHESIS RESULT: score=%.2f, human_review=%s", score, bool(critical))

    return {
        "narrative_score": score,
        "human_review_required": bool(critical) or any(f.get("requires_human_review") for f in findings),
        "synthesis_trace": "; ".join(trace_parts + evidence_parts),
    }
