from __future__ import annotations

from agents.state import UplanState


def run_adversarial_audit(state: UplanState) -> dict:
    """
    Construct the strongest plausible rejection case, then the rebuttal case.
    This mirrors an officer-style review posture instead of confirmation bias.
    """
    findings = state.get("findings", [])
    critical = [f["message"] for f in findings if f.get("severity") == "critical"]
    warnings = [f["message"] for f in findings if f.get("severity") == "warning"]

    if critical:
        rejection = "Strongest rejection case: " + " ".join(critical[:3])
    elif warnings:
        rejection = "Strongest rejection case: " + " ".join(warnings[:3])
    else:
        rejection = (
            "Strongest rejection case: no major rule failure found, but officer may "
            "still request corroboration for source of funds and identity consistency."
        )

    evidence = []
    if state.get("financial_accounts"):
        evidence.append("declared bank/fixed-deposit evidence")
    if state.get("income_sources"):
        evidence.append("itemised income sources")
    if state.get("properties"):
        evidence.append("property evidence")
    if state.get("i_tax"):
        evidence.append("tax baseline")
    rebuttal = (
        "Rebuttal case: " + (", ".join(evidence) if evidence else "insufficient corroborating evidence yet")
        + ". Remaining gaps should be answered with bank statements, tax computation, and balance certificates."
    )

    return {
        "rejection_case": rejection,
        "rebuttal_case": rebuttal,
        "human_review_required": state.get("human_review_required", False) or bool(critical),
    }
