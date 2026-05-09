from __future__ import annotations

from agents.state import AgentFinding, UplanState


def run_tax_agent(state: UplanState) -> dict:
    """
    Agent B: Tax & Income Coherence.
    Rules 4 and 5 from the formal spec.
    """
    assert state["raw_purge_confirmed"], "PRIVACY GATE"

    findings: list[AgentFinding] = []
    i_tax = state.get("i_tax")
    i_form = state.get("i_form")
    i_aff = _affidavit_income(state)
    epsilon = state["epsilon"]
    delta_warn = state["delta_warn"]
    delta_crit = state["delta_crit"]

    if i_tax and i_form:
        deviation = abs(i_form - i_tax) / i_tax
        if deviation > epsilon:
            findings.append(AgentFinding(
                agent_id="tax_income",
                rule_id="R4_form_tax_mismatch",
                severity="critical",
                message=(
                    f"Form income {i_form:,.0f} deviates from tax record "
                    f"{i_tax:,.0f} by {deviation * 100:.1f}% "
                    f"(permitted: {epsilon * 100:.0f}%)."
                ),
                requires_human_review=True,
            ))

    if i_tax and i_aff and i_tax > 0:
        delta = i_aff / i_tax
        if delta > delta_crit:
            findings.append(AgentFinding(
                agent_id="tax_income",
                rule_id="R5_extreme_disparity",
                severity="critical",
                message=(
                    f"Affidavit claims {i_aff:,.0f} vs tax-verified {i_tax:,.0f}; "
                    f"ratio delta={delta:.1f}x exceeds critical threshold "
                    f"{delta_crit}x. Verifiable asset proof mandatory."
                ),
                requires_human_review=True,
            ))
        elif delta > delta_warn:
            findings.append(AgentFinding(
                agent_id="tax_income",
                rule_id="R5_elevated_disparity",
                severity="warning",
                message=(
                    f"Affidavit-to-tax ratio delta={delta:.1f}x is elevated. "
                    f"Asset documentation recommended."
                ),
                requires_human_review=False,
            ))

    return {"findings": findings}


def _affidavit_income(state: UplanState) -> float | None:
    sources = state.get("income_sources", [])
    source_total = sum(float(item.get("amount") or 0.0) for item in sources)
    scalar = state.get("i_aff") or 0.0
    value = max(float(scalar), source_total)
    return value or None
