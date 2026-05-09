from __future__ import annotations

from agents.state import PolicySource, UplanState


def distribute_policy(state: UplanState) -> dict:
    """
    Policy layer.

    This is the seam between deterministic rule agents and updateable policy.
    Today it uses a local policy table; Layer 2 replaces retrieve_policy_sources
    with ChromaDB/BYOG retrieval without changing agent logic.
    """
    visa_type = state.get("visa_type") or "student"
    jurisdiction = state.get("destination_jurisdiction") or "JP"
    percentile = state.get("applicant_income_percentile")

    policy_sources = retrieve_policy_sources(visa_type, jurisdiction)
    thresholds, trace = condition_thresholds(visa_type, jurisdiction, percentile)

    return {
        **thresholds,
        "threshold_trace": trace,
        "policy_sources": policy_sources,
        "currency_normalized": True,
    }


def retrieve_policy_sources(visa_type: str, jurisdiction: str) -> list[PolicySource]:
    return [
        {
            "source_id": "local-policy-student-funds-v0",
            "title": "Student visa financial evidence baseline",
            "jurisdiction": jurisdiction,
            "visa_type": visa_type,
            "excerpt": (
                "Applicant should demonstrate sufficient maintained funds, "
                "credible sponsor income, and corroborating bank/tax evidence."
            ),
        },
        {
            "source_id": "local-policy-income-coherence-v0",
            "title": "Income and sponsor coherence heuristic",
            "jurisdiction": jurisdiction,
            "visa_type": visa_type,
            "excerpt": (
                "Large unexplained disparities between affidavit claims and "
                "tax-verified income require additional proof."
            ),
        },
    ]


def condition_thresholds(
    visa_type: str,
    jurisdiction: str,
    applicant_income_percentile: float | None,
) -> tuple[dict[str, float], dict[str, str]]:
    thresholds = {
        "alpha": 0.25,
        "epsilon": 0.10,
        "delta_warn": 2.5,
        "delta_crit": 5.0,
        "w_late": 1.5,
        "kappa": 0.80,
    }
    trace = {
        "base": "student/default baseline",
        "jurisdiction": jurisdiction,
        "visa_type": visa_type,
    }

    if visa_type in {"skilled_worker", "work"}:
        thresholds["alpha"] = 0.35
        thresholds["delta_warn"] = 2.0
        thresholds["delta_crit"] = 4.0
        trace["visa_adjustment"] = "work visa: income consistency weighted higher than balance spikes"
    else:
        trace["visa_adjustment"] = "student visa: maintained liquid funds weighted strongly"

    if applicant_income_percentile is not None and applicant_income_percentile < 35:
        thresholds["alpha"] = 0.35
        thresholds["delta_warn"] = 3.0
        thresholds["delta_crit"] = 6.0
        trace["income_context"] = "lower-income context: larger sponsor/help deposits tolerated before critical escalation"
    elif applicant_income_percentile is not None and applicant_income_percentile > 80:
        thresholds["alpha"] = 0.20
        thresholds["delta_warn"] = 2.0
        thresholds["delta_crit"] = 4.0
        trace["income_context"] = "higher-income context: unexplained deposits/disparities held to tighter tolerance"
    else:
        trace["income_context"] = "income percentile unavailable or mid-band: default tolerances"

    return thresholds, trace
