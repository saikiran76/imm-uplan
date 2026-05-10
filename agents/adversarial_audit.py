from __future__ import annotations

import logging

from agents.state import UplanState

logger = logging.getLogger("uplan.adversarial_audit")


def run_adversarial_audit(state: UplanState) -> dict:
    """
    Construct the strongest plausible rejection case, then the rebuttal case.
    This mirrors an officer-style review posture instead of confirmation bias.
    """
    findings = state.get("findings", [])
    critical = [f["message"] for f in findings if f.get("severity") == "critical"]
    warnings = [f["message"] for f in findings if f.get("severity") == "warning"]

    logger.info(
        "ADVERSARIAL AUDIT: %d findings (%d critical, %d warnings), "
        "balance_series=%d, financial_accounts=%d, i_tax=%s",
        len(findings), len(critical), len(warnings),
        len(state.get("balance_series", [])),
        len(state.get("financial_accounts", [])),
        state.get("i_tax"),
    )

    docs = []
    if state.get("financial_accounts"):
        docs.append("bank balance certificate / financial account evidence")
    if state.get("balance_series"):
        docs.append("bank statement series")
    if state.get("i_tax"):
        docs.append("tax return or computation")
    if state.get("name_variants"):
        docs.append("identity/name evidence")

    if critical:
        rejection = "Strongest rejection case: " + " ".join(critical[:3])
    elif warnings:
        rejection = "Strongest rejection case: " + " ".join(warnings[:3])
    else:
        rejection = (
            "Strongest rejection case: no major rule failure found, but officer may "
            "still request corroboration for source of funds and identity consistency."
        )
    if docs:
        rejection += " Evidence reviewed: " + ", ".join(docs) + "."
    if state.get("financial_accounts") and not state.get("balance_series"):
        rejection += (
            " The balance certificate supports existence of funds at issue date, "
            "but it does not by itself prove accumulation history or rule out show-money deposits."
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
    if state.get("spon_relationship"):
        rebuttal += f" Relationship evidence currently states {state['spon_relationship']}."

    logger.info("ADVERSARIAL RESULT: rejection_len=%d, rebuttal_len=%d", len(rejection), len(rebuttal))

    return {
        "rejection_case": rejection,
        "rebuttal_case": rebuttal,
        "human_review_required": state.get("human_review_required", False) or bool(critical),
    }
