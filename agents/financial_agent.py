from __future__ import annotations

import statistics

from agents.state import AgentFinding, UplanState


def run_financial_agent(state: UplanState) -> dict:
    """
    Agent A: Financial Flow.

    Rules 1-3 use statement time series when available. For affidavit-only
    packs, Rule 1 falls back to liquid financial accounts so the agent can
    still reason over the extracted support evidence.
    """
    assert state["raw_purge_confirmed"], "PRIVACY GATE"

    findings: list[AgentFinding] = []
    balances = state["balance_series"]
    deposits = state["deposit_entries"]
    t_req = state["t_req"]
    alpha = state["alpha"]
    w_late = state["w_late"]

    if t_req <= 0:
        return {"findings": findings}

    if balances:
        _evaluate_balance_series(
            findings=findings,
            balances=balances,
            deposits=deposits,
            t_req=t_req,
            alpha=alpha,
            w_late=w_late,
        )
    else:
        _evaluate_affidavit_liquidity(
            findings=findings,
            accounts=state.get("financial_accounts", []),
            movable_assets=state.get("movable_assets", []),
            t_req=t_req,
        )
        if state.get("financial_accounts"):
            findings.append(AgentFinding(
                agent_id="financial_flow",
                rule_id="R1_bank_statement_missing",
                severity="warning",
                message=(
                    "Bank or fixed-deposit balances were found, but no month-by-month "
                    "bank statement series was available. A balance certificate proves "
                    "point-in-time funds, not accumulation history; maintained "
                    "liquidity and late-deposit risk cannot be verified."
                ),
                requires_human_review=False,
            ))

    return {"findings": findings, "completed_agents": ["financial_agent"]}


def _evaluate_balance_series(
    findings: list[AgentFinding],
    balances: list[float],
    deposits: list[tuple[float, float]],
    t_req: float,
    alpha: float,
    w_late: float,
) -> None:
    b_avg = statistics.mean(balances)
    b_min = min(balances)
    kappa = 0.80

    if b_avg < t_req or b_min < kappa * t_req:
        findings.append(AgentFinding(
            agent_id="financial_flow",
            rule_id="R1_liquidity",
            severity="critical",
            message=(
                f"Avg balance {b_avg:,.0f} or floor {b_min:,.0f} below required "
                f"{t_req:,.0f}. Sustained liquidity not demonstrated."
            ),
            requires_human_review=True,
        ))

    window = len(balances)
    for month_offset, amount in deposits:
        if amount > alpha * t_req and month_offset > (window - w_late):
            findings.append(AgentFinding(
                agent_id="financial_flow",
                rule_id="R2_deposit_anomaly",
                severity="warning",
                message=(
                    f"Late large deposit of {amount:,.0f} "
                    f"({amount / t_req * 100:.0f}% of required funds) in month "
                    f"{month_offset:.1f}. Provenance documentation required."
                ),
                requires_human_review=False,
            ))

    if b_avg > 0:
        sigma = (max(balances) - min(balances)) / b_avg
        if sigma > 0.40:
            findings.append(AgentFinding(
                agent_id="financial_flow",
                rule_id="R3_volatility",
                severity="warning",
                message=(
                    f"Balance volatility sigma={sigma:.2f} exceeds 0.40. "
                    f"Possible show-money pattern; verify month-by-month."
                ),
                requires_human_review=False,
            ))


def _evaluate_affidavit_liquidity(
    findings: list[AgentFinding],
    accounts: list[dict],
    movable_assets: list[dict],
    t_req: float,
) -> None:
    liquid_total = sum(float(item.get("amount") or 0.0) for item in accounts)
    movable_total = sum(float(item.get("amount") or 0.0) for item in movable_assets)

    if liquid_total < t_req:
        findings.append(AgentFinding(
            agent_id="financial_flow",
            rule_id="R1_affidavit_liquidity",
            severity="critical",
            message=(
                f"Affidavit liquid accounts total {liquid_total:,.0f}, below "
                f"required funds {t_req:,.0f}. Additional liquid proof needed."
            ),
            requires_human_review=True,
        ))

    if movable_total > 0:
        findings.append(AgentFinding(
            agent_id="financial_flow",
            rule_id="R1_movable_asset_review",
            severity="warning",
            message=(
                f"Movable assets of {movable_total:,.0f} were declared. "
                f"Treat as supporting evidence unless independently valued."
            ),
            requires_human_review=False,
        ))
