from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentFinding(TypedDict):
    agent_id: str
    rule_id: str
    severity: str
    message: str
    requires_human_review: bool


class MoneyItem(TypedDict, total=False):
    source: str
    amount: float
    description: str


class UplanState(TypedDict):
    # Populated by extraction layer
    balance_series: list[float]
    deposit_entries: list[tuple[float, float]]
    t_req: float
    i_form: Optional[float]
    i_tax: Optional[float]
    i_aff: Optional[float]
    i_spon: Optional[float]
    spon_relationship: Optional[str]
    currency_code: Optional[str]
    name_variants: dict[str, str]
    financial_accounts: list[MoneyItem]
    income_sources: list[MoneyItem]
    movable_assets: list[MoneyItem]
    properties: list[MoneyItem]

    # Context thresholds, hardcoded now and RAG-fed later
    alpha: float
    epsilon: float
    delta_warn: float
    delta_crit: float
    w_late: float

    # Agent outputs. LangGraph appends parallel node outputs via reducer.
    findings: Annotated[list[AgentFinding], operator.add]

    # Synthesis output
    narrative_score: Optional[float]
    human_review_required: bool
    synthesis_trace: Optional[str]

    # Privacy gate
    raw_purge_confirmed: bool
