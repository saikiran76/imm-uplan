from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.adversarial_audit import run_adversarial_audit
from agents.financial_agent import run_financial_agent
from agents.policy import distribute_policy
from agents.state import UplanState
from agents.synthesis_agent import run_synthesis_agent
from agents.tax_agent import run_tax_agent


def intake_node(state: UplanState) -> dict:
    if not state["raw_purge_confirmed"]:
        raise RuntimeError("PRIVACY GATE: agents cannot run before raw purge is confirmed.")
    return {}


def build_graph():
    graph = StateGraph(UplanState)

    graph.add_node("intake", intake_node)
    graph.add_node("policy_distribution", distribute_policy)
    graph.add_node("financial_agent", run_financial_agent)
    graph.add_node("tax_agent", run_tax_agent)
    graph.add_node("narrative_synthesis", run_synthesis_agent)
    graph.add_node("adversarial_audit", run_adversarial_audit)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "policy_distribution")
    graph.add_edge("policy_distribution", "financial_agent")
    graph.add_edge("policy_distribution", "tax_agent")
    graph.add_edge(["financial_agent", "tax_agent"], "narrative_synthesis")
    graph.add_edge("narrative_synthesis", "adversarial_audit")
    graph.add_edge("adversarial_audit", END)

    return graph.compile()
