from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.financial_agent import run_financial_agent
from agents.state import UplanState
from agents.tax_agent import run_tax_agent


def build_graph():
    graph = StateGraph(UplanState)

    graph.add_node("financial_agent", run_financial_agent)
    graph.add_node("tax_agent", run_tax_agent)

    # Parallel fan-out. The findings reducer in UplanState merges outputs.
    graph.add_edge(START, "financial_agent")
    graph.add_edge(START, "tax_agent")
    graph.add_edge("financial_agent", END)
    graph.add_edge("tax_agent", END)

    return graph.compile()
