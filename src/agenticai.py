import logging
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from types import SimpleNamespace
from agents import parser_agent, solver_agent, verifier_agent, explanation_agent
from states import StateSchema

logger = logging.getLogger("math_solver.graph")


# ── Helpers ────────────────────────────────────────────────────────────

# Default attribute maps for each state sub-object
_DEFAULTS = {
    "agent_input":          {"image_input": None, "text_input": None, "audio_input": None},
    "parse_agent_output":   {"problem_text": "", "topic": "", "variables": [], "constraints": [], "need_clarification": False},
    "solve_agent_output":   {"solution": "", "answer": None},
    "verify_agent_output":  {"confidence_score": None, "message": ""},
    "explain_agent_output": {"problem_statement": "", "solution": "", "context": ""},
}


def _to_namespace(value, key=None):
    """Convert a dict/None to SimpleNamespace with sensible defaults."""
    defaults = _DEFAULTS.get(key, {})
    if value is None:
        return SimpleNamespace(**defaults)
    if isinstance(value, SimpleNamespace):
        # Ensure missing attrs get defaults
        for k, v in defaults.items():
            if not hasattr(value, k):
                setattr(value, k, v)
        return value
    if isinstance(value, dict):
        merged = {**defaults, **value}
        return SimpleNamespace(**merged)
    return value


def _to_dict(value):
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value


# ── Graph nodes ────────────────────────────────────────────────────────

def parser_node(state: StateSchema):
    """Run the parser agent and capture results + logs."""
    agent_input = _to_namespace(state.get("agent_input"), "agent_input")
    parse_output = _to_namespace(state.get("parse_agent_output"), "parse_agent_output")

    result = parser_agent.agent_parser(agent_input, parse_output)
    logs = result.get("logs", [])

    return {
        "parse_agent_output": _to_dict(parse_output),
        "current_step": "parser",
        "logs": logs,
    }


def hitl_wait_node(state: StateSchema):
    """Pause point for HITL — the graph interrupts here.
    When resumed, clarification_action will be set by the FastAPI endpoint."""
    logger.info("HITL wait node reached — graph will interrupt")
    return {"current_step": "awaiting_clarification"}


def solver_node(state: StateSchema):
    parse_output = _to_namespace(state.get("parse_agent_output"), "parse_agent_output")
    solve_output = _to_namespace(state.get("solve_agent_output"), "solve_agent_output")

    result = solver_agent.solve_agent(solve_output, parse_output)
    logs = result.get("logs", [])

    return {
        "solve_agent_output": _to_dict(solve_output),
        "current_step": "solver",
        "logs": logs,
    }


def verifier_node(state: StateSchema):
    parse_output = _to_namespace(state.get("parse_agent_output"), "parse_agent_output")
    solve_output = _to_namespace(state.get("solve_agent_output"), "solve_agent_output")
    verify_output = _to_namespace(state.get("verify_agent_output"), "verify_agent_output")

    result = verifier_agent.verify_agent(solve_output, verify_output, parse_output)
    logs = result.get("logs", [])

    return {
        "verify_agent_output": _to_dict(verify_output),
        "current_step": "verifier",
        "logs": logs,
    }


def explanation_node(state: StateSchema):
    parse_output = _to_namespace(state.get("parse_agent_output"), "parse_agent_output")
    solve_output = _to_namespace(state.get("solve_agent_output"), "solve_agent_output")
    verify_output = _to_namespace(state.get("verify_agent_output"), "verify_agent_output")
    explain_output = _to_namespace(state.get("explain_agent_output"), "explain_agent_output")

    result = explanation_agent.explain_agent(explain_output, verify_output, parse_output, solve_output)
    logs = result.get("logs", [])

    return {
        "explain_agent_output": _to_dict(explain_output),
        "current_step": "explanation",
        "logs": logs,
    }


# ── Conditional edges ─────────────────────────────────────────────────

def after_parser(state: StateSchema) -> str:
    """Route after parser: if clarification needed → HITL wait, else → solver."""
    parse_out = state.get("parse_agent_output", {})
    needs = parse_out.get("need_clarification", False)
    if needs in (True, "True", "true"):
        logger.info("Clarification needed — routing to HITL wait")
        return "hitl_wait"
    logger.info("No clarification needed — routing to solver")
    return "solver"


def after_hitl(state: StateSchema) -> str:
    """Route after HITL: proceed → solver, retry → re-parse."""
    action = state.get("clarification_action", "proceed")
    if action == "retry":
        logger.info("User chose retry — re-running parser")
        return "parser"
    logger.info("User chose proceed — moving to solver")
    return "solver"


# ── Build the graph ───────────────────────────────────────────────────

def build_graph():
    """Build and compile the multi-agent graph with HITL interrupt."""
    workflow = StateGraph(StateSchema)

    # Nodes
    workflow.add_node("parser", parser_node)
    workflow.add_node("hitl_wait", hitl_wait_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("explanation", explanation_node)

    # Edges
    workflow.add_edge(START, "parser")
    workflow.add_conditional_edges("parser", after_parser, {"hitl_wait": "hitl_wait", "solver": "solver"})
    workflow.add_conditional_edges("hitl_wait", after_hitl, {"parser": "parser", "solver": "solver"})
    workflow.add_edge("solver", "verifier")
    workflow.add_edge("verifier", "explanation")
    workflow.add_edge("explanation", END)

    # Compile with checkpointer for HITL interrupt support
    checkpointer = MemorySaver()
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_wait"],
    )
    return compiled, checkpointer


app, checkpointer = build_graph()


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    text_input = input("Enter math problem: ").strip()
    if not text_input:
        print("No input provided.")
        return

    config = {"configurable": {"thread_id": "cli-session"}}
    state = {
        "agent_input": {
            "text_input": text_input,
            "image_input": None,
            "audio_input": None,
        }
    }

    # First invocation — may pause at hitl_wait
    result = app.invoke(state, config=config)

    # Check if interrupted for clarification
    snapshot = app.get_state(config)
    while snapshot.next:
        print("\n⚠️  The parser needs clarification.")
        print(f"Parsed problem: {result.get('parse_agent_output', {})}")
        action = input("Type 'proceed' to continue or 'retry' to re-parse: ").strip().lower()
        if action not in ("proceed", "retry"):
            action = "proceed"

        app.update_state(config, {"clarification_action": action})
        result = app.invoke(None, config=config)
        snapshot = app.get_state(config)

    print("\n📊 Final Result:")
    print(result)


if __name__ == "__main__":
    main()