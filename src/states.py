from langgraph.graph import StateGraph
from typing import Optional, TypedDict, List, Union, Annotated
import operator


def _merge_logs(left: List[str], right: List[str]) -> List[str]:
    """Reducer that appends new log entries to existing ones."""
    return (left or []) + (right or [])


class AgentInput(TypedDict, total=False):
    image_input: Optional[bytearray]
    text_input: Optional[str]
    audio_input: Optional[bytes]

class ParseAgentOutput(TypedDict, total=False):
    problem_text: Optional[str]
    topic: Optional[str]
    variables: Optional[List[str]]
    constraints: Optional[List[str]]
    need_clarification: bool

class SolveAgentOutput(TypedDict, total=False):
    solution: Optional[str]
    answer: Optional[Union[str, int]]

class VerifyAgentOutput(TypedDict, total=False):
    confidence_score: Optional[int]
    message: Optional[str]

class ExplainAgentOutput(TypedDict, total=False):
    problem_statement: Optional[str]
    solution: Optional[str]
    context: Optional[str]


class StateSchema(TypedDict, total=False):
    agent_input: AgentInput
    parse_agent_output: ParseAgentOutput
    solve_agent_output: SolveAgentOutput
    verify_agent_output: VerifyAgentOutput
    explain_agent_output: ExplainAgentOutput
    # HITL fields
    clarification_action: Optional[str]       # "proceed" | "retry" | None
    # Progress tracking
    current_step: Optional[str]               # which agent is active
    # Logging
    logs: Annotated[List[str], _merge_logs]   # accumulated log lines