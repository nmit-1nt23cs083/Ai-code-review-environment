"""
Typed Pydantic models for the AI Code Review RL Environment.
"""
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    FLAG_BUG = "FLAG_BUG"
    SUGGEST_FIX = "SUGGEST_FIX"
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    ADD_COMMENT = "ADD_COMMENT"


class IssueType(str, Enum):
    SYNTAX = "syntax"
    LOGIC = "logic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DESIGN = "design"
    NONE = "none"


class CodeIssue(BaseModel):
    """Represents a known issue in the code sample."""
    line_number: int = Field(..., description="Line number where the issue occurs (1-indexed)")
    issue_type: IssueType = Field(..., description="Category of the issue")
    description: str = Field(..., description="Human-readable description of the issue")
    severity: str = Field(default="medium", description="low | medium | high | critical")


class Observation(BaseModel):
    """What the agent sees at each step."""
    code_snippet: str = Field(..., description="The code under review")
    language: str = Field(..., description="Programming language of the snippet")
    context: str = Field(..., description="Brief description / task context")
    step_number: int = Field(..., description="Current step in the episode")
    max_steps: int = Field(..., description="Maximum steps allowed in the episode")
    issues_found_so_far: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Issues the agent has correctly flagged so far"
    )
    remaining_issues_hint: int = Field(
        ..., description="Number of issues still not found (count only, no details)"
    )
    task_name: str = Field(..., description="Name of the current task")
    task_difficulty: str = Field(..., description="easy | medium | hard")


class Action(BaseModel):
    """An action the agent takes during a review step."""
    action_type: ActionType = Field(..., description="The type of review action")
    line_number: Optional[int] = Field(
        default=None,
        description="Line number being flagged or commented on (required for FLAG_BUG, SUGGEST_FIX, ADD_COMMENT)"
    )
    issue_type: Optional[IssueType] = Field(
        default=None,
        description="The type of issue being flagged (required for FLAG_BUG)"
    )
    comment: Optional[str] = Field(
        default=None,
        description="Free-text comment or suggestion"
    )


class StepResult(BaseModel):
    """Full result returned by step()."""
    observation: Observation
    reward: float = Field(..., description="Immediate reward for this action")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra diagnostic info")


class TaskInfo(BaseModel):
    """Metadata about a task."""
    name: str
    difficulty: str
    description: str
    total_issues: int
    max_steps: int


class ResetResponse(BaseModel):
    """Response from /reset endpoint."""
    observation: Observation
    task_info: TaskInfo


class StateResponse(BaseModel):
    """Response from /state endpoint."""
    observation: Observation
    total_reward: float
    steps_taken: int
    done: bool
    task_info: TaskInfo
