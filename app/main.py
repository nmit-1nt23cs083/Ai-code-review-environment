"""
FastAPI server exposing the OpenEnv interface for the AI Code Review environment.

Endpoints:
  POST /reset          – Start a new episode
  POST /step           – Execute an action
  GET  /state          – Inspect current state
  GET  /tasks          – List available tasks
  GET  /health         – Health check
  GET  /               – Environment info
"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment import CodeReviewEnv
from app.models.schemas import Action, IssueType, StepResult, ResetResponse, StateResponse

app = FastAPI(
    title="AI Code Review RL Environment",
    description=(
        "A realistic reinforcement learning environment where an AI agent "
        "reviews code snippets, flags bugs, and suggests fixes. "
        "Implements the OpenEnv interface (reset/step/state)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per-server; for production use session IDs)
env = CodeReviewEnv()


# ------------------------------------------------------------------
# Request / response bodies
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = None


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/")
def root():
    """Environment information."""
    return {
        "name": "ai-code-review-env",
        "version": "1.0.0",
        "description": "RL environment for AI-driven code review",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
        "openenv_spec": "1.0",
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    from app.tasks.task_definitions import ALL_TASKS
    return {
        "tasks": [
            {
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "total_issues": len(t["issues"]),
                "max_steps": t["max_steps"],
            }
            for t in ALL_TASKS
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    - Pass `task_name` to select a specific task.
    - If omitted, cycles through easy → medium → hard.
    """
    try:
        result = env.reset(task_name=request.task_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """
    Execute one action in the current episode.

    Action types:
      - FLAG_BUG: flag a bug at a specific line (requires line_number, issue_type, comment)
      - SUGGEST_FIX: suggest a fix at a specific line (requires line_number, comment)
      - APPROVE: approve the code as-is
      - REQUEST_CHANGES: request general changes (no line needed)
      - ADD_COMMENT: add a general comment (no scoring impact)
    """
    try:
        # Sanitize: default issue_type to NONE if not provided
        if action.issue_type is None:
            action.issue_type = IssueType.NONE
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state():
    """Inspect the current state of the environment."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/score")
def get_score():
    """Get the final grader score for the current episode (0.0 – 1.0)."""
    score = env.get_final_score()
    return {"score": score, "note": "deterministic grader result"}