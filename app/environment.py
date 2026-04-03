"""
CodeReviewEnv – Core RL environment implementing OpenEnv interface.

Implements:
  reset()  → ResetResponse
  step()   → StepResult
  state()  → StateResponse
"""
from typing import Any, Dict, List, Optional
import copy

from app.models.schemas import (
    Action, ActionType, IssueType,
    Observation, StepResult, TaskInfo,
    ResetResponse, StateResponse,
)
from app.tasks.task_definitions import ALL_TASKS, TASK_MAP, grade_episode


class CodeReviewEnv:
    """
    A closed-loop RL environment where an AI agent reviews code snippets,
    flags bugs, suggests fixes, and approves clean code.

    Episode lifecycle:
      1. reset(task_name) initialises state for a task.
      2. step(action) evaluates each action and returns (obs, reward, done, info).
      3. state() returns full current state at any time.
      4. grade_episode() gives a final 0-1 score.
    """

    def __init__(self) -> None:
        self._task: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._action_history: List[Dict[str, Any]] = []
        self._found_issue_indices: set = set()
        self._current_task_index: int = 0  # cycles through easy→medium→hard

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_name: Optional[str] = None) -> ResetResponse:
        """
        Initialise a new episode.

        Args:
            task_name: Name of the task to load. If None, cycles through
                       tasks in order (easy → medium → hard → easy …).

        Returns:
            ResetResponse with initial Observation and TaskInfo.
        """
        if task_name and task_name in TASK_MAP:
            self._task = copy.deepcopy(TASK_MAP[task_name])
        else:
            self._task = copy.deepcopy(ALL_TASKS[self._current_task_index % len(ALL_TASKS)])
            self._current_task_index += 1

        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._action_history = []
        self._found_issue_indices = set()

        obs = self._build_observation()
        task_info = self._build_task_info()
        return ResetResponse(observation=obs, task_info=task_info)

    def step(self, action: Action) -> StepResult:
        """
        Execute one action in the environment.

        Reward shaping:
          +1.0   correct FLAG_BUG / SUGGEST_FIX on right line & matching keywords
          +0.5   correct line but only type match (partial credit)
          -0.2   FLAG_BUG on wrong line / incorrect type
          +1.5   APPROVE when no issues remain (bonus for clean close)
          -0.5   APPROVE when issues still remain (premature approval)
          -0.1   ADD_COMMENT (neutral / slight cost to avoid spam)
          +0.0   REQUEST_CHANGES (neutral; no info gained)

        Returns:
            StepResult with updated observation, reward, done flag, and info dict.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        reward, info = self._evaluate_action(action)
        self._total_reward += reward

        # Record action for grader
        self._action_history.append({
            "action_type": action.action_type,
            "line_number": action.line_number,
            "issue_type": action.issue_type,
            "comment": action.comment or "",
            "reward": reward,
        })

        # Episode ends when all issues found, APPROVE after all found, or step limit
        all_found = len(self._found_issue_indices) == len(self._task["issues"])
        step_limit = self._step_count >= self._task["max_steps"]
        approved = action.action_type == ActionType.APPROVE

        if approved or all_found or step_limit:
            self._done = True
            # Final score bonus: reward correct completion
            if all_found and (approved or step_limit):
                bonus = 0.5
                reward += bonus
                self._total_reward += bonus
                info["completion_bonus"] = bonus

        obs = self._build_observation()
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> StateResponse:
        """Return full current state of the environment."""
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return StateResponse(
            observation=self._build_observation(),
            total_reward=round(self._total_reward, 4),
            steps_taken=self._step_count,
            done=self._done,
            task_info=self._build_task_info(),
        )

    def get_final_score(self) -> float:
        """Run the deterministic grader and return 0-1 score for the episode."""
        if self._task is None:
            return 0.0
        return grade_episode(self._task, self._action_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_action(self, action: Action):
        """Score a single action against the current task's expected issues."""
        issues = self._task["issues"]
        reward = 0.0
        info: Dict[str, Any] = {"action": action.action_type, "matched_issue": None}

        if action.action_type == ActionType.APPROVE:
            remaining = len(issues) - len(self._found_issue_indices)
            if remaining == 0:
                reward = 1.5
                info["message"] = "Correct APPROVE: no issues remain."
            else:
                reward = -0.5
                info["message"] = f"Premature APPROVE: {remaining} issue(s) still unfound."
            return reward, info

        if action.action_type == ActionType.ADD_COMMENT:
            reward = -0.1
            info["message"] = "Comment noted (no scoring impact)."
            return reward, info

        if action.action_type == ActionType.REQUEST_CHANGES:
            reward = 0.0
            info["message"] = "Request for changes noted."
            return reward, info

        # FLAG_BUG or SUGGEST_FIX — check against expected issues
        if action.line_number is None:
            reward = -0.2
            info["message"] = "FLAG_BUG/SUGGEST_FIX requires a line_number."
            return reward, info

        for idx, issue in enumerate(issues):
            if idx in self._found_issue_indices:
                continue  # already found

            line_match = (action.line_number == issue["line_number"])
            keyword_hit = self._keyword_match(action.comment or "", issue["keywords"])
            type_match = (
                action.issue_type is not None
                and action.issue_type.value == issue["issue_type"]
            )

            if line_match and (keyword_hit or type_match):
                self._found_issue_indices.add(idx)
                reward = 1.0
                info["matched_issue"] = issue["description"]
                info["message"] = f"Correct! Issue on line {action.line_number} identified."
                return reward, info

            if line_match:
                # Right line, wrong keywords/type → partial
                reward = 0.3
                info["message"] = f"Partially correct: right line ({action.line_number}) but description unclear."
                return reward, info

        # Wrong line
        reward = -0.2
        info["message"] = f"No expected issue on line {action.line_number}."
        return reward, info

    @staticmethod
    def _keyword_match(comment: str, keywords: List[str]) -> bool:
        if not comment:
            return False
        c = comment.lower()
        return any(kw in c for kw in keywords)

    def _build_observation(self) -> Observation:
        remaining = len(self._task["issues"]) - len(self._found_issue_indices)
        found_list = [
            {
                "line_number": self._task["issues"][i]["line_number"],
                "issue_type": self._task["issues"][i]["issue_type"],
                "description": self._task["issues"][i]["description"],
            }
            for i in self._found_issue_indices
        ]
        return Observation(
            code_snippet=self._task["code"],
            language=self._task["language"],
            context=self._task["description"],
            step_number=self._step_count,
            max_steps=self._task["max_steps"],
            issues_found_so_far=found_list,
            remaining_issues_hint=remaining,
            task_name=self._task["name"],
            task_difficulty=self._task["difficulty"],
        )

    def _build_task_info(self) -> TaskInfo:
        return TaskInfo(
            name=self._task["name"],
            difficulty=self._task["difficulty"],
            description=self._task["description"],
            total_issues=len(self._task["issues"]),
            max_steps=self._task["max_steps"],
        )
