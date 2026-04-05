"""
Gym environment wrapper for AI Code Review RL training.
Interfaces with the FastAPI server at http://localhost:7860.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import json
from typing import Any, Dict, Tuple


class CodeReviewEnv(gym.Env):
    """
    Gym environment for training RL agents on code review.
    
    Action space:
      - action_type: 0=FLAG_BUG, 1=SUGGEST_FIX, 2=APPROVE, 3=REQUEST_CHANGES, 4=ADD_COMMENT
      - line_number: 0-49 (maps to 1-50)
      - issue_type: 0=syntax, 1=logic, 2=security, 3=performance, 4=style, 5=design, 6=none
      - comment_summary: 0-99 (discretized comment embeddings or predefined phrases)
    
    Observation space:
      - code_lines: 50 lines of code, each encoded as small int
      - step_number: current step (0-49)
      - max_steps: max allowed steps (0-49)
      - remaining_issues: count of unfound issues (0-20)
      - task_difficulty: 0=easy, 1=medium, 2=hard
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, base_url: str = "http://localhost:7860", task_name: str = None):
        super().__init__()
        self.base_url = base_url
        self.task_name = task_name
        self.current_obs = None
        self.episode_steps = 0
        
        # Action space: Single Discrete with 250 actions
        # Maps: action = action_type * 50 + line_number
        # issue_type and comment are hardcoded intelligently
        self.action_space = spaces.Discrete(250)  # 5 action types × 50 lines
        
        # Observation space
        self.observation_space = spaces.Dict({
            "code_length": spaces.Box(low=0, high=2000, shape=(1,), dtype=np.int32),
            "step_number": spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
            "max_steps": spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
            "remaining_issues": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "difficulty": spaces.Discrete(3),  # 0=easy, 1=medium, 2=hard
            "found_count": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
        })
        
        # Predefined comment templates
        self.comment_templates = [
            "Bug detected",
            "Logic error found",
            "Security vulnerability",
            "Performance issue",
            "Style violation",
            "Design flaw",
            "Potential defect",
            "Code smell",
            "Issue detected",
            "Problem found",
        ]
    
    def reset(self, seed=None, options=None):
        """Reset environment and start new episode."""
        super().reset(seed=seed)
        try:
            resp = requests.post(f"{self.base_url}/reset", json={"task_name": self.task_name}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.current_obs = data["observation"]
            self.episode_steps = 0
            return self._encode_observation(self.current_obs), {}
        except Exception as e:
            print(f"Reset error: {e}")
            raise
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one action and return (obs, reward, terminated, truncated, info)."""
        if self.current_obs is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Decode single Discrete action → (action_type, line_number, issue_type, comment)
        action = int(action)
        action_type_idx = action // 50
        line_number = (action % 50) + 1  # Convert to 1-50
        
        action_types = ["FLAG_BUG", "SUGGEST_FIX", "APPROVE", "REQUEST_CHANGES", "ADD_COMMENT"]
        action_type = action_types[min(action_type_idx, 4)]
        
        issue_types = ["syntax", "logic", "security", "performance", "style", "design", "none"]
        issue_type = issue_types[action % 7]  # Rotate through issue types
        
        comment = self.comment_templates[action % len(self.comment_templates)]
        
        # Build action dict for API
        action_dict = {
            "action_type": action_type,
            "line_number": line_number if action_type in ["FLAG_BUG", "SUGGEST_FIX", "ADD_COMMENT"] else None,
            "issue_type": issue_type if action_type in ["FLAG_BUG", "SUGGEST_FIX"] else None,
            "comment": comment,
        }
        
        try:
            resp = requests.post(f"{self.base_url}/step", json=action_dict, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.current_obs = data["observation"]
            reward = data["reward"]
            done = data["done"]
            self.episode_steps += 1
            
            encoded_obs = self._encode_observation(self.current_obs)
            truncated = self.episode_steps >= self.current_obs.get("max_steps", 20)
            
            info = {"raw_reward": reward, "done": done}
            if done:
                try:
                    score_resp = requests.get(f"{self.base_url}/score", timeout=10)
                    score_resp.raise_for_status()
                    score_data = score_resp.json()
                    info["accuracy_score"] = score_data.get("score")
                except Exception:
                    info["accuracy_score"] = None
            return encoded_obs, float(reward), done, truncated, info
        except Exception as e:
            print(f"Step error: {e}")
            reward = -1.0
            terminated = True
            truncated = False
            info = {"error": str(e)}
            return self._encode_observation(self.current_obs), reward, terminated, truncated, info
    
    def _encode_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert API observation to Gym observation space."""
        code_length = len(obs.get("code_snippet", ""))
        step_number = obs.get("step_number", 0)
        max_steps = obs.get("max_steps", 20)
        remaining = obs.get("remaining_issues_hint", 0)
        difficulty = {"easy": 0, "medium": 1, "hard": 2}.get(obs.get("task_difficulty", "easy"), 0)
        found_count = len(obs.get("issues_found_so_far", []))
        
        return {
            "code_length": np.array([code_length], dtype=np.int32),
            "step_number": np.array([step_number], dtype=np.int32),
            "max_steps": np.array([max_steps], dtype=np.int32),
            "remaining_issues": np.array([remaining], dtype=np.int32),
            "difficulty": difficulty,
            "found_count": np.array([found_count], dtype=np.int32),
        }
    
    def render(self):
        """Print current observation (simple render)."""
        if self.current_obs:
            print(f"Step {self.current_obs.get('step_number', 0)}/{self.current_obs.get('max_steps', 0)}")
            print(f"Remaining issues: {self.current_obs.get('remaining_issues_hint', 0)}")
            print(f"Found so far: {len(self.current_obs.get('issues_found_so_far', []))}")


if __name__ == "__main__":
    # Quick test
    env = CodeReviewEnv()
    obs, info = env.reset()
    print(f"Initial obs: {obs}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result: obs={obs}, reward={reward}, done={terminated}")
