"""
trained_inference.py – Run the trained RL agent on all 3 tasks.

Loads the PPO model and runs inference on the environment, logging results.

Usage:
  python trained_inference.py
  
Output:
  Logs [START], [STEP], [END] for each task
  Computes final score from grader
  Prints summary results
"""
import os
import sys
import httpx
from stable_baselines3 import PPO
from train_env import CodeReviewEnv


ENV_NAME = "ai-code-review-env"
TASKS = [
    "fix_syntax_and_obvious_bugs",
    "logic_and_security_review",
    "design_and_architecture_review",
]


def call_env(method: str, path: str, payload: dict = None) -> dict:
    """Call the OpenEnv server."""
    url = f"http://localhost:7860{path}"
    with httpx.Client(timeout=30) as http:
        if method == "POST":
            resp = http.post(url, json=payload or {})
        else:
            resp = http.get(url)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"ERROR calling {url}: {exc.response.status_code} {exc.response.text}")
        raise
    return resp.json()


def run_task_with_model(model: PPO, task_name: str) -> dict:
    """
    Run a single episode using the trained RL model.
    Returns dict with {task_name, score, steps, rewards, success}.
    """
    env = CodeReviewEnv(task_name=task_name)
    obs, _ = env.reset()
    
    print(f"[START] Task={task_name} Env={ENV_NAME} Model=PPO-trained")
    
    rewards = []
    steps = 0
    done = False
    
    while not done and steps < 20:  # 20 step limit
        # Get action from trained model
        action_int, _ = model.predict(obs, deterministic=True)
        
        # Step environment (pass int action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated
        steps += 1
        rewards.append(reward)
        
        # Decode action for logging
        action_int = int(action_int)
        action_type_idx = action_int // 50
        line_number = (action_int % 50) + 1
        
        action_types = ["FLAG_BUG", "SUGGEST_FIX", "APPROVE", "REQUEST_CHANGES", "ADD_COMMENT"]
        action_type = action_types[min(action_type_idx, 4)]
        
        print(
            f"[STEP] step={steps} "
            f"action={action_type} "
            f"line={line_number} "
            f"reward={reward:.2f} "
            f"done={done}"
        )
    
    # Get final score
    score_resp = call_env("GET", "/score")
    score = score_resp.get("score", 0.0)
    success = score >= 0.8
    
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}"
    )
    print()
    
    env.close()
    
    return {
        "task_name": task_name,
        "score": score,
        "steps": steps,
        "rewards": rewards,
        "success": success,
    }


def main():
    """Run trained agent on all tasks."""
    print("=" * 70)
    print("=== AI Code Review RL Training - Trained Agent Inference ===")
    print("=" * 70)
    print()
    
    # Load trained model
    model_path = "ppo_code_review_models/ppo_codereview_final"
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ ERROR: Trained model not found at {model_path}.zip")
        print("Run 'python train.py' first to train the agent.")
        sys.exit(1)
    
    print(f"📦 Loading trained model from: {model_path}.zip")
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print()
    
    # Run all tasks
    results = []
    for task in TASKS:
        result = run_task_with_model(model, task)
        results.append(result)
    
    # Print summary
    print("=" * 70)
    print("TRAINED AGENT RESULTS SUMMARY")
    print("=" * 70)
    
    for result in results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {status}  {result['task_name']:35s} score={result['score']:.4f}  steps={result['steps']:2d}")
    
    avg_score = sum(r["score"] for r in results) / len(results)
    print()
    print(f"  Average Score: {avg_score:.4f}")
    print("=" * 70)
    print()
    print(f"TRAINED_FINAL_SCORE={avg_score:.4f}")


if __name__ == "__main__":
    main()
