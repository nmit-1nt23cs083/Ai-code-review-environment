"""
inference.py – Baseline inference script for the AI Code Review RL Environment.

Reads config from environment variables:
  OPENAI_API_KEY   – API key (required)
  API_BASE_URL     – Base URL of the OpenEnv server (default: http://localhost:7860)
  MODEL_NAME       – Model to use (default: gpt-4o-mini)
  HF_TOKEN         – Hugging Face token (used if API_BASE_URL points to HF Space)

Logs in required format:
  [START] Task={task_name} Env=ai-code-review-env Model={model}
  [STEP]  step=N action=... reward=... done=...
  [END]   success=True/False steps=N score=0.XX rewards=[...]

Usage:
  export OPENAI_API_KEY=sk-...
  export API_BASE_URL=http://localhost:7860
  export MODEL_NAME=gpt-4o-mini
  python inference.py
"""

import os
import sys
import json
import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
GROQ_BASE_URL: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# Build OpenAI (Grok-compatible) client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=GROQ_BASE_URL)

# Validate model availability (log so you can fix MODEL_NAME early)
try:
    models = client.models.list()
    if hasattr(models, "data"):
        print(f"Grok client connected. Available models: {len(models.data)}")
    else:
        print("Grok client connected; model list response received.")
except Exception as e:
    print(f"WARNING: Could not list models from Grok endpoint: {e}")

ENV_NAME = "ai-code-review-env"
TASKS = [
    "fix_syntax_and_obvious_bugs",
    "logic_and_security_review",
    "design_and_architecture_review",
]

# ---------------------------------------------------------------------------
# System prompt for the agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert software engineer performing a code review.
You will be given a code snippet and must identify bugs, security issues, logic errors, and design problems.

For each step, respond with ONLY a valid JSON object - no markdown, no code fences, no extra text:
{
  "action_type": "FLAG_BUG",
  "line_number": 5,
  "issue_type": "syntax",
  "comment": "Syntax error: = used instead of == for equality check. Fix: change to =="
}

STRICT RULES:
- action_type MUST be one of: FLAG_BUG, SUGGEST_FIX, APPROVE, REQUEST_CHANGES, ADD_COMMENT
- issue_type MUST be one of: syntax, logic, security, performance, style, design, none  -- NEVER null or missing
- line_number MUST be an integer when using FLAG_BUG or SUGGEST_FIX
- comment MUST use technical terms that name the exact bug, for example:
    sql injection, off-by-one, race condition, mutable default argument,
    hardcoded password, resource leak, zerodivision, swallowed exception,
    srp violation, missing authentication, indexerror
- Flag ONE issue per response. Count lines from 1 (including blank lines).
- Only use APPROVE when remaining_issues_hint is 0.
- Output ONLY the JSON. Nothing else.
"""


def build_user_prompt(observation: dict) -> str:
    found = observation.get("issues_found_so_far", [])
    remaining = observation.get("remaining_issues_hint", 0)

    # Number the lines so the model references correct line numbers
    code = observation['code_snippet']
    numbered_lines = "\n".join(
        f"{i+1:3d} | {line}" for i, line in enumerate(code.splitlines())
    )

    found_text = ""
    if found:
        found_text = "\nIssues you have ALREADY correctly identified (do NOT flag these again):\n"
        for f in found:
            found_text += f"  - Line {f['line_number']}: [{f['issue_type']}] {f['description']}\n"

    approve_hint = ""
    if remaining == 0:
        approve_hint = "\nAll issues found! Use APPROVE now.\n"

    return f"""\
Task: {observation['task_name']} ({observation['task_difficulty']})
Context: {observation['context']}

Code to review ({observation['language']}) — line numbers shown on left:
{numbered_lines}
{found_text}{approve_hint}
Remaining issues NOT YET found: {remaining}
Step {observation['step_number']} of {observation['max_steps']}.

Find ONE new issue and respond with ONLY the JSON object. Use the exact line number from the code above.
"""



def call_env(method: str, path: str, payload: dict = None) -> dict:
    """Call the OpenEnv server."""
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    url = f"{API_BASE_URL}{path}"
    with httpx.Client(timeout=30) as http:
        if method == "POST":
            resp = http.post(url, json=payload or {}, headers=headers)
        else:
            resp = http.get(url, headers=headers)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"ERROR calling {url}: {exc.response.status_code} {exc.response.text}")
        raise
    return resp.json()


def parse_action(raw_text: str) -> dict:
    """Parse JSON action from model output, sanitizing all fields."""
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    # Extract first {...} block if model added surrounding text
    if "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        text = text[start:end]

    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        return {
            "action_type": "REQUEST_CHANGES",
            "line_number": None,
            "issue_type": "NONE",
            "comment": raw_text[:200],
        }

    # --- sanitize issue_type ---
    valid_issue_types = {"syntax", "logic", "security", "performance", "style", "design", "none"}
    if not action.get("issue_type") or str(action.get("issue_type")).lower() not in valid_issue_types:
        action["issue_type"] = "none"
    else:
        action["issue_type"] = str(action["issue_type"]).lower()

    # --- sanitize line_number ---
    ln = action.get("line_number")
    if ln is not None:
        try:
            action["line_number"] = int(ln)
        except (ValueError, TypeError):
            action["line_number"] = None

    # --- NEW FIX: enforce required fields per action_type ---
    action_type = action.get("action_type")

    valid_action_types = {"FLAG_BUG", "SUGGEST_FIX", "APPROVE", "REQUEST_CHANGES", "ADD_COMMENT"}
    if action_type not in valid_action_types:
        action_type = "REQUEST_CHANGES"

    if action_type == "FLAG_BUG":
        if action.get("line_number") is None:
            action["line_number"] = 1
        if action.get("issue_type") == "none":
            action["issue_type"] = "logic"
        if not action.get("comment"):
            action["comment"] = "Detected issue in code."

    elif action_type == "SUGGEST_FIX":
        if action.get("line_number") is None:
            action["line_number"] = 1
        if not action.get("comment"):
            action["comment"] = "Suggested fix."

    elif action_type == "ADD_COMMENT":
        if action.get("line_number") is None:
            action["line_number"] = 1
        if not action.get("comment"):
            action["comment"] = "General comment."

    elif action_type in {"APPROVE", "REQUEST_CHANGES"}:
        action["line_number"] = None
        action["issue_type"] = "none"

    action["action_type"] = action_type

    return action


def run_task(task_name: str) -> dict:
    """
    Run a single episode on the given task.
    Returns dict with {task_name, score, steps, rewards, success}.
    """
    # Reset
    reset_resp = call_env("POST", "/reset", {"task_name": task_name})
    obs = reset_resp["observation"]
    task_info = reset_resp["task_info"]

    print(f"[START] Task={task_name} Env={ENV_NAME} Model={MODEL_NAME}")

    conversation = []
    rewards = []
    steps = 0
    done = False

    while not done:
        # Build prompt
        user_msg = build_user_prompt(obs)
        conversation.append({"role": "user", "content": user_msg})

        # Ask LLM
        # Truncate conversation to avoid TPM limits: keep system + last 4 messages (2 exchanges)
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
        if len(full_messages) > 5:  # system + 4 messages
            full_messages = full_messages[:1] + full_messages[-4:]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            temperature=0,
            max_tokens=256,
        )
        raw = completion.choices[0].message.content
        conversation.append({"role": "assistant", "content": raw})

        # Parse action
        action_dict = parse_action(raw)

        # Step environment
        step_resp = call_env("POST", "/step", action_dict)
        reward = step_resp["reward"]
        done = step_resp["done"]
        obs = step_resp["observation"]
        steps += 1
        rewards.append(reward)

        print(
            f"[STEP] step={steps} "
            f"action={action_dict.get('action_type')} "
            f"line={action_dict.get('line_number')} "
            f"reward={reward:.2f} "
            f"done={done}"
        )

    # Get final score from grader
    score_resp = call_env("GET", "/score")
    score = score_resp.get("score", 0.0)
    success = score >= 0.8

    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}"
    )
    print()

    return {
        "task_name": task_name,
        "score": score,
        "steps": steps,
        "rewards": rewards,
        "success": success,
    }


def main():
    print(f"=== AI Code Review Baseline Inference ===")
    print(f"Environment: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print()

    all_results = []
    for task_name in TASKS:
        result = run_task(task_name)
        all_results.append(result)

    # Summary
    print("=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    total_score = 0.0
    for r in all_results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"  {status}  {r['task_name']:<40}  score={r['score']:.4f}  steps={r['steps']}")
        total_score += r["score"]

    avg_score = total_score / len(all_results)
    print(f"\n  Average Score: {avg_score:.4f}")
    print("=" * 50)

    # Final normalized score (0-1)
    print(f"\nFINAL_SCORE={avg_score:.4f}")


if __name__ == "__main__":
    main()