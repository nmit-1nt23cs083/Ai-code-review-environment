"""
validate_openenv.py

Pre-submission validation script. Checks all requirements from the hackathon rubric:
  ✓ openenv.yaml exists and has required fields
  ✓ All 3 tasks exist with graders that return [0,1] scores
  ✓ Graders are deterministic
  ✓ Required files exist (Dockerfile, inference.py, README.md)
  ✓ Environment class has reset/step/state methods
  ✓ Pydantic models defined for Observation, Action, StepResult

Run with: python validate_openenv.py
"""
import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

errors = []
warnings = []


def check(label: str, condition: bool, message: str = "", fatal: bool = True):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f" — {message}" if message else ""))
        if fatal:
            errors.append(label)
        else:
            warnings.append(label)


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print('─' * 55)


# -----------------------------------------------------------------------
# 1. File structure
# -----------------------------------------------------------------------
section("1. Required files")

required_files = [
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "inference.py",
    "README.md",
    "app/__init__.py",
    "app/main.py",
    "app/environment.py",
    "app/models/schemas.py",
    "app/tasks/task_definitions.py",
    "tests/test_environment.py",
]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f))


# -----------------------------------------------------------------------
# 2. openenv.yaml
# -----------------------------------------------------------------------
section("2. openenv.yaml content")

try:
    import yaml
    with open("openenv.yaml") as fh:
        config = yaml.safe_load(fh)

    for field in ["name", "version", "description", "observation", "action", "tasks", "endpoints"]:
        check(f"openenv.yaml has '{field}'", field in config)

    if "tasks" in config:
        check("openenv.yaml has exactly 3 tasks", len(config["tasks"]) == 3,
              f"found {len(config.get('tasks', []))}")

    difficulties = [t.get("difficulty") for t in config.get("tasks", [])]
    check("Tasks include easy difficulty",  "easy"   in difficulties)
    check("Tasks include medium difficulty","medium" in difficulties)
    check("Tasks include hard difficulty",  "hard"   in difficulties)

except ImportError:
    print(f"{WARN} pyyaml not installed; skipping YAML validation (ok for CI)")
    warnings.append("pyyaml not installed")


# -----------------------------------------------------------------------
# 3. Task definitions
# -----------------------------------------------------------------------
section("3. Task definitions and graders")

try:
    from app.tasks.task_definitions import ALL_TASKS, TASK_MAP, grade_episode

    check("Exactly 3 tasks defined", len(ALL_TASKS) == 3, f"found {len(ALL_TASKS)}")
    check("TASK_MAP populated",      len(TASK_MAP) == 3)

    for task in ALL_TASKS:
        name = task["name"]
        n_issues = len(task["issues"])
        check(f"Task '{name}' has ≥1 issue", n_issues >= 1)
        check(f"Task '{name}' has code",     bool(task.get("code", "").strip()))
        check(f"Task '{name}' has max_steps > 0", task.get("max_steps", 0) > 0)

        # Perfect actions
        perfect = [
            {"action_type": "FLAG_BUG",
             "line_number": i["line_number"],
             "issue_type":  i["issue_type"],
             "comment":     i["keywords"][0]}
            for i in task["issues"]
        ]
        score_perfect = grade_episode(task, perfect)
        score_empty   = grade_episode(task, [])

        check(f"Grader perfect=1.0 on '{name}'", abs(score_perfect - 1.0) < 0.001,
              f"got {score_perfect}")
        check(f"Grader empty=0.0 on '{name}'",   abs(score_empty - 0.0) < 0.001,
              f"got {score_empty}")
        check(f"Grader score in [0,1] on '{name}'", 0.0 <= score_perfect <= 1.0)

        # Determinism
        scores = [grade_episode(task, perfect) for _ in range(3)]
        check(f"Grader is deterministic on '{name}'", len(set(scores)) == 1)

        # Partial credit
        partial_score = grade_episode(task, [perfect[0]])
        check(f"Grader gives partial credit on '{name}'",
              0.0 < partial_score < 1.0, f"got {partial_score}")

except Exception as e:
    print(f"{FAIL} Task definitions failed to load: {e}")
    errors.append("task_definitions import failed")


# -----------------------------------------------------------------------
# 4. Schemas (without pydantic — check source structure)
# -----------------------------------------------------------------------
section("4. Pydantic model definitions")

try:
    schema_src = open("app/models/schemas.py").read()
    for model in ["Observation", "Action", "StepResult", "TaskInfo", "ResetResponse", "StateResponse"]:
        check(f"Model '{model}' defined in schemas.py", f"class {model}" in schema_src)
    for enum in ["ActionType", "IssueType"]:
        check(f"Enum '{enum}' defined", f"class {enum}" in schema_src)
    for field in ["code_snippet", "language", "action_type", "line_number", "reward", "done"]:
        check(f"Field '{field}' in schemas.py", field in schema_src)
except Exception as e:
    print(f"{FAIL} Could not read schemas.py: {e}")
    errors.append("schemas.py read failed")


# -----------------------------------------------------------------------
# 5. Environment class
# -----------------------------------------------------------------------
section("5. Environment class interface")

try:
    env_src = open("app/environment.py").read()
    for method in ["def reset", "def step", "def state", "def get_final_score"]:
        check(f"Environment has '{method}'", method in env_src)
    check("Environment uses CodeReviewEnv class", "class CodeReviewEnv" in env_src)
    check("Environment imports grade_episode", "grade_episode" in env_src)
except Exception as e:
    print(f"{FAIL} Could not read environment.py: {e}")
    errors.append("environment.py read failed")


# -----------------------------------------------------------------------
# 6. FastAPI endpoints
# -----------------------------------------------------------------------
section("6. FastAPI endpoint definitions")

try:
    main_src = open("app/main.py").read()
    for route in ['"/reset"', '"/step"', '"/state"', '"/health"', '"/tasks"', '"/score"']:
        check(f"Endpoint {route} defined", route in main_src)
    check("CORS middleware added", "CORSMiddleware" in main_src)
    check("FastAPI app created",   "FastAPI(" in main_src)
except Exception as e:
    print(f"{FAIL} Could not read main.py: {e}")
    errors.append("main.py read failed")


# -----------------------------------------------------------------------
# 7. Dockerfile
# -----------------------------------------------------------------------
section("7. Dockerfile")

try:
    dockerfile = open("Dockerfile").read()
    check("Dockerfile uses Python base image", "python" in dockerfile.lower())
    check("Dockerfile exposes port 7860",      "7860" in dockerfile)
    check("Dockerfile copies requirements.txt", "requirements.txt" in dockerfile)
    check("Dockerfile runs uvicorn",            "uvicorn" in dockerfile)
except Exception as e:
    print(f"{FAIL} Could not read Dockerfile: {e}")
    errors.append("Dockerfile read failed")


# -----------------------------------------------------------------------
# 8. inference.py
# -----------------------------------------------------------------------
section("8. inference.py")

try:
    inf_src = open("inference.py").read()
    for env_var in ["OPENAI_API_KEY", "API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        check(f"inference.py reads {env_var}", env_var in inf_src)
    for log_tag in ["[START]", "[STEP]", "[END]"]:
        check(f"inference.py logs {log_tag}", log_tag in inf_src)
    check("inference.py uses OpenAI client", "OpenAI(" in inf_src)
    check("inference.py has main() function", "def main" in inf_src)
    check("inference.py runs all 3 tasks",
          all(t["name"] in inf_src for t in ALL_TASKS))
except Exception as e:
    print(f"{FAIL} Could not read inference.py: {e}")
    errors.append("inference.py read failed")


# -----------------------------------------------------------------------
# 9. README
# -----------------------------------------------------------------------
section("9. README completeness")

try:
    readme = open("README.md", encoding="utf-8", errors="ignore").read()
    for section_kw in ["Action", "Observation", "Reward", "Task", "Baseline", "Docker", "Setup"]:
        check(f"README mentions '{section_kw}'", section_kw in readme, fatal=False)
except Exception as e:
    print(f"{FAIL} Could not read README.md: {e}")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'═' * 55}")
print(f"  VALIDATION SUMMARY")
print(f"{'═' * 55}")
total_checks = 43  # approximate
print(f"  Errors:   {len(errors)}")
print(f"  Warnings: {len(warnings)}")

if errors:
    print(f"\n  FAILED CHECKS:")
    for e in errors:
        print(f"    ✗ {e}")

if warnings:
    print(f"\n  WARNINGS:")
    for w in warnings:
        print(f"    ! {w}")

if not errors:
    print(f"\n  🎉 All critical checks passed! Ready to submit.")
    sys.exit(0)
else:
    print(f"\n  🚫 Fix the above errors before submitting.")
    sys.exit(1)