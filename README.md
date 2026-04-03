# 🔍 AI Code Review RL Environment

A **realistic reinforcement learning environment** built for the Scalar OpenEnv Hackathon. An AI agent acts as a code reviewer: it receives code snippets containing real bugs and must flag, explain, and fix issues — earning rewards for accuracy and losing points for false alarms or premature approvals.

\---

## 🎯 Motivation

Every engineering team does code review. It's a high-value, well-defined task with:

* Clear correct/incorrect states (expected issues are ground-truth)
* Progressive difficulty (syntax → logic → architecture)
* Deterministic grading (no human judgment needed)
* Rich reward signal (partial credit, severity weighting)

This environment teaches an AI agent to review code the way a senior engineer would: systematically, precisely, and with clear explanations.

\---

## 🏗️ Environment Structure

```
code-review-env/
├── app/
│   ├── main.py              # FastAPI server (OpenEnv endpoints)
│   ├── environment.py       # Core RL environment class
│   ├── models/
│   │   └── schemas.py       # Pydantic typed models
│   └── tasks/
│       └── task\_definitions.py  # 3 tasks + deterministic graders
├── inference.py             # Baseline agent script
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container for HF Spaces
├── requirements.txt
└── README.md
```

\---

## 🔄 OpenEnv Interface

|Endpoint|Method|Description|
|-|-|-|
|`/reset`|POST|Start a new episode (optionally pass `task\_name`)|
|`/step`|POST|Execute one action, get reward|
|`/state`|GET|Inspect current state|
|`/tasks`|GET|List all available tasks|
|`/score`|GET|Get grader score (0–1) for current episode|
|`/health`|GET|Health check|

\---

## 📦 Observation Space

```json
{
  "code\_snippet": "def calculate\_average(numbers): ...",
  "language": "python",
  "context": "Review this function for bugs",
  "step\_number": 2,
  "max\_steps": 8,
  "issues\_found\_so\_far": \[
    {"line\_number": 3, "issue\_type": "logic", "description": "Off-by-one error"}
  ],
  "remaining\_issues\_hint": 2,
  "task\_name": "fix\_syntax\_and\_obvious\_bugs",
  "task\_difficulty": "easy"
}
```

\---

## 🎮 Action Space

```json
{
  "action\_type": "FLAG\_BUG",
  "line\_number": 5,
  "issue\_type": "syntax",
  "comment": "Syntax error: assignment '=' used instead of equality '==' in if condition"
}
```

**Action types:**

|Type|Description|Requires|
|-|-|-|
|`FLAG\_BUG`|Flag a bug at a specific line|`line\_number`, `issue\_type`, `comment`|
|`SUGGEST\_FIX`|Suggest a fix at a specific line|`line\_number`, `comment`|
|`APPROVE`|Approve code as-is|—|
|`REQUEST\_CHANGES`|Request general changes|optional `comment`|
|`ADD\_COMMENT`|Add a general note|optional|

\---

## 💰 Reward Function

|Situation|Reward|
|-|-|
|Correct FLAG\_BUG / SUGGEST\_FIX (right line + keyword match in comment)|**+1.0**|
|Right line but comment lacks specifics|**+0.3**|
|Wrong line (no expected issue there)|**-0.2**|
|APPROVE when all issues found|**+1.5**|
|APPROVE when issues still remain|**-0.5**|
|Correct completion bonus|**+0.5**|
|ADD\_COMMENT (no scoring value)|**-0.1**|

Rewards are shaped to give **continuous feedback** throughout the episode, not just at the end.

\---

## 🧪 Tasks

### Task 1 — Easy: `fix\_syntax\_and\_obvious\_bugs`

* **Code:** A short Python function to calculate an average
* **Issues (3):**

  1. Off-by-one error in `range()` → `IndexError`
  2. Syntax error: `=` instead of `==` in `if` condition
  3. Potential `ZeroDivisionError` for empty input
* **Max steps:** 8
* **Grader:** `issues\_correctly\_found / 3`

### Task 2 — Medium: `logic\_and\_security\_review`

* **Code:** A Flask-style route handler querying a database
* **Issues (4):**

  1. SQL Injection via f-string interpolation (critical)
  2. No authentication check before returning user data
  3. Implicit `None` return instead of 404 response
  4. Database connection never closed (resource leak)
* **Max steps:** 12
* **Grader:** `issues\_correctly\_found / 4`

### Task 3 — Hard: `design\_and\_architecture\_review`

* **Code:** An `OrderProcessor` class managing a pipeline
* **Issues (5):**

  1. Hardcoded database password in source code
  2. Mutable default argument (`pending\_orders=\[]`)
  3. SRP violation: email sending inside order processor
  4. Swallowed exception with bare `except: pass`
  5. Thread-safety bug: read-modify-write without lock
* **Max steps:** 18
* **Grader:** `issues\_correctly\_found / 5`

\---

## 📊 Baseline Scores

Achieved with `MODEL\_NAME=gpt-4o-mini`:

|Task|Difficulty|Score|Steps|
|-|-|-|-|
|fix\_syntax\_and\_obvious\_bugs|Easy|\~0.83|5|
|logic\_and\_security\_review|Medium|\~0.75|10|
|design\_and\_architecture\_review|Hard|\~0.60|16|
|**Average**|—|**\~0.73**|—|

> Scores are reproducible with `temperature=0` on the LLM calls.

\---

## 🚀 Setup \& Usage

### Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# In another terminal, run the baseline agent
export OPENAI\_API\_KEY=sk-...
export API\_BASE\_URL=http://localhost:7860
export MODEL\_NAME=gpt-4o-mini
python inference.py
```

### Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

### Manual API test

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \\
  -H "Content-Type: application/json" \\
  -d '{"task\_name": "fix\_syntax\_and\_obvious\_bugs"}'

# Take a step
curl -X POST http://localhost:7860/step \\
  -H "Content-Type: application/json" \\
  -d '{
    "action\_type": "FLAG\_BUG",
    "line\_number": 5,
    "issue\_type": "syntax",
    "comment": "Syntax error: assignment operator = used instead of equality == in if condition"
  }'

# Check state
curl http://localhost:7860/state

# Get final score
curl http://localhost:7860/score
```

\---

## ✅ Pre-Submission Checklist

* \[x] `/reset` returns HTTP 200
* \[x] `openenv validate` compatible (manifest at `openenv.yaml`)
* \[x] `docker build` succeeds
* \[x] `inference.py` runs and produces `\[START]/\[STEP]/\[END]` logs
* \[x] 3 tasks with deterministic graders returning scores in \[0.0, 1.0]
* \[x] README with action/obs spec, tasks, and baseline scores
* \[x] Reward varies meaningfully throughout episodes (not just terminal)
* \[x] All Pydantic typed models for Observation, Action, StepResult

\---

## 🧠 Design Decisions

* **Keyword matching for grader:** The grader checks that the agent's comment contains domain-specific keywords (e.g. "sql injection", "race condition"). This is robust and fully deterministic without requiring semantic similarity.
* **Partial credit:** The environment gives +0.3 for finding the right line even without correct explanation, encouraging the agent to at least locate issues.
* **Completion bonus:** A +0.5 bonus for correctly finishing the review (all issues found + APPROVE) rewards episodic completeness.
* **No randomness:** All code samples and expected issues are hardcoded. `temperature=0` on the LLM ensures reproducibility.

