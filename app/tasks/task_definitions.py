"""
Task definitions for the AI Code Review environment.
Each task has: code snippet, expected issues, difficulty, and a grader function.
All tasks are deterministic.
"""
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Task 1 – EASY: Syntax & obvious bugs in a small Python function
# ---------------------------------------------------------------------------
TASK_EASY = {
    "name": "fix_syntax_and_obvious_bugs",
    "difficulty": "easy",
    "language": "python",
    "description": (
        "Review a short Python function for syntax errors and clear bugs. "
        "The code has 3 issues: a syntax error, a wrong operator, and an off-by-one error."
    ),
    "max_steps": 8,
    "code": """\
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers) + 1):   # Issue 1: off-by-one (should be len(numbers))
        total = total + numbers[i]
    if total = 0:                        # Issue 2: syntax error (= instead of ==)
        return None
    average = total / len(numbers)
    return average
""",
    "issues": [
        {
            "line_number": 3,
            "issue_type": "logic",
            "description": "Off-by-one error: range(len(numbers) + 1) will cause IndexError",
            "severity": "high",
            "keywords": ["off-by-one", "index", "range", "indexerror", "+1", "loop"],
        },
        {
            "line_number": 5,
            "issue_type": "syntax",
            "description": "Syntax error: assignment '=' used instead of equality '==' in if condition",
            "severity": "critical",
            "keywords": ["syntax", "==", "assignment", "condition", "if", "operator"],
        },
        {
            "line_number": 7,
            "issue_type": "logic",
            "description": "Division after early return for total==0 is fine, but dividing by len(numbers) when numbers is empty causes ZeroDivisionError",
            "severity": "medium",
            "keywords": ["zerodivision", "empty", "division", "len", "zero"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Task 2 – MEDIUM: Logic & security issues in a web-style snippet
# ---------------------------------------------------------------------------
TASK_MEDIUM = {
    "name": "logic_and_security_review",
    "difficulty": "medium",
    "language": "python",
    "description": (
        "Review a Python Flask-style route handler. "
        "Find 4 issues spanning logic errors, a SQL injection vulnerability, "
        "missing authentication check, and an unhandled edge case."
    ),
    "max_steps": 12,
    "code": """\
from flask import request
import sqlite3

def get_user_data():
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Issue 1: SQL Injection – user input directly in query
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()

    # Issue 2: No authentication check before returning data
    if result:
        return {"user": result}

    # Issue 3: Logic – returns nothing (None) instead of a proper 404 response
    # Issue 4: Connection never closed (resource leak)
    conn.close()   # This line is unreachable when result is truthy
""",
    "issues": [
        {
            "line_number": 9,
            "issue_type": "security",
            "description": "SQL Injection: user_id is directly interpolated into the query without parameterisation",
            "severity": "critical",
            "keywords": ["sql injection", "injection", "parameteris", "f-string", "user_id", "sanitize", "escape"],
        },
        {
            "line_number": 12,
            "issue_type": "security",
            "description": "No authentication/authorisation check before returning user data",
            "severity": "high",
            "keywords": ["auth", "authentication", "authoris", "access control", "permission", "login"],
        },
        {
            "line_number": 15,
            "issue_type": "logic",
            "description": "Function returns None implicitly when result is falsy; should return explicit 404 error",
            "severity": "medium",
            "keywords": ["none", "404", "return", "error", "missing", "not found", "implicit"],
        },
        {
            "line_number": 17,
            "issue_type": "performance",
            "description": "Database connection is never closed when result is found (resource leak); conn.close() is unreachable",
            "severity": "medium",
            "keywords": ["leak", "close", "connection", "resource", "unreachable", "finally", "context manager"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Task 3 – HARD: Design & architecture issues in a class-based system
# ---------------------------------------------------------------------------
TASK_HARD = {
    "name": "design_and_architecture_review",
    "difficulty": "hard",
    "language": "python",
    "description": (
        "Review a Python class that manages an order-processing pipeline. "
        "Identify 5 issues: violation of Single Responsibility Principle, "
        "hardcoded secrets, mutable default argument, swallowed exception, "
        "and a thread-safety problem."
    ),
    "max_steps": 18,
    "code": """\
import smtplib
import threading

DB_PASSWORD = "super_secret_123"   # Issue 1: Hardcoded secret / credential

class OrderProcessor:
    def __init__(self, pending_orders=[]):   # Issue 2: Mutable default argument
        self.pending_orders = pending_orders
        self.counter = 0
        self.lock = threading.Lock()

    def process_all(self):
        for order in self.pending_orders:
            try:
                self._validate(order)
                self._save_to_db(order)
                self._send_confirmation_email(order)  # Issue 3: SRP violation – email in processor
            except Exception:
                pass   # Issue 4: Swallowed exception – silent failure

    def increment_counter(self):
        # Issue 5: Thread-safety – read-modify-write without holding lock
        temp = self.counter
        temp += 1
        self.counter = temp

    def _validate(self, order):
        assert order.get("amount") > 0, "Amount must be positive"

    def _save_to_db(self, order):
        pass  # placeholder

    def _send_confirmation_email(self, order):
        pass  # placeholder
""",
    "issues": [
        {
            "line_number": 4,
            "issue_type": "security",
            "description": "Hardcoded credential: DB_PASSWORD stored as plaintext in source code; use environment variables or a secrets manager",
            "severity": "critical",
            "keywords": ["hardcoded", "secret", "password", "credential", "env", "plaintext", "vault"],
        },
        {
            "line_number": 6,
            "issue_type": "logic",
            "description": "Mutable default argument: pending_orders=[] is shared across all instances, leading to unexpected state sharing",
            "severity": "high",
            "keywords": ["mutable default", "default argument", "shared state", "list", "none", "instance"],
        },
        {
            "line_number": 15,
            "issue_type": "design",
            "description": "Single Responsibility Principle violation: OrderProcessor handles email sending; this should be delegated to a NotificationService",
            "severity": "medium",
            "keywords": ["srp", "single responsibility", "email", "separation", "concern", "delegate", "notification"],
        },
        {
            "line_number": 17,
            "issue_type": "logic",
            "description": "Swallowed exception: bare 'except: pass' hides all errors silently, making debugging impossible",
            "severity": "high",
            "keywords": ["swallow", "exception", "pass", "silent", "bare except", "error handling", "log"],
        },
        {
            "line_number": 22,
            "issue_type": "performance",
            "description": "Thread-safety bug: counter read-modify-write is not protected by the lock, causing race conditions",
            "severity": "high",
            "keywords": ["thread", "race condition", "lock", "atomic", "concurrent", "safe", "mutex"],
        },
    ],
}


ALL_TASKS = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
TASK_MAP = {t["name"]: t for t in ALL_TASKS}


# ---------------------------------------------------------------------------
# Grader functions – deterministic, return float in [0.0, 1.0]
# ---------------------------------------------------------------------------

def _keyword_match(action_comment: str, issue_keywords: List[str]) -> bool:
    """Check if the agent's comment mentions at least one keyword for an issue."""
    if not action_comment:
        return False
    comment_lower = action_comment.lower()
    return any(kw in comment_lower for kw in issue_keywords)


def grade_episode(
    task: Dict[str, Any],
    action_history: List[Dict[str, Any]],
) -> float:
    """
    Deterministic grader. Returns a float in [0.0, 1.0].

    Scoring:
      - For each expected issue, check if any action in history correctly
        flagged the right line_number AND comment matches issue keywords.
      - Score = correctly_found_issues / total_issues
    """
    issues = task["issues"]
    total = len(issues)
    if total == 0:
        return 1.0

    found = 0
    for issue in issues:
        for action in action_history:
            line_match = (action.get("line_number") == issue["line_number"])
            keyword_hit = _keyword_match(
                action.get("comment", ""), issue["keywords"]
            )
            type_match = (
                action.get("issue_type") == issue["issue_type"]
                or action.get("action_type") in ("FLAG_BUG", "SUGGEST_FIX", "REQUEST_CHANGES")
            )
            if line_match and keyword_hit:
                found += 1
                break
            elif line_match and type_match:
                # partial credit counted as found if line is right and type matches
                found += 0.5
                break

    score = min(found / total, 1.0)
    return round(score, 4)
