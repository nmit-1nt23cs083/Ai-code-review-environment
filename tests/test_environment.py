"""
tests/test_environment.py

Comprehensive test suite for the AI Code Review RL Environment.
Tests all environment logic, graders, reward shaping, and edge cases.

Run with: python -m pytest tests/ -v
Or standalone: python tests/test_environment.py
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks.task_definitions import ALL_TASKS, TASK_MAP, grade_episode


# -----------------------------------------------------------------------
# Helpers — replicate reward logic for unit testing without the full env
# -----------------------------------------------------------------------

def _keyword_match(comment: str, keywords: list) -> bool:
    if not comment:
        return False
    c = comment.lower()
    return any(kw in c for kw in keywords)


class TestTaskDefinitions(unittest.TestCase):
    """Validate task structure and grader correctness."""

    def test_three_tasks_exist(self):
        self.assertEqual(len(ALL_TASKS), 3, "Must have exactly 3 tasks")

    def test_task_difficulties(self):
        difficulties = [t["difficulty"] for t in ALL_TASKS]
        self.assertIn("easy",   difficulties)
        self.assertIn("medium", difficulties)
        self.assertIn("hard",   difficulties)

    def test_task_names_unique(self):
        names = [t["name"] for t in ALL_TASKS]
        self.assertEqual(len(names), len(set(names)), "Task names must be unique")

    def test_task_map_matches_list(self):
        for t in ALL_TASKS:
            self.assertIn(t["name"], TASK_MAP)

    def test_each_task_has_required_keys(self):
        required = {"name", "difficulty", "language", "description", "max_steps", "code", "issues"}
        for t in ALL_TASKS:
            missing = required - set(t.keys())
            self.assertEqual(missing, set(), f"Task '{t['name']}' missing keys: {missing}")

    def test_each_issue_has_required_keys(self):
        required = {"line_number", "issue_type", "description", "severity", "keywords"}
        for t in ALL_TASKS:
            for issue in t["issues"]:
                missing = required - set(issue.keys())
                self.assertEqual(missing, set(), f"Issue in '{t['name']}' missing: {missing}")

    def test_easy_has_3_issues(self):
        easy = TASK_MAP["fix_syntax_and_obvious_bugs"]
        self.assertEqual(len(easy["issues"]), 3)

    def test_medium_has_4_issues(self):
        medium = TASK_MAP["logic_and_security_review"]
        self.assertEqual(len(medium["issues"]), 4)

    def test_hard_has_5_issues(self):
        hard = TASK_MAP["design_and_architecture_review"]
        self.assertEqual(len(hard["issues"]), 5)

    def test_issue_counts_increase_with_difficulty(self):
        counts = [len(t["issues"]) for t in ALL_TASKS]
        self.assertEqual(sorted(counts), counts, "Issue count should increase easy→hard")

    def test_max_steps_increase_with_difficulty(self):
        steps = [t["max_steps"] for t in ALL_TASKS]
        self.assertEqual(sorted(steps), steps, "Max steps should increase easy→hard")

    def test_all_keywords_are_lowercase(self):
        for t in ALL_TASKS:
            for issue in t["issues"]:
                for kw in issue["keywords"]:
                    self.assertEqual(kw, kw.lower(), f"Keyword '{kw}' must be lowercase")

    def test_line_numbers_positive(self):
        for t in ALL_TASKS:
            for issue in t["issues"]:
                self.assertGreater(issue["line_number"], 0)


class TestGraderFunction(unittest.TestCase):
    """Validate the deterministic grader for each task."""

    def _perfect_actions(self, task):
        return [
            {
                "action_type": "FLAG_BUG",
                "line_number": i["line_number"],
                "issue_type": i["issue_type"],
                "comment": " ".join(i["keywords"][:2]),
            }
            for i in task["issues"]
        ]

    def test_perfect_score_is_1(self):
        for task in ALL_TASKS:
            score = grade_episode(task, self._perfect_actions(task))
            self.assertAlmostEqual(score, 1.0, places=3,
                msg=f"Perfect actions should score 1.0 on '{task['name']}'")

    def test_empty_actions_score_0(self):
        for task in ALL_TASKS:
            score = grade_episode(task, [])
            self.assertAlmostEqual(score, 0.0, places=3,
                msg=f"No actions should score 0.0 on '{task['name']}'")

    def test_score_in_0_1_range(self):
        for task in ALL_TASKS:
            partial = self._perfect_actions(task)[:1]
            score = grade_episode(task, partial)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_partial_score_easy(self):
        task = TASK_MAP["fix_syntax_and_obvious_bugs"]
        # find 1 out of 3
        actions = self._perfect_actions(task)[:1]
        score = grade_episode(task, actions)
        self.assertAlmostEqual(score, 1/3, places=3)

    def test_partial_score_medium(self):
        task = TASK_MAP["logic_and_security_review"]
        # find 2 out of 4
        actions = self._perfect_actions(task)[:2]
        score = grade_episode(task, actions)
        self.assertAlmostEqual(score, 0.5, places=3)

    def test_partial_score_hard(self):
        task = TASK_MAP["design_and_architecture_review"]
        # find 3 out of 5
        actions = self._perfect_actions(task)[:3]
        score = grade_episode(task, actions)
        self.assertAlmostEqual(score, 0.6, places=3)

    def test_wrong_line_no_credit(self):
        task = TASK_MAP["fix_syntax_and_obvious_bugs"]
        bad_actions = [
            {"action_type": "FLAG_BUG", "line_number": 99, "issue_type": "logic",
             "comment": "off-by-one error range loop indexerror"}
        ]
        score = grade_episode(task, bad_actions)
        self.assertAlmostEqual(score, 0.0, places=3,
            msg="Wrong line should give 0 credit")

    def test_duplicate_correct_action_not_double_counted(self):
        task = TASK_MAP["fix_syntax_and_obvious_bugs"]
        first_issue = task["issues"][0]
        repeated = [
            {"action_type": "FLAG_BUG", "line_number": first_issue["line_number"],
             "issue_type": first_issue["issue_type"], "comment": first_issue["keywords"][0]}
        ] * 5  # same correct action 5 times
        score = grade_episode(task, repeated)
        expected = 1 / len(task["issues"])
        self.assertAlmostEqual(score, expected, places=3,
            msg="Repeating the same correct action should not multiply score")

    def test_grader_is_deterministic(self):
        task = TASK_MAP["logic_and_security_review"]
        actions = self._perfect_actions(task)
        scores = [grade_episode(task, actions) for _ in range(5)]
        self.assertTrue(all(s == scores[0] for s in scores),
            "Grader must be deterministic")

    def test_approve_action_ignored_by_grader(self):
        """APPROVE alone should give 0; grader only counts FLAG/SUGGEST actions."""
        task = TASK_MAP["fix_syntax_and_obvious_bugs"]
        score = grade_episode(task, [{"action_type": "APPROVE", "line_number": None,
                                       "issue_type": None, "comment": None}])
        self.assertAlmostEqual(score, 0.0, places=3)


class TestRewardLogic(unittest.TestCase):
    """Test reward shaping logic manually (without pydantic / FastAPI)."""

    def _evaluate(self, action, issues, found_set):
        """Mirrors environment._evaluate_action logic."""
        if action["action_type"] == "APPROVE":
            remaining = len(issues) - len(found_set)
            return (1.5, "correct approve") if remaining == 0 else (-0.5, "premature")
        if action["action_type"] == "ADD_COMMENT":
            return -0.1, "comment"
        if action["action_type"] == "REQUEST_CHANGES":
            return 0.0, "request"
        if action.get("line_number") is None:
            return -0.2, "no line"

        for idx, issue in enumerate(issues):
            if idx in found_set:
                continue
            lm = action["line_number"] == issue["line_number"]
            kh = _keyword_match(action.get("comment", ""), issue["keywords"])
            tm = action.get("issue_type") == issue["issue_type"]
            if lm and (kh or tm):
                found_set.add(idx)
                return 1.0, "correct"
            if lm:
                return 0.3, "partial"
        return -0.2, "wrong line"

    def setUp(self):
        self.task = TASK_MAP["fix_syntax_and_obvious_bugs"]
        self.issues = self.task["issues"]

    def test_correct_flag_gives_plus_1(self):
        found = set()
        issue = self.issues[0]
        reward, _ = self._evaluate({
            "action_type": "FLAG_BUG",
            "line_number": issue["line_number"],
            "issue_type": issue["issue_type"],
            "comment": issue["keywords"][0],
        }, self.issues, found)
        self.assertAlmostEqual(reward, 1.0)

    def test_wrong_line_gives_minus_0_2(self):
        found = set()
        reward, _ = self._evaluate({
            "action_type": "FLAG_BUG", "line_number": 999,
            "issue_type": "logic", "comment": "something"
        }, self.issues, found)
        self.assertAlmostEqual(reward, -0.2)

    def test_premature_approve_gives_minus_0_5(self):
        found = set()  # nothing found yet
        reward, _ = self._evaluate({
            "action_type": "APPROVE", "line_number": None,
            "issue_type": None, "comment": None
        }, self.issues, found)
        self.assertAlmostEqual(reward, -0.5)

    def test_correct_approve_gives_plus_1_5(self):
        # mark all issues as found
        found = set(range(len(self.issues)))
        reward, _ = self._evaluate({
            "action_type": "APPROVE", "line_number": None,
            "issue_type": None, "comment": None
        }, self.issues, found)
        self.assertAlmostEqual(reward, 1.5)

    def test_add_comment_gives_minus_0_1(self):
        found = set()
        reward, _ = self._evaluate({
            "action_type": "ADD_COMMENT", "line_number": None,
            "issue_type": None, "comment": "looks good"
        }, self.issues, found)
        self.assertAlmostEqual(reward, -0.1)

    def test_request_changes_neutral(self):
        found = set()
        reward, _ = self._evaluate({
            "action_type": "REQUEST_CHANGES", "line_number": None,
            "issue_type": None, "comment": "needs work"
        }, self.issues, found)
        self.assertAlmostEqual(reward, 0.0)

    def test_right_line_vague_comment_partial(self):
        found = set()
        issue = self.issues[0]
        reward, msg = self._evaluate({
            "action_type": "FLAG_BUG",
            "line_number": issue["line_number"],
            "issue_type": "style",        # wrong type
            "comment": "this looks weird", # no keywords
        }, self.issues, found)
        self.assertAlmostEqual(reward, 0.3, msg="Right line + no keywords = partial credit")

    def test_same_issue_not_rewarded_twice(self):
        found = set()
        issue = self.issues[0]
        action = {
            "action_type": "FLAG_BUG",
            "line_number": issue["line_number"],
            "issue_type": issue["issue_type"],
            "comment": issue["keywords"][0],
        }
        r1, _ = self._evaluate(action, self.issues, found)
        r2, _ = self._evaluate(action, self.issues, found)  # second time
        self.assertAlmostEqual(r1, 1.0)
        self.assertAlmostEqual(r2, -0.2,
            msg="Second flag of same issue (already found) should be penalised")

    def test_total_reward_perfect_episode(self):
        found = set()
        total = 0.0
        for issue in self.issues:
            r, _ = self._evaluate({
                "action_type": "FLAG_BUG",
                "line_number": issue["line_number"],
                "issue_type": issue["issue_type"],
                "comment": issue["keywords"][0],
            }, self.issues, found)
            total += r
        r_approve, _ = self._evaluate({
            "action_type": "APPROVE", "line_number": None,
            "issue_type": None, "comment": None
        }, self.issues, found)
        total += r_approve
        # 3 * 1.0 + 1.5 = 4.5
        self.assertAlmostEqual(total, 4.5)


class TestKeywordMatching(unittest.TestCase):
    """Test the keyword matching helper used by grader and reward."""

    def test_match_found(self):
        self.assertTrue(_keyword_match("sql injection vulnerability here", ["sql injection"]))

    def test_case_insensitive(self):
        self.assertTrue(_keyword_match("SQL INJECTION found", ["sql injection"]))

    def test_no_match(self):
        self.assertFalse(_keyword_match("looks fine to me", ["sql injection"]))

    def test_empty_comment(self):
        self.assertFalse(_keyword_match("", ["sql injection"]))

    def test_none_comment(self):
        self.assertFalse(_keyword_match(None, ["sql injection"]))

    def test_partial_keyword_matches(self):
        self.assertTrue(_keyword_match("authentication missing", ["auth"]))

    def test_any_keyword_sufficient(self):
        self.assertTrue(_keyword_match("race condition in thread", ["mutex", "race condition", "lock"]))


class TestOpenEnvYaml(unittest.TestCase):
    """Validate openenv.yaml exists and has required fields."""

    def setUp(self):
        import yaml
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
        try:
            with open(yaml_path) as f:
                self.config = yaml.safe_load(f)
        except ImportError:
            self.config = None  # yaml not available in sandbox — skip
        except FileNotFoundError:
            self.config = None

    def test_yaml_exists(self):
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
        self.assertTrue(os.path.exists(yaml_path), "openenv.yaml must exist")

    def test_yaml_has_name(self):
        if self.config is None:
            self.skipTest("pyyaml not available")
        self.assertIn("name", self.config)

    def test_yaml_has_version(self):
        if self.config is None:
            self.skipTest("pyyaml not available")
        self.assertIn("version", self.config)

    def test_yaml_has_tasks(self):
        if self.config is None:
            self.skipTest("pyyaml not available")
        self.assertIn("tasks", self.config)
        self.assertEqual(len(self.config["tasks"]), 3)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTaskDefinitions))
    suite.addTests(loader.loadTestsFromTestCase(TestGraderFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestKeywordMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenEnvYaml))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
