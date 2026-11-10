"""Helpers only: official JUnit identity matching and control-input integrity."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import control_case_v070 as case
import run_controls_v070 as controller


class ControlHelpersTests(unittest.TestCase):
    def test_junit_requires_the_named_target_node(self):
        task = SimpleNamespace(FAIL_TO_PASS=["tests/test_api.py::test_target"])
        text = '<testsuite tests="1"><testcase name="test_target" classname="tests.test_api"/></testsuite>'
        result = case.junit_evidence(text, task)
        self.assertTrue(result["target_tests_verified"])
        self.assertEqual(result["failing"], [])

    def test_empty_patch_junit_tracks_actual_target_failure(self):
        task = SimpleNamespace(FAIL_TO_PASS=["tests/test_api.py::test_target"])
        text = '<testsuite tests="1"><testcase name="test_target" classname="tests.test_api"><failure message="expected"/></testcase></testsuite>'
        result = case.junit_evidence(text, task)
        self.assertTrue(result["target_tests_verified"])
        self.assertEqual(result["failing"], task.FAIL_TO_PASS)

    def test_missing_target_cannot_pass_from_unrelated_test(self):
        task = SimpleNamespace(FAIL_TO_PASS=["tests/test_api.py::test_target"])
        text = '<testsuite tests="1"><testcase name="test_unrelated" classname="tests.test_api"/></testsuite>'
        self.assertFalse(case.junit_evidence(text, task)["target_tests_verified"])

    def test_junit_setup_error_is_not_an_empty_patch_failure(self):
        task = SimpleNamespace(FAIL_TO_PASS=["tests/test_api.py::test_target"])
        text = '<testsuite tests="1"><testcase name="test_target" classname="tests.test_api"><error message="setup failed"/></testcase></testsuite>'
        evidence = case.junit_evidence(text, task)
        self.assertFalse(evidence["target_tests_verified"])
        self.assertEqual(evidence["failing"], [])
        self.assertEqual(evidence["junit_error_count"], 1)

    def test_junit_skipped_target_cannot_validate_control(self):
        task = SimpleNamespace(FAIL_TO_PASS=["tests/test_api.py::test_target"])
        text = '<testsuite tests="1"><testcase name="test_target" classname="tests.test_api"><skipped/></testcase></testsuite>'
        evidence = case.junit_evidence(text, task)
        self.assertFalse(evidence["target_tests_verified"])
        self.assertEqual(evidence["failing"], [])
        self.assertEqual(evidence["skipped"], task.FAIL_TO_PASS)

    def test_real_bound_submit_tool_returns_dict_and_is_recorded(self):
        from swegemma.tools.execution import make_submit_patch
        manager = SimpleNamespace(exec=lambda *args: SimpleNamespace(
            exit_code=0, stdout="", stderr="", timed_out=False))
        context = SimpleNamespace(docker=manager, container_id="fixture", repo="test/mock_repo",
                                  check_budget=lambda **kwargs: None)
        observations = []
        observed = case.observe_submit_tool(make_submit_patch(context), observations)
        response = observed()
        self.assertIsInstance(response, dict)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(observations, [response])
        self.assertTrue(context.patch_submitted)

    def test_preserved_pypika_exact_identity_regression(self):
        base = Path("PROJECT_ROOT/codex/experiments/v070-git-r1/controls/evidence/cases/kayak__pypika-267--empty")
        xml = (base / "verification-junit.xml").read_bytes()
        self.assertEqual(hashlib.sha256(xml).hexdigest(), "453cdf824ec80bb12cf33e8de87bf8a64024322a9f6105b7d9911949b2f82416")
        original = json.loads((base / "case-report.json").read_text())
        task = SimpleNamespace(FAIL_TO_PASS=original["junit"]["required"])
        evidence = case.junit_evidence(xml.decode(), task)
        self.assertEqual(evidence["collected"], 214)
        self.assertEqual(len(evidence["required"]), 7)
        self.assertEqual(evidence["failing"], task.FAIL_TO_PASS)
        self.assertEqual(evidence["passed"], [])
        self.assertEqual(evidence["missing"], [])
        self.assertEqual(evidence["ambiguous"], [])
        self.assertTrue(case.control_passed("empty", evidence, False, 1, True, None))

    def test_longer_passing_sibling_cannot_satisfy_short_target(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Checks::test_equals"])
        xml = '<testsuite><testcase classname="pkg.test_api.Checks" name="test_equals"><failure/></testcase><testcase classname="pkg.test_api.Checks" name="test_equals_and"/></testsuite>'
        evidence = case.junit_evidence(xml, task)
        self.assertEqual(evidence["failing"], task.FAIL_TO_PASS)
        self.assertEqual(evidence["passed"], [])
        self.assertTrue(case.control_passed("empty", evidence, False, 1, True, None))

    def test_different_class_same_method_is_missing(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Expected::test_equals"])
        xml = '<testsuite><testcase classname="pkg.test_api.Sibling" name="test_equals"><failure/></testcase></testsuite>'
        evidence = case.junit_evidence(xml, task)
        self.assertFalse(evidence["target_tests_verified"])
        self.assertEqual(evidence["missing"], task.FAIL_TO_PASS)

    def test_full_parameterized_name_is_preserved_including_separator(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Outer::Inner::test_equals[a::b]"])
        xml = '<testsuite><testcase classname="pkg.test_api.Outer.Inner" name="test_equals[a::b]"><failure/></testcase><testcase classname="pkg.test_api.Outer.Inner" name="test_equals[a::c]"/></testsuite>'
        evidence = case.junit_evidence(xml, task)
        self.assertTrue(evidence["target_tests_verified"])
        self.assertEqual(evidence["failing"], task.FAIL_TO_PASS)
        self.assertEqual(evidence["passed"], [])

    def test_unparameterized_required_name_does_not_match_parameter(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::test_equals"])
        xml = '<testsuite><testcase classname="pkg.test_api" name="test_equals[a]"><failure/></testcase></testsuite>'
        self.assertFalse(case.junit_evidence(xml, task)["target_tests_verified"])

    def test_duplicate_exact_identities_are_ambiguous(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Checks::test_equals"])
        xml = '<testsuite><testcase classname="pkg.test_api.Checks" name="test_equals"><failure/></testcase><testcase classname="pkg.test_api.Checks" name="test_equals"/></testsuite>'
        evidence = case.junit_evidence(xml, task)
        self.assertFalse(evidence["target_tests_verified"])
        self.assertEqual(evidence["ambiguous"], task.FAIL_TO_PASS)
        self.assertFalse(case.control_passed("empty", evidence, False, 1, True, None))

    def test_explicit_file_mismatch_is_rejected(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Checks::test_equals"])
        xml = '<testsuite><testcase file="other/test_api.py" classname="pkg.test_api.Checks" name="test_equals"><failure/></testcase></testsuite>'
        self.assertFalse(case.junit_evidence(xml, task)["target_tests_verified"])

    def test_explicit_path_separator_and_py_representation(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Checks::test_equals"])
        xml = '<testsuite><testcase classname="pkg/test_api.py.Checks" name="test_equals"><failure/></testcase></testsuite>'
        self.assertTrue(case.junit_evidence(xml, task)["target_tests_verified"])

    def test_unrelated_skip_is_reported_without_rejecting_target(self):
        task = SimpleNamespace(FAIL_TO_PASS=["pkg/test_api.py::Checks::test_equals"])
        xml = '<testsuite><testcase classname="pkg.test_api.Checks" name="test_equals"><failure/></testcase><testcase classname="pkg.test_api.Checks" name="test_else"><skipped/></testcase></testsuite>'
        evidence = case.junit_evidence(xml, task)
        self.assertEqual(evidence["skipped_testcases"], 1)
        self.assertEqual(evidence["skipped"], [])
        self.assertTrue(case.control_passed("empty", evidence, False, 1, True, None))

    def test_control_source_change_is_rejected_without_launch(self):
        base = Path("PROJECT_ROOT/codex/experiments/v070-git-r2/controls")
        with tempfile.TemporaryDirectory(prefix="unit-plan-", dir=base) as temp:
            root = Path(temp)
            amendment = root / "amendment-FROZEN.md"
            amendment.write_text("fixture")
            clarification = root / "operational-temp-clarification.json"
            clarification.write_text("fixture")
            source = root / "source.py"
            source.write_text("original")
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "amendment_sha256": controller.sha(amendment),
                "operational_temp_clarification_sha256": controller.sha(clarification),
                "source_pins": {str(source): controller.sha(source)},
                "parent": {"path": str(amendment), "sha256": controller.sha(amendment)},
                "runner_reference": {"path": str(clarification), "sha256": controller.sha(clarification)},
                "cases": []}))
            digest = controller.sha(plan)
            source.write_text("changed")
            with patch.object(controller, "WORK", root), patch.object(controller, "PLAN", plan):
                with self.assertRaisesRegex(RuntimeError, "Control source changed"):
                    controller.verify_plan(digest)


if __name__ == "__main__":
    unittest.main()
