"""Isolated real-Git fixtures; no shared PIN, locks, GPU, or production workspaces."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import git_baseline_v070 as repair


class LocalSandbox:
    def __init__(self, root):
        self.root = Path(root)

    def exec(self, container_id, command):
        command = command.replace("/workspace", str(self.root))
        result = subprocess.run(["/bin/sh", "-c", command], text=True, capture_output=True,
                                cwd=self.root, env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout,
                               stderr=result.stderr, timed_out=False)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v070-git-fixture-",
            dir="PROJECT_ROOT/codex/experiments/v070-git-r2/wrapper")
        self.root = Path(self.temp.name)
        (self.root / "source.py").write_text("value = 1\n")
        self.sandbox = LocalSandbox(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_initializes_preexisting_git_info_and_preserves_exclude(self):
        info = self.root / ".git/info"
        info.mkdir(parents=True)
        original = "__pycache__/\n*.pyc\n"
        (info / "exclude").write_text(original)
        repair.setup_baseline_commit(self.sandbox, "fixture")
        self.assertEqual((info / "exclude").read_text(), original)
        self.assertEqual((self.root / "source.py").read_text(), "value = 1\n")
        row = repair.entries()[-1]
        self.assertTrue(row["clean"])
        self.assertEqual(row["source_before_sha256"], row["source_after_sha256"])
        self.assertEqual(row["head"], row["baseline_tag"])

    def test_actual_submit_captures_tracked_edit_and_new_file(self):
        from swegemma.context import SwegemmaContext
        from swegemma.tools.execution import submit_patch
        repair.setup_baseline_commit(self.sandbox, "fixture")
        (self.root / "source.py").write_text("value = 2\n")
        (self.root / "new_module.py").write_text("created = True\n")
        ctx = SwegemmaContext(docker_manager=self.sandbox, container_id="fixture",
                              repo="test/mock_repo_fixture")
        response = json.loads(submit_patch(ctx))
        self.assertEqual(response["status"], "ok")
        self.assertTrue(ctx.patch_submitted)
        self.assertIn("source.py", ctx.submitted_patch)
        self.assertIn("new_module.py", ctx.submitted_patch)
        self.assertEqual(response["patch_size"], len(ctx.submitted_patch))

    def test_empty_submission_is_successfully_captured_empty(self):
        from swegemma.context import SwegemmaContext
        from swegemma.tools.execution import submit_patch
        repair.setup_baseline_commit(self.sandbox, "fixture")
        ctx = SwegemmaContext(docker_manager=self.sandbox, container_id="fixture",
                              repo="test/mock_repo_fixture")
        response = json.loads(submit_patch(ctx))
        self.assertEqual(response["status"], "ok")
        self.assertTrue(ctx.patch_submitted)
        self.assertEqual(ctx.submitted_patch, "")

    def test_checked_error_has_infrastructure_prefix(self):
        broken = SimpleNamespace(exec=lambda *args: SimpleNamespace(
            exit_code=7, stdout="", stderr="fixture failure", timed_out=False))
        with self.assertRaisesRegex(repair.GitBaselineError, "V070_GIT_SETUP_ERROR"):
            repair.setup_baseline_commit(broken, "fixture")
        self.assertFalse(repair.entries()[-1]["completed"])

    def test_symlink_git_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / ".git").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(repair.GitBaselineError):
            repair.setup_baseline_commit(self.sandbox, "fixture")

    def test_workspace_inside_parent_repository_gets_own_git_root(self):
        parent = self.root / "parent"
        parent.mkdir()
        subprocess.run(["git", "init", "--quiet", str(parent)], check=True)
        workspace = parent / "workspace"
        workspace.mkdir()
        (workspace / "module.py").write_text("owned = True\n")
        sandbox = LocalSandbox(workspace)
        repair.setup_baseline_commit(sandbox, "fixture")
        self.assertEqual(Path(repair.entries()[-1]["git_root"]), workspace)
        self.assertTrue((workspace / ".git/HEAD").is_file())
        self.assertFalse((parent / ".git/refs/heads/master").exists())

    def test_guarded_entry_installs_three_aliases_and_preserves_argv(self):
        import sys
        import run_guarded_git_v070 as entry
        from swegemma.harness import container_setup, agent_runner, verification
        receipt = self.root / "guard.json"
        saved_path = list(sys.path)
        observed = []
        def fake_run(path, run_name):
            if Path(path) == entry.ROOT / "run_eval.py":
                return {}
            self.assertEqual(Path(path), entry.GUARD)
            for module in (container_setup, agent_runner, verification):
                self.assertIs(module.setup_baseline_commit, repair.setup_baseline_commit)
            observed.append(list(sys.argv))
            raise SystemExit(0)
        try:
            with patch.object(entry.runpy, "run_path", side_effect=fake_run):
                code = entry.main(["--guard-receipt", str(receipt), "--",
                                   str(entry.ROOT / "run_eval.py"), "--tasks", "fixture.jsonl"])
            self.assertEqual(code, 0)
            self.assertEqual(observed, [[str(entry.GUARD), "--guard-receipt", str(receipt), "--",
                                        str(entry.ROOT / "run_eval.py"), "--tasks", "fixture.jsonl"]])
            self.assertTrue(json.loads(Path(str(receipt) + ".git-setup.json").read_text())["completed"])
        finally:
            sys.path[:] = saved_path

    def test_receipt_refuses_overwrite(self):
        p = self.root / "receipt.json"
        repair.write_receipt(p, completed=True)
        with self.assertRaises(FileExistsError):
            repair.write_receipt(p, completed=True)


if __name__ == "__main__":
    unittest.main()
