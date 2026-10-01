"""Guard against silently under-testing unfamiliar or deleted inputs."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location("dev_checks", Path(__file__).parents[1] / "tools/dev_checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_unclassified_and_shared_changes_require_full_suite():
    for path in ("src/anygeometry/model.py", "tests/conftest.py", "pyproject.toml",
                 ".github/workflows/development.yml", "tools/new_tool.py", "docs/LICENSE.md"):
        assert checks.select(["docs/testing.md", path])["tests"] == ["tests"]


def test_new_tests_are_included_and_deleted_tests_require_full(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_new.py").write_text("", encoding="utf-8")
    assert checks.select(["tests/test_new.py"], tmp_path)["tests"] == ["tests/test_new.py"]
    assert checks.select(["tests/test_deleted.py"], tmp_path)["scope"] == "full"


def test_coordinates_cover_automation_and_docs_are_not_numerical_evidence():
    plan = checks.select(["src/anygeometry/coordinates.py"])
    assert "tests/test_automation_contract.py" in plan["tests"]
    assert checks.select(["docs/testing.md", "CHANGELOG.md"])["scope"] == "documentation"
    assert checks.select(["src/anygeometry/surprise.md"])["scope"] == "full"


def test_failed_test_execution_propagates_and_saves_report(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(checks, "changed_paths", lambda base: ["tests/test_coordinates.py"])
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7))
    report = tmp_path / "report.json"
    assert checks.main(["--report", str(report)]) == 7
    assert json.loads(report.read_text())["exit_code"] == 7


def test_git_selection_includes_committed_staged_deleted_and_new_files(monkeypatch, tmp_path):
    import subprocess
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    tests = tmp_path / "tests"
    tests.mkdir()
    old = tests / "test_old.py"
    old.write_text("# old\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
    git("tag", "base")
    (tmp_path / "README.md").write_text("# committed\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "change")
    (tests / "test_staged.py").write_text("", encoding="utf-8")
    git("add", "tests/test_staged.py")
    old.unlink()
    (tests / "test_new.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    paths = checks.changed_paths("base")
    assert set(paths) == {"README.md", "tests/test_old.py", "tests/test_staged.py", "tests/test_new.py"}
    assert checks.select(paths, tmp_path)["scope"] == "full"


def accepted_run():
    run = {"status": "completed", "conclusion": "success", "event": "pull_request",
           "pull_requests": [{"number": 11, "base": {"sha": "base"}}]}
    jobs = [{"name": name, "status": "completed", "conclusion": "success"}
            for name in checks.DEVELOPMENT_JOBS]
    return run, jobs


@pytest.mark.parametrize("invalid", ["source", "license", "different_base", "different_pr",
                                     "pending", "failed", "skipped", "missing", "duplicate"])
def test_evidence_reuse_rejects_unaccepted_or_changed_inputs(invalid):
    run, jobs = accepted_run()
    base, pr, delta = "base", 11, ["docs/testing.md"]
    assert checks.reusable_evidence(run, jobs, pr=pr, base=base, delta=delta)
    if invalid == "source": delta = ["src/anygeometry/coordinates.py"]
    elif invalid == "license": delta = ["docs/LICENSE.md"]
    elif invalid == "different_base": base = "another"
    elif invalid == "different_pr": pr = 12
    elif invalid == "pending": run["status"] = "in_progress"
    elif invalid == "failed": run["conclusion"] = "failure"
    elif invalid == "skipped": jobs[0]["conclusion"] = "skipped"
    elif invalid == "missing": jobs.pop()
    elif invalid == "duplicate": jobs[-1] = jobs[0]
    assert not checks.reusable_evidence(run, jobs, pr=pr, base=base, delta=delta)


def test_api_and_missing_commit_fail_closed(monkeypatch):
    for key, value in {"GITHUB_TOKEN": "test-fixture", "GITHUB_REPOSITORY": "owner/repo",
                       "PR_NUMBER": "11", "BASE_SHA": "base"}.items():
        monkeypatch.setenv(key, value)
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(checks.urllib.request, "urlopen", unavailable)
    assert checks.ci_reuse() is None


def test_reuse_binds_actual_git_delta_and_successful_jobs(monkeypatch):
    import io
    import json
    run, jobs = accepted_run()
    run.update(id=123, head_sha="a" * 40)
    for key, value in {"GITHUB_TOKEN": "test-fixture", "GITHUB_REPOSITORY": "owner/repo",
                       "PR_NUMBER": "11", "BASE_SHA": "base"}.items():
        monkeypatch.setenv(key, value)
    def response(request, **kwargs):
        data = {"jobs": jobs} if "/123/jobs" in request.full_url else {"workflow_runs": [run]}
        return io.StringIO(json.dumps(data))
    monkeypatch.setattr(checks.urllib.request, "urlopen", response)
    monkeypatch.setattr(checks, "git", lambda *args: "docs/testing.md\0")
    assert checks.ci_reuse()["head_sha"] == "a" * 40
    monkeypatch.setattr(checks, "git", lambda *args: "src/anygeometry/model.py\0")
    assert checks.ci_reuse() is None
