"""Guard against silently under-testing unfamiliar or deleted inputs."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("dev_checks", Path(__file__).parents[1] / "tools/dev_checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_unclassified_and_shared_changes_require_full_suite():
    for path in ("src/anygeometry/model.py", "tests/conftest.py", "pyproject.toml",
                 ".github/workflows/development.yml", "tools/new_tool.py"):
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
