"""The installed-wheel gate must reject contamination and substituted code."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installation(tmp_path, monkeypatch):
    probe = _load("compatibility_probe")
    environment = tmp_path / "venv"
    site = environment / "lib"
    module = site / "anygeometry" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"__version__ = '0.4.3'\n")
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("anygeometry/__init__.py", module.read_bytes())
    dist = SimpleNamespace(locate_file=lambda name: site / name,
                           read_text=lambda name: json.dumps({"dir_info": {"editable": False}}))
    monkeypatch.setattr(probe, "sys", SimpleNamespace(prefix=str(environment), base_prefix="other",
                                                       executable=str(environment / "python")))
    original_import = probe.importlib.import_module
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(module))
                        if name == "anygeometry" else original_import(name))
    monkeypatch.setattr(probe.metadata, "distribution", lambda name: dist)
    monkeypatch.setattr(probe.metadata, "version", lambda name: "0.4.3")
    config = {"environment": str(environment), "modules": {"ANYgeometry": "anygeometry"},
              "versions": {"ANYgeometry": "0.4.3"}, "wheel": str(wheel),
              "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    return probe, config, module, dist


def test_installed_source_bytes_are_verified_not_just_version(installation):
    probe, config, module, _ = installation
    assert probe.verify_environment(config)["verified_files"] == 1
    # An older wheel with the same version label must not pass this gate.
    module.write_bytes(b"__version__ = '0.4.3'\nold_behavior = True\n")
    with pytest.raises(AssertionError):
        probe.verify_environment(config)


def test_source_checkout_and_editable_origins_cannot_pass(installation, monkeypatch, tmp_path):
    probe, config, _, dist = installation
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(tmp_path / "source.py")))
    with pytest.raises(AssertionError):
        probe.verify_environment(config)


def test_editable_metadata_is_rejected_even_inside_prefix(installation):
    probe, config, _, dist = installation
    dist.read_text = lambda name: '{"dir_info": {"editable": true}}'
    with pytest.raises(AssertionError):
        probe.verify_environment(config)


def test_input_wheel_tampering_is_rejected(installation):
    probe, config, _, _ = installation
    Path(config["wheel"]).write_bytes(b"replacement")
    with pytest.raises(AssertionError, match="input wheel changed"):
        probe.verify_environment(config)


def test_shared_consumer_failure_never_counts_as_passing():
    runner = _load("check_compatibility")
    assert runner.classification("failed", "failed") == "existing_consumer_or_environment_failure"
    assert runner.classification("failed", "passed") == "candidate_regression"
    assert runner.classification("passed", "failed") == "baseline_failure_requires_review"


def test_failed_command_keeps_diagnostics_and_nonzero_status(tmp_path):
    import sys
    runner = _load("check_compatibility").Runner(tmp_path)
    with pytest.raises(RuntimeError, match=r"failed \(7\)"):
        runner.run([sys.executable, "-I", "-c", "import sys; print('failure evidence'); sys.exit(7)"])
    assert runner.commands[0]["exit_code"] == 7
    assert "failure evidence" in (tmp_path / runner.commands[0]["log"]).read_text()
