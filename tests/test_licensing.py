"""Release licensing and dependency-policy contracts."""

from __future__ import annotations

from email.message import Message
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = [
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "dependency-licenses.json",
    "docs/LICENSE.md",
]


def test_project_metadata_and_notices_define_the_042_license_boundary() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["version"] == "0.4.7"
    assert project["license"] == "MPL-2.0"
    assert project["license-files"] == EXPECTED_FILES
    assert all((ROOT / relative).is_file() for relative in EXPECTED_FILES)
    assert "Copyright (c) Audun Nyhus" in (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Starting with version 0.4.2" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_repository_license_is_the_approved_mpl_20_text() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.startswith("Mozilla Public License Version 2.0\n")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest().upper() == (
        "3F3D9E0024B1921B067D6F7F88DEB4A60CBE7A78E76C64E3F1D7FC3B779B9D04"
    )


def test_dependency_inventory_is_canonical_and_complete() -> None:
    raw = (ROOT / "dependency-licenses.json").read_text(encoding="utf-8")
    inventory = json.loads(raw)
    assert raw.replace("\r\n", "\n") == (
        json.dumps(inventory, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    assert inventory["project_license"] == "MPL-2.0"
    assert {row["name"] for row in inventory["dependencies"]} == {
        "build",
        "mpmath",
        "numpy",
        "psutil",
        "pytest",
        "scipy",
        "setuptools",
        "shapely",
        "threadpoolctl",
        "twine",
        "wheel",
    }
    assert all(row["bundled"] is False for row in inventory["dependencies"])


def test_static_license_checker_accepts_repository() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_licenses.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "license check passed" in completed.stdout


@pytest.mark.parametrize("name", ["psutil", "threadpoolctl"])
@pytest.mark.parametrize("field", ["License", "License-Expression"])
@pytest.mark.parametrize("license_text", ["BSD-3-Clause", "MIT", "BSD", "GPL-3.0", ""])
def test_runtime_monitoring_licenses_accept_only_verified_expression(
    monkeypatch: pytest.MonkeyPatch, name: str, field: str, license_text: str
) -> None:
    spec = importlib.util.spec_from_file_location("check_runtime_licenses", ROOT / "tools" / "check_licenses.py")
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    metadata = Message()
    metadata[field] = license_text
    distribution = type("Distribution", (), {"metadata": metadata})()
    monkeypatch.setattr(checker.importlib_metadata, "distribution", lambda _: distribution)
    rows = [{"name": name, "scope": "runtime"}]

    if license_text == "BSD-3-Clause":
        checker._check_installed(rows)
    else:
        with pytest.raises(SystemExit, match="license check failed"):
            checker._check_installed(rows)
