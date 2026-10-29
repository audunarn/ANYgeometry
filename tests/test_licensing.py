"""Release licensing and dependency-policy contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from email.message import Message
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

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
        "pytest",
        "scipy",
        "setuptools",
        "shapely",
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


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("geometry_license_checker", ROOT / "tools" / "check_licenses.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distribution(license="MIT", *, expression=None, classifier=None, version="1.17.1"):
    metadata = Message()
    if license is not None:
        metadata["License"] = license
    if expression is not None:
        metadata["License-Expression"] = expression
    if classifier is not None:
        metadata["Classifier"] = classifier
    return SimpleNamespace(metadata=metadata, version=version)


def _installed(monkeypatch, checker, present):
    def find(name):
        if name not in present:
            raise checker.importlib_metadata.PackageNotFoundError(name)
        return present[name]
    monkeypatch.setattr(checker.importlib_metadata, "distribution", find)


@pytest.mark.parametrize("profile", [None, {"planar"}, {"sketch"}])
@pytest.mark.parametrize("scope", ["runtime", "development"])
def test_required_runtime_and_development_absence_is_never_relaxed(checker, monkeypatch, profile, scope):
    _installed(monkeypatch, checker, {})
    with pytest.raises(SystemExit, match="not installed"):
        checker._check_installed([{"name": "numpy", "scope": scope}], installed_extras=profile)


@pytest.mark.parametrize("profile", [None, {"sketch"}])
def test_missing_optional_is_refused_by_strict_default_or_selected_profile(checker, monkeypatch, profile):
    _installed(monkeypatch, checker, {})
    with pytest.raises(SystemExit, match="scipy.*not installed"):
        checker._check_installed([{"name": "scipy", "scope": "optional:sketch"}], installed_extras=profile)


def test_explicit_profile_allows_only_absent_unselected_optionals(checker, monkeypatch):
    _installed(monkeypatch, checker, {"numpy": _distribution("BSD-3-Clause")})
    checker._check_installed([{"name": "numpy", "scope": "runtime"},
                             {"name": "scipy", "scope": "optional:sketch"},
                             {"name": "mpmath", "scope": "optional:test"}], installed_extras={"planar"})


@pytest.mark.parametrize("name,license", [("scipy", "GPL-3.0"), ("mpmath", "AGPL-3.0"),
                                         ("shapely", "unreviewed custom license"), ("pytest", None)])
def test_present_unselected_optional_is_still_audited(checker, monkeypatch, name, license):
    _installed(monkeypatch, checker, {name: _distribution(license)})
    with pytest.raises(SystemExit, match="license"):
        checker._check_installed([{"name": name, "scope": "optional:unselected"}], installed_extras={"planar"})


@pytest.fixture
def reviewed_legacy(checker, monkeypatch):
    # Controlled metadata exercises version/text identity without vendoring the
    # long upstream license. The real installed SciPy is checked separately.
    text = "Controlled upstream BSD notice\nGPL-3.0-or-later WITH GCC-exception-3.1"
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    monkeypatch.setattr(checker, "SCIPY_REVIEWED_LEGACY", {("1.17.1", fingerprint)})
    return text


@pytest.mark.parametrize("expression", [None, "BSD-3-Clause"])
def test_reviewed_scipy_metadata_is_recognized_only_with_matching_spdx(checker, reviewed_legacy, expression):
    assert checker._installed_license("scipy", _distribution(reviewed_legacy, expression=expression)) == "BSD-3-Clause"


@pytest.mark.parametrize("mutation", ["version", "removed_exception", "added_text", "spaces", "conflicting_spdx"])
def test_reviewed_scipy_does_not_accept_changed_composite_or_version(checker, reviewed_legacy, mutation):
    text, version, expression = reviewed_legacy, "1.17.1", None
    if mutation == "version":
        version = "1.17.2"
    elif mutation == "removed_exception":
        text = text.replace(" WITH GCC-exception-3.1", "")
    elif mutation == "added_text":
        text += "\nAGPL-3.0"
    elif mutation == "spaces":
        text = text.replace("upstream BSD", "upstream  BSD")
    else:
        expression = "MIT"
    with pytest.raises(SystemExit, match="unreviewed|conflicting"):
        checker._installed_license("scipy", _distribution(text, expression=expression, version=version))


@pytest.mark.parametrize("legacy", ["GPL-3.0", "AGPL-3.0", "unreviewed composite\nGCC-exception-3.1"])
def test_bsd_classifier_or_expression_cannot_hide_unreviewed_scipy_legacy(checker, legacy):
    with pytest.raises(SystemExit, match="unreviewed"):
        checker._installed_license("scipy", _distribution(legacy, expression="BSD-3-Clause",
            classifier="License :: OSI Approved :: BSD License"))


def test_profile_does_not_relax_static_declaration_inventory(checker, monkeypatch):
    inventory = json.loads((ROOT / "dependency-licenses.json").read_text(encoding="utf-8"))
    inventory["dependencies"] = [row for row in inventory["dependencies"] if row["name"] != "scipy"]
    monkeypatch.setattr(checker, "_read_inventory", lambda: inventory)
    with pytest.raises(SystemExit, match="inventory mismatch"):
        checker.check(check_installed=True, installed_extras=["dev", "planar"])


@pytest.mark.parametrize("installed,profile,match", [(False, ["planar"], "requires"),
                                                    (True, ["typo"], "unknown installed extras")])
def test_invalid_profile_is_refused(checker, installed, profile, match):
    with pytest.raises(SystemExit, match=match):
        checker.check(check_installed=installed, installed_extras=profile)


def test_cli_forwards_explicit_profile_without_changing_default(checker, monkeypatch):
    calls = []
    monkeypatch.setattr(checker, "check", lambda **options: calls.append(options))
    assert checker.main(["--check-installed", "--installed-extra", "dev", "--installed-extra", "planar"]) == 0
    assert checker.main(["--check-installed"]) == 0
    assert calls == [{"check_installed": True, "installed_extras": ["dev", "planar"]},
                     {"check_installed": True, "installed_extras": None}]


def test_actual_reviewed_scipy_metadata_when_present(checker):
    try:
        distribution = checker.importlib_metadata.distribution("scipy")
    except checker.importlib_metadata.PackageNotFoundError:
        pytest.skip("optional SciPy is absent; controlled decoder tests still exercise policy")
    # Optional installations remain subject to real metadata validation even
    # when the environment profile does not select sketch.
    checker._check_installed([{"name": "scipy", "scope": "optional:sketch"}], installed_extras={"dev", "planar"})
    legacy = distribution.metadata.get("License", "").strip().replace("\r\n", "\n")
    if (distribution.version, hashlib.sha256(legacy.encode()).hexdigest()) in checker.SCIPY_REVIEWED_LEGACY:
        assert "GPL-3.0-or-later WITH GCC-exception-3.1" in legacy
