"""Reviewed installed-licence exception of tools/check_licenses.py (hash-pinned, data-driven)."""

from __future__ import annotations

from email.message import Message
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("check_licenses", ROOT / "tools" / "check_licenses.py")
assert _SPEC is not None and _SPEC.loader is not None
check_licenses = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_licenses)

COMPONENTS = [
    ["OpenBLAS", "BSD-3-Clause"],
    ["GCC runtime library", "GPL-3.0-or-later WITH GCC-exception-3.1"],
    ["libquadmath", "LGPL-2.1-or-later"],
]
REVIEWED_TEXT = (
    "Copyright (c) SciPy Developers. BSD-3-Clause terms.\n\n"
    " Name: OpenBLAS\n Files: x/libopenblas*.so\n Description: bundled\n"
    " Availability: https://example.invalid/\n License: BSD-3-Clause\n\n"
    " Name: GCC runtime library\n Files: x/libgfortran*.so\n Description: linked\n"
    " License: GPL-3.0-or-later WITH GCC-exception-3.1\n\n"
    " Name: libquadmath\n Files: x/libquadmath*.so\n Description: linked\n"
    " License: LGPL-2.1-or-later\n"
)


class _FakeDistribution:
    def __init__(self, text: str, version: str = "1.18.1") -> None:
        self.version = version
        self.metadata = Message()
        self.metadata["License"] = text


def _sha(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _row(name: str = "scipy", text: str = REVIEWED_TEXT, with_record: bool = True) -> dict[str, object]:
    row: dict[str, object] = {
        "bundled": False,
        "license_expression": "BSD-3-Clause",
        "name": name,
        "requirement": f"{name}>=1.13",
        "scope": "optional:sketch",
    }
    if with_record:
        row["reviewed_installed_license"] = {
            "components": COMPONENTS,
            "metadata_sha256": _sha(text),
            "package": name,
            "project_license": "BSD-3-Clause",
            "rationale": "test record",
            "version": "1.18.1",
        }
    return row


def _run(monkeypatch: pytest.MonkeyPatch, row: dict[str, object], installed: str, version: str = "1.18.1") -> None:
    monkeypatch.setattr(
        check_licenses.importlib_metadata,
        "distribution",
        lambda name: _FakeDistribution(installed, version),
    )
    monkeypatch.setitem(check_licenses.INSTALLED_LICENSE_ALIASES, str(row["name"]), {"BSD-3-Clause"})
    check_licenses._check_installed([row])


def _rejected(monkeypatch: pytest.MonkeyPatch, row: dict[str, object], installed: str, **kw: str) -> str:
    with pytest.raises(SystemExit) as caught:
        _run(monkeypatch, row, installed, **kw)
    return str(caught.value)


def test_reviewed_text_with_matching_hash_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _run(monkeypatch, _row(), REVIEWED_TEXT)
    _run(monkeypatch, _row(), REVIEWED_TEXT.replace("\n", "\r\n"))


def test_modified_text_is_rejected_with_re_review_message(monkeypatch: pytest.MonkeyPatch) -> None:
    message = _rejected(monkeypatch, _row(), REVIEWED_TEXT + " tampered\n")
    assert "re-review" in message


def test_text_gaining_a_new_copyleft_component_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    extra = REVIEWED_TEXT + "\n Name: libnew\n Files: y\n Description: z\n License: GPL-2.0-only\n"
    assert "re-review" in _rejected(monkeypatch, _row(), extra)


def test_changed_version_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "re-review" in _rejected(monkeypatch, _row(), REVIEWED_TEXT, version="1.19.0")


def test_same_text_under_another_package_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "review-required" in _rejected(monkeypatch, _row("other", with_record=False), REVIEWED_TEXT)
    stolen = _row("other")
    stolen["reviewed_installed_license"]["package"] = "scipy"  # type: ignore[index]
    assert "recorded for 'scipy'" in _rejected(monkeypatch, stolen, REVIEWED_TEXT)


def test_plain_gpl_package_is_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "review-required" in _rejected(monkeypatch, _row("gplpkg", with_record=False), "GPL-3.0-only")
    assert "re-review" in _rejected(monkeypatch, _row(), "GPL-3.0-only")


def test_record_components_must_match_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row()
    row["reviewed_installed_license"]["components"] = COMPONENTS[:2]  # type: ignore[index]
    assert "re-review" in _rejected(monkeypatch, row, REVIEWED_TEXT)


def test_repository_inventory_records_scipy_review_and_static_check_passes() -> None:
    inventory = json.loads((ROOT / "dependency-licenses.json").read_text(encoding="utf-8"))
    (scipy,) = [r for r in inventory["dependencies"] if r["name"] == "scipy"]
    record = scipy["reviewed_installed_license"]
    assert record["package"] == "scipy"
    assert record["metadata_sha256"] == "2a463073ea6816d22b763c8e2f03886f4e8bf4f2a234f3c9267401d688588d17"
    assert record["project_license"] == "BSD-3-Clause"
    assert "2026-10-30" in record["rationale"]
    assert ["libquadmath", "LGPL-2.1-or-later"] in record["components"]
    assert [r for r in inventory["dependencies"] if "reviewed_installed_license" in r] == [scipy]
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_licenses.py")], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
