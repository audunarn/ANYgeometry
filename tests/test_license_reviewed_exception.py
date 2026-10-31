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


WINDOWS_TEXT = (
    "Copyright (c) SciPy Developers. BSD-3-Clause terms.\n\n"
    " Name: OpenBLAS\n Files: x\\libopenblas*.dll\n Description: bundled\n"
    " Availability: https://example.invalid/\n License: BSD-3-Clause\n\n"
    " Name: GCC runtime library\n Files: x\\libopenblas*.dll\n Description: statically linked\n"
    " License: GPL-3.0-or-later WITH GCC-exception-3.1\n"
)


def _record(text: str, components: list[list[str]], package: str = "scipy") -> dict[str, object]:
    return {
        "components": components,
        "metadata_sha256": _sha(text),
        "package": package,
        "project_license": "BSD-3-Clause",
        "rationale": "test record",
        "version": "1.18.1",
    }


def _multi_row() -> dict[str, object]:
    row = _row()
    row["additional_reviewed_installed_licenses"] = [_record(WINDOWS_TEXT, [COMPONENTS[0], COMPONENTS[1]])]
    return row


class _FakeDistribution:
    def __init__(
        self,
        text: str | None,
        version: str | None = "1.18.1",
        *,
        name: str = "scipy",
        expression: str | None = None,
        classifiers: tuple[str, ...] = ("License :: OSI Approved :: BSD License",),
    ) -> None:
        self.version = version
        self.metadata = Message()
        self.metadata["Name"] = name
        if text is not None:
            self.metadata["License"] = text
        if expression is not None:
            self.metadata["License-Expression"] = expression
        for classifier in classifiers:
            self.metadata["Classifier"] = classifier


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


def _run(
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, object],
    installed: str | None,
    version: str | None = "1.18.1",
    *,
    extra: tuple[_FakeDistribution, ...] = (),
    **fields: object,
) -> None:
    primary = _FakeDistribution(installed, version, name=str(fields.pop("name", row["name"])), **fields)  # type: ignore[arg-type]
    monkeypatch.setattr(check_licenses.importlib_metadata, "distribution", lambda name: primary)
    monkeypatch.setattr(check_licenses.importlib_metadata, "distributions", lambda **kw: [primary, *extra])
    monkeypatch.setitem(check_licenses.INSTALLED_LICENSE_ALIASES, str(row["name"]), {"BSD-3-Clause"})
    check_licenses._check_installed([row])


def _rejected(monkeypatch: pytest.MonkeyPatch, row: dict[str, object], installed: str | None, **kw: object) -> str:
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


def test_pin_runs_even_when_the_expression_is_an_allowed_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _run(monkeypatch, _row(), REVIEWED_TEXT, expression="BSD-3-Clause")
    assert "re-review" in _rejected(monkeypatch, _row(), REVIEWED_TEXT + " tampered\n", expression="BSD-3-Clause")
    extra = REVIEWED_TEXT + "\n Name: libnew\n Files: y\n Description: z\n License: GPL-2.0-only\n"
    assert "re-review" in _rejected(monkeypatch, _row(), extra, expression="BSD-3-Clause")
    assert "re-review" in _rejected(monkeypatch, _row(), REVIEWED_TEXT, version="1.19.0", expression="BSD-3-Clause")


def test_absent_or_empty_legacy_text_fails_re_review_for_a_pinned_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for text in (None, "", "UNKNOWN"):
        assert "re-review" in _rejected(monkeypatch, _row(), text, expression="BSD-3-Clause")
    assert "re-review" in _rejected(monkeypatch, _row(), None)


@pytest.mark.parametrize("expression", ["GPL-3.0-only", "AGPL-3.0-only", "LGPL-2.1-or-later", "MIT"])
def test_expression_checks_still_run_for_a_pinned_package(monkeypatch: pytest.MonkeyPatch, expression: str) -> None:
    message = _rejected(monkeypatch, _row(), REVIEWED_TEXT, expression=expression)
    assert "license" in message and "re-review" not in message


@pytest.mark.parametrize(
    "classifier",
    ["License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
     "License :: OSI Approved :: GNU Affero General Public License v3"],
)
def test_forbidden_classifier_is_rejected_for_a_pinned_package(monkeypatch: pytest.MonkeyPatch, classifier: str) -> None:
    assert "classifier" in _rejected(monkeypatch, _row(), REVIEWED_TEXT, classifiers=(classifier,))


@pytest.mark.parametrize("version", [None, "", "1.18.1+local"])
def test_missing_or_different_version_is_rejected(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    assert "re-review" in _rejected(monkeypatch, _row(), REVIEWED_TEXT, version=version)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("rationale"),
        lambda r: r.update(extra="x"),
        lambda r: r.update(metadata_sha256="ABC"),
        lambda r: r.update(metadata_sha256=None),
        lambda r: r.update(version=None),
        lambda r: r.update(version=""),
        lambda r: r.update(version=1),
        lambda r: r.update(rationale=" "),
        lambda r: r.update(project_license="MIT"),
        lambda r: r.update(components=[]),
        lambda r: r.update(components="x"),
        lambda r: r.update(components=[["only-one"]]),
        lambda r: r.update(components=COMPONENTS + [["libextra", "LGPL-2.1-or-later"]]),
    ],
)
def test_malformed_or_mismatched_record_fields_are_rejected(monkeypatch: pytest.MonkeyPatch, mutate: object) -> None:
    row = _row()
    mutate(row["reviewed_installed_license"])  # type: ignore[operator,index]
    _rejected(monkeypatch, row, REVIEWED_TEXT)


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_shape_malformed_records_fail_the_static_check_too(index: int) -> None:
    mutations = [
        lambda r: r.pop("rationale"), lambda r: r.update(extra="x"), lambda r: r.update(metadata_sha256="ABC"),
        lambda r: r.update(metadata_sha256=None), lambda r: r.update(version=None), lambda r: r.update(version=""),
        lambda r: r.update(version=1), lambda r: r.update(rationale=" "), lambda r: r.update(project_license="MIT"),
        lambda r: r.update(components=[]), lambda r: r.update(components="x"), lambda r: r.update(components=[["a"]]),
    ]
    row = _row()
    mutations[index](row["reviewed_installed_license"])  # type: ignore[index]
    with pytest.raises(SystemExit):
        check_licenses._check_reviewed_record(row)


def test_name_case_and_separator_normalisation(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row("My_Pkg")
    _run(monkeypatch, row, REVIEWED_TEXT, name="my-pkg")
    assert "re-review" in _rejected(monkeypatch, row, REVIEWED_TEXT, name="other-pkg")


@pytest.mark.parametrize(
    "duplicate",
    [
        _FakeDistribution(REVIEWED_TEXT, expression="GPL-3.0-only"),
        _FakeDistribution(REVIEWED_TEXT, expression="BSD-3-Clause"),
        _FakeDistribution(REVIEWED_TEXT, classifiers=("License :: OSI Approved :: GNU General Public License v3 (GPLv3)",)),
        _FakeDistribution(REVIEWED_TEXT, "1.17.0"),
        _FakeDistribution(REVIEWED_TEXT + " other\n", "1.17.0"),
        _FakeDistribution(REVIEWED_TEXT),
    ],
)
def test_any_second_distribution_of_a_pinned_package_fails_closed(
    monkeypatch: pytest.MonkeyPatch, duplicate: _FakeDistribution
) -> None:
    assert "re-review" in _rejected(monkeypatch, _row(), REVIEWED_TEXT, extra=(duplicate,))
    assert "re-review" in _rejected(monkeypatch, _row(), REVIEWED_TEXT, extra=(duplicate,), expression="BSD-3-Clause")


def test_a_single_distribution_still_passes_and_other_packages_do_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _run(monkeypatch, _row(), REVIEWED_TEXT)
    _run(monkeypatch, _row(), REVIEWED_TEXT, expression="BSD-3-Clause")
    unrelated = _FakeDistribution("MIT", name="unrelated")
    _run(monkeypatch, _row(), REVIEWED_TEXT, extra=(unrelated,))


def test_each_reviewed_platform_text_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _run(monkeypatch, _multi_row(), REVIEWED_TEXT)
    _run(monkeypatch, _multi_row(), WINDOWS_TEXT)
    _run(monkeypatch, _multi_row(), WINDOWS_TEXT.replace("\n", "\r\n"))
    _run(monkeypatch, _multi_row(), WINDOWS_TEXT, expression="BSD-3-Clause")


def test_modified_or_unmatched_platform_text_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "re-review" in _rejected(monkeypatch, _multi_row(), WINDOWS_TEXT + " tampered\n")
    assert "re-review" in _rejected(monkeypatch, _multi_row(), "Name: x\n License: MIT\n")
    assert "re-review" in _rejected(monkeypatch, _multi_row(), WINDOWS_TEXT, version="1.19.0")
    assert "re-review" in _rejected(monkeypatch, _row(), WINDOWS_TEXT)  # no Windows record on a plain row


def test_a_record_with_an_unreviewed_copyleft_component_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_text = WINDOWS_TEXT + "\n Name: libnew\n Files: y\n Description: z\n License: GPL-2.0-only\n"
    row = _row()
    row["additional_reviewed_installed_licenses"] = [
        _record(bad_text, [COMPONENTS[0], COMPONENTS[1], ["libnew", "GPL-2.0-only"]])
    ]
    assert "copyleft" in _rejected(monkeypatch, row, bad_text)
    with pytest.raises(SystemExit):
        check_licenses._check_reviewed_record(row)
    swapped = _row()
    swapped["additional_reviewed_installed_licenses"] = [
        _record(WINDOWS_TEXT, [COMPONENTS[0], ["GCC runtime library", "GPL-2.0-only"]])
    ]
    with pytest.raises(SystemExit):
        check_licenses._check_reviewed_record(swapped)


def test_additional_records_are_validated_and_must_be_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row()
    row["additional_reviewed_installed_licenses"] = [_record(REVIEWED_TEXT, COMPONENTS)]
    assert "re-review" in _rejected(monkeypatch, row, REVIEWED_TEXT)  # same hash twice is ambiguous
    other = _row()
    other["additional_reviewed_installed_licenses"] = [_record(WINDOWS_TEXT, COMPONENTS[:2], package="other")]
    with pytest.raises(SystemExit):
        check_licenses._check_reviewed_record(other)
    broken = _row()
    broken["additional_reviewed_installed_licenses"] = "x"
    with pytest.raises(SystemExit):
        check_licenses._check_reviewed_record(broken)


def test_repository_inventory_records_all_reviewed_scipy_platforms() -> None:
    inventory = json.loads((ROOT / "dependency-licenses.json").read_text(encoding="utf-8"))
    (scipy,) = [r for r in inventory["dependencies"] if r["name"] == "scipy"]
    records = [scipy["reviewed_installed_license"], *scipy["additional_reviewed_installed_licenses"]]
    assert sorted(r["metadata_sha256"] for r in records) == sorted(
        [
            "2a463073ea6816d22b763c8e2f03886f4e8bf4f2a234f3c9267401d688588d17",
            "7f2f9e131a09f25323e392aac37fb2e263c600bb9d1ffc7c3b6ca3e1d017573e",
            "c85f6c4154a28ccb2a49fe9a1cbcacb17939c14ffad7a2f63d32655610652bb7",
        ]
    )
    assert all(r["version"] == "1.18.1" and r["package"] == "scipy" for r in records)


def test_runtime_scope_cannot_carry_a_reviewed_record(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row()
    row["scope"] = "runtime"
    assert "scope" in _rejected(monkeypatch, row, REVIEWED_TEXT)


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
