"""The source distribution must not ship internal working material.

Release 0.4.7 preparation (2026-11-24): the trial sdist carried internal plans,
probes and notes with local workstation paths, the workstation user directory
or customer-project names.  The wheel never carried them.  This test computes
the exact sdist file list (``setup.py egg_info`` writes it to ``SOURCES.txt``,
the same manifest ``sdist`` uses) into a temporary directory and scans every
listed file.

``README.md`` and ``ECOSYSTEM_GUIDE.md`` are named exceptions: they are
user-facing documents whose local-path wording is an open owner decision, not
internal material.  The pattern literals are split so this file does not match itself.
Package metadata (author name and e-mail) is intentional
and not matched.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

INTERNAL = re.compile(
    r"Audun" r"ArnesenNyhus"
    r"|(?<![A-Za-z])[A-Za-z]:[\\/]+(?:Github|ANYworkspace\w*|Users)\b"
    r"|/mnt" r"/c/"
    r"|Hbeam" r"Both",
    re.IGNORECASE,
)

NAMED_EXCEPTIONS = {"README.md", "ECOSYSTEM_GUIDE.md"}


def _sdist_sources(tmp_path: Path) -> list[str]:
    egg_base = tmp_path / "egg"
    egg_base.mkdir()
    completed = subprocess.run(
        [sys.executable, "setup.py", "-q", "egg_info", "--egg-base", str(egg_base)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    (sources,) = egg_base.glob("*.egg-info/SOURCES.txt")
    return [line.strip() for line in sources.read_text("utf-8").splitlines() if line.strip()]


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the source checkout, not an unpacked sdist")
def test_sdist_ships_no_internal_working_material(tmp_path: Path) -> None:
    listed = _sdist_sources(tmp_path)
    assert len(listed) > 500, f"implausibly small sdist manifest: {len(listed)} files"
    offenders = []
    for name in listed:
        if name in NAMED_EXCEPTIONS or name.endswith(".egg-info/PKG-INFO") or ".egg-info/" in name:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        match = INTERNAL.search(text)
        if match:
            offenders.append(f"{name}: {match.group(0)!r}")
    assert not offenders, "internal material in the sdist:\n" + "\n".join(offenders)
