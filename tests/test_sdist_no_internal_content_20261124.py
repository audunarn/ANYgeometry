"""The source distribution must not ship internal working material.

Release 0.4.7 preparation (2026-11-24): the trial sdist carried 27 internal
plans, probes and notes with workstation paths, the workstation user directory
or a customer-project name; the wheel never carried them.  The binding check is
``tools/check_sdist_content.py``, which the publish workflow runs on the built
archive before the release bundle is made.  This test builds a real sdist into
a temporary directory and runs the same scanner, and pins the scanner's
patterns and encodings with small probes.  The pattern literals in the probes
are split so this file does not match itself.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_sdist_content as scanner  # noqa: E402

USER = "Audun" + "ArnesenNyhus"
CUSTOMER = "Hbeam" + "Both"


def _archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "probe-0.0.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo("probe-0.0.0/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("text", [
    "C:" + "/Github/ANYgeometry",
    "E:" + "\\ANYworkspaceWork\\x",
    "D:" + "\\Users\\someone\\notes",
    "\\\\" + "fileserver\\share\\notes.md",
    "/home" + "/alice/work",
    "/Users" + "/alice/work",
    "/mnt" + "/c/Github",
    "\\\\" + "192.168.1.10\\share\\note.md",
    "\\\\" + "fileserver\\x\\note.md",
    "/home" + "/1001/private/work",
    "[notes](file://" + "/home/alice/private/work)",
    "workdir:" + "/home/alice/private/work",
    USER,
    CUSTOMER,
])
def test_scanner_flags_internal_markers(text: str) -> None:
    assert scanner.PATTERN.search(text), text


@pytest.mark.parametrize("text", [
    "https://github.com/audunarn/ANYgeometry",
    "C:\\Python\\Python314\\python.exe",
    "/home/runner/work/_temp/x",
    "/Users/runner/work/_temp/x",
    '"\\\\x\\n"',
    "https://api.github.com/users/octocat",
    "see docs/home/users.md",
    "https://example.com/home/x",
    "HOME:/home/runner",
    "file:///home/runner",
])
def test_scanner_leaves_public_and_ci_paths(text: str) -> None:
    assert not scanner.PATTERN.search(text), text


@pytest.mark.parametrize("codec", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32"])
def test_scanner_reads_wide_encodings(tmp_path: Path, codec: str) -> None:
    path = _archive(tmp_path, {"docs/note.md": f"see {CUSTOMER} drawing".encode(codec)})
    assert scanner.offenders(str(path)), codec


def test_scanner_reads_utf32_without_bom(tmp_path: Path) -> None:
    text = "Saved to 中" + "D:" + "\\Users\\someone\\private"
    path = _archive(tmp_path, {"docs/note.md": text.encode("utf-32-le")})
    assert scanner.offenders(str(path))


def test_scanner_checks_link_targets_metadata_headers_and_egg_info_lookalikes(tmp_path: Path) -> None:
    path = tmp_path / "links-0.0.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        link = tarfile.TarInfo("links-0.0.0/docs/example.md")
        link.type, link.linkname = tarfile.SYMTYPE, "/home" + "/alice/private/note.md"
        archive.addfile(link)
        for name, data in {
            "links-0.0.0/PKG-INFO": f"Metadata-Version: 2.4\nSummary: {CUSTOMER} tools\n\nC:".encode()
            + b"/Github/readme-body",
            "links-0.0.0/docs/notes.egg-info/case.md": CUSTOMER.encode(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    found = scanner.offenders(str(path))
    assert len(found) == 3, found
    assert scanner.offenders(str(_archive(tmp_path, {"PKG-INFO": b"Metadata-Version: 2.4\n\nC:" + b"/Github/readme"}))) == []


def test_scanner_checks_member_names_and_keeps_named_exceptions(tmp_path: Path) -> None:
    path = _archive(tmp_path, {
        f"docs/{CUSTOMER}-notes.md": b"plain",
        "README.md": b"pip install -e C:" + b"\\Github\\ANYgeometry",
        "src/clean.py": b"x = 1\n",
    })
    found = scanner.offenders(str(path))
    assert len(found) == 1 and "member name" in found[0], found
    assert scanner.main([str(path)]) == 1


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the source checkout, not an unpacked sdist")
def test_built_sdist_ships_no_internal_working_material(tmp_path: Path) -> None:
    # The archive itself, not a separately computed manifest: sdist can add files while it writes.
    pytest.importorskip("setuptools", reason="building the sdist needs setuptools; the publish workflow check is binding")
    completed = subprocess.run(
        [sys.executable, "setup.py", "-q", "egg_info", "--egg-base", str(tmp_path),
         "sdist", "--dist-dir", str(tmp_path / "dist")],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    (archive,) = (tmp_path / "dist").glob("*.tar.gz")
    with tarfile.open(archive, "r:gz") as opened:
        assert len(opened.getnames()) > 500, "implausibly small sdist"
    found = scanner.offenders(str(archive))
    assert not found, "internal material in the sdist:\n" + "\n".join(found)
