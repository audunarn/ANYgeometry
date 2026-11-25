"""Refuse a source distribution that carries internal working material.

Usage: ``python tools/check_sdist_content.py dist/anygeometry-<version>.tar.gz``

Scans the built archive itself (every member name and every regular file's
content), not a manifest computed separately, because ``sdist`` can add files
while it writes the archive.  Text is decoded from UTF-8, or from UTF-16/32 when
a byte-order mark or NUL bytes show a wide encoding, so an internal note cannot
pass by encoding alone.  Exits 1 and lists the offenders; prints nothing else.

Internal material means: a workstation user directory on any drive or under
``/home`` or ``/Users`` (CI runner home ``runner`` excepted), the workstation
checkout roots (``<drive>:/Github``, ``<drive>:/ANYworkspace*``), a UNC share
path, WSL mounts of Windows drives, or a customer-project marker.  Named
exceptions are user-facing documents whose wording is an open owner decision.
The pattern literals are split so this file does not match itself.

Contract (owner decision S (a), 2026-11-25): this is a tripwire for known
markers, not a proof that no private path form exists; the primary control is
the explicit MANIFEST.in exclusion list.  Known limits (Codex round 3 Lows):
a BOM-less UTF-32 ``PKG-INFO`` is split at the first blank line of the
NUL-stripped text, so its header may be read from the wrong decoding; and a
backslash-escaped literal such as ``"\\\\alpha\\n"`` in a document is read as a
UNC path (a false positive that blocks the release until the text changes).
Also (Codex round 4 Low): a URL with a doubled slash before ``home`` or
``Users`` (``https://example.com//home/x``) is read as a home path.
"""

from __future__ import annotations

import codecs
import re
import sys
import tarfile
from pathlib import PurePosixPath

_ANY_CASE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+(?:Users[\\/]|Github\b|ANYworkspace)"
    r"|(?:^|[^\\\w])\\\\[\w.-]{2,}\\[\w$.-]+"
    r"|/mnt" r"/[a-z]/"
    r"|Audun" r"ArnesenNyhus"
    r"|Hbeam" r"Both",
    re.IGNORECASE,
)
# POSIX home directories are case-sensitive; ``/users/`` inside a URL is not one.
_EXACT_CASE = re.compile(r"(?<![\w.])/(?:home|Users)/(?!runner(?![\w.-]))[\w.-]+")


class _Pattern:
    def search(self, text: str):
        return _ANY_CASE.search(text) or _EXACT_CASE.search(text)


PATTERN = _Pattern()

# Relative to the archive root directory (``anygeometry-<version>/``): whole-file exceptions.
NAMED_EXCEPTIONS = frozenset({"README.md", "ECOSYSTEM_GUIDE.md"})
# Metadata whose body is README.md: only the header block is scanned.
_METADATA = frozenset({"PKG-INFO", "src/ANYgeometry.egg-info/PKG-INFO"})


def decode(data: bytes) -> str:
    """Decode text without letting a wide encoding hide its content."""
    for bom, codec in ((codecs.BOM_UTF32_LE, "utf-32-le"), (codecs.BOM_UTF32_BE, "utf-32-be"),
                       (codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"),
                       (codecs.BOM_UTF16_BE, "utf-16-be")):
        if data.startswith(bom):
            return data[len(bom):].decode(codec, errors="replace")
    text = data.decode("utf-8", errors="replace")
    if b"\x00" in data:
        # No BOM but NUL bytes: also read every wide interpretation and the bytes with NULs removed.
        wide = [data.decode(codec, errors="replace") for codec in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")]
        text = "\n".join([text, data.replace(b"\x00", b"").decode("utf-8", errors="replace"), *wide])
    return text


def offenders(archive_path: str) -> list[str]:
    found = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            relative = "/".join(parts[1:])
            match = PATTERN.search(member.name) or (PATTERN.search(member.linkname) if member.linkname else None)
            if match:
                found.append(f"{member.name}: member name or link target {match.group(0)!r}")
                continue
            if not member.isfile() or relative in NAMED_EXCEPTIONS:
                continue
            handle = archive.extractfile(member)
            text = decode(handle.read()) if handle is not None else ""
            if relative in _METADATA:
                text = text.replace("\r\n", "\n").split("\n\n", 1)[0]
            match = PATTERN.search(text)
            if match:
                found.append(f"{member.name}: {match.group(0)!r}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_sdist_content.py <sdist.tar.gz>", file=sys.stderr)
        return 2
    found = offenders(argv[0])
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
