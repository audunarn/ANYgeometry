"""Source-pinned read-only adapter for frozen private ANYgeometry primitives.

Pins identify the reviewed current-source compatibility candidate, not new
numerical qualification or equivalence to the unrecoverable dirty baseline.
Only CRLF line endings are canonicalized to LF; every other byte is pinned.
This experimental package never edits the shared modules it checks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ._source import NurbsSurfaceIntersectionIntegrityError


PINS = (
    ("nurbs.py","fb0550ff48f5fae8e346f672183744a5dd9c6e981276286cccc074799ba5db06"),
    ("_cert_exact_runtime.py","68427a0294a160c10c32af712fb374eb4ac95d249fc719940f629e2a798c4afa"),
    ("plane_nurbs.py","77c8155418b33b7c1dbb22b5ee7f831ea43ef25ef89db206150b8fdf137868aa"),
    ("tolerance.py","994f4ce8306e796c5ac77b8e960e4cee917665a657ec0974059e32fca35b425e"),
    ("errors.py","a6da540fff83f27e8f4fc5cc2b187f8413da6d2d3460461fa64b820429e04c89"),
)


def verify_dependencies():
    base = Path(__file__).resolve().parent.parent
    for name,expected in PINS:
        try:
            actual = hashlib.sha256((base/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        except OSError as error:
            raise NurbsSurfaceIntersectionIntegrityError(f"source-pinned dependency unavailable: {name}") from error
        if actual!=expected:
            raise NurbsSurfaceIntersectionIntegrityError(f"source-pinned dependency changed: {name}")
