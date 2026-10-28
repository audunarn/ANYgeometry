"""Source-pinned read-only adapter for frozen private ANYgeometry primitives.

Pins come from the captured dirty baseline, not upstream HEAD. Only this
experimental package checks these source files; it never edits shared modules.
Checking actual module files also rejects an installed incompatible primitive.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ._source import NurbsSurfaceIntersectionIntegrityError


PINS = (
    ("nurbs.py","ab19c366cbe6ec3646592809bde77fffd60df336b21c7525b8add7b7f5d00545"),
    ("plane_nurbs.py","ad3c58f02479861aae44b1250203daa80ef3120060e2ab937043c1817d4dab53"),
    ("tolerance.py","f7709e89d2294dab1b71aecdb6a6a92f212a42905531e21ace78e89f1e0ab7f5"),
    ("errors.py","a6da540fff83f27e8f4fc5cc2b187f8413da6d2d3460461fa64b820429e04c89"),
)


def verify_dependencies():
    base = Path(__file__).resolve().parent.parent
    for name,expected in PINS:
        try:
            actual = hashlib.sha256((base/name).read_bytes()).hexdigest()
        except OSError as error:
            raise NurbsSurfaceIntersectionIntegrityError(f"source-pinned dependency unavailable: {name}") from error
        if actual!=expected:
            raise NurbsSurfaceIntersectionIntegrityError(f"source-pinned dependency changed: {name}")
