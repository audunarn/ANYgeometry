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
    ("nurbs.py","53310819017985a2bd6153a3449f27b24d087ecee713eb9c379f1eb021f21b77"),
    ("_cert_exact_runtime.py","68427a0294a160c10c32af712fb374eb4ac95d249fc719940f629e2a798c4afa"),
    ("plane_nurbs.py","f60c8b6d80ab2c652af2db32df3adfd0817af467b2aac20d775281991aa21db2"),
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
