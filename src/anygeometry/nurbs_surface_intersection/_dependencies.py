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
    ("_cert_exact_runtime.py","5b600f1be3d641fe1f6f4feee05542ef72b665f946c160aa745bd296e38e7725"),
    ("plane_nurbs.py","f60c8b6d80ab2c652af2db32df3adfd0817af467b2aac20d775281991aa21db2"),
    ("tolerance.py","f7709e89d2294dab1b71aecdb6a6a92f212a42905531e21ace78e89f1e0ab7f5"),
    ("errors.py","8dd49f04145b15f5d6bde5a91458b0df13d1c889a7dd760f04092dfcd064b2bd"),
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
