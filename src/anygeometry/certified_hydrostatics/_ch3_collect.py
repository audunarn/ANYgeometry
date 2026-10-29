"""CH3a2: the curved part of a snapshot (design CH3 sections 2.3 and 3.4).

``collect_curved`` runs the curved-face admission of CH3a1 (C1-C4, H1, H2), the four simplicity tests and the
certified ``sigma_chart`` of every curved face's loop (check 2 for curved faces), and records which edges are
section arcs of which curved face. It runs while the snapshot is collected because the snapshot document (design
3.4, convention version 2) records the exact supports, section graphs, corner identities and ``sigma_chart``."""
from __future__ import annotations

from dataclasses import dataclass, field

from ._ch3_admission_trims import CurvedFace, CurvedTrim, admit_curved_faces
from ._ch3_loops import check_loop, loop_sigma
from ._ch3_poly import Kernel, refusal
from ._errors import RefusalCategory
from ._work import Meter

__all__ = ["CurvedData", "collect_curved"]


@dataclass(frozen=True)
class CurvedData:
    """``faces``: curved face id -> ``CurvedFace``; ``sigma``: face id -> ``sigma_chart``; ``arcs``: section edge
    id -> ``(curved face id, CurvedTrim)``; ``hints``: edge id -> ``(domain, path)`` of the stored curve."""

    faces: dict
    sigma: dict
    arcs: dict
    hints: dict = field(default_factory=dict)


def collect_curved(model, curved_ids, meter: Meter, policy) -> CurvedData:
    faces = admit_curved_faces(model, curved_ids, meter, policy)
    k = Kernel(meter, policy.kernel_limits())
    sigma, arcs, hints, owner = {}, {}, {}, {}
    for fid, face in faces.items():
        check_loop(k, face)
        sigma[fid] = loop_sigma(k, face)
        for trim in face.trims:
            k.tick("admission_element")
            if trim.edge in arcs:
                raise refusal(RefusalCategory.UNSUPPORTED_TRIM,
                              f"edge {trim.edge} is a section arc of two curved faces", "curved_edge_shared")
            arcs[trim.edge] = (fid, trim)
            curve = model.edges[trim.edge].curve
            hints[trim.edge] = (tuple(curve.domain), int(curve.path))
        for vertex in face.corners:
            k.tick("admission_element")
            if vertex in owner:
                raise refusal(RefusalCategory.UNSUPPORTED_TRIM,
                              f"vertex {vertex} is a corner of two curved faces", "vertex_on_two_curved_faces")
            owner[vertex] = fid
    return CurvedData(faces, sigma, arcs, hints)
