"""Closed shells and their enclosed volume and centroid (generality slice B1-P5c; seed of the H1 track).

``closed_shell(model, faces)``: the faces form a closed, consistently oriented shell when every edge
of their loops is used by exactly two of them, in opposite directions (then the loops orient every
face the same way, inwards or outwards, across each shared edge).

``enclosed_volume(model, faces)``: volume and centroid by the divergence theorem, with the fields
``(0, 0, z)`` (volume), ``(0, 0, x z)``, ``(0, 0, y z)``, ``(0, 0, z^2 / 2)`` (first moments), each face
by the chart moment engine of ``face_moments`` (H1a: plane faces in any orientation, NURBS faces;
section edges, straight lines and straight chart segments as boundaries). The result is a quadrature
value with an error estimate (the change from a doubled order), not a rigorous bound; a negative
volume means inward loops, and all signs flip.
"""
from __future__ import annotations


import numpy as np

from .errors import GeometryError


def closed_shell(model, faces) -> list[str]:
    """Problems that keep the faces from being a closed, consistently oriented shell (empty: closed).

    Also connected through shared edges (review B1-P5 round 1: two components oriented independently
    would add with opposite signs)."""
    faces = list(faces)
    uses: dict = {}
    for face_id in faces:
        face = model.faces[face_id]
        for loop in (face.loop,) + tuple(face.holes):
            for item in loop:
                uses.setdefault(item.edge, []).append((face_id, item.forward))
    problems = []
    for edge_id, owners in sorted(uses.items()):
        if len(owners) != 2:
            problems.append(f"edge {edge_id} bounds {len(owners)} faces of the shell (needs exactly 2)")
        elif owners[0][1] == owners[1][1]:
            problems.append(f"edge {edge_id} runs the same way in faces {owners[0][0]} and {owners[1][0]} "
                            "(inconsistent orientation)")
    if faces:
        neighbours: dict = {face: set() for face in faces}
        for owners in uses.values():
            for first, _f in owners:
                for second, _s in owners:
                    if first != second:
                        neighbours[first].add(second)
        reached, pending = {faces[0]}, [faces[0]]
        while pending:
            for other in neighbours[pending.pop()]:
                if other not in reached:
                    reached.add(other)
                    pending.append(other)
        if len(reached) != len(set(faces)):
            problems.append("the shell's faces are not connected through shared edges")
    return problems


def _nurbs_face_moments(model, face, order, origin):
    """The volume moments of one face in the world frame at ``origin`` (kept for the P5 tests)."""
    from .face_moments import Frame, face_moments
    return face_moments(model, face.id, Frame.world(origin), order=order)


def _moments(model, faces, order, origin):
    """The four field integrals in the local frame centred at ``origin`` (review B1-P5 round 3: absolute
    coordinates far from the world origin cancel); H1a: every face by the chart moment engine."""
    from .face_moments import Frame, shell_moments
    return shell_moments(model, faces, Frame.world(origin), order=order)


def enclosed_volume(model, faces, order: int = 24, rtol: float = 1e-8) -> dict:
    """``{"volume", "centroid", "estimate"}`` of the region the closed shell ``faces`` encloses.

    Integrated in a frame centred at the shell's vertices' mean (translated back); refused when the
    order-doubling difference exceeds ``rtol`` of the volume (volume) or of volume x extent (first
    moments; review B1-P5 round 4: one scale of volume x extent^2 accepted a 50 % volume error)."""
    faces = list(faces)
    problems = closed_shell(model, faces)
    if problems:
        raise GeometryError("not a closed shell: " + "; ".join(problems))
    positions = np.asarray([model.vertices[model.oriented_start_vertex(item)].position
                            for face_id in faces for loop in (model.faces[face_id].loop,) + tuple(model.faces[face_id].holes)
                            for item in loop], dtype=float)
    origin = positions.mean(axis=0)
    extent = float(np.max(np.linalg.norm(positions - origin, axis=1)))
    coarse, fine = _moments(model, faces, order, origin), _moments(model, faces, 2 * order, origin)
    if fine[0] < 0:
        coarse, fine = -coarse, -fine                            # inward loops
    volume = float(fine[0])
    if volume <= 0:
        raise GeometryError("the shell encloses no volume")
    difference = np.abs(fine - coarse)
    estimate = float(np.max(difference))
    if not (difference[0] <= rtol * volume and np.all(difference[1:] <= rtol * volume * extent)):
        raise GeometryError(f"the volume quadrature did not settle (moment difference {estimate:.3g})")
    centroid = fine[1:] / volume + origin
    return {"volume": volume, "centroid": centroid, "estimate": estimate}
