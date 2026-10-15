"""Closed shells and their enclosed volume and centroid (generality slice B1-P5c; seed of the H1 track).

``closed_shell(model, faces)``: the faces form a closed, consistently oriented shell when every edge
of their loops is used by exactly two of them, in opposite directions (then the loops orient every
face the same way, inwards or outwards, across each shared edge).

``enclosed_volume(model, faces)``: volume and centroid by the divergence theorem, with the fields
``(0, 0, z)`` (volume), ``(0, 0, x z)``, ``(0, 0, y z)``, ``(0, 0, z^2 / 2)`` (first moments). Each
face contributes ``int f n_z dA`` with the normal its loop orientation gives:

- plane faces with ``n_z = 0`` (exactly) contribute nothing;
- horizontal plane faces (``z = z0``): Green's theorem in ``(x, y)`` along their loops,
  ``int f n_z dA = oint F dy`` with ``dF/dx = f(x, y, z0)``; exact on lines (Gauss-Legendre of
  sufficient degree), Gauss-Legendre per arc piece on section edges;
- NURBS faces: Green's theorem in the chart, ``int int g du dv = oint G dv`` with
  ``g = f(S) (S_u x S_v)_z`` and ``G(u, v) = int_0^u g(s, v) ds`` (inner Gauss-Legendre); section
  edges and straight chart segments as boundaries.

Other faces refuse. The result is a quadrature value with an error estimate (the change from a
doubled order), not a rigorous bound; a negative volume means inward loops, and all signs flip.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .curves import Straight
from .errors import GeometryError
from .support_curves import PlaneNurbsCurve


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


def _fields(points):
    """Integrands f of the four fields at world points (n, 3): z, x z, y z, z^2 / 2."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack((z, x * z, y * z, .5 * z * z), axis=1)


def _edge_samples(model, item, order):
    """World points and derivatives d/dt along an oriented edge (Gauss nodes per smooth piece) with
    weights: ``(points, derivatives, weights)`` in the loop direction."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edge = model.edges[item.edge]
    curve = edge.curve
    if isinstance(curve, Straight):
        start, end = model.vertices[edge.start].position, model.vertices[edge.end].position
        if not item.forward:
            start, end = end, start
        t = .5 * (nodes + 1.)
        points = start + t[:, None] * (end - start)
        return points, np.tile(end - start, (order, 1)), .5 * weights
    if isinstance(curve, PlaneNurbsCurve):
        knots = sorted({0.0, 1.0, *curve.breakpoints()})
        points, derivatives, ws = [], [], []
        for lo, hi in zip(knots, knots[1:]):
            t = .5 * (hi - lo) * (nodes + 1.) + lo
            points.append(curve.evaluate(t))
            derivatives.append(curve.derivative(t))
            ws.append(.5 * (hi - lo) * weights)
        points, derivatives, ws = np.vstack(points), np.vstack(derivatives), np.concatenate(ws)
        if not item.forward:
            derivatives = -derivatives
        return points, derivatives, ws
    raise GeometryError(f"edge {item.edge}: only straight and section edges bound the volume's plane faces")


def _plane_face_moments(model, face, order):
    from .surfaces import Plane
    plane = face.surface
    normal = np.cross(plane.u_vector, plane.v_vector)
    if normal[0] == 0.0 and normal[1] == 0.0 and normal[2] == 0.0:
        raise GeometryError("degenerate plane")
    exact = [Fraction(float(value)) for value in normal]
    if exact[2] == 0:
        return np.zeros(4)                                       # vertical: n_z = 0
    if exact[0] != 0 or exact[1] != 0:
        raise GeometryError(f"face {face.id}: tilted plane faces are not supported by the volume yet")
    z0 = float(plane.origin[2])
    total = np.zeros(4)
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            points, derivatives, weights = _edge_samples(model, item, order)
            x, y = points[:, 0], points[:, 1]
            dy = derivatives[:, 1]
            # F with dF/dx = f(x, y, z0): z0 x, z0 x^2 / 2, z0 x y, z0^2 x / 2
            antiderivative = np.stack((z0 * x, .5 * z0 * x * x, z0 * x * y, .5 * z0 * z0 * x), axis=1)
            total += weights @ (antiderivative * dy[:, None])
    return total


def _nurbs_face_moments(model, face, order):
    from .nurbs import CurveOnSurface
    surface = face.surface
    nodes, weights = np.polynomial.legendre.leggauss(order)
    (r0, r1), (q0, q1) = surface.u_range, surface.v_range
    u_knots = sorted({(k - r0) / (r1 - r0) for k in surface.knots_u})
    v_knots = sorted({(k - q0) / (q1 - q0) for k in surface.knots_v})
    total = np.zeros(4)
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            curve = model.edges[item.edge].curve
            if isinstance(curve, PlaneNurbsCurve):
                knots = sorted({0.0, 1.0, *curve.breakpoints()})
                chart, rates, ws = [], [], []
                for lo, hi in zip(knots, knots[1:]):
                    t = .5 * (hi - lo) * (nodes + 1.) + lo
                    chart.append(curve.chart(t))
                    rates.append(curve.chart_derivative(t))
                    ws.append(.5 * (hi - lo) * weights)
                chart, rates, ws = np.vstack(chart), np.vstack(rates), np.concatenate(ws)
            elif isinstance(curve, CurveOnSurface) and curve.pcurve.degree == 1 and len(curve.pcurve.controls) == 2:
                p0, p1 = (np.asarray([float(v) for v in curve.pcurve.exact_point(s)]) for s in (0, 1))
                # pieces between the segment's crossings of knot lines (smooth on each)
                cuts = {0.0, 1.0}
                for axis, knots in ((0, u_knots), (1, v_knots)):
                    if p1[axis] != p0[axis]:
                        cuts.update(c for c in ((k - p0[axis]) / (p1[axis] - p0[axis]) for k in knots) if 0.0 < c < 1.0)
                cuts = sorted(cuts)
                chart, rates, ws = [], [], []
                for lo, hi in zip(cuts, cuts[1:]):
                    t = .5 * (hi - lo) * (nodes + 1.) + lo
                    chart.append(p0 + t[:, None] * (p1 - p0))
                    rates.append(np.tile(p1 - p0, (order, 1)))
                    ws.append(.5 * (hi - lo) * weights)
                chart, rates, ws = np.vstack(chart), np.vstack(rates), np.concatenate(ws)
            else:
                raise GeometryError(f"edge {item.edge}: only section edges and straight chart segments bound "
                                    "the volume's NURBS faces")
            if not item.forward:
                rates = -rates
            # G(u, v) = int_0^u g(s, v) ds by an inner Gauss-Legendre rule at every boundary node, split at
            # the chart images of the interior u knots (review B1-P5 round 1: a narrow span was missed)
            inner = np.empty((len(chart), 4))
            for k, (u, v) in enumerate(chart):
                cuts = [0.0] + [c for c in u_knots if 0.0 < c < u] + [u]
                value = np.zeros(4)
                for lo, hi in zip(cuts, cuts[1:]):
                    s = .5 * (hi - lo) * (nodes + 1.) + lo
                    uv = np.stack((s, np.full_like(s, v)), axis=1)
                    points = surface.evaluate_many(uv)
                    s_u, s_v = surface.derivatives_many(uv)
                    jacobian_z = s_u[:, 0] * s_v[:, 1] - s_u[:, 1] * s_v[:, 0]
                    value += (.5 * (hi - lo) * weights) @ (_fields(points) * jacobian_z[:, None])
                inner[k] = value
            total += ws @ (inner * rates[:, 1][:, None])
    return total


def _moments(model, faces, order):
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    total = np.zeros(4)
    for face_id in faces:
        face = model.faces[face_id]
        if isinstance(face.surface, Plane):
            total += _plane_face_moments(model, face, order)
        elif isinstance(face.surface, NurbsSurface):
            total += _nurbs_face_moments(model, face, order)
        else:
            raise GeometryError(f"face {face_id}: only plane and NURBS faces bound the volume")
    return total


def enclosed_volume(model, faces, order: int = 24) -> dict:
    """``{"volume", "centroid", "estimate"}`` of the region the closed shell ``faces`` encloses."""
    faces = list(faces)
    problems = closed_shell(model, faces)
    if problems:
        raise GeometryError("not a closed shell: " + "; ".join(problems))
    coarse, fine = _moments(model, faces, order), _moments(model, faces, 2 * order)
    if fine[0] < 0:
        coarse, fine = -coarse, -fine                            # inward loops
    volume = float(fine[0])
    if volume <= 0:
        raise GeometryError("the shell encloses no volume")
    centroid = fine[1:] / volume
    estimate = float(np.max(np.abs(fine - coarse)))
    return {"volume": volume, "centroid": centroid, "estimate": estimate}
