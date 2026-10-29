"""Read-only materialization of authoritative face supports for interchange.

This capability constructs stored-coefficient geometry, never a fitted surface.
It does not validate trims or certify a downstream CAD writer.
"""
from fractions import Fraction as Q
from math import comb

from .errors import GeometryError

__all__ = ["InterchangeSupportError", "face_support_for_interchange"]


class InterchangeSupportError(GeometryError):
    """The authoritative support has no admitted exact explicit representation."""

    def __init__(self, message, *, code="geometry.interchange.support_unsupported"):
        self.code = code
        super().__init__(message)


def _stored(value):
    try:
        result = float(value)
    except OverflowError as error:
        raise InterchangeSupportError("explicit support coefficient overflows binary64") from error
    if Q(result) != value:
        raise InterchangeSupportError(
            "exact explicit support coefficient is not representable in binary64",
            code="geometry.interchange.coefficient_unrepresentable",
        )
    return result


def _elevate(points, degree):
    """Exact Bernstein degree elevation; no sampled point is a control point."""
    old = len(points) - 1
    return tuple(tuple(sum(
        (points[j][axis] * Q(comb(old, j) * comb(degree-old, i-j), comb(degree, i))
         for j in range(max(0, i-degree+old), min(old, i)+1)), Q(0))
        for axis in range(3)) for i in range(degree+1))


def _polynomial_coons(model, face):
    from .curves import Straight, Spline
    from .nurbs import NurbsSurface

    if len(face.corners) != 4 or any(len(side) != 1 for side in face.sides()):
        raise InterchangeSupportError("exact polynomial Coons materialization requires four single-edge sides")
    sides = []
    for side in face.sides():
        use = side[0]
        edge = model.edges[use.edge]
        if type(edge.curve) is Straight:
            identifiers = (edge.start, edge.end)
        elif type(edge.curve) is Spline:
            identifiers = (edge.start, *edge.curve.control_vertices, edge.end)
        else:
            raise InterchangeSupportError("topology Coons sides are not admitted polynomial curves")
        points = tuple(tuple(Q(float(x)) for x in model.vertex_position(i)) for i in identifiers)
        sides.append(points if use.forward else points[::-1])
    bottom, right, top, left = sides
    top, left = top[::-1], left[::-1]
    if not (bottom[0] == left[0] and bottom[-1] == right[0]
            and top[0] == left[-1] and top[-1] == right[-1]):
        raise InterchangeSupportError("polynomial Coons boundaries do not close exactly")
    p, q = max(len(bottom), len(top))-1, max(len(left), len(right))-1
    bottom, top = _elevate(bottom, p), _elevate(top, p)
    left, right = _elevate(left, q), _elevate(right, q)
    c00, c10, c01, c11 = bottom[0], bottom[-1], top[0], top[-1]
    net = []
    for i in range(p+1):
        u = Q(i, p)
        row = []
        for j in range(q+1):
            v = Q(j, q)
            row.append(tuple(_stored(
                (1-v)*bottom[i][k] + v*top[i][k]
                + (1-u)*left[j][k] + u*right[j][k]
                - ((1-u)*(1-v)*c00[k] + u*(1-v)*c10[k]
                   + (1-u)*v*c01[k] + u*v*c11[k])
            ) for k in range(3)))
        net.append(tuple(row))
    return NurbsSurface(p, q, (0.,)*(p+1)+(1.,)*(p+1),
                        (0.,)*(q+1)+(1.,)*(q+1), tuple(net))


def face_support_for_interchange(model, face_id):
    """Return an explicit exact authoritative support without changing ``model``.

    Existing Plane, Cylinder, Cone, RuledSurface, explicit CoonsSurface,
    ExtrudedSurface and NurbsSurface objects are returned unchanged. Topology
    Coons supports admit proven NURBS translations (including declared sweep
    charts), or four single-edge Straight/Spline sides whose exact Bernstein
    tensor coefficients fit binary64. Other cases raise GeometryError.

    The return represents the same geometric support. A recognized translation
    may rotate/reverse the chart using the owner's existing sweep convention;
    callers must not reuse source UV coordinates without chart transport.
    Trims, holes, topology and downstream conversion remain the caller's task.
    A separate face parameterization never replaces its authoritative support.
    """
    from .model import GeometryModel
    from .nurbs import NurbsSurface
    from .surfaces import Plane, Cylinder, Cone, RuledSurface, CoonsSurface, ExtrudedSurface
    from .nurbs_sweep_chart import has_sweep_chart, chart_model, _tensor
    from ._translational_pairs import _sweep

    if not isinstance(model, GeometryModel):
        raise TypeError("model must be a GeometryModel")
    if isinstance(face_id, bool) or not isinstance(face_id, int) or face_id not in model.faces:
        raise InterchangeSupportError("face_id must identify an existing face")
    face = model.faces[face_id]
    support = face.surface
    if type(support) in (Plane, Cylinder, Cone, RuledSurface, ExtrudedSurface, NurbsSurface):
        return support
    if type(support) is not CoonsSurface:
        raise InterchangeSupportError("face has no admitted authoritative explicit support")
    if support.has_boundaries:
        return support
    if has_sweep_chart(face):
        return chart_model(model, face_id).support
    recognized = _sweep(model, face_id)
    if recognized is not None:
        return _tensor(recognized[0], recognized[1])
    return _polynomial_coons(model, face)
