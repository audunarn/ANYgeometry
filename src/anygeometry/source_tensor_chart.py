"""Exact tensor charts for authored polynomial and sampled four-sided surfaces."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np

from .curves import Straight, Spline
from .errors import GeometryError
from .extrusions import BezierDirectrix
from .model import GeometryModel
from .nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from .nurbs_sweep_chart import _Edges, _Faces, _stored, _tensor
from .surfaces import CoonsSurface, ExtrudedSurface, RuledSurface

_UNIT = (0., 0., 1., 1.)
_CORNERS = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
_METHODS = {
    cls: {name: (getattr(cls, name), getattr(cls, name).__code__) for name in names}
    for cls, names in (
        (GeometryModel, ('face_point', 'sample_edge', '_sample_edge_object', '_spline_points',
                         '_topology_face_point', '_topology_face_point_and_derivatives',
                         '_chain_eval', '_chain_data')),
        (BezierDirectrix, ('point', '_de_casteljau')),
        (ExtrudedSurface, ('evaluate', 'evaluate_many', '_parameters')),
        (CoonsSurface, ('evaluate', '_sample')),
        (RuledSurface, ('evaluate', '_sample')),
    )
}


def _canonical_evaluators(value):
    methods = _METHODS.get(type(value))
    if methods is None or any(name in vars(value) or getattr(type(value), name) is not method
                              or method.__code__ is not code for name, (method, code) in methods.items()):
        raise GeometryError('source tensor requires the canonical authored evaluators')


def _require_tensor_runtime(model, face):
    """Audit the nonserialized evaluators of supported original face definitions."""
    surface = face.surface
    if face.parameterization is not None or face.holes or len(face.loop) != 4:
        return
    if type(surface) is ExtrudedSurface and type(surface.directrix) is BezierDirectrix:
        _canonical_evaluators(surface.directrix)
    elif type(surface) is CoonsSurface:
        if not surface.has_boundaries:
            curves = [type(model.edges[use.edge].curve) for use in face.loop]
            if curves != [Spline, Straight, Spline, Straight]:
                return
            cached = model._topology_corner_cache.get(face.id)
            if cached is not None and cached[0] == model.revision and cached[1] is face:
                _, _, chains, corners = cached
                if len(chains) != 4 or len(corners) != 4:
                    raise GeometryError('source tensor topology cache lacks four sides')
                for use, chain, corner in zip(face.loop, chains, corners):
                    uses, lengths, total, cumulative, edges = chain
                    if (uses != (use,) or len(edges) != 1 or edges[0] is not model.edges[use.edge]
                            or not total > 0 or not np.array_equal(lengths, (total,))
                            or not np.array_equal(cumulative, (0.,total))
                            or not np.array_equal(corner, model.vertices[model.oriented_start_vertex(use)].position)):
                        raise GeometryError('source tensor topology cache differs from its directed rim')
    elif type(surface) is not RuledSurface:
        return
    _canonical_evaluators(model)
    _canonical_evaluators(surface)


def _point(row):
    return tuple(Q(float(x)) for x in row)


def _mix(a, b, t):
    return tuple(x + t * (y - x) for x, y in zip(a, b))


def _bezier(points):
    n = len(points)
    return NurbsCurve(n - 1, (0.,) * n + (1.,) * n, points)


def _edge_curve(model, edge):
    if type(edge.curve) is NurbsCurve:
        from .nurbs_runtime_binding import validate_curve_runtime
        validate_curve_runtime(edge.curve)
        return edge.curve
    if type(edge.curve) not in (Straight, Spline):
        return None
    interior = edge.curve.control_vertices if type(edge.curve) is Spline else ()
    return _bezier(tuple(tuple(model.vertices[v].position) for v in (edge.start, *interior, edge.end)))


def _breaks(points):
    return tuple(Q(i, len(points)-1) for i in range(len(points)))


def _sample(points, t):
    position = t * (len(points)-1)
    index = min(int(position), len(points)-2)
    return _mix(_point(points[index]), _point(points[index+1]), position-index)


def _sampled(surface):
    if type(surface) is RuledSurface:
        bottom, top = surface.first_boundary, surface.second_boundary
        left, right = (bottom[0], top[0]), (bottom[-1], top[-1])
    else:
        bottom, right, top, left = surface.bottom, surface.right, surface.top, surface.left
    if any(_point(a) != _point(b) for a, b in (
            (bottom[0], left[0]), (bottom[-1], right[0]),
            (top[0], left[-1]), (top[-1], right[-1]))):
        raise GeometryError('source tensor requires exact sampled corner correspondence')
    us = sorted(set(_breaks(bottom)) | set(_breaks(top)))
    vs = sorted(set(_breaks(left)) | set(_breaks(right)))
    def control(u, v):
        a = _mix(_sample(bottom, u), _sample(top, u), v)
        b = _mix(_sample(left, v), _sample(right, v), u)
        c = _mix(_mix(_point(bottom[0]), _point(bottom[-1]), u),
                 _mix(_point(top[0]), _point(top[-1]), u), v)
        return tuple(_stored(x+y-z) for x, y, z in zip(a,b,c))
    def knots(values):
        return (0., *map(_stored, values), 1.)
    return NurbsSurface(1, 1, knots(us), knots(vs),
                        tuple(tuple(control(u, v) for v in vs) for u in us))


def _matches_edge(support, side, curve, forward):
    # On an open knot span, homogeneous curve components have degree d.
    # The tensor restriction along an axis is polynomial of degree k.
    # Equality at d+k+1 distinct exact parameters proves the identity after
    # multiplying by the positive NURBS denominator. Include every knot too.
    a, b = map(Q, curve.domain)
    start, end = _CORNERS[side], _CORNERS[(side+1)%4]
    if not forward:
        start, end = end, start
    axis = 0 if start[0] != end[0] else 1
    degree, knots = ((support.degree_u, support.knots_u) if axis == 0
                     else (support.degree_v, support.knots_v))
    values = {Q(0), Q(1)}
    values.update(t for x in curve.knots if 0 < (t := (Q(x)-a)/(b-a)) < 1)
    values.update(t for x in knots if 0 < (t := (Q(x)-Q(start[axis]))/
                                          (Q(end[axis])-Q(start[axis]))) < 1)
    cuts = sorted(values)
    # A knot value may select the neighboring span. Use d+k+1 strictly
    # interior parameters on each span, in addition to both knot values.
    count = curve.degree + degree + 2
    uv0, uv1 = _point(start), _point(end)
    for lo, hi in zip(cuts, cuts[1:]):
        for j in range(count+2):
            t = lo + (hi-lo)*j/(count+1)
            if curve.exact_point(t) != support.exact_point(*_mix(uv0, uv1, t)):
                raise GeometryError('source tensor rim differs from its directed authored edge')
    return NurbsCurve(1, _UNIT, (start, end))


class _TensorView:
    def __init__(self, model, face, support, edges):
        self.raw, self.face_id, self._support = model, face.id, support
        self._revision = model.revision
        self._edges = edges
        self.edges, self.faces = _Edges(self), _Faces(self)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def support(self):
        if self.raw.revision != self._revision or self.raw._transaction_journal is not None:
            raise GeometryError('source tensor view is stale or has an open transaction')
        return self._support

    def chart_edge(self, edge):
        self.support
        return self._edges.get(edge.id, edge)

    def chart_face(self, face):
        return replace(face, surface=self.support) if face.id == self.face_id else face

    def face_boundary_uv(self, face_id, edge_id, parameters):
        self.support
        if face_id != self.face_id or edge_id not in self._edges:
            raise GeometryError('source tensor rim does not own this edge')
        return self._edges[edge_id].curve.chart(parameters)

    def face_point(self, face_id, u, v):
        if face_id == self.face_id:
            return self.support.evaluate(u, v)
        return self.raw.face_point(face_id, u, v)


def source_tensor_chart(model, face_id):
    """Return the exact original unit-square tensor and its directed edge maps.

    This derives a source chart only. Material, whole-cell approximation,
    source coverage and transfer all require their ordinary owner proofs.
    """
    face = model.faces[face_id]
    surface = face.surface
    if type(surface) not in (RuledSurface, CoonsSurface, ExtrudedSurface):
        return None
    if (face.parameterization is not None or face.holes or len(face.loop) != 4
            or tuple(face.corners) != (0, 1, 2, 3) or len({u.edge for u in face.loop}) != 4):
        return None
    if type(surface) is ExtrudedSurface and type(surface.directrix) is BezierDirectrix:
        if surface.u_range != (0., 1.) or surface.v_range != (0., 1.):
            return None
        if (not np.array_equal(surface.directrix.controls, surface.directrix._array)
                or not np.array_equal(surface.vector, surface._vector)):
            raise GeometryError('source tensor extrusion differs from evaluated coefficients')
        support = _tensor(_bezier(surface.directrix.controls), _point(surface.vector))
    elif type(surface) is RuledSurface or type(surface) is CoonsSurface and surface.has_boundaries:
        support = _sampled(surface)
    elif type(surface) is CoonsSurface:
        from .native_support_snapshots import _coons_snapshot
        from .polynomial_extrusion_support import _compile, COONS_POLYNOMIAL
        curves = [type(model.edges[use.edge].curve) for use in face.loop]
        if curves != [Spline, Straight, Spline, Straight]:
            return None
        kind, data = _coons_snapshot(model, face, surface)
        if kind != COONS_POLYNOMIAL:
            return None
        proof = _compile(kind, data, lambda: None)
        support = _tensor(_bezier(tuple(tuple(map(_stored, row)) for row in proof.controls)),
                          proof.direction)
    else:
        return None
    _require_tensor_runtime(model, face)
    if model._transaction_journal is not None:
        raise GeometryError('source tensor chart requires a committed source')
    edges = {}
    for side, use in enumerate(face.loop):
        edge = model.edges[use.edge]
        curve = _edge_curve(model, edge)
        if curve is None:
            return None
        if (curve.exact_point(0) != _point(model.vertices[edge.start].position) or
                curve.exact_point(1) != _point(model.vertices[edge.end].position)):
            raise GeometryError('source tensor rim does not bind its topology vertices')
        pcurve = _matches_edge(support, side, curve, use.forward)
        edges[edge.id] = replace(edge, curve=CurveOnSurface(support, pcurve))
    return _TensorView(model, face, support, edges)
