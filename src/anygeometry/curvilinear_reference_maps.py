"""Candidate local discrepancy tools; no material coverage authority.

These local maps alone do not establish a mesh's global material coverage.
Their certificates must be combined with true symbolic source vertices,
authenticated boundary interval/true-endpoint transport and oriented cell
incidence/degree/collar proofs before any public receipt is issued. The moved
boundary helper below bounds a target discrepancy only; it must never stand
in for the original source boundary in a material partition proof.
"""
from dataclasses import dataclass
from fractions import Fraction as Q

from .errors import GeometryError
from .nurbs import CurveOnSurface, _float_down, _float_up
from .support_curves import PlaneNurbsCurve
from .plane_nurbs import charge_box, charge_split, depth_limit
from .nurbs_metric import add, sub, mul, scale, sum_intervals, derivative_intervals


def constant(value):
    return value, value


def _norm_upper(magnitudes):
    """A rational upper bound of the Euclidean norm of nonnegative exact ``magnitudes``.

    Two independent exact upper bounds are available: the coordinate sum (``|v|_2 <= |v|_1``) and the
    rounded-up root ``sqrt(x) <= (isqrt(ceil(x * 2**128)) + 1) / 2**64``. The smaller is returned, so the
    result is never below the true norm, never above the former coordinate sum, and an exact zero vector
    stays exactly zero. All arithmetic is exact (no float).
    """
    from math import ceil, isqrt
    values = tuple(Q(v) for v in magnitudes)
    total = sum((v*v for v in values), Q(0))
    return min(sum(values, Q(0)), Q(isqrt(ceil(total*2**128))+1, 2**64))


def interval(value, lo, hi):
    return add(constant(lo), scale(value, hi-lo))


def native_derivatives(curve, lo, hi):
    """Exact chart derivative bounds in the edge's normalized parameter."""
    if isinstance(curve, CurveOnSurface):
        p = curve.pcurve
        if p.degree != 1 or len(p.controls) != 2:
            raise GeometryError('source-cell pcurve derivative family is not yet qualified')
        w0, w1 = (Q(1), Q(1)) if p.weights is None else map(Q, p.weights)
        a, b = map(Q, p.domain)
        k0, k1 = map(Q, (p.knots[1], p.knots[2]))
        r = interval((lo, hi), (a-k0)/(k1-k0), (b-k0)/(k1-k0))
        weights = sorted(w0+(w1-w0)*t for t in r)
        if weights[0] <= 0:
            raise GeometryError('source-cell chart denominator is not positive')
        factor = scale((1/weights[1]**2, 1/weights[0]**2), w0*w1*(b-a)/(k1-k0))
        return tuple(scale(factor, Q(y)-Q(x)) for x, y in zip(p.controls[0], p.controls[1]))
    if not isinstance(curve, PlaneNurbsCurve):
        raise GeometryError('source-cell curve derivative family is not yet qualified')
    from .branch_predicates import _Arc
    values = []
    lower, upper = max(0., _float_down(lo)), min(1., _float_up(hi))
    count = len(curve._arcs())
    for arc, s, x in curve._arc_rects(lower, upper):
        for root in (arc.first, arc.second):
            rs, _rx = curve._ordered_root_rect(arc, root)
            if rs[0] < rs[1] and not (s[1] < rs[0] or s[0] > rs[1]):
                # P2 clamps its nominal slicing gap to the true end root.
                # The implicit graph derivative is not the derivative of
                # that constant interval. A regular-image reparameterization
                # with true algebraic endpoint provenance is required here.
                raise GeometryError('source-cell true arc-end reparameterization is not yet certified')
        view = object.__new__(_Arc)
        view.slicing, view.u_box, view.v_box = arc.slicing, arc.box.u_range, arc.box.v_range
        du, dv = view.derivative_ranges(arc.box.coefficients, s, x)
        gs, gx = (du, dv) if arc.slicing == 0 else (dv, du)
        if gx[0] <= 0 <= gx[1]:
            raise GeometryError('source-cell section derivative denominator is not regular')
        ds = (Q(curve.domain[1])-Q(curve.domain[0]))*count*(arc.s1-arc.s0)
        dx = scale(mul(gs, tuple(sorted((1/gx[0], 1/gx[1])))), -ds)
        physical = (constant(ds), dx) if arc.slicing == 0 else (dx, constant(ds))
        values.append(tuple(scale(d, 1/(Q(r[1])-Q(r[0]))) for d, r in
                            zip(physical, (curve.surface.u_range, curve.surface.v_range))))
    if not values:
        raise GeometryError('source-cell section interval is empty')
    return tuple((min(row[i][0] for row in values), max(row[i][1] for row in values)) for i in (0, 1))


@dataclass(frozen=True)
class BoundaryMap:
    support: object
    curve: object
    parameters: tuple
    targets: tuple
    endpoint_boxes: tuple
    corrections: tuple

    def box(self, lo, hi):
        from .prepared_material_curvilinear import _uv_interval
        a, b = self.parameters
        t0, t1 = a+(b-a)*lo, a+(b-a)*hi
        source = _uv_interval(self.support, self.curve, min(t0, t1), max(t0, t1))
        return tuple(add(coordinate, add(scale(d0, 1-hi), scale(d1, lo)))
                     if lo == hi else add(coordinate, (min(d0[0], d1[0]), max(d0[1], d1[1])))
                     for coordinate, d0, d1 in zip(source, self.corrections[0], self.corrections[1]))

    def derivative(self, lo, hi):
        a, b = self.parameters
        first, last = a+(b-a)*lo, a+(b-a)*hi
        source = native_derivatives(self.curve, min(first, last), max(first, last))
        return tuple(add(scale(d, b-a), sub(d1, d0))
                     for d, d0, d1 in zip(source, self.corrections[0], self.corrections[1]))


def boundary_map(support, curve, parameters, targets):
    from .prepared_material_curvilinear import _uv_box
    endpoints = tuple(_uv_box(support, curve, t) for t in parameters)
    correction = tuple(tuple(sub(constant(Q(x)), bound) for x, bound in zip(target, point))
                       for target, point in zip(targets, endpoints))
    return BoundaryMap(support, curve, parameters, targets, endpoints, correction)


def _cross(a, b):
    return sub(mul(a[0], b[1]), mul(a[1], b[0]))


def _source_box(support, box):
    from .surfaces import Plane
    if isinstance(support, Plane):
        from .planar_exact import ExactPlane
        from itertools import product
        frame = ExactPlane.from_plane(support)
        points = [frame.world(*p) for p in product(*box)]
        return tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in range(3))
    from .surface_capabilities import capabilities
    if any(a < 0 or b > 1 for a, b in box):
        raise GeometryError('source-curved map exceeds its declared support chart')
    enclosed = capabilities(support).enclosure(tuple((_float_down(a), _float_up(b)) for a, b in box))
    return tuple((Q(float(a)), Q(float(b))) for a, b in zip(enclosed.lower, enclosed.upper))


def _quad_fan(xyz, side, s, t):
    """Literal bilinear4-node FE map and its derivatives on a centre fan."""
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    first, last = corners[side], corners[(side+1)%4]
    edge = tuple(interval(t, Q(a), Q(b)) for a, b in zip(first, last))
    reference = tuple(mul((1-s[1], 1-s[0]), x) for x in edge)
    ds = tuple(scale(x, -1) for x in edge)
    dt = tuple(scale((1-s[1], 1-s[0]), Q(b-a)) for a, b in zip(first, last))
    output, derivative_s, derivative_t = [], [], []
    for values in zip(*xyz):
        a, b, c, d = map(Q, values)
        c0, cx, cy, cxy = (a+b+c+d)/4, (-a+b+c-d)/4, (-a-b+c+d)/4, (a-b+c-d)/4
        x, y = reference
        output.append(add(add(constant(c0), scale(x, cx)), add(scale(y, cy), scale(mul(x, y), cxy))))
        fx, fy = add(constant(cx), scale(y, cxy)), add(constant(cy), scale(x, cxy))
        derivative_s.append(add(mul(fx, ds[0]), mul(fy, ds[1])))
        derivative_t.append(add(mul(fx, dt[0]), mul(fy, dt[1])))
    return tuple(output), tuple(derivative_s), tuple(derivative_t)


def _source_quad_fan(uv_boxes, boundary, side, s, t, *, corner_hull=False):
    """True bilinear corner map plus its true curved-side correction.

    The t derivative is returned AFTER removing the known (1-s) apex
    factor. Shared true endpoint symbols make correction endpoints exactly
    zero, so adjacent proof fans have identical bilinear radial maps.
    """
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    first, last = corners[side], corners[(side+1)%4]
    edge = tuple(interval(t, Q(a), Q(b)) for a, b in zip(first, last))
    one_minus_s = (1-s[1], 1-s[0])
    reference = tuple(mul(one_minus_s, x) for x in edge)
    derivative_s = tuple(scale(x, -1) for x in edge)
    edge_t = tuple(Q(b-a) for a, b in zip(first, last))
    base, base_s, base_reduced_t = [], [], []
    for a, b, c, d in zip(*uv_boxes):
        c0 = scale(add(add(a, b), add(c, d)), Q(1, 4))
        cx = scale(add(sub(b, a), sub(c, d)), Q(1, 4))
        cy = scale(add(sub(c, b), sub(d, a)), Q(1, 4))
        cxy = scale(sub(add(a, c), add(b, d)), Q(1, 4))
        x, y = reference
        base.append(add(add(c0, mul(cx, x)), add(mul(cy, y), mul(cxy, mul(x, y)))))
        fx, fy = add(cx, mul(cxy, y)), add(cy, mul(cxy, x))
        base_s.append(add(mul(fx, derivative_s[0]), mul(fy, derivative_s[1])))
        base_reduced_t.append(add(scale(fx, edge_t[0]), scale(fy, edge_t[1])))
    # Bilinear shape weights are nonnegative on this reference square.
    # Intersect the algebraic enclosure with its exact corner convex hull.
    base = [((max(value[0], min(p[axis][0] for p in uv_boxes))),
             (min(value[1], max(p[axis][1] for p in uv_boxes)))) for axis, value in enumerate(base)]
    if boundary.curve is None:
        correction = derivative_correction = (constant(Q(0)),)*2
    else:
        from .regular_source_reference import SourceSide
        line = SourceSide(boundary.start, boundary.end)
        correction = tuple(sub(q, l) for q, l in zip(boundary.box(*t), line.box(*t)))
        if t[0] == t[1] and t[0] in (0, 1):
            correction = (constant(Q(0)),)*2
        derivative_correction = tuple(sub(q, l) for q, l in zip(boundary.derivative(*t), line.derivative(*t)))
    r = tuple(add(b, mul(one_minus_s, e)) for b, e in zip(base, correction))
    if corner_hull:
        # The SAME endpoint symbols make L(t) the quad's own side. Writing
        # u=s/2+(1-s)t, the remaining four corner weights (cyclic from side)
        # are s²/4+s(1-s)t/2, s²/4+s(1-s)(1-t)/2,
        # (s/2)u, (s/2)(1-u). They are nonnegative and sum to s;
        # Q(t) has weight 1-s. Intersect with that true convex hull, never
        # with an arbitrary global chart clamp. The map/derivatives stay fixed.
        q = boundary.box(*t)
        hull = tuple((min(q[axis][0],*(p[axis][0] for p in uv_boxes)),
                      max(q[axis][1],*(p[axis][1] for p in uv_boxes))) for axis in (0,1))
        r = tuple((max(a,c),min(b,d)) for (a,b),(c,d) in zip(r,hull))
    rs = tuple(sub(b, e) for b, e in zip(base_s, correction))
    reduced_t = tuple(add(b, e) for b, e in zip(base_reduced_t, derivative_correction))
    return r, rs, reduced_t


class _SideMemo:
    """Invocation-local memo of one boundary side's exact enclosures inside ONE fan proof.

    ``SourceSide.box`` and ``SourceSide.derivative`` each ran the full ``_ranges(lo, hi)`` (true-root
    rectangles, derivative ranges, chart rectangles) and the proof loop asks both for the same parameter
    interval at every ``(s, t)`` box that shares it, plus again for the midpoint. Here ``_ranges`` runs ONCE
    per exact interval and serves both values; every value is exactly what the side itself returns for that
    interval (deterministic function of immutable exact operands). The memo is created per
    ``certify_fan_subtriangle`` call, never stored, shared or consulted for identity, and every examined
    box is still charged by the caller; exceptions are not cached. Other attributes delegate to the side.

    Eligibility is the EXACT owner ``SourceSide`` type only (immutable exact operands, deterministic
    ``_ranges``). Subclasses, ``BoundaryMap`` and any custom boundary are never wrapped: every query is
    delegated, so callbacks, refusals and Work behaviour of such objects can never be hidden by a cache.
    """
    def __init__(self, side):
        from .regular_source_reference import SourceSide
        if type(side) is not SourceSide:
            raise TypeError('only an exact SourceSide may be memoized')
        self._side = side
        self._both = {}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._side, name)

    def _ranges(self, lo, hi):
        key = (lo, hi)
        if key not in self._both:
            self._both[key] = self._side._ranges(lo, hi)
        return self._both[key]

    def box(self, lo, hi):
        if lo == hi == 0:                           # SourceSide.box's exact endpoint cases
            return self._side.start.box
        if lo == hi == 1:
            return self._side.end.box
        return self._ranges(lo, hi)[0]

    def derivative(self, lo, hi):
        return self._ranges(lo, hi)[1]


def _polynomial_fast_path(support, boundary, centre_uv, xyz_side, centre_xyz, orientation, quad_xyz, quad_uv,
                          quad_side=None):
    """Exact whole-cell residual bound for a total-degree<=3 support and an uncurved straight side, or
    ``None`` (use the general interval proof). Sufficient proof only: a bound above the requested error
    also falls back, so a genuine violation is still decided by the unchanged general proof."""
    from .regular_source_reference import SourceSide
    from .polynomial_cell_bounds import triangle_residual_bound, quad_residual_bound
    if type(boundary) is not SourceSide or boundary.curve is not None or orientation not in (1, -1):
        return None
    # The true corner/vertex charts may be certified intervals; the bounds enclose EVERY choice of true
    # points in the boxes, require strict orientation for all of them and keep the chart domain contained.
    if quad_uv is None:
        if quad_xyz is not None:
            return None
        return triangle_residual_bound(support, (boundary.start.box, boundary.end.box, centre_uv),
                                       (*xyz_side, centre_xyz), orientation)
    if quad_xyz is None or len(quad_uv) != 4 or quad_side is None:
        return None
    # the fan side must be the quad's own side: same certified boxes, not merely nearby ones
    if boundary.start.box != quad_uv[quad_side] or boundary.end.box != quad_uv[(quad_side+1) % 4]:
        return None
    return quad_residual_bound(support, quad_uv, quad_xyz, orientation)


def certify_fan_subtriangle(support, boundary, centre_uv, xyz_side, centre_xyz, *,
                           orientation, surface_error, quad_xyz=None, quad_side=None, quad_uv=None,
                           cancellation_check=None, quad_nodes=None):
    """Bound a ruled source-UV map against its literal affine FE subtriangle.

    The expected collapsed-apex factor(1-s) is removed from the Jacobian.
    The reduced determinant must remain strictly oriented on each certified
    t interval; source support immersion is a separate owner obligation.
    """
    # explicit cancellation is honoured at ENTRY, before any exact early exit or fast path; a callback
    # exception propagates unchanged
    if cancellation_check is not None and cancellation_check():
        raise GeometryError('source-curved reference map proof cancelled')
    centre_uv, centre_xyz = tuple(map(Q, centre_uv)), tuple(map(Q, centre_xyz))
    xyz_side = tuple(tuple(map(Q, p)) for p in xyz_side)
    from .regular_source_reference import SourceSide
    from .regular_source_reference import SourceNode
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    correlated_quad = False
    if quad_nodes is not None:
        from .prepared_material_curvilinear import _literal
        if (type(quad_nodes) is not tuple or len(quad_nodes) != 4 or type(quad_side) is not int
                or not 0 <= quad_side < 4 or quad_uv is None or len(quad_uv) != 4
                or any(type(node) is not SourceNode for node in quad_nodes)
                or _literal(tuple(node.box for node in quad_nodes)) != _literal(quad_uv)):
            raise GeometryError('source quad canonical corner correspondence is malformed')
        if type(boundary) is SourceSide:
            corners = quad_nodes[quad_side],quad_nodes[(quad_side+1)%4]
            if any(type(node) is not SourceNode for node in (boundary.start,boundary.end)) or any(
                    _literal(node.identity) != _literal(corner.identity) or _literal(node.box) != _literal(corner.box)
                    for node,corner in zip((boundary.start,boundary.end),corners)):
                raise GeometryError('source quad side does not have its canonical corner identities and boxes')
            correlated_quad = True
    if (isinstance(support, Plane) and type(boundary) is SourceSide and boundary.curve is None
            and all(a == b for node in (boundary.start, boundary.end) for a, b in node.box)):
        from .planar_exact import ExactPlane
        frame = ExactPlane.from_plane(support)
        first, last = [tuple(a for a, b in node.box) for node in (boundary.start, boundary.end)]
        if quad_uv is not None and all(a == b for point in quad_uv for a, b in point):
            points = [tuple(a for a, b in point) for point in quad_uv]
            targets_equal = all(frame.world(*p) == tuple(map(Q, xyz)) for p, xyz in zip(points, quad_xyz))
            turns = [(points[(k+1)%4][0]-points[k][0])*(points[(k+2)%4][1]-points[(k+1)%4][1])-
                     (points[(k+1)%4][1]-points[k][1])*(points[(k+2)%4][0]-points[(k+1)%4][0]) for k in range(4)]
            if targets_equal and all(orientation*turn > 0 for turn in turns):
                # Affine Plane evaluation commutes EXACTLY with the literal
                # bilinear FE interpolation. Convex turns certify its Jacobian.
                charge_box()
                return Q(0), 1
        wedge = (last[0]-first[0])*(centre_uv[1]-first[1])-(last[1]-first[1])*(centre_uv[0]-first[0])
        if (quad_xyz is None and orientation*wedge > 0 and frame.world(*first) == xyz_side[0]
                and frame.world(*last) == xyz_side[1] and frame.world(*centre_uv) == centre_xyz):
            charge_box()
            return Q(0), 1
    weights = getattr(support, 'weights', None)
    uniform = weights is None or len({Q(w) for row in weights for w in row}) == 1
    if (isinstance(support, NurbsSurface) and type(boundary) is SourceSide and boundary.curve is None and support.degree_u == support.degree_v == 1
            and len(support.controls) == len(support.controls[0]) == 2 and uniform
            and all(a == b for node in (boundary.start, boundary.end) for a, b in node.box)):
        corners = [support.exact_point(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
        affine = all(a-b+c-d == 0 for a, b, c, d in zip(*corners))
        first, last = [tuple(a for a, b in node.box) for node in (boundary.start, boundary.end)]
        wedge = (last[0]-first[0])*(centre_uv[1]-first[1])-(last[1]-first[1])*(centre_uv[0]-first[0])
        exact_targets = (tuple(support.exact_point(*first)) == xyz_side[0]
                         and tuple(support.exact_point(*last)) == xyz_side[1]
                         and tuple(support.exact_point(*centre_uv)) == centre_xyz)
        quad_affine = quad_xyz is None or all(Q(a)-Q(b)+Q(c)-Q(d) == 0 for a, b, c, d in zip(*quad_xyz))
        if affine and quad_uv is not None and all(a == b for point in quad_uv for a, b in point):
            points = [tuple(a for a, b in point) for point in quad_uv]
            exact_quad = all(tuple(support.exact_point(*p)) == tuple(map(Q, xyz)) for p, xyz in zip(points, quad_xyz))
            # A strict convex quadrilateral has an oriented bilinear Jacobian
            # at all four corners. That determinant is affine in reference
            # coordinates, hence strictly oriented everywhere on the square.
            turns = [(points[(k+1)%4][0]-points[k][0])*(points[(k+2)%4][1]-points[(k+1)%4][1])-
                     (points[(k+1)%4][1]-points[k][1])*(points[(k+2)%4][0]-points[(k+1)%4][0]) for k in range(4)]
            if exact_quad and all(orientation*turn > 0 for turn in turns):
                charge_box()
                return Q(0), 1
        if affine and exact_targets and quad_affine and orientation*wedge > 0:
            charge_box()
            return Q(0), 1
    fast = _polynomial_fast_path(support, boundary, centre_uv, xyz_side, centre_xyz, orientation,
                                 quad_xyz, quad_uv, quad_side)
    if fast is not None and fast <= surface_error:
        charge_box()
        return fast, 1
    if quad_uv is None and quad_xyz is None:
        from ._structural_fan_bounds import structural_fan_bound
        structural = structural_fan_bound(support, boundary, centre_uv, xyz_side, centre_xyz,
            orientation=orientation, surface_error=surface_error, cancellation_check=cancellation_check)
        if structural is not None:
            return structural
    from .regular_source_reference import SourceSide as _ExactSide
    if type(boundary) is _ExactSide:
        boundary = _SideMemo(boundary)      # proof-local: one exact enclosure per parameter interval
    pending, maximum, examined = [((Q(0), Q(1)), (Q(0), Q(1)), 0)], Q(0), 0
    while pending:
        if cancellation_check is not None and cancellation_check():
            raise GeometryError('source-curved reference map proof cancelled')
        s, t, depth = pending.pop()
        charge_box()
        examined += 1
        if examined > 32768:
            raise GeometryError('source-curved cell proof box budget exhausted')
        q, dq = boundary.box(*t), boundary.derivative(*t)
        radial = tuple(sub(constant(c), x) for c, x in zip(centre_uv, q))
        wedge = scale(_cross(dq, radial), Q(orientation))
        r = tuple((min(x[0], c), max(x[1], c)) for x, c in zip(q, centre_uv))
        # Keep the convex combinations as interval expressions on the actual
        # proof subpatch, avoiding the full triangle at every refinement.
        r = tuple(add(mul((1-s[1], 1-s[0]), x), scale(s, c)) for x, c in zip(q, centre_uv))
        r = tuple((max(a, min(qi[0], c)), min(b, max(qi[1], c))) for (a, b), qi, c in zip(r, q, centre_uv))
        if any(a > b for a, b in r):
            raise GeometryError('source-curved reference interval is empty')
        rs, reduced_t = radial, dq
        if quad_uv is not None:
            r, rs, reduced_t = _source_quad_fan(quad_uv, boundary, quad_side, s, t, corner_hull=correlated_quad)
            wedge = scale(_cross(reduced_t, rs), Q(orientation))
        if wedge[0] > 0 and (isinstance(support, Plane) or all(0 <= a <= b <= 1 for a, b in r)):
            du, dv = derivative_intervals(support, r)
            rt = tuple(mul((1-s[1], 1-s[0]), d) for d in reduced_t)
            world_s = tuple(add(mul(du[i], rs[0]), mul(dv[i], rs[1])) for i in range(3))
            world_t = tuple(add(mul(du[i], rt[0]), mul(dv[i], rt[1])) for i in range(3))
            side = tuple(interval(t, a, b) for a, b in zip(*xyz_side))
            fe_s = tuple(sub(constant(c), x) for c, x in zip(centre_xyz, side))
            fe_t = tuple(scale((1-s[1], 1-s[0]), b-a) for a, b in zip(*xyz_side))
            if quad_xyz is not None:
                _fe, fe_s, fe_t = _quad_fan(quad_xyz, quad_side, s, t)
            sm, tm = sum(s)/2, sum(t)/2
            qm = boundary.box(tm, tm)
            rm = tuple(add(scale(x, 1-sm), constant(sm*c)) for x, c in zip(qm, centre_uv))
            if quad_uv is not None:
                rm = _source_quad_fan(quad_uv, boundary, quad_side, (sm, sm), (tm, tm), corner_hull=correlated_quad)[0]
            source_mid = _source_box(support, rm)
            fe_mid = tuple((1-sm)*(a+(b-a)*tm)+sm*c for a, b, c in zip(*xyz_side, centre_xyz))
            if quad_xyz is not None:
                fe_mid = tuple(a for a, b in _quad_fan(quad_xyz, quad_side, (sm, sm), (tm, tm))[0])
            # Euclidean mean-value bound: |G(p)| <= |G(mid)| + sup|G_s| |ds| + sup|G_t| |dt| for
            # G = source - FE over the half widths; every vector norm is a rounded-up Euclidean norm of
            # the per-coordinate maximum magnitudes (never above the former coordinate sum).
            discrepancy = _norm_upper(max(abs(a-c), abs(b-c)) for (a, b), c in zip(source_mid, fe_mid))
            discrepancy += _norm_upper(max(abs(a), abs(b)) for a, b in
                (sub(x, y) for x, y in zip(world_s, fe_s)))*(s[1]-s[0])/2
            discrepancy += _norm_upper(max(abs(a), abs(b)) for a, b in
                (sub(x, y) for x, y in zip(world_t, fe_t)))*(t[1]-t[0])/2
            if discrepancy <= surface_error:
                maximum = max(maximum, discrepancy)
                continue
        if depth >= depth_limit(12):
            raise GeometryError('source-curved cell Jacobian or approximation is not certified')
        charge_split()
        axis = 0 if s[1]-s[0] >= t[1]-t[0] else 1
        ranges = (s, t)
        middle = sum(ranges[axis])/2
        for part in ((ranges[axis][0], middle), (middle, ranges[axis][1])):
            child = list(ranges)
            child[axis] = part
            pending.append((*child, depth+1))
    return maximum, examined
