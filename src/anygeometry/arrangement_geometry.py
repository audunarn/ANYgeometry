"""Immutable curve paths and analytic predicates for material arrangements.

This module has no mutable topology. It is shared by planning and surface-chart
queries; discretization samples are never used for trim or junction predicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np

from ._cross3 import _cross3
from .analytic_roots import isolate_real_roots, trigonometric_roots
from .curves import Arc, Straight, Spline
from .errors import GeometryError
from .exact_curves import EXACT_CURVES, EllipticArc, CylinderIntersectionCurve
from .branch_algebra import directrix_of
from .plane_ruled import PlaneRuledSupport
from .branch_curves import BezierQuadricCurve
from .quadric_curves import QuadricIntersectionCurve
from .surfaces import Cylinder, Plane
from contextvars import ContextVar
from functools import lru_cache
from contextlib import contextmanager

# Retained for controlled experiments; full-case benefit is not established.
_JUNCTION_BOUNDS = ContextVar('junction_bounds', default=False)


# Piecewise line/branch rejection (default on). A predicate budget or an explicit
# ``_junction_bounds_policy(False)`` restores the unfiltered route for exact accounting.
_SWEEP_REJECTION = ContextVar('junction_sweep_rejection', default=True)


@contextmanager
def _junction_bounds_policy(enabled):
    token = _JUNCTION_BOUNDS.set(enabled)
    sweep = _SWEEP_REJECTION.set(bool(enabled))
    try:
        yield
    finally:
        _SWEEP_REJECTION.reset(sweep)
        _JUNCTION_BOUNDS.reset(token)


_SWEEP_EPS = float(np.finfo(float).eps)
_SWEEP_PIECES = 8


def _segment_clear_of_box(p, q, lo, hi):
    """``True`` only when the segment ``p``-``q`` (3 floats each) and the box ``lo``-``hi`` are disjoint
    (separating-axis test: the three box axes, then the three ``d x e_k`` axes)."""
    for k in range(3):
        a, b = p[k], q[k]
        if a > b:
            a, b = b, a
        if b < lo[k] or a > hi[k]:
            return True
    d = (q[0]-p[0], q[1]-p[1], q[2]-p[2])
    gap = [.5*(lo[k]+hi[k])-.5*(p[k]+q[k]) for k in range(3)]
    half = [.5*(hi[k]-lo[k]) for k in range(3)]
    for axis in ((0., d[2], -d[1]), (-d[2], 0., d[0]), (d[1], -d[0], 0.)):
        if abs(axis[0]*gap[0]+axis[1]*gap[1]+axis[2]*gap[2]) > (
                abs(axis[0])*half[0]+abs(axis[1])*half[1]+abs(axis[2])*half[2]):
            return True
    return False


_IDENTITY_ROWS = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))


@lru_cache(maxsize=16384)
def _curve_setup_benign(curve):
    """``True`` when the curve-only set-up of the original route (directrix, plan, end evaluations) does not refuse.

    The original ``curve_junctions`` inverts the end points of each curve on the other; that builds the
    directrix and the plan of a branch curve, which can refuse (for example a directrix that degenerates to a
    line). A shortcut must not turn such a refusal into ``()``, so it declines when the set-up raises.
    """
    try:
        if type(curve) is BezierQuadricCurve and not isinstance(curve.first, PlaneRuledSupport):
            directrix_of(curve.first)
        if type(curve) in (BezierQuadricCurve, CylinderIntersectionCurve):
            curve.evaluate(0.)
            curve.evaluate(1.)
        return True
    except Exception:
        return False


def _shortcut_curve_ok(curve):
    """``False`` for a curve the rejection shortcuts must not decide; they then decline to the original route.

    The single guard of every shortcut path (``_separated_line_pair``, ``_line_clear_of_branch``,
    ``_curve_pair_clear``). Contract: with the shortcuts on, ``curve_junctions`` returns the same values and
    raises the same typed refusals as the original route. Cancellation poll counts beyond the replay in
    ``_poll_like_original`` and exceptions under ``np.errstate(...='raise')`` are not part of the contract
    (production never sets errstate to raise).

    A branch curve with a non-identity transform measures its endpoint inversion tolerance in reference
    space (a scale or shear widens the world-space window beyond the padding); an elliptic arc with a
    huge angular phase loses the angle arithmetic (``start + sweep == start``) the enclosure assumes.
    Both keep the original route.
    """
    kind = type(curve)
    if kind is BezierQuadricCurve:
        return bool(curve._identity) and _curve_setup_benign(curve)
    if kind is CylinderIntersectionCurve:
        if not (tuple(tuple(float(v) for v in row) for row in curve.transform) == _IDENTITY_ROWS
                and _curve_setup_benign(curve)):
            return False
        try:
            return _angle_error_ok(float(curve.start_angle), float(curve.sweep_angle))
        except Exception:
            return False
    if kind is EllipticArc:
        start, sweep = float(curve.start_angle), float(curve.sweep_angle)
        return _angle_error_ok(start, sweep)
    return True


def _angle_error_ok(start, sweep):
    """The rounding of the angular phase stays below half of the rounding allowance of the enclosures.

    The evaluated angle ``start + sweep * value`` and the angles the bounds are built from differ by a few
    ulp of the largest angle (taken as 8). The enclosures pad their box by ``1e3 * eps`` times the
    coordinate scale, and the angle error moves a point by at most that error times a lever arm no larger
    than the coordinate scale, so ``8 ulp(max |angle|) <= 500 eps`` (a largest angle of about 60 rad) keeps the
    error inside half of the allowance for every radius, axis length and axial range. A phase so large that
    ``start + sweep == start`` is always declined.
    """
    top = max(abs(start), abs(start+sweep), 1.)
    return (math.isfinite(top) and start+sweep != start
            and 8*float(np.spacing(top)) <= 500*float(np.finfo(float).eps))


def _poll_like_original(cancellation_check):
    """The two cancellation polls the original route makes for a pair that returns ``()``.

    ``curve_junctions`` polls on entry (already done), before the ``t = 1`` endpoint candidates and on
    entry to ``_curve_junctions``; a pair certainly apart reaches no other poll.
    """
    if cancellation_check is not None:
        for _ in range(2):
            if cancellation_check():
                raise GeometryError('curve intersection predicate cancelled')


def _coplanar_with_ellipse(ellipse, curve, tolerance):
    """``True`` when the original route could refuse or take its coplanar branch for this ellipse/branch pair."""
    try:
        normal = _cross3(ellipse.u_vector, ellipse.v_vector)
        normal /= np.linalg.norm(normal)
        return plane_roots(curve, normal, float(normal @ ellipse.center), tolerance=tolerance) is None
    except Exception:
        return True


def _line_clear_of_branch(first, second, tolerance):
    """``True`` only when ``curve_junctions(first, second)`` certainly returns ``()``.

    Every junction needs a curve point within ``tolerance`` of the line's segment extended by
    ``tolerance`` in parameter (the window ``point_parameters`` accepts). The outward interval
    enclosures of the curve (whole, then eight parameter pieces) padded by ``tolerance`` and a
    rounding allowance must all be disjoint from that extended segment. Anything unusual
    (other curve types, a degenerate line, non-finite data) keeps the original route.
    """
    if not _SWEEP_REJECTION.get():
        return False
    if type(first) is LinePath and type(second) in (BezierQuadricCurve, CylinderIntersectionCurve):
        line, curve = first, second
    elif type(second) is LinePath and type(first) in (BezierQuadricCurve, CylinderIntersectionCurve):
        line, curve = second, first
    else:
        return False
    if not _shortcut_curve_ok(curve):
        return False
    try:
        start = [float(v) for v in line.start]
        end = [float(v) for v in line.end]
        d = [end[k]-start[k] for k in range(3)]
        size = math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])
        if not (size > tolerance and math.isfinite(size) and 0 < tolerance < 1e100):
            return False
        p = [start[k]-tolerance*d[k] for k in range(3)]
        q = [end[k]+tolerance*d[k] for k in range(3)]
        scale = max(1., *(abs(v) for v in p), *(abs(v) for v in q))
        base = tolerance*(1+1e-9)

        def clear(lower, upper):
            lo, hi = curve.bounds(lower, upper)
            lo, hi = lo.tolist(), hi.tolist()
            top = max(scale, *(abs(v) for v in lo), *(abs(v) for v in hi))
            if not math.isfinite(top):
                return False
            pad = base+1e3*_SWEEP_EPS*top
            return _segment_clear_of_box(p, q, [v-pad for v in lo], [v+pad for v in hi])

        if clear(0., 1.):
            return True
        return all(clear(i/_SWEEP_PIECES, (i+1)/_SWEEP_PIECES) for i in range(_SWEEP_PIECES))
    except (GeometryError, ArithmeticError, ValueError, TypeError):
        return False


def _separated_line_pair(first, second, tolerance):
    """Conservative rejection within the existing line/analytic admission.

    Parameter tolerances for lines and ellipses are deliberately included in
    world bounds. Unsupported or transformed-reference predicates retain the
    original route. This does not certify an intersection or skip clipping.
    """
    if not _JUNCTION_BOUNDS.get():
        return False
    if type(first) is LinePath:
        line, other = first, second
    elif type(second) is LinePath:
        line, other = second, first
    else:
        return False
    if type(other) not in (LinePath, EllipticArc, CylinderIntersectionCurve, BezierQuadricCurve, BezierPath):
        return False
    if type(other) is BezierQuadricCurve and not other._identity:
        return False
    if not _shortcut_curve_ok(other):
        return False
    direction = np.asarray(line.end) - line.start
    if np.linalg.norm(direction) <= tolerance:
        raise GeometryError('degenerate arrangement line')
    if type(other) is LinePath and float(np.dot(np.asarray(other.end)-other.start,
                                               np.asarray(other.end)-other.start)) == 0:
        raise GeometryError('degenerate arrangement line')

    def envelope(curve):
        try:
            lo, hi = curve.bounds()
        except (GeometryError, ArithmeticError, ValueError):
            return None
        lo, hi = np.asarray(lo), np.asarray(hi)
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            return None
        factor = 1.
        if type(curve) is LinePath:
            factor += float(np.linalg.norm(np.asarray(curve.end)-curve.start))
        elif type(curve) is EllipticArc:
            factor = 2. + float(np.linalg.norm(curve.u_vector) + np.linalg.norm(curve.v_vector))
        rounding = 64*np.finfo(float).eps*max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))), 1.)
        pad = tolerance*factor + rounding
        if not np.isfinite(pad):
            return None
        return np.nextafter(lo-pad, -np.inf), np.nextafter(hi+pad, np.inf)
    a, b = envelope(first), envelope(second)
    return (a is not None and b is not None
            and (np.any(a[1] < b[0]) or np.any(b[1] < a[0])))


@lru_cache(maxsize=16384)
def _cached_enclosure(curve, lower, upper):
    """``curve.bounds(lower, upper)`` as float lists (the curves are immutable values)."""
    lo, hi = curve.bounds(lower, upper)
    return lo.tolist(), hi.tolist()


def _box_pair_clear(lo1, hi1, lo2, hi2, pad):
    for k in range(3):
        if lo2[k]-hi1[k] > pad or lo1[k]-hi2[k] > pad:
            return True
    return False


def _curve_pair_clear(first, second, tolerance):
    """``True`` only when ``curve_junctions(first, second)`` certainly returns ``()`` (curve x curve).

    A junction needs a point of each curve within ``tolerance`` times the curve's acceptance
    window of a point of the other (lines: the extended segment; arcs: ``2 + |u| + |v|``,
    the angular window). Outward interval enclosures (whole, then eight pieces) padded by
    those windows and a rounding allowance must be pairwise disjoint. Lines against branch
    curves use ``_line_clear_of_branch``; other types, degenerate or non-finite data keep the
    original route.
    """
    if not _SWEEP_REJECTION.get():
        return False
    kinds = (LinePath, EllipticArc, BezierQuadricCurve, CylinderIntersectionCurve)
    if type(first) not in kinds or type(second) not in kinds or (type(first) is LinePath and type(second) is LinePath):
        return False
    if type(first) is LinePath and type(second) in (BezierQuadricCurve, CylinderIntersectionCurve):
        return False
    if type(second) is LinePath and type(first) in (BezierQuadricCurve, CylinderIntersectionCurve):
        return False
    if not (_shortcut_curve_ok(first) and _shortcut_curve_ok(second)):
        return False
    for ellipse, other in ((first, second), (second, first)):
        if type(ellipse) is EllipticArc and type(other) in (BezierQuadricCurve, CylinderIntersectionCurve):
            if _coplanar_with_ellipse(ellipse, other, tolerance):
                return False
    try:
        window = 0.
        objects = []
        for curve in (first, second):
            if type(curve) is LinePath:
                start = [float(v) for v in curve.start]
                end = [float(v) for v in curve.end]
                d = [end[k]-start[k] for k in range(3)]
                size = math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])
                if not (size > tolerance and math.isfinite(size)):
                    return False
                objects.append((curve, (start, end, d)))
                window += 1.
            else:
                if type(curve) is EllipticArc:
                    u = [float(v) for v in curve.u_vector]
                    v = [float(x) for x in curve.v_vector]
                    nu, nv = math.sqrt(sum(x*x for x in u)), math.sqrt(sum(x*x for x in v))
                    cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                    if not (math.isfinite(nu+nv) and nu > 0 and nv > 0
                            and math.sqrt(sum(x*x for x in cross)) > 1e-6*nu*nv
                            and math.isfinite(float(curve.sweep_angle)) and float(curve.sweep_angle) != 0.):
                        return False
                    window += 2.+nu+nv
                else:
                    window += 1.
                objects.append((curve, None))
        if not (0 < tolerance < 1e100) or not math.isfinite(window):
            return False
        base = tolerance*(1+1e-9)*window

        def enclosure(item, lower, upper):
            curve, line = item
            if line is not None:
                start, end, d = line
                return ('s', [start[k]-tolerance*d[k] for k in range(3)], [end[k]+tolerance*d[k] for k in range(3)])
            lo, hi = _cached_enclosure(curve, lower, upper)
            return ('b', lo, hi)

        whole = [enclosure(item, 0., 1.) for item in objects]
        top = max(1., *(abs(x) for obj in whole for part in obj[1:] for x in part))
        if not math.isfinite(top):
            return False
        pad = base+1e3*_SWEEP_EPS*top

        def apart(x, y):
            if x[0] == 'b' and y[0] == 'b':
                return _box_pair_clear(x[1], x[2], y[1], y[2], pad)
            seg, box = (x, y) if x[0] == 's' else (y, x)
            return _segment_clear_of_box(seg[1], seg[2], [v-pad for v in box[1]], [v+pad for v in box[2]])

        if apart(whole[0], whole[1]):
            return True
        n = _SWEEP_PIECES

        def pieces(index):
            item = objects[index]
            if item[1] is not None:
                return [whole[index]]
            return [enclosure(item, i/n, (i+1)/n) for i in range(n)]

        second_pieces = pieces(1)
        for piece in pieces(0):
            if apart(piece, whole[1]):
                continue
            if not all(apart(piece, other) for other in second_pieces):
                return False
        return True
    except (GeometryError, ArithmeticError, ValueError, TypeError):
        return False


@dataclass(frozen=True, slots=True)
class LinePath:
    start: tuple
    end: tuple

    def evaluate(self, parameters):
        # Scalar fast path: per coordinate start + t*(end-start), the same
        # float operations in the same order as the vectorized route below.
        # Only exact float and np.float64 scalars and coordinates are
        # eligible (production freeze_edge stores exact np.float64 values);
        # float subclasses and non-scalar parameters fall back to it. A
        # finite result certifies finite intermediates for overflow and
        # invalid operations, so any exceptional value (overflow, inf, nan)
        # also falls back and keeps the vectorized route's behavior,
        # including its np.errstate exceptions. Underflow yields finite
        # results, so the fast path is disabled entirely while NumPy
        # underflow handling is enabled.
        if type(parameters) is float or type(parameters) is np.float64:
            start, end = self.start, self.end
            if (type(start) is tuple and type(end) is tuple
                    and len(start) == 3 and len(end) == 3
                    and all(type(value) is float or type(value) is np.float64
                            for value in start+end)
                    and np.geterr()['under'] == 'ignore'):
                t = float(parameters)
                s0, s1, s2 = map(float, start)
                e0, e1, e2 = map(float, end)
                x0 = s0+t*(e0-s0)
                x1 = s1+t*(e1-s1)
                x2 = s2+t*(e2-s2)
                if x0-x0 == 0. and x1-x1 == 0. and x2-x2 == 0.:
                    return np.array((x0, x1, x2))
        t = np.asarray(parameters, dtype=float)
        return np.asarray(self.start)+t[..., None]*(np.asarray(self.end)-self.start)

    def derivative(self, parameters):
        return np.broadcast_to(np.asarray(self.end)-self.start, np.shape(parameters)+(3,)).copy()

    def subcurve(self, lower, upper):
        return LinePath(tuple(self.evaluate(lower)), tuple(self.evaluate(upper)))

    def bounds(self, lower=0., upper=1.):
        points = self.evaluate((lower, upper))
        return np.nextafter(points.min(axis=0), -np.inf), np.nextafter(points.max(axis=0), np.inf)


@dataclass(frozen=True, slots=True)
class BezierPath:
    controls: tuple

    def evaluate(self, parameters):
        t = np.asarray(parameters, dtype=float)
        work = np.broadcast_to(self.controls, t.shape+np.shape(self.controls)).copy()
        for size in range(len(self.controls)-1, 0, -1):
            work[..., :size, :] = ((1-t[..., None, None])*work[..., :size, :]
                                   + t[..., None, None]*work[..., 1:size+1, :])
        return work[..., 0, :].copy()

    def derivative(self, parameters):
        if len(self.controls) == 1:
            return np.zeros(np.shape(parameters)+(3,))
        return BezierPath(tuple(map(tuple, (len(self.controls)-1)*np.diff(self.controls, axis=0)))).evaluate(parameters)

    def _split(self, t):
        work = np.asarray(self.controls).copy()
        origin = work[0].copy()
        work -= origin
        left, right = [work[0].copy()], [work[-1].copy()]
        for size in range(len(work)-1, 0, -1):
            work[:size] = (1-t)*work[:size]+t*work[1:size+1]
            left.append(work[0].copy()); right.append(work[size-1].copy())
        left, right = np.asarray(left)+origin, np.asarray(right[::-1])+origin
        # Translation back need not recover mixed-magnitude outer endpoints.
        # They are exact inherited controls, not newly computed split points.
        left[0], right[-1] = self.controls[0], self.controls[-1]
        return BezierPath(tuple(map(tuple, left))), BezierPath(tuple(map(tuple, right)))

    def subcurve(self, lower, upper):
        if upper < lower:
            return BezierPath(self.subcurve(upper, lower).controls[::-1])
        left = self if upper == 1 else self._split(upper)[0]
        return left if lower == 0 else left._split(lower/upper)[1]

    def bounds(self, lower=0., upper=1.):
        controls = np.asarray(self.subcurve(lower, upper).controls)
        return np.nextafter(controls.min(axis=0), -np.inf), np.nextafter(controls.max(axis=0), np.inf)

    def transformed(self,matrix):
        matrix=np.asarray(matrix,dtype=float)
        points=np.asarray(self.controls) @ matrix[:3,:3].T+matrix[:3,3]
        return BezierPath(tuple(tuple(point) for point in points))


def freeze_edge(model, edge_id):
    edge = model.edges[edge_id]
    if isinstance(edge.curve, Straight):
        return LinePath(tuple(model.vertex_position(edge.start)), tuple(model.vertex_position(edge.end)))
    if isinstance(edge.curve, Arc):
        frame = model.arc_frame(edge_id)
        return EllipticArc(frame.center, frame.radius*frame.e1, frame.radius*frame.e2, 0., frame.sweep)
    if isinstance(edge.curve, Spline):
        return BezierPath(tuple(map(tuple, model._spline_points(edge))))
    if isinstance(edge.curve, EXACT_CURVES):
        return edge.curve
    raise GeometryError("arrangement curve definition is unsupported")


def _angle_parameter(curve, angle):
    # Root isolation already clips to the stored angular endpoints. Recover
    # their identities before dividing: on a tiny arc, (start+sweep-start)/sweep
    # can leave [0, 1] by many ulps even though the angle is exactly the endpoint.
    if angle == curve.start_angle:
        return 0.0
    if angle == curve.start_angle + curve.sweep_angle:
        return 1.0
    fraction = (angle-curve.start_angle)/curve.sweep_angle
    if -128*np.finfo(float).eps <= fraction <= 1+128*np.finfo(float).eps:
        fraction=min(1.,max(0.,fraction))
    if isinstance(curve, CylinderIntersectionCurve):
        if curve.parameterization == "left_square":
            return math.sqrt(max(0., fraction))
        if curve.parameterization == "right_square":
            return 1-math.sqrt(max(0., 1-fraction))
        if curve.parameterization == "both_sine":
            return 2/math.pi*math.asin(math.sqrt(min(1., max(0., fraction))))
    return fraction


def plane_roots(curve, normal, offset, *, tolerance=1e-12, cancellation_check=None):
    """Every curve/plane root; None denotes coincidence with the whole curve."""
    normal = np.asarray(normal)
    if isinstance(curve, LinePath):
        start = float(normal @ curve.start-offset)
        delta = float(normal @ (np.asarray(curve.end)-curve.start))
        # A line's residual is affine; its endpoint maximum bounds the whole
        # segment. An affine chart can leave a tiny nonzero dot product on a
        # horizontal boundary, which must not lose the winding crossing.
        if max(abs(start),abs(start+delta))<=tolerance:
            return None
        if delta == 0:
            return ()
        root = -start/delta
        return (min(1., max(0., root)),) if -tolerance <= root <= 1+tolerance else ()
    if isinstance(curve, EllipticArc):
        coefficients = (float(normal @ curve.center-offset), float(normal @ curve.u_vector),
                        float(normal @ curve.v_vector), 0., 0.)
        if sum(abs(value) for value in coefficients) <= tolerance:
            return None
        angles = trigonometric_roots(coefficients, start=curve.start_angle,
                                      sweep=curve.sweep_angle, tolerance=4*np.finfo(float).eps,
                                      cancellation_check=cancellation_check)
        return tuple(_angle_parameter(curve, angle) for angle in angles)
    if isinstance(curve, CylinderIntersectionCurve):
        from .quadric_curve_events import quadric_roots
        return quadric_roots(curve,np.zeros((3,3)),.5*normal,-offset,
                             tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(curve, (QuadricIntersectionCurve, BezierQuadricCurve)):
        from .quadric_algebra import QuadricSupport
        normal = np.asarray(normal, dtype=float)
        origin = tuple(float(v) for v in normal*(offset/float(normal @ normal)))
        return curve.roots_on(QuadricSupport("plane", origin, tuple(float(v) for v in normal)),
                              tolerance=tolerance, cancellation_check=cancellation_check)
    if isinstance(curve, BezierPath):
        # Bernstein-to-power conversion of the scalar plane constraint.
        values = np.asarray(curve.controls) @ normal-offset
        if np.max(np.abs(values)) <= tolerance:
            return None
        degree = len(values)-1
        coefficients = [sum((-1)**(k-i)*math.comb(degree, i)*math.comb(degree-i, k-i)*float(values[i])
                            for i in range(k+1)) for k in range(degree+1)]
        return tuple(min(1., max(0., root.witness)) for root in isolate_real_roots(
            coefficients, tolerance=4*np.finfo(float).eps, cancellation_check=cancellation_check)
            if -tolerance <= root.witness <= 1+tolerance)
    raise GeometryError("arrangement plane predicate is unsupported")


def point_parameters(curve, point, *, tolerance=1e-10):
    """All analytic parameters, retaining exact chart endpoint identities.

    Inverting a regularized branch at its transition can amplify an angular
    rounding error. The immutable endpoint definition is an independent exact
    candidate and must not be lost (or misclassified as an interior station).
    """
    point=np.asarray(point,dtype=float)
    endpoints=tuple(t for t in (0.,1.) if np.linalg.norm(curve.evaluate(t)-point)<=tolerance)
    roots=_point_parameters(curve,point,tolerance=tolerance)
    values=[]
    for parameter in (*endpoints,*roots):
        snapped=next((t for t in endpoints if abs(parameter-t)<=128*np.finfo(float).eps),parameter)
        if snapped not in values:
            values.append(snapped)
    return tuple(values)


def _point_parameters(curve, point, *, tolerance=1e-10):
    """All parameters for an exact point on a line or ellipse."""
    point = np.asarray(point)
    if isinstance(curve, LinePath):
        direction = np.asarray(curve.end)-curve.start
        length2 = float(direction @ direction)
        if length2 == 0:
            raise GeometryError("degenerate arrangement line")
        parameter = float((point-curve.start) @ direction)/length2
        if -tolerance <= parameter <= 1+tolerance and np.linalg.norm(curve.evaluate(parameter)-point) <= tolerance:
            return (min(1., max(0., parameter)),)
        return ()
    if isinstance(curve, EllipticArc):
        matrix = np.column_stack((curve.u_vector, curve.v_vector))
        coefficients = np.linalg.lstsq(matrix, point-curve.center, rcond=None)[0]
        if abs(float(coefficients @ coefficients)-1) > tolerance/max(np.linalg.norm(matrix), tolerance):
            return ()
        if np.linalg.norm(matrix @ coefficients+curve.center-point) > tolerance:
            return ()
        angle = math.atan2(coefficients[1], coefficients[0])
        lower, upper = sorted((curve.start_angle, curve.start_angle+curve.sweep_angle))
        roots = []
        for turn in range(math.ceil((lower-angle-tolerance)/math.tau), math.floor((upper-angle+tolerance)/math.tau)+1):
            parameter = (angle+turn*math.tau-curve.start_angle)/curve.sweep_angle
            roots.append(min(1., max(0., parameter)))
        return tuple(roots)
    if isinstance(curve, CylinderIntersectionCurve):
        matrix = np.asarray(curve.transform)
        original = np.linalg.solve(matrix[:3, :3], point-matrix[:3, 3])
        offset = original-curve.first.origin
        e1 = np.asarray(curve.first.radial_direction)
        e2 = _cross3(curve.first.axis, e1)
        angle = math.atan2(float(offset @ e2), float(offset @ e1))
        lower, upper = sorted((curve.start_angle, curve.start_angle+curve.sweep_angle))
        roots = []
        for turn in range(math.ceil((lower-angle-tolerance)/math.tau), math.floor((upper-angle+tolerance)/math.tau)+1):
            parameter = min(1., max(0., _angle_parameter(curve, angle+turn*math.tau)))
            if np.linalg.norm(curve.evaluate(parameter)-point) <= tolerance:
                roots.append(parameter)
        return tuple(roots)
    if isinstance(curve, (QuadricIntersectionCurve, BezierQuadricCurve)):
        return curve.parameters_of(point, tolerance=tolerance)
    if isinstance(curve, BezierPath):
        controls = np.asarray(curve.controls)
        coordinate = int(np.argmax(np.ptp(controls, axis=0)))
        normal = np.eye(3)[coordinate]
        roots = plane_roots(curve, normal, float(point[coordinate]), tolerance=tolerance)
        return tuple(t for t in (() if roots is None else roots)
                     if np.linalg.norm(curve.evaluate(t)-point) <= tolerance)
    raise GeometryError("arrangement point predicate is unsupported")


def line_curve_junctions(line, curve, *, tolerance=1e-10,cancellation_check=None):
    """All isolated crossings and overlap endpoints of a line and a curve."""
    direction = np.asarray(line.end)-line.start
    if np.linalg.norm(direction) <= tolerance:
        raise GeometryError("degenerate arrangement line")
    # Two independent planes contain the infinite line. Any isolated common
    # point must be a root of at least one non-coincident plane constraint.
    coordinate = np.eye(3)[int(np.argmin(np.abs(direction)))]
    normal = _cross3(direction, coordinate)
    normal /= np.linalg.norm(normal)
    first_roots = plane_roots(curve, normal, float(normal @ line.start), tolerance=tolerance,
                              cancellation_check=cancellation_check)
    other_normal = _cross3(direction, normal)
    other_normal /= np.linalg.norm(other_normal)
    second_roots = plane_roots(curve, other_normal, float(other_normal @ line.start), tolerance=tolerance,
                               cancellation_check=cancellation_check)
    if first_roots is None and second_roots is None:
        roots = (0., 1., *point_parameters(curve, line.start, tolerance=tolerance),
                 *point_parameters(curve, line.end, tolerance=tolerance))
    else:
        # Either plane can be tangent to a curved path. Enumerate both root
        # sets, then qualify the full line residual; a tangent scalar root's
        # roundoff must not hide a transverse root of the other constraint.
        roots = (*(first_roots or ()), *(second_roots or ()))
    result = []
    for parameter in roots:
        point = curve.evaluate(parameter)
        for line_parameter in point_parameters(line, point, tolerance=tolerance):
            result.append((line_parameter, parameter))
    return tuple(result)


def ellipse_curve_junctions(ellipse, curve, *, tolerance=1e-10,cancellation_check=None):
    """Analytic ellipse intersections with coplanar lines and ellipses."""
    if isinstance(curve, LinePath):
        return tuple((second, first) for first, second in line_curve_junctions(curve, ellipse, tolerance=tolerance,
                                                                           cancellation_check=cancellation_check))
    normal = _cross3(ellipse.u_vector, ellipse.v_vector)
    normal /= np.linalg.norm(normal)
    roots = plane_roots(curve, normal, float(normal @ ellipse.center), tolerance=tolerance,
                        cancellation_check=cancellation_check)
    if roots is None:
        if isinstance(curve,CylinderIntersectionCurve):
            from .quadric_curve_events import quadric_roots
            if ellipse_on_branch_supports(ellipse,curve,tolerance):
                roots=None
            else:
                inverse=np.linalg.pinv(np.column_stack((ellipse.u_vector,ellipse.v_vector)))
                matrix=inverse.T @ inverse
                center=np.asarray(ellipse.center)
                roots=quadric_roots(curve,matrix,-matrix @ center,float(center @ matrix @ center-1),
                                    tolerance=tolerance/max(np.linalg.norm(inverse),tolerance),
                                    cancellation_check=cancellation_check)
            if roots is None:
                roots=(0.,1.,*point_parameters(curve,ellipse.evaluate(0.),tolerance=tolerance),
                       *point_parameters(curve,ellipse.evaluate(1.),tolerance=tolerance))
        elif isinstance(curve, BezierPath):
            inverse = np.linalg.pinv(np.column_stack((ellipse.u_vector, ellipse.v_vector)))
            controls = (np.asarray(curve.controls)-ellipse.center) @ inverse.T
            degree = len(controls)-1
            power = [[sum(Fraction(float(controls[i, coordinate]))*((-1)**(k-i)
                      * math.comb(degree, i)*math.comb(degree-i, k-i)) for i in range(k+1))
                      for k in range(degree+1)] for coordinate in range(2)]
            polynomial = [Fraction(0)]*(2*degree+1)
            polynomial[0] = Fraction(-1)
            for values in power:
                for i, first in enumerate(values):
                    for j, second in enumerate(values):
                        polynomial[i+j] += first*second
            roots = tuple(root.witness for root in isolate_real_roots(polynomial,
                          tolerance=4*np.finfo(float).eps,cancellation_check=cancellation_check) if 0 <= root.witness <= 1)
        elif not isinstance(curve, EllipticArc):
            raise GeometryError("coincident-plane curve/ellipse predicate is not implemented")
        else:
            inverse = np.linalg.pinv(np.column_stack((ellipse.u_vector, ellipse.v_vector)))
            q0 = inverse @ (np.asarray(curve.center)-ellipse.center)
            qc, qs = inverse @ curve.u_vector, inverse @ curve.v_vector
            coefficients = (float(q0 @ q0+.5*(qc @ qc+qs @ qs)-1),
                            float(2*q0 @ qc), float(2*q0 @ qs),
                            float(.5*(qc @ qc-qs @ qs)), float(qc @ qs))
            if sum(abs(value) for value in coefficients) <= tolerance:
                roots = (0., 1., *point_parameters(curve, ellipse.evaluate(0.), tolerance=tolerance),
                         *point_parameters(curve, ellipse.evaluate(1.), tolerance=tolerance))
            else:
                angles = trigonometric_roots(coefficients, start=curve.start_angle,
                                              sweep=curve.sweep_angle, tolerance=4*np.finfo(float).eps,
                                              cancellation_check=cancellation_check)
                roots = tuple(_angle_parameter(curve, angle) for angle in angles)
    result = []
    for parameter in roots:
        point = curve.evaluate(parameter)
        for ellipse_parameter in point_parameters(ellipse, point, tolerance=tolerance):
            result.append((ellipse_parameter, parameter))
    return tuple(result)


def ellipse_on_branch_supports(ellipse,curve,tolerance):
    """Whole-curve implicit coefficient certificates on both supports.

    This avoids losing an exact common ellipse through floating pseudoinverse
    cancellation. Branch selection remains a separate chart-point inversion.
    """
    from .member_arrangements import _support_roots
    original=ellipse.transformed(np.linalg.inv(np.asarray(curve.transform)))
    return all(_support_roots(original,parent.surface(),tolerance,lambda:None) is None
               for parent in (curve.first,curve.second))


def curve_junctions(first, second, *, tolerance=1e-10,cancellation_check=None):
    if cancellation_check is not None and cancellation_check():
        raise GeometryError('curve intersection predicate cancelled')
    if _separated_line_pair(first, second, tolerance):
        from .runtime_diagnostics import _increment
        _increment('junction_line_bounds_pruned')
        return ()
    if _line_clear_of_branch(first, second, tolerance) or _curve_pair_clear(first, second, tolerance):
        _poll_like_original(cancellation_check)
        return ()
    # Named interval endpoints are independent qualified candidates. A root
    # of a clipped conic can round just beyond its new angular chart; retaining
    # its full world-space endpoint avoids an artificial detached port.
    candidates=[]
    for t in (0.,1.):
        if t != 0. and cancellation_check is not None and cancellation_check():
            raise GeometryError('curve intersection predicate cancelled')
        candidates.extend((t,s) for s in point_parameters(second,first.evaluate(t),tolerance=tolerance))
        candidates.extend((s,t) for s in point_parameters(first,second.evaluate(t),tolerance=tolerance))
    candidates.extend(_curve_junctions(first,second,tolerance=tolerance,
                                      cancellation_check=cancellation_check))
    result = tuple(sorted(dict.fromkeys(candidates)))
    if isinstance(first, _BRANCHES) or isinstance(second, _BRANCHES):
        result = _merge_touching(first, second, result, tolerance)
    return result


_TOUCH = 1e-6
_BRANCHES = (QuadricIntersectionCurve, BezierQuadricCurve)


def _merge_touching(first, second, junctions, tolerance):
    """One junction for a tangential contact.

    Where two curves touch, their exact crossings split into a pair as far apart as
    the square root of the data's rounding, many times the tolerance. Junctions that are
    close along both curves while the curves coincide between them (within tolerance)
    are the same contact: keep the one at a curve end, else their mean.
    """
    if len(junctions) < 2:
        return junctions
    clusters = [[junctions[0]]]
    for item in junctions[1:]:
        t, s = item
        t0, s0 = clusters[-1][-1]
        if (np.linalg.norm(first.evaluate(t)-first.evaluate(t0)) <= _TOUCH
                and np.linalg.norm(second.evaluate(s)-second.evaluate(s0)) <= _TOUCH
                and np.linalg.norm(first.evaluate(.5*(t+t0))-second.evaluate(.5*(s+s0))) <= tolerance):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    result = []
    for cluster in clusters:
        if len(cluster) == 1:
            result.append(cluster[0])
            continue
        ends = [item for item in cluster if item[0] in (0., 1.) or item[1] in (0., 1.)]
        result.append(ends[0] if ends else (sum(item[0] for item in cluster)/len(cluster),
                                            sum(item[1] for item in cluster)/len(cluster)))
    return tuple(result)


def _curve_junctions(first, second, *, tolerance=1e-10,cancellation_check=None):
    if cancellation_check is not None and cancellation_check():
        raise GeometryError('curve intersection predicate cancelled')
    if isinstance(first, QuadricIntersectionCurve) or isinstance(second, QuadricIntersectionCurve):
        from .quadric_events import quadric_pair_junctions
        return quadric_pair_junctions(first, second, tolerance=tolerance, cancellation_check=cancellation_check)
    if isinstance(first, LinePath):
        return line_curve_junctions(first, second, tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(second, LinePath):
        return tuple((b, a) for a, b in line_curve_junctions(second, first, tolerance=tolerance,
                                                         cancellation_check=cancellation_check))
    if isinstance(first, EllipticArc):
        return ellipse_curve_junctions(first, second, tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(second, EllipticArc):
        return tuple((b, a) for a, b in ellipse_curve_junctions(second, first, tolerance=tolerance,
                                                            cancellation_check=cancellation_check))
    if isinstance(first, CylinderIntersectionCurve) and isinstance(second, CylinderIntersectionCurve):
        from .cylinder_curve_events import branch_junctions
        return branch_junctions(first, second, tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(first,BezierPath) and isinstance(second,BezierPath):
        from .bezier_intersections import bezier_junctions
        return bezier_junctions(first,second,tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(first,BezierPath) and isinstance(second,CylinderIntersectionCurve):
        from .bezier_intersections import bezier_branch_junctions
        return bezier_branch_junctions(first,second,tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(second,BezierPath) and isinstance(first,CylinderIntersectionCurve):
        from .bezier_intersections import bezier_branch_junctions
        return tuple((b,a) for a,b in bezier_branch_junctions(second,first,tolerance=tolerance,
                                                           cancellation_check=cancellation_check))
    if isinstance(first, BezierQuadricCurve) or isinstance(second, BezierQuadricCurve):
        from .branch_events import bezier_quadric_junctions
        return bezier_quadric_junctions(first, second, tolerance=tolerance, cancellation_check=cancellation_check)
    raise GeometryError("general analytic curve-pair arrangement predicate is not implemented")
