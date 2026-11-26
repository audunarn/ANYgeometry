"""Face display meshes: the N2 chart mesh of a face's N1 rings, lifted onto its support (N3).

Private helper of :func:`anygeometry.display_tessellation.face_display_mesh` (SPEC-N3, owner
decisions 2026-11-26).  N3a: faces on a ``Plane``; a ``Cylinder`` or ``Cone`` face is
``SUPPORT_UNSUPPORTED`` until N3b / N3c.  Same trust model as N1 and N2: exact argument types,
fixed refusal text, the module's arithmetic under ``np.errstate(all="ignore")`` with explicit
guards, model calls through N1 (NumPy's default state), no raw arithmetic error escapes (typed
``TRIM_INVALID`` at worst), and every failure is a typed refusal, never a partial mesh.

Steps: (1) support check; (2) rings: the N1 samples of each coedge (reversed for a reversed
coedge, last point dropped; consecutive coedges must meet bit for bit); (3) chart of every ring
point (an orthonormal frame of the plane); (4) seam refusal; (5) boundary-chord certificate against
T_face; (6) slab step (none on a plane: the map is affine); (7) N2; (8) lift by index, cut points
ON the 3D N1 chord, ``A + f (B - A)``; (9) combinatorial, orientation and deviation certificates.

Plane bounds (``eps`` the float epsilon).  The chart is ``c = ((P - o).e1, (P - o).e2)`` with the
module's own frame ``e1 = u / |u|``, ``n = (u x v) / |u x v|``, ``e2 = n x e1``.  Let ``e1*, e2*``
be the exactly orthonormal frame of the support plane (``e1* = u / |u|`` exactly, ``e2*`` its
exact in-plane perpendicular) and ``S*(c) = o + c_x e1* + c_y e2*``: an isometric, affine chart of
the support plane, so the paired deviation of a mesh point ``sum w_i X_i`` is ``|sum w_i (X_i -
S*(c_i))| <= max_i |X_i - S*(c_i)|`` (no curvature term on a plane).  With ``s = |u x v| / (|u||v|)``
(computed, lowered by 1e-3 relative for its own rounding; ``s < 1e-12`` is ``OUT_OF_DOMAIN``),
every computed frame vector lies within ``alpha = 8 eps / s + 16 eps`` of the exact one (``e1``:
two roundings; ``n``: the cross product's ``3 eps |u||v|`` over ``|u x v|`` plus the
normalisation; ``e2``: both plus its cross product).  Then
``|X_i - S*(c_i)| <= delta_i (1 + 8 eps) + A_face`` with ``delta_i = |X_i - S(c_i)|`` computed and

    A_face = 16 eps (|o| + max |X| + H) + 2 alpha max |c|

(``H`` the largest coordinate extent of the mesh points): the evaluation of ``S(c)`` rounds by at
most ``2 eps (|o| + 2 |c|)``, the subtraction and norm by ``2 eps (|X| + |S(c)|)``, and the frame
by ``2 alpha |c|``.  ``deviation_bound = max over triangles of max_i delta_i (1 + 8 eps) + A_face``.

Orientation (Q7): on a plane the support normal is constant, so the sign holds over a triangle's
whole footprint iff it holds once: ``nT . (sense n) > (16 eps + 2 alpha) |d1| |d2|`` (``nT = d1 x
d2`` computed from the stored vertices; the differences, the cross product, the dot and the
frame's tilt each round by at most a few eps of ``|d1| |d2|``).  A triangle whose 3D width
``|nT| / longest edge`` is at most ``A_face`` is rounding-thin: exempt and counted in
``degenerate_count``.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import display_tessellation as dt
from ._slab_triangulation import _integers, triangulate_rings
from .curves import Arc, Straight
from .errors import GeometryError
from .exact_curves import EllipticArc
from .line_ranges import LineRange
from .surfaces import Cone, Cylinder, Plane

_EPS = dt._EPS
#: Shewchuk's ccwerrboundA: a filtered orientation decides when |det| exceeds this x the sum of
#: the magnitudes of its two products (plus an underflow slack); otherwise exact integers decide
_ORIENT_BOUND = (3.0 + 16.0 * 2.0 ** -53) * 2.0 ** -53
_UNDERFLOW = 2.0 ** -1000
#: smallest sine of the angle between a plane's vectors (the frame's tilt bound grows as 1 / s)
_MIN_SINE = 1.0e-12

# fixed refusal texts (owner decision N2 r5 (1); TRIM_INVALID covers both bad trims and
# uncertifiable lifts, distinguished here: owner decision 2026-11-26, REFUSAL_CODES unchanged)
_GAP = "consecutive coedges do not meet bit for bit"
_EMPTY = "a loop has no coedges"
_SEAM = "the face uses one edge twice (a seam or slit)"
_RING_COINCIDE = "two ring points coincide in the chart"
_COINCIDE = "two mesh vertices coincide in the chart (lift not certified: a collapsed trim interval or a cut on another ring's vertex)"
_CHAIN = "the mesh boundary is not the chain of the ring chords (lift not certified)"
_EDGE_USE = "a mesh edge is not shared by exactly two oppositely oriented triangles (lift not certified)"
_ORIENT2D = "a chart triangle is not strictly counter-clockwise (lift not certified)"
_FOLD = "a lifted triangle folds against the face normal (lift not certified)"
_DEVIATION = "a lifted triangle's deviation bound exceeds the face tolerance (lift not certified)"
_ARITHMETIC = "an arithmetic error in the lift"


def _refuse(code, face_id, text):
    return dt.DisplayTrimRefused(code, f"face {dt._id_text(face_id)}: {text}")


def _invalid(face_id, text):
    return _refuse("TRIM_INVALID", face_id, text)


# ----------------------------------------------------------------------------- input
def _check_input(model, face_id, tol):
    from .model import GeometryModel
    if not issubclass(type(model), GeometryModel):
        raise GeometryError("model must be a GeometryModel")
    if not issubclass(type(tol), dt.DisplayTolerance):
        raise GeometryError("tol must be a DisplayTolerance")
    if not dt._exact_type_in(face_id, dt._ID_TYPES):
        raise GeometryError("face id must be an int or a NumPy integer scalar")
    face_id = int(face_id)
    if not -dt._ID_BOUND < face_id < dt._ID_BOUND:
        raise GeometryError("face id is out of range (|id| < 2**63)")
    if face_id not in model.faces:
        raise GeometryError(f"face {dt._id_text(face_id)} is not in the model")
    return face_id, model.faces[face_id]


def _check_extra(extra_x):
    if extra_x is None:
        return None
    extra = np.array(extra_x, dtype=np.float64).reshape(-1)
    return extra


# ----------------------------------------------------------------------------- rings
class _Ring:
    __slots__ = ("points", "source", "edges")

    def __init__(self, points, source, edges):
        self.points, self.source, self.edges = points, source, edges


def _ring(model, face_id, loop, tol) -> _Ring:
    """The N1 samples of each coedge in loop direction, last point dropped, consecutive coedges
    meeting bit for bit; ``source[j] = (edge, interval of the chord starting at point j)``."""
    if not loop:
        raise _invalid(face_id, _EMPTY)
    pieces, sources, edges = [], [], []
    for use in loop:
        samples = dt.edge_display_samples(model, use.edge, tol)
        points = samples.points if use.forward else samples.points[::-1]
        count = len(points) - 1
        index = np.arange(count, dtype=np.int64)
        sources.append(np.column_stack((np.full(count, samples.edge_id, dtype=np.int64),
                                        index if use.forward else count - 1 - index)))
        pieces.append(points)
        edges.append(samples)
    for here, there in zip(pieces, pieces[1:] + pieces[:1]):
        if not np.array_equal(here[-1], there[0]):
            raise _invalid(face_id, _GAP)
    return _Ring(np.vstack([p[:-1] for p in pieces]), np.vstack(sources), edges)


def _check_loops(face_id, loops):
    seen = set()
    for loop in loops:
        for use in loop:
            if use.edge in seen:
                raise _refuse("TRIM_WINDS_SUPPORT", face_id, _SEAM)
            seen.add(use.edge)


# ----------------------------------------------------------------------------- analytic trim (Q6)
def _trim_sag(face_id, model, samples, tol) -> float:
    """A certified upper bound of the distance of the edge's DRAWN chords from its analytic curve
    (Q6): per interval the exact chord sag (N1's :meth:`_Conic.sag`; 0 for a line) raised by its
    rounding (``64 eps + 8 eps x aspect``, N1's inflation), plus the vertex offset from the curve
    at an end interval, plus the rounding of the points: ``3 E`` for a conic (the evaluator's
    rounding of the curve point and of both chord ends, as N1's sag allowance), for a line its ends'
    ``X`` (and ``E`` with interior samples).  The drawn chords end at the N1 samples, whose first
    and last points are the edge's vertices.

    Enforced only when ``tol.sag`` is given (owner decision Q6): the exact part (sag and offsets)
    must be at most ``tol.sag (1 + 64 eps)`` plus the offsets N1 treats as rounding (at most
    ``4 eps max(1, |vertex coordinate|)``), the bound N1's own refinement meets; else
    ``EDGE_UNSUPPORTED`` (fail closed)."""
    edge_id = samples.edge_id
    edge = model.edges[edge_id]
    curve = edge.curve
    start, end = samples.points[0], samples.points[-1]
    interior = len(samples.parameters) > 2
    conic = None
    if isinstance(curve, Straight):
        offsets, rounding = (0.0, 0.0), dt._line_end_rounding(start, end)
        if interior:
            rounding += 8.0 * _EPS * (dt._norm(start) + dt._norm(end))
    elif isinstance(curve, LineRange):
        a, b = np.asarray(curve.a, dtype=float), np.asarray(curve.b, dtype=float)
        offsets = (dt._norm(start - a), dt._norm(end - b))
        rounding = dt._line_end_rounding(a, b)
        if interior:
            rounding += 8.0 * _EPS * (dt._norm(a) + dt._norm(b))
    else:
        if isinstance(curve, EllipticArc):
            conic = dt._Conic(edge_id, curve.center, curve.u_vector, curve.v_vector, curve.start_angle,
                              curve.sweep_angle)
        elif isinstance(curve, Arc):
            frame = dt._model_call(model.arc_frame, edge_id)
            conic = dt._Conic(edge_id, frame.center, frame.radius * frame.e1, frame.radius * frame.e2, 0.0,
                              frame.sweep)
        else:                                                # N1 accepted it: cannot happen
            raise _refuse("EDGE_UNSUPPORTED", face_id, "an edge's curve has no trim distance bound")
        ends = dt._evaluate(model, edge_id, [0.0, 1.0])
        offsets = (dt._norm(start - ends[0]), dt._norm(end - ends[1]))
        rounding = 3.0 * conic.point_rounding
    worst = max(offsets)
    if conic is not None:
        inflate = 1.0 + 64.0 * _EPS + 8.0 * _EPS * (1.0 if conic.circle else conic.aspect)
        parameters = samples.parameters.tolist()
        last = len(parameters) - 2
        for i, (a, b) in enumerate(zip(parameters[:-1], parameters[1:])):
            offset = max(offsets[0] if i == 0 else 0.0, offsets[1] if i == last else 0.0)
            worst = max(worst, conic.sag(a, b) * inflate + offset)
    if tol.sag is not None:
        resolution = 4.0 * _EPS * max(1.0, float(np.abs(np.vstack((start, end))).max()))
        hidden = max([0.0] + [offset for offset in offsets if offset <= resolution])
        if not worst <= tol.sag * (1.0 + 64.0 * _EPS) + hidden:
            raise _refuse("EDGE_UNSUPPORTED", face_id, f"edge {dt._id_text(edge_id)}'s drawn chords leave its "
                                                       f"curve by up to {worst + rounding!r}, above the sag {tol.sag!r}")
    return worst + rounding


# ----------------------------------------------------------------------------- exact signs
def _ring_sign(ring) -> int:
    """The exact sign of a ring's area (float filter, N2's bound; else exact integers)."""
    x, y = ring[:, 0], ring[:, 1]
    terms = (x - x[0]) * (np.roll(y, -1) - np.roll(y, 1))
    doubled, magnitude = float(np.sum(terms)), float(np.sum(np.abs(terms)))
    error = (len(ring) + 8) * _EPS * magnitude + _UNDERFLOW
    if doubled > error:
        return 1
    if doubled < -error:
        return -1
    ints, _ = _integers(x.tolist() + y.tolist())
    n = len(ring)
    ix, iy = ints[:n], ints[n:]
    exact = sum(ix[i] * (iy[(i + 1) % n] - iy[i - 1]) for i in range(n))
    return (exact > 0) - (exact < 0)


def _orient_positive(chart, triangles) -> bool:
    """Every chart triangle strictly counter-clockwise, exactly (filter, then rationals)."""
    a, b, c = chart[triangles[:, 0]], chart[triangles[:, 1]], chart[triangles[:, 2]]
    left = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
    right = (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    det = left - right
    bound = _ORIENT_BOUND * (np.abs(left) + np.abs(right)) + _UNDERFLOW
    if bool(np.all(det > bound)):
        return True
    if bool(np.any(det < -bound)):
        return False
    for row in np.flatnonzero(~(det > bound)).tolist():
        p, q, r = ([Fraction(float(v)) for v in point] for point in (a[row], b[row], c[row]))
        if not (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) > 0:
            return False
    return True


# ----------------------------------------------------------------------------- certificates
def _ring_structure(rings):
    """Each ring vertex's ring and the index of the next vertex along its ring (cyclic)."""
    sizes = np.array([len(r.points) for r in rings], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    ring_of = np.repeat(np.arange(len(rings)), sizes)
    position = np.arange(int(sizes.sum()), dtype=np.int64)
    return ring_of, starts[ring_of] + (position - starts[ring_of] + 1) % sizes[ring_of]


def _combinatorial(face_id, rings, chart, triangles, cut_ends, signs) -> None:
    """Q2 (i)-(iv): injective chart; every directed triangle edge used once; an edge without its
    reverse lies on one ring chord and these edges are exactly the chain of every ring (its
    vertices with the cut points of each chord in order, directed so the region is on the left);
    every chart triangle strictly CCW.  With all triangles positive, the 2-chain's boundary is the
    rings' chain, so every point off the edges is covered by exactly its winding number (1 inside
    the chart region, 0 outside): a conforming, non-overlapping covering."""
    n = len(chart)
    k = sum(len(r.points) for r in rings)
    if len(np.unique(chart, axis=0)) != n:
        raise _invalid(face_id, _COINCIDE)
    ring_of, following = _ring_structure(rings)
    position = np.arange(k, dtype=np.int64)
    # the chord of each cut point: its ends must be consecutive ring vertices
    a, b = cut_ends[:, 0].astype(np.int64), cut_ends[:, 1].astype(np.int64)
    if len(a) and not (bool(np.all((a >= 0) & (a < k) & (b >= 0) & (b < k)))):
        raise _invalid(face_id, _CHAIN)
    forward = following[a] == b if len(a) else np.zeros(0, dtype=bool)
    backward = following[b] == a if len(a) else np.zeros(0, dtype=bool)
    if not bool(np.all(forward | backward)):
        raise _invalid(face_id, _CHAIN)
    chord = np.where(forward, a, b)                         # the chord's first ring vertex
    x = chart[:, 0]
    x0, x1 = x[chord], x[following[chord]]
    cut_x = x[k:]
    rising = x0 < x1
    if not bool(np.all(np.where(rising, (x0 < cut_x) & (cut_x < x1), (x1 < cut_x) & (cut_x < x0)))):
        raise _invalid(face_id, _CHAIN)
    along = np.where(rising, cut_x, -cut_x)
    # the ordered chain: ring vertex i, then its chord's cuts by their position along it
    primary = np.concatenate((position, chord))
    secondary = np.concatenate((np.zeros(k), np.ones(len(chord))))
    tertiary = np.concatenate((np.zeros(k), along))
    order = np.lexsort((tertiary, secondary, primary))
    ordered_primary, ordered_tertiary = primary[order], tertiary[order]
    same = (ordered_primary[1:] == ordered_primary[:-1]) & (order[1:] >= k) & (order[:-1] >= k)
    if bool(np.any(same & (ordered_tertiary[1:] <= ordered_tertiary[:-1]))):
        raise _invalid(face_id, _CHAIN)
    node_ring = ring_of[ordered_primary]
    expected = []
    for r in range(len(rings)):
        nodes = order[node_ring == r]
        tail, head = nodes, np.roll(nodes, -1)
        along_ring = (signs[r] > 0) if r == 0 else (signs[r] < 0)
        expected.append((tail, head) if along_ring else (head, tail))
    expected_codes = np.concatenate([t * n + h for t, h in expected])
    t0, t1, t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    tails = np.concatenate((t0, t1, t2))
    heads = np.concatenate((t1, t2, t0))
    codes = tails * n + heads
    unique = np.unique(codes)
    if len(unique) != len(codes):
        raise _invalid(face_id, _EDGE_USE)
    reverse = heads * n + tails
    boundary = np.sort(codes[~np.isin(reverse, unique)])
    if not np.array_equal(boundary, np.sort(expected_codes)):
        raise _invalid(face_id, _CHAIN)
    if not _orient_positive(chart, triangles):
        raise _invalid(face_id, _ORIENT2D)


# ----------------------------------------------------------------------------- plane
class _PlaneChart:
    def __init__(self, face_id, support):
        self.o = np.asarray(support.origin, dtype=float)
        u = np.asarray(support.u_vector, dtype=float)
        v = np.asarray(support.v_vector, dtype=float)
        nu, nv = dt._norm(u), dt._norm(v)
        cross = np.cross(u, v)
        ncross = dt._norm(cross)
        sine = ncross / (nu * nv) if nu > 0.0 and nv > 0.0 else 0.0
        if not (math.isfinite(sine) and sine >= _MIN_SINE):
            raise dt.DisplayDomainRefused(f"face {dt._id_text(face_id)}: its plane's vectors are parallel to "
                                          f"within a sine of {_MIN_SINE:g}")
        self.e1 = u / nu
        self.n = cross / ncross
        self.e2 = np.cross(self.n, self.e1)
        self.alpha = 8.0 * _EPS / (sine * (1.0 - 1.0e-3)) + 16.0 * _EPS

    def chart(self, points):
        d = points - self.o
        e1, e2 = self.e1, self.e2
        return np.column_stack((d[:, 0] * e1[0] + d[:, 1] * e1[1] + d[:, 2] * e1[2],
                                d[:, 0] * e2[0] + d[:, 1] * e2[1] + d[:, 2] * e2[2]))

    def offsets(self, points, chart):
        """``delta_i = |X_i - S(c_i)|`` computed (the plane frame's own map)."""
        mapped = self.o + chart[:, :1] * self.e1 + chart[:, 1:] * self.e2
        return np.sqrt(np.sum((points - mapped) ** 2, axis=1))

    def rounding(self, points, chart):
        """``A_face`` (module docstring)."""
        size = float(np.sqrt(np.sum(points ** 2, axis=1)).max())
        extent = float(np.ptp(points, axis=0).max())
        reach = float(np.sqrt(np.sum(chart ** 2, axis=1)).max())
        return 16.0 * _EPS * (dt._norm(self.o) + size + extent) + 2.0 * self.alpha * reach


def _lift_plane(model, face_id, tol, support, rings, extra):
    plane = _PlaneChart(face_id, support)
    face_tolerance = float(tol.sag) if tol.sag is not None else float(model.tolerance.surface_residual)
    ring_points = np.vstack([r.points for r in rings])
    charts = [plane.chart(r.points) for r in rings]
    ring_chart = np.vstack(charts)
    k = len(ring_points)
    if len(np.unique(ring_chart, axis=0)) != k:
        raise _invalid(face_id, _RING_COINCIDE)
    # (5) boundary chords: on a plane a chord's paired deviation is its ends' (affine map)
    deltas = plane.offsets(ring_points, ring_chart)
    rounding = plane.rounding(ring_points, ring_chart)
    worst = float(deltas.max()) * (1.0 + 8.0 * _EPS) + rounding
    if not worst <= face_tolerance:
        raise _refuse("EDGE_UNSUPPORTED", face_id, f"a boundary chord deviates from the plane by up to {worst!r}, "
                                                   f"above the face tolerance {face_tolerance!r}")
    # (6) no slab step on a plane: (7) N2, refusals typed
    mesh = triangulate_rings(charts, extra)
    chart = mesh.points
    if not np.array_equal(chart[:k], ring_chart):
        raise _invalid(face_id, _CHAIN)
    signs = [_ring_sign(c) for c in charts]
    sense = signs[0]
    if sense == 0:
        raise _invalid(face_id, _ORIENT2D)
    # (8) lift by index: ring vertices are N1's points, cut points on the 3D N1 chord
    ends = mesh.cut_ends.astype(np.int64)
    start, finish = ring_points[ends[:, 0]], ring_points[ends[:, 1]]
    cuts = start + mesh.cut_fraction[:, None] * (finish - start)
    points = np.vstack((ring_points, cuts))
    triangles = mesh.triangles.astype(np.int64)
    # (9) certificates
    _combinatorial(face_id, rings, chart, triangles, ends, signs)
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    rounding = plane.rounding(points, chart)
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = np.cross(d1, d2)
    size = np.sqrt(np.sum(normal ** 2, axis=1))
    longest = np.sqrt(np.maximum.reduce([np.sum(d1 ** 2, axis=1), np.sum(d2 ** 2, axis=1),
                                         np.sum((c - b) ** 2, axis=1)]))
    thin = ~(size > rounding * longest)                     # 3D width |nT| / longest <= A_face
    lengths = np.sqrt(np.sum(d1 ** 2, axis=1)) * np.sqrt(np.sum(d2 ** 2, axis=1))
    dot = normal @ (sense * plane.n)
    margin = (16.0 * _EPS + 2.0 * plane.alpha) * lengths
    if not bool(np.all(thin | (dot > margin))):
        raise _invalid(face_id, _FOLD)
    deltas = plane.offsets(points, chart)
    per_triangle = np.max(deltas[triangles], axis=1) * (1.0 + 8.0 * _EPS) + rounding
    deviation = float(per_triangle.max())
    if not deviation <= face_tolerance:
        raise _invalid(face_id, _DEVIATION)
    # Q6: the analytic trim distance, always reported, enforced only with tol.sag
    trim = max(_trim_sag(face_id, model, samples, tol) for ring in rings for samples in ring.edges)
    # a cut point's source is its chord's: the interval starting at the chord's first ring vertex
    source = np.vstack([r.source for r in rings])
    _, following = _ring_structure(rings)
    first = np.where(following[ends[:, 0]] == ends[:, 1], ends[:, 0], ends[:, 1])
    source = np.vstack((source, source[first]))
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(source), int(sense), face_tolerance, deviation, float(trim),
                              int(np.count_nonzero(thin)))


# ----------------------------------------------------------------------------- entry
def _lift(model, face_id, tol, *, _extra_x=None):
    """The face's certified display mesh, else a typed refusal (``_extra_x``: a private test seam
    forcing extra N2 slab lines)."""
    with np.errstate(all="ignore"):
        face_id, face = _check_input(model, face_id, tol)
        support = face.surface
        if face.parameterization is not None:
            raise _refuse("SUPPORT_UNSUPPORTED", face_id, "the face has an explicit parameterization")
        if not isinstance(support, (Plane, Cylinder, Cone)):
            raise _refuse("SUPPORT_UNSUPPORTED", face_id, "its support is not a plane, cylinder or cone")
        if not isinstance(support, Plane):
            raise _refuse("SUPPORT_UNSUPPORTED", face_id, "a face lift on a cylinder or cone is not supported yet")
        dt._check_support_domain(support)
        loops = (face.loop, *face.holes)
        _check_loops(face_id, loops)
        extra = _check_extra(_extra_x)
        rings = [_ring(model, face_id, loop, tol) for loop in loops]
        try:
            return _lift_plane(model, face_id, tol, support, rings, extra)
        except dt.DisplayTrimRefused:
            raise
        except (ArithmeticError, ValueError) as error:      # no raw arithmetic error escapes
            raise _invalid(face_id, _ARITHMETIC) from error
