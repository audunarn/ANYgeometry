"""Face display meshes: the N2 chart mesh of a face's N1 rings, lifted onto its support (N3).

Private helper of :func:`anygeometry.display_tessellation.face_display_mesh` (SPEC-N3, owner
decisions 2026-11-26; SPEC-N3b decisions D1-D4).  N3a: faces on a ``Plane``; N3b: on a
``Cylinder``; a ``Cone`` face is ``SUPPORT_UNSUPPORTED`` until N3c.  Same trust model as N1 and
N2: exact argument types,
fixed refusal text, the module's arithmetic under ``np.errstate(all="ignore")`` with explicit
guards, model calls through N1 (NumPy's default state), no raw arithmetic error escapes (typed
``TRIM_INVALID`` at worst), and every failure is a typed refusal, never a partial mesh.

Steps: (1) support check; (2) rings: the N1 samples of each coedge (reversed for a reversed
coedge, last point dropped; consecutive coedges must meet bit for bit); (3) chart of every ring
point (an orthonormal frame of the plane; on a cylinder ``(theta, h)`` about the module's own
axis frame, theta unwrapped along each ring); (4) seam and winding refusals; (5) boundary-chord
certificate against T_face; (6) slab refinement (none on a plane: the map is affine); (7) N2;
(8) lift by index, cut points ON the 3D N1 chord, ``A + f (B - A)``; (9) combinatorial,
orientation and deviation certificates.

Merged cut points (owner decision D1): two or more CUT points (points N2 made on a ring chord,
not ring vertices) whose chart points are bitwise equal become one mesh vertex (the first, its
own chord's position and source) before the certificates; the boundary chain is then taken as a
1-chain, a chain edge occurring in both directions (the zero-width spike between the two chords)
cancelling, and every certificate runs on the merged mesh.  A chart point shared with a ring
vertex (an N1 sample) stays ``TRIM_INVALID``.

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

Cylinder (N3b).  Chart ``c = (theta, h)``: ``h = (P - o).a`` (``a`` the normalised stored axis),
``theta = atan2((P - o).e2, (P - o).e1)`` unwrapped along each ring (each step reduced into
(-pi, pi]: N1 bounds every azimuth step by the angle plus 2 %); a ring whose steps, the closing
one included, sum to more than pi in magnitude winds the axis (``TRIM_WINDS_SUPPORT``); each hole
is shifted by a multiple of 2 pi onto the outer ring's sheet.  ``(e1, e2)`` is the module's own
frame (:func:`display_tessellation._axis_frame`), within ``FE = _FRAME_ERROR`` of an exactly
orthonormal frame ``(e1*, e2*, a*)`` about the axis direction, and ``S*(theta, h) = o + r (cos
theta e1* + sin theta e2*) + h a*`` an exact map of the support.  The chart values are numbers; the
offsets ``delta_i = |X_i - S(c_i)|`` (computed with the float frame) measure how far each stored
point is from ``S*`` of its chart point: ``|X_i - S*(c_i)| <= delta_i (1 + 8 eps) + A_face``,

    A_face = 32 eps (|o| + r + H + max |X|) + 2 FE (r + H)

(``H = max |h|``: the cosine and sine (an ulp each), the products and the three sums of ``S(c)``,
the subtraction and the frame's ``2 FE (r + |h|)``).

Paired deviation.  A mesh point at barycentric weights ``w`` of a triangle is ``x = sum w_i X_i``
with chart point ``c = sum w_i c_i``.  A ring vertex is its own N1 sample; a cut is ``X = A + f
(B - A) + rho`` on its chord's ring vertices ``A, B`` (``|rho| <= 16 eps max |X|``) with chart
point ``c = (1 - f) c_A + f c_B + kappa`` (``|kappa|`` computed, plus its rounding).  So ``x`` is a
convex combination ``sum W_j X_j`` of ring vertices (to ``rho``) and ``c`` the same combination of
their chart points (to ``kappa``).  Taylor with integral remainder about ``cbar = sum W_j c_j``
(``S*_tt = -r c(theta)``, every other second derivative 0, the first-order terms cancel):

    |x - S*(c)| <= max_j |X_j - S*(c_j)| + (r / 2) Var_W(theta) + r |kappa_t| + |kappa_h| + |rho|.

By the law of total variance ``Var_W(theta) = sum_i w_i V_i + Var_w(m_i)``, ``V_i = f_i (1 - f_i)
(theta_B - theta_A)^2`` (0 for a ring vertex) and ``m_i = (1 - f_i) theta_A + f_i theta_B``
within ``kappa_t`` of ``theta_i``, so ``Var_w(m) <= Var_w(theta) + 2 kappa_t dT + kappa_t^2``.  The
maximum over the triangle of ``G(w) = sum w_i V_i + Var_w(theta_i) = sum w_i (V_i + theta_i^2) -
(sum w_i theta_i)^2`` is attained on an edge (``G`` depends on ``w`` through the image point
``(sum w_i (V_i + theta_i^2), sum w_i theta_i)``, and for each second coordinate the largest first
one lies on the boundary of the image triangle, the image of the edges); on the edge ``p, q``
(``tau = theta_q - theta_p``) ``G = (1 - l) V_p + l V_q + l (1 - l) tau^2``, whose maximum is
``max(V_p, V_q) + max(tau^2 - |V_q - V_p|, 0)^2 / (4 tau^2)`` exactly.  Per triangle

    E(T) = max_j delta_j (1 + 8 eps) + (r / 2) (U (1 + 32 eps) + 8 eps (dT^2 + max V))
           + A_face + 16 eps max |X| + r k_t + k_h + (r / 2) (2 k_t dT_max + k_t^2)

(``U`` the edge maximum, ``j`` over the ring vertices under the triangle's vertices, ``k`` the
largest ``kappa``; the 32 eps and 8 eps terms the rounding of ``V``, ``tau^2`` and the closed
form), the whole sum raised by ``(1 + 8 eps)`` for its own rounding.  This is SPEC-N3 section
3's Taylor/Popoviciu bound taken over the ring vertices a cut
stands for: its ``max_i delta_i + r dT^2 / 8`` with a cut's own offset counted again would double
the chord's sag (288 of the 2663 tripod triangles over T_face at 7.5 deg; none with this form).
Boundary chords (two ring vertices): ``max delta (1 + 8 eps) + r dT^2 / 8 (1 + 8 eps) + A_face``
must meet T_face (else ``EDGE_UNSUPPORTED``, a fixed N1 chord).  T_face = ``tol.sag`` if given,
else ``r (tol.angle + a)^2 / 8 + surface_residual``, ``a`` the largest N1 ``allowance.angle`` of
the face's edges (owner decision Q5).

Refinement: every slab holding a triangle with ``E(T) > T_face`` gets a slab line at its
middle (an N2 extra line), N2 runs again and every certificate is re-run on the new mesh, up to
``_REFINE_ROUNDS`` rounds; still failing -> ``TRIM_INVALID`` (deviation), a slab too narrow to
bisect likewise, more than ``MAX_SLAB_INCIDENCES`` lines -> ``OUT_OF_DOMAIN``.  Bisection rather
than one uniform step: as ``tau`` shrinks the edge maximum falls to ``max(V_p, V_q)`` as soon as
``tau^2 <= |V_q - V_p|``, so a staggered cut near a chord's middle (``V`` near T_face's own
``dT^2 / 4``) usually needs one or two halvings, where a uniform step from the worst chord's
variance would need thousands of lines.  A face needing none is N2's mesh itself.

Orientation (Q7) on a cylinder: the reference normal is ``sense x c(theta)`` (``S_t x S_h = r
c(theta)``, outward).  ``n . c(theta) = p cos theta + q sin theta = R cos(theta - phi)``: its set
above a margin ``m >= 0`` is an arc shorter than pi, so on a footprint ``[theta_lo, theta_hi]``
narrower than pi the minimum exceeds ``m`` iff both end values do (an interval with both ends in
that arc cannot contain its complement, longer than the interval).  Checked as ``sense n .
c(theta_end) > 40 eps |d1| |d2|`` at both ends (the cross product, the frame and the trigonometric
values each round by a few eps of ``|d1| |d2|``); a footprint of pi or more is refused.  Rounding-
thin triangles (3D width at most ``A_face``) are exempt and counted.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import display_tessellation as dt
from ._slab_triangulation import MAX_SLAB_INCIDENCES, _integers, triangulate_rings
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
_TWO_PI = 2.0 * math.pi
#: chart azimuths closer than this x max(1, |theta|) are joined (rounding-level: a few ulps)
_SNAP = 16.0 * 2.0 ** -52
#: bisection rounds of the slabs holding a triangle over T_face (at most 2^8 sub-slabs each)
_REFINE_ROUNDS = 8

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
_WINDS = "a loop winds around the support's axis"


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
    __slots__ = ("points", "source", "edges", "allowance")

    def __init__(self, points, source, edges, allowance):
        self.points, self.source, self.edges, self.allowance = points, source, edges, allowance


def _ring(model, face_id, loop, tol) -> _Ring:
    """The N1 samples of each coedge in loop direction, last point dropped, consecutive coedges
    meeting bit for bit; ``source[j] = (edge, interval of the chord starting at point j)``."""
    if not loop:
        raise _invalid(face_id, _EMPTY)
    pieces, sources, edges = [], [], []
    allowance = 0.0
    for use in loop:
        samples, rounding, trim = dt._edge_display_samples_and_allowance(model, use.edge, tol)
        allowance = max(allowance, rounding.angle)
        points = samples.points if use.forward else samples.points[::-1]
        count = len(points) - 1
        index = np.arange(count, dtype=np.int64)
        sources.append(np.column_stack((np.full(count, samples.edge_id, dtype=np.int64),
                                        index if use.forward else count - 1 - index)))
        pieces.append(points)
        edges.append((samples, trim))
    for here, there in zip(pieces, pieces[1:] + pieces[:1]):
        if not np.array_equal(here[-1], there[0]):
            raise _invalid(face_id, _GAP)
    return _Ring(np.vstack([p[:-1] for p in pieces]), np.vstack(sources), edges, allowance)


def _check_loops(face_id, loops):
    seen = set()
    for loop in loops:
        for use in loop:
            if use.edge in seen:
                raise _refuse("TRIM_WINDS_SUPPORT", face_id, _SEAM)
            seen.add(use.edge)


# ----------------------------------------------------------------------------- analytic trim (Q6)
def _trim_sag(face_id, model, samples, tol, trim) -> float:
    """A certified upper bound of the distance of the edge's DRAWN chords from its analytic curve
    (Q6): per interval the exact chord sag (N1's :meth:`_Conic.sag`; 0 for a line) raised by its
    rounding (``64 eps + 8 eps x aspect``, N1's inflation), plus the vertex offset from the curve
    at an end interval, plus the rounding of the points: ``3 E`` for a conic (the evaluator's
    rounding of the curve point and of both chord ends, as N1's sag allowance), for a line its ends'
    ``X`` (and ``E`` with interior samples).  The drawn chords end at the N1 samples, whose first
    and last points are the edge's vertices.  ``trim`` is N1's ``(conic, evaluated curve ends)``
    of the edge (the same objects this bound would build: N1 constructs the conic from the same
    fields, an ``Arc``'s from its frame).

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
        conic, ends = trim
        if not isinstance(curve, (EllipticArc, Arc)) or conic is None:   # N1 accepted it: cannot happen
            raise _refuse("EDGE_UNSUPPORTED", face_id, "an edge's curve has no trim distance bound")
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


def _equal_runs(chart):
    """The chart points in (x, y) order (stable: equal points by index) and which of them equal
    their predecessor (float equality, as ``np.unique(axis=0)``)."""
    order = np.lexsort((chart[:, 1], chart[:, 0]))
    ordered = chart[order]
    same = (ordered[1:, 0] == ordered[:-1, 0]) & (ordered[1:, 1] == ordered[:-1, 1])
    return order, same


def _merge(face_id, k, chart):
    """D1: the merged index of every mesh vertex and the kept vertices (``None, None``: the chart
    is injective).  Equal chart points are allowed only among cut points; the first of them
    stays."""
    order, same = _equal_runs(chart)
    if not bool(np.any(same)):
        return None, None
    n = len(chart)
    starts = np.flatnonzero(np.concatenate(([True], ~same)))
    group = np.cumsum(np.concatenate(([True], ~same))) - 1
    representative = np.empty(n, dtype=np.int64)
    representative[order] = order[starts][group]
    if bool(np.any(representative[:k] != np.arange(k))) or bool(np.any(representative[k:] < k)):
        raise _invalid(face_id, _COINCIDE)
    kept = np.flatnonzero(representative == np.arange(n))
    index = np.empty(n, dtype=np.int64)
    index[kept] = np.arange(len(kept), dtype=np.int64)
    return index[representative], kept


def _combinatorial(face_id, rings, chart, triangles, cut_ends, signs, remap=None, merged_chart=None,
                   structure=None):
    """Q2 (i)-(iv): injective chart; every directed triangle edge used once; an edge without its
    reverse lies on one ring chord and these edges are exactly the chain of every ring (its
    vertices with the cut points of each chord in order, directed so the region is on the left);
    every chart triangle strictly CCW.  With all triangles positive, the 2-chain's boundary is the
    rings' chain, so every point off the edges is covered by exactly its winding number (1 inside
    the chart region, 0 outside): a conforming, non-overlapping covering.

    ``chart``, ``triangles`` and ``cut_ends`` are N2's.  With merged cut points (D1) ``remap`` maps
    every N2 index to its merged vertex and ``merged_chart`` is the merged (injective) chart: the
    chain is built on N2's indices, mapped, and taken as a 1-chain (a chain edge and its reverse,
    the zero-width spike between two chords, cancel); the triangles are mapped and must stay
    proper; every other check runs on the merged mesh (the caller, :func:`_merge`, has decided
    injectivity).  Returns the (mapped) triangles."""
    n = len(chart)
    k = sum(len(r.points) for r in rings)
    ring_of, following = _ring_structure(rings) if structure is None else structure
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
    if remap is None:
        expected_codes = np.concatenate([t * n + h for t, h in expected])
    else:
        tails = remap[np.concatenate([t for t, _ in expected])]
        heads = remap[np.concatenate([h for _, h in expected])]
        triangles = remap[triangles]
        chart, n = merged_chart, len(merged_chart)
        if bool(np.any((triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2])
                       | (triangles[:, 2] == triangles[:, 0]))):
            raise _invalid(face_id, _COINCIDE)
        if bool(np.any(tails == heads)):
            raise _invalid(face_id, _CHAIN)
        expected_codes = tails * n + heads
        ordered = np.sort(expected_codes)
        if bool(np.any(ordered[1:] == ordered[:-1])):
            raise _invalid(face_id, _CHAIN)
        expected_codes = expected_codes[~_member(heads * n + tails, ordered)]           # spikes cancel
    t0, t1, t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    tails = np.concatenate((t0, t1, t2))
    heads = np.concatenate((t1, t2, t0))
    codes = tails * n + heads
    unique = np.sort(codes)
    if bool(np.any(unique[1:] == unique[:-1])):
        raise _invalid(face_id, _EDGE_USE)
    reverse = heads * n + tails
    boundary = np.sort(codes[~_member(reverse, unique)])
    if not np.array_equal(boundary, np.sort(expected_codes)):
        raise _invalid(face_id, _CHAIN)
    if not _orient_positive(chart, triangles):
        raise _invalid(face_id, _ORIENT2D)
    return triangles


def _member(values, ordered):
    """``values`` found in the sorted array ``ordered`` (as ``np.isin``)."""
    if not len(ordered):
        return np.zeros(len(values), dtype=bool)
    at = np.minimum(np.searchsorted(ordered, values), len(ordered) - 1)
    return ordered[at] == values


class _Assembled:
    """The mesh whose combinatorics are certified: N2's, merged cut points (D1) folded in."""
    __slots__ = ("points", "triangles", "chart", "ends", "fraction", "source", "sense", "following")


def _assemble(face_id, rings, ring_points, charts, ring_chart, mesh) -> _Assembled:
    """Steps (8)-(9a): N2's ring vertices checked, the sense (the outer ring's exact chart area),
    cut points ON the 3D N1 chord (``A + f (B - A)``), merged cut points (D1), the combinatorial
    certificate and the sources."""
    k = len(ring_points)
    chart = mesh.points
    if not np.array_equal(chart[:k], ring_chart):
        raise _invalid(face_id, _CHAIN)
    signs = [_ring_sign(c) for c in charts]
    sense = signs[0]
    if sense == 0:
        raise _invalid(face_id, _ORIENT2D)
    ends = mesh.cut_ends.astype(np.int64)
    fraction = mesh.cut_fraction
    start, finish = ring_points[ends[:, 0]], ring_points[ends[:, 1]]
    cuts = start + fraction[:, None] * (finish - start)
    triangles = mesh.triangles.astype(np.int64)
    remap, kept = _merge(face_id, k, chart)
    structure = _ring_structure(rings)
    if remap is None:
        triangles = _combinatorial(face_id, rings, chart, triangles, ends, signs, structure=structure)
    else:
        merged_chart = chart[kept]
        triangles = _combinatorial(face_id, rings, chart, triangles, ends, signs, remap, merged_chart, structure)
        keep = kept[k:] - k
        chart, cuts, ends, fraction = merged_chart, cuts[keep], ends[keep], fraction[keep]
    out = _Assembled()
    out.points = np.vstack((ring_points, cuts))
    out.triangles, out.chart, out.ends, out.fraction, out.sense = triangles, chart, ends, fraction, sense
    # a cut point's source is its chord's: the interval starting at the chord's first ring vertex
    source = np.vstack([r.source for r in rings])
    following = structure[1]
    out.following = following
    first = np.where(following[ends[:, 0]] == ends[:, 1], ends[:, 0], ends[:, 1])
    out.source = np.vstack((source, source[first]))
    return out


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
    if bool(np.any(_equal_runs(ring_chart)[1])):
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
    # (8) lift by index: ring vertices are N1's points, cut points on the 3D N1 chord
    built = _assemble(face_id, rings, ring_points, charts, ring_chart, mesh)
    points, chart, triangles, sense = built.points, built.chart, built.triangles, built.sense
    # (9) certificates
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
    trim = max(_trim_sag(face_id, model, samples, tol, curve) for ring in rings for samples, curve in ring.edges)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, float(trim),
                              int(np.count_nonzero(thin)))


# ----------------------------------------------------------------------------- cylinder
class _CylinderChart:
    """Chart ``(theta, h)`` about the module's own axis frame (module docstring, Cylinder)."""

    def __init__(self, support):
        self.o, self.e1, self.e2 = dt._axis_frame(support)
        axis = np.asarray(support.axis, dtype=float)
        self.a = axis / dt._norm(axis)
        self.r = float(support.radius)

    def _project(self, points):
        d = points - self.o
        e1, e2, a = self.e1, self.e2, self.a
        x = d[:, 0] * e1[0] + d[:, 1] * e1[1] + d[:, 2] * e1[2]
        y = d[:, 0] * e2[0] + d[:, 1] * e2[1] + d[:, 2] * e2[2]
        h = d[:, 0] * a[0] + d[:, 1] * a[1] + d[:, 2] * a[2]
        return np.arctan2(y, x), h

    def charts(self, face_id, rings):
        """Each ring's chart, theta unwrapped along it (steps reduced into (-pi, pi]); a ring
        winding the axis is refused; holes shifted onto the outer ring's sheet."""
        out = []
        for ring in rings:
            theta, h = self._project(ring.points)
            if len(theta) > 1:
                wraps = np.round(np.diff(theta) / _TWO_PI)
                theta = theta - _TWO_PI * np.concatenate(([0.0], np.cumsum(wraps)))
            closing = float(theta[0] - theta[-1])
            total = float(theta[-1] - theta[0]) + (closing - _TWO_PI * round(closing / _TWO_PI))
            if not (math.isfinite(total) and abs(total) < math.pi):
                raise _refuse("TRIM_WINDS_SUPPORT", face_id, _WINDS)
            out.append((theta, h))
        middle = 0.5 * (float(out[0][0].min()) + float(out[0][0].max()))
        thetas = [out[0][0]]
        for theta, _ in out[1:]:
            shift = _TWO_PI * round((middle - 0.5 * (float(theta.min()) + float(theta.max()))) / _TWO_PI)
            thetas.append(theta + shift if shift else theta)
        # azimuths within rounding of each other (a ruling's two ends, a hole's ruling against a
        # vertex of another ring) become one value, the smallest: no ulp-wide slabs, whose cut
        # points on a long chord would stand tangentially off the chart order (a true sliver fold
        # of the lifted mesh); the offsets delta measure the change (a few ulps x r)
        joined = np.concatenate(thetas)
        order = np.argsort(joined, kind="stable")
        ordered = joined[order]
        gaps = np.diff(ordered)
        # the rounding of a computed azimuth: its own (|theta| ulps) and that of the projection,
        # eps (|P| + |o|) over the radius
        reach = max(float(np.abs(ring.points).max()) for ring in rings) + float(np.abs(self.o).max())
        close = gaps <= _SNAP * (np.maximum(1.0, np.abs(ordered[1:])) + reach / self.r)
        if bool(np.any(close)):
            group = np.concatenate(([0], np.cumsum(~close)))
            first = np.concatenate(([0], np.flatnonzero(~close) + 1))
            joined[order] = ordered[first[group]]
        charts, start = [], 0
        for theta, (_, h) in zip(thetas, out):
            charts.append(np.column_stack((joined[start:start + len(theta)], h)))
            start += len(theta)
        return charts

    def offsets(self, points, chart):
        """``delta_i = |X_i - S(c_i)|`` computed with the module's frame."""
        theta, h = chart[:, 0], chart[:, 1]
        mapped = (self.o + (self.r * np.cos(theta))[:, None] * self.e1 + (self.r * np.sin(theta))[:, None] * self.e2
                  + h[:, None] * self.a)
        return np.sqrt(np.sum((points - mapped) ** 2, axis=1))

    def rounding(self, points, chart):
        """``A_face`` (module docstring)."""
        size = float(np.sqrt(np.sum(points ** 2, axis=1)).max())
        height = float(np.abs(chart[:, 1]).max())
        return (32.0 * _EPS * (dt._norm(self.o) + self.r + height + size)
                + 2.0 * dt._FRAME_ERROR * (self.r + height))


def _edge_maximum(vp, vq, tau):
    """The exact maximum over an edge of ``(1 - l) V_p + l V_q + l (1 - l) tau^2`` (module
    docstring), before its rounding allowance."""
    tau2 = tau * tau
    gap = np.maximum(tau2 - np.abs(vq - vp), 0.0)
    extra = np.where(tau2 > 0.0, gap * gap / np.where(tau2 > 0.0, 4.0 * tau2, 1.0), 0.0)
    return np.maximum(vp, vq) + extra


class _CylinderBound:
    """The per-triangle paired-deviation bound ``E(T)`` of an assembled mesh (module docstring)."""

    def __init__(self, cyl, k, ring_deltas, built):
        r = cyl.r
        chart, ends, f = built.chart, built.ends, built.fraction
        theta, h = chart[:, 0], chart[:, 1]
        n = len(chart)
        ja = np.concatenate((np.arange(k, dtype=np.int64), ends[:, 0]))
        jb = np.concatenate((np.arange(k, dtype=np.int64), ends[:, 1]))
        ff = np.concatenate((np.zeros(k), f))
        ta, tb, ha, hb = theta[ja], theta[jb], h[ja], h[jb]
        self.variance = ff * (1.0 - ff) * (tb - ta) ** 2
        kt = np.abs(theta - (ta + ff * (tb - ta))) + 4.0 * _EPS * (np.abs(ta) + np.abs(tb))
        kh = np.abs(h - (ha + ff * (hb - ha))) + 4.0 * _EPS * (np.abs(ha) + np.abs(hb))
        kt[:k], kh[:k] = 0.0, 0.0
        self.under = np.maximum(ring_deltas[ja], ring_deltas[jb])
        self.theta = theta
        kt_max = float(kt.max()) if n else 0.0
        kh_max = float(kh.max()) if n else 0.0
        size = float(np.sqrt(np.sum(built.points ** 2, axis=1)).max())
        self.rounding = cyl.rounding(built.points, chart)
        self.kt_max, self.kh_max, self.r = kt_max, kh_max, r
        self.extra = self.rounding + 16.0 * _EPS * size + r * kt_max + kh_max

    def per_triangle(self, triangles):
        r = self.r
        tt = self.theta[triangles]
        vt = self.variance[triangles]
        u = np.maximum.reduce([_edge_maximum(vt[:, 0], vt[:, 1], tt[:, 1] - tt[:, 0]),
                               _edge_maximum(vt[:, 1], vt[:, 2], tt[:, 2] - tt[:, 1]),
                               _edge_maximum(vt[:, 2], vt[:, 0], tt[:, 0] - tt[:, 2])])
        width = tt.max(axis=1) - tt.min(axis=1)
        widest = float(width.max()) if len(width) else 0.0
        kt = self.kt_max
        spread = 0.5 * r * (2.0 * kt * widest + kt * kt)
        # the sum's own rounding (a few positive terms): (1 + 8 eps)
        return (np.max(self.under[triangles], axis=1) * (1.0 + 8.0 * _EPS)
                + 0.5 * r * (u * (1.0 + 32.0 * _EPS) + 8.0 * _EPS * (width ** 2 + vt.max(axis=1)))
                + self.extra + spread) * (1.0 + 8.0 * _EPS), tt, width


def _bisection(face_id, tt, failing, added):
    """One refinement round: a slab line at the middle of every slab holding a failing triangle
    (module docstring, Refinement)."""
    slabs = np.unique(np.column_stack((tt[failing].min(axis=1), tt[failing].max(axis=1))), axis=0)
    left, right = slabs[:, 0], slabs[:, 1]
    middle = 0.5 * (left + right)
    if not bool(np.all((left < middle) & (middle < right))):
        raise _invalid(face_id, _DEVIATION)
    if not added + len(middle) <= MAX_SLAB_INCIDENCES:
        raise dt.DisplayDomainRefused(f"face {dt._id_text(face_id)}: its refinement needs more than "
                                      f"{MAX_SLAB_INCIDENCES} slab lines (too complex)")
    return middle


def _lift_cylinder(model, face_id, tol, support, rings, extra):
    cyl = _CylinderChart(support)
    r = cyl.r
    if tol.sag is not None:
        face_tolerance = float(tol.sag)
    else:
        a = max(ring.allowance for ring in rings)
        face_tolerance = r * (tol.angle + a) ** 2 / 8 + float(model.tolerance.surface_residual)
    charts = cyl.charts(face_id, rings)
    ring_points = np.vstack([ring.points for ring in rings])
    ring_chart = np.vstack(charts)
    k = len(ring_points)
    # (5) boundary chords: the two-point bound
    ring_deltas = cyl.offsets(ring_points, ring_chart)
    _, following = _ring_structure(rings)
    span = np.abs(ring_chart[following, 0] - ring_chart[:, 0])
    chord = (np.maximum(ring_deltas, ring_deltas[following]) * (1.0 + 8.0 * _EPS)
             + r * span ** 2 / 8.0 * (1.0 + 8.0 * _EPS) + cyl.rounding(ring_points, ring_chart))
    worst = float(chord.max())
    if not worst <= face_tolerance:
        raise _refuse("EDGE_UNSUPPORTED", face_id, f"a boundary chord deviates from the cylinder by up to {worst!r}, "
                                                   f"above the face tolerance {face_tolerance!r}")
    # (6)-(9): N2, assembly, the deviation bound; failing slabs bisected, re-certified each round
    lines, added = extra, np.zeros(0)
    for attempt in range(_REFINE_ROUNDS + 1):
        mesh = triangulate_rings(charts, lines)
        built = _assemble(face_id, rings, ring_points, charts, ring_chart, mesh)
        bound = _CylinderBound(cyl, k, ring_deltas, built)
        per_triangle, tt, width = bound.per_triangle(built.triangles)
        failing = ~(per_triangle <= face_tolerance)
        if not bool(np.any(failing)):
            break
        if attempt == _REFINE_ROUNDS:
            raise _invalid(face_id, _DEVIATION)
        added = np.concatenate((added, _bisection(face_id, tt, failing, len(added))))
        lines = added if extra is None else np.concatenate((extra, added))
    points, chart, triangles, sense = built.points, built.chart, built.triangles, built.sense
    # Q7: the sign over each footprint (both ends of its theta range), rounding-thin exempt
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = np.cross(d1, d2)
    size = np.sqrt(np.sum(normal ** 2, axis=1))
    longest = np.sqrt(np.maximum.reduce([np.sum(d1 ** 2, axis=1), np.sum(d2 ** 2, axis=1),
                                         np.sum((c - b) ** 2, axis=1)]))
    thin = ~(size > bound.rounding * longest)               # 3D width |nT| / longest <= A_face
    margin = 40.0 * _EPS * np.sqrt(np.sum(d1 ** 2, axis=1)) * np.sqrt(np.sum(d2 ** 2, axis=1))
    p = normal @ cyl.e1
    q = normal @ cyl.e2
    lo, hi = tt.min(axis=1), tt.max(axis=1)
    at_lo = sense * (p * np.cos(lo) + q * np.sin(lo))
    at_hi = sense * (p * np.cos(hi) + q * np.sin(hi))
    if not bool(np.all(thin | ((at_lo > margin) & (at_hi > margin) & (width < math.pi)))):
        raise _invalid(face_id, _FOLD)
    deviation = float(per_triangle.max())
    # Q6: the analytic trim distance, always reported, enforced only with tol.sag
    trim = max(_trim_sag(face_id, model, samples, tol, curve) for ring in rings for samples, curve in ring.edges)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, float(trim),
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
        if isinstance(support, Cone):
            raise _refuse("SUPPORT_UNSUPPORTED", face_id, "a face lift on a cone is not supported yet")
        dt._check_support_domain(support)
        loops = (face.loop, *face.holes)
        _check_loops(face_id, loops)
        extra = _check_extra(_extra_x)
        rings = [_ring(model, face_id, loop, tol) for loop in loops]
        try:
            if isinstance(support, Plane):
                return _lift_plane(model, face_id, tol, support, rings, extra)
            return _lift_cylinder(model, face_id, tol, support, rings, extra)
        except dt.DisplayTrimRefused:
            raise
        except (ArithmeticError, ValueError) as error:      # no raw arithmetic error escapes
            raise _invalid(face_id, _ARITHMETIC) from error
