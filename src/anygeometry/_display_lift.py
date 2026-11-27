"""Face display meshes: the N2 chart mesh of a face's N1 rings, lifted onto its support (N3).

Private helper of :func:`anygeometry.display_tessellation.face_display_mesh` (SPEC-N3, owner
decisions 2026-11-26; SPEC-N3b decisions D1-D4; SPEC-N3c decisions E1-E3).  N3a: faces on a
``Plane``; N3b: on a ``Cylinder``; N3c: on a ``Cone``.  The contract and the derivations are in
``docs/DISPLAY_TESSELLATION.md``, "Face lift (N3)".  Same trust model as N1 and N2: exact argument types,
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
``|nT| / longest edge`` is at most ``A_face`` and that this test cannot certify is decided exactly
(M1, :func:`_thin_plane`); only an exactly zero normal is exempt and counted in
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
thin triangles (3D width at most ``A_face``) the test cannot certify are decided exactly (M1).

Cone (N3c).  ``S*(theta, h) = o + rho(h) c(theta) + h a*``, ``rho = r0 + k h``, ``k = (r1 - r0) /
(height |axis|)``; ``A_face`` as the cylinder's with ``rho_max`` for ``r`` plus the rounding of
``rho``, ``8 eps (|r0| + |k| H)``.  ``rho`` is linear in ``h``, so exactly ``sum W_j S*(c_j) -
S*(cbar) = k Cov_W(theta, h) c'(thetabar) + sum W_j rho_j R_j`` (``|R_j| <= dt_j^2 / 2``): the
paired deviation is bounded by ``max delta`` plus the maximum over the triangle of ``H(w) = sum
w_i P_i + sum_{i<j} w_i w_j D_ij`` (each cut's own chord variance and covariance in ``P``, the
vertices' variance and covariance in ``D``), taken exactly (edges in closed form, the critical
point where ``H`` is strictly concave), plus the kappa and rounding terms (:class:`_ConeBound`).
Boundary chords: ``rho dT^2 / 8 + |k| dT dH / 4`` plus offsets and ``A_face``.  Orientation: the
sign of ``n . (c(theta) - k a)`` over the footprint, ends plus the sinusoid's minimum where it may
lie inside.  A trim reaching the apex (or ``rho <= 0`` at a ring point) is ``OUT_OF_DOMAIN``.
Azimuth join (E2): on a cone the azimuth's rounding uses the smallest ``rho`` at the ring points.

Codex round 1 (owner decisions H1, M1-M4, L1).  H1: a D1 merge and the chain cancellation it
allows must hold in 3D: the merge distance ``d`` plus the cancelled chord length ``L``
(:func:`_identification`) is added to ``trim_sag_bound`` and must not exceed T_face.  M1: a
rounding-thin triangle the float sign test cannot certify is decided from its exact normal
(:func:`_thin_plane`, :func:`_thin_curved`); exempt only a zero normal or, on a curved support, a
footprint minimum within the support normal's rounding.  M2: the cut points' rounding
(:func:`_cut_rounding`) is in ``trim_sag_bound``.  M3: where the world-scaled bound exceeds
T_face, an origin-relative bound (and on a plane an exact rational decision,
:meth:`_PlaneChart.exact_bound`) decides; a face the first bound accepts is unchanged.  M4: N2's
ring caps are enforced while the rings are sampled (:func:`_ring`).
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import _slab_triangulation as _slab
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
_IDENTIFY = ("merged cut points, or the boundary interval their merge cancels, lie apart in 3D by more than the "
             "face tolerance (lift not certified)")


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


def _ring(model, face_id, loop, tol, budget=None) -> _Ring:
    """The N1 samples of each coedge in loop direction, last point dropped, consecutive coedges
    meeting bit for bit; ``source[j] = (edge, interval of the chord starting at point j)``.

    ``budget`` (a one-element list, default a fresh one) is the running count of ring points over
    the face's loops:
    as soon as it exceeds N2's ``MAX_SLAB_RING_POINTS`` (read at call time) the face is refused
    ``OUT_OF_DOMAIN`` with N2's own fixed text, before any further coedge is sampled (Codex N3 r1
    M4: budgets before work)."""
    if not loop:
        raise _invalid(face_id, _EMPTY)
    if budget is None:
        budget = [0]
    pieces, sources, edges = [], [], []
    allowance = 0.0
    for use in loop:
        samples, rounding, trim = dt._edge_display_samples_and_allowance(model, use.edge, tol)
        allowance = max(allowance, rounding.angle)
        points = samples.points if use.forward else samples.points[::-1]
        count = len(points) - 1
        budget[0] += count
        cap = _slab.MAX_SLAB_RING_POINTS
        if budget[0] > cap:
            raise _slab._too_complex(f"{dt._id_text(budget[0])} ring points (MAX_SLAB_RING_POINTS = {cap})")
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
    terms = (x - x[0]) * (np.concatenate((y[1:], y[:1])) - np.concatenate((y[-1:], y[:-1])))   # np.roll(y, -1), (y, 1)
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
    """D1: the merged index of every mesh vertex, the kept vertices and each vertex's kept
    representative (N2 index) (``None, None, None``: the chart is injective).  Equal chart points
    are allowed only among cut points; the first of them stays."""
    order, same = _equal_runs(chart)
    if not bool(np.any(same)):
        return None, None, None
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
    return index[representative], kept, representative


def _combinatorial(face_id, rings, chart, triangles, cut_ends, signs, remap=None, merged_chart=None,
                   structure=None, cancelled=None):
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
    injectivity).  The cancelled chain edges (their N2 tail and head indices) are appended to
    ``cancelled`` when given.  Returns the (mapped) triangles."""
    n = len(chart)
    k = sum(len(r.points) for r in rings)
    ring_of, following = _ring_structure(rings) if structure is None else structure
    position = np.arange(k, dtype=np.int64)
    a, b = cut_ends[:, 0].astype(np.int64), cut_ends[:, 1].astype(np.int64)
    if not len(a) and remap is None:
        # no cut points: the chain is every ring's chords, directed so the region is on the left
        # (the general construction below gives the same set of edges)
        along = np.array([(signs[r] > 0) if r == 0 else (signs[r] < 0) for r in range(len(rings))])[ring_of]
        expected_codes = np.where(along, position * n + following, following * n + position)
        return _triangle_edges(face_id, chart, triangles, n, expected_codes)
    # the chord of each cut point: its ends must be consecutive ring vertices
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
        tail, head = nodes, np.concatenate((nodes[1:], nodes[:1]))          # np.roll(nodes, -1)
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
        cancel = _member(heads * n + tails, ordered)                                   # spikes cancel
        # H1: a ring whose whole chain cancels leaves no boundary to cover its interval
        sizes = np.cumsum([0] + [len(t) for t, _ in expected])
        if any(bool(np.all(cancel[sizes[r]:sizes[r + 1]])) for r in range(len(rings))):
            raise _invalid(face_id, _CHAIN)
        if cancelled is not None:
            cancelled.append((np.concatenate([t for t, _ in expected])[cancel],
                              np.concatenate([h for _, h in expected])[cancel]))
        expected_codes = expected_codes[~cancel]
    return _triangle_edges(face_id, chart, triangles, n, expected_codes)


def _triangle_edges(face_id, chart, triangles, n, expected_codes):
    """Q2 (ii)-(iv) on the (mapped) triangles: every directed edge once, the unpaired ones exactly
    the expected chain, every chart triangle strictly CCW.  Returns the triangles."""
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


def _cross(u, v):
    """``np.cross`` of (n, 3) arrays, the same products and differences (bit for bit), without
    its axis handling (a few microseconds per call on small meshes)."""
    out = np.empty(np.broadcast_shapes(u.shape, v.shape))
    out[:, 0] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    out[:, 1] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    out[:, 2] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return out


def _cross3(u, v):
    """``np.cross`` of two 3-vectors, bit for bit (the same float products and differences)."""
    u0, u1, u2 = float(u[0]), float(u[1]), float(u[2])
    v0, v1, v2 = float(v[0]), float(v[1]), float(v[2])
    return np.array([u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0])


def _axis_frame(support):
    """:func:`display_tessellation._axis_frame` (the module's own frame about the unit axis), the
    same operations bit for bit, with :func:`_cross3` for ``np.cross`` (pinned by a test)."""
    origin = np.asarray(support.origin, dtype=float)
    axis = np.asarray(support.axis, dtype=float)
    axis = axis / dt._norm(axis)
    basis = np.zeros(3)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    cross = _cross3(axis, basis)
    e1 = cross / dt._norm(cross)
    return origin, e1, _cross3(axis, e1)


def _member(values, ordered):
    """``values`` found in the sorted array ``ordered`` (as ``np.isin``)."""
    if not len(ordered):
        return np.zeros(len(values), dtype=bool)
    at = np.minimum(np.searchsorted(ordered, values), len(ordered) - 1)
    return ordered[at] == values


class _Assembled:
    """The mesh whose combinatorics are certified: N2's, merged cut points (D1) folded in.

    ``cut_rounding``: a bound of the distance of every cut point (as computed, ``A + f (B - A)``)
    from its exact 3D chord (M2); ``identify``: a bound of how far the lifted boundary moves by the
    D1 identifications, ``d + L`` (H1, :func:`_identification`)."""
    __slots__ = ("points", "triangles", "chart", "ends", "fraction", "source", "sense", "following",
                 "cut_rounding", "identify")


def _assemble(face_id, rings, ring_points, charts, ring_chart, mesh, structure=None) -> _Assembled:
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
    remap, kept, representative = _merge(face_id, k, chart)
    if structure is None:
        structure = _ring_structure(rings)
    out = _Assembled()
    # M2: a computed cut A + f (B - A) (f in [0, 1]) is within 2 eps (|A_i| + |B_i|) of its exact
    # chord per coordinate (the difference, the product and the sum, half an ulp of each result)
    out.cut_rounding = _cut_rounding(start, finish) if len(cuts) else 0.0
    out.identify = 0.0
    if remap is None:
        triangles = _combinatorial(face_id, rings, chart, triangles, ends, signs, structure=structure)
    else:
        merged_chart = chart[kept]
        cancelled = []
        triangles = _combinatorial(face_id, rings, chart, triangles, ends, signs, remap, merged_chart, structure,
                                   cancelled)
        out.identify = _identification(ring_points, cuts, representative, *cancelled[0])
        keep = kept[k:] - k
        chart, cuts, ends, fraction = merged_chart, cuts[keep], ends[keep], fraction[keep]
    out.points = np.vstack((ring_points, cuts))
    out.triangles, out.chart, out.ends, out.fraction, out.sense = triangles, chart, ends, fraction, sense
    # a cut point's source is its chord's: the interval starting at the chord's first ring vertex
    source = np.vstack([r.source for r in rings])
    following = structure[1]
    out.following = following
    first = np.where(following[ends[:, 0]] == ends[:, 1], ends[:, 0], ends[:, 1])
    out.source = np.vstack((source, source[first]))
    return out


def _cut_rounding(start, finish):
    """``4 eps |(|A| + |B|)|`` over the cut points' chords, raised for its own rounding (M2)."""
    span = np.abs(start) + np.abs(finish)
    return float(np.sqrt(np.sum(span * span, axis=1)).max()) * 4.0 * _EPS * (1.0 + 8.0 * _EPS)


def _identification(ring_points, cuts, representative, tails, heads):
    """H1: how far the D1 identifications move the lifted boundary from the lifted ring chords.

    A merged cut is replaced by its representative (a cut on another chord with the same chart
    point): ``d`` = the largest 3D distance between the two, as computed.  A chain edge that
    cancels with its reverse (the zero-width spike between two chords) leaves the piece of chord
    it stood for off the mesh boundary: every point of such a piece is within the total 3D length
    ``L`` of the cancelled pieces of a node that stays on the boundary (a cancelled run along a
    ring ends at nodes of that ring's remaining chain; a ring whose whole chain cancels is refused
    by :func:`_combinatorial`).  So every lifted chord point is within ``d + L`` of the mesh
    boundary and every boundary point within ``d`` of a chord: ``d + L`` is added to
    ``trim_sag_bound`` and must not exceed T_face (else ``TRIM_INVALID``).  Both measured on the
    computed points (the cuts' own rounding is ``cut_rounding``), raised for the rounding of the
    differences, norms and sum."""
    points = np.vstack((ring_points, cuts))
    moved = representative != np.arange(len(points))
    d = (float(np.sqrt(np.sum((points[moved] - points[representative[moved]]) ** 2, axis=1)).max())
         if bool(np.any(moved)) else 0.0)
    length = float(np.sum(np.sqrt(np.sum((points[tails] - points[heads]) ** 2, axis=1)))) if len(tails) else 0.0
    return (d + length) * (1.0 + 8.0 * _EPS + 4.0 * _EPS * (len(tails) + 1))


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
        self._exact = tuple([Fraction(float(x)) for x in vec] for vec in (self.o, u, v))
        eu, ev = self._exact[1], self._exact[2]
        self.exact_normal = (eu[1] * ev[2] - eu[2] * ev[1], eu[2] * ev[0] - eu[0] * ev[2], eu[0] * ev[1] - eu[1] * ev[0])

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

    def local_bounds(self, points, chart):
        """Per point ``|X - S*(c)| <= delta_loc (1 + 8 eps) + A_loc`` with origin-relative
        magnitudes only (M3): ``delta_loc = |w - c_x e1 - c_y e2|`` computed, ``w = X - o`` (the
        same rounded difference the chart is made from, within half an ulp of the exact one per
        coordinate), and ``A_loc = 16 eps (W + |c|max) + 2 alpha |c|max`` (``W = max |w|``): the
        difference (``eps/2 W``), the two products and two sums per coordinate (``2 eps (W +
        2 |c|)``), the frame (``alpha (|c_x| + |c_y|) <= 2 alpha |c|``); the norm's own rounding is
        in ``1 + 8 eps``.  No world-scale term: a face's decision does not depend on where it is."""
        w = points - self.o
        mapped = chart[:, :1] * self.e1 + chart[:, 1:] * self.e2
        delta = np.sqrt(np.sum((w - mapped) ** 2, axis=1))
        size = float(np.sqrt(np.sum(w ** 2, axis=1)).max())
        reach = float(np.sqrt(np.sum(chart ** 2, axis=1)).max())
        return delta * (1.0 + 8.0 * _EPS) + (16.0 * _EPS * (size + reach) + 2.0 * self.alpha * reach)

    def exact_bound(self, point, chart_point, limit):
        """The deviation ``D = |X - S*(c)|`` of one point decided against ``limit`` exactly (M3):
        ``D^2 = |w|^2 + |c|^2 - 2 c_x (w.u) / sqrt(u.u) - 2 c_y (w.v') / sqrt(v'.v')`` with ``w = X -
        o`` and ``v' = v - (v.u / u.u) u`` in rationals (``e1* = u / |u|``, ``e2* = v' / |v'|``, the
        exactly orthonormal frame), the two square roots enclosed in rational intervals of relative
        width ``2^-bits`` (bits 128, then 512, then 2048).  Returns an upper bound of ``D`` (at most
        ``limit``) when ``D <= limit`` is proven, ``None`` when ``D > limit`` is proven or the
        interval cannot decide (fail closed)."""
        o, u, v = self._exact
        w = [Fraction(float(x)) - y for x, y in zip(point, o)]
        cx, cy = Fraction(float(chart_point[0])), Fraction(float(chart_point[1]))
        uu = sum(x * x for x in u)
        vu = sum(x * y for x, y in zip(v, u))
        vp = [x - vu / uu * y for x, y in zip(v, u)]
        vv = sum(x * x for x in vp)
        alpha_ = sum(x * y for x, y in zip(w, u))
        beta = sum(x * y for x, y in zip(w, vp))
        q = sum(x * x for x in w) + cx * cx + cy * cy
        t2 = Fraction(limit) ** 2
        for bits in (128, 512, 2048):
            lo_terms, hi_terms = q, q
            for coefficient, base in ((-2 * cx * alpha_, uu), (-2 * cy * beta, vv)):
                enclosure = _inverse_sqrt(base, bits)           # 1 / sqrt(base) in [low, high]
                if enclosure is None:
                    break
                a, b = coefficient * enclosure[0], coefficient * enclosure[1]
                lo_terms += min(a, b)
                hi_terms += max(a, b)
            else:
                enclosure = True
            if enclosure is None:
                continue
            if lo_terms > t2:
                return None
            if hi_terms <= t2:
                bound = math.sqrt(float(max(hi_terms, Fraction(0))))
                bound = math.nextafter(math.nextafter(bound, math.inf), math.inf)
                return min(float(limit), bound)
        return None


def _inverse_sqrt(x, bits):
    """Rationals ``low <= 1 / sqrt(x) <= high`` (``x > 0`` rational), relative width ~``2^-bits``
    (``None`` when ``x`` is too small for this precision)."""
    scale = 1 << bits
    n = x.numerator * scale * scale // x.denominator
    root = math.isqrt(n)                                   # root <= sqrt(x) scale < root + 1 (n floored)
    if root == 0:
        return None
    return Fraction(scale, root + 1), Fraction(scale, root)


def _plane_point_bounds(face_id, plane, points, chart, face_tolerance, rounding, code, text):
    """Certified per-point bounds of ``|X - S*(c)|`` (M3).  First N3a's bound ``delta (1 + 8 eps)
    + A_face`` (outputs bit-identical wherever it decides); where it exceeds the face tolerance,
    the origin-relative bound (:meth:`_PlaneChart.local_bounds`), then the exact decision
    (:meth:`_PlaneChart.exact_bound`).  A point proven over, or not decidable: refusal ``code``."""
    bounds = plane.offsets(points, chart) * (1.0 + 8.0 * _EPS) + rounding
    if bool(np.all(bounds <= face_tolerance)):
        return bounds
    bounds = np.minimum(bounds, plane.local_bounds(points, chart))
    for i in np.flatnonzero(~(bounds <= face_tolerance)).tolist():
        exact = plane.exact_bound(points[i], chart[i], face_tolerance)
        if exact is None:
            raise _refuse(code, face_id, text(float(bounds[i])))
        bounds[i] = exact
    return bounds


def _exact_cross(points, rows):
    """The exact cross products ``(b - a) x (c - a)`` (rationals) of the stored points of the
    triangles ``rows`` (an (m, 3) index array)."""
    out = []
    for i, j, l in rows.tolist():
        a, b, c = ([Fraction(float(x)) for x in points[t]] for t in (i, j, l))
        d1 = [y - x for x, y in zip(a, b)]
        d2 = [y - x for x, y in zip(a, c)]
        out.append((d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]))
    return out


def _thin_plane(points, rows, plane_normal, sense):
    """M1 on a plane: the orientation of each thin triangle decided exactly from the stored
    points, ``sense (nT . (u x v))`` in rationals (the support normal is exactly representable).
    Returns ``(exempt, folded)`` masks: exempt only for an exactly zero normal."""
    exempt = np.zeros(len(rows), dtype=bool)
    folded = np.zeros(len(rows), dtype=bool)
    for r, n in enumerate(_exact_cross(points, rows)):
        if n == (0, 0, 0):
            exempt[r] = True
        elif not sense * sum(x * y for x, y in zip(n, plane_normal)) > 0:
            folded[r] = True
    return exempt, folded


def _thin_curved(points, rows, lo, hi, e1, e2, axis, k, sense):
    """M1 on a cylinder (``k = 0``) or cone: the orientation of each thin triangle over its
    footprint ``[lo, hi]`` decided from the EXACT normal of its stored points (rationals, then
    correctly rounded: a relative error of half an ulp per coordinate), not from the world-scaled
    width test.  The support normal ``c(theta) - k a`` is not representable (a transcendental
    frame), so ``g(theta) = n . (c(theta) - k a)`` is evaluated with the module's frame and
    decided against the margin ``m = 32 eps (1 + |k|) |n|``: the normal's rounding (eps / 2), the
    frame (``2 FE``, 8 eps), the cosine and sine (an ulp each), the dot products and the sum
    (6 eps), each times ``(1 + |k|)``.  Positive over the footprint (both ends above ``m``; on a
    cone also the sinusoid's minimum where it may lie inside): certified, not exempt.  A value
    below ``-m`` (a fold the stored floats decide) or a footprint of pi or more: folded.  Exempt
    (counted in ``degenerate_count``) only an exactly zero normal or a footprint minimum within
    ``[-m, m]`` (the sign is undecidable because the support normal is not representable).
    Returns ``(exempt, folded)`` masks."""
    exempt = np.zeros(len(rows), dtype=bool)
    folded = np.zeros(len(rows), dtype=bool)
    slack = 1.0 + abs(k)
    for r, n in enumerate(_exact_cross(points, rows)):
        if n == (0, 0, 0):
            exempt[r] = True
            continue
        nf = np.array([float(x) for x in n])
        margin = 32.0 * _EPS * slack * float(np.sqrt(nf @ nf)) + _UNDERFLOW
        p, q = sense * float(nf @ e1), sense * float(nf @ e2)
        level = -sense * k * float(nf @ axis)
        a, b = float(lo[r]), float(hi[r])
        if not b - a < math.pi:
            folded[r] = True
            continue
        at_lo = p * math.cos(a) + q * math.sin(a) + level
        at_hi = p * math.cos(b) + q * math.sin(b) + level
        inside = (-p * math.sin(a) + q * math.cos(a) <= margin) and (-p * math.sin(b) + q * math.cos(b) >= -margin)
        bottom = level - math.hypot(p, q)
        low = min(at_lo, at_hi)
        if k == 0.0:
            if low > margin:                       # both ends positive on a footprint < pi: positive
                continue
        elif low > margin and (not inside or bottom > margin):
            continue
        if inside:
            low = min(low, bottom)
        if low < -margin:
            folded[r] = True
        else:
            exempt[r] = True
    return exempt, folded


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
    def over(worst):
        return f"a boundary chord deviates from the plane by up to {worst!r}, above the face tolerance {face_tolerance!r}"
    _plane_point_bounds(face_id, plane, ring_points, ring_chart, face_tolerance,
                        plane.rounding(ring_points, ring_chart), "EDGE_UNSUPPORTED", over)
    # (6) no slab step on a plane: (7) N2, refusals typed
    mesh = triangulate_rings(charts, extra)
    # (8) lift by index: ring vertices are N1's points, cut points on the 3D N1 chord
    built = _assemble(face_id, rings, ring_points, charts, ring_chart, mesh)
    _check_identification(face_id, built, face_tolerance)
    points, chart, triangles, sense = built.points, built.chart, built.triangles, built.sense
    # (9) certificates
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    rounding = plane.rounding(points, chart)
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = _cross(d1, d2)
    size = np.sqrt(np.sum(normal ** 2, axis=1))
    longest = np.sqrt(np.maximum.reduce([np.sum(d1 ** 2, axis=1), np.sum(d2 ** 2, axis=1),
                                         np.sum((c - b) ** 2, axis=1)]))
    thin = ~(size > rounding * longest)                     # 3D width |nT| / longest <= A_face
    lengths = np.sqrt(np.sum(d1 ** 2, axis=1)) * np.sqrt(np.sum(d2 ** 2, axis=1))
    dot = normal @ (sense * plane.n)
    margin = (16.0 * _EPS + 2.0 * plane.alpha) * lengths
    positive = dot > margin                                 # the float filter certifies these
    if not bool(np.all(thin | positive)):
        raise _invalid(face_id, _FOLD)
    # M1: a thin triangle the filter cannot certify is decided exactly; exempt only a zero normal
    exempt_count = 0
    undecided = np.flatnonzero(thin & ~positive)
    if len(undecided):
        exempt, folded = _thin_plane(points, triangles[undecided], plane.exact_normal, sense)
        if bool(np.any(folded)):
            raise _invalid(face_id, _FOLD)
        exempt_count = int(np.count_nonzero(exempt))
    bounds = _plane_point_bounds(face_id, plane, points, chart, face_tolerance, rounding, "TRIM_INVALID",
                                 lambda worst: _DEVIATION)
    per_triangle = np.max(bounds[triangles], axis=1)
    deviation = float(per_triangle.max())
    if not deviation <= face_tolerance:
        raise _invalid(face_id, _DEVIATION)
    trim = _trim_bound(face_id, model, tol, rings, built)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, trim, exempt_count)


def _check_identification(face_id, built, face_tolerance):
    """H1: the D1 identifications may move the lifted boundary by at most T_face."""
    if not built.identify <= face_tolerance:
        raise _invalid(face_id, _IDENTIFY)


def _trim_bound(face_id, model, tol, rings, built):
    """Q6 ``trim_sag_bound``: the N1 chords' certified distance from the analytic trim (always
    reported, enforced only with tol.sag) plus every point N3 creates on the boundary: the cut
    points' rounding (M2) and the D1 identifications' displacement (H1)."""
    trim = max(_trim_sag(face_id, model, samples, tol, curve) for ring in rings for samples, curve in ring.edges)
    if built.cut_rounding or built.identify:
        trim = (trim + built.cut_rounding + built.identify) * (1.0 + 4.0 * _EPS)
    return float(trim)


# ----------------------------------------------------------------------------- cylinder
class _CylinderChart:
    """Chart ``(theta, h)`` about the module's own axis frame (module docstring, Cylinder)."""

    def __init__(self, support):
        self.o, self.e1, self.e2 = _axis_frame(support)
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
        return self._join(face_id, rings, self._unwrap(face_id, rings), self.r)

    def _unwrap(self, face_id, rings):
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
        return out

    def _join(self, face_id, rings, out, radius):
        """Holes onto the outer ring's sheet, then azimuths within rounding joined (E2);
        ``radius`` the smallest support radius at the ring points (the azimuth's rounding is
        ``eps |P| / radius``)."""
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
        close = gaps <= _SNAP * (np.maximum(1.0, np.abs(ordered[1:])) + reach / radius)
        joined_any = bool(np.any(close))
        if joined_any:
            group = np.concatenate(([0], np.cumsum(~close)))
            first = np.concatenate(([0], np.flatnonzero(~close) + 1))
            joined[order] = ordered[first[group]]
        charts, start = [], 0
        for theta, (_, h) in zip(thetas, out):
            charts.append(np.column_stack((joined[start:start + len(theta)], h)))
            start += len(theta)
        # E2: the join moves chart azimuths only (the ring vertices stay N1's points); it must
        # never make two distinct N1 samples one chart point
        if joined_any and bool(np.any(_equal_runs(np.vstack(charts))[1])):
            raise _invalid(face_id, _RING_COINCIDE)
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

    def radius(self, chart):
        """The support radius at each chart point (a cylinder's is constant)."""
        return np.full(len(chart), self.r)

    def local_offsets(self, points, chart):
        """``delta_loc = |w - S_loc(c)|`` with ``w = X - o`` and ``S_loc = rho (cos e1 + sin e2) +
        h a`` (no ``o``: origin-relative magnitudes, M3 class)."""
        theta, h = chart[:, 0], chart[:, 1]
        rho = self.radius(chart)
        mapped = (rho * np.cos(theta))[:, None] * self.e1 + (rho * np.sin(theta))[:, None] * self.e2 + h[:, None] * self.a
        return np.sqrt(np.sum((points - self.o - mapped) ** 2, axis=1))

    def local_rounding(self, points, chart):
        """``A_loc``: ``A_face`` with ``|o| + max |X|`` replaced by ``W = max |X - o|`` (the
        difference ``X - o`` rounds by half an ulp of its result): the rounding of the cosine, sine,
        products and sums of ``S_loc``, of ``w - S_loc`` and of the frame; no world-scale term."""
        size = float(np.sqrt(np.sum((points - self.o) ** 2, axis=1)).max())
        height = float(np.abs(chart[:, 1]).max())
        return 32.0 * _EPS * (self.r + height + size) + 2.0 * dt._FRAME_ERROR * (self.r + height)


def _edge_maximum(vp, vq, tau):
    """The exact maximum over an edge of ``(1 - l) V_p + l V_q + l (1 - l) tau^2`` (module
    docstring), before its rounding allowance."""
    tau2 = tau * tau
    gap = np.maximum(tau2 - np.abs(vq - vp), 0.0)
    extra = np.where(tau2 > 0.0, gap * gap / np.where(tau2 > 0.0, 4.0 * tau2, 1.0), 0.0)
    return np.maximum(vp, vq) + extra


class _CylinderBound:
    """The per-triangle paired-deviation bound ``E(T)`` of an assembled mesh (module docstring).
    ``local``: the origin-relative variant (offsets and ``A_loc`` from
    :meth:`_CylinderChart.local_offsets` / ``local_rounding``; the cut points' own rounding
    ``rho`` bounded by ``cut_rounding`` from their chords' ends), used where N3b's bound exceeds
    T_face (M3 class)."""

    def __init__(self, cyl, k, ring_deltas, built, ring_rounding, local=False):
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
        if local:
            # the cuts' own rounding |rho| from their chords' ends (cut_rounding), not the world size
            self.rounding = ring_rounding if n == k else cyl.local_rounding(built.points, chart)
            cut = built.cut_rounding
        else:
            size = float(np.sqrt(np.sum(built.points ** 2, axis=1)).max())
            # A_face of the mesh points; without cut points they are the ring's (the same value)
            self.rounding = ring_rounding if n == k else cyl.rounding(built.points, chart)
            cut = 16.0 * _EPS * size
        self.kt_max, self.kh_max, self.r = kt_max, kh_max, r
        self.extra = self.rounding + cut + r * kt_max + kh_max

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
    structure = _ring_structure(rings)
    following = structure[1]
    span = np.abs(ring_chart[following, 0] - ring_chart[:, 0])
    ring_rounding = cyl.rounding(ring_points, ring_chart)
    chord = (np.maximum(ring_deltas, ring_deltas[following]) * (1.0 + 8.0 * _EPS)
             + r * span ** 2 / 8.0 * (1.0 + 8.0 * _EPS) + ring_rounding)
    worst = float(chord.max())
    local = None
    if not worst <= face_tolerance:                         # M3 class: the origin-relative bound
        local = (cyl.local_offsets(ring_points, ring_chart), cyl.local_rounding(ring_points, ring_chart))
        chord = np.minimum(chord, np.maximum(local[0], local[0][following]) * (1.0 + 8.0 * _EPS)
                           + r * span ** 2 / 8.0 * (1.0 + 8.0 * _EPS) + local[1])
        worst = float(chord.max())
    if not worst <= face_tolerance:
        raise _refuse("EDGE_UNSUPPORTED", face_id, f"a boundary chord deviates from the cylinder by up to {worst!r}, "
                                                   f"above the face tolerance {face_tolerance!r}")
    # (6)-(9): N2, assembly, the deviation bound; failing slabs bisected, re-certified each round
    lines, added = extra, np.zeros(0)
    for attempt in range(_REFINE_ROUNDS + 1):
        mesh = triangulate_rings(charts, lines)
        built = _assemble(face_id, rings, ring_points, charts, ring_chart, mesh, structure)
        _check_identification(face_id, built, face_tolerance)
        bound = _CylinderBound(cyl, k, ring_deltas, built, ring_rounding)
        per_triangle, tt, width = bound.per_triangle(built.triangles)
        failing = ~(per_triangle <= face_tolerance)
        if bool(np.any(failing)):
            if local is None:
                local = (cyl.local_offsets(ring_points, ring_chart), cyl.local_rounding(ring_points, ring_chart))
            per_triangle = np.minimum(per_triangle, _CylinderBound(cyl, k, local[0], built, local[1],
                                                                   local=True).per_triangle(built.triangles)[0])
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
    normal = _cross(d1, d2)
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
    positive = (at_lo > margin) & (at_hi > margin) & (width < math.pi)
    if not bool(np.all(thin | positive)):
        raise _invalid(face_id, _FOLD)
    exempt_count = _thin_decision(face_id, points, triangles, thin & ~positive, lo, hi, cyl, 0.0, sense)
    deviation = float(per_triangle.max())
    trim = _trim_bound(face_id, model, tol, rings, built)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, trim, exempt_count)


def _thin_decision(face_id, points, triangles, undecided, lo, hi, chart, slope, sense):
    """M1: the thin triangles the float filter cannot certify, decided by :func:`_thin_curved`;
    a fold refuses, the exempt ones are counted."""
    rows = np.flatnonzero(undecided)
    if not len(rows):
        return 0
    exempt, folded = _thin_curved(points, triangles[rows], lo[rows], hi[rows], chart.e1, chart.e2, chart.a, slope,
                                  sense)
    if bool(np.any(folded)):
        raise _invalid(face_id, _FOLD)
    return int(np.count_nonzero(exempt))


# ----------------------------------------------------------------------------- cone
class _ConeChart(_CylinderChart):
    """Chart ``(theta, h)`` of a cone about the module's own axis frame (module docstring, Cone):
    ``rho(h) = r0 + k h``, ``k = (r1 - r0) / (height |axis|)`` (the support's ``evaluate`` puts
    ``v height axis`` at radius ``(1 - v) r0 + v r1``)."""

    def __init__(self, support):
        self.o, self.e1, self.e2 = _axis_frame(support)
        axis = np.asarray(support.axis, dtype=float)
        length = dt._norm(axis)
        self.a = axis / length
        self.r0, self.r1 = float(support.radius_start), float(support.radius_end)
        self.r = max(self.r0, self.r1)
        self.k = (self.r1 - self.r0) / (float(support.height) * length)

    def charts(self, face_id, rings):
        out = self._unwrap(face_id, rings)
        h = np.concatenate([h for _, h in out])
        low = float((self.r0 + self.k * h).min())
        if not low > self.rho_rounding(float(np.abs(h).max())):      # the apex or the other nappe (to rounding)
            raise dt.DisplayDomainRefused(_apex_text(face_id))
        return self._join(face_id, rings, out, low)

    def offsets(self, points, chart):
        """``delta_i = |X_i - S(c_i)|`` computed with the module's frame."""
        theta, h = chart[:, 0], chart[:, 1]
        rho = self.r0 + self.k * h
        mapped = (self.o + (rho * np.cos(theta))[:, None] * self.e1 + (rho * np.sin(theta))[:, None] * self.e2
                  + h[:, None] * self.a)
        return np.sqrt(np.sum((points - mapped) ** 2, axis=1))

    def rho_rounding(self, height):
        """The rounding of a computed ``rho(h) = r0 + k h`` (``k`` itself a few roundings)."""
        return 8.0 * _EPS * (abs(self.r0) + abs(self.k) * height)

    def rounding(self, points, chart):
        """``A_face`` (module docstring, Cone)."""
        size = float(np.sqrt(np.sum(points ** 2, axis=1)).max())
        height = float(np.abs(chart[:, 1]).max())
        rho = float(np.abs(self.r0 + self.k * chart[:, 1]).max()) + self.rho_rounding(height)
        return (32.0 * _EPS * (dt._norm(self.o) + rho + height + size)
                + 2.0 * dt._FRAME_ERROR * (rho + height) + self.rho_rounding(height))

    def radius(self, chart):
        return self.r0 + self.k * chart[:, 1]

    def local_rounding(self, points, chart):
        """``A_loc`` (:meth:`_CylinderChart.local_rounding` with ``rho`` and its rounding)."""
        size = float(np.sqrt(np.sum((points - self.o) ** 2, axis=1)).max())
        height = float(np.abs(chart[:, 1]).max())
        rho = float(np.abs(self.r0 + self.k * chart[:, 1]).max()) + self.rho_rounding(height)
        return (32.0 * _EPS * (rho + height + size) + 2.0 * dt._FRAME_ERROR * (rho + height)
                + self.rho_rounding(height))


def _check_apex(face_id, model, support, cone, ring_points):
    """A trim reaching the apex (a ring point within N1's apex rule of an exact zero radius) is
    ``OUT_OF_DOMAIN``, checked before the azimuths (an apex point has none); a support radius
    ``rho(h) <= 0`` at a ring point (the other nappe) likewise, in :meth:`_ConeChart.charts`.
    Then ``rho > 0`` on the whole chart region (``rho`` is linear in ``h`` and every mesh point's
    ``h`` lies between ring points' ``h``)."""
    apexes = []
    origin = np.asarray(support.origin, dtype=float)
    if support.radius_start == 0.0:
        apexes.append(origin)
    if support.radius_end == 0.0:
        apexes.append(origin + support.height * np.asarray(support.axis, dtype=float))
    length = float(model.tolerance.length)
    at_apex = any(bool(np.any(np.sqrt(np.sum((ring_points - apex) ** 2, axis=1))
                              <= length + dt._ON_AXIS * max(1.0, float(np.abs(apex).max())))) for apex in apexes)
    if at_apex:
        raise dt.DisplayDomainRefused(_apex_text(face_id))


def _apex_text(face_id):
    return (f"face {dt._id_text(face_id)}: its trim reaches the cone's apex (a face lift needs a positive "
            f"support radius at every trim point)")


class _ConeBound:
    """The per-triangle paired-deviation bound ``E(T)`` of an assembled cone mesh (module
    docstring, Cone): the cylinder's bound with ``r`` replaced by the largest ``rho`` over the
    ring vertices under the triangle, plus the mixed term ``|k| |Cov_W(theta, h)|``."""

    def __init__(self, cone, k, ring_deltas, ring_rho, built, ring_rounding=None, local=False):
        chart, ends, f = built.chart, built.ends, built.fraction
        theta, h = chart[:, 0], chart[:, 1]
        n = len(chart)
        ja = np.concatenate((np.arange(k, dtype=np.int64), ends[:, 0]))
        jb = np.concatenate((np.arange(k, dtype=np.int64), ends[:, 1]))
        ff = np.concatenate((np.zeros(k), f))
        ta, tb, ha, hb = theta[ja], theta[jb], h[ja], h[jb]
        spread = ff * (1.0 - ff)
        self.vt = spread * (tb - ta) ** 2
        # a cut's own covariance on its chord, f (1 - f) dtheta dh (0 on a circle: dh = 0)
        self.cov = np.abs(spread * (tb - ta) * (hb - ha))
        kt = np.abs(theta - (ta + ff * (tb - ta))) + 4.0 * _EPS * (np.abs(ta) + np.abs(tb))
        kh = np.abs(h - (ha + ff * (hb - ha))) + 4.0 * _EPS * (np.abs(ha) + np.abs(hb))
        kt[:k], kh[:k] = 0.0, 0.0
        self.under = np.maximum(ring_deltas[ja], ring_deltas[jb])
        self.theta, self.h = theta, h
        self.kt_max = float(kt.max()) if n else 0.0
        self.kh_max = float(kh.max()) if n else 0.0
        self.slope = abs(cone.k)
        height = float(np.abs(h).max())
        # rho on every Taylor segment (ring vertices under the triangle to the mesh point, and on
        # to the true chart point kappa away): its ends' largest, computed, plus its rounding
        self.rho_under = np.maximum(ring_rho[ja], ring_rho[jb])
        self.rho_slack = cone.rho_rounding(height) + self.slope * self.kh_max
        rho_all = float(self.rho_under.max()) + self.rho_slack
        if local:                        # the origin-relative variant (see :class:`_CylinderBound`)
            self.rounding = ring_rounding if n == k else cone.local_rounding(built.points, chart)
            cut = built.cut_rounding
        else:
            size = float(np.sqrt(np.sum(built.points ** 2, axis=1)).max())
            self.rounding = cone.rounding(built.points, chart)
            cut = 16.0 * _EPS * size
        self.extra = (self.rounding + cut + rho_all * self.kt_max
                      + (1.0 + self.slope) * self.kh_max)

    def per_triangle(self, triangles):
        tt, th = self.theta[triangles], self.h[triangles]
        rho = np.max(self.rho_under[triangles], axis=1) + self.rho_slack
        half = 0.5 * rho
        slope = self.slope
        # H(w) = sum_i w_i P_i + sum_{i<j} w_i w_j D_ij bounds (rho / 2) Var_W(theta) + |k| |Cov_W|
        # (module docstring, Cone); its maximum over the triangle, exactly: the edges' closed form
        # and, where H is strictly concave on the plane, the critical point
        width = tt.max(axis=1) - tt.min(axis=1)
        height = th.max(axis=1) - th.min(axis=1)
        kt, kh = self.kt_max, self.kh_max
        # a cut's own chord: its variance at the chord's own largest rho, its covariance, and the
        # cross term |k| C_i |m_i - theta_bar| (rho varies along a skew chord)
        p = (0.5 * (self.rho_under[triangles] + self.rho_slack) * self.vt[triangles]
             + slope * self.cov[triangles] * (1.0 + (width + 2.0 * kt))[:, None])
        dt_ = tt[:, [1, 2, 2]] - tt[:, [0, 0, 1]]                   # edges 01, 02, 12
        dh_ = th[:, [1, 2, 2]] - th[:, [0, 0, 1]]
        d = half[:, None] * dt_ * dt_ + slope * np.abs(dt_ * dh_)
        d01, d02, d12 = d[:, 0], d[:, 1], d[:, 2]
        p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
        top = np.maximum.reduce([_quadratic_edge(p0, p1, d01), _quadratic_edge(p0, p2, d02),
                                 _quadratic_edge(p1, p2, d12)])
        m = d01 + d02 - d12
        det = 4.0 * d01 * d02 - m * m
        det_error = 16.0 * _EPS * (4.0 * d01 * d02 + m * m)
        g1, g2 = p1 - p0 + d01, p2 - p0 + d02
        flat = d01 + d02
        # strictly concave with a bounded condition number (<= 4e6): the critical point's own
        # rounding is far below the 1e-3 inside slack; otherwise the boundary plus the ridge term
        concave = (d01 > 0.0) & (det > det_error) & (det > 1.0e-6 * flat * flat)
        safe = np.where(concave, det, 1.0)
        a = (2.0 * d02 * g1 - m * g2) / safe
        b = (2.0 * d01 * g2 - m * g1) / safe
        slack = 1.0e-3
        inside = concave & (a >= -slack) & (b >= -slack) & (a + b <= 1.0 + slack)
        # a concave H's critical value is its global maximum (an upper bound wherever it lies);
        # a nearly singular H exceeds its boundary maximum by at most det / (D01 + D02)
        crest = np.where(inside, p0 + 0.5 * (a * g1 + b * g2), 0.0)
        ridge = np.where(~concave & (flat > 0.0), (np.maximum(det, 0.0) + det_error) / np.where(flat > 0.0, flat, 1.0),
                         0.0)
        h_max = (np.maximum(top, crest) + ridge) * (1.0 + 64.0 * _EPS) + 32.0 * _EPS * (d.sum(axis=1) + p.max(axis=1))
        widest = float(width.max()) if len(width) else 0.0
        # kappa: Var_w(m) <= Var_w(theta) + 2 kt dT + kt^2; |Cov_w(m)| <= |Cov_w| + (dT kh + dH kt) / 2 + kt kh
        spread = half * (2.0 * kt * widest + kt * kt) + slope * (0.5 * (kh * width + kt * height) + kt * kh)
        return (np.max(self.under[triangles], axis=1) * (1.0 + 8.0 * _EPS)
                + h_max + spread * (1.0 + 8.0 * _EPS) + self.extra) * (1.0 + 8.0 * _EPS), tt, width


def _quadratic_edge(vp, vq, d):
    """The exact maximum over ``l`` in [0, 1] of ``(1 - l) vp + l vq + l (1 - l) d`` (``d >= 0``):
    ``max(vp, vq) + max(d - |vq - vp|, 0)^2 / (4 d)``, before its rounding allowance."""
    gap = np.maximum(d - np.abs(vq - vp), 0.0)
    return np.maximum(vp, vq) + np.where(d > 0.0, gap * gap / np.where(d > 0.0, 4.0 * d, 1.0), 0.0)


def _lift_cone(model, face_id, tol, support, rings, extra):
    cone = _ConeChart(support)
    if tol.sag is not None:
        face_tolerance = float(tol.sag)
    else:
        a = max(ring.allowance for ring in rings)
        face_tolerance = cone.r * (tol.angle + a) ** 2 / 8 + float(model.tolerance.surface_residual)
    ring_points = np.vstack([ring.points for ring in rings])
    _check_apex(face_id, model, support, cone, ring_points)
    charts = cone.charts(face_id, rings)
    ring_chart = np.vstack(charts)
    k = len(ring_points)
    # (5) boundary chords: the two-point bound rho dT^2 / 8 + |k| dT dH / 4
    ring_deltas = cone.offsets(ring_points, ring_chart)
    ring_rho = cone.r0 + cone.k * ring_chart[:, 1]
    structure = _ring_structure(rings)
    following = structure[1]
    span = np.abs(ring_chart[following, 0] - ring_chart[:, 0])
    rise = np.abs(ring_chart[following, 1] - ring_chart[:, 1])
    rho = np.maximum(ring_rho, ring_rho[following]) + cone.rho_rounding(float(np.abs(ring_chart[:, 1]).max()))
    chord = (np.maximum(ring_deltas, ring_deltas[following]) * (1.0 + 8.0 * _EPS)
             + (rho * span ** 2 / 8.0 + abs(cone.k) * span * rise / 4.0) * (1.0 + 8.0 * _EPS)
             + cone.rounding(ring_points, ring_chart))
    worst = float(chord.max())
    local = None
    if not worst <= face_tolerance:                         # M3 class: the origin-relative bound
        local = (cone.local_offsets(ring_points, ring_chart), cone.local_rounding(ring_points, ring_chart))
        chord = np.minimum(chord, np.maximum(local[0], local[0][following]) * (1.0 + 8.0 * _EPS)
                           + (rho * span ** 2 / 8.0 + abs(cone.k) * span * rise / 4.0) * (1.0 + 8.0 * _EPS)
                           + local[1])
        worst = float(chord.max())
    if not worst <= face_tolerance:
        raise _refuse("EDGE_UNSUPPORTED", face_id, f"a boundary chord deviates from the cone by up to {worst!r}, "
                                                   f"above the face tolerance {face_tolerance!r}")
    # (6)-(9): N2, assembly, the deviation bound; failing slabs bisected, re-certified each round
    lines, added = extra, np.zeros(0)
    for attempt in range(_REFINE_ROUNDS + 1):
        mesh = triangulate_rings(charts, lines)
        built = _assemble(face_id, rings, ring_points, charts, ring_chart, mesh, structure)
        _check_identification(face_id, built, face_tolerance)
        bound = _ConeBound(cone, k, ring_deltas, ring_rho, built)
        per_triangle, tt, width = bound.per_triangle(built.triangles)
        failing = ~(per_triangle <= face_tolerance)
        if bool(np.any(failing)):
            if local is None:
                local = (cone.local_offsets(ring_points, ring_chart), cone.local_rounding(ring_points, ring_chart))
            per_triangle = np.minimum(per_triangle, _ConeBound(cone, k, local[0], ring_rho, built, local[1],
                                                               local=True).per_triangle(built.triangles)[0])
            failing = ~(per_triangle <= face_tolerance)
        if not bool(np.any(failing)):
            break
        if attempt == _REFINE_ROUNDS:
            raise _invalid(face_id, _DEVIATION)
        added = np.concatenate((added, _bisection(face_id, tt, failing, len(added))))
        lines = added if extra is None else np.concatenate((extra, added))
    points, chart, triangles, sense = built.points, built.chart, built.triangles, built.sense
    # Q7: the sign of n . (c(theta) - k a) over each footprint, exactly: both ends, and the
    # sinusoid's minimum where it may lie inside; rounding-thin exempt
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = _cross(d1, d2)
    size = np.sqrt(np.sum(normal ** 2, axis=1))
    longest = np.sqrt(np.maximum.reduce([np.sum(d1 ** 2, axis=1), np.sum(d2 ** 2, axis=1),
                                         np.sum((c - b) ** 2, axis=1)]))
    thin = ~(size > bound.rounding * longest)               # 3D width |nT| / longest <= A_face
    margin = (40.0 * _EPS * (1.0 + abs(cone.k)) * np.sqrt(np.sum(d1 ** 2, axis=1))
              * np.sqrt(np.sum(d2 ** 2, axis=1)))
    p = sense * (normal @ cone.e1)
    q = sense * (normal @ cone.e2)
    level = -sense * cone.k * (normal @ cone.a)
    lo, hi = tt.min(axis=1), tt.max(axis=1)
    cos_lo, sin_lo, cos_hi, sin_hi = np.cos(lo), np.sin(lo), np.cos(hi), np.sin(hi)
    at_lo = p * cos_lo + q * sin_lo + level
    at_hi = p * cos_hi + q * sin_hi + level
    # the minimum of p cos + q sin lies inside (lo, hi) only if the slope is falling at lo and
    # rising at hi (footprint < pi); "possibly inside" within rounding is taken as inside
    inside = (-p * sin_lo + q * cos_lo <= margin) & (-p * sin_hi + q * cos_hi >= -margin)
    bottom = level - np.hypot(p, q)
    positive = (at_lo > margin) & (at_hi > margin) & (width < math.pi) & (~inside | (bottom > margin))
    if not bool(np.all(thin | positive)):
        raise _invalid(face_id, _FOLD)
    exempt_count = _thin_decision(face_id, points, triangles, thin & ~positive, lo, hi, cone, cone.k, sense)
    deviation = float(per_triangle.max())
    trim = _trim_bound(face_id, model, tol, rings, built)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, trim, exempt_count)


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
        dt._check_support_domain(support)
        loops = (face.loop, *face.holes)
        _check_loops(face_id, loops)
        extra = _check_extra(_extra_x)
        if len(loops) > _slab.MAX_SLAB_RINGS:                  # N2's ring cap, before any sampling
            raise _slab._too_complex(f"{dt._id_text(len(loops))} rings (MAX_SLAB_RINGS = {_slab.MAX_SLAB_RINGS})")
        budget = [0]
        rings = [_ring(model, face_id, loop, tol, budget) for loop in loops]
        try:
            if isinstance(support, Plane):
                return _lift_plane(model, face_id, tol, support, rings, extra)
            if isinstance(support, Cylinder):
                return _lift_cylinder(model, face_id, tol, support, rings, extra)
            return _lift_cone(model, face_id, tol, support, rings, extra)
        except dt.DisplayTrimRefused:
            raise
        except (ArithmeticError, ValueError) as error:      # no raw arithmetic error escapes
            raise _invalid(face_id, _ARITHMETIC) from error
