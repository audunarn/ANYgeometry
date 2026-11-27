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
vertex (an N1 sample) stays ``TRIM_INVALID``.  Since owner decision C1 (2026-11-27) a merge needs
the merged cuts' LIFTED 3D points bitwise equal, and a cancellation both lifted ends of the
cancelled chain edge bitwise equal (:func:`_check_identified`); otherwise ``TRIM_INVALID``.

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
frame's tilt each round by at most a few eps of ``|d1| |d2|``).  A triangle this test cannot
certify is decided by a tight filter, else exactly (M1, :func:`_plane_orientation`); an exactly
zero normal is dropped from the mesh and counted in ``degenerate_count`` (C3, :func:`_drop_zero`).

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
that arc cannot contain its complement, longer than the interval).  Filter: ``sense n .
c(theta_end) > 40 eps |d1| |d2|`` at both ends certifies (an error bound, not a margin: the
rounded differences and cross product about 2.5 eps, the dots 2 eps, the frame ``sqrt(2) FE``
5.7 eps, cosine and sine (libm, assumed within a few ulps) and the sums 2.5 eps: about 13 eps
of ``|d1| |d2|``, a threefold slack); a footprint of pi or more is refused.  Every triangle the
filter cannot certify is decided by an exact enclosure (C3, :func:`_thin_curved`): kept only if
certifiably positive, dropped if its exact normal is zero, otherwise a fold.

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
lie inside (filter ``40 eps (1 + |k|) |d1| |d2|``, the budget above plus about 17 eps |k|; the
rest the exact enclosure).  A trim reaching the apex (or ``rho <= 0`` at a ring point) is ``OUT_OF_DOMAIN``.
Azimuth join (E2): on a cone the azimuth's rounding uses the smallest ``rho`` at the ring points.

Codex round 1 (owner decisions H1, M1-M4, L1).  H1: a D1 merge and the chain cancellation it
allows must hold in 3D: the merge distance ``d`` plus the cancelled chord length ``L`` was
added to ``trim_sag_bound`` (and, round 2, had to stay within ``A_face``; round 3, C1: replaced by
bitwise equality).  M1: a triangle the float sign test cannot certify is decided from its exact
normal (:func:`_plane_orientation`, :func:`_thin_curved`); r1/r2 exempted a zero normal or, on a
curved support, a footprint minimum within the support normal's rounding (round 3, C3: no
exemption).  M2: the cut points' rounding
(:func:`_cut_rounding`) is in ``trim_sag_bound``.  M3: where the world-scaled bound exceeds
T_face, an origin-relative bound (and on a plane an exact rational decision,
:meth:`_PlaneChart.exact_bound`) decides; a face the first bound accepts is unchanged.  M4: N2's
ring caps are enforced while the rings are sampled (:func:`_ring`).

Codex round 2 (owner decisions H1, H2, M1-M3, L1-L3, P1; docs "Codex round 2").  H1: an
identification was allowed only within the rounding allowance ``A_face`` (round 3, C1: only
between bitwise-equal points).  H2: on a cylinder or cone the rings' unwrapped azimuth extent must be certifiably
below 2 pi (:func:`_check_extent`; the chart -> 3D map is then injective).  M1/M2: the magnitude
domain (:func:`_check_magnitudes`, nonzero magnitudes at least ``2^-200``), bound norms that never
underflow (:func:`_norms`), every float orientation filter routing what it cannot certify to an
exact decision, the exact normal normalised by a power of two (:func:`_thin_curved`), exact ->
binary64 conversions rounded up (:func:`_sqrt_up`).  M3: the E2 threshold from support-frame
magnitudes plus the projected representation (:meth:`_CylinderChart.reach`).  L1: the exact plane
decision by squared comparison (:func:`_sign_two_roots`).  L3: budgets before the seam scan.  P1:
the planar exact orientation behind a tight vectorised filter (:func:`_plane_orientation`).

Codex round 3: the retry limit reached on three classes; owner contract decisions C1-C3
(2026-11-27, replacing r1/r2 H1 and the r1/r2 M1/M3 rules where they conflict; docs "Codex round
3").  C1: identifications are exact (bitwise-equal lifted points, :func:`_check_identified`, also
the coedge joins of :func:`_ring`); no allowance, ``trim_sag_bound`` without an identification
term.  C2: translation invariance is not a requirement (a placement-dependent refusal is allowed,
an accepted mesh must be certified); thresholds unchanged.  C3: no orientation margin: a triangle
is kept only if certifiably positive over its footprint (exact normal, an enclosure of the
support normal, :func:`_thin_curved`, :func:`_cos_sin`), a zero exact normal is dropped and
counted (:func:`_drop_zero`), anything else is a fold.  L2: the ring cap before the ring list is
built.

Codex round 4 (owner standing approval, the recommended option; docs "Codex round 4"): the class
"chart non-overlap plus per-triangle orientation does not imply that the LIFTED triangles do not
overlap in 3D" (r2 H2 via winding, r4 H1 via a notch narrower than the chord deviation) is closed by
a global certificate (:func:`_check_projection`): a projection P from 3D back to the chart (the
plane's orthogonal-frame projection; on a cylinder or cone (azimuth, axial coordinate)), defined on
the whole lifted mesh, maps it injectively: (i) every published triangle's P-image is positively
oriented, (ii) the P-image of the lifted boundary is a set of disjoint simple closed curves, (iii)
exactly one of them positively oriented (the trim's nesting).  A local homeomorphism of the chart
region whose boundary maps injectively onto curves of winding number at most 1 has degree at most 1,
so no two published triangles' P-images (hence lifted triangles) share an interior point.
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
#: the lift's magnitude domain (Codex N3 r2 M1/M2): every nonzero ring-point coordinate and support
#: parameter is at least this in magnitude (N1 bounds them above by MAX_DISPLAY_COORDINATE)
_MIN_MAGNITUDE = 2.0 ** -200
_TINY_NORMAL = 2.0 ** -1022
#: fixed-point bits of the exact fallback's cosine and sine enclosures (C3, :func:`_cos_sin`)
_COS_SIN_BITS = 128
#: a rational upper bound of sqrt(2) (C3: |cos t| + |sin t| <= sqrt(2))
_SQRT2_UP = Fraction(141422, 100000)

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
_IDENTIFY = ("merged cut points, or the ends of a boundary interval their merge cancels, are not bitwise equal in 3D "
             "(lift not certified: a gap in the trim)")
_ZERO_UNCOVERED = ("a zero-area lifted triangle is not covered by the triangles kept beside it (lift not certified: "
                   "it cannot be dropped)")
_EXTENT = "the trim's azimuth extent is not certifiably below 2 pi (the support would be covered twice)"


def _magnitude_text(face_id):
    return (f"face {dt._id_text(face_id)}: a nonzero coordinate or support parameter is below 2**-200 in magnitude "
            f"(the lift's domain)")


# ----------------------------------------------------------------------------- arithmetic helpers
def _norms(v):
    """Row norms of an (n, 3) array, ``sqrt(sum v^2)`` bit for bit wherever the sum of squares is
    normal; a row whose squares may underflow is scaled by an exact power of two first and its
    norm, if subnormal, raised by two ulps (an upper bound to a few eps, never 0 for a nonzero
    row: Codex N3 r2 M1/M2)."""
    s = np.sum(v * v, axis=1)
    out = np.sqrt(s)
    if not len(s) or s.min() >= 2.0 ** -960:          # the common case (NaN takes the slow path)
        return out
    low = ~(s >= 2.0 ** -960)
    if bool(np.any(low)):
        rows = v[low]
        top = np.max(np.abs(rows), axis=1)
        nonzero = top > 0.0
        _, e = np.frexp(np.where(nonzero, top, 1.0))
        scaled = np.ldexp(rows, -e[:, None])
        value = np.ldexp(np.sqrt(np.sum(scaled * scaled, axis=1)), e)
        value = np.where(value < _TINY_NORMAL, np.nextafter(np.nextafter(value, np.inf), np.inf), value)
        out[low] = np.where(nonzero, value, 0.0)
    return out


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def _sign_root(a, b, p) -> int:
    """The exact sign of ``a + b sqrt(p)`` (rationals, ``p >= 0``)."""
    sa, sb = _sign(a), (_sign(b) if p > 0 else 0)
    if sb == 0 or sa == sb:
        return sa if sa else sb
    if sa == 0:
        return sb
    t = _sign(a * a - b * b * p)
    return sa if t > 0 else (sb if t < 0 else 0)


def _sign_two_roots(a, b, p, c, q) -> int:
    """The exact sign of ``a + b sqrt(p) + c sqrt(q)`` (rationals, ``p, q >= 0``): the sign of
    ``g = a + b sqrt(p)`` against ``c sqrt(q)``, comparing squares where they differ in sign
    (``g^2 - c^2 q = (a^2 + b^2 p - c^2 q) + 2 a b sqrt(p)``)."""
    sg, sc = _sign_root(a, b, p), (_sign(c) if q > 0 else 0)
    if sc == 0 or sg == sc:
        return sg if sg else sc
    if sg == 0:
        return sc
    t = _sign_root(a * a + b * b * p - c * c * q, 2 * a * b, p)
    return sg if t > 0 else (sc if t < 0 else 0)


def _sqrt_up(x) -> float:
    """A binary64 upper bound of ``sqrt(x)`` (``x >= 0`` rational), within two ulps: the integer
    square root at 64 bits or more below the leading bit, then correctly rounded division and one
    step up; a positive ``x`` never gives 0 (Codex N3 r2 M2: no exact -> binary64 conversion of a
    bound rounds down)."""
    if x <= 0:
        return 0.0
    n, d = x.numerator, x.denominator
    shift = max(0, (130 - n.bit_length() + d.bit_length()) // 2 + 1)
    root = math.isqrt((n << (2 * shift)) // d) + 1          # sqrt(x) 2^shift < root
    value = root / (1 << shift) if shift < 1074 else float(Fraction(root, 1 << shift))
    return math.nextafter(value, math.inf)


def _exact_normals(points, rows):
    """The exact cross products ``(b - a) x (c - a)`` of the stored points of the triangles
    ``rows``, as Python integers over one common power-of-two scale ``q^2`` (the points as integers
    over ``q``: the sign and every ratio are the exact ones).  Returns ``(normals, edges2, q)`` with
    ``edges2`` each triangle's largest squared edge in the same integers (scale ``q^2``)."""
    used = np.unique(rows)
    values = points[used].ravel().tolist()
    ints, q = _integers(values)
    at = {int(u): (ints[3 * i], ints[3 * i + 1], ints[3 * i + 2]) for i, u in enumerate(used.tolist())}
    normals, edges2 = [], []
    for i, j, l in rows.tolist():
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = at[i], at[j], at[l]
        ux, uy, uz, vx, vy, vz = bx - ax, by - ay, bz - az, cx - ax, cy - ay, cz - az
        normals.append((uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx))
        wx, wy, wz = cx - bx, cy - by, cz - bz
        edges2.append(max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz, wx * wx + wy * wy + wz * wz))
    return normals, edges2, q


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
        if _bits(here[-1:]).tobytes() != _bits(there[:1]).tobytes():          # bit for bit (C1)
            raise _invalid(face_id, _GAP)
    return _Ring(np.vstack([p[:-1] for p in pieces]), np.vstack(sources), edges, allowance)


def _support_parameters(support):
    """The support's numbers the lift computes with (origin, vectors, radii, height)."""
    if isinstance(support, Plane):
        return [*support.origin, *support.u_vector, *support.v_vector]
    if isinstance(support, Cylinder):
        return [*support.origin, *support.axis, support.radius]
    return [*support.origin, *support.axis, support.radius_start, support.radius_end, support.height]


def _check_magnitudes(face_id, support, rings):
    """The lift's magnitude domain (Codex N3 r2 M1/M2, docs): every nonzero ring-point coordinate
    and support parameter is at least ``2^-200`` in magnitude (N1 bounds them above), else
    ``OUT_OF_DOMAIN``.  Then a nonzero difference of two such numbers is at least ``2^-253`` and a
    product of up to four of them is normal, so the float filters' and bounds' relative error
    analyses hold; what still underflows (derived quantities) is decided exactly or bounded by
    :func:`_norms`."""
    values = np.abs(np.concatenate([r.points.ravel() for r in rings]
                                   + [np.asarray(_support_parameters(support), dtype=float)]))
    if bool(np.any((values > 0.0) & (values < _MIN_MAGNITUDE))):
        raise dt.DisplayDomainRefused(_magnitude_text(face_id))


def _check_extent(face_id, charts, radius, reach):
    """H2 (Codex N3 r2): the combinatorial certificate is a statement about the chart, and the
    exact map ``S*`` of a cylinder or cone is injective only on a strip of azimuth width below
    2 pi.  Every mesh chart point's azimuth lies between the rings' (cut points on ring chords,
    slab lines strictly inside the rings' range), so the unwrapped extent of the rings, raised by
    twice E2's azimuth rounding (``radius`` the smallest support radius at the ring points,
    ``reach`` the largest ``|P - o|``), must be below 2 pi, else ``TRIM_WINDS_SUPPORT`` (a
    zero-winding loop over more than a turn).  Then distinct chart points have distinct support
    points, and the certified chart covering (no overlap, Q2) maps to a covering of the face's
    part of the support without a second sheet; the lifted mesh is within ``deviation_bound`` of
    it pointwise (docs, "Injectivity")."""
    theta = np.concatenate([c[:, 0] for c in charts])
    lo, hi = float(theta.min()), float(theta.max())
    rounding = 2.0 * _SNAP * (max(1.0, abs(lo), abs(hi)) + reach / radius)
    if not (hi - lo) + rounding < _TWO_PI * (1.0 - 4.0 * _EPS):
        raise _refuse("TRIM_WINDS_SUPPORT", face_id, _EXTENT)


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
    ``cancelled`` when given.  Returns the (mapped) triangles and ``(codes, n)``: the chain's
    directed edges ``tail * n + head`` (merged indices, the region on their left; Codex N3 r4)."""
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
        return _triangle_edges(face_id, chart, triangles, n, expected_codes), (expected_codes, n)
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
    return _triangle_edges(face_id, chart, triangles, n, expected_codes), (expected_codes, n)


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
    from its exact 3D chord (M2).  The D1 identifications move no lifted point (C1,
    :func:`_check_identified`)."""
    __slots__ = ("points", "triangles", "chart", "ends", "fraction", "source", "sense", "following",
                 "cut_rounding", "chain")


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
    if remap is None:
        triangles, (codes, size) = _combinatorial(face_id, rings, chart, triangles, ends, signs, structure=structure)
    else:
        merged_chart = chart[kept]
        cancelled = []
        triangles, (codes, size) = _combinatorial(face_id, rings, chart, triangles, ends, signs, remap, merged_chart,
                                                  structure, cancelled)
        _check_identified(face_id, np.vstack((ring_points, cuts)), representative, *cancelled[0])
        keep = kept[k:] - k
        chart, cuts, ends, fraction = merged_chart, cuts[keep], ends[keep], fraction[keep]
    out.points = np.vstack((ring_points, cuts))
    out.triangles, out.chart, out.ends, out.fraction, out.sense = triangles, chart, ends, fraction, sense
    out.chain = (codes // size, codes % size)                 # the boundary chain (Codex N3 r4)
    # a cut point's source is its chord's: the interval starting at the chord's first ring vertex
    source = np.vstack([r.source for r in rings])
    following = structure[1]
    out.following = following
    first = np.where(following[ends[:, 0]] == ends[:, 1], ends[:, 0], ends[:, 1])
    out.source = np.vstack((source, source[first]))
    return out


def _cut_rounding(start, finish):
    """``4 eps |(|A| + |B|)|`` over the cut points' chords, raised for its own rounding (M2)."""
    return float(_norms(np.abs(start) + np.abs(finish)).max()) * 4.0 * _EPS * (1.0 + 8.0 * _EPS)


def _bits(points):
    """The binary64 bit patterns of an (n, 3) array (bitwise comparison: ``-0.0 != 0.0``)."""
    return np.ascontiguousarray(points, dtype=np.float64).view(np.uint64)


def _check_identified(face_id, points, representative, tails, heads):
    """C1 (owner decision 2026-11-27, replacing the r1/r2 ``d + L`` allowance): identifications are
    exact.  A merged cut (D1: a cut point whose chart point equals another cut's) is identified with
    its representative only when their LIFTED 3D points (as computed, ``A + f (B - A)`` on their
    own chords) are bitwise equal; a chain edge cancelling with its reverse (the zero-width spike
    between two chords) only when both of its lifted endpoints are bitwise equal (its 3D length
    ``L`` is exactly 0).  Otherwise ``TRIM_INVALID`` (fixed text).  So no identification moves a
    lifted point and no cancelled piece has a 3D extent: the mesh boundary is the lifted chain of
    the ring chords, point for point, and ``trim_sag_bound`` carries no identification term.  No
    allowance enters (r3: an allowance from ``A_face`` reached 15646 on an ill-conditioned plane)."""
    bits = _bits(points)
    moved = np.flatnonzero(representative != np.arange(len(points)))
    if len(moved) and not np.array_equal(bits[moved], bits[representative[moved]]):
        raise _invalid(face_id, _IDENTIFY)
    if len(tails) and not np.array_equal(bits[tails], bits[heads]):
        raise _invalid(face_id, _IDENTIFY)


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
        exact = (eu[1] * ev[2] - eu[2] * ev[1], eu[2] * ev[0] - eu[0] * ev[2], eu[0] * ev[1] - eu[1] * ev[0])
        scale = math.lcm(*(x.denominator for x in exact))
        #: u x v exactly, as integers (a positive multiple): the sign reference of Q7 on a plane
        self.exact_normal = tuple(int(x * scale) for x in exact)
        # u x v in floats and a bound of each component's error (a 2 x 2 determinant of the stored
        # vectors: Shewchuk's ccwerrboundA without the differences, which are exact here)
        self.normal_f = np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])
        self.normal_error = _ORIENT_BOUND * np.array([abs(u[1] * v[2]) + abs(u[2] * v[1]), abs(u[2] * v[0]) + abs(u[0] * v[2]),
                                                      abs(u[0] * v[1]) + abs(u[1] * v[0])]) + _UNDERFLOW

    def chart(self, points):
        d = points - self.o
        e1, e2 = self.e1, self.e2
        return np.column_stack((d[:, 0] * e1[0] + d[:, 1] * e1[1] + d[:, 2] * e1[2],
                                d[:, 0] * e2[0] + d[:, 1] * e2[1] + d[:, 2] * e2[2]))

    def offsets(self, points, chart):
        """``delta_i = |X_i - S(c_i)|`` computed (the plane frame's own map)."""
        mapped = self.o + chart[:, :1] * self.e1 + chart[:, 1:] * self.e2
        return _norms(points - mapped)

    def rounding(self, points, chart):
        """``A_face`` (module docstring)."""
        size = float(_norms(points).max())
        extent = float(np.ptp(points, axis=0).max())
        reach = float(_norms(chart).max())
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
        delta = _norms(w - mapped)
        size = float(_norms(w).max())
        reach = float(_norms(chart).max())
        return delta * (1.0 + 8.0 * _EPS) + (16.0 * _EPS * (size + reach) + 2.0 * self.alpha * reach)

    def exact_bound(self, point, chart_point, limit):
        """The deviation ``D = |X - S*(c)|`` of one point decided against ``limit`` exactly (M3;
        Codex N3 r2 L1/M2).  With ``w = X - o``, ``U = u.u``, ``v' = v - (v.u / U) u``, ``V = v'.v'``
        (``e1* = u / sqrt(U)``, ``e2* = v' / sqrt(V)``, the exactly orthonormal frame), all
        rationals: ``D^2 = q + b1 sqrt(U) + b2 sqrt(V)``, ``q = |w|^2 + |c|^2``, ``b1 = -2 c_x (w.u) /
        U``, ``b2 = -2 c_y (w.v') / V``, and ``D <= limit`` iff ``q - limit^2 + b1 sqrt(U) + b2
        sqrt(V) <= 0``, decided exactly by comparing squares (:func:`_sign_two_roots`; no
        square-root enclosure, so equality is decided).  Returns an upper bound of ``D`` (at most
        ``limit``) when ``D <= limit``, else ``None``.  The bound: rational enclosures of the two
        square roots, refined until ``D^2``'s enclosure is within ``2^-60`` relative (or 8192
        bits), and :func:`_sqrt_up` of its upper end (rounded up: a positive distance never
        becomes 0)."""
        o, u, v = self._exact
        w = [Fraction(float(x)) - y for x, y in zip(point, o)]
        cx, cy = Fraction(float(chart_point[0])), Fraction(float(chart_point[1]))
        uu = sum(x * x for x in u)
        vu = sum(x * y for x, y in zip(v, u))
        vp = [x - vu / uu * y for x, y in zip(v, u)]
        vv = sum(x * x for x in vp)
        q = sum(x * x for x in w) + cx * cx + cy * cy
        b1 = -2 * cx * sum(x * y for x, y in zip(w, u)) / uu
        b2 = -2 * cy * sum(x * y for x, y in zip(w, vp)) / vv
        if _sign_two_roots(q - Fraction(limit) ** 2, b1, uu, b2, vv) > 0:
            return None
        for bits in (128, 512, 2048, 8192):
            lo, hi = q, q
            for b, base in ((b1, uu), (b2, vv)):
                low, high = _sqrt_enclosure(base, bits)
                lo += min(b * low, b * high)
                hi += max(b * low, b * high)
            if lo > 0 and hi - lo <= lo * Fraction(1, 1 << 60):
                break
        return min(float(limit), _sqrt_up(max(hi, Fraction(0))))


def _sqrt_enclosure(x, bits):
    """Rationals ``low <= sqrt(x) <= high`` (``x >= 0`` rational), ``high - low = 2^-bits``."""
    scale = 1 << bits
    root = math.isqrt(x.numerator * scale * scale // x.denominator)       # root <= sqrt(x) scale < root + 1
    return Fraction(root, scale), Fraction(root + 1, scale)


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


def _plane_orientation(face_id, plane, points, rows, sense):
    """Q7 on a plane for the triangles ``rows`` the first float filter cannot certify (M1; Codex N3
    r2 M1, P1): the sign of ``sense (nT . (u x v))``, ``nT`` the exact normal of the stored points,
    decided by a second, tight filter and else exactly.  Filter: each component of ``nT`` is a
    2 x 2 orientation of the stored points (Shewchuk's ``ccwerrboundA`` bounds its error, the
    differences' rounding included, by ``B (|l| + |r|)``), ``u x v`` is the float cross product
    within ``normal_error`` per component; then ``|computed - exact| <= sum (|N_f| B P + dN (|nT_f|
    + B P)) + 2 eps sum |N_f nT_f|`` (the dot's own rounding), raised by ``4 eps`` and an underflow
    slack (the undecided go exact, so the slack only routes).  Exactly (integers over a common
    power-of-two scale, :func:`_exact_normals`): positive certified, a zero normal returned (C3:
    dropped and counted by the caller, :func:`_drop_zero`), anything else a fold
    (``TRIM_INVALID``).  The reference ``u x v`` is exact here, so this is the exact sign, no
    margin.  Returns the positions (in ``rows``) of the zero normals."""
    a, b, c = points[rows[:, 0]], points[rows[:, 1]], points[rows[:, 2]]
    d1, d2 = b - a, c - a
    left = np.column_stack((d1[:, 1] * d2[:, 2], d1[:, 2] * d2[:, 0], d1[:, 0] * d2[:, 1]))
    right = np.column_stack((d1[:, 2] * d2[:, 1], d1[:, 0] * d2[:, 2], d1[:, 1] * d2[:, 0]))
    nt = left - right
    perm = _ORIENT_BOUND * (np.abs(left) + np.abs(right))
    nf, dn = plane.normal_f, plane.normal_error
    terms = nt * nf
    value = sense * terms.sum(axis=1)
    error = ((perm @ np.abs(nf) + (np.abs(nt) + perm) @ dn + 2.0 * _EPS * np.abs(terms).sum(axis=1))
             * (1.0 + 4.0 * _EPS) + _UNDERFLOW * (1.0 + float(np.abs(nf).sum())))
    if bool(np.any(value < -error)):
        raise _invalid(face_id, _FOLD)
    rest = np.flatnonzero(~(value > error))
    if not len(rest):
        return rest
    normals, _, _ = _exact_normals(points, rows[rest])
    reference = plane.exact_normal
    zero = []
    for position, n in zip(rest.tolist(), normals):
        dot = sense * (n[0] * reference[0] + n[1] * reference[1] + n[2] * reference[2])
        if dot > 0:
            continue
        if n != (0, 0, 0):
            raise _invalid(face_id, _FOLD)
        zero.append(position)
    return np.array(zero, dtype=np.int64)


def _cos_sin(theta):
    """Rational enclosures ``(cos_lo, cos_hi, sin_lo, sin_hi)`` of the cosine and sine of the float
    ``theta`` (exact as a rational), or ``None`` for ``|theta| > 64`` (undecided: the caller
    folds).  Fixed-point Taylor series in integers (C3: no libm accuracy assumption): with ``theta =
    N / D`` each term ``|theta|^j / j!`` is ``floor(|N|^j 2^b / (D^j j!))`` (``b`` =
    :data:`_COS_SIN_BITS`), less than one unit ``2^-b`` low; the series stops at the first term
    that is 0 past ``j > 2 |theta| + 2``, where every later term is below half its predecessor (the
    tail is below one unit).  So each enclosure is the sum plus or minus ``(terms + 2)`` units."""
    x = Fraction(theta)
    if abs(x) > 64:
        return None
    numerator, denominator = x.numerator, x.denominator
    base = abs(numerator)
    scale = 1 << _COS_SIN_BITS
    cosine = sine = 0
    power, factorial = 1, 1                      # |N|^j and D^j j!
    limit = 2 * abs(x) + 2
    j = 0
    while True:
        term = power * scale // factorial
        sign = -1 if (j // 2) % 2 else 1
        if j % 2:
            sine += (-sign if numerator < 0 else sign) * term
        else:
            cosine += sign * term
        if term == 0 and j > limit:
            break
        j += 1
        power *= base
        factorial *= denominator * j
    error = j + 3
    return (Fraction(cosine - error, scale), Fraction(cosine + error, scale),
            Fraction(sine - error, scale), Fraction(sine + error, scale))


def _thin_curved(points, rows, lo, hi, e1, e2, axis, k, sense, frame=None):
    """M1 / C3 on a cylinder (``k = 0``) or cone: the orientation of each triangle the float filter
    cannot certify, over its footprint ``[lo, hi]``, from the EXACT normal ``n`` of its stored
    points (integers, :func:`_exact_normals`) and an ENCLOSURE of the support normal (owner
    decision C3, 2026-11-27: no orientation margin).  The exact map ``S*`` uses an exactly
    orthonormal frame ``(e1*, e2*, a*)`` within ``FE = _FRAME_ERROR`` of the module's float frame
    ``(e1, e2, a)`` (each unit vector; ``a`` the normalised stored axis) and the exact slope ``k*``
    within ``kerr = 8 eps |k|`` of the computed ``k`` (a difference, a norm, a product and a
    quotient); its normal direction is ``c*(t) - k* a*``.  With ``g*(t) = sense n . (cos t e1* +
    sin t e2* - k* a*)`` and ``g(t) = P cos t + Q sin t + L`` (``P = sense n . e1``, ``Q = sense n .
    e2``, ``L = -sense k n . a``, exact rationals of the stored floats),

        |g*(t) - g(t)| <= E = |n| (sqrt(2) FE + (|k| + kerr) FE + kerr (1 + FE))

    (``|n|`` rounded up by an integer square root).  ``cos t`` and ``sin t`` at the footprint ends
    are enclosed rationally (:func:`_cos_sin`), so the lower ends below are exact lower bounds:

    - a footprint of pi or more (exactly: ``hi - lo`` against the float pi, which is below pi):
      folded;
    - both ends' lower bounds of ``g - E`` positive, else folded;
    - on a cylinder (``L = 0``) that suffices: ``{g > E}`` is an arc shorter than pi, and an
      interval narrower than pi with both ends in it lies in it;
    - on a cone also ``g - E > 0`` everywhere (``L - E > sqrt(P^2 + Q^2)``, decided by squares), or
      the minimum of ``g`` provably not inside: ``g'(lo) > 0`` (rising after its minimum, the next
      one is pi past the maximum ahead) or ``g'(hi) < 0`` (enclosed); else folded.

    Then ``g* >= g - E > 0`` over the footprint: certifiably positive.  An exactly zero normal is
    returned as zero (dropped and counted by the caller).  Everything is exact rational arithmetic
    on the stored floats; scale-invariant (Codex N3 r2 M1).  Returns ``(zero, folded)``.

    ``frame`` (Codex N3 r4, :func:`_projection_triangles`): three exact rational vectors used as
    ``(e1, e2, axis)`` with no frame error (``E = 0``): the sign of ``sense n . (cos t f1 + sin t
    f2)`` exactly, the Jacobian of the projection back to the chart (``k`` must then be 0)."""
    zero = np.zeros(len(rows), dtype=bool)
    folded = np.zeros(len(rows), dtype=bool)
    normals, _, _ = _exact_normals(points, np.asarray(rows))
    slope = Fraction(float(k))
    if frame is None:
        frame = [[Fraction(float(x)) for x in vector] for vector in (e1, e2, axis)]
        kerr = 8 * Fraction(_EPS) * abs(slope)
        fe = Fraction(dt._FRAME_ERROR)
        factor = _SQRT2_UP * fe + (abs(slope) + kerr) * fe + kerr * (1 + fe)
    else:
        factor = Fraction(0)
    half_turn = Fraction(math.pi)                       # the float pi, below pi
    ends = {}

    def enclosure(theta):
        if theta not in ends:
            ends[theta] = _cos_sin(theta)
        return ends[theta]
    for r, n in enumerate(normals):
        if n == (0, 0, 0):
            zero[r] = True
            continue
        a, b = float(lo[r]), float(hi[r])
        if not Fraction(b) - Fraction(a) < half_turn:
            folded[r] = True
            continue
        at_a, at_b = enclosure(a), enclosure(b)
        if at_a is None or at_b is None:
            folded[r] = True
            continue
        p = sense * sum(x * y for x, y in zip(n, frame[0]))
        q = sense * sum(x * y for x, y in zip(n, frame[1]))
        level = -sense * slope * sum(x * y for x, y in zip(n, frame[2]))
        size = math.isqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) + 1
        slack = size * factor
        lowest = []
        for cl, ch, sl, sh in (at_a, at_b):
            lowest.append((p * cl if p >= 0 else p * ch) + (q * sl if q >= 0 else q * sh) + level - slack)
        if not (lowest[0] > 0 and lowest[1] > 0):
            folded[r] = True
            continue
        if k == 0.0:
            continue
        floor_ = level - slack
        if floor_ > 0 and floor_ * floor_ > p * p + q * q:
            continue
        cl, ch, sl, sh = at_a
        rising = (-p * sh if p >= 0 else -p * sl) + (q * cl if q >= 0 else q * ch)            # g'(lo), lower end
        cl, ch, sl, sh = at_b
        falling = (-p * sl if p >= 0 else -p * sh) + (q * ch if q >= 0 else q * cl)           # g'(hi), upper end
        if rising > 0 or falling < 0:
            continue
        folded[r] = True
    return zero, folded


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
    points, chart, triangles, sense = built.points, built.chart, built.triangles, built.sense
    rounding = plane.rounding(points, chart)
    # (9) certificates
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = _cross(d1, d2)
    lengths = np.sqrt(np.sum(d1 ** 2, axis=1)) * np.sqrt(np.sum(d2 ** 2, axis=1))
    dot = normal @ (sense * plane.n)
    filter_bound = (16.0 * _EPS + 2.0 * plane.alpha) * lengths + _UNDERFLOW
    positive = dot > filter_bound                                 # the float filter certifies these
    # M1 / C3: every triangle the filter cannot certify is decided by a tight filter, else exactly;
    # a zero normal is dropped (the slack only routes: the undecided go exact)
    zero = np.zeros(0, dtype=np.int64)
    undecided = np.flatnonzero(~positive)
    if len(undecided):
        zero = undecided[_plane_orientation(face_id, plane, points, triangles[undecided], sense)]
    bounds = _plane_point_bounds(face_id, plane, points, chart, face_tolerance, rounding, "TRIM_INVALID",
                                 lambda worst: _DEVIATION)
    per_triangle = np.max(bounds[triangles], axis=1)
    deviation = float(per_triangle.max())
    if not deviation <= face_tolerance:
        raise _invalid(face_id, _DEVIATION)
    triangles = _drop_zero(face_id, points, triangles, zero)
    _check_projection(face_id, _PlaneProjection(plane, points), points, triangles, sense, built.chain)
    trim = _trim_bound(face_id, model, tol, rings, built)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, trim, len(zero))


def _drop_zero(face_id, points, triangles, zero):
    """C3: the triangles whose exact 3D normal is zero (``zero``: their rows) are dropped from the
    published mesh (the caller counts them in ``degenerate_count``).  A zero-normal triangle's three
    lifted points are collinear (or coincide), so its point set is its longest edge, and any two of
    its edges cover it.  It is dropped only if that set provably lies in the kept triangles: call an
    edge COVERED when its reverse is in a kept triangle, or in a dropped triangle already shown
    covered (the edge is then part of that triangle's set); a dropped triangle is covered when two
    of its edges are, or one longest edge (exact squared lengths) is; iterated to a fixpoint (ulp
    slabs give strips of zero-area triangles along one chord).  Any triangle left uncovered:
    ``TRIM_INVALID`` (fail closed).  Then the kept triangles contain, as point sets, every dropped
    one: the lifted surface and the lifted boundary chain (whose dropped chain edges lie in the kept
    mesh) are unchanged as sets, so a neighbour sharing the trim meets this mesh without a gap; the
    chart covering (Q2) is certified on the full chart mesh, and ``deviation_bound`` is taken over
    all of it, so the support over a dropped triangle's footprint is within ``deviation_bound`` of
    the kept mesh."""
    if not len(zero):
        return triangles
    n = len(points)
    drop = np.zeros(len(triangles), dtype=bool)
    drop[zero] = True
    kept = triangles[~drop]
    kept_edges = set(np.concatenate((kept[:, 0] * n + kept[:, 1], kept[:, 1] * n + kept[:, 2],
                                     kept[:, 2] * n + kept[:, 0])).tolist())
    rows = [triangles[row].tolist() for row in np.asarray(zero).tolist()]
    owner = {(t[i], t[(i + 1) % 3]): r for r, t in enumerate(rows) for i in range(3)}
    longest = []
    for t in rows:
        exact = [[Fraction(float(x)) for x in points[i]] for i in t]
        lengths = [sum((exact[(i + 1) % 3][d] - exact[i][d]) ** 2 for d in range(3)) for i in range(3)]
        longest.append([value == max(lengths) for value in lengths])
    covered = [False] * len(rows)
    changed = True
    while changed:
        changed = False
        for r, t in enumerate(rows):
            if covered[r]:
                continue
            ok = []
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                ok.append(b * n + a in kept_edges or ((b, a) in owner and covered[owner[(b, a)]]))
            if sum(ok) >= 2 or any(o and l for o, l in zip(ok, longest[r])):
                covered[r] = changed = True
    if not all(covered):
        raise _invalid(face_id, _ZERO_UNCOVERED)
    return kept


def _trim_bound(face_id, model, tol, rings, built):
    """Q6 ``trim_sag_bound``: the N1 chords' certified distance from the analytic trim (always
    reported, enforced only with tol.sag) plus every point N3 creates on the boundary: the cut
    points' rounding (M2).  The D1 identifications move no lifted point (C1), so they add
    nothing."""
    trim = max(_trim_sag(face_id, model, samples, tol, curve) for ring in rings for samples, curve in ring.edges)
    if built.cut_rounding:
        trim = (trim + built.cut_rounding) * (1.0 + 4.0 * _EPS)
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
        # the rounding of a computed azimuth: its own (|theta| ulps) and that of the projection
        # (:meth:`reach`: the difference P - o and the stored coordinates' representation in the
        # radial plane, not world reach: Codex N3 r2 M3), over the radius
        close = gaps <= _SNAP * (np.maximum(1.0, np.abs(ordered[1:])) + self.reach(rings) / radius)
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

    def reach(self, rings):
        """The magnitude whose ``eps / radius`` multiple is the rounding of a computed azimuth (E2,
        H2): the computation from the difference ``P - o`` (``max |P - o|``: support-frame), plus
        the stored coordinates' own representation (half an ulp of each coordinate of ``P`` and
        ``o``) projected onto the support's radial plane, ``sum_i (|P_i| + |o_i|) (|e1_i| +
        |e2_i|)``.  An axial translation along a coordinate axis leaves it unchanged (Codex N3 r2
        M3); a lateral one coarsens the stored points themselves, which the join must follow
        (ruling ends then differ by that representation: measured, a window face at a lateral
        1e3)."""
        cached = getattr(self, "_reach", None)
        if cached is not None and cached[0] is rings:
            return cached[1]
        weight = np.abs(self.e1) + np.abs(self.e2)
        local = max(float(np.abs(ring.points - self.o).max()) for ring in rings)
        stored = max(float(((np.abs(ring.points) + np.abs(self.o)) @ weight).max()) for ring in rings)
        self._reach = (rings, local + stored)
        return local + stored

    def offsets(self, points, chart):
        """``delta_i = |X_i - S(c_i)|`` computed with the module's frame."""
        theta, h = chart[:, 0], chart[:, 1]
        mapped = (self.o + (self.r * np.cos(theta))[:, None] * self.e1 + (self.r * np.sin(theta))[:, None] * self.e2
                  + h[:, None] * self.a)
        return _norms(points - mapped)

    def rounding(self, points, chart):
        """``A_face`` (module docstring)."""
        size = float(_norms(points).max())
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
        return _norms(points - self.o - mapped)

    def local_rounding(self, points, chart):
        """``A_loc``: ``A_face`` with ``|o| + max |X|`` replaced by ``W = max |X - o|`` (the
        difference ``X - o`` rounds by half an ulp of its result): the rounding of the cosine, sine,
        products and sums of ``S_loc``, of ``w - S_loc`` and of the frame; no world-scale term."""
        size = float(_norms(points - self.o).max())
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
            size = float(_norms(built.points).max())
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
    _check_extent(face_id, charts, r, cyl.reach(rings))
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
    # Q7: the sign over each footprint (both ends of its theta range); the float filter certifies
    # (its bound covers its own rounding, C3: no margin admits anything), the rest exact
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = _cross(d1, d2)
    filter_bound = 40.0 * _EPS * np.sqrt(np.sum(d1 ** 2, axis=1)) * np.sqrt(np.sum(d2 ** 2, axis=1)) + _UNDERFLOW
    p = normal @ cyl.e1
    q = normal @ cyl.e2
    lo, hi = tt.min(axis=1), tt.max(axis=1)
    at_lo = sense * (p * np.cos(lo) + q * np.sin(lo))
    at_hi = sense * (p * np.cos(hi) + q * np.sin(hi))
    positive = (at_lo > filter_bound) & (at_hi > filter_bound) & (width < math.pi)
    zero = _thin_decision(face_id, points, triangles, ~positive, lo, hi, cyl, 0.0, sense)
    deviation = float(per_triangle.max())
    triangles = _drop_zero(face_id, points, triangles, zero)
    _check_projection(face_id, _CurvedProjection(face_id, cyl, points, chart[:, 0]), points, triangles, sense,
                      built.chain)
    trim = _trim_bound(face_id, model, tol, rings, built)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, trim, len(zero))


def _thin_decision(face_id, points, triangles, undecided, lo, hi, chart, slope, sense):
    """M1 / C3: every triangle the float filter cannot certify (the filter's bound and underflow
    slack only route), decided by :func:`_thin_curved`; a triangle not certifiably positive refuses
    (``TRIM_INVALID`` fold), the zero normals are returned (rows of ``triangles``: dropped and
    counted, :func:`_drop_zero`).  No width limit, no margin (owner decision C3)."""
    rows = np.flatnonzero(undecided)
    if not len(rows):
        return rows
    zero, folded = _thin_curved(points, triangles[rows], lo[rows], hi[rows], chart.e1, chart.e2, chart.a, slope,
                                sense)
    if bool(np.any(folded)):
        raise _invalid(face_id, _FOLD)
    return rows[zero]


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
        return _norms(points - mapped)

    def rho_rounding(self, height):
        """The rounding of a computed ``rho(h) = r0 + k h`` (``k`` itself a few roundings)."""
        return 8.0 * _EPS * (abs(self.r0) + abs(self.k) * height)

    def rounding(self, points, chart):
        """``A_face`` (module docstring, Cone)."""
        size = float(_norms(points).max())
        height = float(np.abs(chart[:, 1]).max())
        rho = float(np.abs(self.r0 + self.k * chart[:, 1]).max()) + self.rho_rounding(height)
        return (32.0 * _EPS * (dt._norm(self.o) + rho + height + size)
                + 2.0 * dt._FRAME_ERROR * (rho + height) + self.rho_rounding(height))

    def radius(self, chart):
        return self.r0 + self.k * chart[:, 1]

    def local_rounding(self, points, chart):
        """``A_loc`` (:meth:`_CylinderChart.local_rounding` with ``rho`` and its rounding)."""
        size = float(_norms(points - self.o).max())
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
            size = float(_norms(built.points).max())
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
    _check_extent(face_id, charts, float((cone.r0 + cone.k * ring_chart[:, 1]).min()), cone.reach(rings))
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
    # sinusoid's minimum where it may lie inside; the float filter certifies, the rest exact (C3)
    if sense < 0:
        triangles = triangles[:, [0, 2, 1]]
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = _cross(d1, d2)
    filter_bound = (40.0 * _EPS * (1.0 + abs(cone.k)) * np.sqrt(np.sum(d1 ** 2, axis=1))
              * np.sqrt(np.sum(d2 ** 2, axis=1))) + _UNDERFLOW
    p = sense * (normal @ cone.e1)
    q = sense * (normal @ cone.e2)
    level = -sense * cone.k * (normal @ cone.a)
    lo, hi = tt.min(axis=1), tt.max(axis=1)
    cos_lo, sin_lo, cos_hi, sin_hi = np.cos(lo), np.sin(lo), np.cos(hi), np.sin(hi)
    at_lo = p * cos_lo + q * sin_lo + level
    at_hi = p * cos_hi + q * sin_hi + level
    # the minimum of p cos + q sin lies inside (lo, hi) only if the slope is falling at lo and
    # rising at hi (footprint < pi); "possibly inside" within rounding is taken as inside
    inside = (-p * sin_lo + q * cos_lo <= filter_bound) & (-p * sin_hi + q * cos_hi >= -filter_bound)
    bottom = level - np.hypot(p, q)
    positive = (at_lo > filter_bound) & (at_hi > filter_bound) & (width < math.pi) & (~inside | (bottom > filter_bound))
    zero = _thin_decision(face_id, points, triangles, ~positive, lo, hi, cone, cone.k, sense)
    deviation = float(per_triangle.max())
    triangles = _drop_zero(face_id, points, triangles, zero)
    _check_projection(face_id, _CurvedProjection(face_id, cone, points, chart[:, 0]), points, triangles, sense,
                      built.chain)
    trim = _trim_bound(face_id, model, tol, rings, built)
    return dt.FaceDisplayMesh(face_id, dt._frozen(points), dt._frozen(triangles), dt._frozen(chart.copy()), k,
                              dt._frozen(built.source), int(sense), face_tolerance, deviation, trim, len(zero))


# ----------------------------------------------------------------------------- global injectivity (Codex N3 r4)
#
# The chart certificate (Q2) and the per-triangle orientation (Q7) do not imply that the LIFTED
# triangles do not overlap in 3D (Codex r2 H2 via winding, r4 H1 via a notch narrower than the chord
# deviation).  So the lift also certifies that a projection P from 3D back to the chart maps the
# published mesh injectively (docs, "Global injectivity"):
#
#   Plane     P(X) = ((X - o).e1, (X - o).e2)                       (a linear map of rank 2)
#   Cylinder  P(X) = (azimuth of (x, y), z),  (x, y, z) = M (X - o)  (M the rows e1, e2, a)
#   Cone      the cylinder's P (valid away from the axis)
#
# with the module's FLOAT frame vectors taken as exact rationals (P is then an exactly defined
# function whose predicates are rational; it need not be orthonormal).  Certified:
#   (i)   every published triangle's P-image is positively oriented (the Jacobian of P on the flat
#         triangle: ``n . (e1 x e2)`` on a plane, ``n . (cos t f1 + sin t f2)`` with ``f1 = e2 x a``,
#         ``f2 = a x e1`` over the triangle's P-azimuth range on a cylinder or cone);
#   (ii)  the P-image of the lifted boundary chain is a set of pairwise disjoint simple closed
#         curves (non-adjacent boundary edges disjoint, adjacent ones meeting only at their shared
#         vertex; exact rational predicates after a sweep over the edges' bounding boxes);
#   (iii) exactly one of these loops is positively oriented (the outer one), the others negative.
# Then the degree of P on the chart region is at most 1 everywhere (the winding number of the
# boundary image), every published triangle's P-image counts +1 where it lies, so no two published
# triangles' P-images share an interior point; and two lifted triangles sharing an interior point
# in 3D would have P-images sharing one.

_NOT_SIMPLE = ("the lifted boundary is not simple: its projection back to the chart crosses or touches itself "
               "(lift not certified: lifted triangles may overlap)")
_NESTING = ("the lifted boundary's loops are not nested as the trim's: not exactly one positively oriented loop in "
            "the projection back to the chart (lift not certified)")
_PROJECTED = ("a lifted triangle's projection back to the chart is not positively oriented (lift not certified: "
              "lifted triangles may overlap)")
_NEAR_AXIS = ("a lifted point is too close to the support's axis, or too far from its chart point, for the "
              "projection back to the chart (lift not certified)")
#: the largest P-azimuth span of one lifted boundary edge (exact comparisons need pairs within pi)
_EDGE_SPAN = 0.5 * math.pi
#: the most box-overlapping boundary edge pairs the sweep enumerates (else OUT_OF_DOMAIN)
_MAX_BOX_PAIRS = 4_000_000


# Float filters below bound a product of inputs known to within absolute errors: ``|p~ q~ - p q| <=
# |p~| eq + |q~| ep + ep eq`` plus the product's own rounding (summed over the 1-norms of the xy
# parts); a sign is certain beyond the bound times ``1 + 64 eps`` plus an underflow slack (the
# bound's own rounding; what the filters cannot decide goes exact).


def _scale_bits(*arrays):
    """The smallest ``k`` (at most 1074) with every value of the arrays times ``2^k`` an integer:
    ``53 - e`` for the smallest ``frexp`` exponent ``e`` of a nonzero value (its ulp is ``2^(e -
    53)``)."""
    values = np.abs(np.concatenate([np.asarray(x, dtype=float).ravel() for x in arrays]))
    values = values[values > 0.0]
    if not len(values):
        return 0
    return int(min(1074, max(0, 53 - int(np.frexp(values)[1].min()))))


def _fixed(vector, k):
    """Binary64 values as exact integers over the scale ``2^k`` (:func:`_scale_bits`: every value an
    integer multiple of ``2^-k``): homogeneous exact predicates without Fraction normalisation."""
    out = []
    for x in vector:
        n, d = float(x).as_integer_ratio()
        if (1 << k) % d:                                    # cannot happen with _scale_bits' k
            raise ArithmeticError("a value is not on the exact scale")
        out.append(n * ((1 << k) // d))
    return out


def _rational_cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _cross2(p, q):
    return p[0] * q[1] - p[1] * q[0]


def _differences(points, ends, base, rows):
    """``M (X_end - X_base)`` per frame row (``rows`` the float frame vectors, exact as rationals:
    P's coordinates are linear in ``X``, so this is exactly the difference of the P coordinates),
    from the 3D difference ``fl(X_end - X_base)`` (half an ulp per component) and the dot's three
    products and two sums: each component within ``e = 4 eps |fl(X_end - X_base)|_1`` (raised for
    its own rounding), relative to the DIFFERENCE, not to the points' distance from the origin, so
    ulp-short boundary edges decide in floats.  ``ends`` may have a leading axis."""
    d = points[ends] - points[base]
    out = np.stack([d[..., 0] * r[0] + d[..., 1] * r[1] + d[..., 2] * r[2] for r in rows], axis=-1)
    e = 4.0 * _EPS * np.sum(np.abs(d), axis=-1) * (1.0 + 8.0 * _EPS) + _UNDERFLOW
    return out, e


class _PlaneProjection:
    """``P(X) = ((X - o).e1, (X - o).e2)``, the plane chart's float frame as exact rationals.
    ``u, v``: the computed P of every mesh point (the chart's own formula), ``err`` a bound of the
    absolute error of each (``fl(X - o)`` half an ulp, the dot's three products and two sums:
    within ``4 eps |fl(X - o)|_1``, raised for its own rounding)."""
    curved = False

    def __init__(self, plane, points):
        d = points - plane.o
        e1, e2 = plane.e1, plane.e2
        self.u = d[:, 0] * e1[0] + d[:, 1] * e1[1] + d[:, 2] * e1[2]
        self.v = d[:, 0] * e2[0] + d[:, 1] * e2[1] + d[:, 2] * e2[2]
        self.err = 4.0 * _EPS * np.sum(np.abs(d), axis=1) * (1.0 + 8.0 * _EPS) + _UNDERFLOW
        self.uv = np.column_stack((self.u, self.v))
        self.points, self._plane = points, plane
        #: P's kernel direction e1 x e2 (float; exactly in :attr:`normal`): the Jacobian sign is
        #: n . (e1 x e2)
        self.normal_f = _cross3(e1, e2)
        self._o = None
        self._cache = {}

    def _setup(self):
        plane = self._plane
        self._k = _scale_bits(self.points, plane.o)
        k = _scale_bits(plane.e1, plane.e2)
        self._o, self._e1, self._e2 = _fixed(plane.o, self._k), _fixed(plane.e1, k), _fixed(plane.e2, k)

    @property
    def normal(self):
        if self._o is None:
            self._setup()
        return _rational_cross(self._e1, self._e2)

    def exact(self, i):
        """The exact P of mesh point ``i`` (integers over a common power-of-two scale)."""
        hit = self._cache.get(i)
        if hit is None:
            if self._o is None:
                self._setup()
            d = [x - y for x, y in zip(_fixed(self.points[i], self._k), self._o)]
            hit = (sum(x * y for x, y in zip(d, self._e1)), sum(x * y for x, y in zip(d, self._e2)))
            self._cache[i] = hit
        return hit

    # (ii) adjacent boundary edges (A -> B -> C): meet only at B
    def adjacent_decided(self, a, b, c):
        """Float filter: True where ``BA`` and ``BC`` certainly meet only at ``B`` (not collinear, or
        pointing apart).  ``W[0] = U``, ``W[1] = V`` within ``e`` per component (the ends' err plus
        the difference's rounding); products bounded by the rule above, on 1-norms."""
        W, e = _differences(self.points, np.stack((a, c)), b, (self._plane.e1, self._plane.e2))
        aw = np.abs(W)
        nw = aw[..., 0] + aw[..., 1]
        orient = W[0, :, 0] * W[1, :, 1] - W[0, :, 1] * W[1, :, 0]
        dot = W[0, :, 0] * W[1, :, 0] + W[0, :, 1] * W[1, :, 1]
        bound = (nw[0] * e[1] + nw[1] * e[0] + 2.0 * e[0] * e[1] + 2.0 * _EPS * nw[0] * nw[1]) * (1.0 + 64.0 * _EPS)
        bound = bound + _UNDERFLOW
        # -orient (= (C - B) x (A - B)) is also the loop's turn at B (:meth:`turn`)
        return (np.abs(orient) > bound) | (dot < -bound), -orient, bound

    def apart(self, p, q, r, s):
        """Float filter for non-adjacent segments ``pq``, ``rs``: True where ``r`` and ``s`` lie
        certainly strictly on one side of the line ``pq``, or ``p`` and ``q`` of the line ``rs``."""
        rows = (self._plane.e1, self._plane.e2)

        def side(base, tip, x, y):
            W, e = _differences(self.points, np.stack((tip, x, y)), base, rows)
            aw = np.abs(W)
            nw = aw[..., 0] + aw[..., 1]
            o = W[0, :, 0] * W[1:, :, 1] - W[0, :, 1] * W[1:, :, 0]
            bound = (nw[0] * e[1:] + nw[1:] * e[0] + 2.0 * e[0] * e[1:] + 2.0 * _EPS * nw[0] * nw[1:]) * (
                1.0 + 64.0 * _EPS) + _UNDERFLOW
            return ((o[0] > bound[0]) & (o[1] > bound[1])) | ((o[0] < -bound[0]) & (o[1] < -bound[1]))
        return side(p, q, r, s) | side(r, s, p, q)

    def adjacent_meet(self, a, b, c):
        pb, pa, pc = self.exact(b), self.exact(a), self.exact(c)
        u = (pa[0] - pb[0], pa[1] - pb[1])
        v = (pc[0] - pb[0], pc[1] - pb[1])
        return _cross2(u, v) == 0 and u[0] * v[0] + u[1] * v[1] >= 0

    def meet(self, a, b, c, d):
        """The closed segments ``P(a) P(b)`` and ``P(c) P(d)`` (no shared index) intersect."""
        pa, pb, pc, pd = self.exact(a), self.exact(b), self.exact(c), self.exact(d)
        o1 = _sign(_orient_points(pa, pb, pc))
        o2 = _sign(_orient_points(pa, pb, pd))
        o3 = _sign(_orient_points(pc, pd, pa))
        o4 = _sign(_orient_points(pc, pd, pb))
        if o1 * o2 > 0 or o3 * o4 > 0:
            return False
        if o1 or o2 or o3 or o4:
            return True
        # collinear (both segments non-degenerate: a zero-length edge is refused as adjacent first):
        # their projections onto the line overlap
        axis = 0 if pa[0] != pb[0] else 1
        lo1, hi1 = sorted((pa[axis], pb[axis]))
        lo2, hi2 = sorted((pc[axis], pd[axis]))
        return lo1 <= hi2 and lo2 <= hi1

    # (iii) orientation of a loop at its lexicographically smallest vertex
    def smaller(self, i, j):
        return self.exact(i) < self.exact(j)

    def turn(self, prev, vertex, nxt):
        """The sign of the loop's turn at its lexicographic minimum: ``(P(next) - P(v)) x (P(prev) -
        P(v))`` (positive: counter-clockwise)."""
        pv, pw, pu = self.exact(vertex), self.exact(nxt), self.exact(prev)
        return _sign(_orient_points(pv, pw, pu))


def _orient_points(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


class _CurvedProjection:
    """``P(X) = (azimuth of (x, y), z)`` with ``(x, y, z) = M (X - o)``, ``M`` the rows ``e1, e2, a``
    of the module's float axis frame as exact rationals (the cylinder's and the cone's P).  ``u`` is
    the computed azimuth of every mesh point, unwrapped onto its chart azimuth's sheet; ``v = z``.
    Error bounds: ``x, y, z`` within ``err = 4 eps |fl(X - o)|_1`` each; the azimuth within ``aerr =
    2.2 err / rho + 8 eps (1 + |u|) + |turns| (2.5e-16 + 8 eps)`` (the angle subtended by the
    ``sqrt(2) err`` disk at distance ``rho >= 16 err``, the libm ``atan2`` within a few ulps (the
    module's documented assumption, as for the float filters), the unwrapping by the float 2 pi).
    A point with ``rho <= 16 err``, or whose azimuth is more than pi / 4 from its chart azimuth, is
    refused (``TRIM_INVALID``): P is certified only away from the axis."""
    curved = True

    def __init__(self, face_id, chart, points, theta):
        d = points - chart.o
        e1, e2, a = chart.e1, chart.e2, chart.a
        x = d[:, 0] * e1[0] + d[:, 1] * e1[1] + d[:, 2] * e1[2]
        y = d[:, 0] * e2[0] + d[:, 1] * e2[1] + d[:, 2] * e2[2]
        z = d[:, 0] * a[0] + d[:, 1] * a[1] + d[:, 2] * a[2]
        err = 4.0 * _EPS * np.sum(np.abs(d), axis=1) * (1.0 + 8.0 * _EPS) + _UNDERFLOW
        rho = np.hypot(x, y)
        if not bool(np.all(rho > 16.0 * err)):
            raise _invalid(face_id, _NEAR_AXIS)
        raw = np.arctan2(y, x)
        turns = np.round((theta - raw) / _TWO_PI)
        u = raw + turns * _TWO_PI
        if not bool(np.all(np.abs(u - theta) <= 0.25 * math.pi)):
            raise _invalid(face_id, _NEAR_AXIS)
        self.xyz, self.u, self.v, self.err = np.column_stack((x, y, z)), u, z, err
        self.aerr = (2.2 * err / rho + 8.0 * _EPS * (1.0 + np.abs(u))
                     + np.abs(turns) * (2.5e-16 + 8.0 * _EPS)) * (1.0 + 8.0 * _EPS)
        self.points, self._chart = points, chart
        #: the Jacobian frame f1 = e2 x a, f2 = a x e1 (float; exactly in :attr:`frame`)
        self.f1, self.f2 = _cross3(e2, a), _cross3(a, e1)
        self._m = None
        self._cache = {}

    def _setup(self):
        chart = self._chart
        self._k = _scale_bits(self.points, chart.o)
        k = _scale_bits(chart.e1, chart.e2, chart.a)
        self._o, self._m = _fixed(chart.o, self._k), [_fixed(chart.e1, k), _fixed(chart.e2, k), _fixed(chart.a, k)]

    @property
    def frame(self):
        """``(f1, f2, a)`` exactly (integers over a common power-of-two scale)."""
        if self._m is None:
            self._setup()
        return [_rational_cross(self._m[1], self._m[2]), _rational_cross(self._m[2], self._m[0]), self._m[2]]

    def exact(self, i):
        hit = self._cache.get(i)
        if hit is None:
            if self._m is None:
                self._setup()
            d = [x - y for x, y in zip(_fixed(self.points[i], self._k), self._o)]
            hit = tuple(sum(x * y for x, y in zip(d, row)) for row in self._m)
            self._cache[i] = hit
        return hit

    def adjacent_decided(self, a, b, c):
        """Float filter for ``A -> B -> C``: True where the two edges' P-images certainly meet only at
        ``B``: they leave ``B`` to opposite sides in azimuth (``cross2(B, BA)``, ``cross2(B, BC)``) or
        in height, or ``|alpha'| > min(|Uz|, |Vz|) |beta|`` (docs: a second meeting needs ``0 < s, t
        <= 1``)."""
        # W[0] = U = A - B, W[1] = V = C - B (M-space floats), each component within e (the ends'
        # err plus the difference's own rounding); products bounded by |p q - p~ q~| <= |p| eq +
        # |q| ep + ep eq plus their rounding (the rule above, on 1-norms of the xy parts)
        xyz, err = self.xyz, self.err
        B = xyz[b]
        chart = self._chart
        W, e = _differences(self.points, np.stack((a, c)), b, (chart.e1, chart.e2, chart.a))
        eb = err[b]
        aw = np.abs(W)
        nw = aw[..., 0] + aw[..., 1]
        nb = np.abs(B[:, 0]) + np.abs(B[:, 1])
        cross = B[:, 0] * W[..., 1] - B[:, 1] * W[..., 0]                    # cross2(B, U), cross2(B, V)
        ecross = nb * e + nw * eb + 2.0 * eb * e + 2.0 * _EPS * nb * nw
        z, ez = W[..., 2], e
        margin = 1.0 + 64.0 * _EPS
        cpos, cneg = cross > ecross * margin + _UNDERFLOW, cross < -(ecross * margin + _UNDERFLOW)
        zpos, zneg = z > ez * margin + _UNDERFLOW, z < -(ez * margin + _UNDERFLOW)
        apart = (cpos[0] & cneg[1]) | (cneg[0] & cpos[1]) | (zpos[0] & zneg[1]) | (zneg[0] & zpos[1])
        # alpha' = Uz cross2(B, V) - Vz cross2(B, U); beta = cross2(U, V)
        az_, ac = np.abs(z), np.abs(cross)
        alpha = z[0] * cross[1] - z[1] * cross[0]
        ealpha = (az_[0] * ecross[1] + ac[1] * ez[0] + ez[0] * ecross[1] + az_[1] * ecross[0] + ac[0] * ez[1]
                  + ez[1] * ecross[0] + 2.0 * _EPS * (az_[0] * ac[1] + az_[1] * ac[0]))
        beta = W[0, :, 0] * W[1, :, 1] - W[0, :, 1] * W[1, :, 0]
        ebeta = nw[0] * e[1] + nw[1] * e[0] + 2.0 * e[0] * e[1] + 2.0 * _EPS * nw[0] * nw[1]
        reach = np.minimum(az_[0] + ez[0], az_[1] + ez[1]) * (np.abs(beta) + ebeta) * (1.0 + 4.0 * _EPS)
        gap = np.abs(alpha) - reach
        far = gap > (ealpha + _EPS * (np.abs(alpha) + reach)) * margin + _UNDERFLOW
        # alpha' is also the loop's turn at B (:meth:`turn` with prev A, next C)
        return apart | far, alpha, ealpha * margin + _UNDERFLOW

    def apart(self, p, q, r, s):
        """No float filter for non-adjacent curve pairs (rare after the boxes): all go exact."""
        return np.zeros(len(p), dtype=bool)

    def adjacent_meet(self, a, b, c):
        """Exactly: the P-images of ``BA`` and ``BC`` meet at a point other than ``P(B)``.  With ``U =
        A - B``, ``V = C - B`` (M-space), a second meeting is ``B + s U``, ``B + t V`` at the same
        height (``s Uz = t Vz``) and on the same ray (``cross2 = 0``; the edges span less than pi / 2,
        so a zero cross product is the same azimuth), ``0 < s, t <= 1``."""
        pb, pa, pc = self.exact(b), self.exact(a), self.exact(c)
        u = [x - y for x, y in zip(pa, pb)]
        v = [x - y for x, y in zip(pc, pb)]
        cu, cv = _cross2(pb, u), _cross2(pb, v)
        if u[2] == 0 and v[2] == 0:
            return cu == 0 or cv == 0 or _sign(cu) == _sign(cv)
        if u[2] == 0 or v[2] == 0:
            return (cu if u[2] == 0 else cv) == 0                  # a horizontal radial edge collapses
        if _sign(u[2]) != _sign(v[2]):
            return False
        alpha = u[2] * cv - v[2] * cu
        beta = _cross2(u, v)
        if beta == 0:
            return alpha == 0
        s = Fraction(-alpha, u[2] * beta)
        t = Fraction(-alpha, v[2] * beta)
        return 0 < s <= 1 and 0 < t <= 1

    def meet(self, a, b, c, d):
        """Exactly: the P-images of the segments ``ab`` and ``cd`` (no shared index) meet: a common
        height and a zero ``cross2`` (same azimuth: the pair's azimuths lie within pi)."""
        p1, q1, p2, q2 = self.exact(a), self.exact(b), self.exact(c), self.exact(d)
        u = [x - y for x, y in zip(q1, p1)]
        v = [x - y for x, y in zip(q2, p2)]
        if u[2] == 0 and v[2] == 0:
            if p1[2] != p2[2]:
                return False
            return _angular_overlap(p1, q1, p2, q2)
        if v[2] == 0:
            p1, u, p2, v = p2, v, p1, u
        # t(s) = t0 + t1 s from the heights; the s-interval where 0 <= t <= 1
        t0, t1 = Fraction(p1[2] - p2[2], v[2]), Fraction(u[2], v[2])
        lo, hi = Fraction(0), Fraction(1)
        if t1 == 0:
            if not 0 <= t0 <= 1:
                return False
        else:
            ends = sorted(((-t0) / t1, (1 - t0) / t1))
            lo, hi = max(lo, ends[0]), min(hi, ends[1])
            if lo > hi:
                return False
        w = [p2[0] + t0 * v[0], p2[1] + t0 * v[1]]
        c0 = _cross2(p1, w)
        c1 = _cross2(u, w) + t1 * _cross2(p1, v)
        c2 = t1 * _cross2(u, v)

        def q(s):
            return c0 + s * (c1 + s * c2)
        ql, qh = q(lo), q(hi)
        if ql == 0 or qh == 0 or _sign(ql) != _sign(qh):
            return True
        if c2 != 0:
            vertex = -c1 / (2 * c2)
            if lo < vertex < hi and _sign(q(vertex)) != _sign(ql):
                return True
        return False

    def smaller(self, i, j):
        """Lexicographic (azimuth, height) order of two mesh points whose azimuths are close."""
        pi_, pj = self.exact(i), self.exact(j)
        c = _cross2(pi_, pj)
        if c != 0:
            return c > 0
        return pi_[2] < pj[2]

    def turn(self, prev, vertex, nxt):
        """The sign of ``t_out x t_back`` at the loop's lexicographic minimum, ``t = (cross2(v, W) /
        |v|^2, Wz)`` the P-image tangent of the edge toward ``W``."""
        pv, pw, pu = self.exact(vertex), self.exact(nxt), self.exact(prev)
        w = [x - y for x, y in zip(pw, pv)]
        u = [x - y for x, y in zip(pu, pv)]
        return _sign(_cross2(pv, w) * u[2] - w[2] * _cross2(pv, u))


def _angular_overlap(p1, q1, p2, q2):
    """Two azimuth intervals (of horizontal edges at one height, each under pi / 2 and within pi of
    each other) intersect: exact ``cross2`` comparisons."""
    def ordered(p, q):
        return (p, q) if _cross2(p, q) >= 0 else (q, p)
    a1, b1 = ordered(p1, q1)
    a2, b2 = ordered(p2, q2)
    return _cross2(a1, b2) >= 0 and _cross2(a2, b1) >= 0


def _projection_triangles(face_id, proj, points, triangles, sense):
    """(i): every published triangle's P-image positively oriented (``triangles`` in the published
    orientation; the N2 orientation's normal is ``sense`` times theirs).  Float filter, else exact
    (C3 style, no margin)."""
    if not len(triangles):
        return
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    d1, d2 = b - a, c - a
    normal = _cross(d1, d2)
    lengths = np.sqrt(np.sum(d1 ** 2, axis=1)) * np.sqrt(np.sum(d2 ** 2, axis=1))
    if not proj.curved:
        # |nT_f - nT| about 3 sqrt(3) eps |d1||d2|, |N_f - N| 2 sqrt(3) eps, the dot 3 eps: 24 eps
        value = sense * (normal @ proj.normal_f)
        undecided = np.flatnonzero(~(value > 24.0 * _EPS * lengths + _UNDERFLOW))
        if not len(undecided):
            return
        normals, _, _ = _exact_normals(points, triangles[undecided])
        reference = proj.normal
        for n in normals:
            if not sense * (n[0] * reference[0] + n[1] * reference[1] + n[2] * reference[2]) > 0:
                raise _invalid(face_id, _PROJECTED)
        return
    # curved: the triangle's P-azimuth range, its vertices' azimuths widened by their bounds
    u, aerr = proj.u[triangles], proj.aerr[triangles]
    lo = np.nextafter(np.min(u - aerr, axis=1), -np.inf)
    hi = np.nextafter(np.max(u + aerr, axis=1), np.inf)
    if not bool(np.all(hi - lo < math.pi)):
        raise _invalid(face_id, _PROJECTED)
    p = sense * (normal @ proj.f1)
    q = sense * (normal @ proj.f2)
    bound = 40.0 * _EPS * lengths + _UNDERFLOW
    positive = ((p * np.cos(lo) + q * np.sin(lo) > bound) & (p * np.cos(hi) + q * np.sin(hi) > bound))
    rows = np.flatnonzero(~positive)
    if not len(rows):
        return
    # exact: the vertices' exact azimuths inside [lo, hi] (no libm assumption), then the sign of
    # n . (cos t f1 + sin t f2) over [lo, hi] exactly (:func:`_thin_curved`, E = 0)
    for r in rows.tolist():
        at_lo, at_hi = _cos_sin(float(lo[r])), _cos_sin(float(hi[r]))
        if at_lo is None or at_hi is None:
            raise _invalid(face_id, _PROJECTED)
        for i in triangles[r].tolist():
            x, y, _ = proj.exact(i)
            cl, ch, sl, sh = at_lo
            after_lo = (y * cl if y >= 0 else y * ch) - (x * sh if x >= 0 else x * sl)        # cross2(c(lo), v)
            cl, ch, sl, sh = at_hi
            before_hi = (x * sl if x >= 0 else x * sh) - (y * ch if y >= 0 else y * cl)       # cross2(v, c(hi))
            if not (after_lo > 0 and before_hi > 0):
                raise _invalid(face_id, _PROJECTED)
    zero, folded = _thin_curved(points, triangles[rows], lo[rows], hi[rows], None, None, None, 0.0, sense,
                                frame=proj.frame)
    if bool(np.any(folded | zero)):
        raise _invalid(face_id, _PROJECTED)


def _box_pairs(lo_u, hi_u, lo_v, hi_v):
    """The pairs of boxes that overlap (closed), by a sweep over the boxes sorted by ``lo_u``."""
    m = len(lo_u)
    order = np.argsort(lo_u, kind="stable")
    slo, shi = lo_u[order], hi_u[order]
    end = np.searchsorted(slo, shi, side="right")
    counts = np.maximum(end - np.arange(m) - 1, 0)
    total = int(counts.sum())
    if total > _MAX_BOX_PAIRS:
        raise _slab._too_complex(f"{total} overlapping boundary edge boxes (more than {_MAX_BOX_PAIRS})")
    first = np.repeat(np.arange(m), counts)
    offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    i, j = order[first], order[first + 1 + offset]
    keep = (lo_v[i] <= hi_v[j]) & (lo_v[j] <= hi_v[i])
    return i[keep], j[keep]


def _check_boundary(face_id, proj, tails, heads):
    """(ii) and (iii) on the boundary chain ``tails[e] -> heads[e]`` (mesh indices, the region on
    the left in the chart): every vertex once a tail and once a head; adjacent edges meet only at
    their shared vertex, non-adjacent ones not at all (P-images; boxes, float filters, exact
    predicates); exactly one positively oriented loop, and it holds vertex 0 if the chain does
    (the outer ring's first vertex).  ``TRIM_INVALID`` otherwise."""
    tails, heads = np.asarray(tails, dtype=np.int64), np.asarray(heads, dtype=np.int64)
    u, v, err = proj.u, proj.v, proj.err
    # chain vertices whose lifted points are bitwise equal are one point (C3: two cuts an ulp apart
    # in the chart lift onto one point; the zero-length chain edge between them lies in a dropped
    # zero-area triangle): identify them and drop the zero-length edges.  A genuine self-touch then
    # shows as a vertex used twice (refused below).  Equal points have equal computed P: only
    # P-equal pairs are compared bitwise.
    order = np.argsort(u[tails], kind="stable")
    ut, vt = u[tails][order], v[tails][order]
    if bool(np.any((ut[1:] == ut[:-1]) & (vt[1:] == vt[:-1]))):
        bits = _bits(proj.points[tails])
        order = np.lexsort((bits[:, 2], bits[:, 1], bits[:, 0]))
        same = np.all(bits[order][1:] == bits[order][:-1], axis=1)
        if bool(np.any(same)):
            canon = np.arange(len(proj.points), dtype=np.int64)
            group = np.concatenate(([0], np.cumsum(~same)))
            first = order[np.concatenate(([0], np.flatnonzero(~same) + 1))]
            canon[tails[order]] = tails[first[group]]
            tails, heads = canon[tails], canon[heads]
            keep = tails != heads
            tails, heads = tails[keep], heads[keep]
    m = len(tails)
    order = np.argsort(tails, kind="stable")
    ordered_tails = tails[order]
    if m < 3 or bool(np.any(ordered_tails[1:] == ordered_tails[:-1])) or not np.array_equal(ordered_tails,
                                                                                            np.sort(heads)):
        raise _invalid(face_id, _NOT_SIMPLE)
    following = order[np.searchsorted(ordered_tails, heads)]          # the edge after edge e
    ends = np.stack((tails, heads))
    eu = proj.aerr if proj.curved else err
    ue, eue, ve, eve = u[ends], eu[ends], v[ends], err[ends]
    if proj.curved and not bool(np.all(np.abs(ue[0] - ue[1]) + eue[0] + eue[1] < _EDGE_SPAN)):
        raise _invalid(face_id, _NOT_SIMPLE)
    lo_u = np.nextafter(np.min(ue - eue, axis=0), -np.inf)
    hi_u = np.nextafter(np.max(ue + eue, axis=0), np.inf)
    lo_v = np.nextafter(np.min(ve - eve, axis=0), -np.inf)
    hi_v = np.nextafter(np.max(ve + eve, axis=0), np.inf)
    # adjacent edges e = (A -> B) and following[e] = (B -> C)
    a, b, c = tails, heads, heads[following]
    decided, turns, turn_errors = proj.adjacent_decided(a, b, c)
    for e in np.flatnonzero(~decided).tolist():
        if proj.adjacent_meet(int(a[e]), int(b[e]), int(c[e])):
            raise _invalid(face_id, _NOT_SIMPLE)
    # non-adjacent edges whose boxes overlap (all pairs for a short chain, else the sweep)
    if m <= 12:
        lu, hu, lv, hv, nxt = lo_u.tolist(), hi_u.tolist(), lo_v.tolist(), hi_v.tolist(), following.tolist()
        near = [(e, f) for e in range(m) for f in range(e + 1, m)
                if lu[f] <= hu[e] and lu[e] <= hu[f] and lv[f] <= hv[e] and lv[e] <= hv[f] and nxt[e] != f and nxt[f] != e]
    else:
        first, second = _box_pairs(lo_u, hi_u, lo_v, hi_v)
        keep = (following[first] != second) & (following[second] != first)
        near = list(zip(first[keep].tolist(), second[keep].tolist()))
    if near:
        pair = np.array(near, dtype=np.int64)
        e_, f_ = pair[:, 0], pair[:, 1]
        near = pair[~proj.apart(tails[e_], heads[e_], tails[f_], heads[f_])].tolist()
    for e, f in near:
        if proj.meet(int(tails[e]), int(heads[e]), int(tails[f]), int(heads[f])):
            raise _invalid(face_id, _NOT_SIMPLE)
    # (iii) the loops: each edge's smallest edge index along its cycle (a walk for short chains,
    # pointer jumping for long ones)
    if m <= 256:
        label, nxt = [-1] * m, following.tolist()
        for start in range(m):
            if label[start] < 0:
                e = start
                while label[e] < 0:
                    label[e] = start
                    e = nxt[e]
        label = np.array(label)
    else:
        label, jump = np.arange(m), following.copy()
        for _ in range(int(m).bit_length() + 1):
            label = np.minimum(label, label[jump])
            jump = jump[jump]
    previous = np.empty(m, dtype=np.int64)
    previous[following] = np.arange(m)
    positive = []
    loops = [0] if bool(np.all(label == 0)) else np.unique(label).tolist()
    for loop in loops:
        edges = np.flatnonzero(label == loop)
        key, key_err = u[tails[edges]], eu[tails[edges]]
        candidates = edges[key - key_err <= float(np.min(key + key_err))].tolist()
        best = candidates[0]
        for e in candidates[1:]:
            if proj.smaller(int(tails[e]), int(tails[best])):
                best = e
        # the turn at the vertex: the pair (previous[best], best) has B = tails[best]; its float
        # value decides when certain, else exactly
        pair = int(previous[best])
        if turns[pair] > turn_errors[pair]:
            turn = 1
        elif turns[pair] < -turn_errors[pair]:
            turn = -1
        else:
            turn = proj.turn(int(tails[pair]), int(tails[best]), int(heads[best]))
        if turn == 0:
            raise _invalid(face_id, _NESTING)
        if turn > 0:
            positive.append(edges)
    if len(positive) != 1:
        raise _invalid(face_id, _NESTING)
    if bool(np.any(tails == 0)) and not bool(np.any(tails[positive[0]] == 0)):
        raise _invalid(face_id, _NESTING)


def _check_projection(face_id, proj, points, triangles, sense, chain):
    """Codex N3 r4: the global injectivity certificate (module section above; docs "Global
    injectivity"): (i) on the published triangles, (ii)-(iii) on the boundary chain; on a curved
    support also the whole mesh's P-azimuth range below 2 pi (P a single-valued function there)."""
    if proj.curved:
        lo, hi = float(np.min(proj.u - proj.aerr)), float(np.max(proj.u + proj.aerr))
        if not hi - lo < _TWO_PI * (1.0 - 4.0 * _EPS):
            raise _refuse("TRIM_WINDS_SUPPORT", face_id, _EXTENT)
    _projection_triangles(face_id, proj, points, triangles, sense)
    _check_boundary(face_id, proj, *chain)


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
        extra = _check_extra(_extra_x)
        # budgets before any scan proportional to the trim (Codex N3 r2 L3): N2's ring cap (before
        # the ring list is even built: Codex N3 r3 L2, memory O(cap)), then its ring-point cap
        # against the coedge count (each coedge gives at least one ring point)
        count = len(face.holes) + 1
        if count > _slab.MAX_SLAB_RINGS:
            raise _slab._too_complex(f"{dt._id_text(count)} rings (MAX_SLAB_RINGS = {_slab.MAX_SLAB_RINGS})")
        loops = (face.loop, *face.holes)
        coedges = sum(len(loop) for loop in loops)
        if coedges > _slab.MAX_SLAB_RING_POINTS:
            raise _slab._too_complex(f"at least {dt._id_text(coedges)} ring points "
                                     f"(MAX_SLAB_RING_POINTS = {_slab.MAX_SLAB_RING_POINTS})")
        _check_loops(face_id, loops)
        budget = [0]
        rings = [_ring(model, face_id, loop, tol, budget) for loop in loops]
        _check_magnitudes(face_id, support, rings)
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
