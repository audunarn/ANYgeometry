"""Exact separating-plane (SP) certificate for face pairs that share model topology.

ADR-T2 v2 section 4 (T2-1a). This module is a pure decision procedure: it is not
called by any planning route (T2-1b wires it in behind ``planning_mode``), so the
default behaviour of the library is unchanged by its presence.

For two faces F1, F2 with Plane or Cylinder supports and Line / EllipticArc
boundary curves it returns, per pair,

* ``Certified(r_pair, evidence)``: the true intersection of the certified regions
  G_F1 and G_F2 (section 4.1: projected stored curves closed by connectors,
  nonzero winding) lies within ``r_pair`` of the stored shared topology T_out
  (shared edges, and shared vertices not on a shared edge), with
  ``r_pair <= r_max`` = tol/1000 rounded down; or
* ``Undecided(reason, evidence)``: the exact path must run for the pair.

A cancellation raised by the ``check`` callback propagates unchanged.

Numeric rules (ADR N1-N5): stored floats are read as exact rationals; every
irrational quantity (square roots, pi, cos/sin) is an outward rational interval
(``_I``/``_Proof`` of ``cylinder_charts``, plus a fixed-point certified sine and
cosine with an enclosed pi for large arguments); a decision uses the safe-side
bound; stored floats outside zero or [2^-960, 2^960] give UNDECIDED; no float
containment decision is taken; a property used to exclude points is certified
over the whole parameter interval on which it is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
import itertools
import math

import numpy as np

from .cylinder_charts import _I, _Proof, _Refusal

__all__ = (
    "CERTIFICATE_VERSION", "S_MIN", "SPCertificatePolicy", "PlaneSupport", "CylinderSupport",
    "BoundaryUse", "FaceRecord", "Certified", "Undecided", "SPCertifier", "certify_face_pair",
    "face_record_from_model", "contract_r_max", "certified_sincos", "first_harmonic_range",
    "degree_two_bound", "vertex_separation_bound", "PI_ENCLOSURE", "SPRefusal",
)

CERTIFICATE_VERSION = "sp-exact-t2-1a-1"
S_MIN = Fraction(1, 50)                      # s_min = 0.02 (ADR 4.3 step 3/4)
SPRefusal = _Refusal
_WINDOW_LO = Fraction(1, 1 << 960)
_WINDOW_HI = Fraction(1 << 960)
_PIECE = Fraction(3, 2)                      # piece length < pi without enclosing pi (4.2a)


@dataclass(frozen=True)
class SPCertificatePolicy:
    """Proof budget of one certificate call (its own; never ``max_predicates``)."""
    max_interval_operations: int = 1_500_000
    max_distance_boxes: int = 1024
    end_refinement: int = 6
    trig_argument_bits: int = 400
    s_min: Fraction = S_MIN

    def __post_init__(self):
        # Configuration is validated (ValueError) when built, so a certificate never meets a
        # float budget or a float s_min in its exact arithmetic.
        for name in ("max_interval_operations", "max_distance_boxes", "end_refinement", "trig_argument_bits"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"SPCertificatePolicy.{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        try:
            s_min = _rational(self.s_min, "s_min")
        except _Refusal:
            raise ValueError("SPCertificatePolicy.s_min must be a finite real number") from None
        if s_min <= 0:
            raise ValueError("SPCertificatePolicy.s_min must be positive")
        object.__setattr__(self, "s_min", s_min)


# ----------------------------------------------------------------------------------- records

@dataclass(frozen=True)
class PlaneSupport:
    origin: tuple
    u_vector: tuple
    v_vector: tuple


@dataclass(frozen=True)
class CylinderSupport:
    origin: tuple
    axis: tuple
    radius: object
    radial_direction: tuple = ()


@dataclass(frozen=True)
class BoundaryUse:
    """One stored loop entry: the face's path curve from ``start_vertex`` to ``end_vertex``.

    ``edge_curve`` is the model edge's own (forward) curve, the output representative of
    the edge; ``None`` means the path curve itself.
    """
    curve: object
    edge: int
    start_vertex: int
    end_vertex: int
    forward: bool = True
    edge_curve: object = None


@dataclass(frozen=True)
class FaceRecord:
    face_id: int
    support: object
    loops: tuple
    vertices: tuple                          # ((vertex_id, (x, y, z)), ...)


@dataclass(frozen=True, eq=False)
class Certified:
    r_pair: float                            # certified upper bound, rounded up
    evidence: dict = field(default_factory=dict)

    @property
    def certified(self):
        return True


@dataclass(frozen=True, eq=False)
class Undecided:
    reason: str
    evidence: dict = field(default_factory=dict)

    @property
    def certified(self):
        return False


# ----------------------------------------------------------------------------------- numerics

def _rational(value, what):
    """The exact rational value of one stored scalar (N1), or a typed refusal BEFORE any conversion.

    Every numeric input of the module enters here. Accepted, each read exactly and never through
    ``float()``: int and numpy integers (``int()``), Fraction, and finite float, numpy floating
    and Decimal values (``as_integer_ratio``). bool and numpy.bool_, str, complex, arrays and
    every other type are refused, as are NaN and infinities. numpy.longdouble is refused
    explicitly: its precision is platform dependent (binary64 on Windows, 80- or 128-bit
    elsewhere), so neither the stored-binary64 model nor the tests cover it.
    """
    if type(value) is Fraction:
        return value
    if isinstance(value, (bool, np.bool_, np.longdouble)):
        raise _Refusal(f"numeric_window:type:{what}")
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, Fraction):                 # a subclass: rebuild from its integer parts
        n, d = value.numerator, value.denominator
        if type(n) is int and type(d) is int and d > 0:
            return Fraction(n, d)
        raise _Refusal(f"numeric_window:type:{what}")
    if isinstance(value, (float, np.floating, Decimal)):
        try:
            n, d = value.as_integer_ratio()
        except (OverflowError, ValueError):
            raise _Refusal(f"numeric_window:nonfinite:{what}") from None
        return Fraction(n, d)
    raise _Refusal(f"numeric_window:type:{what}")


def _input_in_window(made):
    """The N3 window: zero or a magnitude in [2^-960, 2^960]."""
    return made == 0 or _WINDOW_LO <= abs(made) <= _WINDOW_HI


def _float_held(value):
    """A stored binary64 (float, numpy floating): the values N3's window applies to on input."""
    return isinstance(value, (float, np.floating))


def _exact(value, what):
    """A stored value as an exact rational (N1). N3: a float-held value must lie in the window;
    exact rationals (int, Fraction, Decimal) are exempt, since the arithmetic on them is exact.
    The window is enforced again wherever a float is produced (``_up``/``_down``, r_max) or
    enters a forward-error model (``_binary64`` of delta_eval)."""
    made = _rational(value, what)
    if _float_held(value) and not _input_in_window(made):
        raise _Refusal(f"numeric_window:{what}")
    if max(abs(made.numerator).bit_length(), made.denominator.bit_length()) > 4096:
        raise _Refusal("rational_bit_budget")
    return made


def _to_float(value, what):
    """Nearest binary64 of an exact rational; an overflow is a typed refusal, never an exception."""
    try:
        return float(value)
    except OverflowError:
        raise _Refusal(f"numeric_window:{what}") from None


def _items(value, what, count=3):
    """The ``count`` entries of a stored vector, or a typed refusal (no unbounded iteration)."""
    if isinstance(value, (str, bytes)):
        raise _Refusal(f"malformed_vector:{what}")
    try:
        items = tuple(itertools.islice(iter(value), count + 1))
    except TypeError:
        raise _Refusal(f"malformed_vector:{what}") from None
    if len(items) != count:
        raise _Refusal(f"malformed_vector:{what}")
    return items


def _vec(value, what):
    return tuple(_exact(x, what) for x in _items(value, what))


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _vsub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _vscale(a, k):
    return (a[0]*k, a[1]*k, a[2]*k)


def _up(value):
    """Rational upper bound as a float rounded up (N2), kept in the N3 window (else a refusal)."""
    value = Fraction(value)
    if value <= 0:
        return 0.0 if value == 0 else -_down(-value)
    if value < _WINDOW_LO:
        return float(_WINDOW_LO)
    if value > _WINDOW_HI:
        raise _Refusal("numeric_window:upper_bound")
    result = _to_float(value, "upper_bound")
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _down(value):
    """Rational lower bound as a float rounded down (N2); below the window it becomes 0, above it
    a refusal (N3: no clamped value)."""
    value = Fraction(value)
    if value <= 0:
        return 0.0 if value == 0 else -_up(-value)
    if value < _WINDOW_LO:
        return 0.0
    if value > _WINDOW_HI:
        raise _Refusal("numeric_window:lower_bound")
    result = _to_float(value, "lower_bound")
    if Fraction(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


# ------------------------------------------------- certified pi, sine and cosine (fixed point)

_PI_BITS = 640


def _arctan_inverse(n, bits):
    """atan(1/n) * 2^bits as an integer, with an error bound in units of 2^-bits."""
    x = (1 << bits) // n
    square, total, k, terms = n*n, x, 0, 1
    while x:
        x //= square
        k += 1
        terms += 1
        term = x // (2*k + 1)
        total = total - term if k % 2 else total + term
    # x_k is within 2 units below its true value, so each term is within 3 units;
    # the alternating tail after x reaches 0 is below 2 units.
    return total, 3*terms + 2


def _pi_fixed():
    a, ea = _arctan_inverse(5, _PI_BITS)
    b, eb = _arctan_inverse(239, _PI_BITS)
    return 16*a - 4*b, 16*ea + 4*eb


_PI_FIX, _PI_ERR = _pi_fixed()
PI_ENCLOSURE = _I(Fraction(_PI_FIX - _PI_ERR, 1 << _PI_BITS), Fraction(_PI_FIX + _PI_ERR, 1 << _PI_BITS))
_TRIG_BITS = 128
_TRIG_CACHE = {}
_TRIG_CACHE_LIMIT = 1 << 18


def _sincos_fixed(x, max_bits):
    """Outward enclosures (sin x, cos x) of a rational x, as ``_I`` with 2^-128 denominators."""
    whole = abs(x.numerator) // x.denominator
    bits = whole.bit_length()
    if bits > max_bits:
        raise _Refusal("trig_argument_budget")
    P = _TRIG_BITS
    Pr = P + bits + 16
    shift = _PI_BITS + 1 - Pr
    if shift < 64:
        raise _Refusal("trig_argument_budget")
    half_pi = _PI_FIX >> shift                         # pi/2 * 2^Pr, floor
    half_err = (_PI_ERR >> shift) + 2                  # units of 2^-Pr
    X = (x.numerator << Pr) // x.denominator           # |X - x 2^Pr| < 1
    k = (2*X + half_pi) // (2*half_pi)
    R = X - k*half_pi
    err_r_pr = 1 + abs(k)*half_err                     # |R - r 2^Pr|, r = x - k pi/2
    drop = Pr - P
    Rp = R >> drop
    err_r = 1 + -(-err_r_pr // (1 << drop)) + 1        # units of 2^-P
    one = 1 << P
    if abs(Rp) > (one*13)//16:
        raise _Refusal("trig_reduction_unqualified")
    R2 = (Rp*Rp) >> P
    sine, term, n = Rp, Rp, 1
    while term:
        term = -((term*R2) >> P) // ((2*n)*(2*n + 1))
        sine += term
        n += 1
    sine_terms = n
    cosine, term, n = one, one, 1
    while term:
        term = -((term*R2) >> P) // ((2*n - 1)*(2*n))
        cosine += term
        n += 1
    err = 8*(max(sine_terms, n) + 1) + err_r
    s_lo, s_hi, c_lo, c_hi = sine - err, sine + err, cosine - err, cosine + err
    quadrant = k % 4
    if quadrant == 0:
        sl, sh, cl, ch = s_lo, s_hi, c_lo, c_hi
    elif quadrant == 1:
        sl, sh, cl, ch = c_lo, c_hi, -s_hi, -s_lo
    elif quadrant == 2:
        sl, sh, cl, ch = -s_hi, -s_lo, -c_hi, -c_lo
    else:
        sl, sh, cl, ch = -c_hi, -c_lo, s_lo, s_hi

    def clip(lo, hi):
        return _I(Fraction(max(lo, -one), one), Fraction(min(hi, one), one))
    return clip(sl, sh), clip(cl, ch)


def certified_sincos(x, *, max_bits=400):
    """Public test hook: ((sin_lo, sin_hi), (cos_lo, cos_hi)) of an exact rational ``x``."""
    s, c = _sincos_fixed(_rational(x, "x"), max_bits)
    return (s.lo, s.hi), (c.lo, c.hi)


class _CallbackRaised(BaseException):
    """Carrier of an exception raised by the caller's ``check`` (ADR 4.5 CANCELLED).

    A BaseException outside the ``_Refusal`` hierarchy, so no refusal handler can turn a
    cancellation into UNDECIDED (even when the callback itself raises an ``SPRefusal``);
    ``SPCertifier.certify`` re-raises the original object.
    """

    def __init__(self, original):
        super().__init__()
        self.original = original


class _Ar:
    """Outward interval arithmetic on ``_Proof`` (budget, cancellation polling), plus trig."""

    __slots__ = ("p", "policy")

    def __init__(self, policy, check):
        callback = None
        if check is not None:
            def callback(phase, _check=check):
                try:
                    _check()
                except BaseException as error:      # noqa: BLE001 - every callback exception cancels
                    raise _CallbackRaised(error) from None
        self.p = _Proof(policy, callback)
        self.policy = policy

    def I(self, value, hi=None):
        return self.p.i(value, hi)

    def add(self, a, b):
        return self.p.add(a, b)

    def sub(self, a, b):
        return self.p.sub(a, b)

    def mul(self, a, b):
        return self.p.mul(a, b)

    def div(self, a, b):
        return self.p.div(a, b)

    def sqrt(self, a):
        a = self.p.i(a)
        if a.lo < 0:
            if a.hi < 0:
                raise _Refusal("negative_sqrt_enclosure")
            a = _I(Fraction(0), a.hi)
        return self.p.sqrt(a)

    def tick(self, count=1):
        for _ in range(count):
            self.p.charge()

    def phase(self, name):
        self.p.cancel("separating-plane certificate: " + name)

    def sincos(self, x):
        # A fixed charge per lookup keeps the outcome independent of cache state.
        for _ in range(8):
            self.p.charge("series_terms")
        hit = _TRIG_CACHE.get(x)
        if hit is None:
            hit = _sincos_fixed(x, self.policy.trig_argument_bits)
            if len(_TRIG_CACHE) >= _TRIG_CACHE_LIMIT:
                _TRIG_CACHE.clear()
            _TRIG_CACHE[x] = hit
        return hit

    def norm_hi(self, lows, highs, centre=None):
        """Upper bound of |p - centre| over the box [lows, highs]."""
        total = Fraction(0)
        for i in range(3):
            c = 0 if centre is None else centre[i]
            m = max(abs(lows[i] - c), abs(highs[i] - c))
            total += m*m
        return self.sqrt(total).hi


def _sqrt_outward(value):
    """Outward 2^-80 grid enclosure of the square root of an exact nonnegative rational."""
    value = Fraction(value)
    if value < 0:
        raise _Refusal("negative_sqrt_enclosure")
    grid = 1 << 80
    low = math.isqrt((value.numerator*grid*grid)//value.denominator)
    high = low if Fraction(low*low, grid*grid) == value else low + 1
    return _I(Fraction(low, grid), Fraction(high, grid))


def _sqrt_up(value):
    """Exact rational upper bound of sqrt(value): (x + t^2)/(2t) >= sqrt(x) for every t > 0."""
    value = Fraction(value)
    if value < 0:
        raise _Refusal("negative_sqrt_enclosure")
    if value == 0:
        return Fraction(0)
    try:
        t = Fraction(math.sqrt(_to_float(value, "sqrt")))
    except (_Refusal, OverflowError, ValueError):
        t = Fraction(1)
    if t <= 0:
        t = Fraction(1)
    return (value + t*t)/(2*t)


def _fh_value(ar, a, b, c, cs):
    s, co = cs
    return ar.add(a, ar.add(ar.mul(b, co), ar.mul(c, s)))


def _fh_deriv(ar, b, c, cs):
    s, co = cs
    return ar.sub(ar.mul(c, co), ar.mul(b, s))


def _sign(interval):
    return 1 if interval.lo > 0 else -1 if interval.hi < 0 else 0


def _piece_range(vp, vq, dp, dq, crit_lo, crit_hi):
    """Range of a first harmonic on a piece shorter than pi (ADR 4.2a step 3).

    The derivative A cos(phi - psi) has simple zeros exactly pi apart, so the closed piece holds
    at most one. Equal certified end signs: monotone. Otherwise a zero z may exist; the sign
    after z is -(sign before z). If one end sign is certified, the zero can only be of the kind
    compatible with it (a certified negative derivative at the right end, or a positive one at
    the left end, allows only a maximum a + A; the mirrored cases only a minimum a - A). Only
    when both end signs are undecided are both a +- A included (the ADR's widening rule).
    """
    lo, hi = min(vp.lo, vq.lo), max(vp.hi, vq.hi)
    sp, sq = _sign(dp), _sign(dq)
    if sp and sp == sq:
        return lo, hi                        # no zero of the derivative: monotone
    if sp == 1 or sq == -1:
        return lo, max(hi, crit_hi)          # at most one critical point, a maximum a + A
    if sp == -1 or sq == 1:
        return min(lo, crit_lo), hi          # at most one critical point, a minimum a - A
    return min(lo, crit_lo), max(hi, crit_hi)   # both end signs undecided: include both a +- A


def _fh_interval_range(ar, a, b, c, p, q, amplitude=None):
    """Certified range of a + b cos + c sin over [p, q] (rational, q - p <= 3/2)."""
    if not q - p <= _PIECE:
        raise _Refusal("internal:piece_too_long")
    if amplitude is None:
        amplitude = ar.sqrt(ar.add(ar.mul(b, b), ar.mul(c, c)))
    cp, cq = ar.sincos(p), ar.sincos(q)
    vp, vq = _fh_value(ar, a, b, c, cp), _fh_value(ar, a, b, c, cq)
    dp, dq = _fh_deriv(ar, b, c, cp), _fh_deriv(ar, b, c, cq)
    return _piece_range(vp, vq, dp, dq, ar.sub(a, amplitude).lo, ar.add(a, amplitude).hi)


def _split(lo, hi):
    """Rational pieces of length <= 3/2 covering [lo, hi]."""
    if hi < lo:
        raise _Refusal("internal:reversed_interval")
    count = max(1, math.ceil((hi - lo)/_PIECE))
    step = (hi - lo)/count
    return [(lo + j*step, lo + (j + 1)*step if j + 1 < count else hi) for j in range(count)]


def first_harmonic_range(coefficients, start, sweep, *, policy=None):
    """Public 4.2a routine: certified range of a + b cos(phi) + c sin(phi) on the arc interval.

    Degree-two input (five coefficients) is rejected: it needs ``degree_two_bound``.
    """
    coefficients = tuple(coefficients)
    if len(coefficients) != 3:
        raise ValueError("first_harmonic_range takes exactly (a, b, c); degree two needs degree_two_bound")
    policy = policy or SPCertificatePolicy()
    ar = _Ar(policy, None)
    a, b, c = (_exact(x, "coefficient") for x in coefficients)
    s0, sw = _exact(start, "start"), _exact(sweep, "sweep")
    lo, hi = min(s0, s0 + sw), max(s0, s0 + sw)
    out_lo, out_hi = None, None
    amplitude = ar.sqrt(ar.add(ar.mul(b, b), ar.mul(c, c)))
    for p, q in _split(lo, hi):
        r = _fh_interval_range(ar, a, b, c, p, q, amplitude)
        out_lo = r[0] if out_lo is None else min(out_lo, r[0])
        out_hi = r[1] if out_hi is None else max(out_hi, r[1])
    return out_lo, out_hi


def _degree_two(a0, a1, b1, a2, b2):
    sigma = abs(a1) + abs(b1) + abs(a2) + abs(b2)
    return a0 - sigma, a0 + sigma


def degree_two_bound(coefficients):
    """Public 4.2b routine: [a0 - S, a0 + S] encloses a0 + a1 cos + b1 sin + a2 cos 2 + b2 sin 2."""
    a0, a1, b1, a2, b2 = (_exact(x, "coefficient") for x in coefficients)
    return _degree_two(a0, a1, b1, a2, b2)


def _quadratic_form_harmonics(W, U, V, Q):
    """Degree-two coefficients of Q(w, w) for w = W + U cos + V sin (Q symmetric bilinear)."""
    QWW, QUU, QVV = Q(W, W), Q(U, U), Q(V, V)
    return (QWW + (QUU + QVV)/2, 2*Q(W, U), 2*Q(W, V), (QUU - QVV)/2, Q(U, V))


# ----------------------------------------------------------------------------------- curves

class _Line:
    kind = "line"

    def __init__(self, curve, what):
        self.S = _vec(curve.start, what + ".start")
        self.E = _vec(curve.end, what + ".end")
        self.D = _vsub(self.E, self.S)
        self.len2 = _dot(self.D, self.D)
        if self.len2 == 0:
            raise _Refusal("degenerate_line")

    def end_point(self, ar, which):
        p = self.S if which == 0 else self.E
        return p, p

    def point_box(self, ar, t0, t1):
        a = tuple(self.S[i] + t0*self.D[i] for i in range(3))
        b = tuple(self.S[i] + t1*self.D[i] for i in range(3))
        return tuple(min(a[i], b[i]) for i in range(3)), tuple(max(a[i], b[i]) for i in range(3))

    def domain(self):
        return Fraction(0), Fraction(1)

    def base_pieces(self):
        return [(Fraction(0), Fraction(1))]


class _Arc:
    kind = "arc"

    def __init__(self, curve, policy, what):
        self.C = _vec(curve.center, what + ".center")
        self.U = _vec(curve.u_vector, what + ".u")
        self.V = _vec(curve.v_vector, what + ".v")
        self.p0 = _exact(curve.start_angle, what + ".start_angle")
        sweep = _exact(curve.sweep_angle, what + ".sweep_angle")
        if sweep == 0:
            raise _Refusal("degenerate_arc")
        if abs(sweep) > Fraction(7):          # > 2 pi + 16 eps is rejected by EllipticArc
            raise _Refusal("arc_sweep_out_of_scope")
        if _cross(self.U, self.V) == (0, 0, 0):
            raise _Refusal("degenerate_arc_basis")
        self.p1 = self.p0 + sweep
        self.direction = 1 if sweep > 0 else -1
        self.lo, self.hi = min(self.p0, self.p1), max(self.p0, self.p1)
        span = self.hi - self.lo
        count = max(1, math.ceil(span/_PIECE))
        step = span/count
        points = {self.lo + j*step for j in range(count)} | {self.hi}
        self.coarse = tuple(sorted(points))
        for k in range(1, policy.end_refinement + 1):
            points.add(self.lo + step/(1 << k))
            points.add(self.hi - step/(1 << k))
        self.breaks = tuple(sorted(points))
        UU, VV, UV = _dot(self.U, self.U), _dot(self.V, self.V), _dot(self.U, self.V)
        self.speed2_hi = (UU + VV)/2 + abs(VV - UU)/2 + abs(UV)     # |x'(phi)|^2, 4.2b

    def angle_at(self, which):
        return self.p0 if which == 0 else self.p1

    def point(self, ar, phi):
        cs = ar.sincos(phi)
        lows, highs = [], []
        for i in range(3):
            v = _fh_value(ar, self.C[i], self.U[i], self.V[i], cs)
            lows.append(v.lo)
            highs.append(v.hi)
        return tuple(lows), tuple(highs)

    def end_point(self, ar, which):
        return self.point(ar, self.angle_at(which))

    def point_box(self, ar, p, q):
        # Pure function of exact inputs: cached on the curve, with a fixed charge per lookup.
        ar.tick(4)
        cache = self.__dict__.setdefault("_boxes", {})
        hit = cache.get((p, q))
        if hit is not None:
            return hit
        amplitudes = self.__dict__.get("_amplitudes")
        if amplitudes is None:
            amplitudes = self._amplitudes = tuple(_sqrt_outward(self.U[i]**2 + self.V[i]**2) for i in range(3))
        lows, highs = [], []
        for i in range(3):
            lo, hi = _fh_interval_range(ar, self.C[i], self.U[i], self.V[i], p, q, amplitudes[i])
            lows.append(lo)
            highs.append(hi)
        hit = cache[(p, q)] = (tuple(lows), tuple(highs))
        return hit

    def domain(self):
        return self.lo, self.hi

    def base_pieces(self, refined=True):
        b = self.breaks if refined else self.coarse
        return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def _make_curve(curve, policy, what):
    name = type(curve).__name__
    if hasattr(curve, "start") and hasattr(curve, "end") and name in ("LinePath", "_RawLine"):
        return _Line(curve, what)
    if name in ("EllipticArc", "_RawArc"):
        return _Arc(curve, policy, what)
    raise _Refusal("unsupported_curve:" + name)


def _same_record(a, b):
    """Exact identity of two curve records (every stored value compared as an exact rational,
    read by ``_rational``); a value that cannot be read exactly makes the records differ."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    names = ("start", "end") if type(a).__name__ in ("LinePath", "_RawLine") else \
        ("center", "u_vector", "v_vector", "start_angle", "sweep_angle")

    def exact(name, value):
        if name in ("start_angle", "sweep_angle"):
            return _rational(value, "record")
        return tuple(_rational(x, "record") for x in _items(value, "record"))
    try:
        return all(exact(n, getattr(a, n)) == exact(n, getattr(b, n)) for n in names)
    except (AttributeError, _Refusal):
        return False


# ----------------------------------------------------------------------------------- faces

class _Use:
    __slots__ = ("geom", "edge", "vs", "ve", "forward", "dS", "ends", "rep_hi", "edge_end", "Hhi",
                 "loop", "index", "representative")


class _UseRecord:
    __slots__ = ("curve", "edge", "start_vertex", "end_vertex", "forward", "edge_curve")


def _record_parts(record):
    """(face_id, support, vertices, loops) of a FaceRecord-like object, with every structural field
    read once and checked (hashable ids, (id, position) pairs, loops of uses with all fields); a
    malformed record is UNDECIDED (``record:malformed``), never a TypeError/ValueError/AttributeError."""
    try:
        face_id, support = record.face_id, record.support
        hash(face_id)
        vertices = []
        for item in tuple(record.vertices):
            vid, position = item
            hash(vid)
            vertices.append((vid, position))
        loops = []
        for loop in tuple(record.loops):
            uses = []
            for item in tuple(loop):
                use = _UseRecord()
                use.curve, use.edge = item.curve, item.edge
                use.start_vertex, use.end_vertex = item.start_vertex, item.end_vertex
                use.forward, use.edge_curve = bool(item.forward), item.edge_curve
                hash(use.edge), hash(use.start_vertex), hash(use.end_vertex)
                uses.append(use)
            loops.append(tuple(uses))
    except (AttributeError, TypeError, ValueError):
        raise _Refusal("record:malformed") from None
    return face_id, support, tuple(vertices), tuple(loops)


class _Face:
    """Per-face certified data (closure rule and defects, ADR 4.1); budget of its own."""

    def __init__(self, record, policy, check):
        ar = _Ar(policy, check)
        ar.phase("face preparation")
        self.id, support, record_vertices, record_loops = _record_parts(record)
        if isinstance(support, PlaneSupport):
            self.kind = "plane"
            origin = _vec(support.origin, "plane.origin")
            u, v = _vec(support.u_vector, "plane.u"), _vec(support.v_vector, "plane.v")
            self.nF = _cross(u, v)
            if self.nF == (0, 0, 0):
                raise _Refusal("degenerate_plane_support")
            self.kF = _dot(self.nF, origin)
            self.nF2 = _dot(self.nF, self.nF)
            self.nF_lo = ar.sqrt(self.nF2).lo
            if self.nF_lo <= 0:
                raise _Refusal("degenerate_plane_support")
            self.support_key = ("plane", tuple(origin), tuple(u), tuple(v))
        elif isinstance(support, CylinderSupport):
            self.kind = "cyl"
            self.A = _vec(support.origin, "cylinder.origin")
            self.a = _vec(support.axis, "cylinder.axis")
            self.R = _exact(support.radius, "cylinder.radius")
            if self.R <= 0 or self.a == (0, 0, 0):
                raise _Refusal("degenerate_cylinder_support")
            self.a2 = _dot(self.a, self.a)
            try:
                radial = tuple(itertools.islice(iter(support.radial_direction), 4))
            except TypeError:
                raise _Refusal("malformed_vector:cylinder.radial") from None
            if radial:
                radial = _vec(radial, "cylinder.radial")
            self.support_key = ("cyl", self.A, self.a, self.R, tuple(radial))
        else:
            raise _Refusal("unsupported_support:" + type(support).__name__)

        self.vertices = {}
        for vid, position in record_vertices:
            self.vertices[vid] = _vec(position, f"vertex.{vid}")
        self.loops = []
        self.edges = {}
        for li, loop in enumerate(record_loops):
            uses = []
            for i, item in enumerate(loop):
                u = _Use()
                u.geom = _make_curve(item.curve, policy, f"face{self.id}.loop{li}.{i}")
                u.edge, u.vs, u.ve, u.forward = item.edge, item.start_vertex, item.end_vertex, item.forward
                u.loop, u.index = li, i
                if u.vs not in self.vertices or u.ve not in self.vertices:
                    raise _Refusal("missing_vertex")
                if u.edge in self.edges:
                    raise _Refusal("repeated_edge_in_face")
                self.edges[u.edge] = u
                u.rep_hi, edge_geom = self._representative(ar, item, u, policy)
                u.representative = item.curve if item.edge_curve is None else item.edge_curve
                u.edge_end = {}
                # The model edge curve's own end at each of its vertices, against the stored vertex.
                start_vid, end_vid = (u.vs, u.ve) if u.forward else (u.ve, u.vs)
                for which, vid in ((0, start_vid), (1, end_vid)):
                    lows, highs = edge_geom.end_point(ar, which)
                    distance = ar.norm_hi(lows, highs, self.vertices[vid])
                    u.edge_end[vid] = min(u.edge_end.get(vid, distance), distance)
                u.ends = (u.geom.end_point(ar, 0), u.geom.end_point(ar, 1))
                uses.append(u)
            if not uses:
                raise _Refusal("empty_loop")
            for i, u in enumerate(uses):
                if u.ve != uses[(i + 1) % len(uses)].vs:
                    raise _Refusal("loop_vertex_mismatch")
            self.loops.append(uses)
        self.uses = [u for loop in self.loops for u in loop]
        if not self.uses:
            raise _Refusal("face_without_boundary")
        self.vertex_ids = {u.vs for u in self.uses} | {u.ve for u in self.uses}

        # Defects: delta_S of curves and vertices, delta_V at loop vertices.
        self.dSV = {vid: self._vertex_support_distance(ar, self.vertices[vid]) for vid in self.vertex_ids}
        for u in self.uses:
            u.dS, u.Hhi = self._curve_support_distance(ar, u.geom)
        self.dV = {}
        self.loop_vertices = []                      # (loop, i, vid): between use i and use i+1
        for loop in self.loops:
            for i, u in enumerate(loop):
                nxt = loop[(i + 1) % len(loop)]
                vid = u.ve
                V = self.vertices[vid]
                d1 = ar.norm_hi(*u.ends[1], V)
                d2 = ar.norm_hi(*nxt.ends[0], V)
                d = max(d1, d2)
                self.dV[vid] = max(self.dV.get(vid, Fraction(0)), d)
                self.loop_vertices.append((u, nxt, vid, d))
        if self.kind == "cyl":
            half = self.R/2
            if any(u.dS > half for u in self.uses) or any(x > half for x in self.dSV.values()):
                raise _Refusal("cylinder_projection_unqualified")
            delta = max([u.dS for u in self.uses] + list(self.dSV.values()))
            reach = max(d for *_, d in self.loop_vertices)
            base = self.R - delta - reach
            if base < half:
                raise _Refusal("cylinder_projection_unqualified")
            lipschitz = self.R/base
            chord = lipschitz*reach
            x = chord/self.R
            if x >= 1:
                raise _Refusal("cylinder_connector_unqualified")
            self.lam = lipschitz/(1 - x*x/6)
            self._check_net_turn(ar)
        else:
            self.lam = Fraction(1)
        self.dGamma = max(max(u.dS, nxt.dS) + self.lam*d for u, nxt, _vid, d in self.loop_vertices)
        self.dS_max = max(u.dS for u in self.uses)
        self.dS_arc_cyl = max((u.dS for u in self.uses if self.kind == "cyl" and u.geom.kind == "arc"),
                              default=Fraction(0))
        self.dV_max = max(d for *_, d in self.loop_vertices)

    # -- representatives and support distances

    def _representative(self, ar, item, u, policy):
        """Hausdorff bound between the face's path curve and the model edge curve."""
        if item.edge_curve is None or _same_record(item.edge_curve, item.curve):
            return Fraction(0), u.geom
        edge_geom = _make_curve(item.edge_curve, policy, f"edge{u.edge}")
        path = u.geom
        if path.kind == "line" and edge_geom.kind == "line":
            pairs = ((path.S, edge_geom.S), (path.E, edge_geom.E)) if u.forward else \
                ((path.S, edge_geom.E), (path.E, edge_geom.S))
            bound = max(ar.sqrt(_dot(_vsub(p, q), _vsub(p, q))).hi for p, q in pairs)
            return bound, edge_geom
        if path.kind == "arc" and edge_geom.kind == "arc" and (path.C, path.U, path.V) == \
                (edge_geom.C, edge_geom.U, edge_geom.V):
            shift = max(abs(path.lo - edge_geom.lo), abs(path.hi - edge_geom.hi))
            return ar.mul(ar.sqrt(path.speed2_hi), shift).hi, edge_geom
        raise _Refusal("edge_record_mismatch")

    def _H(self, w):
        """|a|^2 |w|^2 - (w.a)^2 = |a|^2 * (squared axis distance)."""
        wa = _dot(w, self.a)
        return self.a2*_dot(w, w) - wa*wa

    def _vertex_support_distance(self, ar, V):
        if self.kind == "plane":
            return ar.div(abs(_dot(self.nF, V) - self.kF), self.nF_lo).hi
        h = self._H(_vsub(V, self.A))/self.a2
        return abs(h - self.R*self.R)/self.R

    def _curve_support_distance(self, ar, geom):
        """(delta_S upper bound, H upper bound) for a stored curve (ADR 4.1 table, 4.2b)."""
        if self.kind == "plane":
            if geom.kind == "line":
                num = max(abs(_dot(self.nF, geom.S) - self.kF), abs(_dot(self.nF, geom.E) - self.kF))
            else:
                b, c = _dot(self.nF, geom.U), _dot(self.nF, geom.V)
                num = abs(_dot(self.nF, geom.C) - self.kF) + ar.sqrt(b*b + c*c).hi
            return ar.div(num, self.nF_lo).hi, None
        R2 = self.R*self.R
        if geom.kind == "line":
            w0 = _vsub(geom.S, self.A)
            d = geom.D
            wa, da = _dot(w0, self.a), _dot(d, self.a)
            H0 = self.a2*_dot(w0, w0) - wa*wa
            H1 = 2*(self.a2*_dot(w0, d) - wa*da)
            H2 = self.a2*_dot(d, d) - da*da                # >= 0: convex in t
            values = [H0, H0 + H1 + H2]
            if H2 > 0:
                t = -H1/(2*H2)
                if 0 < t < 1:
                    values.append(H0 + H1*t + H2*t*t)
            hmax, hmin = max(values), min(values)
            dS = max(abs(hmax/self.a2 - R2), abs(hmin/self.a2 - R2))/self.R
            return dS, max(H0, H0 + H1 + H2)
        W = _vsub(geom.C, self.A)

        def Q(x, y):
            return self.a2*_dot(x, y) - _dot(x, self.a)*_dot(y, self.a)
        a0, a1, b1, a2, b2 = _quadratic_form_harmonics(W, geom.U, geom.V, Q)
        lo, hi = _degree_two(a0, a1, b1, a2, b2)
        sigma = hi - a0
        dS = (abs(a0/self.a2 - R2) + sigma/self.a2)/self.R
        return dS, hi

    def _check_net_turn(self, ar):
        """Each loop must close in the unrolled chart (net turn 0), else UNDECIDED (ADR 4.1)."""
        two_pi_lo = 2*PI_ENCLOSURE.lo
        a = self.a

        def perp(p):
            w = _vsub(p, self.A)
            return _vsub(w, _vscale(a, _dot(w, a)/self.a2))     # exact axis-perpendicular part

        def turn_bound(p, q):
            """Upper bound of the radial turn along a short element from p to q (boxes)."""
            mp_, ep = _box_turn_error(p, self.R)
            mq_, eq = _box_turn_error(q, self.R)
            rp, rq = perp(mp_), perp(mq_)
            if _dot(rp, rq) <= 0:
                raise _Refusal("cylinder_turn_unqualified")
            cr = _cross(rp, rq)
            sin2 = _dot(cr, cr)/(_dot(rp, rp)*_dot(rq, rq))
            # |angle| <= (pi/2) |sin angle| for |angle| <= pi/2
            return PI_ENCLOSURE.hi/2*_sqrt_up(sin2) + ep + eq

        for loop in self.loops:
            exact_sum, error = Fraction(0), Fraction(0)
            for i, u in enumerate(loop):
                nxt = loop[(i + 1) % len(loop)]
                V = self.vertices[u.ve]
                error += turn_bound(u.ends[1], (V, V)) + turn_bound((V, V), nxt.ends[0])
                g = u.geom
                if g.kind == "line":
                    error += turn_bound(g.end_point(ar, 0), g.end_point(ar, 1))
                    continue
                triple = _dot(a, _cross(g.U, g.V))
                if triple == 0:
                    raise _Refusal("cylinder_turn_unqualified")
                sigma = 1 if triple > 0 else -1
                # Q(phi) = U_perp cos + sigma (a_hat x U_perp) sin turns by exactly sigma*sweep.
                Vp = _vsub(g.V, _vscale(a, _dot(g.V, a)/self.a2))
                axU = _cross(a, g.U)
                inv_len = ar.div(1, ar.sqrt(self.a2))
                diff = [ar.sub(Vp[i], ar.mul(ar.mul(sigma, axU[i]), inv_len)) for i in range(3)]
                diff_hi = ar.sqrt(sum((max(abs(x.lo), abs(x.hi))**2 for x in diff), Fraction(0))).hi
                Wp_hi = ar.div(ar.sqrt(self._H(_vsub(g.C, self.A))), ar.sqrt(self.a2)).hi
                Up_lo = ar.div(ar.sqrt(self._H(g.U)), ar.sqrt(self.a2)).lo
                eta = Wp_hi + diff_hi
                if Up_lo <= 0 or eta >= Up_lo:
                    raise _Refusal("cylinder_turn_unqualified")
                exact_sum += sigma*(g.p1 - g.p0)
                error += PI_ENCLOSURE.hi*eta/Up_lo        # 2 asin(x) <= pi x
            if not abs(exact_sum) + error < two_pi_lo:
                raise _Refusal("cylinder_loop_winds")


def _box_turn_error(box, R):
    """(midpoint, bound on the chart-angle error of using it) for a box holding a true point.

    The true point's radial vector r has |r| >= R/2 (projection check of the face); the box
    midpoint differs from it by d with |d_perp| <= |d| <= hw (sum of half-widths). If |d_perp|
    < |r|, angle(r, r + d_perp) <= asin(hw/(R/2)) <= (pi/2) hw/(R/2) = pi hw/R. That hypothesis
    (addendum A5, H5.4) is checked here, conservatively as 2 hw < R/2, instead of being relied on
    structurally; otherwise UNDECIDED.
    """
    lows, highs = box
    hw = sum((highs[i] - lows[i])/2 for i in range(3))
    if not 2*hw < R/2:
        raise _Refusal("cylinder_turn_unqualified:box_width")
    mid = tuple((lows[i] + highs[i])/2 for i in range(3))
    return mid, PI_ENCLOSURE.hi*hw/R


# ----------------------------------------------------------------------------------- functionals

class _Fn:
    """v = n.x - k on one stored curve: pieces in path order with v- and dv/dparam-ranges."""
    __slots__ = ("pieces", "v0", "v1", "sgn", "speed_hi", "lo", "hi")


def _negated_fn(f):
    """The table of -v from the table of v (exact: every bound is negated and swapped)."""
    g = _Fn()
    g.pieces = [(-vh, -vl, -wh, -wl) for vl, vh, wl, wh in f.pieces]
    g.v0, g.v1 = _I(-f.v0.hi, -f.v0.lo), _I(-f.v1.hi, -f.v1.lo)
    g.sgn, g.speed_hi, g.lo, g.hi = f.sgn, f.speed_hi, -f.hi, -f.lo
    return g


def _curve_fn(ar, geom, n, k, refined=False):
    f = _Fn()
    if geom.kind == "line":
        v0 = _dot(n, geom.S) - k
        v1 = _dot(n, geom.E) - k
        w = v1 - v0
        f.pieces = [(min(v0, v1), max(v0, v1), w, w)]
        f.v0, f.v1 = ar.I(v0), ar.I(v1)
        f.sgn = 1
        f.speed_hi = ar.sqrt(geom.len2).hi
        f.lo, f.hi = min(v0, v1), max(v0, v1)
        return f
    a = _dot(n, geom.C) - k
    b, c = _dot(n, geom.U), _dot(n, geom.V)
    A = ar.sqrt(ar.add(ar.mul(b, b), ar.mul(c, c)))
    crit_v = (ar.sub(a, A).lo, ar.add(a, A).hi)
    crit_w = (-A.hi, A.hi)
    values = {}
    for phi in (geom.breaks if refined else geom.coarse):
        s, co = ar.sincos(phi)
        bco, cs_, cco, bs = ar.mul(b, co), ar.mul(c, s), ar.mul(c, co), ar.mul(b, s)
        harmonic = ar.add(bco, cs_)                  # b cos + c sin
        v = ar.add(a, harmonic)
        w = ar.sub(cco, bs)                          # dv/dphi = c cos - b sin
        w2 = _I(-harmonic.hi, -harmonic.lo)          # d2v/dphi2 = -(b cos + c sin), exact negation
        values[phi] = (v, w, w2)
    pieces = []
    for p, q in geom.base_pieces(refined):
        vp, wp, w2p = values[p]
        vq, wq, w2q = values[q]
        vr = _piece_range(vp, vq, wp, wq, *crit_v)
        wr = _piece_range(wp, wq, w2p, w2q, *crit_w)
        pieces.append((vr[0], vr[1], wr[0], wr[1]))
    if geom.direction < 0:
        pieces.reverse()
    f.pieces = pieces
    f.v0, f.v1 = values[geom.p0][0], values[geom.p1][0]
    f.sgn = geom.direction
    f.speed_hi = ar.sqrt(geom.speed2_hi).hi
    f.lo, f.hi = min(p[0] for p in pieces), max(p[1] for p in pieces)
    return f


def _flip(lo, hi, s):
    return (lo, hi) if s > 0 else (-hi, -lo)


# ----------------------------------------------------------------------------------- distances

def _boxes_gap2(a, b):
    total = Fraction(0)
    for i in range(3):
        gap = max(Fraction(0), b[0][i] - a[1][i], a[0][i] - b[1][i])
        total += gap*gap
    return total


class _Piece:
    __slots__ = ("geom", "p", "q", "point")

    def __init__(self, geom, p, q, point=None):
        self.geom, self.p, self.q, self.point = geom, p, q, point

    def box(self, ar):
        if self.point is not None:
            return self.point, self.point
        return self.geom.point_box(ar, self.p, self.q)

    def split(self):
        if self.point is not None:
            return None
        m = (self.p + self.q)/2
        if m == self.p or m == self.q:
            return None
        return _Piece(self.geom, self.p, m), _Piece(self.geom, m, self.q)


def _pieces(geom, lo=None, hi=None):
    dlo, dhi = geom.domain()
    lo = dlo if lo is None else lo
    hi = dhi if hi is None else hi
    if hi < lo:
        return []
    if geom.kind == "line":
        return [_Piece(geom, lo, hi)]
    return [_Piece(geom, p, q) for p, q in _split(lo, hi)]


def _apart(ar, first, second, rho, policy):
    """Certified d(first, second) > rho for lists of pieces, by box subdivision (whole curves)."""
    if not first or not second:
        return True
    limit = rho*rho
    work = [(a, b) for a in first for b in second]
    boxes = 0
    while work:
        a, b = work.pop()
        boxes += 1
        if boxes > policy.max_distance_boxes:
            return False
        ar.tick()
        ba, bb = a.box(ar), b.box(ar)
        if _boxes_gap2(ba, bb) > limit:
            continue
        sa = sum(bb_ - ba_ for ba_, bb_ in zip(*ba))
        sb = sum(bb_ - ba_ for ba_, bb_ in zip(*bb))
        if sa >= sb:
            parts = a.split()
            if parts is None:
                parts = b.split()
                if parts is None:
                    return False
                work.extend((a, x) for x in parts)
            else:
                work.extend((x, b) for x in parts)
        else:
            parts = b.split()
            if parts is None:
                parts = a.split()
                if parts is None:
                    return False
                work.extend((x, b) for x in parts)
            else:
                work.extend((a, x) for x in parts)
    return True


# ----------------------------------------------------------------------------- vertex separation

def _cones(ar, geom, which, policy):
    """Whole-piece tangent cones at the curve end ``which`` (ADR 4.2a, step 6(iii) S1).

    Returns [(J pieces, complement pieces, t, c_alpha, s_alpha_hi)].
    """
    if geom.kind == "line":
        t = geom.D if which == 0 else _vscale(geom.D, -1)
        return [([_Piece(geom, Fraction(0), Fraction(1))], [], t, Fraction(1), Fraction(0))]
    phiV = geom.angle_at(which)
    at_lo = phiV == geom.lo
    sigma = 1 if at_lo else -1                         # away from V: increasing phi when V is the low end
    cs = _sincos_fixed(phiV, policy.trig_argument_bits)
    sm, cm = cs[0].mid, cs[1].mid
    # A nearby binary64 direction (any t works: c_alpha below is certified for this exact t).
    t = tuple(Fraction(_to_float(sigma*(-geom.U[i]*sm + geom.V[i]*cm), "cone_direction")) for i in range(3))
    if t == (0, 0, 0):
        return []
    t_norm_hi = ar.sqrt(_dot(t, t)).hi
    M = ar.sqrt(geom.speed2_hi).hi
    b, c = sigma*_dot(t, geom.V), -sigma*_dot(t, geom.U)
    lengths = sorted({abs(x - phiV) for x in geom.breaks if 0 < abs(x - phiV) <= _PIECE})
    out = []
    for L in lengths:
        J = (phiV, phiV + L) if at_lo else (phiV - L, phiV)
        lower = None
        for p, q in [(x, y) for x, y in geom.base_pieces() if x >= J[0] and y <= J[1]]:
            lo, _hi = _fh_interval_range(ar, 0, b, c, p, q)
            lower = lo if lower is None else min(lower, lo)
        if lower is None or lower <= 0:
            break                                       # larger J cannot have a cone either
        c_alpha = min(Fraction(1), ar.div(lower, ar.mul(M, t_norm_hi)).lo)
        if c_alpha <= 0:
            break
        s_alpha = ar.sqrt(1 - c_alpha*c_alpha).hi
        if at_lo:
            rest = _pieces(geom, J[1], geom.hi) if J[1] < geom.hi else []
        else:
            rest = _pieces(geom, geom.lo, J[0]) if J[0] > geom.lo else []
        Jp = [_Piece(geom, p, q) for p, q in geom.base_pieces() if p >= J[0] and q <= J[1]]
        out.append((Jp, rest, t, c_alpha, s_alpha))
    return out


def _s3_rest_apart(ar, rest, other, rho12, policy):
    """(S3): the rest of one curve (outside its initial piece J) is certified farther than rho12."""
    return _apart(ar, rest, other, rho12, policy)


def _sin_theta_lower(ar, cone1, cone2):
    """Lower bound of sin(theta_V), theta_V = min(pi/2, angle(t1, t2) - alpha1 - alpha2) (S2).

    alpha_j is the certified half-angle arccos(c_alpha_j) itself (cos alpha_j = c_alpha_j exactly,
    sin alpha_j = sqrt(1 - c_alpha_j^2) enclosed), never formed as a float angle (N1). With
    x = angle(t1, t2) - alpha1 - alpha2 <= pi: if alpha1 + alpha2 < pi/2 and cos x <= 0, then
    x >= pi/2 and theta_V = pi/2; otherwise sin x > 0 gives x in (0, pi) and
    sin(theta_V) >= min(1, sin x).
    """
    t1, t2 = cone1[2], cone2[2]
    n1, n2 = ar.sqrt(_dot(t1, t1)), ar.sqrt(_dot(t2, t2))
    den = ar.mul(n1, n2)
    cos12 = ar.div(_dot(t1, t2), den)
    cr = _cross(t1, t2)
    sin12 = ar.div(ar.sqrt(_dot(cr, cr)), den)
    c1, c2 = cone1[3], cone2[3]
    s1, s2 = ar.sqrt(1 - c1*c1), ar.sqrt(1 - c2*c2)
    cos_s = ar.sub(c1*c2, ar.mul(s1, s2))
    sin_s = ar.add(ar.mul(s1, c2), ar.mul(s2, c1))
    cos_x = ar.add(ar.mul(cos12, cos_s), ar.mul(sin12, sin_s))
    sin_x = ar.sub(ar.mul(sin12, cos_s), ar.mul(cos12, sin_s))
    if cos_s.lo > 0 and cos_x.hi <= 0:
        return Fraction(1)
    return min(Fraction(1), sin_x.lo)


def _vertex_separation(ar, geom1, which1, geom2, which2, V, rho1, rho2, policy, r_limit=None, tries=4):
    """Step 6(iii): bound on d(x, V) from whole-piece cones (S1), angle (S2), separated rest (S3).

    ``r_limit`` only skips candidates whose bound could not be used anyway (r_V > r_max).
    """
    rho12 = rho1 + rho2
    d1 = ar.norm_hi(*geom1.end_point(ar, which1), V)
    d2 = ar.norm_hi(*geom2.end_point(ar, which2), V)
    dV = max(d1, d2)
    cones1, cones2 = _cones(ar, geom1, which1, policy), _cones(ar, geom2, which2, policy)
    candidates = []
    for c1 in cones1:
        for c2 in cones2:
            sin_lo = _sin_theta_lower(ar, c1, c2)
            if sin_lo > 0:
                r = min(rho1, rho2) + ar.div(rho12 + 2*dV, sin_lo).hi + dV
                candidates.append((r, sin_lo, c1, c2))
    candidates.sort(key=lambda item: item[0])
    if r_limit is not None:
        usable = [c for c in candidates if c[0] <= r_limit]
        if candidates and not usable:
            raise _Refusal("separation:bound_above_r_max")
        candidates = usable
    whole1, whole2 = _pieces(geom1), _pieces(geom2)
    for r, sin_lo, c1, c2 in candidates[:tries]:
        if _s3_rest_apart(ar, c1[1], whole2, rho12, policy) and _s3_rest_apart(ar, whole1, c2[1], rho12, policy):
            return r, {"sin_theta_lo": _down(sin_lo), "delta_V": _up(dV)}
    if not candidates:
        raise _Refusal("separation:no_cone")
    raise _Refusal("separation:rest_not_separated")


def vertex_separation_bound(curve1, curve2, vertex, rho1, rho2, *, policy=None):
    """Public test hook for step 6(iii): r_V for two curves that both end at ``vertex``.

    Curves are LinePath / EllipticArc (or objects with the same fields holding exact
    Fractions); each must have one end at the vertex. Raises ``SPRefusal`` when undecided.
    """
    policy = policy or SPCertificatePolicy()
    ar = _Ar(policy, None)
    V = _vec(vertex, "vertex")
    geoms = [_make_curve(curve1, policy, "c1"), _make_curve(curve2, policy, "c2")]
    ends = []
    for g in geoms:
        distances = [ar.norm_hi(*g.end_point(ar, w), V) for w in (0, 1)]
        ends.append(0 if distances[0] <= distances[1] else 1)
    r, _info = _vertex_separation(ar, geoms[0], ends[0], geoms[1], ends[1], V,
                                  _exact(rho1, "rho1"), _exact(rho2, "rho2"), policy)
    return r


# ----------------------------------------------------------------------------------- the pair

def contract_r_max(tolerance):
    """r_max: the exact rational tol/1000, rounded DOWN to binary64 (ADR 4.3 step 7, N2-N3)."""
    tol = _exact(tolerance, "tolerance")
    if tol <= 0:
        raise _Refusal("tolerance_not_positive")
    exact = tol/1000
    value = _to_float(exact, "r_max")                  # tol is inside the window: no overflow
    if Fraction(value) > exact:
        value = math.nextafter(value, -math.inf)
    if not (value > 0 and _WINDOW_LO <= Fraction(value) <= _WINDOW_HI) or value < 2.0**-1022:
        raise _Refusal("numeric_window:r_max")
    return value


def _normalise(n, k):
    """Scale a plane equation by a power of two so that |n| is near 1 (exact; the plane is unchanged)."""
    size = max(abs(x) for x in n)
    e = -(size.numerator.bit_length() - size.denominator.bit_length())
    scale = Fraction(2)**e
    return tuple(x*scale for x in n), k*scale


def _proportional(n1, k1, n2, k2):
    if _cross(n1, n2) != (0, 0, 0):
        return False
    den = _dot(n2, n2)
    lam = _dot(n1, n2)/den
    return lam != 0 and k1 == lam*k2


class _PairCtx:
    __slots__ = ("ar", "policy", "fns", "F1", "F2", "SE", "SV", "SVE", "TV", "r_max", "threshold_r")


def _fn(ctx, face, use, n, k, refined=False):
    refined = refined and use.geom.kind == "arc"
    key = (face.id, use.loop, use.index, n, k, refined)
    hit = ctx.fns.get(key)
    if hit is None:
        neg = (tuple(-x for x in n), -k)
        other = ctx.fns.get((face.id, use.loop, use.index, neg[0], neg[1], refined))
        if other is not None:
            hit = _negated_fn(other)
        else:
            hit = _curve_fn(ctx.ar, use.geom, n, k, refined)
        ctx.fns[key] = hit
    return hit


def _side_sup(ctx, face, n, k, s):
    return max(_flip(f.lo, f.hi, s)[1] for f in (_fn(ctx, face, u, n, k) for u in face.uses))


def _end_local(ctx, f, eps_c, s, smin):
    """Class (b) search: final pieces at one or both ends with certified sign (ADR 4.3 step 3(b)).

    Returns None or {"start": s_c or None, "end": s_c or None}.
    """
    pieces = f.pieces
    N = len(pieces)
    S = [i for i, p in enumerate(pieces) if _flip(p[0], p[1], s)[1] >= -eps_c]
    if not S:
        return None

    def slope_to_end(i):          # lower bound of d(s v)/ds moving towards the path end
        lo, hi = _flip(pieces[i][2], pieces[i][3], s*f.sgn)
        return lo/f.speed_hi if lo > 0 else None

    def slope_to_start(i):
        lo, hi = _flip(pieces[i][2], pieces[i][3], -s*f.sgn)
        return lo/f.speed_hi if lo > 0 else None
    best = None
    for k0 in range(N + 1):
        later = [i for i in S if i >= k0]
        k1 = N - min(later) if later else 0
        if k0 == 0 and k1 == 0:
            continue
        if k0 > 0 and k1 > 0 and k0 + k1 >= N:
            continue                                  # two final pieces need a nonempty rest
        if k0 + k1 > N:
            continue
        sc0 = sc1 = None
        if k0:
            sl = [slope_to_start(i) for i in range(k0)]
            if any(x is None or x < smin for x in sl):
                continue
            sc0 = min(sl)
        if k1:
            sl = [slope_to_end(i) for i in range(N - k1, N)]
            if any(x is None or x < smin for x in sl):
                continue
            sc1 = min(sl)
        worst = max(1/x for x in (sc0, sc1) if x is not None)
        if best is None or worst < best[0]:
            best = (worst, {"start": sc0, "end": sc1})
    return None if best is None else best[1]


def _contacts(ctx, face, n, k, s, eps, n_hi):
    """Contact elements of a face in the band |v| <= eps (ADR 4.3 step 3, Lemma B)."""
    smin = ctx.policy.s_min
    reach = {}                                        # element -> reach
    classes = {"o": 0, "a": 0, "b": 0, "vertex": 0}
    s_c_min = None

    def add(element, value):
        reach[element] = max(reach.get(element, Fraction(0)), value)
    for u in face.uses:
        f = _fn(ctx, face, u, n, k)
        eps_c = eps + n_hi*u.dS
        lo, hi = _flip(f.lo, f.hi, s)
        if hi < -eps_c:
            classes["o"] += 1
            continue
        local = _end_local(ctx, f, eps_c, s, smin)
        if local is None and u.geom.kind == "arc":
            # Retry on the end-refined pieces (finer final pieces near both ends).
            local = _end_local(ctx, _fn(ctx, face, u, n, k, refined=True), eps_c, s, smin)
        if local is None:
            classes["a"] += 1
            add(("E", u.edge), u.dS)
            continue
        classes["b"] += 1
        for key, vid in (("start", u.vs), ("end", u.ve)):
            sc = local[key]
            if sc is not None:
                s_c_min = sc if s_c_min is None else min(s_c_min, sc)
                add(("V", vid), u.dS + 2*eps_c/sc + face.dV[vid])
    for u, nxt, vid, d in face.loop_vertices:
        margin = n_hi*face.lam*d
        e1 = _flip(_fn(ctx, face, u, n, k).v1.lo, _fn(ctx, face, u, n, k).v1.hi, s)[1]
        e2 = _flip(_fn(ctx, face, nxt, n, k).v0.lo, _fn(ctx, face, nxt, n, k).v0.hi, s)[1]
        if e1 >= -(eps + n_hi*u.dS) - margin or e2 >= -(eps + n_hi*nxt.dS) - margin:
            classes["vertex"] += 1
            add(("V", vid), face.lam*d + face.dSV[vid])
    rho_gamma = max(reach.values(), default=Fraction(0))
    return reach, rho_gamma, classes, s_c_min


def _gradient(ctx, face, n, k, s, n2):
    """Certified lower bound g_F of the ascent rate over the chart hull (ADR 4.3 step 4)."""
    ar, smin = ctx.ar, ctx.policy.s_min
    if face.kind == "plane":
        nt2 = n2 - _dot(n, face.nF)**2/face.nF2
        if nt2 <= 0:
            raise _Refusal("gradient:plane_parallel_to_P")
        g = ar.sqrt(nt2).lo
        family = "plane"
    else:
        na = _dot(n, face.a)
        g = ar.div(abs(na), ar.sqrt(face.a2)).lo if na != 0 else Fraction(0)
        family = "generator"
        if g < smin:
            # Circumferential family: circles at constant z, rate r_hat . (n x a_hat) =
            # (w . (n x a)) / sqrt(H), certified over every stored curve and vertex (the chart
            # hull's angles; the rate is a sinusoid in the angle, concave where positive, so
            # connectors are covered by their end points). ADR 4.3 step 4 states this family
            # for axial planes (n.a = 0); neither the rate formula nor Lemma C uses n.a = 0, so
            # it is also tried when the generator rate is below s_min (a plate on a generator
            # whose stored normal is not exactly perpendicular to the axis). A support tangent
            # to P still fails: its rate vanishes at the tangent generator.
            m = _cross(n, face.a)
            km = _dot(m, face.A)
            samples = []
            for u in face.uses:
                f = _fn(ctx, face, u, m, km)
                samples.append((f.lo, f.hi, u.Hhi))
            for vid in face.vertex_ids:
                w = _vsub(face.vertices[vid], face.A)
                val = _dot(w, m)
                samples.append((val, val, face._H(w)))
            sign = 0
            for lo, hi, _h in samples:
                if lo > 0:
                    sign = 1
                    break
                if hi < 0:
                    sign = -1
                    break
            if sign == 0:
                raise _Refusal("gradient:circumferential_rate")
            g = None
            for lo, hi, h in samples:
                low = lo if sign > 0 else -hi
                if low <= 0 or h is None or h <= 0:
                    raise _Refusal("gradient:circumferential_rate")
                ratio = ar.div(low, ar.sqrt(h)).lo
                g = ratio if g is None else min(g, ratio)
            family = "circumferential"
    if g < smin:
        raise _Refusal("gradient:below_s_min")
    return g, family


def _shared_bound(ctx, face, element, rho):
    """Bound on d(x, T_out) when the contact element is shared topology (case (i)), else None."""
    kind, ident = element
    if kind == "E":
        if ident in ctx.SE:
            return rho + face.edges[ident].rep_hi
        return None
    if ident in ctx.TV:
        return rho
    if ident in ctx.SVE:
        best = None
        for e in ctx.SE:
            use = ctx.F1.edges[e]
            if ident in use.edge_end:
                d = use.edge_end[ident]
                best = d if best is None else min(best, d)
        return None if best is None else rho + best
    return None


def _band_hopeless(ctx, eps, n_hi):
    """Decline a plane whose band alone exceeds r_max (a decline only: never a soundness step).

    Every gradient rate is at most |n| (|n_t|, |n.a_hat| and |n x a_hat| are), so rho_F >=
    2 eps/|n|, and every bound of case (i), (iii) or the branch is >= min(rho1, rho2). Such a
    plane can only certify when no pair is decided by (i)/(iii) (all contact pairs apart, or
    no contact elements); that capability is given up for speed.
    """
    if 2*eps/n_hi > ctx.r_max:
        raise _Refusal("sides:band_above_r_max")


def _element_vertices(face, element):
    kind, ident = element
    if kind == "V":
        return {ident}
    use = face.edges[ident]
    return {use.vs, use.ve}


def _element_pieces(face, element):
    kind, ident = element
    if kind == "V":
        V = face.vertices[ident]
        return [_Piece(None, 0, 0, point=V)]
    return _pieces(face.edges[ident].geom)


def _attempt(ctx, n, k, source):
    """One candidate plane with F1 on v <= eps and F2 on v >= -eps. Returns (r_hi, evidence)."""
    ar, F1, F2, policy = ctx.ar, ctx.F1, ctx.F2, ctx.policy
    ar.phase("plane attempt")
    n2 = _dot(n, n)
    n_hi = ar.sqrt(n2).hi
    in1 = F1.kind == "plane" and _proportional(n, k, F1.nF, F1.kF)
    in2 = F2.kind == "plane" and _proportional(n, k, F2.nF, F2.kF)
    ev = {"plane": source}
    if in1 and in2:
        raise _Refusal("branch:coplanar")
    if in1 or in2:
        # Plane-support branch (step 5): only F_other's sides, contacts and gradient are used.
        other, s = (F2, -1) if in1 else (F1, 1)
        U = _side_sup(ctx, other, n, k, s)
        eps = max(U, Fraction(0)) + n_hi*other.dGamma
        _band_hopeless(ctx, eps, n_hi)
        reach, rho_gamma, classes, s_c = _contacts(ctx, other, n, k, s, eps, n_hi)
        g, family = _gradient(ctx, other, n, k, s, n2)
        rho = 2*eps/g + rho_gamma
        r = Fraction(0)
        for element in reach:
            b = _shared_bound(ctx, other, element, rho)
            if b is None:
                raise _Refusal("branch:contact_not_shared")
            r = max(r, b)
        ev.update(case="plane_support_branch", eps=_up(eps), g={other.id: _down(g)}, family={other.id: family},
                  s_c=None if s_c is None else _down(s_c), rho={other.id: _up(rho)},
                  rho_gamma={other.id: _up(rho_gamma)}, classes={other.id: classes},
                  deciding={"branch": len(reach)}, dominant=_dominant(eps, g, rho_gamma, None))
        return r, ev
    U1 = _side_sup(ctx, F1, n, k, 1)
    U2 = _side_sup(ctx, F2, n, k, -1)
    eps = max(max(U1, Fraction(0)) + n_hi*F1.dGamma, max(U2, Fraction(0)) + n_hi*F2.dGamma)
    _band_hopeless(ctx, eps, n_hi)
    reach1, rg1, cl1, sc1 = _contacts(ctx, F1, n, k, 1, eps, n_hi)
    reach2, rg2, cl2, sc2 = _contacts(ctx, F2, n, k, -1, eps, n_hi)
    g1, fam1 = _gradient(ctx, F1, n, k, 1, n2)
    g2, fam2 = _gradient(ctx, F2, n, k, -1, n2)
    rho1, rho2 = 2*eps/g1 + rg1, 2*eps/g2 + rg2
    rho12 = rho1 + rho2
    r = Fraction(0)
    deciding = {"i": 0, "ii": 0, "iii": 0}
    worst_iii = None
    for e1 in reach1:
        for e2 in reach2:
            ar.tick()
            options = [b for b in (_shared_bound(ctx, F1, e1, rho1), _shared_bound(ctx, F2, e2, rho2))
                       if b is not None]
            if options:
                deciding["i"] += 1
                r = max(r, min(options))
                if r > ctx.r_max:
                    ctx.threshold_r = max(ctx.threshold_r or 0, r)
                    raise _Refusal("threshold:r_pair_above_r_max")
                continue
            # (ii) is skipped for elements incident to a common vertex id: they touch, so a
            # distance proof cannot succeed and would only spend budget (capability, not soundness).
            if not (_element_vertices(F1, e1) & _element_vertices(F2, e2)) and \
                    _apart(ar, _element_pieces(F1, e1), _element_pieces(F2, e2), rho12, policy):
                deciding["ii"] += 1
                continue
            if e1[0] == "E" and e2[0] == "E":
                u1, u2 = F1.edges[e1[1]], F2.edges[e2[1]]
                common = ({u1.vs, u1.ve} & {u2.vs, u2.ve}) & ctx.SV
                done = None
                for vid in sorted(common):
                    w1 = 0 if u1.vs == vid else 1
                    w2 = 0 if u2.vs == vid else 1
                    if u1.vs == u1.ve or u2.vs == u2.ve:
                        continue
                    try:
                        rv, _info = _vertex_separation(ar, u1.geom, w1, u2.geom, w2, F1.vertices[vid],
                                                       rho1, rho2, policy, r_limit=ctx.r_max)
                    except _Refusal as exc:
                        if str(exc).startswith("qualification_budget") or str(exc).startswith("rational_bit"):
                            raise
                        continue
                    extra = 0 if vid in ctx.TV else _shared_bound(ctx, F1, ("V", vid), 0)
                    if extra is None:
                        continue
                    done = rv + extra
                    break
                if done is not None:
                    deciding["iii"] += 1
                    r = max(r, done)
                    worst_iii = done if worst_iii is None else max(worst_iii, done)
                    continue
            raise _Refusal("combination:pair_undecided")
    sc = [x for x in (sc1, sc2) if x is not None]
    ev.update(case="separating_plane", eps=_up(eps), g={F1.id: _down(g1), F2.id: _down(g2)},
              family={F1.id: fam1, F2.id: fam2}, s_c=_down(min(sc)) if sc else None,
              rho={F1.id: _up(rho1), F2.id: _up(rho2)}, rho_gamma={F1.id: _up(rg1), F2.id: _up(rg2)},
              classes={F1.id: cl1, F2.id: cl2}, deciding=deciding,
              dominant=_dominant(eps, min(g1, g2), max(rg1, rg2), worst_iii))
    return r, ev


def _dominant(eps, g, rho_gamma, iii):
    terms = {"band_2eps_over_g": 2*eps/g, "boundary_rho_gamma": rho_gamma}
    if iii is not None:
        terms["vertex_separation"] = iii
    return max(terms, key=terms.get)


def _candidate_planes(F1, F2, SE, SV):
    out = []
    for face in (F1, F2):
        if face.kind == "plane":
            out.append((("support", face.id), face.nF, face.kF))
        for u in face.uses:
            touches = u.edge in SE or u.vs in SV or u.ve in SV
            if not touches:
                continue
            g = u.geom
            if g.kind == "arc":
                n = _cross(g.U, g.V)
                out.append((("arc", face.id, u.edge), n, _dot(n, g.C)))
            elif face.kind == "cyl":
                n = _cross(face.a, _vsub(g.S, face.A))
                if n != (0, 0, 0):
                    out.append((("axial", face.id, u.edge), n, _dot(n, g.S)))
    planes = []
    priority = {"support": 0, "axial": 1, "arc": 2}
    for source, n, k in sorted(out, key=lambda item: priority[item[0][0]]):
        n, k = _normalise(n, k)
        if any(_proportional(n, k, m, j) for _s, m, j in planes):
            continue
        planes.append((source, n, k))
    return planes


def _coincident_cylinders(F1, F2):
    if F1.kind != "cyl" or F2.kind != "cyl":
        return False
    return (_cross(F1.a, F2.a) == (0, 0, 0) and F1.R == F2.R
            and _cross(_vsub(F2.A, F1.A), F1.a) == (0, 0, 0))


class SPCertifier:
    """Certificate engine with per-face preparation reused across pairs (pure functions of records)."""

    def __init__(self, policy=None):
        self.policy = policy or SPCertificatePolicy()
        self._faces = {}

    def _face(self, record, check):
        key = id(record)
        hit = self._faces.get(key)
        if hit is not None and hit[0] is record:
            if isinstance(hit[1], str):
                raise _Refusal(hit[1])
            return hit[1]
        try:
            face = _Face(record, self.policy, check)
        except _Refusal as exc:
            self._faces[key] = (record, "face:" + str(exc))
            raise _Refusal("face:" + str(exc)) from None
        self._faces[key] = (record, face)
        return face

    def certify(self, first, second, tolerance, *, check=None):
        """CERTIFIED(r_pair, evidence) or UNDECIDED(reason, evidence) for one face pair.

        An exception raised by ``check`` propagates as the same object (CANCELLED, ADR 4.5);
        it is never converted to UNDECIDED and never cached with a face preparation.
        """
        try:
            return self._certify(first, second, tolerance, check)
        except _CallbackRaised as carrier:
            original = carrier.original
        raise original

    def _certify(self, first, second, tolerance, check):
        evidence = {"version": CERTIFICATE_VERSION,
                    "faces": (getattr(first, "face_id", None), getattr(second, "face_id", None))}
        try:
            r_max = contract_r_max(tolerance)
            evidence["r_max"] = r_max
            F1, F2 = self._face(first, check), self._face(second, check)
            ctx = _PairCtx()
            ctx.ar, ctx.policy, ctx.fns, ctx.F1, ctx.F2 = _Ar(self.policy, check), self.policy, {}, F1, F2
            ctx.r_max = Fraction(r_max)
            ctx.threshold_r = None
            ctx.SE = set(F1.edges) & set(F2.edges)
            ctx.SV = F1.vertex_ids & F2.vertex_ids
            ctx.SVE = {v for e in ctx.SE for v in (F1.edges[e].vs, F1.edges[e].ve)}
            ctx.TV = ctx.SV - ctx.SVE
            evidence.update(shared_edges=len(ctx.SE), shared_vertices=len(ctx.SV))
            if not ctx.SE and not ctx.SV:
                return Undecided("no_shared_topology", evidence)
            for e in ctx.SE:
                a, b = F1.edges[e], F2.edges[e]
                if {a.vs, a.ve} != {b.vs, b.ve}:
                    return Undecided("shared_edge_vertex_mismatch", evidence)
                # T_out holds ONE stored curve per shared edge: both faces must name the same record.
                if not _same_record(a.representative, b.representative):
                    return Undecided("shared_edge_representative_mismatch", evidence)
            if F1.kind == "plane" and F2.kind == "plane" and _proportional(F1.nF, F1.kF, F2.nF, F2.kF):
                return Undecided("coplanar_supports", evidence)
            coincident = _coincident_cylinders(F1, F2)
            evidence["coincident_support"] = coincident
            if coincident:
                if F1.support_key[:4] != F2.support_key[:4]:
                    return Undecided("coincident_support_ineligible:support_data", evidence)
                if any(F1.edges[e].forward == F2.edges[e].forward for e in ctx.SE):
                    return Undecided("coincident_support_ineligible:traversal", evidence)
                evidence["shadow_required"] = "_coplanar_traces"
            evidence["delta"] = {F1.id: _up(F1.dGamma), F2.id: _up(F2.dGamma)}
            evidence["delta_S"] = {F1.id: _up(F1.dS_max), F2.id: _up(F2.dS_max)}
            evidence["delta_V"] = {F1.id: _up(F1.dV_max), F2.id: _up(F2.dV_max)}
            evidence["delta_S_arc_on_cylinder"] = _up(max(F1.dS_arc_cyl, F2.dS_arc_cyl))
            planes = _candidate_planes(F1, F2, ctx.SE, ctx.SV)
            evidence["candidate_planes"] = len(planes)
            best, failures = None, []
            # The first attempt whose bound meets r_max decides the pair (any certified plane gives
            # a valid bound); further planes are only tried while no attempt has met it.
            for source, n, k in planes:
                if best is not None and best[0] <= ctx.r_max:
                    break
                for sign in (1, -1):
                    try:
                        r, ev = _attempt(ctx, tuple(sign*x for x in n), sign*k, source)
                    except _Refusal as exc:
                        text = str(exc)
                        if text.startswith(("qualification_budget", "rational_bit", "series_budget")):
                            raise
                        failures.append(text)
                        continue
                    ev["sign"] = sign
                    if best is None or r < best[0]:
                        best = (r, ev)
                    if r <= ctx.r_max:
                        break
            if best is None:
                evidence["attempts"] = failures
                if ctx.threshold_r is not None:
                    evidence["r_pair_lower_at_refusal"] = _down(ctx.threshold_r)
                return Undecided(_best_reason(failures) if failures else "no_candidate_plane", evidence)
            r, ev = best
            evidence.update(ev)
            evidence["r_pair_upper"] = _up(r)
            evidence["output_membership"] = self._membership(ctx)
            if not r <= Fraction(r_max):
                return Undecided("r_pair_above_r_max", evidence)
            evidence["delta_eval_max"] = _up(max((_delta_eval(F1.edges[e].representative, self.policy)
                                                  for e in ctx.SE), default=Fraction(0)))
            return Certified(_up(r), evidence)
        except _Refusal as exc:
            return Undecided(str(exc), evidence)

    @staticmethod
    def _membership(ctx):
        """delta_out per face: every output entity lies within it of that face's G_F (step 7)."""
        out = {}
        for face in (ctx.F1, ctx.F2):
            values = [face.edges[e].dS + face.edges[e].rep_hi for e in ctx.SE]
            values += [face.dSV[v] for v in ctx.TV]
            out[face.id] = _up(max(values, default=Fraction(0)))
        return out


_UNIT = Fraction(1, 1 << 53)                  # binary64 unit roundoff
_LIBM_TRIG_ERROR = Fraction(1, 1 << 50)       # assumed |fl(cos x) - cos x| bound of numpy's cos/sin


def _in_window(value):
    return value == 0 or _WINDOW_LO <= abs(value) <= _WINDOW_HI


def _delta_eval(curve, policy):
    """delta_eval (ADR 4.1 table): float ``evaluate`` at the curve ends vs the mathematical curve.

    Forward-error model from the operation count; it assumes no underflow or overflow, so every
    intermediate must lie in the N3 window, else UNDECIDED. numpy's cos/sin are assumed accurate
    to 2^-50 absolute (an assumption of this diagnostic, not used by any accept decision).
    """
    name = type(curve).__name__
    if name in ("LinePath", "_RawLine"):
        s = tuple(_binary64(x) for x in curve.start)
        e = tuple(_binary64(x) for x in curve.end)
        total = Fraction(0)
        for i in range(3):
            # evaluate(1.0) = s + 1.0*(e - s): every operand is a known float, so each
            # intermediate is computed exactly as IEEE does and checked (t = 0 returns s exactly).
            diff = _float_op(s[i], e[i], lambda x, y: y - x)
            result = _float_op(s[i], diff, lambda x, y: x + y)
            _window_value(diff)
            _window_value(result)
            total += abs(result - e[i])
        return total
    if name in ("EllipticArc", "_RawArc"):
        C, U, V = (tuple(_binary64(x) for x in getattr(curve, k)) for k in ("center", "u_vector", "v_vector"))
        start, sweep = _binary64(curve.start_angle), _binary64(curve.sweep_angle)
        end_angle = _float_op(start, sweep, lambda x, y: x + y)        # the float angle evaluate forms
        _window_value(end_angle)
        phase = abs(end_angle - (start + sweep))
        worst = Fraction(0)
        for angle, shift in ((start, Fraction(0)), (end_angle, phase)):
            if angle == 0:                      # sin(0) = 0 and cos(0) = 1 exactly, in libm too
                s, c = (Fraction(0),), (Fraction(1),)
            else:
                si, ci = _sincos_fixed(angle, policy.trig_argument_bits)
                s = (si.lo - _LIBM_TRIG_ERROR, si.hi + _LIBM_TRIG_ERROR)
                c = (ci.lo - _LIBM_TRIG_ERROR, ci.hi + _LIBM_TRIG_ERROR)
            total = Fraction(0)
            for i in range(3):
                # numpy: (C + cos*U) + sin*V, in that order; every intermediate is checked.
                p1 = _model_mul(U[i], c)
                p2 = _model_mul(V[i], s)
                s1 = _model_add(C[i], p1)
                s2 = _model_add(s1, p2)
                for item in (p1, p2, s1, s2):
                    _window_model(item)
                trig_error = (0 if angle == 0 else _LIBM_TRIG_ERROR) + shift
                total += (abs(U[i]) + abs(V[i]))*trig_error + 4*_UNIT*(abs(C[i]) + abs(U[i]) + abs(V[i]))
            worst = max(worst, total)
        return worst
    raise _Refusal("unsupported_curve:" + name)


def _binary64(value):
    """A stored value that must be an exact binary64 inside the N3 window (the evaluate model)."""
    made = _exact(value, "eval")
    if not _input_in_window(made):                     # a float input of the model: N3, before conversion
        raise _Refusal("numeric_window:delta_eval")
    if Fraction(_to_float(made, "delta_eval")) != made:
        raise _Refusal("numeric_window:delta_eval_not_binary64")
    return made


def _float_op(x, y, op):
    result = op(_to_float(x, "delta_eval"), _to_float(y, "delta_eval"))
    if not math.isfinite(result):
        raise _Refusal("numeric_window:delta_eval")
    return Fraction(result)


def _window_value(value):
    if not _in_window(value):
        raise _Refusal("numeric_window:delta_eval")


def _model_mul(coef, trig):
    """Float product coef*trig: an exact float when trig is exact, else an enclosure (lo, hi)."""
    if len(trig) == 1:
        return ("exact", _float_op(coef, trig[0], lambda x, y: x*y))
    if coef == 0:
        return ("exact", Fraction(0))
    a, b = coef*trig[0], coef*trig[1]
    lo, hi = min(a, b), max(a, b)
    pad = _UNIT*max(abs(lo), abs(hi))
    return ("interval", lo - pad, hi + pad)


def _model_add(x, y):
    """Float sum of a known float x (or model value) and a model value y."""
    if not isinstance(x, tuple):
        x = ("exact", x)
    if x[0] == "exact" and y[0] == "exact":
        return ("exact", _float_op(x[1], y[1], lambda a, b: a + b))
    xl, xh = (x[1], x[1]) if x[0] == "exact" else (x[1], x[2])
    yl, yh = (y[1], y[1]) if y[0] == "exact" else (y[1], y[2])
    lo, hi = xl + yl, xh + yh
    pad = _UNIT*max(abs(lo), abs(hi))
    return ("interval", lo - pad, hi + pad)


def _window_model(item):
    if item[0] == "exact":
        _window_value(item[1])
        return
    lo, hi = item[1], item[2]
    if lo <= 0 <= hi or min(abs(lo), abs(hi)) < _WINDOW_LO or max(abs(lo), abs(hi)) > _WINDOW_HI:
        raise _Refusal("numeric_window:delta_eval")


def _best_reason(failures):
    order = {"sides": 0, "gradient": 1, "branch": 1, "combination": 2, "separation": 3, "threshold": 4}
    return max(failures, key=lambda t: order.get(t.split(":")[0], 0))


def certify_face_pair(first, second, tolerance, *, policy=None, check=None):
    """One-shot API (no reuse of face preparation)."""
    return SPCertifier(policy).certify(first, second, tolerance, check=check)


# ----------------------------------------------------------------------------------- model reader

def face_record_from_model(model, face_id):
    """FaceRecord of a model face: MaterialDomain support and paths, model vertex ids (input only)."""
    from .material_arrangement import MaterialDomain
    from .arrangement_geometry import freeze_edge
    from .surfaces import Cylinder, Plane
    domain = MaterialDomain.from_model(model, face_id)
    face = model.faces[face_id]
    support = domain.support
    def floats(values, what):
        return tuple(_stored_float(x, what) for x in values)
    if type(support) is Plane:
        record_support = PlaneSupport(floats(support.origin, "plane.origin"), floats(support.u_vector, "plane.u"),
                                      floats(support.v_vector, "plane.v"))
    elif type(support) is Cylinder:
        record_support = CylinderSupport(floats(support.origin, "cylinder.origin"),
                                         floats(support.axis, "cylinder.axis"),
                                         _stored_float(support.radius, "cylinder.radius"),
                                         floats(support.radial_direction, "cylinder.radial"))
    else:
        record_support = support
    loops, vids = [], set()
    model_loops = (face.loop, *face.holes)
    if len(model_loops) != len(domain.boundaries):
        raise _Refusal("record:loop_count")
    for paths, uses in zip(domain.boundaries, model_loops):
        if len(paths) != len(uses):
            raise _Refusal("record:loop_length")
        items = []
        for path, use in zip(paths, uses):
            if path.source_edge != use.edge:
                raise _Refusal("record:edge_order")
            edge = model.edges[use.edge]
            start = edge.start if use.forward else edge.end
            end = edge.end if use.forward else edge.start
            vids.update((start, end))
            items.append(BoundaryUse(path.curve, use.edge, start, end, bool(use.forward),
                                     freeze_edge(model, use.edge)))
        loops.append(tuple(items))
    vertices = tuple((v, floats(model.vertex_position(v), f"vertex.{v}")) for v in sorted(vids))
    return FaceRecord(face_id, record_support, tuple(loops), vertices)


def _stored_float(value, what):
    """A model value copied into a record as the binary64 it already is; a value that is not
    exactly a binary64 (or not a number) is a refusal instead of a silent rounding."""
    made = _rational(value, what)
    if isinstance(value, float):                       # float and numpy.float64: exact, keeps -0.0
        return float(value)
    result = _to_float(made, what)
    if Fraction(result) != made:
        raise _Refusal(f"record:not_binary64:{what}")
    return result
