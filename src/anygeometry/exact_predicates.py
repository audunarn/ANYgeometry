"""Certified decisions independent of floating-point rounding (robust-predicate toolkit, slice R0).

Architecture: docs/agent_plans/GENERALITY_ARCHITECTURE.md, section R.2. Nothing
in the engine calls this module yet (R0 changes no engine route); slices R1-R6
move decisions onto it one inventory entry at a time.

* T1 exact rational layer: stored doubles are exact rationals; dot and cross
  products, squared distances and small linear solves are evaluated in
  ``Fraction`` and compared exactly.
* T2 canonical enclosures for non-rational quantities (square roots,
  sin/cos/atan2 and anything depending on an isolated root): one
  deterministic procedure in ``Fraction`` interval arithmetic, outward-rounded
  to a grid of ``2**-(80 + 32 L)`` at level ``L = 0 .. L_MAX``.
* T3 one decision policy: :func:`certified_compare` returns ``BELOW`` or
  ``ABOVE`` (or ``EQUAL`` where a rational predicate allows it) or raises
  :class:`DegenerateWithinBound`; a faster route may decide through
  :func:`filter_compare` only when it is separated from the threshold by more
  than its own error bound plus ``W_max`` (the bound of the canonical level-
  ``L_MAX`` enclosure width), so every route returns the same outcome and the
  same refusals.
* T0 stations: a decision point is a definition plus an exact parameter (a
  stored double or an isolating interval of an exact polynomial root), not an
  evaluated tuple.
* T4 exact chart predicates on rationals (orientation, point in polygon).
* T7 operand-order symmetry: binary predicates evaluate on canonically
  ordered operands.
* T8 decision trace: every certified decision is counted in
  :mod:`anygeometry.runtime_diagnostics` and recorded by
  :func:`collect_decisions`.

No symbolic perturbation: a degenerate decision is refused, never resolved by
an invented order (T5).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from typing import Sequence

import numpy as np

from .errors import GeometryError
from .runtime_diagnostics import _increment

__all__ = [
    "Box",
    "Canonical",
    "CertificationPolicyError",
    "DegenerateWithinBound",
    "L_MAX",
    "Outcome",
    "RootInterval",
    "Station",
    "canonical_extent",
    "canonical_order",
    "correctly_rounded_sqrt",
    "certified_compare",
    "collect_decisions",
    "exact",
    "exact_cross",
    "exact_dot",
    "exact_squared_distance",
    "filter_compare",
    "orientation_2d",
    "point_in_polygon",
    "w_max",
]

L_MAX = 4
_BASE_BITS = 80
_LEVEL_BITS = 32
_MAX_OPERATIONS = 200_000


def grid_bits(level: int) -> int:
    """Binary grid of the canonical procedure at ``level`` (enclosures are multiples of 2**-bits)."""
    return _BASE_BITS + _LEVEL_BITS * level


# ----------------------------------------------------------------------------------- outcomes, refusals, trace


class Outcome(Enum):
    BELOW = "below"
    EQUAL = "equal"
    ABOVE = "above"


def _describe(value: Fraction) -> str:
    """A short, always-representable description of an exact rational."""
    try:
        return repr(float(value))
    except OverflowError:
        return f"{'-' if value < 0 else ''}~2**{abs(value.numerator).bit_length() - value.denominator.bit_length()}"


class CertificationPolicyError(GeometryError):
    """A registered predicate kind exceeded its documented width bound (an internal contract violation).

    Raised instead of deciding: the filter/canonical agreement of that kind is
    no longer guaranteed, so the operation fails closed.
    """


class DegenerateWithinBound(GeometryError):
    """A decision the canonical procedure cannot make.

    Either its level-``L_MAX`` enclosure ``enclosure`` still contains the
    threshold (degenerate within the certified bound), or the quantity could
    not be evaluated at any level (``enclosure`` is ``None``: for example a
    denominator enclosure that contains zero, or a branch cut).
    """

    def __init__(self, predicate: str, enclosure, threshold, entities=()):
        self.predicate, self.enclosure, self.threshold, self.entities = predicate, enclosure, threshold, tuple(entities)
        if enclosure is None:
            detail = "not evaluable by the canonical procedure at any level"
        else:
            detail = (f"degenerate within the certified bound (enclosure [{_describe(enclosure.lo)}, "
                      f"{_describe(enclosure.hi)}] at level {L_MAX})")
        super().__init__(f"{predicate}: {detail} against threshold {_describe(threshold)}; entities {self.entities}")


_TRACE: ContextVar[tuple] = ContextVar("anygeometry_exact_predicates_trace", default=())


@dataclass(frozen=True, slots=True)
class Decision:
    """One certified decision (T8)."""
    predicate: str
    entities: tuple
    route: str          # "exact", "filter" or "canonical"
    level: int | None   # canonical level that decided, if any
    outcome: str        # Outcome value or "refused"


@contextmanager
def collect_decisions():
    """Collect every certified decision made in this scope (nested collectors each receive all)."""
    records: list[Decision] = []
    token = _TRACE.set((*_TRACE.get(), records))
    try:
        yield records
    finally:
        _TRACE.reset(token)


def _record(predicate, entities, route, level, outcome):
    _increment("certified_decisions")
    _increment({"exact": "exact_decisions", "filter": "filter_decisions", "canonical": "canonical_decisions"}[route])
    if outcome == "refused":
        _increment("undecided_refusals")
    collectors = _TRACE.get()
    if collectors:
        decision = Decision(predicate, tuple(entities), route, level, outcome)
        for records in collectors:
            records.append(decision)


# ----------------------------------------------------------------------------------- T1 exact rational layer


def exact(value) -> Fraction:
    """The exact rational value of a stored double (or int / Fraction); non-finite values are refused."""
    if type(value) is Fraction:
        return value
    if isinstance(value, bool):
        raise GeometryError("a boolean is not a geometric quantity")
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise GeometryError("non-finite geometric quantity")
        return Fraction(*value.as_integer_ratio())
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    raise GeometryError(f"not an exact geometric quantity: {type(value).__name__}")


def exact_vector(values: Sequence) -> tuple[Fraction, ...]:
    return tuple(exact(value) for value in values)


def exact_dot(a: Sequence, b: Sequence) -> Fraction:
    return sum((exact(x) * exact(y) for x, y in zip(a, b, strict=True)), Fraction(0))


def exact_cross(a: Sequence, b: Sequence) -> tuple[Fraction, Fraction, Fraction]:
    (a0, a1, a2), (b0, b1, b2) = exact_vector(a), exact_vector(b)
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


def exact_squared_distance(a: Sequence, b: Sequence) -> Fraction:
    return sum(((exact(x) - exact(y)) ** 2 for x, y in zip(a, b, strict=True)), Fraction(0))


def correctly_rounded_sqrt(value) -> float:
    """The double nearest to ``sqrt(value)`` for an exact non-negative rational (ties impossible).

    Integer square root with 64 guard bits and a sticky bit, then CPython's
    correctly rounded Fraction -> float conversion: a deterministic function of
    the exact input, independent of libm and of summation order.
    """
    value = exact(value)
    if value < 0:
        raise GeometryError("square root of a negative quantity")
    if value == 0:
        return 0.0
    shift = max(0, 64 + 53 - (value.numerator.bit_length() - value.denominator.bit_length()) // 2)
    scaled = value * (1 << (2 * shift))
    whole = scaled.numerator // scaled.denominator
    root = math.isqrt(whole)
    is_exact = root * root == whole and scaled.denominator == 1
    return float(Fraction(2 * root + (0 if is_exact else 1), 1 << (shift + 1)))


def canonical_extent(lower: Sequence, upper: Sequence) -> float:
    """Correctly rounded length of the box diagonal ``upper - lower`` (exact differences and squares)."""
    return correctly_rounded_sqrt(sum(((exact(b) - exact(a)) ** 2 for a, b in zip(lower, upper)), Fraction(0)))


# ----------------------------------------------------------------------------------- T2 canonical interval procedure


@dataclass(frozen=True, slots=True)
class Box:
    """A closed rational interval ``[lo, hi]``."""
    lo: Fraction
    hi: Fraction

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def __contains__(self, value) -> bool:
        return self.lo <= exact(value) <= self.hi


class _BudgetExhausted(Exception):
    """Undecided at this level (budget, zero-straddling denominator, ambiguous quadrant); a finer level may succeed."""


class _NotEvaluable(_BudgetExhausted):
    """Undefined or discontinuous for the enclosed inputs at every level (e.g. the atan2 branch cut)."""


class Canonical:
    """Outward-rounded ``Fraction`` interval arithmetic on the grid of one level.

    Deterministic: the result depends only on the inputs and the level. Each
    operation rounds outward to multiples of ``2**-grid_bits(level)``.
    """

    def __init__(self, level: int):
        if not 0 <= level <= L_MAX:
            raise ValueError("canonical level out of range")
        self.level = level
        self.bits = grid_bits(level)
        self.grid = 1 << self.bits
        self.operations = 0
        self._pi = None

    # -- primitives

    def _charge(self) -> None:
        self.operations += 1
        if self.operations > _MAX_OPERATIONS:
            raise _BudgetExhausted()

    def _round(self, lo: Fraction, hi: Fraction) -> Box:
        g = self.grid
        return Box(Fraction((lo.numerator * g) // lo.denominator, g),
                   Fraction(-((-hi.numerator * g) // hi.denominator), g))

    def box(self, value, hi=None) -> Box:
        """An exact point (stored double) or interval, without rounding."""
        if isinstance(value, Box) and hi is None:
            return value
        lo = exact(value)
        upper = lo if hi is None else exact(hi)
        if lo > upper:
            raise ValueError("reversed interval")
        return Box(lo, upper)

    def add(self, a, b) -> Box:
        self._charge()
        a, b = self.box(a), self.box(b)
        return self._round(a.lo + b.lo, a.hi + b.hi)

    def neg(self, a) -> Box:
        a = self.box(a)
        return Box(-a.hi, -a.lo)

    def sub(self, a, b) -> Box:
        return self.add(a, self.neg(b))

    def mul(self, a, b) -> Box:
        self._charge()
        a, b = self.box(a), self.box(b)
        products = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
        return self._round(min(products), max(products))

    def div(self, a, b) -> Box:
        self._charge()
        a, b = self.box(a), self.box(b)
        if b.lo <= 0 <= b.hi:
            raise _BudgetExhausted()          # division by an interval containing zero: undecided at this level
        quotients = (a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi)
        return self._round(min(quotients), max(quotients))

    def square(self, a) -> Box:
        self._charge()
        a = self.box(a)
        values = (a.lo * a.lo, a.hi * a.hi)
        low = Fraction(0) if a.lo <= 0 <= a.hi else min(values)
        return self._round(low, max(values))

    def sqrt(self, a) -> Box:
        self._charge()
        a = self.box(a)
        if a.lo < 0:
            if a.hi < 0:
                raise GeometryError("square root of a negative quantity")
            a = Box(Fraction(0), a.hi)
        g = self.grid

        def floor_root(x: Fraction) -> int:
            return math.isqrt((x.numerator * g * g) // x.denominator)

        low, high = floor_root(a.lo), floor_root(a.hi)
        if Fraction(high * high, g * g) < a.hi:
            high += 1
        return Box(Fraction(low, g), Fraction(high, g))

    def dot(self, a: Sequence, b: Sequence) -> Box:
        result = self.box(0)
        for x, y in zip(a, b, strict=True):
            result = self.add(result, self.mul(x, y))
        return result

    def norm(self, vector: Sequence) -> Box:
        boxes = [self.box(value) for value in vector]
        if all(item.lo == item.hi for item in boxes):
            # Exact inputs: the sum of squares is exact; round once, in the root
            # (width at most one grid unit at any magnitude, also near zero).
            self._charge()
            return self.sqrt(Box(*(sum((item.lo * item.lo for item in boxes), Fraction(0)),) * 2))
        result = self.box(0)
        for item in boxes:
            result = self.add(result, self.square(item))
        return self.sqrt(result)

    # -- transcendental functions

    def _remainder_padding(self, bound: Fraction) -> Box:
        return Box(-bound, bound)

    def _atan_series(self, x: Box) -> Box:
        """atan on |x| <= 1/2 by its alternating series with a rigorous tail bound."""
        magnitude = max(abs(x.lo), abs(x.hi))
        if magnitude > Fraction(1, 2):
            raise ValueError("atan series argument out of range")
        power, square, result = x, self.square(x), self.box(0)
        target = Fraction(1, self.grid * 16)
        for n in range(4 * self.bits):
            term = self.div(power, 2 * n + 1)
            result = self.add(result, term if n % 2 == 0 else self.neg(term))
            power = self.mul(power, square)
            tail = magnitude ** (2 * n + 3) / (2 * n + 3)
            if tail <= target:
                return self.add(result, self._remainder_padding(tail))
        raise _BudgetExhausted()

    def pi(self) -> Box:
        if self._pi is None:
            # Machin: pi = 16 atan(1/5) - 4 atan(1/239).
            self._pi = self.sub(self.mul(16, self._atan_series(self.box(Fraction(1, 5)))),
                                self.mul(4, self._atan_series(self.box(Fraction(1, 239)))))
        return self._pi

    def atan(self, x) -> Box:
        x = self.box(x)
        if x.hi < 0:
            return self.neg(self.atan(self.neg(x)))
        if x.lo < 0:
            return Box(self.atan(self.box(x.lo)).lo, self.atan(self.box(x.hi)).hi)
        if x.lo > 1:
            return self.sub(self.div(self.pi(), 2), self.atan(self.div(1, x)))
        if x.hi > 1:
            return Box(self.atan(self.box(x.lo)).lo, self.atan(self.box(x.hi)).hi)
        # atan x = 2 atan(x / (1 + sqrt(1 + x^2))), argument <= tan(pi/8) < 1/2
        reduced = self.div(x, self.add(1, self.sqrt(self.add(1, self.square(x)))))
        return self.mul(2, self._atan_series(reduced))

    def atan2(self, y, x) -> Box:
        y, x = self.box(y), self.box(x)
        if x.lo > 0:
            return self.atan(self.div(y, x))
        if y.lo > 0:
            return self.sub(self.div(self.pi(), 2), self.atan(self.div(x, y)))
        if y.hi < 0:
            return self.sub(self.neg(self.div(self.pi(), 2)), self.atan(self.div(x, y)))
        if x.hi < 0 and y.lo >= 0:
            # Closed upper side of the cut: atan2(0, x<0) = +pi (exact y = 0 is the real number 0).
            return self.add(self.pi(), self.atan(self.div(y, x)))
        if x.hi < 0 and y.hi < 0:
            return self.sub(self.atan(self.div(y, x)), self.pi())
        # y straddles or touches zero from below with x < 0 (the branch cut), or the
        # origin is enclosed: atan2 is discontinuous there; no single interval is valid.
        raise _NotEvaluable()

    def sincos(self, x) -> tuple[Box, Box]:
        """Enclosures of (sin x, cos x)."""
        x = self.box(x)
        pi = self.pi()
        half = self.div(pi, 2)
        quotient = self.div(x, half)
        low = math.floor(quotient.lo + Fraction(1, 2))
        high = math.floor(quotient.hi + Fraction(1, 2))
        if low != high:
            raise _BudgetExhausted()      # quadrant not unique at this level
        reduced = self.sub(x, self.mul(low, half))
        magnitude = max(abs(reduced.lo), abs(reduced.hi))
        if magnitude > Fraction(4, 5):
            raise _BudgetExhausted()
        power, factorial, sine, cosine = self.box(1), 1, self.box(0), self.box(0)
        target = Fraction(1, self.grid * 16)
        for n in range(4 * self.bits):
            term = self.div(power, factorial)
            if n % 2:
                sine = self.add(sine, term if n % 4 == 1 else self.neg(term))
            else:
                cosine = self.add(cosine, term if n % 4 == 0 else self.neg(term))
            power = self.mul(power, reduced)
            factorial *= n + 1
            tail = magnitude ** (n + 1) / factorial
            if n > 3 and tail <= target:
                pad = self._remainder_padding(tail)
                sine, cosine = self.add(sine, pad), self.add(cosine, pad)
                return ((sine, cosine), (cosine, self.neg(sine)),
                        (self.neg(sine), self.neg(cosine)), (self.neg(cosine), sine))[low % 4]
        raise _BudgetExhausted()

    def sin(self, x) -> Box:
        return self.sincos(x)[0]

    def cos(self, x) -> Box:
        return self.sincos(x)[1]


# ----------------------------------------------------------------------------------- T3 decision policy


# Filterable predicate kinds and their width bounds. A kind names a fixed
# expression shape whose canonical level-L_MAX enclosure width is bounded a
# priori: one canonical function applied to exact stored inputs (sqrt of an
# exact value; sin/cos of an exact angle; atan/atan2 of exact arguments away
# from the branch cut; the norm of an exact vector). Each of these performs a
# bounded number of outward roundings (at most a few hundred series terms and
# reductions) on quantities of magnitude ``magnitude``, so its width is below
# 2**_W_SLACK_BITS grid units times max(1, magnitude). Composite or
# ill-conditioned expressions (e.g. a quotient of near-zero differences) have
# no such bound and are NOT filterable: certified_compare alone decides them.
# certified_compare re-checks the bound whenever it refuses a registered kind.
_W_SLACK_BITS = 40
_W_FACTORS = {
    "sqrt": 1,
    "distance": 1,
    "trigonometric": 4,
    "angle": 4,
}


def w_max(kind: str, magnitude: float) -> Fraction:
    """Documented bound of the canonical level-``L_MAX`` enclosure width for ``kind``."""
    factor = _W_FACTORS.get(kind)
    if factor is None:
        raise ValueError(f"unknown predicate kind {kind!r}")
    scale = max(Fraction(1), exact(float(magnitude)))
    return factor * scale / (1 << (grid_bits(L_MAX) - _W_SLACK_BITS))


def _compare_rational(value: Fraction, threshold: Fraction) -> Outcome:
    return Outcome.BELOW if value < threshold else Outcome.ABOVE if value > threshold else Outcome.EQUAL


def certified_compare(predicate: str, quantity, threshold, *, kind: str | None = None, magnitude: float | None = None,
                      on_equal: Outcome | None = None, entities=()) -> Outcome:
    """Compare ``quantity`` with ``threshold`` by the one decision policy (T3).

    ``quantity`` is either an exact value (a stored double, int or Fraction,
    compared exactly) or a callable ``f(engine) -> Box`` evaluated by the
    canonical procedure at levels ``0 .. L_MAX``. Exact equality returns
    ``on_equal`` (``Outcome.EQUAL`` when ``None``). A canonical enclosure that
    still contains the threshold at ``L_MAX`` raises
    :class:`DegenerateWithinBound`; it is never snapped to a side.
    """
    limit = exact(threshold)
    if not callable(quantity):
        outcome = _compare_rational(exact(quantity), limit)
        if outcome is Outcome.EQUAL and on_equal is not None:
            outcome = on_equal
        _record(predicate, entities, "exact", None, outcome.value)
        return outcome
    enclosure = None
    for level in range(L_MAX + 1):
        try:
            enclosure = quantity(Canonical(level))
        except _NotEvaluable:
            enclosure = None
            break
        except _BudgetExhausted:
            enclosure = None                    # only the level-L_MAX enclosure is evidence
            continue
        if enclosure.hi < limit:
            _record(predicate, entities, "canonical", level, Outcome.BELOW.value)
            return Outcome.BELOW
        if enclosure.lo > limit:
            _record(predicate, entities, "canonical", level, Outcome.ABOVE.value)
            return Outcome.ABOVE
        if enclosure.lo == enclosure.hi == limit:
            outcome = Outcome.EQUAL if on_equal is None else on_equal
            _record(predicate, entities, "canonical", level, outcome.value)
            return outcome
    if kind is not None and magnitude is None:
        raise ValueError("a registered kind needs the magnitude of its inputs")
    if enclosure is not None and kind is not None and enclosure.width > w_max(kind, magnitude):
        raise CertificationPolicyError(f"{predicate}: canonical enclosure width {_describe(enclosure.width)} exceeds W_max "
                           f"of kind {kind!r}; the quantity is not of that registered shape")
    _record(predicate, entities, "canonical", L_MAX, "refused")
    raise DegenerateWithinBound(predicate, enclosure, limit, entities)


def filter_compare(predicate: str, estimate: float, error_bound: float, threshold, *, kind: str,
                   magnitude: float, entities=()) -> Outcome | None:
    """A fast route's decision, or ``None`` to defer to :func:`certified_compare`.

    The fast route claims ``|exact quantity - estimate| <= error_bound``. It
    decides only when it is separated from the threshold by more than
    ``error_bound + w_max``: the canonical level-``L_MAX`` enclosure (which
    contains the exact quantity and is at most ``w_max`` wide) is then
    separated too, so both routes return the same outcome. Closer cases defer.

    ``magnitude`` is required: the largest absolute input of the function
    (the angle for sin/cos, the arguments for atan/atan2, the coordinates for
    a distance, the radicand's scale for sqrt). ``W_max`` scales with it, so a
    huge angle's argument reduction is covered and such cases defer.

    ``kind`` must be a registered kind (see ``_W_FACTORS``): the quantity is
    one canonical function of exact inputs, evaluable at level ``L_MAX``. A
    composite or ill-conditioned expression has no a-priori width bound and
    must not be filtered; calling with an unregistered kind raises
    ``ValueError``.
    """
    gap = exact(estimate) - exact(threshold)
    margin = exact(error_bound) + w_max(kind, magnitude)
    if gap > margin:
        _record(predicate, entities, "filter", None, Outcome.ABOVE.value)
        return Outcome.ABOVE
    if -gap > margin:
        _record(predicate, entities, "filter", None, Outcome.BELOW.value)
        return Outcome.BELOW
    return None


# ----------------------------------------------------------------------------------- T0 stations


@dataclass(frozen=True, slots=True)
class RootInterval:
    """The unique root of an exact polynomial inside ``(lo, hi)``.

    ``coefficients`` are exact power-basis coefficients (constant first). The
    constructor requires a strict sign change over the interval and no root
    at its ends; uniqueness is the caller's isolation claim (from
    ``analytic_roots.isolate_real_roots``) and is part of the definition.
    """
    coefficients: tuple
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        coefficients = tuple(exact(value) for value in self.coefficients)
        lo, hi = exact(self.lo), exact(self.hi)
        if not lo < hi:
            raise GeometryError("root interval must have lo < hi")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        if self.value(lo) * self.value(hi) >= 0:
            raise GeometryError("root interval needs a strict sign change and no root at its ends")

    def value(self, x: Fraction) -> Fraction:
        result = Fraction(0)
        for coefficient in reversed(self.coefficients):
            result = result * x + coefficient
        return result

    def refined(self, steps: int) -> "RootInterval":
        """Exact bisection ``steps`` times (an exact midpoint root ends the refinement as a point)."""
        lo, hi = self.lo, self.hi
        low_sign = self.value(lo) > 0
        for _ in range(steps):
            middle = (lo + hi) / 2
            at = self.value(middle)
            if at == 0:
                return _PointRoot(self.coefficients, middle)
            if (at > 0) == low_sign:
                lo = middle
            else:
                hi = middle
        return RootInterval(self.coefficients, lo, hi)

    def box(self) -> Box:
        return Box(self.lo, self.hi)


@dataclass(frozen=True, slots=True)
class _PointRoot:
    coefficients: tuple
    root: Fraction

    def refined(self, steps: int) -> "_PointRoot":
        return self

    def box(self) -> Box:
        return Box(self.root, self.root)


@dataclass(frozen=True, slots=True)
class Station:
    """A decision point: a frozen curve definition and an exact parameter (T0).

    ``parameter`` is a stored double / Fraction or a :class:`RootInterval`.
    Positions are derived: exact rationals for line and Bezier paths at
    rational parameters, otherwise a canonical enclosure.
    """
    definition: object
    parameter: object

    def __post_init__(self):
        parameter = self.parameter
        if not isinstance(parameter, (RootInterval, _PointRoot)):
            parameter = exact(parameter)
        object.__setattr__(self, "parameter", parameter)

    def exact_position(self) -> tuple[Fraction, ...] | None:
        """The exact position for rational parameters of line and Bezier paths, else ``None``."""
        if not isinstance(self.parameter, Fraction):
            return None
        kind = type(self.definition).__name__
        t = self.parameter
        if kind == "LinePath":
            start, end = exact_vector(self.definition.start), exact_vector(self.definition.end)
            return tuple(a + t * (b - a) for a, b in zip(start, end))
        if kind == "BezierPath":
            points = [exact_vector(point) for point in self.definition.controls]
            while len(points) > 1:
                points = [tuple((1 - t) * a + t * b for a, b in zip(p, q)) for p, q in zip(points, points[1:])]
            return points[0]
        return None


# ----------------------------------------------------------------------------------- T4 exact chart predicates


def orientation_2d(a: Sequence, b: Sequence, c: Sequence) -> int:
    """Sign of the exact cross product ``(b - a) x (c - a)``: +1 left turn, -1 right turn, 0 collinear."""
    (ax, ay), (bx, by), (cx, cy) = exact_vector(a), exact_vector(b), exact_vector(c)
    value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (value > 0) - (value < 0)


def point_in_polygon(point: Sequence, polygon: Sequence[Sequence]) -> Outcome:
    """Exact containment of a point in a straight-edged polygon (any orientation).

    Returns ``ABOVE`` inside, ``BELOW`` outside and ``EQUAL`` on the boundary,
    decided by exact rational ray parity (no tolerance, no sampling).
    """
    px, py = exact_vector(point)
    vertices = [exact_vector(vertex) for vertex in polygon]
    inside = False
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        if cross == 0 and min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1):
            return Outcome.EQUAL
        if (y0 > py) != (y1 > py):
            # crossing of the horizontal ray to +x: x0 + (py - y0)(x1 - x0)/(y1 - y0) > px, exactly
            if (cross > 0) == (y1 > y0):
                inside = not inside
    return Outcome.ABOVE if inside else Outcome.BELOW


# ----------------------------------------------------------------------------------- T7 operand order


def canonical_order(first, second) -> tuple[object, object, bool]:
    """Operands ordered by their content checksum, and whether they were swapped (T7)."""
    from .definition_binding import definition_checksum

    a, b = definition_checksum(first), definition_checksum(second)
    return (first, second, False) if a <= b else (second, first, True)
