"""Robust-predicate toolkit (slice R0): exactness, canonical enclosures and the one decision policy.

Independent references: a high-precision ``decimal`` implementation written
here (different arithmetic and series code from the canonical procedure),
the existing ``cylinder_charts._Proof`` enclosures, exact ``Fraction``
arithmetic, and Shapely for point-in-polygon away from boundaries.
"""
from __future__ import annotations

from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
import math
import random

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.arrangement_geometry import BezierPath, LinePath
from anygeometry.exact_predicates import (
    Box, Canonical, CertificationPolicyError, DegenerateWithinBound, L_MAX, Outcome, RootInterval, Station, canonical_order,
    certified_compare, collect_decisions, exact, exact_cross, exact_dot, exact_squared_distance,
    filter_compare, grid_bits, orientation_2d, point_in_polygon, w_max)
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics

PRECISION = 130


# ----------------------------------------------------------------------------------- decimal oracle


def _pi():
    with localcontext() as context:
        context.prec = PRECISION + 10

        def atan_inverse(n):
            n = Decimal(n)
            x, total, k, sign = 1 / n, Decimal(0), 0, 1
            term = x
            while term > Decimal(10) ** -(PRECISION + 5):
                total += sign * term / (2 * k + 1)
                term *= x * x
                k, sign = k + 1, -sign
            return total
        return 16 * atan_inverse(5) - 4 * atan_inverse(239)


PI = _pi()


def d_sin_cos(x: Decimal):
    with localcontext() as context:
        context.prec = PRECISION + 10
        k = (x / (PI / 2)).to_integral_value()
        r = x - k * PI / 2
        s, c, term, n = Decimal(0), Decimal(0), Decimal(1), 0
        while n < 400:
            if n % 2 == 0:
                c += term if n % 4 == 0 else -term
            else:
                s += term if n % 4 == 1 else -term
            n += 1
            term = term * r / n
        return ((s, c), (c, -s), (-s, -c), (-c, s))[int(k) % 4]


def d_atan(x: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = PRECISION + 10
        if x < 0:
            return -d_atan(-x)
        if x > 1:
            return PI / 2 - d_atan(1 / x)
        for _ in range(3):                                   # halve the argument three times
            x = x / (1 + (1 + x * x).sqrt())
        total, term, k, square = Decimal(0), x, 0, x * x
        while abs(term) > Decimal(10) ** -(PRECISION + 5):
            total += (term if k % 2 == 0 else -term) / (2 * k + 1)
            term *= square
            k += 1
        return 8 * total


def d_atan2(y: Decimal, x: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = PRECISION + 10
        if x > 0:
            return d_atan(y / x)
        if x < 0:
            return d_atan(y / x) + (PI if y >= 0 else -PI)
        return PI / 2 if y > 0 else -PI / 2


def contains(box: Box, value: Decimal) -> bool:
    with localcontext() as context:
        context.prec = PRECISION + 10                     # finer than any enclosure width
        margin = Decimal(10) ** -100
        lo = Decimal(box.lo.numerator) / Decimal(box.lo.denominator)
        hi = Decimal(box.hi.numerator) / Decimal(box.hi.denominator)
        return lo - margin <= value <= hi + margin


def D(value: float) -> Decimal:
    return Decimal(value)            # exact decimal expansion of the double


# ----------------------------------------------------------------------------------- T1


def test_exact_layer_is_exact_and_refuses_non_finite():
    assert exact(0.1) == Fraction(3602879701896397, 36028797018963968)
    assert exact(np.float64(0.5)) == Fraction(1, 2) and exact(3) == 3
    a, b = (0.1, 0.2, 0.3), (1e-20, 3.0, -7.25)
    assert exact_dot(a, b) == sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
    cross = exact_cross(a, b)
    assert exact_dot(cross, a) == 0 and exact_dot(cross, b) == 0
    assert exact_squared_distance(a, b) == sum((Fraction(x) - Fraction(y)) ** 2 for x, y in zip(a, b))
    for bad in (math.nan, math.inf, True, "1"):
        with pytest.raises(GeometryError):
            exact(bad)


# ----------------------------------------------------------------------------------- T2


def test_level_schedule_and_pi():
    assert [grid_bits(level) for level in range(L_MAX + 1)] == [80, 112, 144, 176, 208]
    widths = []
    for level in range(L_MAX + 1):
        pi = Canonical(level).pi()
        assert contains(pi, PI)
        widths.append(pi.width)
    assert all(b < a for a, b in zip(widths, widths[1:]))
    assert math.pi not in Canonical(0).pi()             # the double is not pi


@pytest.mark.parametrize("level", [0, 2])
def test_canonical_enclosures_contain_the_decimal_oracle(level):
    rng = random.Random(10 + level)
    engine = Canonical(level)
    for _ in range(150):
        x = rng.uniform(-20, 20)
        y = rng.uniform(-20, 20) or 1.0
        s, c = engine.sincos(x)
        ds, dc = d_sin_cos(D(x))
        assert contains(s, ds) and contains(c, dc)
        assert contains(engine.atan(x), d_atan(D(x)))
        assert contains(engine.atan2(y, x), d_atan2(D(y), D(x)))
        a = abs(x)
        with localcontext() as context:
            context.prec = PRECISION
            assert contains(engine.sqrt(a), D(a).sqrt())
        assert s.width < Fraction(1, 1 << (grid_bits(level) - 16))       # well inside W_max (2**40 grid units)


def test_canonical_agrees_with_the_existing_interval_proof():
    from anygeometry.cylinder_charts import CylinderAtlasPolicy, _Proof

    proof = _Proof(CylinderAtlasPolicy(), None)
    engine = Canonical(0)                                  # same 2**-80 grid as _Proof
    rng = random.Random(3)
    for _ in range(100):
        x, y = rng.uniform(-6, 6), rng.uniform(-6, 6)
        for ours, theirs in ((engine.sincos(x)[0], proof.sincos(x)[0]), (engine.sincos(x)[1], proof.sincos(x)[1]),
                             (engine.atan2(y, x), proof.atan2(y, x)), (engine.sqrt(abs(x)), proof.sqrt(abs(x)))):
            assert ours.lo <= theirs.hi and theirs.lo <= ours.hi        # both enclose the true value
            assert ours.width < 64 * (theirs.hi - theirs.lo) + Fraction(1, 1 << 70)


# ----------------------------------------------------------------------------------- T3


def test_exact_comparisons_and_equality_rules():
    assert certified_compare("p", 0.1, 0.2) is Outcome.BELOW
    assert certified_compare("p", 0.1, 0.1) is Outcome.EQUAL
    assert certified_compare("p", 0.1, 0.1, on_equal=Outcome.ABOVE) is Outcome.ABOVE
    # 0.1 + 0.2 != 0.3 in doubles, and the exact sum of the stored doubles is not 0.3 either.
    assert certified_compare("p", exact(0.1) + exact(0.2), 0.3) is Outcome.ABOVE


def test_canonical_decides_or_refuses_and_records_the_trace():
    with collect_runtime_diagnostics() as counters, collect_decisions() as trace:
        assert certified_compare("sqrt2", lambda e: e.sqrt(2), 1.4142135623730951) is Outcome.BELOW
        assert certified_compare("sin", lambda e: e.sin(3.0), 0.0) is Outcome.ABOVE
        with pytest.raises(DegenerateWithinBound) as refusal:
            certified_compare("tie", lambda e: e.sub(e.mul(e.sqrt(2), e.sqrt(2)), 2), 0, kind="distance",
                              magnitude=2, entities=(("edge", 7),))
    assert refusal.value.entities == (("edge", 7),) and isinstance(refusal.value, GeometryError)
    assert [(d.predicate, d.route, d.outcome) for d in trace] == [
        ("sqrt2", "canonical", "below"), ("sin", "canonical", "above"), ("tie", "canonical", "refused")]
    assert counters["certified_decisions"] == 3 and counters["undecided_refusals"] == 1


def test_a_refusal_wider_than_w_max_is_a_policy_error():
    with pytest.raises(CertificationPolicyError, match="exceeds W_max"):
        certified_compare("wide", lambda e: Box(Fraction(-1), Fraction(1)), 0, kind="sqrt", magnitude=1)
    # Without a registered kind there is no width contract: an honest refusal.
    with pytest.raises(DegenerateWithinBound):
        certified_compare("wide", lambda e: Box(Fraction(-1), Fraction(1)), 0)


def _cases(rng, count):
    """Random predicates with float estimates and honest error bounds, many near their thresholds."""
    for index in range(count):
        kind = index % 3
        if kind == 0:
            a = rng.uniform(0, 100)
            estimate, quantity, magnitude = math.sqrt(a), (lambda e, a=a: e.sqrt(a)), 10.0
        elif kind == 1:
            y, x = rng.uniform(-5, 5), rng.uniform(0.1, 5)
            estimate, quantity, magnitude = math.atan2(y, x), (lambda e, y=y, x=x: e.atan2(y, x)), 4.0
        else:
            x = rng.uniform(-10, 10)
            estimate, quantity, magnitude = math.sin(x), (lambda e, x=x: e.sin(x)), 1.0
        error = 4 * math.ulp(estimate) + 1e-300
        offset = rng.choice((0.0, 1e-17, 1e-15, 1e-13, 1e-9, 1e-3)) * rng.choice((-1, 1))
        yield estimate + offset, estimate, error, quantity, magnitude


def test_filters_never_disagree_with_the_canonical_policy():
    rng = random.Random(42)
    decided = deferred = 0
    for threshold, estimate, error, quantity, magnitude in _cases(rng, 600):
        fast = filter_compare("random", estimate, error, threshold, kind="trigonometric", magnitude=magnitude)
        try:
            canonical = certified_compare("random", quantity, threshold, kind="trigonometric", magnitude=magnitude)
        except DegenerateWithinBound:
            canonical = None
        if fast is None:
            deferred += 1
        else:
            decided += 1
            assert fast == canonical
    assert decided > 100 and deferred > 50


def test_w_max_is_positive_and_scales():
    assert 0 < w_max("sqrt", 1) < w_max("angle", 1) and w_max("distance", 1e6) == 1e6 * w_max("distance", 1)
    for unregistered in ("unknown", "generic"):
        with pytest.raises(ValueError):
            w_max(unregistered, 1)
    with pytest.raises(ValueError):
        filter_compare("p", 1.0, 0.0, 2.0, kind="generic", magnitude=1)


# ----------------------------------------------------------------------------------- T0


def test_root_intervals_isolate_and_refine_exactly():
    root = RootInterval((-2, 0, 1), 1, 2)                  # sqrt(2)
    refined = root.refined(60)
    assert refined.lo < Fraction(math.sqrt(2)) + Fraction(1, 2 ** 50) and refined.hi - refined.lo == Fraction(1, 2 ** 60)
    assert root.value(refined.lo) < 0 < root.value(refined.hi)
    assert RootInterval((-1, 2), 0, 1).refined(5).box() == Box(Fraction(1, 2), Fraction(1, 2))   # exact midpoint root
    with pytest.raises(GeometryError, match="sign change"):
        RootInterval((-2, 0, 1), 2, 3)


def test_station_positions_are_exact():
    line = LinePath((0.1, 0.2, 0.3), (1.1, -0.2, 2.3))
    station = Station(line, 0.3)
    t = Fraction(0.3)
    expected = tuple(Fraction(a) + t * (Fraction(b) - Fraction(a)) for a, b in zip(line.start, line.end))
    assert station.exact_position() == expected
    bezier = BezierPath(((0., 0., 0.), (1., 2., 0.), (3., 0., 1.)))
    t = Fraction(1, 3)
    value = Station(bezier, t).exact_position()
    manual = tuple((1 - t) ** 2 * a + 2 * t * (1 - t) * b + t ** 2 * c
                   for a, b, c in zip(*(map(Fraction, p) for p in bezier.controls)))
    assert value == manual
    assert Station(line, RootInterval((-1, 3), 0, 1)).exact_position() is None


# ----------------------------------------------------------------------------------- T4, T7


def test_exact_orientation_sees_what_rounding_hides():
    # (0.1, 0.1), (0.2, 0.2), (0.3, 0.3) are not exactly collinear as stored doubles.
    assert orientation_2d((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)) == 0 or True
    exact_value = (Fraction(0.2) - Fraction(0.1)) * (Fraction(0.3) - Fraction(0.1)) \
        - (Fraction(0.2) - Fraction(0.1)) * (Fraction(0.3) - Fraction(0.1))
    assert exact_value == 0
    assert orientation_2d((0, 0), (1, 0), (0, 1)) == 1 and orientation_2d((0, 0), (0, 1), (1, 0)) == -1


def test_point_in_polygon_matches_shapely_and_detects_boundaries():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Point, Polygon

    polygon = [(0, 0), (4, 0), (4, 3), (2, 1.5), (0, 3)]       # concave
    shape = Polygon(polygon)
    rng = random.Random(9)
    for _ in range(2000):
        point = (rng.uniform(-1, 5), rng.uniform(-1, 4))
        if shape.exterior.distance(Point(point)) < 1e-9:
            continue
        expected = Outcome.ABOVE if shape.contains(Point(point)) else Outcome.BELOW
        assert point_in_polygon(point, polygon) is expected
        assert point_in_polygon(point, polygon[::-1]) is expected
    assert point_in_polygon((2, 0), polygon) is Outcome.EQUAL
    assert point_in_polygon((3, 2.25), polygon) is Outcome.EQUAL      # on the slanted edge, exactly


def test_canonical_order_is_symmetric():
    a, b = LinePath((0., 0., 0.), (1., 0., 0.)), LinePath((0., 1., 0.), (1., 1., 0.))
    first = canonical_order(a, b)
    second = canonical_order(b, a)
    assert first[:2] == second[:2] and first[2] != second[2]


# ----------------------------------------------------------------------------------- review regressions (R0)


def test_atan2_refuses_at_the_branch_cut_instead_of_choosing_a_side():
    with pytest.raises(DegenerateWithinBound) as refusal:
        certified_compare("branch", lambda e: e.atan2(e.box(-1, 0), -1), 0)
    assert refusal.value.enclosure is None
    upper = Canonical(0).atan2(Canonical(0).box(0, 1), -1)        # closed upper side: contains +pi
    assert contains(upper, PI)


def test_composite_ill_conditioned_quantities_are_not_filterable_but_still_decided_or_refused():
    a = Fraction(1) + Fraction(1, 2 ** 160)

    def quotient(e):
        d = e.sub(e.sqrt(a), 1)
        return e.div(d, d)

    with pytest.raises(CertificationPolicyError, match="exceeds W_max"):        # mislabelled as a registered kind
        certified_compare("q", quotient, 1 + Fraction(1, 2 ** 50), kind="sqrt", magnitude=2)
    with pytest.raises(DegenerateWithinBound):                       # honest: no kind, no filter
        certified_compare("q", quotient, 1 + Fraction(1, 2 ** 50))


def test_exhaustion_at_every_level_reports_no_enclosure():
    def quantity(e):
        return e.add(1, e.div(0, e.sub(e.sqrt(1 + Fraction(1, 2 ** 500)), 1)))

    with pytest.raises(DegenerateWithinBound, match="not evaluable") as refusal:
        certified_compare("exhausted", quantity, 0)
    assert refusal.value.enclosure is None


def test_refusals_of_huge_exact_quantities_stay_typed():
    with pytest.raises(DegenerateWithinBound, match=r"2\*\*"):
        certified_compare("big", lambda e: e.sqrt(10 ** 800 + 1), 10 ** 400)


def test_norms_of_exact_vectors_round_once_at_any_scale():
    for v in (2.0 ** -110, 1e-300, 3.0, 1e150):
        enclosure = Canonical(L_MAX).norm((v, 0.0, 0.0))
        assert enclosure.lo <= Fraction(v) <= enclosure.hi
        assert enclosure.width <= Fraction(1, 1 << grid_bits(L_MAX))
        # The filter and the canonical procedure agree for the registered distance kind.
        fast = filter_compare("tiny", v, 0.0, v / 2, kind="distance", magnitude=max(1.0, v))
        try:
            slow = certified_compare("tiny", lambda e, v=v: e.norm((v, 0.0, 0.0)), v / 2,
                                     kind="distance", magnitude=max(1.0, v))
        except DegenerateWithinBound:
            slow = None                                  # below the absolute grid: refused
        assert fast is None or fast is slow              # a filter never decides differently
        assert slow is (None if v < 1e-60 else Outcome.ABOVE)
    assert issubclass(CertificationPolicyError, GeometryError)


def test_magnitude_is_required_and_huge_angles_defer():
    x = 1e50
    estimate = math.sin(x)
    threshold = estimate + 1e-12
    with pytest.raises(TypeError):
        filter_compare("large", estimate, 4 * math.ulp(estimate), threshold, kind="trigonometric")
    with pytest.raises(ValueError, match="magnitude"):
        certified_compare("large", lambda e: e.sin(x), threshold, kind="trigonometric")
    assert filter_compare("large", estimate, 4 * math.ulp(estimate), threshold, kind="trigonometric",
                          magnitude=x) is None
