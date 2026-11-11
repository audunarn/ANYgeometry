"""The integer (gcd-free) exact kernels against frozen copies of the Fraction kernels they replace.

``_isolate_real_roots`` bisects on integer numerators over a common denominator and ``build_abc`` /
``PolyPlan`` work on integer polynomials over one positive denominator. Every result must equal the Fraction
route exactly: isolating intervals (values and types), multiplicities, float witnesses, refusals (exception type
and message) and the number of cancellation polls.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from anygeometry import analytic_roots
from anygeometry.analytic_roots import (GeometryError, IsolatedRoot, _division, _integer_value, _square_free, _sturm,
                                        _trim, _variations)
from anygeometry.branch_algebra import (BezierRuledSupport, PolyPlan, _integer_row, _padd, _resultant,
                                        build_abc, poly_roots)
from anygeometry.branch_algebra import _trim as _btrim
from anygeometry.quadric_algebra import QuadricSupport, _chart_roots, _fraction_vec


# ---------------------------------------------------------------------------
# Frozen Fraction references (verbatim logic of commit 739305b)


def _reference_isolate(coefficients, tolerance, cancellation_check, interval):
    try:
        p = _trim([Fraction(value) for value in coefficients])
        tolerance = Fraction(tolerance)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise GeometryError("root isolation needs finite coefficients and tolerance") from exc
    if not p or p == (0,) or tolerance <= 0:
        raise GeometryError("root isolation needs a nonzero polynomial and positive tolerance")
    if cancellation_check is not None and cancellation_check():
        raise GeometryError("analytic root isolation cancelled")
    if interval is not None:
        try:
            lower, upper = (Fraction(value) for value in interval)
            if lower >= upper:
                raise ValueError('empty interval')
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            raise GeometryError('root isolation interval must have two finite increasing endpoints') from exc
    if len(p) == 1:
        return ()
    p = _square_free(p)
    sequence = _sturm(p)
    if interval is None:
        bound = Fraction(2)+max(abs(value/p[-1]) for value in p[:-1])
        lower, upper = -bound, bound
    else:
        for endpoint in (lower, upper):
            if _integer_value(sequence[0], endpoint) == 0:
                quotient, remainder = _division(p, (-endpoint, Fraction(1)))
                if remainder != (0,):
                    raise GeometryError('inconsistent interval endpoint root')
                remaining = _reference_isolate(quotient, tolerance, cancellation_check, (lower, upper))
                return tuple(sorted((IsolatedRoot(endpoint, endpoint), *remaining), key=lambda item: item.lower))
    variation_cache = {}

    def variations(point):
        value = variation_cache.get(point)
        if value is None:
            value = variation_cache[point] = _variations(sequence, point)
        return value

    pending = [(lower, upper, None, 0)]
    roots = []
    while pending:
        if cancellation_check is not None and cancellation_check():
            raise GeometryError("analytic root isolation cancelled")
        lower, upper, count, lower_sign = pending.pop()
        if count is None:
            count = variations(lower)-variations(upper)
        if count == 0:
            continue
        if count == 1 and upper-lower <= tolerance:
            roots.append(IsolatedRoot(lower, upper))
            continue
        middle = (lower+upper)/2
        middle_value = _integer_value(sequence[0], middle)
        if middle_value == 0:
            roots.append(IsolatedRoot(middle, middle))
            quotient, remainder = _division(p, (-middle, Fraction(1)))
            if remainder != (0,):
                raise GeometryError("inconsistent exact root isolation")
            remaining = _reference_isolate(quotient, tolerance, cancellation_check, interval)
            return tuple(sorted((IsolatedRoot(middle, middle), *remaining), key=lambda item: item.lower))
        middle_sign = 1 if middle_value > 0 else -1
        if count == 1:
            if not lower_sign:
                lower_value = _integer_value(sequence[0], lower)
                lower_sign = 1 if lower_value > 0 else -1
            left = int(lower_sign != middle_sign)
        else:
            left = variations(lower)-variations(middle)
        pending.extend(((middle, upper, count-left, middle_sign), (lower, middle, left, lower_sign)))
    return tuple(sorted(roots, key=lambda item: item.lower))


def _reference_build_abc(first, second):
    second = QuadricSupport.from_surface(second)
    m, l, c0 = second.exact_relative(first.origin)
    d = _fraction_vec(first.direction)
    a = first.exact_power()
    n = first.degree
    md = [sum(m[i][j] * d[j] for j in range(3)) for i in range(3)]
    lead = sum(d[i] * md[i] for i in range(3))
    b = [2 * sum(l[i] * d[i] for i in range(3))] + [2 * sum(md[i] * a[k][i] for i in range(3)) for k in range(1, n + 1)]
    ma = [[sum(m[i][j] * a[k][j] for j in range(3)) for i in range(3)] for k in range(n + 1)]
    c = [c0] + [Fraction(0)] * (2 * n)
    for i in range(1, n + 1):
        for k in range(1, n + 1):
            c[i + k] += sum(a[i][r] * ma[k][r] for r in range(3))
    for j in range(1, n + 1):
        c[j] += 2 * sum(l[r] * a[j][r] for r in range(3))
    return [lead], _btrim(b), _btrim(c)


def _reference_bound(A, B, C, s0):
    s = Fraction(float(s0))
    roots = poly_roots(_padd(_padd([A[0] * s * s], [value * s for value in B]), C))
    return [] if roots is None else roots


def _reference_common(A, B, C):
    from anygeometry.analytic_roots import _integer_gcd
    current = None
    for poly in (A, B, C):
        if all(value == 0 for value in poly):
            continue
        row = _integer_row(_btrim(poly))
        current = row if current is None else _integer_gcd(current, row)
    if current is None or len(current) < 2:
        return []
    return [(t, 1) for t, _m in _chart_roots(list(current), interval=(Fraction(0), Fraction(1)))]


# ---------------------------------------------------------------------------
# Comparison helpers


def _outcome(function, *args):
    try:
        return ("ok", function(*args))
    except Exception as error:                                   # the refusal itself is part of the contract
        return ("error", type(error), str(error))


def _typed(result):
    if result[0] != "ok":
        return result
    value = result[1]
    if isinstance(value, tuple) and all(isinstance(item, IsolatedRoot) for item in value):
        return ("ok", tuple((r.lower, type(r.lower), r.upper, type(r.upper), _witness(r)) for r in value))
    return ("ok", _deep(value))


def _witness(root):
    try:
        return root.witness.hex()
    except OverflowError:
        return OverflowError


def _deep(value):
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_deep(item) for item in value))
    if isinstance(value, float):
        return (float, value.hex())
    return (type(value), value)


class _Polls:
    def __init__(self, cancel_at=None):
        self.count, self.cancel_at = 0, cancel_at

    def __call__(self):
        self.count += 1
        return self.cancel_at is not None and self.count >= self.cancel_at


def assert_isolation_identical(coefficients, tolerance=4 * 2.0 ** -52, interval=None, cancel_at=None):
    old_polls, new_polls = _Polls(cancel_at), _Polls(cancel_at)
    old = _typed(_outcome(_reference_isolate, coefficients, tolerance, old_polls, interval))
    new = _typed(_outcome(analytic_roots._isolate_real_roots, coefficients, tolerance, new_polls, interval))
    assert new == old, (coefficients, tolerance, interval)
    assert new_polls.count == old_polls.count
    return old


# ---------------------------------------------------------------------------
# Root isolation


def _from_roots(roots, lead=1):
    poly = [Fraction(lead)]
    for root in roots:
        poly = [(poly[i - 1] if i else 0) - root * (poly[i] if i < len(poly) else 0) for i in range(len(poly) + 1)]
    return poly


ADVERSARIAL = [
    [0], [], [Fraction(0), 0], [3], [0, 0, 0, 5], [1, 0, 0, 0, 0],               # zero, constants, zero leading
    [-1, 1], [0, 1], [1, 1], [-1, 0, 1], [0, 0, 1], [1, 0, 1],                    # roots at 0, chart ends, none
    _from_roots([0, 0, 1, 1, Fraction(1, 2)]), _from_roots([Fraction(1, 3)] * 3 + [Fraction(2, 3)] * 2),
    _from_roots([-1, 1, 0, Fraction(1, 2), Fraction(-1, 2)]), _from_roots([Fraction(1, 1024), Fraction(3, 1024)]),
    _from_roots([2.0 ** -40, 2.0 ** -40 + 2.0 ** -90]),                          # a pair closer than the tolerance
    [2.0 ** -1000, -(2.0 ** 1000)], [1e300, -1e-300, 1e-300], [5e-324, 1.0, -5e-324],
    [1.0, float("nan")], [float("inf"), 1.0], ["x", 1],
    [-2, 0, 0, 0, 0, 0, 0, 1],                                                    # 2^(1/7), Cauchy bound
    _from_roots([Fraction(1, 7), Fraction(1, 7) + Fraction(1, 10 ** 20)]),
]
INTERVALS = [None, (0, 1), (-1, 1), (Fraction(0), Fraction(1)), (0.0, 0.5), (Fraction(1, 3), Fraction(2, 3)),
             (1, 0), (0, 0), (0, float("inf")), ("a", 1), (Fraction(-1, 2), Fraction(1, 2))]


@pytest.mark.parametrize("coefficients", ADVERSARIAL, ids=range(len(ADVERSARIAL)))
def test_adversarial_isolation_identical(coefficients):
    for interval in INTERVALS:
        for tolerance in (4 * 2.0 ** -52, 1e-13, 0.25, 0.0, -1.0, 2.0 ** -200):
            assert_isolation_identical(coefficients, tolerance, interval)


def test_cancellation_polls_identical():
    poly = _from_roots([Fraction(k, 9) for k in range(-4, 5)])
    total = _Polls()
    analytic_roots._isolate_real_roots(poly, 4 * 2.0 ** -52, total, None)
    assert total.count > 10
    for cancel_at in range(1, total.count + 2):
        assert_isolation_identical(poly, interval=None, cancel_at=cancel_at)
        assert_isolation_identical(poly, interval=(0, 1), cancel_at=cancel_at)


def test_random_isolation_identical():
    rng = random.Random(20261110)
    for case in range(1500):
        kind = case % 5
        degree = rng.randint(1, 9)
        if kind == 0:
            poly = [rng.randint(-50, 50) for _ in range(degree + 1)]
        elif kind == 1:
            poly = [rng.uniform(-1, 1) * 2.0 ** rng.randint(-60, 60) for _ in range(degree + 1)]
        elif kind == 2:                                          # repeated dyadic and rational roots
            roots = [Fraction(rng.randint(-8, 8), rng.choice((1, 2, 4, 8, 3, 6))) for _ in range(rng.randint(1, 6))]
            roots += rng.sample(roots, min(len(roots), rng.randint(0, 3)))
            poly = _from_roots(roots, lead=rng.choice((1, -3, Fraction(5, 7))))
        elif kind == 3:                                          # float coefficients exactly as the planners pass them
            poly = [float(rng.uniform(-3, 3)) for _ in range(degree + 1)]
            if rng.random() < .3:
                poly[-1] = 0.0
        else:                                                    # integer rows of clustered roots
            centre = Fraction(rng.randint(0, 64), 64)
            poly = [int(v * 2 ** 40) for v in _from_roots([centre + Fraction(rng.randint(-3, 3), 2 ** 30)
                                                           for _ in range(rng.randint(2, 5))])]
        interval = rng.choice(INTERVALS[:6])
        assert_isolation_identical(poly, rng.choice((4 * 2.0 ** -52, 1e-12, 1e-6)), interval)


def test_memoized_entry_identical():
    poly = _from_roots([Fraction(1, 2), Fraction(1, 3), Fraction(1, 3)])
    with analytic_roots.root_isolation_memo():
        first = analytic_roots.isolate_real_roots(poly, interval=(0, 1))
        second = analytic_roots.isolate_real_roots(poly, interval=(0, 1))
    assert first == second == _reference_isolate(poly, Fraction(1e-13), None, (0, 1))


# ---------------------------------------------------------------------------
# Bezier ruled support x quadric


def _random_support(rng):
    degree = rng.randint(2, 4)
    pick = rng.random()

    def coordinate():
        if pick < .3:
            return rng.randint(-8, 8) / rng.choice((1, 2, 4, 8))
        if pick < .5:
            return rng.uniform(-1, 1) * 2.0 ** rng.randint(-30, 30)
        return rng.uniform(-3, 3)
    controls = tuple((coordinate(), coordinate(), coordinate()) for _ in range(degree + 1))
    direction = (coordinate(), coordinate(), coordinate())
    if not any(direction):
        direction = (0., 0., 1.)
    if rng.random() < .2:                                         # a planar directrix ruled out of its plane
        controls = tuple((x, y, 0.) for x, y, _z in controls)
        direction = (0., 0., rng.choice((1., -2.5, 3.0e-7)))
    return BezierRuledSupport(controls, direction)


def _random_quadric(rng, support):
    vec = lambda: tuple(rng.uniform(-2, 2) for _ in range(3))
    kind = rng.choice(("plane", "cylinder", "cone", "elliptic", "parabolic", "general", "sphere", "through"))
    try:
        if kind == "plane":
            return QuadricSupport("plane", vec(), vec())
        if kind == "cylinder":
            return QuadricSupport("cylinder", vec(), vec(), radius=rng.uniform(.1, 2))
        if kind == "cone":
            return QuadricSupport("cone", vec(), vec(), radius=rng.uniform(.1, 2), slope=rng.uniform(-1, 1))
        if kind == "elliptic":
            return QuadricSupport("elliptic", vec(), vec(), u_vector=vec(), v_vector=vec())
        if kind == "parabolic":
            return QuadricSupport("parabolic", vec(), vec(), u_vector=vec(), v_vector=vec())
        if kind == "general":
            return QuadricSupport("general", matrix=tuple(rng.uniform(-1, 1) for _ in range(9)), linear=vec(),
                                  constant=rng.uniform(-1, 1))
        if kind == "sphere":
            return QuadricSupport.sphere(vec(), rng.uniform(.1, 3))
        # a plane containing the ruling direction through the first control point (zero A, shared rulings)
        d = support.direction
        normal = (d[1], -d[0], 0.) if d[0] or d[1] else (1., 0., 0.)
        return QuadricSupport("plane", support.origin, normal)
    except GeometryError:
        return QuadricSupport("plane", (0., 0., 0.), (0., 0., 1.))


def _plan_outcomes(plan, others, stations):
    out = [("abc", _deep((plan.A, plan.B, plan.C))), ("disc", _deep(plan.disc)), ("linear", plan.linear)]
    out.append(("common", _typed(_outcome(plan.common_roots))))
    out.extend(("bound", _typed(_outcome(plan.bound_roots, s))) for s in stations)
    out.extend(("res", _typed(_outcome(plan.resultant_roots, other))) for other in others)
    return out


def _reference_outcomes(first, second, others, stations):
    A, B, C = _reference_build_abc(first, second)
    disc = _btrim(_padd(_pmul_ref(B, B), _pmul_ref([4 * A[0]], C), -1))
    out = [("abc", _deep((A, B, C))), ("disc", _deep(disc)), ("linear", all(v == 0 for v in A))]
    out.append(("common", _typed(_outcome(_reference_common, A, B, C))))
    out.extend(("bound", _typed(_outcome(_reference_bound, A, B, C, s))) for s in stations)
    for other in others:
        def resultant(other=other):
            A3, B3, C3 = _reference_build_abc(first, other)
            return poly_roots(_resultant(A, B, C, A3, B3, C3))
        out.append(("res", _typed(_outcome(resultant))))
    return out


def _pmul_ref(p, q):
    from anygeometry.quadric_algebra import _pmul
    return _pmul(p, q)


def test_random_plans_identical():
    rng = random.Random(1110)
    for _case in range(30):
        first = _random_support(rng)
        second = _random_quadric(rng, first)
        others = [_random_quadric(rng, first) for _ in range(3)] + [second]
        stations = [0.0, rng.uniform(-2, 2), 2.0 ** rng.randint(-40, 40), -0.0]
        expected = _reference_outcomes(first, second, others, stations)
        plan = PolyPlan(first, second)
        assert _plan_outcomes(plan, others, stations) == expected
        assert _deep(build_abc(first, second)) == _deep(_reference_build_abc(first, second))


def test_degenerate_plans_identical():
    controls = ((0., 0., 0.), (1., 0., 0.), (2., 0., 0.))                # a straight directrix
    first = BezierRuledSupport(controls, (0., 0., 1.))
    cases = [QuadricSupport("plane", (0., 0., 0.), (0., 1., 0.)),          # coincident: A = B = C = 0
             QuadricSupport("plane", (0., 0., 0.), (0., 0., 1.)),          # A = 0, B constant
             QuadricSupport("plane", (0., 0., 0.), (1., 0., 0.)),          # rulings parallel to the plane
             QuadricSupport("cylinder", (1., 0., 0.), (0., 0., 1.), radius=1.),   # tangent at the ends
             QuadricSupport("cylinder", (0., 0., 0.), (1., 0., 0.), radius=1.)]
    for second in cases:
        others = [c for c in cases]
        stations = [0.0, 1.0, -1.0, .5]
        assert _plan_outcomes(PolyPlan(first, second), others, stations) == _reference_outcomes(
            first, second, others, stations)
