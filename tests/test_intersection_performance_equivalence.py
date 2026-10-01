"""Faster intersection preparation must not change any result.

The 0.4.5 root-isolation loop and exhaustive segment-pair scans are kept below
as oracles. Production paths must agree with them exactly, including the
cancellation polls made by root isolation.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel
from anygeometry import analytic_roots as roots
from anygeometry.analytic_roots import IsolatedRoot, isolate_real_roots, root_isolation_memo


def _reference_square_free(p):
    """The Fraction Euclidean square-free part (the sequences of 0.4.5)."""
    first, second = p, roots._derivative(p)
    while second != (0,):
        first, second = second, roots._division(first, second)[1]
    return roots._division(p, first)[0]


def _reference_sturm(p):
    """The Fraction remainder sequence of 0.4.5, integerized row by row."""
    sequence = [p, roots._derivative(p)]
    while sequence[-1] != (0,):
        remainder = roots._division(sequence[-2], sequence[-1])[1]
        if remainder == (0,):
            break
        magnitude = abs(remainder[-1])
        sequence.append(tuple(-value / magnitude for value in remainder))
    rows = []
    for row in sequence:
        denominator = math.lcm(*(value.denominator for value in row))
        values = tuple(value.numerator * (denominator // value.denominator) for value in row)
        divisor = math.gcd(*values)
        rows.append(tuple(value // divisor for value in values))
    return tuple(rows)


def _reference_isolation(coefficients, *, tolerance=1e-13, cancellation_check=None, interval=None):
    """The 0.4.5 bisection: both endpoint Sturm counts at every node."""
    p = roots._trim([Fraction(value) for value in coefficients])
    tolerance = Fraction(tolerance)
    if not p or p == (0,) or tolerance <= 0:
        raise GeometryError("root isolation needs a nonzero polynomial and positive tolerance")
    if cancellation_check is not None and cancellation_check():
        raise GeometryError("analytic root isolation cancelled")
    if interval is not None:
        lower, upper = (Fraction(value) for value in interval)
    if len(p) == 1:
        return ()
    p = _reference_square_free(p)
    sequence = _reference_sturm(p)
    if interval is None:
        bound = Fraction(2) + max(abs(value / p[-1]) for value in p[:-1])
        lower, upper = -bound, bound
    else:
        for endpoint in (lower, upper):
            if roots._integer_value(sequence[0], endpoint) == 0:
                quotient, _ = roots._division(p, (-endpoint, Fraction(1)))
                remaining = _reference_isolation(quotient, tolerance=tolerance, interval=(lower, upper),
                                                 cancellation_check=cancellation_check)
                return tuple(sorted((IsolatedRoot(endpoint, endpoint), *remaining), key=lambda item: item.lower))
    pending = [(lower, upper)]
    found = []
    while pending:
        if cancellation_check is not None and cancellation_check():
            raise GeometryError("analytic root isolation cancelled")
        lower, upper = pending.pop()
        count = roots._variations(sequence, lower) - roots._variations(sequence, upper)
        if count == 0:
            continue
        if count == 1 and upper - lower <= tolerance:
            found.append(IsolatedRoot(lower, upper))
            continue
        middle = (lower + upper) / 2
        if roots._integer_value(sequence[0], middle) == 0:
            quotient, _ = roots._division(p, (-middle, Fraction(1)))
            remaining = _reference_isolation(quotient, tolerance=tolerance,
                                             cancellation_check=cancellation_check, interval=interval)
            return tuple(sorted((IsolatedRoot(middle, middle), *remaining), key=lambda item: item.lower))
        pending.extend(((middle, upper), (lower, middle)))
    return tuple(sorted(found, key=lambda item: item.lower))


def _polynomials(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        degree = rng.choice((1, 2, 3, 4, 4, 4, 6, 8))
        kind = rng.random()
        if kind < .4:
            yield [rng.uniform(-5, 5) for _ in range(degree + 1)]
        elif kind < .8:
            # product of rational (often dyadic) linear factors: exact and repeated roots
            coefficients = [Fraction(1)]
            for _ in range(degree):
                root = Fraction(rng.randint(-20, 20), 2 ** rng.randint(0, 6))
                coefficients = [(coefficients[i - 1] if i else 0) - root * (coefficients[i] if i < len(coefficients) else 0)
                                for i in range(len(coefficients) + 1)]
            yield [float(value) for value in coefficients]
        else:
            yield [rng.choice((0.0, 1e-9, 1e9, 1.0, -1.0, 3.0)) * rng.uniform(.5, 2) for _ in range(degree + 1)]


def _polled(function, *arguments, cancel_at=None, **options):
    polls = [0]

    def check():
        polls[0] += 1
        return cancel_at is not None and polls[0] >= cancel_at

    try:
        value = function(*arguments, cancellation_check=check, **options)
    except GeometryError as error:
        value = ("error", str(error))
    else:
        value = tuple((item.lower, item.upper) for item in value)
    return value, polls[0]


@pytest.mark.parametrize("options", (
    {}, {"tolerance": 1e-6}, {"interval": (-1, 1)}, {"interval": (Fraction(-1, 2), Fraction(3, 4))},
    {"interval": (-1, 1), "tolerance": 4 * np.finfo(float).eps}))
def test_root_isolation_visits_the_same_intervals_and_polls_as_the_reference(options):
    for coefficients in _polynomials(11, 120):
        for cancel_at in (None, 4):
            assert (_polled(isolate_real_roots, coefficients, cancel_at=cancel_at, **options)
                    == _polled(_reference_isolation, coefficients, cancel_at=cancel_at, **options))


def _integer_polynomials(seed, count):
    """Integer polynomials with repeated factors and coefficients of hundreds of bits (as the exact quadric plans make)."""
    rng = random.Random(seed)
    for _ in range(count):
        factors = rng.randint(1, 5)
        polynomial = [rng.randint(1, 10 ** rng.randint(1, 60))]
        for _ in range(factors):
            if rng.random() < .6:
                factor = [rng.randint(-10 ** rng.randint(1, 40), 10 ** rng.randint(1, 40)), rng.randint(1, 10 ** rng.randint(1, 40))]
            else:
                factor = [rng.randint(-10 ** 30, 10 ** 30), rng.randint(-10 ** 30, 10 ** 30), rng.randint(1, 10 ** 30)]
            for _ in range(rng.choice((1, 1, 1, 2, 3))):                      # a repeated factor
                polynomial = [sum(polynomial[i] * factor[k - i] for i in range(len(polynomial)) if 0 <= k - i < len(factor))
                              for k in range(len(polynomial) + len(factor) - 1)]
        if rng.random() < .2:
            polynomial = [0] * rng.randint(1, 2) + polynomial                  # roots at zero
        yield polynomial


def _proportional(a, b):
    a, b = [Fraction(v) for v in a], [Fraction(v) for v in b]
    return len(a) == len(b) and all(x * b[-1] == y * a[-1] for x, y in zip(a, b))


def test_integer_remainder_sequences_equal_the_fraction_sequences():
    for polynomial in _integer_polynomials(5, 150):
        p = roots._trim([Fraction(v) for v in polynomial])
        if len(p) < 2:
            continue
        assert roots._sturm(p) == _reference_sturm(p)                   # identical rows, not merely the same counts
        assert roots._sturm(tuple(polynomial)) == roots._sturm(p)       # integers and Fractions alike
        assert _proportional(roots._square_free(p), _reference_square_free(p))


def _reference_classes(poly):
    """Yun's decomposition with Fraction remainders (0.4.5)."""
    classes = []
    a = tuple(Fraction(v) for v in poly)
    zero = (Fraction(0),)
    while len(a) > 1:
        g, b = a, roots._derivative(a)
        while b != zero and b != (0,):
            g, b = b, roots._division(g, b)[1]
        classes.append(roots._division(a, g)[0] if len(g) > 1 else a)
        a = g
    return classes


def test_yun_classes_and_common_divisors_equal_the_fraction_versions():
    from anygeometry.quadric_algebra import _classes, _poly_gcd
    cases = list(_integer_polynomials(8, 120))
    for polynomial in cases:
        new, old = _classes(polynomial), _reference_classes(polynomial)
        assert len(new) == len(old) and all(_proportional(x, y) for x, y in zip(new, old))
    for first, second in zip(cases, cases[1:]):
        divisor = _poly_gcd([first, second, [0]])
        reference = tuple(Fraction(v) for v in first)
        other = tuple(Fraction(v) for v in second)
        while any(other):
            reference, other = other, roots._division(reference, other)[1]
        assert _proportional(divisor, reference)
    assert _poly_gcd([[0], [0, 0]]) is None


def test_quadric_chart_roots_agree_with_the_fraction_pipeline():
    from anygeometry.quadric_algebra import _chart_roots, _classes
    from anygeometry.analytic_roots import _integer_value, _variations
    tolerance = 4 * np.finfo(float).eps
    for polynomial in _integer_polynomials(13, 80):
        if len(polynomial) < 2:
            continue
        expected = []
        classes = _reference_classes(polynomial)
        sequences = [_reference_sturm(cls) for cls in classes]
        for root in _reference_isolation(polynomial, tolerance=tolerance, interval=(-1, 1)):
            multiplicity = 1
            for k, (cls, sequence) in enumerate(zip(classes[1:], sequences[1:]), start=2):
                if root.lower == root.upper:
                    hit = _integer_value(cls, root.lower) == 0
                else:
                    hit = _variations(sequence, root.lower) - _variations(sequence, root.upper) >= 1
                if hit:
                    multiplicity = k
            expected.append((root.witness, multiplicity))
        assert _chart_roots(polynomial) == expected


def test_scoped_root_memo_returns_identical_results_and_replays_cancellation():
    cases = list(_polynomials(23, 60))
    unscoped = {tuple(c): _polled(isolate_real_roots, c) for c in cases}
    with root_isolation_memo():
        for _ in range(2):
            for coefficients in cases:
                assert _polled(isolate_real_roots, coefficients) == unscoped[tuple(coefficients)]
        # A hit makes the polls its computation made, so a poll-count
        # cancellation lands on the same poll as without the memo.
        for coefficients in cases:
            _value, polls = unscoped[tuple(coefficients)]
            for cancel_at in range(1, min(polls, 6) + 1):
                assert (_polled(isolate_real_roots, coefficients, cancel_at=cancel_at)
                        == _polled(_reference_isolation, coefficients, cancel_at=cancel_at))


def test_root_memo_is_scoped_and_nested_scopes_share_it():
    assert roots._ROOT_MEMO.get() is None
    with root_isolation_memo():
        outer = roots._ROOT_MEMO.get()
        isolate_real_roots((-2, 0, 1))
        with root_isolation_memo():
            assert roots._ROOT_MEMO.get() is outer
        assert outer
    assert roots._ROOT_MEMO.get() is None


def test_root_memo_never_stores_a_failed_or_cancelled_isolation():
    with root_isolation_memo():
        with pytest.raises(GeometryError):
            isolate_real_roots((0,))
        with pytest.raises(GeometryError, match="cancelled"):
            isolate_real_roots((-2, 0, 1), cancellation_check=lambda: True)
        assert not roots._ROOT_MEMO.get()
        assert [item.witness for item in isolate_real_roots((-2, 0, 1))] == pytest.approx([-math.sqrt(2), math.sqrt(2)], abs=1e-12)


def _segment_batches(seed, batches, size=200):
    rng = np.random.default_rng(seed)
    for _ in range(batches):
        tolerance = float(rng.choice((1e-10, 1e-9, 1e-8, 1e-6)))
        scale = 10 ** rng.uniform(-3, 2)
        kind = rng.choice(("random", "touch", "parallel", "tiny"))
        a0 = rng.uniform(-scale, scale, (size, 2))
        angle = rng.uniform(0, math.tau, size)
        length = scale * rng.uniform(.01, 1.5, size)
        direction = np.column_stack((np.cos(angle), np.sin(angle)))
        a1 = a0 + direction * length[:, None]
        if kind == "random":
            b0 = rng.uniform(-scale, scale, (size, 2))
            b1 = rng.uniform(-scale, scale, (size, 2))
        elif kind == "touch":
            b0 = a0 + (a1 - a0) * rng.uniform(0, 1, size)[:, None] + rng.normal(0, 3 * tolerance, (size, 2))
            b1 = b0 + rng.uniform(-scale, scale, (size, 2))
        elif kind == "parallel":
            rotation = rng.normal(0, 10 ** rng.uniform(-12, -2, size), size)
            c, s = np.cos(rotation), np.sin(rotation)
            d = a1 - a0
            second = np.column_stack((d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c))
            normal = np.column_stack((-d[:, 1], d[:, 0])) / np.linalg.norm(d, axis=1)[:, None]
            b0 = (a0 + d * rng.uniform(-.5, 1.5, size)[:, None]
                  + normal * (rng.normal(0, 4 * tolerance, size) * rng.choice((0, 1, 1e3), size))[:, None])
            b1 = b0 + second * rng.uniform(.2, 2, size)[:, None]
        else:
            # A segment barely longer than the tolerance: the exact predicate
            # treats near-parallel with a far-away long segment as touching.
            a1 = a0 + direction * tolerance * rng.choice((.5, 1.1, 5, 2000), size)[:, None]
            b0 = a0 + np.column_stack((np.zeros(size), -scale * np.ones(size)))
            b1 = b0 + scale * np.column_stack((np.ones(size), 2 * np.ones(size)))
        yield a0, a1, b0, b1, tolerance


def test_segment_pair_filter_never_discards_a_pair_the_exact_predicate_accepts():
    exact = GeometryModel._segments_intersect_2d
    rejected = accepted = 0
    for a0, a1, b0, b1, tolerance in _segment_batches(5, 120):
        keep = GeometryModel._segment_pairs_possible(a0, a1, b0, b1, tolerance)
        for index in range(len(a0)):
            touching = exact(a0[index], a1[index], b0[index], b1[index], tolerance)
            accepted += touching
            rejected += not keep[index]
            assert keep[index] or not touching
    assert accepted and rejected  # both branches of the filter were exercised


def test_segment_pair_filter_keeps_everything_it_cannot_bound():
    a = np.array([[0., 0.], [0., 0.]]); b = np.array([[1., 0.], [np.nan, 0.]])
    c = np.array([[5., 5.], [5., 5.]]); d = np.array([[6., 5.], [6., 5.]])
    assert GeometryModel._segment_pairs_possible(a[:1], b[:1], c[:1], d[:1], 1e-10).tolist() == [False]
    # A non-finite coordinate leaves every pair to the exact predicate.
    assert GeometryModel._segment_pairs_possible(a, b, c, d, 1e-10).all()
    # A tolerance too small for the coordinate scale disables the filter.
    assert GeometryModel._segment_pairs_possible(a, b + 1e6, c, d, 1e-10).all()
    assert GeometryModel._segment_pairs_possible(a, b, c, d, 0.).all()
    assert GeometryModel._segment_pairs_possible(a[:0], b[:0], c[:0], d[:0], 1e-10).shape == (0,)


def _exhaustive_self_intersects(polygon, tolerance):
    count = len(polygon)
    for first in range(count):
        for second in range(first + 1, count):
            if (first + 1) % count == second or (second + 1) % count == first:
                continue
            if GeometryModel._segments_intersect_2d(
                    polygon[first], polygon[(first + 1) % count],
                    polygon[second], polygon[(second + 1) % count], tolerance):
                return True
    return False


def _exhaustive_intersect(first, second):
    return any(
        GeometryModel._segments_intersect_2d(first[i], first[(i + 1) % len(first)],
                                             second[j], second[(j + 1) % len(second)], 1e-10)
        for i in range(len(first)) for j in range(len(second)))


def _polygon(rng, count, scale, kind):
    if kind == "star":  # simple polygon
        angle = np.sort(rng.uniform(0, math.tau, count))
        radius = scale * rng.uniform(.4, 1., count)
        return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    if kind == "rectilinear":
        step = scale / count
        points = [(0., 0.)]
        for index in range(1, count):
            x, y = points[-1]
            points.append((x + step * rng.integers(-2, 3), y) if index % 2 else (x, y + step * rng.integers(-2, 3)))
        return np.asarray(points)
    return rng.uniform(-scale, scale, (count, 2))  # random: usually self-intersecting


def test_polygon_predicates_match_the_exhaustive_scalar_scans():
    rng = np.random.default_rng(31)
    seen = {"self": set(), "pair": set()}
    for _ in range(400):
        scale = 10 ** rng.uniform(-2, 1.5)
        kind = str(rng.choice(("star", "rectilinear", "random")))
        polygon = _polygon(rng, int(rng.integers(3, 30)), scale, kind)
        for tolerance in (1e-10, 1e-6):
            expected = _exhaustive_self_intersects(polygon, tolerance)
            assert GeometryModel._polygon_self_intersects(polygon, tolerance) is expected
            seen["self"].add(expected)
        other = _polygon(rng, int(rng.integers(3, 12)), scale, "star") + rng.uniform(-scale, scale, 2)
        expected = _exhaustive_intersect(polygon, other)
        assert GeometryModel._polygons_intersect(polygon, other) is expected
        seen["pair"].add(expected)
    assert seen == {"self": {True, False}, "pair": {True, False}}
