"""B-spline / NURBS curve kernel (slice N1a): references, independent checks, native agreement.

Independent references: the worked basis-function example of Piegl and Tiller
(The NURBS Book, 2nd ed., Ex. 2.3/2.4); an exact rational circle; Bezier
decomposition by knot insertion evaluated by a separate homogeneous de
Casteljau implementation in this file; Richardson-extrapolated finite
differences for derivatives. The compiled kernel is compared with the
pure-Python reference when it is in use.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import backend, reference
from anygeometry.reference import bspline as ref

NATIVE = backend.backend_info()["backend"] == "native"
needs_native = pytest.mark.skipif(not NATIVE, reason=f"compiled core not in use: {backend.backend_info()['reason']}")

HALF = math.sqrt(0.5)
CIRCLE = (2, np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1.]),
          np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0.]]),
          np.array([1, HALF, 1, HALF, 1, HALF, 1, HALF, 1.]))


def evaluate(degree, knots, controls, weights, t, order=0, route="backend"):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if route == "backend":
        return backend.bspline_curve(degree, knots, controls, weights, t, order)
    out = np.empty((len(t), order + 1, np.shape(controls)[1]))
    ref.bspline_evaluate(degree, np.asarray(knots, float), np.asarray(controls, float),
                         np.empty(0) if weights is None else np.asarray(weights, float), t, order, out)
    return out


def random_curve(rng, degree, count, dim=3, rational=True, repeated=()):
    interior = np.sort(rng.uniform(0.0, 1.0, count - degree - 1))
    for index, times in repeated:          # raise the multiplicity of chosen interior knots
        interior = np.sort(np.concatenate((interior[:len(interior) - times + 1],
                                           np.repeat(interior[index], times - 1))))
    knots = np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))
    controls = rng.standard_normal((count, dim))
    weights = rng.uniform(0.3, 3.0, count) if rational else None
    return degree, knots, controls, weights


def de_casteljau_rational(controls, weights, s):
    """Independent homogeneous de Casteljau evaluation of one rational Bezier segment."""
    points = np.concatenate((controls * weights[:, None], weights[:, None]), axis=1).astype(float)
    while len(points) > 1:
        points = (1 - s) * points[:-1] + s * points[1:]
    return points[0, :-1] / points[0, -1]


# ----------------------------------------------------------------------------------- references


def test_basis_functions_match_the_textbook_example():
    # Piegl & Tiller Ex. 2.3/2.4: p = 2, U = {0,0,0,1,2,3,4,4,5,5,5}, u = 5/2, span 4.
    knots = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5.])
    t = np.array([2.5])
    spans = ref.find_spans(2, knots, 8, t)
    assert spans.tolist() == [4]
    ders = ref.basis_derivatives(2, knots, spans, t, 2)[0]
    np.testing.assert_allclose(ders[0], [1 / 8, 6 / 8, 1 / 8], rtol=0, atol=1e-15)
    np.testing.assert_allclose(ders[1], [-1 / 2, 0, 1 / 2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(ders[2], [1, -2, 1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("route", ["backend", "reference"])
def test_the_rational_circle_is_exact_to_rounding(route):
    t = np.linspace(0.0, 1.0, 20001)
    values = evaluate(*CIRCLE, t, order=2, route=route)
    radius = np.linalg.norm(values[:, 0], axis=1)
    assert np.max(np.abs(radius - 1.0)) <= 4 * np.finfo(float).eps
    # The tangent is perpendicular to the radius; the second derivative is not tangential noise.
    assert np.max(np.abs(np.sum(values[:, 0] * values[:, 1], axis=1))) <= 1e-13
    np.testing.assert_allclose(values[[0, -1], 0], [[1, 0], [1, 0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(evaluate(*CIRCLE, [.125], route=route)[0, 0], [HALF, HALF], atol=1e-15)


@pytest.mark.parametrize("degree,rational", [(1, False), (2, True), (3, False), (3, True), (5, True)])
def test_points_agree_with_bezier_decomposition_and_de_casteljau(degree, rational):
    rng = np.random.default_rng(100 + degree)
    curve = random_curve(rng, degree, degree + 6, rational=rational)
    for t0, t1, controls, weights in ref.bezier_segments(*curve):
        weights = np.ones(len(controls)) if weights is None else weights
        for s in (0.0, 0.3, 0.71, 1.0):
            expected = de_casteljau_rational(controls, weights, s)
            actual = evaluate(*curve, [t0 + s * (t1 - t0)])[0, 0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_derivatives_match_richardson_finite_differences(degree):
    rng = np.random.default_rng(7 + degree)
    curve = random_curve(rng, degree, degree + 5)
    t = np.array([0.11, 0.37, 0.52, 0.86])
    # Stay inside one span so the finite differences see a smooth function.
    t = np.array([value for value in t if not np.any(np.abs(curve[1] - value) < 2e-3)])
    exact = evaluate(*curve, t, order=2)

    def point(x):
        return evaluate(*curve, x)[:, 0]

    def first(h):
        return (point(t + h) - point(t - h)) / (2 * h)

    def second(h):
        return (point(t + h) - 2 * point(t) + point(t - h)) / h**2

    h = 1e-3
    np.testing.assert_allclose((4 * first(h / 2) - first(h)) / 3, exact[:, 1], rtol=0,
                               atol=1e-6 * max(1.0, np.max(np.abs(exact[:, 1]))))
    np.testing.assert_allclose((4 * second(h / 2) - second(h)) / 3, exact[:, 2], rtol=0,
                               atol=1e-4 * max(1.0, np.max(np.abs(exact[:, 2]))))


def test_knot_insertion_leaves_the_curve_unchanged():
    rng = np.random.default_rng(3)
    curve = random_curve(rng, 3, 9)
    t = np.linspace(0.0, 1.0, 401)
    before = evaluate(*curve, t, order=2)
    knots, controls, weights = ref.insert_knot(*curve, 0.4321, times=2)
    after = evaluate(3, knots, controls, weights, t, order=2)
    np.testing.assert_allclose(after[:, 0], before[:, 0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(after[:, 1:], before[:, 1:], rtol=1e-10, atol=1e-10)


def test_degree_one_has_no_second_derivative_and_is_piecewise_linear():
    knots = np.array([0, 0, .5, 1, 1.])
    controls = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0.]])
    values = evaluate(1, knots, controls, None, [0.25, 0.75], order=2)
    np.testing.assert_allclose(values[:, 0], [[1, 0, 0], [2, 1, 0]], atol=1e-15)
    np.testing.assert_allclose(values[:, 1], [[4, 0, 0], [0, 4, 0]], atol=1e-14)
    assert np.all(values[:, 2] == 0)


def test_a_full_multiplicity_interior_knot_keeps_position_continuous():
    rng = np.random.default_rng(11)
    degree, knots, controls, weights = random_curve(rng, 3, 7)
    knots = np.array([0, 0, 0, 0, .5, .5, .5, 1, 1, 1, 1.])  # C0 joint at 0.5
    controls = rng.standard_normal((7, 3))
    left = evaluate(3, knots, controls, weights, [0.5 - 1e-12])[0, 0]
    at = evaluate(3, knots, controls, weights, [0.5])[0, 0]
    np.testing.assert_allclose(at, controls[3], atol=1e-15)        # interpolated at the joint
    np.testing.assert_allclose(left, at, atol=1e-10)


# ----------------------------------------------------------------------------------- validation


@pytest.mark.parametrize("change,message", [
    (dict(t=[1.5]), "outside the knot domain"),
    (dict(t=[-1e-12]), "outside the knot domain"),
    (dict(weights=np.array([1, 0, 1, 1, 1, 1, 1, 1, 1.])), "weights must be positive"),
    (dict(knots=np.array([0, 0, 0, .5, .25, .5, .5, .75, .75, 1, 1, 1])), "non-decreasing"),
    (dict(degree=0), "unsupported"),
    (dict(order=3), "unsupported"),
])
def test_invalid_input_is_refused(change, message):
    arguments = dict(degree=CIRCLE[0], knots=CIRCLE[1], controls=CIRCLE[2], weights=CIRCLE[3], t=[0.5], order=0)
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        backend.bspline_curve(**arguments)


# ----------------------------------------------------------------------------------- native


@needs_native
@pytest.mark.parametrize("seed", range(6))
def test_native_kernel_agrees_with_the_reference(seed):
    rng = np.random.default_rng(seed)
    degree = int(rng.integers(1, 8))
    curve = random_curve(rng, degree, degree + int(rng.integers(1, 12)), dim=int(rng.integers(1, 5)),
                         rational=bool(seed % 2))
    t = np.concatenate((rng.uniform(0, 1, 500), curve[1][degree:-degree], [0.0, 1.0]))
    native = evaluate(*curve, t, order=2)
    expected = evaluate(*curve, t, order=2, route="reference")
    scale = np.maximum(1.0, np.max(np.abs(expected), axis=0))
    assert np.max(np.abs(native - expected) / scale) <= 1e-13


@needs_native
def test_native_kernel_refuses_bad_buffers_itself():
    core = backend._CORE
    degree, knots, controls, weights = CIRCLE
    out = np.empty((1, 1, 2))
    with pytest.raises(ValueError, match="outside the knot domain"):
        core.bspline_evaluate(degree, knots, controls, weights, np.array([2.0]), 0, out)
    with pytest.raises(ValueError, match="inconsistent buffer sizes"):
        core.bspline_evaluate(degree, knots[:-1], controls, weights, np.array([.5]), 0, out)
    with pytest.raises(ValueError, match="unsupported"):
        core.bspline_evaluate(16, knots, controls, weights, np.array([.5]), 0, out)


# ----------------------------------------------------------------------------------- review regressions


@pytest.mark.parametrize("knots,message", [
    ([0, 0, 0, 0], "clamped"),                                   # empty domain
    ([0, 0, 0, 1, 1], r"n \+ degree \+ 1"),                     # wrong length
    ([0, 0, 0, 1], "clamped"),                                   # end multiplicity degree + 2
    ([0, 0, 5e-324, 5e-324], "64 eps"),                          # subnormal domain: no representable span
])
def test_degenerate_knot_vectors_are_refused(knots, message):
    with pytest.raises(ValueError, match=message):
        backend.bspline_curve(1, np.array(knots, float), [[0.], [1.]], None, [0.], 2)


def test_a_span_of_a_few_ulps_is_refused():
    with pytest.raises(ValueError, match="64 eps"):
        backend.bspline_curve(1, [0, 0, 1, 1 + 1e-15, 2, 2.], [[0.], [1.], [2.], [3.]], None, [.5])


def test_interior_multiplicity_above_the_degree_is_refused():
    knots = np.array([0, 0, .5, .5, 1, 1.])                      # degree 1: a discontinuous joint
    controls = np.array([[0.], [1.], [10.], [11.]])
    with pytest.raises(ValueError, match="multiplicity must not exceed the degree"):
        backend.bspline_curve(1, knots, controls, None, [.75])
    with pytest.raises(ValueError, match="multiplicity <= degree"):
        ref.bezier_segments(1, knots, controls, None)


def test_huge_weights_are_scaled_not_overflowed():
    knots = np.array([0, 0, 1, 1.])
    values = backend.bspline_curve(1, knots, [[2.], [2.]], [1e308, 1e308], [.5], 1)
    np.testing.assert_array_equal(values[0], [[2.], [0.]])
    values = backend.bspline_curve(2, *CIRCLE[1:3], CIRCLE[3] * 1e300, [.125], 0)
    np.testing.assert_allclose(values[0, 0], [HALF, HALF], atol=1e-15)


def test_non_representable_results_are_refused():
    knots = np.array([0, 0, 1, 1.])
    # Positions stay in the control hull; a derivative over a tiny span can exceed float64.
    tiny = np.array([0, 0, 1e-290, 1e-290])
    with pytest.raises(ValueError, match="numerical failure"):
        backend.bspline_curve(1, tiny, [[-1e150], [1e150]], None, [5e-291], 1)


@needs_native
def test_native_kernel_refuses_unaligned_buffers():
    raw = np.ndarray((2, 1), dtype="f8", buffer=bytearray(17), offset=1)
    raw[:] = [[0.], [1.]]
    out = np.empty((1, 1, 1))
    with pytest.raises(ValueError, match="aligned"):
        backend._CORE.bspline_evaluate(1, np.array([0, 0, 1, 1.]), raw, np.empty(0), np.array([.5]), 0, out)
    # The validating wrapper copies to aligned storage and evaluates.
    np.testing.assert_array_equal(backend.bspline_curve(1, [0, 0, 1, 1.], raw, None, [.5])[0, 0], [.5])


@needs_native
def test_maximum_degree_agrees_with_the_reference():
    rng = np.random.default_rng(15)
    curve = random_curve(rng, 15, 20, dim=4, rational=True)
    t = np.concatenate((rng.uniform(0, 1, 200), [0., 1.]))
    native = evaluate(*curve, t, order=2)
    expected = evaluate(*curve, t, order=2, route="reference")
    scale = np.maximum(1.0, np.max(np.abs(expected), axis=0))
    assert np.max(np.abs(native - expected) / scale) <= 1e-11


def test_subnormal_weights_and_tiny_coordinates_are_refused_not_rounded_away():
    knots = np.array([0, 0, 1, 1.])
    with pytest.raises(ValueError, match="weight ratio"):
        backend.bspline_curve(1, knots, [[0.25], [1.0]], [1e-323, 1.0], [0.0])
    with pytest.raises(ValueError, match="control coordinates"):
        backend.bspline_curve(1, knots, [[1e-300], [1.0]], None, [0.0])
    with pytest.raises(ValueError, match="control coordinates"):
        backend.bspline_curve(1, knots, [[1e200], [1.0]], None, [0.0])
    # Inside the range the clamped end point is exact.
    np.testing.assert_array_equal(backend.bspline_curve(1, knots, [[0.25], [1.0]], [2.0**-200, 1.0], [0.0])[0, 0], [0.25])
