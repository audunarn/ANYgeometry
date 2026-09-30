from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.analytic_roots import isolate_real_roots, trigonometric_roots


def test_sturm_distinguishes_repeated_clustered_and_missing_real_roots():
    # Independent exact product, including a repeated root and a narrow pair.
    expected = [Fraction(-3), Fraction(1, 7), Fraction(1, 7)+Fraction(1, 10**8)]
    coefficients = [Fraction(1)]
    for root in [*expected, expected[1]]:
        made = [Fraction(0)]*(len(coefficients)+1)
        for index, value in enumerate(coefficients):
            made[index] -= root*value
            made[index+1] += value
        coefficients = made
    intervals = isolate_real_roots(coefficients, tolerance=1e-12)
    assert len(intervals) == 3
    for root, interval in zip(expected, intervals):
        assert interval.lower <= root <= interval.upper
    assert isolate_real_roots((1, 0, 1)) == ()


def test_exact_midpoint_root_does_not_duplicate_prior_isolated_roots():
    roots = isolate_real_roots((0, -2, -1, 1))
    assert [item.witness for item in roots] == pytest.approx([-1, 0, 2], abs=1e-12)


def test_periodic_seam_and_double_tangency_roots_are_complete():
    assert trigonometric_roots((0, 0, 1, 0, 0)) == pytest.approx((0, math.pi, math.tau))
    assert trigonometric_roots((1, -1, 0, 0, 0)) == pytest.approx((0, math.tau))
    # cos(2t)=0 has four roots; reversed charts preserve their orientation.
    expected = np.arange(4)*math.pi/2+math.pi/4
    assert trigonometric_roots((0, 0, 0, 1, 0)) == pytest.approx(expected)
    assert trigonometric_roots((0, 0, 0, 1, 0), start=math.tau, sweep=-math.tau) == pytest.approx(expected[::-1])


def test_coincidence_and_cancellation_fail_without_partial_results():
    with pytest.raises(GeometryError, match="nonzero"):
        isolate_real_roots((0, 0))
    with pytest.raises(GeometryError, match="cancelled"):
        isolate_real_roots((-2, 0, 1), cancellation_check=lambda: True)
