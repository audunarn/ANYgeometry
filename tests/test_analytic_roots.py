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


def test_bounded_projective_charts_enclose_known_roots_without_cauchy_bound():
    # Exact factor oracle includes both chart borders, a repeated border,
    # narrow interior roots and roots far outside the tangent chart.
    expected=(Fraction(-10**80),Fraction(-1),Fraction(-1,7),Fraction(0),
              Fraction(1,7),Fraction(1),Fraction(10**70))
    coefficients=[Fraction(1)]
    for root in (*expected,Fraction(1)):
        product=[Fraction(0)]*(len(coefficients)+1)
        for index,value in enumerate(coefficients):
            product[index]-=root*value
            product[index+1]+=value
        coefficients=product
    intervals=isolate_real_roots(coefficients,interval=(-1,1),tolerance=Fraction(1,10**14))
    assert len(intervals)==5
    for root,interval in zip(expected[1:-1],intervals):
        assert interval.lower<=root<=interval.upper
        assert interval.upper-interval.lower<=Fraction(1,10**14)
    reciprocal=isolate_real_roots(tuple(reversed(coefficients)),interval=(-1,1),tolerance=Fraction(1,10**14))
    reciprocal_expected=sorted(1/root for root in expected if root and abs(root)>=1)
    assert len(reciprocal)==len(reciprocal_expected)
    for root,interval in zip(reciprocal_expected,reciprocal):
        assert interval.lower<=root<=interval.upper


@pytest.mark.parametrize('interval',((1,1),(2,-1),(0,float('inf')),(0,),True))
def test_root_interval_rejects_invalid_bounds(interval):
    with pytest.raises(GeometryError,match='interval'):
        isolate_real_roots((-2,0,1),interval=interval)


def test_bounded_root_cancellation_also_applies_to_constant_and_exact_endpoints():
    for polynomial in ((1,),(-1,1),(0,1)):
        with pytest.raises(GeometryError,match='cancelled'):
            isolate_real_roots(polynomial,interval=(-1,1),cancellation_check=lambda:True)
