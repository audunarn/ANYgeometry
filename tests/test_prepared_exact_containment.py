"""Prepared immutable boundary data must preserve the exact containment route."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from anygeometry.arrangement_geometry import LinePath
from anygeometry.errors import GeometryError
from anygeometry.material_arrangement import (
    ArrangementPath, MaterialDomain, _contains_loop_exact,
    _exact_horizontal_roots, _prepare_exact_boundary_line,
)
from anygeometry.surfaces import Plane


def make_domain(plane=None):
    plane = plane or Plane((2., -3., 1.), (3., .4, .7), (.2, 2., .3))
    loops = (
        ((0., 0.), (4., 0.), (4., 3.), (2., 1.), (0., 3.)),
        ((.3, .3), (.3, .7), (.7, .7), (.7, .3)),
    )
    boundaries = tuple(tuple(
        ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
        for a, b in zip(loop, loop[1:] + loop[:1])) for loop in loops)
    return MaterialDomain(1, plane, boundaries)


@pytest.mark.parametrize("tolerance", (0., 1e-12, 1e-6))
def test_prepared_queries_match_uncached_exact_rational_oracle(tolerance):
    domain = make_domain()
    chart = domain._exact_chart()
    rng = np.random.default_rng(31)
    queries = [tuple(map(Fraction, row)) for row in rng.uniform(-1., 5., (70, 2))]
    queries += [chart.uv(path.curve.start) for loop in domain.boundaries for path in loop]
    for loop in domain.boundaries:
        for uv in queries:
            for boundary in (False, True):
                expected = _contains_loop_exact(chart, loop, uv, tolerance, boundary=boundary)
                assert domain.contains_loop(loop, uv, tolerance, boundary=boundary) == expected
    cache = domain._exact_boundary_cache[2]
    assert len(cache) == len(domain.boundaries)
    assert sum(map(len, cache.values())) <= sum(map(len, domain.boundaries))


@pytest.mark.parametrize("tolerance", (0., 1e-12, 1e-6))
def test_prepared_horizontal_roots_preserve_exact_snapping(tolerance):
    domain = make_domain()
    chart = domain._exact_chart()
    for loop in domain.boundaries:
        for path in loop:
            prepared = _prepare_exact_boundary_line(chart, path.curve)
            start, delta = prepared.start_level, prepared.delta_level
            levels = (start, start + delta, start + delta / 2,
                      start + Fraction(tolerance), start - Fraction(tolerance))
            for level in levels:
                assert _exact_horizontal_roots(
                    chart, path.curve, level, tolerance, prepared=prepared
                ) == _exact_horizontal_roots(chart, path.curve, level, tolerance)


def test_copied_domains_do_not_reuse_boundary_or_support_data():
    domain = make_domain()
    loop = domain.boundaries[0]
    uv = (Fraction(1), Fraction(1))
    before_repr = repr(domain)
    before_hash = hash(domain.boundaries)
    domain.contains_loop(loop, uv, 1e-9)
    assert repr(domain) == before_repr
    assert hash(domain.boundaries) == before_hash
    assert replace(domain) == domain
    other = replace(domain, support=Plane((8., 1., 2.), (2., .4, .7), (.2, 3., .3)))
    other.contains_loop(loop, uv, 1e-9)
    assert other._exact_boundary_cache[2] is not domain._exact_boundary_cache[2]
    assert other.contains_loop(loop, uv, 1e-9) == _contains_loop_exact(
        other._exact_chart(), loop, uv, 1e-9)
    reversed_loop = tuple(reversed(loop))
    replacement = replace(domain, boundaries=(reversed_loop,))
    replacement.contains_loop(reversed_loop, uv, 1e-9)
    assert replacement._exact_boundary_cache[1] is replacement.boundaries
    assert domain._prepared_exact_boundary(tuple(reversed(loop))) is None


def test_mutable_curve_payloads_and_subclasses_use_uncached_route():
    domain = make_domain(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    loop = tuple(replace(path, curve=LinePath(list(path.curve.start), list(path.curve.end)))
                 for path in domain.boundaries[0])
    domain = replace(domain, boundaries=(loop,))
    assert domain._prepared_exact_boundary(loop) is None
    uv = (Fraction(1), Fraction(1))
    domain.contains_loop(loop, uv, 1e-9)
    loop[0].curve.start[0] += .25
    assert domain.contains_loop(loop, uv, 1e-9) == _contains_loop_exact(
        domain._exact_chart(), loop, uv, 1e-9)
    assert domain._exact_boundary_cache is None

    class MutablePlane(Plane):
        pass

    subclass_domain = make_domain(MutablePlane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    assert subclass_domain._prepared_exact_boundary(subclass_domain.boundaries[0]) is None


def test_lazy_preparation_preserves_early_return_and_degenerate_refusal():
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    first = ArrangementPath(LinePath((0., 0., 0.), (1., 0., 0.)))
    invalid = ArrangementPath(LinePath((float("nan"), 0., 0.), (1., 0., 0.)))
    domain = MaterialDomain(1, plane, ((first, invalid),))
    loop = domain.boundaries[0]
    # The later invalid edge must never be prepared after a boundary return.
    assert domain.contains_loop(loop, (Fraction(1, 2), Fraction(0)), 1e-9)
    assert len(domain._exact_boundary_cache[2][id(loop)]) == 1
    bad = ArrangementPath(LinePath((0., 0., 0.), (0., 0., 0.)))
    domain = MaterialDomain(1, plane, ((bad,),))
    for _ in range(2):
        with pytest.raises(GeometryError, match="degenerate arrangement line"):
            domain.contains_loop(domain.boundaries[0], (Fraction(0), Fraction(0)), 1e-9)
