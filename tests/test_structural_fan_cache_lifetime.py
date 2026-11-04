"""A validated native branch survives eviction of its derived cache entry."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryError
from anygeometry import support_curves
from anygeometry.nurbs_runtime_binding import validate_section_runtime
from test_structural_fan_bounds import analytic, bound


@pytest.fixture
def saturated_sections():
    bank = {}
    token = support_curves._RUNTIME_CACHE_BANK.set(bank)
    try:
        args, kwargs = analytic()
        curve = args[1].curve
        arcs = curve._arcs()
        cache = bank['sections']
        # Retain the real branch entry. Unconsumed occupancy sentinels force
        # the existing 64-entry eviction when validation derives its slab.
        cache.clear()
        cache[(curve.plane_record, curve.surface, curve.path)] = arcs
        cache.update({('unconsumed-occupancy', i): () for i in range(63)})
        assert len(cache) == 64
        yield args, kwargs, arcs
    finally:
        support_curves._RUNTIME_CACHE_BANK.reset(token)


def test_validation_returns_the_actual_validated_arcs_across_eviction(saturated_sections):
    args, _, arcs = saturated_sections
    validated = validate_section_runtime(args[1].curve)
    assert validated is arcs
    assert all(actual is retained for actual, retained in zip(validated, arcs))


@pytest.mark.parametrize('outside', (False, True))
def test_eviction_keeps_original_positive_and_chart_refusal(saturated_sections, outside):
    args, kwargs, _ = saturated_sections
    if outside:
        args = (*args[:2], (Q(-1), Q(1, 4)), *args[3:])
    result = bound(args, kwargs)
    if outside:
        assert result is None
    else:
        assert result is not None and 0 <= result[0] <= kwargs['surface_error']


def test_identical_detached_arc_is_not_an_owned_branch(saturated_sections):
    args, kwargs, _ = saturated_sections
    support, side, *rest = args
    lo, hi, arc, first, last = side.pieces[0]
    detached = replace(arc)
    assert detached is not arc
    side = replace(side, pieces=((lo, hi, detached, first, last),))
    with pytest.raises(GeometryError, match='not the current native branch'):
        bound((support, side, *rest), kwargs)


def test_eviction_does_not_hide_retained_arc_callback_mutation(saturated_sections):
    args, kwargs, arcs = saturated_sections
    arc = arcs[0]
    old = arc.s0
    def mutate():
        object.__setattr__(arc, 's0', old + Q(1, 100))
    try:
        with pytest.raises(GeometryError, match='operands changed during proof'):
            bound(args, kwargs, mutate)
    finally:
        object.__setattr__(arc, 's0', old)
