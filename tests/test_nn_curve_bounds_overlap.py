"""Exact overlap-window parity with the former complete component scan."""
from fractions import Fraction as Q

import pytest
from anygeometry.errors import GeometryError

from anygeometry.nurbs_surface_intersection.curve import _covered_arc_indices
from anygeometry.nurbs_surface_intersection._source import _freeze


def scan(start, end, count):
    return tuple(i for i in range(count)
        if not (max(start,Q(i)) > min(end,Q(i+1))
                or max(start,Q(i)) == min(end,Q(i+1)) and start != end))


@pytest.mark.parametrize('count', (1,2,19,4096))
def test_window_matches_original_exact_overlap_predicate(count):
    stations = sorted({Q(0),Q(count),Q(count,2),Q(1,3),Q(2,3),
                       Q(max(0,count-1)),Q(2*count-1,2)})
    for start in stations:
        for end in stations:
            if start <= end:
                assert tuple(_covered_arc_indices(start,end,count)) == scan(start,end,count)


def test_singleton_seams_and_positive_interval_endpoint_ownership():
    assert tuple(_covered_arc_indices(Q(0),Q(0),9)) == (0,)
    assert tuple(_covered_arc_indices(Q(9),Q(9),9)) == (8,)
    assert tuple(_covered_arc_indices(Q(4),Q(4),9)) == (3,4)
    assert tuple(_covered_arc_indices(Q(4),Q(5),9)) == (4,)
    assert tuple(_covered_arc_indices(Q(7,2),Q(4),9)) == (3,)


@pytest.mark.parametrize('kind', ('scalar','rational','generic'))
def test_all_bound_fields_and_world_hulls_match_original_scan(kind, monkeypatch):
    import importlib
    module = importlib.import_module('anygeometry.nurbs_surface_intersection.curve')
    from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection, NurbsIntersectionCurve, IntersectionStatus
    from anygeometry.tolerance import TolerancePolicy
    from types import SimpleNamespace
    from test_prepared_curve_bounds import sources
    first,second = sources(kind)
    policy = SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24) if kind == 'generic' else None
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy(),qualification=policy)
    assert result.status is IntersectionStatus.CERTIFIED
    owner = NurbsIntersectionCurve.from_result(result)
    context = owner.prepare(first=first,second=second,tolerances=TolerancePolicy(),qualification=policy)
    n = int(context.domain[1])
    windows = [(Q(0),Q(0)),(Q(n),Q(n)),(Q(n,3),Q(2*n,3)),(Q(0),Q(n))]
    if n > 1:
        windows += [(Q(1),Q(1)),(Q(1,2),Q(1)),(Q(1),Q(3,2))]
    optimized = module._covered_arc_indices
    for start,end in windows:
        actual = context.enclose(start,end),context._world_bounds(start,end)
        monkeypatch.setattr(module,'_covered_arc_indices',lambda a,b,count:range(count))
        assert (context.enclose(start,end),context._world_bounds(start,end)) == actual
        monkeypatch.setattr(module,'_covered_arc_indices',optimized)
    for reverse in (False,True):
        shape = context.shape().subcurve(Q(3,4),Q(1,4)) if reverse else context.shape()
        for start,end in ((Q(0),Q(0)),(Q(1),Q(1)),(Q(1,3),Q(2,3))):
            actual = shape.enclose(start,end),shape._world_bounds(start,end)
            monkeypatch.setattr(module,'_covered_arc_indices',lambda a,b,count:range(count))
            assert (shape.enclose(start,end),shape._world_bounds(start,end)) == actual
            monkeypatch.setattr(module,'_covered_arc_indices',optimized)


def test_native_freeze_retains_complete_recursive_mutation_checks_and_subclass_fallback():
    nested = [(1,2.,'three'),[4,5]]
    original = _freeze(nested)
    nested[1][0] = 99
    assert _freeze(nested) != original
    class Sequence(tuple):
        pass
    assert _freeze(Sequence((1,[2,3]))) == (1,(2,3))
    with pytest.raises(GeometryError,match='unsupported'):
        _freeze((object(),))
