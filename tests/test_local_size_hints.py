from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry._local_size_hints import curvature_size_hints
from test_local_spatial_allocations import curved_source


def test_initial_size_hints_are_spatial_and_error_scales_without_admission():
    model,face=curved_source();before=owner.to_dict(model)
    first=curvature_size_hints(model,face,Q(1,1000))
    second=curvature_size_hints(model,face,Q(4,1000))
    assert len(first)>=16 and len({row[1] for row in first})>1
    assert all(a[0]==b[0] and a[1]*2==b[1] for a,b in zip(first,second))
    assert owner.to_dict(model)==before


def test_size_hint_callback_exception_and_mutation_stay_hard():
    model,face=curved_source();sentinel=RuntimeError('caller failure')
    def stop(phase):raise sentinel
    with pytest.raises(RuntimeError) as caught:curvature_size_hints(model,face,.01,cancellation_check=stop)
    assert caught.value is sentinel
    changed=False
    def mutate(phase):
        nonlocal changed
        if not changed:changed=True;model.add_point(3,4,5)
        return False
    with pytest.raises(owner.GeometryError,match='changed'):
        curvature_size_hints(model,face,.01,cancellation_check=mutate)
