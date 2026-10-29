import numpy as np
import pytest
from anygeometry import GeometryError, evaluate_material_surface_region
from anygeometry._detached_geometry_read import (detach_numeric, NumericalBoundary,
    MaterialNumericalRead, canonical_material_context)
from test_prepared_material_curvilinear import square_cells


def test_none_ticks_do_not_requalify_and_closed_lifetime_refuses():
    calls=[]
    boundary=NumericalBoundary(lambda:calls.append(1),None)
    for _ in range(20): boundary.tick()
    assert len(calls)==1
    boundary.finish()
    assert len(calls)==2
    with pytest.raises(GeometryError,match='closed'): boundary.tick()


def test_callback_frontiers_and_original_exception_identity():
    calls=[]
    token=RuntimeError('external callback')
    def fail(phase): raise token
    boundary=NumericalBoundary(lambda:calls.append(1),fail)
    with pytest.raises(RuntimeError) as caught: boundary.tick()
    assert caught.value is token
    assert len(calls)==2
    cancelled=NumericalBoundary(lambda:None,lambda phase:True)
    with pytest.raises(GeometryError,match='cancelled'):cancelled.tick()


def test_detached_support_and_final_source_mutation_refusal():
    model,regions,chords,cells=square_cells()
    view=MaterialNumericalRead(model,regions,cells[0].face.id,chords)
    uv=np.array([[.2,.3],[.8,.7]])
    expected=evaluate_material_surface_region(model,regions,cells[0].face.id,uv)
    assert view.support is not regions.regions[0].support
    np.testing.assert_array_equal(view.evaluate(uv),expected)
    old=regions.regions[0].support.controls
    changed=tuple(tuple((x,y,z+1) for x,y,z in row) for row in old)
    object.__setattr__(regions.regions[0].support,'controls',changed)
    np.testing.assert_array_equal(view.evaluate(uv),expected)
    with pytest.raises(GeometryError):view.validate()
    view.close()
    with pytest.raises(GeometryError,match='closed'):view.evaluate(uv)


def test_material_read_exact_derivatives_and_bad_parameters():
    model,regions,chords,cells=square_cells()
    view=MaterialNumericalRead(model,regions,cells[0].face.id,chords)
    uv=np.array([[[0.,0.],[.2,.7]],[[.8,.4],[1.,1.]]])
    expected=evaluate_material_surface_region(model,regions,cells[0].face.id,uv,derivatives=True)
    for actual,wanted in zip(view.evaluate(uv,derivatives=True),expected):
        np.testing.assert_array_equal(actual,wanted)
    for bad in ([[float('nan'),.5]],[[1.1,.5]],[.5]):
        with pytest.raises(GeometryError):view.evaluate(bad)
    view.validate();view.close()


def test_patched_numeric_method_disables_read_context(monkeypatch):
    from anygeometry.nurbs import NurbsSurface
    original=NurbsSurface.evaluate_many
    def wrapper(self,uv):return original(self,uv)
    monkeypatch.setattr(NurbsSurface,'evaluate_many',wrapper)
    assert not canonical_material_context()


def test_actual_callback_mutation_refused_at_same_frontier():
    model,regions,chords,cells=square_cells()
    support=regions.regions[0].support
    def mutate(phase):object.__setattr__(support,'u_range',(.1,.9))
    boundary=NumericalBoundary(lambda:None,mutate,source=model)
    with pytest.raises(GeometryError):boundary.tick()


@pytest.mark.parametrize('external_error',(False,True))
def test_real_field_frontier_preserves_refusal_subtype_and_callback_identity(external_error):
    from anygeometry.local_approximation_field import query_local_approximation_field
    from anygeometry._local_curvature import LocalApproximationNotCertified
    model,regions,chords,cells=square_cells()
    token=GeometryError('actual external callback exception')
    reached=[]
    def callback(phase):
        if phase != 'local approximation field': return False
        reached.append(phase)
        if external_error: raise token
        return True
    with pytest.raises(GeometryError) as caught:
        query_local_approximation_field(model,(cells[0].face.id,),relative_accuracy=.001,
            scale_error=.001,policy_floor=1e-8,cancellation_check=callback)
    assert reached
    if external_error: assert caught.value is token
    else: assert type(caught.value) is LocalApproximationNotCertified
