"""Immutable per-curve preparation preserves the selected kernel's results."""
import copy
import pickle
import textwrap
from pathlib import Path
import numpy as np
import pytest
from anygeometry import GeometryError
from anygeometry.nurbs import NurbsCurve,nurbs_record,nurbs_from_record
from anygeometry import nurbs as module


def curve():
    return NurbsCurve(3,(0,0,0,0,.5,1,1,1,1),((0,0,0),(1,2,0),(2,-1,1),(3,2,0),(4,0,0)),(1.,.5,1.2,.8,1.))


@pytest.mark.parametrize('order',[0,1,2])
@pytest.mark.parametrize('domain',[(0.,1.),(1.,0.),(.1,.5),(.5,.9)])
def test_prepared_result_is_byte_identical_to_public_checked_route(order,domain):
    from dataclasses import replace
    c=replace(curve(),domain=domain)
    values=np.array((0.,.01,.3,.5,.99,1.))
    # The old derivative implementation is reproduced here without using the
    # new private definition cache. It still calls the public checked backend.
    s,scale=c._knot_parameters(values)
    expected=module._call(module.backend.bspline_curve,c.degree,c.knots,c.controls,c.weights,np.ravel(s),order)
    if order:c._inward_end_rows(np.ravel(s),expected,order)
    for k in range(1,order+1):expected[:,k:,:]*=scale
    np.testing.assert_array_equal(c.derivatives(values,order),expected)


def test_prepared_data_immutable_and_not_persisted_or_copied_as_trust():
    original=curve();record=nurbs_record(original)
    for candidate in (original,copy.copy(original),copy.deepcopy(original),pickle.loads(pickle.dumps(original)),nurbs_from_record(record)):
        assert nurbs_record(candidate)==record
        np.testing.assert_array_equal(candidate.derivatives([0,.31,1],2),original.derivatives([0,.31,1],2))
        assert len(candidate.__getstate__())==5
        for array in candidate._prepared[1:]:
            assert not array.flags.writeable
            with pytest.raises(ValueError):array.setflags(write=True)


@pytest.mark.parametrize('name,value', [('controls',((0.,0.,0.),)),('knots',(0.,1.)),('degree',0),('domain',(-1.,2.)),('weights',(1.,))])
def test_frozen_definition_tampering_refuses_instead_of_using_stale_data(name,value):
    c=curve();object.__setattr__(c,name,value)
    with pytest.raises(GeometryError,match='changed after validation'):c.evaluate(.4)


@pytest.mark.parametrize('parameters',[float('nan'),float('inf'),-1e-12,1.00001])
def test_prepared_route_retains_parameter_refusals(parameters):
    with pytest.raises(GeometryError):curve().evaluate(parameters)


def test_endpoint_reuse_is_bounded_detached_and_checks_inputs_before_hits(monkeypatch):
    c=curve();original=module.backend._evaluate_prepared_bspline_curve;calls=[]
    def evaluate(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module.backend,'_evaluate_prepared_bspline_curve',evaluate)
    first=c.evaluate(0.);first[:]=99.
    np.testing.assert_array_equal(c.evaluate(np.array([0.]))[0],np.array((0.,0.,0.)))
    assert len(calls)==1
    c.evaluate(1.);c.evaluate(np.array([[1.]]))
    assert len(calls)==2 and len(c._endpoint_values)==2
    for values in c._endpoint_values:
        with pytest.raises(ValueError):values.setflags(write=True)
    with pytest.raises(GeometryError):c.evaluate(-1e-10)
    with pytest.raises(GeometryError):c.derivatives(0.,-1)
    object.__setattr__(c,'domain',(-1.,2.))
    with pytest.raises(GeometryError,match='changed after validation'):c.evaluate(0.)


def test_copy_pickle_drop_endpoint_caches_and_rebuild_immutable_definition():
    c=curve();c.evaluate(0.);c.evaluate(1.)
    for candidate in (copy.copy(c),copy.deepcopy(c),pickle.loads(pickle.dumps(c))):
        assert candidate._endpoint_values==(None,None)
        for array in candidate._prepared[1:]:
            with pytest.raises(ValueError):array.setflags(write=True)
        np.testing.assert_array_equal(candidate.evaluate(0.),c.evaluate(0.))


def test_public_dataclass_fields_and_legacy_public_state_unchanged():
    from dataclasses import fields,asdict
    c=curve()
    assert [field.name for field in fields(c)]==['degree','knots','controls','weights','domain']
    state=asdict(c)
    assert set(state)=={'degree','knots','controls','weights','domain'}
    restored=object.__new__(NurbsCurve)
    restored.__setstate__({**state,'_prepared':'untrusted obsolete cache'})
    assert nurbs_record(restored)==nurbs_record(c)
    assert restored._prepared!='untrusted obsolete cache'
    np.testing.assert_array_equal(restored.evaluate([0,.3,1]),c.evaluate([0,.3,1]))


def test_exact_scalar_cache_bounded_unquantized_detached_and_reset():
    c=curve()
    for t in np.linspace(.001,.999,150):c.evaluate(t)
    assert len(c._point_values)==128
    t=.3141592653589793
    first=c.evaluate(t);first[:]=1000.
    next_value=np.nextafter(t,1.)
    c.evaluate(next_value)
    s,scale=c._knot_parameters(t);other,scale=c._knot_parameters(next_value)
    assert np.float64(s).tobytes() in c._point_values
    assert np.float64(other).tobytes() in c._point_values
    for candidate in (copy.copy(c),copy.deepcopy(c),pickle.loads(pickle.dumps(c))):
        assert candidate._point_values=={}
        np.testing.assert_array_equal(candidate.evaluate(t),c.evaluate(t))
    for array in c._point_values.values():
        with pytest.raises(ValueError):array.setflags(write=True)
    with pytest.raises(GeometryError):c.evaluate(float('nan'))


def test_parallel_scalar_evictions_preserve_exact_points_and_bound():
    from concurrent.futures import ThreadPoolExecutor
    c=curve();parameters=np.linspace(.001,.999,300)
    expected=[module.backend.bspline_curve(c.degree,c.knots,c.controls,c.weights,[t],0)[0,0] for t in parameters]
    jobs=list(enumerate(parameters))*2
    with ThreadPoolExecutor(max_workers=12) as pool:
        values=list(pool.map(lambda job:(job[0],c.evaluate(job[1])),jobs))
    for index,value in values:np.testing.assert_array_equal(value,expected[index])
    assert len(c._point_values)<=128
    values[0][1][:]=99.
    np.testing.assert_array_equal(c.evaluate(parameters[0]),expected[0])



