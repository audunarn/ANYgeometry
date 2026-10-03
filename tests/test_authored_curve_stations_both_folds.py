"""Independent double-fold equations through actual public preparation."""
from fractions import Fraction as F
from pathlib import Path
import runpy

import numpy as np
import pytest

from anygeometry import (query_prepared_authored_curve_stations as query,
    validate_prepared_authored_curve_station_coordinates as validate,
    evaluate_prepared_authored_face as evaluate, to_dict)


@pytest.fixture(scope='module')
def fold():
    fixture = runpy.run_path(str(Path(__file__).parents[1]/'examples/authored_fold_stations_handoff.py'))
    design, prepared, root, application, correspondence = fixture['build']()
    edges = fixture['double_fold_edges'](prepared, correspondence)
    assert edges, 'The public fixture must exercise a genuine two-fold branch'
    return design, prepared, correspondence, edges


def unpack(rows):
    return tuple(tuple(F(*value) for value in row) for row in rows)


def test_public_two_fold_stations_have_independent_original_chart(fold):
    design, model, corr, edges = fold
    source, current = to_dict(design), to_dict(model)
    tau = np.array((0., 2.**-20, .25, .5, .75, 1.-2.**-20, 1.))
    for edge in edges:
        curve = model.edges[edge].curve
        assert sorted((curve.start, curve.start+curve.sweep)) == [.25, .75]
        result = query(model, corr, edge, tau)
        parameters = np.array(unpack(result.carrier_parameters), dtype=float)
        np.testing.assert_allclose(parameters[:,0],
            curve.start+curve.sweep*np.sin(np.pi*tau/2)**2, rtol=0, atol=4*np.finfo(float).eps)
        for (t,s), uv, xyz in zip(unpack(result.carrier_parameters),
                unpack(result.authored_uv),unpack(result.authored_points)):
            assert uv == (t,s)
            assert xyz == (3*t,F(25,16)-3*t+3*t*t,-1+2*s)
        xyz = np.array(unpack(result.authored_points), dtype=float)
        # Exact construction: the pipe is y^2+z^2=1, independently of
        # the implementation's branch polynomial or witness samples.
        np.testing.assert_allclose(xyz[:,1]**2+xyz[:,2]**2, 1., rtol=0, atol=32*np.finfo(float).eps)
        assert np.all(curve.branch*xyz[:,2] >= -32*np.finfo(float).eps)
        np.testing.assert_array_equal(xyz[[0,-1]],np.array(((3*curve.start,1,0),
            (3*(curve.start+curve.sweep),1,0))))
        registered = model.sample_edge(edge,tau)
        before = registered.tobytes()
        validate(model,result,registered)
        assert registered.tobytes() == before
    assert to_dict(design) == source and to_dict(model) == current


def test_original_differential_at_double_fold_stations(fold):
    _, model, corr, edges = fold
    for edge in edges:
        result = query(model,corr,edge,(0.,.5,1.))
        uv = np.array(unpack(result.authored_uv),dtype=float)
        t,s = uv.T
        expected = np.column_stack((3*t,25/16-3*t+3*t*t,-1+2*s))
        np.testing.assert_allclose(evaluate(model,corr,uv),expected,rtol=0,atol=2e-15)
        du,dv = evaluate(model,corr,uv,derivatives=True)
        np.testing.assert_allclose(du,np.column_stack((np.full_like(t,3),-3+6*t,np.zeros_like(t))),rtol=0,atol=2e-15)
        np.testing.assert_array_equal(dv,np.tile((0.,0.,2.),(len(t),1)))
