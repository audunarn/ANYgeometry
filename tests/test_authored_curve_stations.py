"""Source identity and independent polynomial-carrier station contracts."""
from dataclasses import replace
from fractions import Fraction as F
import math

import numpy as np
import pytest

from anygeometry import (GeometryModel, GeometryError, BezierDirectrix, ExtrudedSurface,
                         Plane, plan_intersections, apply_intersections, to_dict)
from anygeometry.generators import cylinder
from anygeometry.entities import OrientedEdge
from anygeometry.arrangement_geometry import BezierPath, LinePath
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.material_arrangement import MaterialDomain
from anygeometry.authored_boundary_correspondence import query_prepared_authored_boundary_correspondence
from anygeometry.authored_domain_coverage import _original_domain
from anygeometry.authored_curve_stations import (
    query_prepared_authored_curve_stations as query,
    validate_prepared_authored_curve_station_coordinates as validate, _carrier)

C = ((0.,0.,0.), (1.,2.,0.), (2.,-1.,0.), (3.,1.,0.))
D = (.25,0.,1.5)


def build(ranges=None, *, narrow=False):
    model = GeometryModel()
    if ranges is None:
        points = model.add_points(C)
        edge = model.add_spline(points[0], points[1:-1], points[-1])
        wall = model.extrude((edge,), D)[0]
    else:
        urange, vrange = ranges
        base = np.asarray(BezierPath(C).subcurve(min(urange), max(urange)).controls)
        low = model.add_points(base+min(vrange)*np.asarray(D))
        high = model.add_points(base+max(vrange)*np.asarray(D))
        edges = (model.add_spline(low[0], low[1:-1], low[-1]), model.add_line(low[-1], high[-1]),
                 model.add_spline(high[0], high[1:-1], high[-1]), model.add_line(low[0], high[0]))
        wall = model.add_face_from_loop(tuple(OrientedEdge(e, f) for e, f in zip(edges, (True,True,False,False))),
            (0,1,2,3), surface=ExtrudedSurface(BezierDirectrix(C), D, urange, vrange))
    model.insert_model(cylinder(.3 if narrow else .7, 8., origin=(-2.,.5,.7) if narrow else (-2.,.4,.6), axis=(1.,0.,0.),
        radial_direction=(0.,1.,0.), circumferential_segments=8))
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    correspondence = query_prepared_authored_boundary_correspondence(model, wall)
    edges = tuple(e for e, _ in correspondence.interior_incidence if isinstance(model.edges[e].curve, BezierQuadricCurve))
    assert edges
    return model, correspondence, edges


@pytest.fixture(scope='module')
def pipe():
    # Same nine authored faces as verify_pipe_owner.py; no mesher/native calls.
    return build()


def unpack(rows):
    return tuple(tuple(F(*value) for value in row) for row in rows)


def check_formula(result, urange=(0,1), vrange=(0,1)):
    for (t,s), (u,v), xyz in zip(unpack(result.carrier_parameters), unpack(result.authored_uv), unpack(result.authored_points)):
        assert u == (t-F(urange[0]))/(F(urange[1])-F(urange[0]))
        assert v == (s-F(vrange[0]))/(F(vrange[1])-F(vrange[0]))
        assert xyz == (3*t+s/4, 6*t-15*t*t+10*t*t*t, 3*s/2)


def test_all_internal_branches_independent_carrier_chart_and_cylinder(pipe):
    model, corr, edges = pipe; before = to_dict(model)
    tau = np.array([1., .25, 0., .75, .25])
    modes, signs = set(), set()
    for edge in edges:
        curve = model.edges[edge].curve
        result = query(model, corr, edge, tau)
        assert result.parameters == tuple((F(float(x)).numerator, F(float(x)).denominator) for x in tau)
        assert result.occurrences == dict(corr.interior_incidence)[edge]
        assert len(result.occurrences) == 2 and result.occurrences[0][1] != result.occurrences[1][1]
        check_formula(result)
        modes.add(curve.parameterization); signs.add(curve.branch)
        t = np.array([float(row[0]) for row in unpack(result.carrier_parameters)])
        if curve.parameterization == 'linear':
            expected_t = curve.start+curve.sweep*tau
        elif curve.parameterization == 'left_square':
            expected_t = curve.start+curve.sweep*tau**2
        elif curve.parameterization == 'right_square':
            expected_t = curve.start+curve.sweep*(2*tau-tau**2)
        else:
            expected_t = curve.start+curve.sweep*np.sin(math.pi*tau/2)**2
        np.testing.assert_allclose(t, expected_t, rtol=0, atol=4*np.finfo(float).eps)
        xyz = np.array(unpack(result.authored_points), dtype=float)
        # Independent pipe equation and branch sign, including stabilized fold ends.
        np.testing.assert_allclose((xyz[:,1]-.4)**2+(xyz[:,2]-.6)**2, .7**2,
                                   rtol=0, atol=64*np.finfo(float).eps)
        assert np.all(curve.branch*(xyz[:,2]-.6) >= -64*np.finfo(float).eps)
        points = model.sample_edge(edge, tau); snapshot = points.tobytes()
        np.testing.assert_array_equal(np.array(unpack(result.current_curve_points), dtype=float), points)
        validate(model, result, points)
        assert points.tobytes() == snapshot
    assert signs == {-1,1}
    # This packet's radius encloses the entire directrix y range: no folds.
    assert modes == {'linear'}
    print('actual public fixture modes:', sorted(modes), 'branches:', sorted(signs), 'edges:', len(edges))
    assert to_dict(model) == before


@pytest.mark.parametrize('ranges', [((.25,.75),(.125,.875)), ((.75,.25),(.875,.125))])
def test_cropped_and_reversed_original_ranges(ranges):
    model, corr, edges = build(ranges)
    for edge in edges:
        result = query(model, corr, edge, [.75, .125, .5])
        check_formula(result, *ranges)
        validate(model, result, model.sample_edge(edge, np.array([.75,.125,.5])))


@pytest.mark.parametrize('parameters', [[F(1,3)], [1+0j], [np.nan], [np.inf], [True], [np.bool_(False)], [-.1], [1.1], ['0.5']])
def test_invalid_parameters_before_callbacks(pipe, parameters):
    model, corr, edges = pipe
    def callback(_): raise AssertionError('invalid station reached callback')
    with pytest.raises(GeometryError): query(model, corr, edges[0], parameters, cancellation_check=callback)


@pytest.mark.parametrize('parameters,expected', [
    (np.array([.5], dtype=np.float16), ((1,2),)),
    (np.array([.5], dtype=np.float32), ((1,2),)),
    (np.array([0,1], dtype=np.int64), ((0,1),(1,1))),
])
def test_numpy_real_source_station_types_preserve_exact_values(pipe, parameters, expected):
    model, corr, edges = pipe
    result = query(model, corr, edges[0], parameters)
    assert result.parameters == expected
    validate(model, result, model.sample_edge(edges[0], parameters.astype(float)))


def test_order_empty_and_input_detachment(pipe):
    model, corr, edges = pipe
    parameters = np.array([.75, .25, .75])
    def mutate(_): parameters[:] = .5
    result = query(model, corr, edges[0], parameters, cancellation_check=mutate)
    assert result.parameters == ((3,4),(1,4),(3,4))
    assert result.authored_uv[0] == result.authored_uv[2]
    empty = query(model, corr, edges[0], [])
    assert empty.authored_uv == empty.current_curve_points == ()
    validate(model, empty, np.empty((0,3)))


def test_exterior_and_unsupported_internal_identity_refuse(pipe):
    model, corr, _ = pipe
    exterior = corr.exterior_loops[0][0][2][0]
    with pytest.raises(GeometryError, match='paired internal'): query(model, corr, exterior, [.5])
    from test_authored_domain_coverage import planar
    model, corr = planar()
    unsupported = [e for e, _ in corr.interior_incidence if not isinstance(model.edges[e].curve, BezierQuadricCurve)]
    assert unsupported
    with pytest.raises(GeometryError, match='untransformed BezierQuadricCurve'): query(model, corr, unsupported[0], [.5])


def test_narrow_pipe_actual_fold_modes_have_independent_chart_and_carrier_coordinates():
    model, corr, edges = build(narrow=True)
    modes = set()
    tau = np.array([0., .25, .5, .75, 1.])
    for edge in edges:
        curve = model.edges[edge].curve; modes.add(curve.parameterization)
        result = query(model, corr, edge, tau)
        check_formula(result)
        actual_t = np.array([float(row[0]) for row in unpack(result.carrier_parameters)])
        factor = {'linear': lambda x: x, 'left_square': lambda x: x*x,
                  'right_square': lambda x: 2*x-x*x,
                  'both_sine': lambda x: np.sin(math.pi*x/2)**2}[curve.parameterization](tau)
        np.testing.assert_allclose(actual_t, curve.start+curve.sweep*factor, rtol=0, atol=4*np.finfo(float).eps)
        xyz = np.array(unpack(result.authored_points), dtype=float)
        np.testing.assert_allclose((xyz[:,1]-.5)**2+(xyz[:,2]-.7)**2, .3**2,
                                  rtol=0, atol=64*np.finfo(float).eps)
        validate(model, result, model.sample_edge(edge, tau))
    assert modes & {'left_square','right_square','both_sine'}
    print('actual narrow-pipe public modes:', sorted(modes), 'edges:', len(edges))


def test_carrier_refuses_transformed_changed_and_nonextruded_support(pipe):
    model, corr, edges = pipe; curve = model.edges[edges[0]].curve
    domain = _original_domain(corr.authored_definition, lambda: None)
    matrix = np.eye(4); matrix[0,3] = .25
    with pytest.raises(GeometryError, match='untransformed'): _carrier(curve.transformed(matrix), domain)
    changed = replace(domain, support=ExtrudedSurface(BezierDirectrix(C), (.5,0,1.5)))
    with pytest.raises(GeometryError, match='carrier differs'): _carrier(curve, changed)
    controls = (C[0], (1.,2.125,0.), *C[2:])
    changed = replace(domain, support=ExtrudedSurface(BezierDirectrix(controls), D))
    with pytest.raises(GeometryError, match='carrier differs'): _carrier(curve, changed)
    changed = replace(domain, support=Plane((0,0,0),(1,0,0),(0,1,0)))
    with pytest.raises(GeometryError, match='polynomial extrusion'): _carrier(curve, changed)
    with pytest.raises(GeometryError, match='untransformed'): _carrier(LinePath((0,0,0),(1,0,0)), domain)


def test_forged_receipts_and_coordinate_errors(pipe):
    model, corr, edges = pipe; edge = edges[0]
    result = query(model, corr, edge, [.5]); points = model.sample_edge(edge, np.array([.5]))
    for forged in (replace(result, tolerance=(1,1)), replace(result, edge_id=edges[-1]),
                   replace(result, authored_uv=(((0,1),(0,1)),)), replace(result, occurrences=())):
        with pytest.raises(GeometryError): validate(model, forged, points)
    with pytest.raises(GeometryError, match='coordinate error'): validate(model, result, points+1e-4)
    for invalid in (points.astype(complex)+1j, np.full((1,3), np.nan), np.zeros((1,2))):
        with pytest.raises(GeometryError): validate(model, result, invalid)


def test_wrong_owner_and_stale_binding(pipe, monkeypatch):
    model, corr, edges = pipe
    with pytest.raises(GeometryError): query(model.clone(preserve_identity=True), corr, edges[0], [.5])
    monkeypatch.setattr(model, '_revision', model.revision+1)
    with pytest.raises(GeometryError): query(model, corr, edges[0], [.5])


def test_coordinate_copy_and_cancellation_identity(pipe):
    model, corr, edges = pipe; result = query(model, corr, edges[0], [.5])
    source = to_dict(model); points = model.sample_edge(edges[0], np.array([.5]))
    def mutate(_): points[:] = 100
    validate(model, result, points, cancellation_check=mutate)
    error = RuntimeError('cancel curve station')
    def cancel(_): raise error
    for call in (lambda: query(model,corr,edges[0],[.5],cancellation_check=cancel),
                 lambda: validate(model,result,model.sample_edge(edges[0],np.array([.5])),cancellation_check=cancel)):
        with pytest.raises(RuntimeError) as caught: call()
        assert caught.value is error
    with pytest.raises(GeometryError, match='cancelled'): query(model,corr,edges[0],[.5],cancellation_check=lambda _:True)
    assert to_dict(model) == source


@pytest.mark.parametrize('mutation', ['receipt', 'source', 'namespace'])
def test_last_callback_cannot_change_receipt_source_or_namespace(pipe, monkeypatch, mutation):
    model, corr, edges = pipe; edge = edges[0]
    stations = query(model, corr, edge, [.5]); points = model.sample_edge(edge, np.array([.5]))
    phases = []
    validate(model, stations, points, cancellation_check=lambda phase: phases.append(phase))
    remaining = len(phases)
    def callback(_):
        nonlocal remaining
        remaining -= 1
        if remaining == 0:
            if mutation == 'receipt':
                object.__setattr__(stations, 'tolerance', (1,1))
            elif mutation == 'source':
                curve = model.edges[edge].curve
                old = curve.start
                object.__setattr__(curve, 'start', old+2**-40)
            else:
                monkeypatch.setattr(model, '_model_id', GeometryModel().model_id)
    old_start = model.edges[edge].curve.start
    try:
        with pytest.raises(GeometryError): validate(model, stations, points, cancellation_check=callback)
        assert remaining == 0
    finally:
        if mutation == 'source': object.__setattr__(model.edges[edge].curve, 'start', old_start)


@pytest.mark.parametrize('assert_coordinates', [False, True])
def test_last_callback_exception_identity(pipe, assert_coordinates):
    model, corr, edges = pipe; edge = edges[0]
    stations = query(model, corr, edge, [.5]); points = model.sample_edge(edge, np.array([.5]))
    def call(callback):
        if assert_coordinates:
            return validate(model, stations, points, cancellation_check=callback)
        return query(model, corr, edge, [.5], cancellation_check=callback)
    phases = []
    call(lambda phase: phases.append(phase))
    remaining = len(phases); error = RuntimeError('last callback cancellation')
    source = to_dict(model)
    def cancel(_):
        nonlocal remaining
        remaining -= 1
        if remaining == 0: raise error
    with pytest.raises(RuntimeError) as caught: call(cancel)
    assert caught.value is error and remaining == 0 and to_dict(model) == source


def test_integral_edge_identity_is_normalized_once_before_callbacks(pipe):
    model, corr, edges = pipe; source = to_dict(model); original_id = model.model_id
    calls = 0
    class EdgeID(int):
        def __int__(self):
            nonlocal calls
            calls += 1
            if calls > 1:
                model._model_id = GeometryModel().model_id
            return int.__int__(self)
    try:
        result = query(model, corr, EdgeID(edges[0]), [.5])
        assert calls == 1 and type(result.edge_id) is int and result.edge_id == edges[0]
        assert to_dict(model) == source
    finally:
        model._model_id = original_id


@pytest.mark.parametrize('packed', [((1,0),), ((1,),), (('bad',2),), ((1,2,3),)])
def test_malformed_packed_receipt_parameters_refuse_typed(pipe, packed):
    model, corr, edges = pipe
    stations = query(model, corr, edges[0], [.5])
    forged = replace(stations, parameters=packed)
    with pytest.raises(GeometryError):
        validate(model, forged, model.sample_edge(edges[0], np.array([.5])))
