"""Control hulls certify fixed chords without replacing true source witnesses."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryError, GeometryModel, query_material_surface_regions
from anygeometry import query_prepared_material_boundary_sampling
from anygeometry import validate_prepared_material_boundary_sampling_binding
from anygeometry.nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
from anygeometry.line_ranges import LineRange
from anygeometry.registered_world_hulls import registered_world_hull, fixed_chord_hull_bound
from anygeometry.prepared_material_curvilinear import _world_chord, _WorldChordNeedsRefinement
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work
from anygeometry.serialization import to_dict


def _diagonal():
    return NurbsCurve(1, (0., 0., 1., 1.), ((0., 0., 0.), (1., 1., 1.)))


@pytest.mark.parametrize('reverse', (False, True))
def test_native_rational_hull_keeps_exact_controls_and_reversed_ranges(reverse):
    curve = NurbsCurve(2, (0., 0., 0., .5, 1., 1., 1.),
        ((0., 0., 0.), (.25, .5, 0.), (.75, .5, 0.), (1., 0., 0.)), (1., 2., 3., 1.))
    if reverse:
        curve = replace(curve, domain=(.875, .125))
    controls, tube = registered_world_hull(curve, .125, .875)
    assert controls and all(type(x) is Q for p in controls for x in p)
    assert any(Q(float(x)) != x for p in controls for x in p)
    assert all(a <= 0 <= b for a, b in tube)
    endpoints = tuple(curve.exact_point(t) for t in (Q(1, 8), Q(7, 8)))
    bound = fixed_chord_hull_bound(curve, .125, .875, endpoints)
    from anygeometry.nurbs_domain import _exact_segment_distance
    for i in range(33):
        t = Q(1, 8)+Q(3*i, 128)
        assert _exact_segment_distance(curve.exact_point(t), *endpoints) <= bound


def test_diagonal_fixed_chord_certifies_without_axis_box_subdivision():
    curve = _diagonal()
    work = Work(20, 2)
    token = CURRENT_WORK.set(work)
    try:
        error, coverage = _world_chord(curve, Q(9, 32), Q(5, 16),
            (curve.evaluate(9/32), curve.evaluate(5/16)), .0025, lambda: None)
        assert coverage == ((Q(9, 32), Q(5, 16)),)
        assert error <= .0025 and work.subdivisions == 0
    finally:
        CURRENT_WORK.reset(token)


def test_line_range_hull_retains_fraction_domain_and_literal_endpoint_error():
    curve = LineRange((0., 0., 0.), (1., 1., 0.), (Q(2, 3), Q(1, 3)))
    controls, _tube = registered_world_hull(curve, 0., 1.)
    assert controls == (curve.exact_point(0), curve.exact_point(1))
    endpoints = (curve.evaluate(0), curve.evaluate(1))
    error, coverage = _world_chord(curve, Q(0), Q(1), endpoints, 1e-10, lambda: None)
    assert error <= 1e-10 and coverage == ((Q(0), Q(1)),)
    with pytest.raises(GeometryError):
        _world_chord(curve, Q(0), Q(1), ((0., 0., 2.), (1., 1., 2.)), 1e-4, lambda: None)


def test_large_control_hull_only_refines_proof_until_actual_point_violates():
    curve = NurbsCurve(3, (0.,)*4+(1.,)*4,
        ((0., 0., 0.), (.25, 1., 0.), (.75, -1., 0.), (1., 0., 0.)))
    with pytest.raises(_WorldChordNeedsRefinement) as caught:
        _world_chord(curve, Q(0), Q(1), ((0., 0., 0.), (1., 0., 0.)), .02, lambda: None)
    assert caught.value.parameter != Q(1, 2)  # midpoint itself lies on chord


def test_hull_resource_and_cancellation_refuse_before_extraction(monkeypatch):
    curve = _diagonal()
    entered = []
    original = NurbsCurve._active_controls
    def tracked(self, *args):
        entered.append(True)
        return original(self, *args)
    monkeypatch.setattr(NurbsCurve, '_active_controls', tracked)
    token = CURRENT_WORK.set(Work(0, 0))
    try:
        with pytest.raises(GeometryError):
            registered_world_hull(curve, 0., 1.)
    finally:
        CURRENT_WORK.reset(token)
    assert not entered
    marker = RuntimeError('caller cancelled before extraction')
    def cancel():
        raise marker
    with pytest.raises(RuntimeError) as caught:
        _world_chord(curve, Q(0), Q(1), ((0.,)*3, (1.,)*3), .0025, cancel)
    assert caught.value is marker and not entered


def test_detached_curve_callback_mutation_refuses():
    curve = _diagonal()
    domain = curve.domain
    def change():
        object.__setattr__(curve, 'domain', (.25, .75))
    try:
        with pytest.raises(GeometryError, match='changed'):
            _world_chord(curve, Q(0), Q(1), ((0.,)*3, (1.,)*3), .0025, change)
    finally:
        object.__setattr__(curve, 'domain', domain)


def test_registered_hull_refuses_changed_native_evaluator():
    curve = _diagonal()
    prepared = curve._prepared
    try:
        object.__setattr__(curve, '_prepared', ())
        with pytest.raises(GeometryError):
            registered_world_hull(curve, 0., 1.)
    finally:
        object.__setattr__(curve, '_prepared', prepared)


def test_unregistered_world_curve_retains_existing_bbox_route():
    from anygeometry.arrangement_geometry import BezierPath
    curve = BezierPath(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.)))
    assert registered_world_hull(curve, 0., 1.) is None
    bound, coverage = _world_chord(curve, Q(0), Q(1),
        ((0., 0., 0.), (1., 0., 0.)), .0025, lambda: None)
    assert bound <= .0025 and coverage == ((Q(0), Q(1)),)


def test_iso_hull_fresh_extraction_ignores_poisoned_cache(monkeypatch):
    from anygeometry import nurbs
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 1.), (1., 1., 1.))))
    model = GeometryModel()
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = model.add_points(tuple(tuple(support.evaluate(*p)) for p in uv))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1)%4])))) for i, p in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    regions = query_material_surface_regions(model, [face])
    curve = model.edges[edges[0]].curve
    expected = registered_world_hull(curve, 0., 1.)
    native = curve._iso_world(fresh=True)
    degree, knots, controls, sigma, slack, domain = native
    poisoned = (degree, knots, tuple((x+100*w, y, z, w) for x, y, z, w in controls), sigma, slack, domain)
    # Isolate the cache from all other tests and retain the hostile entry: a
    # fresh read must neither consume nor repair that external value.
    monkeypatch.setattr(nurbs, '_ISO_CACHE', {curve: poisoned})
    assert registered_world_hull(curve, 0., 1.) == expected
    assert nurbs._ISO_CACHE[curve] is poisoned
    try:
        result = query_prepared_material_boundary_sampling(model, regions,
            ((model.handle('edge', edges[0]), (0., 1.)),), chord_error=.0025)
    except GeometryError:
        return  # old native box route may conservatively refuse hostile cache
    validate_prepared_material_boundary_sampling_binding(model, regions, result)
    assert all(row.world_error <= .0025 for row in result.chords.intervals)
    for row in result.chords.intervals:
        assert all(abs(x-z) <= .0025 and y == 0 for x, y, z in row.xyz)


def test_genuine_public_iso_sampling_preserves_stations_and_source():
    support = NurbsSurface(3, 1, (0.,)*4+(1.,)*4, (0., 0., 1., 1.),
        tuple(((x, 0., z), (x, 1., z)) for x, z in
              zip((0., .25, .75, 1.), (0., 1., -1., 0.))))
    model = GeometryModel()
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = model.add_points(tuple(tuple(support.evaluate(*p)) for p in uv))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1)%4])))) for i, p in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    regions = query_material_surface_regions(model, [face])
    before = to_dict(model)
    result = query_prepared_material_boundary_sampling(model, regions,
        ((model.handle('edge', edges[0]), (1., .5, 0.)),), chord_error=.02)
    validate_prepared_material_boundary_sampling_binding(model, regions, result)
    stations = result.edge_parameters[0][1]
    assert stations[0] == 1 and stations[-1] == 0 and Q(1, 2) in stations
    assert len(stations) > 3 and all(a > b for a, b in zip(stations, stations[1:]))
    assert all(row.world_error <= .02 for row in result.chords.intervals)
    assert to_dict(model) == before


def test_same_endpoint_poison_cannot_falsely_refine_under_threshold_iso(monkeypatch):
    from anygeometry import nurbs
    support = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
        tuple(((x, 0., z), (x, 1., z)) for x, z in
              ((0., 0.), (.5, .0048), (1., 0.))))
    curve = CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 0.))))
    degree, knots, controls, sigma, slack, domain = curve._iso_world(fresh=True)
    # Same endpoints, but the hostile middle control invents a large bow.
    altered = (controls[0], (Q(1, 2), Q(0), Q(1), Q(1)), controls[2])
    poison = (degree, knots, altered, sigma, slack, domain)
    monkeypatch.setattr(nurbs, '_ISO_CACHE', {curve: poison})
    bound, coverage = _world_chord(curve, Q(0), Q(1),
        ((0., 0., 0.), (1., 0., 0.)), .0025, lambda: None)
    assert bound <= .0025 and len(coverage) > 1
    assert sum(b-a for a, b in coverage) == 1
    assert nurbs._ISO_CACHE[curve] is poison


def test_same_endpoint_poison_cannot_certify_noniso_actual_violation(monkeypatch):
    from anygeometry import nurbs
    knots = (0., 0., 0., 1., 1., 1.)
    support = NurbsSurface(2, 1, knots, (0., 0., 1., 1.),
        tuple(((x, 0., z), (x, 1., z)) for x, z in ((0., 0.), (.5, 2.), (1., 0.))))
    curve = CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 1.))))
    assert curve._iso_world(fresh=True) is None
    # A genuine different support provides the native-format fake straight
    # iso representation, with exactly the actual non-iso curve's endpoints.
    straight_support = NurbsSurface(2, 1, knots, (0., 0., 1., 1.),
        tuple(((x, x, 0.), (x, x+1., 0.)) for x in (0., .5, 1.)))
    straight = CurveOnSurface(straight_support,
        NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 0.))))
    poison = straight._iso_world(fresh=True)
    monkeypatch.setattr(nurbs, '_ISO_CACHE', {curve: poison})
    assert registered_world_hull(curve, 0., 1.) is None
    with pytest.raises(_WorldChordNeedsRefinement) as caught:
        _world_chord(curve, Q(0), Q(1),
            ((0., 0., 0.), (1., 1., 0.)), .0025, lambda: None)
    assert caught.value.parameter == Q(1, 2)
    assert nurbs._ISO_CACHE[curve] is poison
