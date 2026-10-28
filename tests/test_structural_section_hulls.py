"""Actual implicit section evaluator remains inside a structural LENGTH tube."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryError, GeometryModel, Plane, plan_plane_split, apply_plane_split
from anygeometry import query_material_surface_regions, query_prepared_material_boundary_sampling
from anygeometry import validate_prepared_material_boundary_sampling_binding
from anygeometry.nurbs import NurbsCurve, NurbsSurface
from anygeometry.entities import OrientedEdge
from anygeometry.surfaces import CoonsSurface
from anygeometry.support_curves import PlaneNurbsCurve
from anygeometry.structural_section_hulls import structural_section_hull
from anygeometry.registered_world_hulls import fixed_chord_hull_bound
from anygeometry.prepared_material_curvilinear import _world_chord
from anygeometry.serialization import to_dict, from_dict
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work


def _parabolic_section(knots=(0., 1.)):
    lo, hi = knots
    support = NurbsSurface(2, 1, (lo, lo, lo, hi, hi, hi), (0., 0., 1., 1.),
        (((0., 0., 0.), (4., 0., 0.)), ((0., .5, 0.), (4., .5, 0.)),
         ((0., 1., 1.), (4., 1., 1.))), u_range=knots)
    plane = Plane((2., 0., 0.), (-.8, .6, 0.), (0., 0., 1.))
    return support, plane, PlaneNurbsCurve.from_plane(plane, support)


def test_review_ill_conditioned_straight_section_preserves_native_chord_admission():
    eps = 2.**-44
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (1., 0., 0.)), ((0., eps/2, 1.), (1., eps/2, 1.))))
    # Exact normal=(eps,1,0), plane eps*x+y=3eps/4.
    plane = Plane((.75, 0., 0.), (-1., eps, 0.), (0., 0., 1.))
    curve = PlaneNurbsCurve.from_plane(plane, support)
    arcs = curve._arcs()
    assert all(arc.slicing == 0 and arc.first.axis == arc.second.axis == 0 for arc in arcs)
    hull = structural_section_hull(curve, 0., 1.)
    assert hull is not None
    assert hull[1][0][1] < Q(.0025)
    ends = ((.75, 0., 0.), (.25, eps/2, 1.))
    bound, coverage = _world_chord(curve, Q(0), Q(1), ends, .0025, lambda: None)
    assert bound <= .0025 and coverage == ((Q(0), Q(1)),)


def test_effective_initial_bisection_law_is_checked_and_callback_pinned(monkeypatch):
    _support, _plane, curve = _parabolic_section()
    monkeypatch.setattr(PlaneNurbsCurve, '_INITIAL_ROOT_BISECTIONS', 0)
    assert structural_section_hull(curve, 0., 1.) is None
    monkeypatch.setattr(PlaneNurbsCurve, '_INITIAL_ROOT_BISECTIONS', 60)
    assert structural_section_hull(curve, 0., 1.) is not None
    def mutate_law():
        monkeypatch.setattr(PlaneNurbsCurve, '_INITIAL_ROOT_BISECTIONS', 0)
    with pytest.raises(GeometryError, match='evaluator law changed'):
        _world_chord(curve, Q(0), Q(1), ((0.,)*3, (1.,)*3), .0025, mutate_law)


def test_initial_noncanonical_section_law_refuses_before_native_bounds(monkeypatch):
    _support, _plane, curve = _parabolic_section()
    called = []
    def forbidden_bounds(self, *args, **kwargs):
        called.append(True)
        raise AssertionError('uncertified evaluator reached native bounds')
    monkeypatch.setattr(PlaneNurbsCurve, 'bounds', forbidden_bounds)
    monkeypatch.setattr(PlaneNurbsCurve, '_INITIAL_ROOT_BISECTIONS', 0)
    with pytest.raises(GeometryError, match='evaluator law is not certified'):
        _world_chord(curve, Q(0), Q(1), ((0.,)*3, (1.,)*3), .0025, lambda: None)
    assert not called


@pytest.mark.parametrize('knots', ((0., 1.), (2., 6.)))
@pytest.mark.parametrize('domain', ((.125, .875), (.875, .125)))
def test_independent_analytic_position_tube_crop_reverse_and_shifted_knots(knots, domain):
    _support, _plane, full = _parabolic_section(knots)
    curve = replace(full, domain=domain)
    controls, tube = structural_section_hull(curve, 0., 1.)
    assert controls and all(type(x) is Q for point in controls for x in point)
    ratio = Q(.8)/Q(.6)
    for t in (0., .125, .5, .875, 1.):
        u = Q(domain[0])+Q(t)*(Q(domain[1])-Q(domain[0]))
        ideal = (Q(2)-ratio*u, u, u*u)
        actual = tuple(map(Q, curve.evaluate(t)))
        assert all(lo <= x-y <= hi for x, y, (lo, hi) in zip(actual, ideal, tube))
    assert structural_section_hull(curve, .5, .5) is not None


def test_structural_hull_covers_multiple_profile_spans_and_root_branches():
    support = NurbsSurface(2, 1, (0., 0., 0., .5, 1., 1., 1.), (0., 0., 1., 1.),
        tuple(((0., y, z), (4., y, z)) for y, z in
              ((0., 0.), (.25, .5), (.75, .5), (1., 0.))))
    plane = Plane((2., 0., 0.), (-.8, .6, 0.), (0., 0., 1.))
    curve = PlaneNurbsCurve.from_plane(plane, support)
    controls, tube = structural_section_hull(curve, 0., 1.)
    assert len(controls) >= 6
    for t in (0., .125, .25, .5, .75, .875, 1.):
        point = tuple(map(Q, curve.evaluate(t)))
        assert all(min(p[i] for p in controls)+tube[i][0] <= point[i]
                   <= max(p[i] for p in controls)+tube[i][1] for i in range(3))
    ends = tuple(tuple(map(Q, curve.evaluate(t))) for t in (0., 1.))
    bound = fixed_chord_hull_bound(curve, 0., 1., ends)
    from anygeometry.nurbs_domain import _exact_segment_distance
    assert all(_exact_segment_distance(tuple(map(Q, curve.evaluate(t))), *ends) <= bound
               for t in (0., .25, .5, .75, 1.))


def test_literal_endpoint_correction_is_still_guarded():
    _support, _plane, curve = _parabolic_section()
    ends = tuple(tuple(curve.evaluate(t)) for t in (0., 1.))
    # True endpoint correction is represented by existing END_ERROR, not
    # silently zeroed by the structural-image correspondence.
    changed = (tuple(x+1. for x in ends[0]), tuple(x+1. for x in ends[1]))
    with pytest.raises(GeometryError):
        _world_chord(curve, Q(0), Q(1), changed, .0025, lambda: None)


@pytest.mark.parametrize('family', ('nonuniform', 'nontranslation', 'other_axis', 'near_tangent'))
def test_unregistered_section_protocols_keep_native_fallback(family):
    support, plane, curve = _parabolic_section()
    if family == 'nonuniform':
        support = replace(support, weights=((1., 1.), (2., 2.), (1., 1.)))
    elif family == 'nontranslation':
        support = replace(support, controls=(support.controls[0],
            (support.controls[1][0], (4.1, .5, 0.)), support.controls[2]))
    elif family == 'other_axis':
        plane = Plane((0., .5, 0.), (1., 0., 0.), (0., 0., 1.))
    else:
        plane = Plane((2., .5, 0.), (-1., 2.**-20, 0.), (0., 0., 1.))
    curve = PlaneNurbsCurve.from_plane(plane, support)
    assert structural_section_hull(curve, 0., 1.) is None
    assert curve.bounds() is not None  # no native fallback has been disabled


def test_section_hull_zero_work_cancel_runtime_and_detached_mutation(monkeypatch):
    support, _plane, curve = _parabolic_section()
    token = CURRENT_WORK.set(Work(0, 0))
    try:
        with pytest.raises(GeometryError):
            structural_section_hull(curve, 0., 1.)
    finally:
        CURRENT_WORK.reset(token)
    marker = RuntimeError('actual caller cancellation')
    def cancel():
        raise marker
    with pytest.raises(RuntimeError) as caught:
        _world_chord(curve, Q(0), Q(1), ((0.,)*3, (1.,)*3), .0025, cancel)
    assert caught.value is marker
    arc = curve._arcs()[0]
    original = arc.s0
    try:
        object.__setattr__(arc, 's0', (arc.s0+arc.s1)/2)
        with pytest.raises(GeometryError, match='runtime'):
            structural_section_hull(curve, 0., 1.)
    finally:
        object.__setattr__(arc, 's0', original)
    def change():
        object.__setattr__(curve, 'surface', replace(support, controls=tuple(
            tuple(tuple(x+.125 for x in p) for p in row) for row in support.controls)))
    try:
        with pytest.raises(GeometryError, match='changed'):
            _world_chord(curve, Q(0), Q(1), ((0.,)*3, (1.,)*3), .0025, change)
    finally:
        object.__setattr__(curve, 'surface', support)


@pytest.mark.parametrize('reload', (False, True))
def test_genuine_public_curved_sweep_section_sampling(reload):
    model = GeometryModel()
    profile = NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        ((0., 0., 0.), (0., .5, 1.), (0., 1., 0.)))
    opposite = replace(profile, controls=tuple((x+4., y, z) for x, y, z in profile.controls))
    a, b, c, d = model.add_points(((0., 0., 0.), (0., 1., 0.), (4., 1., 0.), (4., 0., 0.)))
    edges = (model.add_curve(a, b, profile), model.add_line(b, c),
             model.add_curve(d, c, opposite), model.add_line(d, a))
    face = model.add_face_from_loop(tuple(OrientedEdge(e, f) for e, f in
        zip(edges, (True, True, False, True))), (0, 1, 2, 3), surface=CoonsSurface())
    plane = Plane((2., 0., 0.), (-.8, .6, 0.), (0., 0., 1.))
    cut = apply_plane_split(model, plan_plane_split(model, (face,), plane), policy='imprint')
    assert len(cut.trace_edges) == 1
    edge = cut.trace_edges[0]
    if reload:
        model = from_dict(to_dict(model))
    assert structural_section_hull(model.edges[edge].curve, 0., 1.) is not None
    child = min(model.faces_using_edge(edge))
    regions = query_material_surface_regions(model, [child])
    before = to_dict(model)
    result = query_prepared_material_boundary_sampling(model, regions,
        ((model.handle('edge', edge), (1., .5, 0.)),), chord_error=.01)
    validate_prepared_material_boundary_sampling_binding(model, regions, result)
    stations = result.edge_parameters[0][1]
    assert stations[0] == 1 and stations[-1] == 0 and Q(1, 2) in stations and len(stations) > 3
    assert all(a > b for a, b in zip(stations, stations[1:]))
    assert all(row.world_error <= .01 for row in result.chords.intervals)
    assert to_dict(model) == before
