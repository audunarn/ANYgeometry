"""Public bounded section owner regressions; no display geometry authority."""
import pytest
from copy import copy
from dataclasses import replace
from anygeometry import (GeometryError, plan_intersections, apply_intersections,
                         query_intersection, plan_imprint, apply_imprint)
from anygeometry.serialization import to_dict, from_dict
from test_plane_split import bowl


def fixture(size=2.):
    model, wall, plane = bowl()
    points = [(-size, -size, .5), (size, -size, .5),
              (size, size, .5), (-size, size, .5)]
    vertices = [model.add_point(*point) for point in points]
    edges = [model.add_line(vertices[i], vertices[(i+1)%4]) for i in range(4)]
    plate = model.add_face(edges, surface=plane)
    return model, wall, plate


def test_public_closed_bounded_imprint_retains_both_materials_and_world_identity():
    model, wall, plate = fixture()
    before = to_dict(model)
    query = query_intersection(model, ('face', wall), ('face', plate))
    assert query.classified and query.components
    assert len(query.components) == 1
    assert query.components[0].curve_traces[0].closed
    assert to_dict(model) == before
    plan = plan_imprint(model, ('face', wall), ('face', plate), policy='imprint')
    assert to_dict(model) == before
    apply_imprint(model, plan, policy='imprint')
    assert len(model.faces) == 4
    traces = model._intersection_application_receipt[3]
    assert traces
    assert all(len(model.faces_using_edge(edge)) == 4 for edge in traces)
    assert not model.validate_topology()
    loaded = from_dict(to_dict(model))
    assert not loaded.validate_topology()


def test_bounded_batch_plan_is_source_bound_and_rejects_raw_edits():
    model, wall, plate = fixture()
    plan = plan_intersections(model, [('face', wall), ('face', plate)], policy='imprint')
    model.faces[plate].metadata['raw-edit'] = True
    before = to_dict(model)
    with pytest.raises(GeometryError, match='stale|changed'):
        apply_intersections(model, plan, policy='imprint')
    assert to_dict(model) == before


def test_copied_bounded_plan_cannot_enter_generic_empty_arrangement_route():
    model, wall, plate = fixture()
    plan = plan_intersections(model, [('face', wall), ('face', plate)], policy='imprint')
    before = to_dict(model)
    with pytest.raises(GeometryError, match='copied|changed|another model'):
        apply_intersections(model, copy(plan), policy='imprint')
    assert to_dict(model) == before


def test_plate_boundary_crossing_refuses_without_mutation():
    model, wall, plate = fixture(.5)
    before = to_dict(model)
    with pytest.raises(GeometryError):
        plan_intersections(model, [('face', wall), ('face', plate)], policy='imprint')
    assert to_dict(model) == before


def test_direct_imprint_plan_has_genuine_epoch_native_binding():
    from anygeometry.preparation_epochs import _issue_epoch, _native_plan_supports
    from anygeometry.native_support_snapshots import capture_native_supports
    model, wall, plate = fixture()
    _issue_epoch(model)
    plan = plan_imprint(model, ('face', wall), ('face', plate), policy='imprint')
    assert _native_plan_supports(plan.batch_plan) == capture_native_supports(model)
    apply_imprint(model, plan, policy='imprint')
    assert len(model.faces) == 4


def test_curved_plate_outer_boundary_retains_two_disjoint_authored_holes():
    from anygeometry.entities import OrientedEdge
    from anygeometry.surfaces import Plane
    model, wall, plate = fixture()
    holes = []
    for centre in (-.25, .25):
        points = [(centre-.05, -.05, .5), (centre-.05, .05, .5),
                  (centre+.05, .05, .5), (centre+.05, -.05, .5)]
        vertices = [model.add_point(*point) for point in points]
        edges = [model.add_line(vertices[i], vertices[(i+1)%4]) for i in range(4)]
        holes.append(tuple(OrientedEdge(edge, True) for edge in edges))
    with model.transaction():
        model._put_entity('face', replace(model.faces[plate], holes=tuple(holes)))
    before = to_dict(model)
    plan = plan_intersections(model, [('face', wall), ('face', plate)], policy='imprint')
    assert to_dict(model) == before
    apply_intersections(model, plan, policy='imprint')
    plane_faces = [face for face in model.faces.values() if isinstance(face.surface, Plane)]
    assert len(plane_faces) == 2 and sorted(len(face.holes) for face in plane_faces) == [1, 2]
    assert not from_dict(to_dict(model)).validate_topology()


@pytest.mark.parametrize('vector,rotation,reverse', [((-5., 0., 0.), 0, False), ((3., 4., 0.), 1, True)])
def test_public_query_uses_original_authored_coons_chart(vector, rotation, reverse):
    import numpy as np
    from anygeometry import GeometryModel, Plane
    from anygeometry.entities import OrientedEdge
    from anygeometry.nurbs import NurbsCurve
    from anygeometry.surfaces import CoonsSurface
    model = GeometryModel()
    controls = ((0., 0., 0.), (0., .5, 1.), (0., 1., 0.))
    if reverse:
        controls = controls[::-1]
    profile = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), controls)
    top = replace(profile, controls=tuple(tuple(a+b for a, b in zip(p, vector)) for p in profile.controls))
    a, b, c, d = model.add_points((profile.evaluate(0.), profile.evaluate(1.), top.evaluate(1.), top.evaluate(0.)))
    edges = (model.add_curve(a, b, profile), model.add_line(b, c),
             model.add_curve(d, c, top), model.add_line(d, a))
    uses = tuple(OrientedEdge(edge, forward) for edge, forward in zip(edges, (True, True, False, True)))
    uses = uses[rotation:]+uses[:rotation]
    wall = model.add_face_from_loop(uses, (0, 1, 2, 3), surface=CoonsSurface())
    x = vector[0]*.2
    plane = Plane((x, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    points = [model.add_point(x, y, z) for y, z in ((-3., -3.), (6., -3.), (6., 3.), (-3., 3.))]
    plate = model.add_face([model.add_line(points[i], points[(i+1)%4]) for i in range(4)], surface=plane)
    before = to_dict(model)
    result = query_intersection(model, ('face', wall), ('face', plate))
    assert result.classified and result.components
    assert to_dict(model) == before and model.faces[wall].parameterization is None
    for component in result.components:
        uv = component.first_parameter_path
        for point, (u, v) in zip(component.witnesses, uv):
            assert np.allclose(model.face_point(wall, u, v), point, rtol=0., atol=1e-9)
        if rotation == 0:
            assert all(abs(v-.2) < 1e-9 for u, v in uv)


def test_profile_pair_qualification_charges_hulls_and_honours_depth():
    from anygeometry.nurbs import NurbsCurve
    from anygeometry.nurbs_audit import _certify_pieces
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    first = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 1.)))
    second = NurbsCurve(1, (0., 0., 1., 1.), ((0., 1.), (1., 0.)))
    calls = []
    def hull(curve, lo, hi):
        calls.append((lo, hi))
        return curve.exact_hull(lo, hi)
    work = Work(100, 1)
    token = CURRENT_WORK.set(work)
    try:
        assert _certify_pieces(first, (0., 1.), second, (0., 1.), (), 4000,
                              _hull=hull, _qualification=True) is None
    finally:
        CURRENT_WORK.reset(token)
    assert work.boxes == len(calls) and work.subdivisions == 2
    assert min(hi-lo for lo, hi in calls) >= .5


def test_low_qualification_limits_refuse_bounded_imprint_without_source_mutation():
    from anygeometry.predicates import IntersectionQualificationPolicy
    model, wall, plate = fixture()
    before = to_dict(model)
    with pytest.raises(GeometryError):
        plan_imprint(model, ('face', wall), ('face', plate), policy='imprint',
                     qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1, max_subdivision_depth=1))
    assert to_dict(model) == before
