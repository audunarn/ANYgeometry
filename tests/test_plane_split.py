import pytest
from dataclasses import replace
from anygeometry import GeometryModel, GeometryError, Plane, plan_plane_split, apply_plane_split
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.serialization import to_dict, from_dict


def bowl(model=None, offset=0.):
    a = (1., -1., 1.)
    surface = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
        tuple(tuple((i-1.+offset, j-1., a[i]+a[j]) for j in range(3)) for i in range(3)))
    model = GeometryModel() if model is None else model
    points = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    vertices = [model.add_point(*surface.evaluate(*p)) for p in points]
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(surface,
                NurbsCurve(1, (0., 0., 1., 1.), (p, points[(i+1)%4]))))
             for i, p in enumerate(points)]
    face = model.add_face(edges, surface=surface)
    return model, face, Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))


def test_closed_split_preserves_source_support_and_shared_trace():
    model, face, plane = bowl()
    before = to_dict(model)
    support = model.faces[face].surface
    plan = plan_plane_split(model, [face], plane)
    assert to_dict(model) == before
    result = apply_plane_split(model, plan, policy='imprint')
    assert len(model.faces) == 2
    assert set(result.positive_faces + result.negative_faces) == set(model.faces)
    assert all(f.surface == support for f in model.faces.values())
    assert all(sum(any(e.edge == trace for loop in (f.loop, *f.holes) for e in loop)
                   for f in model.faces.values()) == 2 for trace in result.trace_edges)
    assert not model.validate_topology()
    loaded = from_dict(to_dict(model))
    assert not loaded.validate_topology()
    with pytest.raises(GeometryError, match='stale'):
        apply_plane_split(model, plan, policy='imprint')


def test_same_revision_edit_refuses_without_further_mutation():
    model, face, plane = bowl()
    plan = plan_plane_split(model, [face], plane)
    model._faces[face] = replace(model.faces[face], metadata={'raw': True})
    before = to_dict(model)
    with pytest.raises(GeometryError, match='stale'):
        apply_plane_split(model, plan, policy='imprint')
    assert to_dict(model) == before


def test_forged_plan_field_refuses():
    model, face, plane = bowl()
    plan = plan_plane_split(model, [face], plane)
    before = to_dict(model)
    object.__setattr__(plan, 'trace_edges', ())
    with pytest.raises(GeometryError, match='fields changed'):
        apply_plane_split(model, plan, policy='imprint')
    assert to_dict(model) == before


def test_two_parents_publish_as_one_revision():
    model, first, plane = bowl()
    _, second, _ = bowl(model, offset=4.)
    before = to_dict(model)
    revision = model.revision
    plan = plan_plane_split(model, [first, second], plane)
    assert to_dict(model) == before
    result = apply_plane_split(model, plan, policy='imprint')
    assert model.revision == revision + 1
    assert result.change_set.revision_before == revision
    assert len(result.parent_children) == 2
    assert len(model.faces) == 4


def test_second_operand_refusal_preserves_source():
    model, first, plane = bowl()
    # A valid planar second operand is outside this intermediate NURBS route.
    second = model.add_plate([model.add_point(x, y, 0.)
                              for x, y in ((4., 0.), (5., 0.), (5., 1.), (4., 1.))])
    before = to_dict(model)
    with pytest.raises(GeometryError, match='translation'):
        plan_plane_split(model, [first, second], plane)
    assert to_dict(model) == before


def test_observer_plan_tampering_cannot_change_application():
    model, face, plane = bowl()
    plan = plan_plane_split(model, [face], plane)
    traces = plan.trace_edges
    model._change_hooks.append(lambda change: object.__setattr__(plan, 'trace_edges', ()))
    result = apply_plane_split(model, plan, policy='imprint')
    assert result.trace_edges == traces


@pytest.mark.parametrize('weights', [None, (1., 4.)])
@pytest.mark.parametrize('reverse', [False, True])
def test_two_open_sections_partition_one_rectangle(weights, reverse):
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
        tuple(((i/2, 0., z), (i/2, 1., z)) for i, z in enumerate((.25, -.25, .25))))
    model = GeometryModel()
    points = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    vertices = [model.add_point(*surface.evaluate(*p)) for p in points]
    edges = []
    for i, p in enumerate(points):
        q = points[(i+1)%4]
        controls = (q, p) if reverse else (p, q)
        pcurve = NurbsCurve(1, (0., 0., 1., 1.), controls, weights,
                           domain=(1., 0.) if reverse else (0., 1.))
        edges.append(model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(surface, pcurve)))
    parent = model.add_face(edges, surface=surface)
    plane = Plane((0., 0., .0625), (1., 0., 0.), (0., 1., 0.))
    before = to_dict(model)
    plan = plan_plane_split(model, [parent], plane)
    assert to_dict(model) == before
    result = apply_plane_split(model, plan, policy='imprint')
    assert len(model.faces) == 3
    assert len(result.trace_edges) >= 2
    assert all(len(model.faces_using_edge(edge)) == 2 for edge in result.trace_edges)
    assert len(result.positive_faces) == 2 and len(result.negative_faces) == 1
    assert all(f.surface == surface for f in model.faces.values())
    assert not model.validate_topology()


@pytest.mark.parametrize('points', [
    [(.65, .4), (.65, .6), (.9, .6), (.9, .4)],
    [(.45, .45), (.45, .55), (.55, .55), (.55, .45)],
])
def test_regular_section_crosses_a_rectangular_chart_hole(points):
    model, parent, plane = bowl()
    support = model.faces[parent].surface
    from anygeometry.entities import OrientedEdge
    with model.transaction():
        vertices = [model.add_point(*support.evaluate(*p)) for p in points]
        edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
                    NurbsCurve(1, (0., 0., 1., 1.), (p, points[(i+1)%4]))))
                 for i, p in enumerate(points)]
        model._put_entity('face', replace(model.faces[parent],
            holes=(tuple(OrientedEdge(edge, True) for edge in edges),)))
    before = to_dict(model)
    plan = plan_plane_split(model, [parent], plane)
    assert to_dict(model) == before
    result = apply_plane_split(model, plan, policy='imprint')
    assert len(result.parent_children[0][1]) == 2
    assert len(result.positive_faces) == len(result.negative_faces) == 1
    assert all(len(model.faces_using_edge(e)) == 2 for e in result.trace_edges)
    assert not model.validate_topology()


def test_second_nonparallel_cut_reuses_original_shared_event_vertices():
    model, parent, plane = bowl()
    first = plan_plane_split(model, [parent], plane)
    apply_plane_split(model, first, policy='imprint')
    vertical = Plane((.2, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    before = to_dict(model)
    second = plan_plane_split(model, tuple(model.faces), vertical)
    assert to_dict(model) == before
    result = apply_plane_split(model, second, policy='imprint')
    assert len(model.faces) == 4
    assert len(result.parent_children) == 2
    assert all(len(model.faces_using_edge(edge)) == 2 for edge in result.trace_edges)
    assert not model.validate_topology()


def test_small_qualification_budget_refuses_without_mutation():
    from anygeometry.predicates import IntersectionQualificationPolicy
    model, parent, plane = bowl()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='budget'):
        plan_plane_split(model, [parent], plane,
            qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1))
    assert to_dict(model) == before

