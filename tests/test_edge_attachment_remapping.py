"""Exact split coordinates for persistent vertex, member and sheet relations."""
import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, from_dict, to_dict
from anygeometry.structural import ParameterRange, JunctionMemberUse


def point_relations(model, edge, parameter):
    point = model.sample_edge(edge, np.asarray([parameter]))[0]
    vertex = model.add_point(*point)
    first = model.add_attachment(None, 'vertex_on_edge', 'edge', edge, ParameterRange(0., 0.),
        (ParameterRange.point(parameter),), source_kind='vertex', source_id=vertex,
        evidence='exact', tolerance_used=1e-9, metadata={'purpose': 'point-load'})
    other = model.add_point(*(point + (0., 0., .2)))
    member = model.add_member((model.add_line(vertex, other),))
    second = model.add_attachment(member, 'member_on_boundary', 'edge', edge,
        ParameterRange.point(0.), (ParameterRange.point(parameter),), evidence='exact', tolerance_used=1e-9)
    return point, (first, second)


@pytest.mark.parametrize('station', [.2, .4, .8])
def test_point_relations_retain_coordinates_and_direct_split_stays_explicit(station):
    model = GeometryModel()
    edge = model.add_line(*model.add_points(((0., 0., 0.), (4., 0., 0.))))
    expected, identifiers = point_relations(model, edge, station)
    before = to_dict(model)
    with pytest.raises(GeometryError, match='explicit parameter remap'):
        model.split_edge(edge, .4)
    assert to_dict(model) == before
    _vertex, children = model.split_edge(edge, .4, remap_attachments=True)
    assert len(children) == 2
    assert model.edges[children[0]].end == _vertex
    assert model.edges[children[1]].start == _vertex
    for identifier in identifiers:
        attachment = model.attachments[identifier]
        assert attachment.target_id == children[0 if station <= .4 else 1]
        assert ('edge', edge) in attachment.lineage
        np.testing.assert_allclose(model.sample_edge(attachment.target_id,
            np.asarray([attachment.target_parameters[0].start]))[0], expected, atol=1e-12)
    assert model.validate_topology() == ()
    assert to_dict(from_dict(to_dict(model))) == to_dict(model)


def test_member_interval_and_junction_expand_at_a_split():
    model = GeometryModel()
    edge = model.add_line(*model.add_points(((0., 0., 0.), (4., 0., 0.))))
    member = model.add_member((edge,))
    attachment = model.add_attachment(member, 'member_on_boundary', 'edge', edge,
        ParameterRange(.1, .9), (ParameterRange(.1, .9),), evidence='exact', tolerance_used=1e-9)
    other = model.add_member((edge,))
    second = model.add_attachment(other, 'member_on_boundary', 'edge', edge,
        ParameterRange(.1, .9), (ParameterRange(.1, .9),), evidence='exact', tolerance_used=1e-9)
    junction = model.ensure_junction('overlap', (JunctionMemberUse(member, ParameterRange(.1, .9)),
        JunctionMemberUse(other, ParameterRange(.1, .9))), attachment_ids=(attachment, second))
    model.split_edge(edge, .4, remap_attachments=True)
    assert len(model.junctions[junction].attachment_ids) == 4
    related = [model.attachments[i] for i in model.junctions[junction].attachment_ids
               if model.attachments[i].member_id == member]
    assert len(related) == 2
    assert [a.member_range for a in related] == [ParameterRange(.1, .4), ParameterRange(.4, .9)]
    assert model.validate_topology() == ()


def regularized_branch():
    from anygeometry.branch_curves import BezierQuadricCurve
    from anygeometry.branch_algebra import BezierRuledSupport
    from anygeometry.quadric_algebra import QuadricSupport
    wall = BezierRuledSupport(((0., 0., 0.), (1., 0., 0.), (2., 1., 0.), (3., 3., 0.)), (0., 0., 1.))
    quadric = QuadricSupport('general', matrix=(0., 0., 0., 0., 0., 0., 0., 0., 1.),
                             linear=(-.5, 0., 0.), constant=0.)
    curve = BezierQuadricCurve(wall, quadric, 0., 1., parameterization='left_square')
    model = GeometryModel()
    edge = model.add_curve(*model.add_points(curve.evaluate(np.array([0., 1.]))), curve)
    return model, edge


def test_regularized_branch_attachment_uses_child_chart_not_linear_fraction():
    model, edge = regularized_branch()
    expected, identifiers = point_relations(model, edge, .8)
    _vertex, children = model.split_edge(edge, .4, remap_attachments=True)
    for identifier in identifiers:
        attachment = model.attachments[identifier]
        parameter = attachment.target_parameters[0].start
        assert attachment.target_id == children[1]
        assert abs(parameter - (.8 - .4) / .6) > .01
        np.testing.assert_allclose(model.sample_edge(children[1], np.asarray([parameter]))[0],
                                   expected, atol=1e-12)
    assert model.validate_topology() == ()


@pytest.mark.parametrize('target_edge_relation', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_regularized_split_preserves_all_owning_member_references(target_edge_relation, reverse):
    from anygeometry.member_joints import _member_point
    model, edge = regularized_branch()
    owner = model.add_member((edge,))
    if reverse:
        model.reverse_member(owner)
    parent = .2 if reverse else .8
    point = model.sample_edge(edge, np.array([.8]))[0]
    a, b = model.add_points((point, point + (0., 0., .5)))
    other_edge = model.add_line(a, b)
    other = model.add_member((other_edge,))
    outgoing = model.add_attachment(owner, 'member_on_boundary', 'edge', other_edge,
        ParameterRange.point(parent), (ParameterRange.point(0.),), evidence='exact', tolerance_used=1e-9)
    incoming = model.add_attachment(other, 'member_endpoint_on_member', 'member', owner,
        ParameterRange.point(0.), (ParameterRange.point(parent),), evidence='exact', tolerance_used=1e-9)
    junction = model.ensure_junction('crossing', (JunctionMemberUse(owner, ParameterRange.point(parent)),
        JunctionMemberUse(other, ParameterRange.point(0.))), attachment_ids=(outgoing, incoming))
    if target_edge_relation:
        point_relations(model, edge, .8)
    _vertex, children = model.split_edge(edge, .4, remap_attachments=True)
    # A second operation must use the current descendant chart and preserve
    # references even when a member now consists of several oriented uses.
    model.split_edge(children[1], .3, remap_attachments=True)
    stations = (model.attachments[outgoing].member_range.start,
                model.attachments[incoming].target_parameters[0].start,
                model.junctions[junction].member_uses[0].member_range.start)
    expected = .4 + .6 * ((.8**2 - .4**2) / (1 - .4**2))
    if reverse:
        expected = 1 - expected
    for station in stations:
        assert station == pytest.approx(expected, abs=1e-14)
        np.testing.assert_allclose(_member_point(model, owner, station), point, atol=1e-12)
    assert model.validate_topology() == ()


def test_batch_multiple_splits_use_current_regularized_descendant_parameters():
    from anygeometry import plan_intersections, apply_intersections
    model, edge = regularized_branch()
    owner = model.add_member((edge,))
    original = model.edges[edge].curve
    faces = [model.add_plate(model.add_points(((x, -1., -1.), (x, 4., -1.),
                                              (x, 4., 3.), (x, -1., 3.))))
             for x in (3 * .4**2, 3 * .8**2)]
    operands = [model.handle('member', owner), *[model.handle('face', f) for f in faces]]
    plan = plan_intersections(model, operands, policy='connect')
    apply_intersections(model, plan, policy='connect')
    uses = [model.member_edge_uses[i] for i in model.members[owner].edge_use_ids]
    assert len(uses) == 3
    for use, interval in zip(uses, ((0., .4), (.4, .8), (.8, 1.))):
        points = model.sample_edge(use.edge_id, np.array([0., 1.]))
        np.testing.assert_allclose(points, original.evaluate(np.array(interval)), atol=1e-12)
    assert model.validate_topology() == ()


def test_failed_explicit_remap_restores_document_and_identifiers(monkeypatch):
    from anygeometry import edge_attachment_remapping
    model = GeometryModel()
    edge = model.add_line(*model.add_points(((0., 0., 0.), (4., 0., 0.))))
    point_relations(model, edge, .8)
    before = to_dict(model)
    def fail(*args):
        raise GeometryError('injected after split')
    monkeypatch.setattr(edge_attachment_remapping, '_one_parameter', fail)
    with pytest.raises(GeometryError, match='injected after split'):
        model.split_edge(edge, .4, remap_attachments=True)
    assert to_dict(model) == before
