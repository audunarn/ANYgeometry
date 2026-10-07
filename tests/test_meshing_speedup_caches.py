"""Focused regression tests for the mesh-time-20261007 cache contracts.

``plan_independent_components`` must certify each unique dependency bound once
per invocation while cached failures still name every affected unit, and its
consumer/dependency closure must expand each (consumer, dependency) pair once
without conflating preexisting owner buckets with already expanded pairs.
``remap_face_attachments`` must analyze each descendant face once per
invocation, with no cache surviving the call.
"""
from dataclasses import replace

import pytest

from anygeometry import (
    GeometryModel, plan_independent_components, validate_component_partition_binding,
    plan_intersections, apply_intersections,
)
from anygeometry import attachment_remapping, component_partition
from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange


def plate(model, x=0, y=0, z=0):
    return model.add_plate(model.add_points(((x,y,z),(x+2,y,z),(x+2,y+1,z),(x,y+1,z))))


def test_unique_dependency_bounds_are_certified_once_per_invocation(monkeypatch):
    model = GeometryModel()
    first = plate(model)
    second = plate(model, 20)
    shared = model.faces[first].loop[0].edge
    members = [model.add_member((shared,)) for _ in range(3)]
    original_bounds = GeometryModel.bounds
    original_face_bound = component_partition._certified_face_bound
    edge_calls = []
    face_calls = []
    def counting_bounds(self, keys):
        edge_calls.append(tuple(keys))
        return original_bounds(self, keys)
    def counting_face_bound(model, face_id, edges):
        face_calls.append(face_id)
        return original_face_bound(model, face_id, edges)
    monkeypatch.setattr(GeometryModel, "bounds", counting_bounds)
    monkeypatch.setattr(component_partition, "_certified_face_bound", counting_face_bound)
    plan = plan_independent_components(model, separation=.1)
    assert plan.certified and len(plan.components) == 2
    assert sorted(face_calls) == sorted((first, second))
    # The shared edge is a dependency of four units but is certified once.
    singles = [call[0] for call in edge_calls if len(call) == 1]
    assert len(singles) == 8 and len(set(singles)) == 8
    grouped_calls = [call for call in edge_calls if len(call) != 1]
    assert len(grouped_calls) == 2 and all(len(call) == 4 for call in grouped_calls)
    grouped = [component.member_ids for component in plan.components]
    assert sorted(grouped) == [(), tuple(sorted(members))]
    validate_component_partition_binding(model, plan)


def test_cached_bound_failure_still_names_every_affected_unit():
    model = GeometryModel()
    points = model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0)))
    unsupported = model.add_face(model.add_polyline(points, close=True))
    axis = model.add_line(*model.add_points(((5,0,0),(6,0,0))))
    members = [model.add_member((axis,), orientation_reference=("face", unsupported))
               for _ in range(2)]
    plan = plan_independent_components(model)
    assert not plan.certified and not plan.components
    affected = [refusal for refusal in plan.refusals if "support family" in refusal.reason]
    assert len(affected) == 3
    assert {entity for refusal in affected for entity in refusal.entities} == \
        {model.handle("face", unsupported),
         model.handle("member", members[0]),
         model.handle("member", members[1])}
    assert len({refusal.reason for refusal in affected}) == 1


def test_orientation_reference_expands_control_vertices_despite_prior_owners():
    model = GeometryModel()
    near = plate(model)
    points = model.add_points(((10,0,0),(1,1,0),(12,0,0)))
    curved = model.add_spline(points[0], (points[1],), points[2])
    axis = model.add_line(*model.add_points(((20,0,0),(21,0,0))))
    member = model.add_member((axis,), orientation_reference=("edge", curved))
    plan = plan_independent_components(model)
    # The spline curve box stays clear of the plate; only its control vertex
    # reaches it, so one component proves the vertex closure was expanded.
    assert plan.certified and len(plan.components) == 1
    assert any("bounds overlap" in reason.reason for reason in plan.merge_reasons)
    validate_component_partition_binding(model, plan)


def test_mutual_attachment_lineage_expands_each_pair_once():
    model = GeometryModel()
    first = plate(model)
    second = plate(model, 20)
    edge = model.faces[first].loop[0].edge
    member = model.add_member((edge,))
    one = model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY,
        AttachmentTargetKind.EDGE, edge, ParameterRange(0,1), (ParameterRange(0,1),),
        lineage=(("face", first),))
    other_edge = model.faces[second].loop[0].edge
    other_member = model.add_member((other_edge,))
    two = model.add_attachment(other_member, AttachmentKind.MEMBER_ON_BOUNDARY,
        AttachmentTargetKind.EDGE, other_edge, ParameterRange(0,1), (ParameterRange(0,1),),
        lineage=(("face", second), ("attachment", one)))
    model._put_structural("attachment", replace(model.attachments[one],
        lineage=(("face", first), ("attachment", two))))
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    assert any("attachment" in reason.reason for reason in plan.merge_reasons)
    validate_component_partition_binding(model, plan)


def test_remap_analyzes_each_descendant_once_per_invocation(monkeypatch):
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    axis = model.add_member((model.add_line(*model.add_points(((0,2,0),(4,2,0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0,1), (ParameterRange(0,1), ParameterRange(.5,.5)),
        evidence='exact', tolerance_used=1e-8)
    point = model.add_member((model.add_line(*model.add_points(((0.5,1,-1),(0.5,1,1)))),))
    model.add_attachment(point, AttachmentKind.MEMBER_THROUGH_FACE, 'face', face,
        ParameterRange(.5,.5), (ParameterRange(.125,.125), ParameterRange(.25,.25)),
        evidence='exact', tolerance_used=1e-8)
    original = attachment_remapping.MaterialDomain.from_model
    analyzed = []
    class CountingDomains:
        @staticmethod
        def from_model(model, face_id):
            analyzed.append(face_id)
            return original(model, face_id)
    monkeypatch.setattr(attachment_remapping, "MaterialDomain", CountingDomains)
    cut = model.add_plate(model.add_points(((2,0,-1),(2,4,-1),(2,4,1),(2,0,1))))
    apply_intersections(model, plan_intersections(model, (face, cut), policy='connect'),
                        policy='connect')
    children = sorted({item.target_id for item in model.attachments.values()
                      if ('face', face) in item.lineage and item.member_id == axis})
    assert len(children) == 2
    assert analyzed == children
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values() if item.member_id == axis)
    assert fragments == [(0.0, 0.5), (0.5, 1.0)]
    retained = [item for item in model.attachments.values() if item.member_id == point]
    assert len(retained) == 1
    assert model.face_point(retained[0].target_id,
        *(r.start for r in retained[0].target_parameters)) == pytest.approx((0.5, 1, 0))
    assert model.validate_topology() == ()
    # A later split recomputes its own descendant domains: no persistent cache.
    before = len(analyzed)
    left = min(children)
    second_cut = model.add_plate(model.add_points(((1,0,-1),(1,4,-1),(1,4,1),(1,0,1))))
    apply_intersections(model, plan_intersections(model, (left, second_cut), policy='connect'),
                        policy='connect')
    grandchildren = sorted({item.target_id for item in model.attachments.values()
                            if ('face', left) in item.lineage and item.member_id == axis})
    assert len(grandchildren) == 2
    assert analyzed[before:] == grandchildren
    assert model.validate_topology() == ()
