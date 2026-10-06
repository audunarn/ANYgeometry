"""Geometry-only regression contracts for component worker handoff."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from anygeometry import (
    GeometryModel, GeometryError, ModelClosure, Plane, CoonsSurface, to_dict, from_dict,
    extract_model_closure, plan_independent_components, validate_component_partition_binding,
    query_trimmed_surface_charts, query_trimmed_surface_charts_by_face,
    validate_trimmed_surface_charts_binding, model_closure_from_dict,
)
from anygeometry.definition_binding import definition_checksum
from anygeometry.structural import (
    AttachmentKind, AttachmentTargetKind, ConnectionIntent, ParameterRange,
    JunctionKind, JunctionMemberUse, SheetTopologyPolicy, ConnectivityPolicy,
)


def plate(model, x=0, y=0, z=0):
    return model.add_plate(model.add_points(((x,y,z),(x+2,y,z),(x+2,y+1,z),(x,y+1,z))))


def stiffened(model, x):
    face = plate(model, x)
    sheet = model.add_sheet((face,))
    edge = model.faces[face].loop[0].edge
    member = model.add_member((edge,))
    attachment = model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY,
        AttachmentTargetKind.EDGE, edge, ParameterRange(0,1), (ParameterRange(0,1),),
        sheet_id=sheet, lineage=(("face",face),))
    model.add_junction(JunctionKind.OVERLAP, (JunctionMemberUse(member, ParameterRange(0,1)),),
                       sheet_ids=(sheet,), attachment_ids=(attachment,))
    return face, member


def test_distant_components_transport_complete_maps_and_stable_ids():
    model = GeometryModel()
    with model.transaction():
        first = stiffened(model, 0)
        second = stiffened(model, 20)
    before = to_dict(model)
    plan = plan_independent_components(model, separation=.1)
    assert plan.certified and len(plan.components) == 2
    assert [component.face_ids for component in plan.components] == [(first[0],),(second[0],)]
    validate_component_partition_binding(model, plan)
    kinds = set()
    for component in plan.components:
        closure = extract_model_closure(model, component.handles)
        assert closure.working_model.model_id != model.model_id
        envelope = closure.to_transport()
        assert json.loads(json.dumps(envelope)) == envelope
        restored = ModelClosure.from_transport(pickle.loads(pickle.dumps(envelope)))
        assert to_dict(restored.working_model) == to_dict(closure.working_model)
        assert dict(restored.work_to_source) == dict(closure.work_to_source)
        assert dict(restored.source_to_work) == dict(closure.source_to_work)
        kinds.update(h.kind for h in restored.work_to_source)
        # Decoding owns its arrays; raw edits cannot mutate the original model.
        vertex = next(iter(restored.working_model.vertices.values()))
        vertex.position.flags.writeable = True
        vertex.position[2] = 9
    assert {"vertex","edge","face","sheet","member","attachment","junction",
            "part","face_use","coedge","member_edge_use"} <= kinds
    assert to_dict(model) == before


def test_separation_is_per_box_and_refuses_invalid_distance():
    model = GeometryModel()
    plate(model, 0)
    plate(model, 2.15)
    assert len(plan_independent_components(model, separation=.05).components) == 2
    merged = plan_independent_components(model, separation=.1)
    assert merged.certified and len(merged.components) == 1
    assert any("bounds overlap" in reason.reason for reason in merged.merge_reasons)
    for value in (-1, float("nan"), float("inf"), True, "oops"):
        with pytest.raises(GeometryError, match="separation"):
            plan_independent_components(model, separation=value)


def test_shared_vertex_and_sheet_membership_merge_regardless_distance():
    model = GeometryModel()
    a = plate(model)
    vertex = model.faces[a].loop[0]
    common = model.edges[vertex.edge].start
    b = model.add_plate((common, *model.add_points(((10,0,0),(10,1,0),(0,2,0)))))
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    assert any("shared vertex" in reason.reason for reason in plan.merge_reasons)
    other = GeometryModel()
    a, b = plate(other), plate(other, 20)
    other.add_sheet((a,b), policy=SheetTopologyPolicy(connectivity=ConnectivityPolicy.ALLOW_DISCONNECTED))
    plan = plan_independent_components(other)
    assert plan.certified and len(plan.components) == 1
    partial = plan_independent_components(other, face_ids=(a,), member_ids=())
    assert not partial.certified and not partial.components
    assert any("partial sheet" in refusal.reason for refusal in partial.refusals)


@pytest.mark.parametrize("intent", list(ConnectionIntent))
def test_declared_distant_attachment_closes_all_intents(intent):
    model = GeometryModel()
    face = plate(model)
    edge = model.add_line(*model.add_points(((100,0,0),(101,0,0))))
    member = model.add_member((edge,))
    model.add_attachment(member, AttachmentKind.MEMBER_ON_FACE, AttachmentTargetKind.FACE,
        face, ParameterRange(0,1), (ParameterRange(0,1), ParameterRange(0,1)), connection_intent=intent)
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    partial = plan_independent_components(model, member_ids=())
    assert not partial.certified and not partial.components
    assert any("omits" in refusal.reason for refusal in partial.refusals)


def test_undeclared_crossings_and_unselected_bridge_refuse_or_merge():
    model = GeometryModel()
    a = plate(model)
    model.add_plate(model.add_points(((1,-1,-1),(1,2,-1),(1,2,1),(1,-1,1))))
    assert len(plan_independent_components(model).components) == 1
    partial = plan_independent_components(model, face_ids=(a,))
    assert not partial.certified and not partial.components


def test_extruded_curved_control_hull_prevents_false_separation():
    model = GeometryModel()
    points = model.add_points(((0,0,0),(1,10,0),(2,0,0)))
    curved = model.extrude((model.add_spline(points[0], (points[1],), points[2]),), (0,0,2))[0]
    other = plate(model, 0, 4, 1)
    plan = plan_independent_components(model)
    assert not plan.certified and not plan.components
    assert any("full trim-domain" in refusal.reason for refusal in plan.refusals)


def test_implicit_support_and_eccentric_metadata_fail_closed():
    model = GeometryModel()
    face = model.add_face(model.add_polyline(model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))), close=True))
    plan = plan_independent_components(model)
    assert not plan.certified and not plan.components
    assert any("support family" in refusal.reason for refusal in plan.refusals)
    other = GeometryModel()
    edge = other.add_line(*other.add_points(((0,0,0),(1,0,0))))
    other.add_member((edge,), metadata={"eccentric_offset": (0,0,10)})
    plan = plan_independent_components(other)
    assert not plan.certified and not plan.components


@pytest.mark.parametrize("kind", ("member", "member_edge_use", "attachment"))
def test_unselected_unknown_physical_extent_refuses_nonempty_selection(kind):
    model = GeometryModel()
    face = plate(model)
    axis = model.add_line(*model.add_points(((100,0,0),(101,0,0))))
    member = model.add_member((axis,))
    if kind == "attachment":
        identifier = model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY,
            AttachmentTargetKind.EDGE, axis, ParameterRange(0,1), (ParameterRange(0,1),),
            metadata={"eccentric_offset": (-100,0,0)})
    else:
        identifier = member if kind == "member" else model.members[member].edge_use_ids[0]
        table = model.members if kind == "member" else model.member_edge_uses
        with model.transaction():
            model._put_structural(kind, replace(table[identifier], metadata={"eccentric_offset": (-100,0,0)}))
    plan = plan_independent_components(model, face_ids=(face,), member_ids=())
    assert not plan.certified and not plan.components
    assert any("extent contract" in refusal.reason and
               model.handle(kind, identifier) in refusal.entities for refusal in plan.refusals)
    empty = plan_independent_components(model, face_ids=(), member_ids=())
    assert empty.certified and not empty.components and not empty.refusals


def test_partition_raw_mutation_tampered_evidence_and_callback_guards():
    model = GeometryModel()
    face = plate(model)
    plan = plan_independent_components(model)
    with pytest.raises(GeometryError, match="evidence"):
        validate_component_partition_binding(model, replace(plan, components=()))
    with pytest.raises(GeometryError, match="another model"):
        validate_component_partition_binding(model.clone(), plan)
    source = model.faces[face].surface
    source.origin.flags.writeable = True
    source.origin[0] += .01
    with pytest.raises(GeometryError, match="binding changed"):
        validate_component_partition_binding(model, plan)
    source.origin[0] -= .01
    def changed(_):
        source.origin[0] += .01
        return False
    with pytest.raises(GeometryError, match="changed during"):
        plan_independent_components(model, cancellation_check=changed)
    with pytest.raises(GeometryError, match="cancelled"):
        plan_independent_components(model, cancellation_check=lambda _: True)


def test_transport_rejects_incomplete_duplicate_or_tampered_correspondence():
    model = GeometryModel()
    face = plate(model)
    closure = extract_model_closure(model, (("face",face),))
    data = closure.to_transport()
    changed = deepcopy(data)
    changed["work_to_source"].pop()
    with pytest.raises(GeometryError, match="checksum"):
        model_closure_from_dict(changed)
    changed["transport_checksum"] = definition_checksum({k:v for k,v in changed.items() if k != "transport_checksum"})
    with pytest.raises(GeometryError, match="every working"):
        model_closure_from_dict(changed)
    changed = deepcopy(data)
    changed["work_to_source"].append(changed["work_to_source"][0])
    changed["transport_checksum"] = definition_checksum({k:v for k,v in changed.items() if k != "transport_checksum"})
    with pytest.raises(GeometryError, match="duplicate"):
        model_closure_from_dict(changed)
    changed = deepcopy(data)
    vertex_rows = [row for row in changed["work_to_source"] if row[0] == "vertex"]
    vertex_rows[1][2] = vertex_rows[0][2]
    changed["transport_checksum"] = definition_checksum({k:v for k,v in changed.items() if k != "transport_checksum"})
    with pytest.raises(GeometryError, match="duplicate source"):
        model_closure_from_dict(changed)


def test_chart_batch_parity_individual_errors_and_success_binding():
    model = GeometryModel()
    good = plate(model)
    bad = model.add_face(model.add_polyline(model.add_points(((10,0,0),(12,0,0),(12,1,1),(10,1,0))), close=True),
                         surface=CoonsSurface())
    batch = query_trimmed_surface_charts_by_face(model)
    assert [row.face.id for row in batch.results] == [good,bad]
    assert batch.results[0].error is None
    expected = query_trimmed_surface_charts(model, (good,))
    assert definition_checksum(batch.results[0].charts) == definition_checksum(expected)
    validate_trimmed_surface_charts_binding(model, batch.results[0].charts)
    with pytest.raises(GeometryError) as error:
        query_trimmed_surface_charts(model, (bad,))
    assert batch.results[1].charts is None and batch.results[1].error == str(error.value)


def test_query_content_qualification_reuse_revalidates_raw_topology(monkeypatch):
    model = GeometryModel()
    face = plate(model)
    count = 0
    validate = model.validate_topology
    def counted():
        nonlocal count
        count += 1
        return validate()
    monkeypatch.setattr(model, "validate_topology", counted)
    query_trimmed_surface_charts(model)
    query_trimmed_surface_charts(model)
    assert count == 1
    before_revision = model.revision
    edge = model.faces[face].loop[0].edge
    old = model.edges[edge]
    model._edges[edge] = replace(old, end=old.start)
    with pytest.raises(GeometryError):
        query_trimmed_surface_charts(model)
    assert model.revision == before_revision and count == 2


@pytest.mark.parametrize("query", [query_trimmed_surface_charts, query_trimmed_surface_charts_by_face])
def test_query_cancel_and_same_revision_callback_mutation_abort(query):
    model = GeometryModel()
    face = plate(model)
    plate(model, 20)
    before = to_dict(model)
    with pytest.raises(GeometryError, match="cancelled"):
        query(model, cancellation_check=lambda _: True)
    assert to_dict(model) == before
    def changed(phase):
        if phase == "trimmed surface chart query complete":
            origin = model.faces[face].surface.origin
            origin.flags.writeable = True
            origin[0] += 1
        return False
    with pytest.raises(GeometryError, match="changed during"):
        query(model, cancellation_check=changed)


def test_current_geometry_document_preserves_sparse_local_ids():
    model = GeometryModel()
    model.add_point(-100,0,0)
    face = plate(model, 20)
    model.remove_vertex(1)
    original = to_dict(model)
    restored = from_dict(json.loads(json.dumps(original)))
    assert to_dict(restored) == original
    assert set(restored.faces) == {face}
    assert set(restored.vertices) == set(model.vertices)


@pytest.mark.parametrize("surface_family", ("cylinder", "cone"))
def test_radial_full_revolution_and_axial_extrapolation_are_conservative(surface_family):
    from anygeometry import Cylinder, Cone
    model = GeometryModel()
    first, via, last = model.add_points(((2,0,0),(2**.5,2**.5,0),(0,2,0)))
    face = model.extrude((model.add_arc(first, via, last),), (0,0,3))[0]
    # Its legal trim goes to v=30 on this geometrically identical support.
    support = (Cylinder((0,0,0),(0,0,1),(1,0,0),2,.1,0,np.pi/2) if surface_family == "cylinder"
               else Cone((0,0,0),(0,0,1),(1,0,0),2,2,.1,0,np.pi/2))
    model.set_face_surface(face, support)
    other = plate(model, -1, -1, 2)
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    assert any("bounds overlap" in reason.reason for reason in plan.merge_reasons)


def test_reused_preparation_cannot_accept_callback_raw_mutation():
    from anygeometry import plan_intersections, apply_intersections, IntersectionBatchPolicy
    model = GeometryModel()
    plate(model)
    model.add_plate(model.add_points(((1,-1,-1),(1,2,-1),(1,2,1),(1,-1,1))))
    plan = plan_intersections(model, tuple(model.faces), policy="connect")
    apply_intersections(model, plan, policy="connect")
    def mutation():
        vertex = next(iter(model.vertices.values()))
        vertex.position.flags.writeable = True
        vertex.position[0] += .01
        return False
    with pytest.raises(GeometryError, match="changed during intersection reuse"):
        apply_intersections(model, plan, policy=IntersectionBatchPolicy(cancellation_check=mutation))


@pytest.mark.parametrize("reference_kind", ("vertex", "edge"))
def test_distant_members_sharing_standalone_orientation_dependency_merge(reference_kind):
    model = GeometryModel()
    origin = model.add_point(50,50,50)
    reference = (reference_kind, origin if reference_kind == "vertex" else
                 model.add_line(origin, model.add_point(51,50,50)))
    for x in (0,100):
        edge = model.add_line(*model.add_points(((x,0,0),(x+1,0,0))))
        model.add_member((edge,), orientation_reference=reference)
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1


def test_distant_members_sharing_standalone_attachment_target_merge():
    model = GeometryModel()
    target = model.add_line(*model.add_points(((50,0,0),(51,0,0))))
    for x in (0,100):
        edge = model.add_line(*model.add_points(((x,0,0),(x+1,0,0))))
        member = model.add_member((edge,))
        model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY, AttachmentTargetKind.EDGE,
            target, ParameterRange(0,1), (ParameterRange(0,1),))
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1


def test_promoted_target_extents_merge_unrelated_crossing_material():
    model = GeometryModel()
    face = plate(model, 50)
    target = model.add_line(*model.add_points(((51,.5,-1),(51,.5,1))))
    axis = model.add_line(*model.add_points(((0,0,0),(1,0,0))))
    member = model.add_member((axis,))
    model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY, AttachmentTargetKind.EDGE,
        target, ParameterRange(0,1), (ParameterRange(0,1),))
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    assert plan.components[0].face_ids == (face,)


def test_shared_standalone_active_lineage_merges_attachment_consumers():
    model = GeometryModel()
    dependency = model.add_point(50,50,50)
    for x in (0,100):
        face = plate(model,x)
        edge = model.faces[face].loop[0].edge
        member = model.add_member((edge,))
        model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY, AttachmentTargetKind.EDGE,
            edge, ParameterRange(0,1), (ParameterRange(0,1),), lineage=(("vertex",dependency),))
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1


def test_certified_extracted_components_have_disjoint_material_dependency_maps():
    model = GeometryModel()
    part = model.add_part()
    for x in (0,20,40):
        face = plate(model,x)
        model.add_sheet((face,), part_id=part)
        edge = model.faces[face].loop[0].edge
        model.add_member((edge,), part_id=part)
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 3
    seen = set()
    for component in plan.components:
        closure = extract_model_closure(model, component.handles)
        dependencies = {h for h in closure.source_to_work if h.kind != "part"}
        assert not seen & dependencies
        seen.update(dependencies)


def test_cached_query_and_chart_validator_reject_same_revision_incidence_corruption():
    model = GeometryModel()
    face = plate(model)
    model.add_sheet((face,))
    charts = query_trimmed_surface_charts(model)
    validate_trimmed_surface_charts_binding(model, charts)
    revision = model.revision
    model._face_structural_uses.clear()
    for check in (lambda: query_trimmed_surface_charts(model),
                  lambda: query_trimmed_surface_charts_by_face(model),
                  lambda: validate_trimmed_surface_charts_binding(model, charts)):
        with pytest.raises(GeometryError, match="reverse incidence is stale"):
            check()
    assert model.revision == revision


@pytest.mark.parametrize("target", ("source", "candidate"))
def test_last_commit_callback_cannot_bypass_derived_incidence_requalification(monkeypatch, target):
    from anygeometry import plan_intersections, apply_intersections, IntersectionBatchPolicy
    import anygeometry.edge_subcurve_preimages as provenance
    model = GeometryModel()
    face = plate(model)
    model.add_sheet((face,))
    model.add_plate(model.add_points(((1,-1,-1),(1,2,-1),(1,2,1),(1,-1,1))))
    before = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy="connect")
    finalize = provenance._finalize_edge_subcurve_preimages
    ready = []
    def seal(candidate, draft, **kwargs):
        result = finalize(candidate, draft, **kwargs)
        ready.append(candidate)
        return result
    monkeypatch.setattr(provenance, "_finalize_edge_subcurve_preimages", seal)
    def corrupt():
        if ready:
            candidate = ready.pop()
            (model if target == "source" else candidate)._face_structural_uses.clear()
        return False
    with pytest.raises(GeometryError, match="reverse incidence is stale"):
        apply_intersections(model, plan, policy=IntersectionBatchPolicy(cancellation_check=corrupt))
    # No candidate material is committed. In the source case the callback's raw
    # index edit is external activity; repair it before comparing persisted data.
    if target == "source":
        model._face_structural_uses = {face: {next(iter(model.face_uses))}}
    assert to_dict(model) == before


@pytest.mark.parametrize("operation", ("plan", "validate"))
def test_partition_last_callback_rejects_equal_document_derived_index_corruption(operation):
    from anygeometry.serialization import _serialized_model_state
    model = GeometryModel()
    face = plate(model)
    model.add_sheet((face,))
    partition = plan_independent_components(model)
    before = to_dict(model)
    revision = model.revision
    calls = 0
    def corrupt_at_final_check(_):
        nonlocal calls
        calls += 1
        # One face root: entry check, face-bound check, then the final check
        # immediately preceding the qualified source freshness guard.
        if calls == 3:
            model._face_structural_uses.clear()
        return False
    with pytest.raises(GeometryError, match="reverse incidence is stale"):
        if operation == "plan":
            plan_independent_components(model, cancellation_check=corrupt_at_final_check)
        else:
            validate_component_partition_binding(model, partition, cancellation_check=corrupt_at_final_check)
    assert calls == 3 and model.revision == revision
    assert _serialized_model_state(model) == before
