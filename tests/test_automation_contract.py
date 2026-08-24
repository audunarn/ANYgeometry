from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from anygeometry import EntityHandle, EntityRef, GeometryModel
from anygeometry.automation import (
    ApplyResult,
    AutomationError,
    AutomationResponse,
    Command,
    CommandBatch,
    PROTOCOL_VERSION,
    SelectionSpec,
    SelectionResult,
    apply_plan,
    automation_dumps,
    automation_json_schema,
    automation_loads,
    describe_capabilities,
    describe_entities,
    describe_model,
    execute_query,
    plan_commands,
    select_entities,
    tool_catalog,
)


def quantity(value, unit="m", frame="model_local"):
    return {"value": value, "unit": unit, "frame": frame}


def header(model: GeometryModel, request_id="request"):
    return {
        "protocol_version": 1,
        "request_id": request_id,
        "model_id": str(model.model_id),
        "expected_revision": model.revision,
    }


def selection(model: GeometryModel, where, **kwargs):
    return SelectionSpec(
        1,
        kwargs.pop("request_id", "select"),
        model.model_id,
        model.revision,
        where,
        **kwargs,
    )


def square(model: GeometryModel):
    vertices = model.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    face = model.add_plate(vertices)
    part = model.add_part(name="deck", metadata={"structural:zone": "deck"})
    sheet = model.add_sheet((face,), part_id=part, name="deck plate")
    return vertices, face, part, sheet


def test_capabilities_and_schema_are_dependency_free_and_versioned():
    capabilities = describe_capabilities()
    assert capabilities["protocol_version"] == PROTOCOL_VERSION == 1
    assert capabilities["geometry_schema_version"] == 4
    assert len(tool_catalog()) == 7
    assert {item["name"] for item in tool_catalog()} == {
        "kernel_capabilities",
        "model_summary",
        "select_entities",
        "describe_entities",
        "query_geometry",
        "plan_edit",
        "apply_edit",
    }
    schema = automation_json_schema()
    assert schema["$defs"]["batch"]["properties"]["commands"]["maxItems"] == 256


def test_strict_json_rejects_duplicate_extra_nonfinite_and_oversized_data():
    model = GeometryModel()
    payload = {**header(model), "where": {"kind": "point"}}
    made = automation_loads("selection", automation_dumps(payload))
    assert isinstance(made, SelectionSpec)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        automation_loads("selection", '{"protocol_version":1,"protocol_version":1}')
    with pytest.raises(AutomationError) as extra:
        automation_loads("selection", json.dumps({**payload, "python": "open('x')"}))
    assert extra.value.code == "UNKNOWN_FIELD"
    with pytest.raises(ValueError, match="non-finite"):
        automation_loads("selection", '{"x":NaN}')
    with pytest.raises(ValueError, match="1 MiB"):
        automation_loads("selection", " " * 1_048_577)
    with pytest.raises(AutomationError, match="descending must be Boolean"):
        automation_loads("selection", automation_dumps({**payload, "descending": "false"}))


def test_aliases_group_tag_owner_topology_and_metadata_return_canonical_kinds():
    model = GeometryModel()
    vertices, face, part, sheet = square(model)
    model.add_to_group("primary", (EntityRef("face", face),))
    model.tag(EntityRef("face", face), "deck")
    # Metadata predicates are exact and namespaced; use the persistent Part name
    # for owner selection and group/tag for geometry semantics.
    point_result = select_entities(model, selection(model, {"kind": "point"}))
    plate_result = select_entities(model, selection(model, {"kind": "plate"}))
    assert {item.kind for item in point_result.entities} == {"vertex"}
    assert SelectionResult.from_dict(json.loads(automation_dumps(point_result))) == point_result
    assert [item.kind for item in plate_result.entities] == ["sheet"]
    owner = model.handle("part", part)
    owned = select_entities(model, selection(model, {"owner": {"model_id": str(model.model_id), "kind": "part", "id": part}}))
    assert model.handle("sheet", sheet) in {item.handle for item in owned.entities}
    grouped = select_entities(model, selection(model, {"all": [{"group": "primary"}, {"tag": "deck"}]}))
    assert [item.handle for item in grouped.entities] == [model.handle("face", face)]
    incident = select_entities(model, selection(model, {"incident_to": {"model_id": str(model.model_id), "kind": "sheet", "id": sheet}}))
    assert any(item.kind == "part" for item in incident.entities)
    summary = describe_entities(model, (owner,), expected_revision=model.revision, detail=True).entities[0]
    assert summary.kind == "part" and summary.topology
    metadata = select_entities(model, selection(model, {"metadata": {"key": "structural:zone", "equals": "deck"}}))
    assert [item.handle for item in metadata.entities] == [owner]


def test_boundary_connectivity_and_owner_selectors_use_reverse_closure(monkeypatch):
    model = GeometryModel()
    _vertices, face, part, sheet = square(model)
    face_use = model.sheets[sheet].face_use_ids[0]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("topology selector fell back to whole-model enumeration")

    monkeypatch.setattr("anygeometry.automation.selection._all_handles", forbidden)
    sheet_handle = {"model_id": str(model.model_id), "kind": "sheet", "id": sheet}
    face_use_handle = {"model_id": str(model.model_id), "kind": "face_use", "id": face_use}
    part_handle = {"model_id": str(model.model_id), "kind": "part", "id": part}
    boundary = select_entities(model, selection(model, {"all": [{"kind": "face_use"}, {"boundary_of": sheet_handle}]}))
    assert [item.handle.id for item in boundary.entities] == [face_use]
    incident = select_entities(model, selection(model, {"all": [{"kind": "face"}, {"incident_to": face_use_handle}]}))
    assert [item.handle.id for item in incident.entities] == [face]
    connected = select_entities(model, selection(model, {"all": [{"kind": "face"}, {"connected_to": part_handle}]}))
    assert [item.handle.id for item in connected.entities] == [face]
    owned = select_entities(model, selection(model, {"all": [{"kind": "sheet"}, {"owner": part_handle}]}))
    assert [item.handle.id for item in owned.entities] == [sheet]


def test_spatial_range_boolean_ordering_pagination_and_stale_cursor():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
    model.add_line(points[0], points[1])
    model.add_line(points[1], points[2])
    where = {
        "all": [
            {"kind": "edge"},
            {"length": {"min": quantity(900, "mm"), "max": quantity(1.1, "m")}},
            {"aabb": {"min": quantity((-1, -1, -1)), "max": quantity((3, 1, 1))}},
        ]
    }
    first = select_entities(model, selection(model, where, page_size=1, order_by="centroid", descending=True))
    assert first.total == 2 and first.next_cursor
    second = select_entities(model, selection(model, where, page_size=1, order_by="centroid", descending=True, cursor=first.next_cursor))
    assert first.entities[0].handle.id > second.entities[0].handle.id
    model.add_point(9, 9, 9)
    with pytest.raises(AutomationError) as stale:
        select_entities(model, SelectionSpec(1, "old", model.model_id, model.revision, where, "centroid", True, 1, first.next_cursor))
    assert stale.value.code == "STALE_CURSOR"


def test_curve_support_area_and_radius_predicates_are_typed_ranges():
    model = GeometryModel()
    vertices, face, _part, _sheet = square(model)
    arc_points = model.add_points(((1, 0, 2), (0, 1, 2), (-1, 0, 2)))
    arc = model.add_arc(*arc_points)
    curved = select_entities(
        model,
        selection(
            model,
            {
                "all": [
                    {"kind": "edge"},
                    {"curve_type": "Arc"},
                    {"radius": {"min": quantity(0.99), "max": quantity(1.01)}},
                ]
            },
        ),
    )
    assert [item.handle.id for item in curved.entities] == [arc]
    planar = select_entities(
        model,
        selection(
            model,
            {
                "all": [
                    {"kind": "face"},
                    {"support_type": "Plane"},
                    {"area": {"min": quantity(1.99, "m^2"), "max": quantity(2.01, "m^2")}},
                ]
            },
        ),
    )
    assert [item.handle.id for item in planar.entities] == [face]


def test_nearest_requires_bound_and_has_deterministic_handle_ties():
    model = GeometryModel()
    model.add_points(((-1, 0, 0), (1, 0, 0)))
    result = select_entities(
        model,
        selection(
            model,
            {"nearest": {"point": quantity((0, 0, 0)), "max_distance": quantity(2), "limit": 2}},
            order_by="distance",
        ),
    )
    assert [item.handle.id for item in result.entities] == [1, 2]
    with pytest.raises(AutomationError):
        select_entities(model, selection(model, {"nearest": {"point": quantity((0, 0, 0)), "limit": 1}}))


def test_world_frame_and_model_units_are_explicit_and_spatially_prefiltered(monkeypatch):
    model = GeometryModel()
    transform = np.eye(4)
    # The document transform is expressed in model units (millimetres here).
    transform[:3, 3] = (10000.0, 20000.0, 30000.0)
    model.set_document_settings(units="mm", coordinate_transform=transform)
    inside = model.add_point(1000, 0, 0)
    model.add_point(3000, 0, 0)
    calls = []
    original = model.spatial_candidates

    def tracked(lower, upper, **kwargs):
        calls.append((tuple(lower), tuple(upper), kwargs))
        return original(lower, upper, **kwargs)

    monkeypatch.setattr(model, "spatial_candidates", tracked)
    result = select_entities(
        model,
        selection(
            model,
            {
                "all": [
                    {"kind": "point"},
                    {
                        "aabb": {
                            "min": quantity((10.9, 19.9, 29.9), "m", "world"),
                            "max": quantity((11.1, 20.1, 30.1), "m", "world"),
                        }
                    },
                ]
            },
        ),
    )
    assert [item.handle.id for item in result.entities] == [inside]
    assert len(calls) == 1


def test_wrong_model_revision_cardinality_and_unknown_units_fail_typed():
    model = GeometryModel()
    other = GeometryModel()
    model.add_points(((0, 0, 0), (1, 0, 0)))
    with pytest.raises(AutomationError) as wrong:
        select_entities(model, SelectionSpec(1, "wrong", other.model_id, model.revision, {"kind": "point"}))
    assert wrong.value.code == "WRONG_MODEL"
    with pytest.raises(AutomationError) as stale:
        select_entities(model, SelectionSpec(1, "stale", model.model_id, 0, {"kind": "point"}))
    assert stale.value.code == "STALE_REVISION"
    with pytest.raises(AutomationError) as ambiguous:
        select_entities(model, selection(model, {"kind": "point"}, expected_cardinality=(1, 1)))
    assert ambiguous.value.code == "AMBIGUOUS_SELECTION"
    with pytest.raises(AutomationError) as unit:
        select_entities(model, selection(model, {"centroid_axis": {"axis": "x", "min": quantity(0, "parsec")}}))
    assert unit.value.code == "UNKNOWN_UNIT"


def symbolic_plate_batch(model: GeometryModel) -> CommandBatch:
    positions = ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))
    commands = [Command(f"p{index}", "create_point", {"position": quantity(position)}) for index, position in enumerate(positions, 1)]
    commands.append(Command("deck", "create_plate", {"vertices": [f"p{index}.vertex" for index in range(1, 5)]}))
    return CommandBatch(1, "symbolic-plate", model.model_id, model.revision, tuple(commands))


def test_symbolic_point_to_owned_plate_is_one_atomic_changeset():
    model = GeometryModel()
    events = []
    model.add_change_hook(events.append)
    before_ids = dict(model._next_id)
    before_structural = dict(model._next_structural_id)
    plan = plan_commands(model, symbolic_plate_batch(model))
    assert model.revision == 0
    assert model._next_id == before_ids
    assert model._next_structural_id == before_structural
    assert plan.expected_entity_counts == {"coedge": 4, "edge": 4, "face": 1, "face_use": 1, "part": 1, "sheet": 1, "vertex": 4}
    assert plan.expected_owners["deck.sheet"] == ("deck.part",)
    assert plan.expected_owners["deck.face"] == ("deck.part",)
    assert plan.affected_bounds == pytest.approx((0, 0, 0, 2, 1, 0))
    result = apply_plan(model, plan)
    assert result.revision_before == 0 and result.revision_after == 1
    assert events == [result.change_set]
    assert model.last_change_set is result.change_set
    assert len(result.outputs["deck.part"]) == len(result.outputs["deck.sheet"]) == len(result.outputs["deck.face"]) == 1
    assert len(result.outputs["deck.coedge"]) == 4
    assert ApplyResult.from_dict(json.loads(automation_dumps(result))) == result
    assert model.validate_topology() == ()
    with pytest.raises(AutomationError) as duplicate:
        apply_plan(model, plan)
    assert duplicate.value.code == "STALE_PLAN"


def test_planning_preserves_revision_high_water_and_existing_cache_objects():
    model = GeometryModel()
    vertices, face, _part, _sheet = square(model)
    model.edge_length(next(iter(model.edges)))
    spatial = model._spatial()
    arc_cache = dict(model._arc_cache)
    length_cache = dict(model._edge_length_cache)
    batch = CommandBatch(
        1,
        "copy-plan",
        model.model_id,
        model.revision,
        (
            Command(
                "copy",
                "copy",
                {
                    "targets": {"where": {"handle": {"model_id": str(model.model_id), "kind": "face", "id": face}}, "expected_cardinality": [1, 1]},
                    "transform": {"translate": quantity((1, 0, 0))},
                },
            ),
        ),
    )
    revision = model.revision
    high = (dict(model._next_id), dict(model._next_structural_id))
    plan = plan_commands(model, batch)
    assert model.revision == revision
    assert (model._next_id, model._next_structural_id) == high
    assert model._spatial_index is spatial
    assert model._arc_cache == arc_cache and model._edge_length_cache == length_cache
    assert plan.expected_entity_counts == {
        "coedge": 4,
        "edge": 4,
        "face": 1,
        "face_use": 1,
        "part": 1,
        "sheet": 1,
        "vertex": 4,
    }
    assert plan.expected_owners == {"copy.entity": ("copy.entity",)}
    assert plan.affected_bounds == pytest.approx((0, 0, 0, 3, 1, 0))


def test_translate_copy_pattern_group_tag_move_and_delete_use_frozen_handles():
    model = GeometryModel()
    vertex = model.add_point(0, 0, 0)
    handle = {"model_id": str(model.model_id), "kind": "vertex", "id": vertex}
    commands = (
        Command("move", "move", {"target": handle, "to": quantity((1, 0, 0))}),
        Command("copy", "copy", {"targets": [handle], "transform": {"translate": quantity((1, 0, 0))}}),
        Command("tagged", "tag", {"targets": "copy.entity", "tags": ["automation"]}),
        Command("array", "pattern", {"targets": [handle], "pattern": {"type": "linear", "direction": quantity((1, 0, 0), "1"), "spacing": quantity(1), "count": 2}}),
        Command("grouped", "group", {"targets": "array.entity", "group": "copies"}),
    )
    result = apply_plan(model, plan_commands(model, CommandBatch(1, "edits", model.model_id, model.revision, commands)))
    assert model.vertex_position(vertex) == pytest.approx((1, 0, 0))
    assert len(result.outputs["copy.entity"]) == 1
    assert model.tags_for(EntityRef("vertex", result.outputs["copy.entity"][0].id)) == ("automation",)
    assert len(model.group("copies")) == 2
    delete = CommandBatch(1, "delete", model.model_id, model.revision, (Command("gone", "delete", {"targets": [{"model_id": str(model.model_id), "kind": "vertex", "id": result.outputs["copy.entity"][0].id}]}),))
    apply_plan(model, plan_commands(model, delete))
    assert result.outputs["copy.entity"][0].id not in model.vertices


def test_failed_apply_rolls_back_and_tampered_or_stale_plans_do_not_mutate():
    model = GeometryModel()
    vertex = model.add_point(0, 0, 0)
    batch = CommandBatch(1, "bad", model.model_id, model.revision, (Command("edge", "create_edge", {"start": {"model_id": str(model.model_id), "kind": "vertex", "id": vertex}, "end": {"model_id": str(model.model_id), "kind": "vertex", "id": vertex}}),))
    plan = plan_commands(model, batch)
    revision, high = model.revision, dict(model._next_id)
    assert plan.diagnostics and plan.diagnostics[0].startswith("APPLICATION_FAILED:")
    with pytest.raises(AutomationError) as failed:
        apply_plan(model, plan)
    assert failed.value.code == "BLOCKED_PLAN"
    assert model.revision == revision and model._next_id == high and not model.edges
    with pytest.raises(AutomationError) as tampered:
        apply_plan(model, replace(plan, diagnostics=("changed",)))
    assert tampered.value.code == "TAMPERED_PLAN"
    model.add_point(1, 0, 0)
    with pytest.raises(AutomationError) as stale:
        apply_plan(model, plan)
    assert stale.value.code == "STALE_PLAN"


def test_execute_query_and_model_description_are_revision_bound():
    model = GeometryModel()
    vertices = model.add_points(((0, 0, 0), (3, 0, 0)))
    edge = model.add_line(*vertices)
    description = describe_model(model, {**header(model, "model")})
    assert description["entity_counts"]["edge"] == 1
    response = execute_query(
        model,
        {
            **header(model, "measure"),
            "operation": "measure",
            "arguments": {"handles": [{"model_id": str(model.model_id), "kind": "edge", "id": edge}], "quantity": "length"},
        },
    )
    assert response["value"] == pytest.approx(3.0)
    assert response["unit"] == "m"
    encoded = automation_dumps(
        {
            **header(model, "query-codec"),
            "operation": "measure",
            "arguments": {
                "handles": [{"model_id": str(model.model_id), "kind": "edge", "id": edge}],
                "quantity": "length",
            },
        }
    )
    assert automation_loads("query", encoded)["operation"] == "measure"
    with pytest.raises(AutomationError) as extra:
        automation_loads(
            "query",
            automation_dumps({**header(model), "operation": "measure", "arguments": {}, "path": "x"}),
        )
    assert extra.value.code == "UNKNOWN_FIELD"


def test_forward_reference_unknown_port_and_command_limits_fail_before_mutation():
    model = GeometryModel()
    with pytest.raises(AutomationError) as forward:
        plan_commands(model, CommandBatch(1, "forward", model.model_id, 0, (Command("edge", "create_edge", {"start": "later.vertex", "end": "later.vertex"}), Command("later", "create_point", {"position": quantity((0, 0, 0))}))))
    assert forward.value.code == "FORWARD_REFERENCE"
    with pytest.raises(AutomationError) as port:
        plan_commands(model, CommandBatch(1, "port", model.model_id, 0, (Command("p", "create_point", {"position": quantity((0, 0, 0))}), Command("edge", "create_edge", {"start": "p.face", "end": "p.vertex"}))))
    assert port.value.code == "UNKNOWN_OUTPUT_PORT"
    with pytest.raises(AutomationError) as many:
        CommandBatch(1, "many", model.model_id, 0, tuple(Command(f"p{i}", "create_point", {"position": quantity((i, 0, 0))}) for i in range(257)))
    assert many.value.code == "PAYLOAD_TOO_LARGE"


def test_selector_depth_and_predicate_limits_apply_even_to_empty_models():
    model = GeometryModel()
    node = {"kind": "point"}
    for _ in range(9):
        node = {"not": node}
    with pytest.raises(AutomationError) as deep:
        select_entities(model, selection(model, node))
    assert deep.value.code == "SELECTOR_TOO_DEEP"
    with pytest.raises(AutomationError) as broad:
        select_entities(model, selection(model, {"all": [{"kind": "point"} for _ in range(65)]}))
    assert broad.value.code == "SELECTOR_TOO_LARGE"


def test_plan_mapping_round_trip_preserves_digest_and_tamper_detection():
    model = GeometryModel()
    plan = plan_commands(
        model,
        CommandBatch(
            1,
            "roundtrip",
            model.model_id,
            0,
            (Command("p", "create_point", {"position": quantity((1, 2, 3))}),),
        ),
    )
    from anygeometry.automation import EditPlan

    restored = EditPlan.from_dict(json.loads(automation_dumps(plan)))
    assert restored == plan
    payload = plan.to_dict()
    payload["commands"][0]["arguments"]["position"]["value"][0] = 999
    tampered = EditPlan.from_dict(payload)
    with pytest.raises(AutomationError) as failure:
        apply_plan(model, tampered)
    assert failure.value.code == "TAMPERED_PLAN"


def test_response_and_error_codecs_are_exact_field_round_trips():
    success = AutomationResponse(1, "ok", True, {"value": 3})
    assert AutomationResponse.from_dict(json.loads(automation_dumps(success))) == success
    error = AutomationError("UNSUPPORTED", "not available", details={"capability": "curve"})
    failure = AutomationResponse(1, "failed", False, None, error)
    restored = AutomationResponse.from_dict(json.loads(automation_dumps(failure)))
    assert restored.to_dict() == failure.to_dict()
    payload = success.to_dict()
    payload["unexpected"] = True
    with pytest.raises(AutomationError) as extra:
        AutomationResponse.from_dict(payload)
    assert extra.value.code == "UNKNOWN_FIELD"


def test_rotate_and_mirror_commands_use_explicit_units_and_frames():
    model = GeometryModel()
    vertex = model.add_point(1, 0, 0)
    handle = {"model_id": str(model.model_id), "kind": "vertex", "id": vertex}
    commands = (
        Command(
            "rotate",
            "rotate",
            {
                "targets": [handle],
                "axis_point": quantity((0, 0, 0)),
                "axis_direction": quantity((0, 0, 1), "1"),
                "angle": {"value": 90, "unit": "deg"},
            },
        ),
        Command(
            "mirror",
            "mirror",
            {
                "targets": [handle],
                "plane_point": quantity((0, 0, 0)),
                "plane_normal": quantity((1, 0, 0), "1"),
            },
        ),
    )
    result = apply_plan(
        model,
        plan_commands(
            model,
            CommandBatch(1, "rotate-mirror", model.model_id, model.revision, commands),
        ),
    )
    assert model.vertex_position(vertex) == pytest.approx((0, 1, 0), abs=1e-12)
    mirrored = result.outputs["mirror.entity"]
    assert len(mirrored) == 1
    assert model.vertex_position(mirrored[0].id) == pytest.approx((0, 1, 0), abs=1e-12)


def test_dependency_safe_delete_blocker_rolls_back_complete_batch():
    model = GeometryModel()
    vertices, face, _part, _sheet = square(model)
    point = Command("new", "create_point", {"position": quantity((9, 9, 9))})
    delete = Command(
        "delete",
        "delete",
        {"targets": [{"model_id": str(model.model_id), "kind": "face", "id": face}]},
    )
    plan = plan_commands(
        model,
        CommandBatch(1, "blocked-delete", model.model_id, model.revision, (point, delete)),
    )
    revision = model.revision
    high = dict(model._next_id)
    with pytest.raises(AutomationError) as failure:
        apply_plan(model, plan)
    assert failure.value.code == "BLOCKED_PLAN"
    assert model.revision == revision
    assert model._next_id == high
    assert len(model.vertices) == len(vertices) and face in model.faces


def test_qualified_face_imprint_command_uses_existing_policy_workflow():
    model = GeometryModel()
    horizontal = model.add_plate(
        model.add_points(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)))
    )
    vertical = model.add_plate(
        model.add_points(((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)))
    )
    command = Command(
        "cut",
        "imprint",
        {
            "first": {"model_id": str(model.model_id), "kind": "face", "id": horizontal},
            "second": {"model_id": str(model.model_id), "kind": "face", "id": vertical},
            "policy": "imprint",
        },
    )
    plan = plan_commands(
        model,
        CommandBatch(1, "imprint", model.model_id, model.revision, (command,)),
    )
    result = apply_plan(model, plan)
    assert result.revision_after == result.revision_before + 1
    assert model.validate_topology() == ()
    assert result.outputs["cut.entity"]
