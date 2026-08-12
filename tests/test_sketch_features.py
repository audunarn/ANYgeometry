"""Persistent face sketches, constraints and extrusion."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    ConnectionIntent,
    EntityRef,
    GeometryModel,
    ImprintOperation,
    IntersectionKind,
    SketchConstraint,
    SketchDefinition,
    apply_imprint,
    face_sketch_plane,
    from_dict,
    plan_imprint,
    query_intersection,
    solve_sketch,
    to_dict,
)
from anygeometry.surfaces import Plane


def _plate() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    points = geometry.add_points(((0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)))
    return geometry, geometry.add_plate(points)


def _definition(extrusion: float = 1.0) -> SketchDefinition:
    return SketchDefinition(
        points={"a": (0.5, 0.5), "b": (1.8, 0.5), "c": (1.8, 1.5), "d": (0.5, 1.5)},
        path=("a", "b", "c", "d"),
        constraints=(SketchConstraint("distance", "a", "b", 1.0),),
        extrusion=extrusion,
    )


def test_distance_and_boundary_coincidence_constraints_solve_in_face_plane():
    geometry, face = _plate()
    plane = face_sketch_plane(geometry, face)
    definition = SketchDefinition(
        points={"a": (0.2, 0.3), "b": (1.7, 0.3), "c": (1.7, 1.0)},
        path=("a", "b", "c"),
        constraints=(
            SketchConstraint("on_edge", "a", boundary_index=0),
            SketchConstraint("distance", "a", "b", 1.0),
        ),
    )

    solved = solve_sketch(definition, plane)

    assert solved["a"][1] == pytest.approx(0.0)
    assert np.linalg.norm(np.asarray(solved["a"]) - solved["b"]) == pytest.approx(1.0)


def test_sketch_feature_extrudes_shell_faces_and_regenerates_editably():
    geometry, face = _plate()
    geometry.features.capture_baseline(geometry)
    feature = geometry.features.append(
        "geometry.sketch.extrude",
        name="Deck sketch",
        parameters=_definition().to_parameters(),
        inputs={"support_face": (EntityRef("face", face),)},
    )

    assert geometry.regenerate_features().success
    first_outputs = dict(geometry.features.get(feature.feature_id).outputs)
    assert {"point/a", "point/b", "profile/edge/0", "extrusion/face/0"} <= set(first_outputs)
    assert len([key for key in first_outputs if key.startswith("extrusion/face/")]) == 4

    changed = _definition(extrusion=2.0).to_parameters()
    changed["points"]["c"] = [2.0, 1.8]
    geometry.features.update(feature.feature_id, parameters=changed)
    report = geometry.regenerate_features()

    assert report.success
    assert all(geometry.resolve_ref(reference) for reference in first_outputs.values())
    new_top = geometry.features.get(feature.feature_id).outputs["point/c"]
    assert geometry.vertices[new_top.id].position[1] == pytest.approx(1.8)


def test_straight_sketch_extrusion_has_exact_planar_support_and_connects() -> None:
    geometry, support_face = _plate()
    geometry.features.capture_baseline(geometry)
    feature = geometry.features.append(
        "geometry.sketch.extrude",
        parameters=_definition().to_parameters(),
        inputs={"support_face": (EntityRef("face", support_face),)},
    )
    assert geometry.regenerate_features().success
    outputs = geometry.features.get(feature.feature_id).outputs
    walls = tuple(
        reference.id
        for key, reference in sorted(outputs.items())
        if key.startswith("extrusion/face/")
    )

    assert len(walls) == 4
    assert all(isinstance(geometry.faces[face_id].surface, Plane) for face_id in walls)
    support_sheet = geometry.add_sheet((support_face,))
    wall_sheet = geometry.add_sheet(walls)

    result = query_intersection(
        geometry,
        geometry.handle("face", support_face),
        geometry.handle("face", walls[0]),
    )
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )
    assert application.face_intersection is not None
    shared_edge = application.face_intersection.edge.id
    sheet_ids = {
        geometry.face_uses[use_id].sheet_id
        for use_id in geometry.face_uses_using_edge(shared_edge)
    }
    assert sheet_ids == {support_sheet, wall_sheet}
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()


def test_nonplanar_spline_extrusion_remains_typed_unsupported() -> None:
    geometry, support_face = _plate()
    start, control, end = geometry.add_points(
        ((0.5, 0.5, 0.0), (1.5, 1.5, 0.0), (2.5, 0.5, 0.0))
    )
    spline = geometry.add_spline(start, (control,), end)
    wall = geometry.extrude((spline,), (0.0, 0.0, 1.0))[0]

    result = query_intersection(
        geometry,
        geometry.handle("face", support_face),
        geometry.handle("face", wall),
    )

    assert result.kind is IntersectionKind.UNSUPPORTED
    assert not result.classified
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
    assert plan.operation is ImprintOperation.NO_TOPOLOGY


def test_sketch_feature_round_trips_with_constraints_and_support_identity():
    geometry, face = _plate()
    geometry.features.capture_baseline(geometry)
    geometry.features.append(
        "geometry.sketch.extrude",
        parameters=_definition().to_parameters(),
        inputs={"support_face": (EntityRef("face", face),)},
    )
    assert geometry.regenerate_features().success

    restored = from_dict(to_dict(geometry))

    assert to_dict(restored) == to_dict(geometry)
    record = restored.features.records[-1]
    assert record.kind == "geometry.sketch.extrude"
    assert record.parameters["constraints"][0]["kind"] == "distance"


def test_point_coincidence_reuses_one_topology_vertex_for_closed_open_path():
    geometry, face = _plate()
    definition = SketchDefinition(
        points={"a": (0.5, 0.5), "b": (1.5, 0.5), "c": (1.0, 1.5), "d": (0.6, 0.6)},
        path=("a", "b", "c", "d"),
        constraints=(SketchConstraint("coincident", "a", "d"),),
        closed=False,
        extrusion=1.0,
    )
    geometry.features.capture_baseline(geometry)
    feature = geometry.features.append(
        "geometry.sketch.extrude",
        parameters=definition.to_parameters(),
        inputs={"support_face": (EntityRef("face", face),)},
    )

    assert geometry.regenerate_features().success
    outputs = geometry.features.get(feature.feature_id).outputs

    assert outputs["point/a"] == outputs["point/d"]
    assert len([key for key in outputs if key.startswith("extrusion/face/")]) == 3
