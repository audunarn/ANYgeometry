"""Affine editing, structural selection, and scalable pattern contracts."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    AffineTransform,
    EntityRef,
    FeatureOutputRef,
    GeometryError,
    GeometryModel,
    ResolutionStatus,
    copy_rotated,
    copy_translated,
    insert_model,
    pattern_entities,
    rectangular_pattern,
    rotate_entities,
    translate_entities,
)


def _owned_plate() -> tuple[GeometryModel, int, int, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    part = geometry.add_part(name="panel")
    sheet = geometry.add_sheet((face,), part_id=part, name="plating")
    geometry.add_member((geometry.faces[face].loop[0].edge,), part_id=part, name="edge")
    return geometry, face, part, sheet


def test_affine_transform_construction_composition_inverse_and_immutability() -> None:
    translated = AffineTransform.translation((2.0, -1.0, 3.0))
    rotated = AffineTransform.rotation(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0
    )
    combined = translated.then(rotated)
    points = np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, -1.0)))

    expected = rotated.apply_points(translated.apply_points(points))
    assert combined.apply_points(points) == pytest.approx(expected)
    assert combined.inverse().apply_points(expected) == pytest.approx(points)
    assert combined == AffineTransform(combined.matrix)
    with pytest.raises(ValueError):
        combined.matrix[0, 0] = 99.0

    reflected = AffineTransform.reflection((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert reflected.apply_points((3.0, 2.0, 0.0)) == pytest.approx((-1.0, 2.0, 0.0))
    assert AffineTransform.scale(2.0, (1.0, 0.0, 0.0)).apply_points(
        (2.0, 0.0, 0.0)
    ) == pytest.approx((3.0, 0.0, 0.0))


def test_translate_and_rotate_structural_selections_preserve_identity() -> None:
    geometry, face, part, sheet = _owned_plate()
    sheet_handle = geometry.handle("sheet", sheet)
    part_handle = geometry.handle("part", part)
    original_ids = {
        "vertices": tuple(geometry.vertices),
        "edges": tuple(geometry.edges),
        "faces": tuple(geometry.faces),
        "parts": tuple(geometry.parts),
        "sheets": tuple(geometry.sheets),
    }
    revision = geometry.revision

    moved = translate_entities(geometry, (sheet_handle,), (4.0, -2.0, 1.0))

    assert moved
    assert geometry.revision == revision + 1
    assert tuple(geometry.vertices) == original_ids["vertices"]
    assert tuple(geometry.edges) == original_ids["edges"]
    assert tuple(geometry.faces) == original_ids["faces"]
    assert tuple(geometry.parts) == original_ids["parts"]
    assert tuple(geometry.sheets) == original_ids["sheets"]
    assert geometry.resolve_handle(sheet_handle).status is ResolutionStatus.ACTIVE
    assert geometry.resolve_handle(part_handle).status is ResolutionStatus.ACTIVE
    assert face in geometry.faces
    assert min(vertex.position[0] for vertex in geometry.vertices.values()) == pytest.approx(4.0)

    rotate_entities(
        geometry,
        (part_handle,),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        np.pi / 2.0,
    )
    assert geometry.resolve_handle(part_handle).status is ResolutionStatus.ACTIVE
    assert geometry.validate_topology() == ()


def test_wrong_model_handle_rejects_move_and_copy_without_mutation() -> None:
    geometry = GeometryModel()
    point = geometry.add_point(1.0, 2.0, 3.0)
    other = GeometryModel()
    wrong = other.handle("vertex", other.add_point(0.0, 0.0, 0.0))
    before = geometry.vertex_position(point).copy()
    revision = geometry.revision

    with pytest.raises(GeometryError, match="wrong-model|another model"):
        translate_entities(geometry, (wrong,), (1.0, 0.0, 0.0))
    with pytest.raises(GeometryError, match="wrong-model|another model"):
        copy_translated(geometry, (wrong,), (1.0, 0.0, 0.0))

    assert geometry.vertex_position(point) == pytest.approx(before)
    assert geometry.revision == revision
    assert tuple(geometry.vertices) == (point,)


def test_structural_single_copy_returns_complete_cross_model_handle_map() -> None:
    geometry, _face, part, sheet = _owned_plate()
    member = next(iter(geometry.members))
    source_part = geometry.handle("part", part)
    source_sheet = geometry.handle("sheet", sheet)
    source_member = geometry.handle("member", member)
    originals = np.asarray([vertex.position for vertex in geometry.vertices.values()])

    copied = copy_translated(geometry, (source_part,), (5.0, 0.0, 0.0))

    for source in (source_part, source_sheet, source_member):
        destination = copied.mapped_handle(source)
        assert destination.model_id == geometry.model_id
        assert destination.id != source.id
        assert geometry.resolve_handle(destination).status is ResolutionStatus.ACTIVE
    assert len(geometry.parts) == 2
    assert len(geometry.sheets) == 2
    assert len(geometry.members) == 2
    copied_vertices = np.asarray(
        [
            geometry.vertex_position(mapped.id)
            for source, mapped in copied.entity_map.items()
            if source.kind == "vertex"
        ]
    )
    assert sorted(copied_vertices[:, 0]) == pytest.approx(
        sorted((originals[:, 0] + 5.0).tolist())
    )
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_insert_and_rotation_copy_map_source_model_handles() -> None:
    source, face, part, _sheet = _owned_plate()
    destination = GeometryModel()
    inserted = insert_model(
        destination,
        source,
        matrix=AffineTransform.translation((3.0, 0.0, 0.0)),
    )
    assert inserted.mapped_handle(source.handle("part", part)).model_id == destination.model_id
    assert inserted.mapped(EntityRef("face", face)).id in destination.faces

    rotated = copy_rotated(
        destination,
        (inserted.mapped(EntityRef("face", face)),),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        np.pi,
    )
    assert rotated.entity_map
    assert destination.validate_topology() == ()


def test_generic_and_rectangular_patterns_have_deterministic_copy_semantics() -> None:
    geometry = GeometryModel()
    point = geometry.add_point(0.0, 0.0, 0.0)
    reference = EntityRef("vertex", point)
    generic = pattern_entities(
        geometry,
        (reference,),
        (
            AffineTransform.translation((1.0, 0.0, 0.0)),
            AffineTransform.rotation(
                (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0
            ) @ AffineTransform.translation((2.0, 0.0, 0.0)),
        ),
    )
    assert len(generic.instances) == len(generic.transforms) == 2
    assert geometry.vertex_position(generic.instances[0].mapped(reference).id) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert geometry.vertex_position(generic.instances[1].mapped(reference).id) == pytest.approx(
        (0.0, 2.0, 0.0), abs=1.0e-12
    )

    rectangular = rectangular_pattern(
        geometry,
        (reference,),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (2.0, 3.0),
        (1, 2),
    )
    assert len(rectangular.instances) == 5
    positions = {
        tuple(np.round(geometry.vertex_position(item.mapped(reference).id), 12))
        for item in rectangular.instances
    }
    assert positions == {
        (0.0, 3.0, 0.0),
        (0.0, 6.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 3.0, 0.0),
        (2.0, 6.0, 0.0),
    }


def test_pattern_failure_rolls_back_inserted_instances_without_reusing_ids() -> None:
    geometry = GeometryModel()
    start, via, end = geometry.add_points(
        ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0))
    )
    arc = geometry.add_arc(start, via, end)
    before_vertices = tuple(geometry.vertices)
    before_edges = tuple(geometry.edges)
    before_state = geometry.id_state()
    revision = geometry.revision

    with pytest.raises(GeometryError, match="anisotropic scale or shear"):
        pattern_entities(
            geometry,
            (EntityRef("edge", arc),),
            (
                AffineTransform.translation((2.0, 0.0, 0.0)),
                AffineTransform.scale((2.0, 1.0, 1.0)),
            ),
        )

    assert tuple(geometry.vertices) == before_vertices
    assert tuple(geometry.edges) == before_edges
    assert geometry.revision == revision
    assert geometry.id_state()["vertex"] > before_state["vertex"]
    assert geometry.id_state()["edge"] > before_state["edge"]


def test_patterns_clone_only_the_selected_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = GeometryModel()
    selected = geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    edge = geometry.add_line(*selected)
    geometry.add_points((float(index), 20.0, 0.0) for index in range(128))
    original_clone = GeometryModel.clone
    cloned_sizes: list[int] = []

    def observed_clone(self: GeometryModel, *, include_features: bool = True) -> GeometryModel:
        assert self is not geometry
        cloned_sizes.append(len(self.vertices))
        return original_clone(self, include_features=include_features)

    monkeypatch.setattr(GeometryModel, "clone", observed_clone)
    result = pattern_entities(
        geometry,
        (EntityRef("edge", edge),),
        (
            AffineTransform.translation((2.0, 0.0, 0.0)),
            AffineTransform.translation((4.0, 0.0, 0.0)),
            AffineTransform.translation((6.0, 0.0, 0.0)),
        ),
    )

    assert len(result.instances) == 3
    assert cloned_sizes == [2, 2, 2]


def test_rectangular_pattern_feature_regenerates_deterministically() -> None:
    geometry = GeometryModel()
    point = geometry.features.append(
        "geometry.point", parameters={"position": [0.0, 0.0, 0.0]}
    )
    pattern = geometry.features.append(
        "geometry.pattern.rectangular",
        parameters={
            "directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "spacings": [2.0, 3.0],
            "counts": [1, 1],
        },
        inputs={
            "entities": (
                FeatureOutputRef(point.feature_id, "point", "vertex"),
            )
        },
        dependencies=(point.feature_id,),
    )

    report = geometry.regenerate_features()

    assert report.success
    outputs = geometry.features.get(pattern.feature_id).outputs
    assert len(outputs) == 3
    assert sorted(outputs) == [
        "instance/0/vertex/1",
        "instance/1/vertex/1",
        "instance/2/vertex/1",
    ]
    assert geometry.validate_topology() == ()


def test_arbitrary_transform_pattern_feature_uses_serializable_matrices() -> None:
    geometry = GeometryModel()
    point = geometry.features.append(
        "geometry.point", parameters={"position": [1.0, 0.0, 0.0]}
    )
    matrices = [
        AffineTransform.translation((2.0, 0.0, 0.0)).matrix.tolist(),
        AffineTransform.rotation(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0
        ).matrix.tolist(),
    ]
    pattern = geometry.features.append(
        "geometry.pattern.transforms",
        parameters={"matrices": matrices},
        inputs={
            "entities": (
                FeatureOutputRef(point.feature_id, "point", "vertex"),
            )
        },
        dependencies=(point.feature_id,),
    )

    assert geometry.regenerate_features().success
    outputs = geometry.features.get(pattern.feature_id).outputs
    positions = {
        tuple(np.round(geometry.vertex_position(reference.id), 12))
        for reference in outputs.values()
    }
    assert positions == {(3.0, 0.0, 0.0), (0.0, 1.0, 0.0)}
