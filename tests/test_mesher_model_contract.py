from __future__ import annotations

import numpy as np
import pytest

from anygeometry.closure import extract_model_closure
from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.evaluation import (
    edge_tangent_many,
    evaluate_edge_many,
    evaluate_face_many,
    face_derivatives_many,
    face_normal_many,
    project_to_face_many,
)
from anygeometry.model import GeometryModel
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    ConnectionIntent,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
)
from anygeometry.surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface
from anygeometry.tolerance import TolerancePolicy


def _surface_face(geometry: GeometryModel, surface: object) -> int:
    if isinstance(surface, (Cylinder, Cone)):
        lower = tuple(
            geometry.add_point(*surface.evaluate(u, 0.0))
            for u in (0.0, 0.5, 1.0)
        )
        upper = tuple(
            geometry.add_point(*surface.evaluate(u, 1.0))
            for u in (0.0, 0.5, 1.0)
        )
        bottom = geometry.add_arc(lower[0], lower[1], lower[2])
        right = geometry.add_line(lower[2], upper[2])
        top = geometry.add_arc(upper[0], upper[1], upper[2])
        left = geometry.add_line(lower[0], upper[0])
        return geometry.add_face_from_loop(
            (
                OrientedEdge(bottom, True),
                OrientedEdge(right, True),
                OrientedEdge(top, False),
                OrientedEdge(left, False),
            ),
            (0, 1, 2, 3),
            surface=surface,
        )
    corners = tuple(
        geometry.add_point(*surface.evaluate(u, v))
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    edges = geometry.add_polyline(corners, close=True)
    return geometry.add_face(edges, (0, 1, 2, 3), surface=surface)


def test_extract_model_closure_preserves_complete_structural_parents() -> None:
    policy = TolerancePolicy(coincidence=2.0e-8, aabb_padding=3.0e-8)
    geometry = GeometryModel(tolerance=policy)
    geometry.set_document_settings(
        units="mm",
        local_origin=(10.0, 20.0, 30.0),
        coordinate_transform=np.eye(4),
        crs_metadata={"authority": "EPSG", "code": 5973},
    )
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    geometry.add_to_group("mesh_faces", (geometry.faces[face].ref,))
    geometry.tag(geometry.faces[face].ref, "shell", "selected")
    edge = geometry.faces[face].loop[0].edge
    part = geometry.add_part(name="panel")
    sheet = geometry.add_sheet((face,), part_id=part)
    member = geometry.add_member((edge,), part_id=part, name="edge member")
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange.point(0.0)),
        connection_intent=ConnectionIntent.CONNECT,
        part_id=part,
        sheet_id=sheet,
        lineage=(("face", face), ("member", member)),
    )
    geometry.add_junction(
        JunctionKind.OVERLAP,
        (JunctionMemberUse(member, ParameterRange(0.0, 1.0)),),
        sheet_ids=(sheet,),
        attachment_ids=(attachment,),
        connection_intent=ConnectionIntent.CONNECT,
    )

    result = extract_model_closure(geometry, (geometry.handle("face", face),))
    work = result.working_model

    assert result.source_model_id == geometry.model_id
    assert result.source_revision == geometry.revision
    assert result.source_handles == (geometry.handle("face", face),)
    assert work.model_id != geometry.model_id
    assert work.tolerance == policy
    assert work.units == "mm"
    assert work.local_origin == pytest.approx((10.0, 20.0, 30.0))
    assert work.coordinate_transform == pytest.approx(np.eye(4))
    assert work.crs_metadata.to_dict() == {"authority": "EPSG", "code": 5973}
    assert len(work.parts) == len(work.sheets) == len(work.members) == 1
    assert len(work.face_uses) == 1
    assert len(work.coedges) == 4
    assert len(work.member_edge_uses) == 1
    assert len(work.attachments) == len(work.junctions) == 1
    mapped_face = result.source_to_work[geometry.handle("face", face)]
    mapped_member = result.source_to_work[geometry.handle("member", member)]
    assert work.group("mesh_faces") == (work.faces[mapped_face.id].ref,)
    assert work.tags_for(work.faces[mapped_face.id].ref) == ("selected", "shell")
    assert next(iter(work.attachments.values())).lineage == (
        ("face", mapped_face.id),
        ("member", mapped_member.id),
    )
    assert work.validate_topology() == ()
    assert work._validate_structural() == ()  # noqa: SLF001
    for source, made in result.source_to_work.items():
        assert source.model_id == geometry.model_id
        assert made.model_id == work.model_id
        assert result.work_to_source[made] == source
    assert result.source_to_work[geometry.handle("member", member)].kind == "member"


def test_extract_model_closure_maps_active_structural_lineage_parents() -> None:
    geometry = GeometryModel()
    primary_vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    primary_face = geometry.add_plate(primary_vertices)
    primary_edge = geometry.faces[primary_face].loop[0].edge
    primary_part = geometry.add_part(name="primary")
    primary_sheet = geometry.add_sheet((primary_face,), part_id=primary_part)
    primary_member = geometry.add_member((primary_edge,), part_id=primary_part)

    lineage_part = geometry.add_part(name="lineage-only part")

    sheet_vertices = geometry.add_points(
        ((10.0, 0.0, 0.0), (12.0, 0.0, 0.0), (12.0, 1.0, 0.0), (10.0, 1.0, 0.0))
    )
    sheet_face = geometry.add_plate(sheet_vertices)
    sheet_part = geometry.add_part(name="lineage sheet owner")
    lineage_sheet = geometry.add_sheet((sheet_face,), part_id=sheet_part)

    member_start, member_end = geometry.add_points(
        ((20.0, 0.0, 0.0), (21.0, 0.0, 0.0))
    )
    member_edge = geometry.add_line(member_start, member_end)
    member_part = geometry.add_part(name="lineage member owner")
    lineage_member = geometry.add_member((member_edge,), part_id=member_part)

    attachment = geometry.add_attachment(
        primary_member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        primary_face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange.point(0.0)),
        part_id=primary_part,
        sheet_id=primary_sheet,
        lineage=(
            ("part", lineage_part),
            ("sheet", lineage_sheet),
            ("member", lineage_member),
        ),
    )

    result = extract_model_closure(
        geometry, (geometry.handle("face", primary_face),)
    )
    work = result.working_model
    mapped_attachment = result.source_to_work[
        geometry.handle("attachment", attachment)
    ]
    mapped_lineage_part = result.source_to_work[
        geometry.handle("part", lineage_part)
    ]
    mapped_lineage_sheet = result.source_to_work[
        geometry.handle("sheet", lineage_sheet)
    ]
    mapped_lineage_member = result.source_to_work[
        geometry.handle("member", lineage_member)
    ]

    assert work.attachments[mapped_attachment.id].lineage == (
        ("part", mapped_lineage_part.id),
        ("sheet", mapped_lineage_sheet.id),
        ("member", mapped_lineage_member.id),
    )
    assert work.sheets[mapped_lineage_sheet.id].part_id == result.source_to_work[
        geometry.handle("part", sheet_part)
    ].id
    assert work.members[mapped_lineage_member.id].part_id == result.source_to_work[
        geometry.handle("part", member_part)
    ].id
    assert geometry.handle("face", sheet_face) in result.source_to_work
    assert geometry.handle("edge", member_edge) in result.source_to_work
    assert work.validate_topology() == ()
    assert work._validate_structural() == ()  # noqa: SLF001


def test_extract_model_closure_does_not_expand_owner_part_siblings() -> None:
    geometry = GeometryModel()
    part = geometry.add_part(name="shared owner")

    first_vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    first_face = geometry.add_plate(first_vertices)
    first_sheet = geometry.add_sheet((first_face,), part_id=part)
    first_axis_vertices = geometry.add_points(
        ((0.5, 1.0, 0.0), (3.5, 1.0, 0.0))
    )
    first_axis = geometry.add_line(*first_axis_vertices)
    first_member = geometry.add_member((first_axis,), part_id=part)
    geometry.add_attachment(
        first_member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        first_face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange.point(0.5)),
        part_id=part,
        sheet_id=first_sheet,
    )

    second_vertices = geometry.add_points(
        ((10.0, 0.0, 0.0), (14.0, 0.0, 0.0), (14.0, 2.0, 0.0), (10.0, 2.0, 0.0))
    )
    second_face = geometry.add_plate(second_vertices)
    second_sheet = geometry.add_sheet((second_face,), part_id=part)
    second_axis_vertices = geometry.add_points(
        ((10.5, 1.0, 0.0), (13.5, 1.0, 0.0))
    )
    second_axis = geometry.add_line(*second_axis_vertices)
    second_member = geometry.add_member((second_axis,), part_id=part)

    result = extract_model_closure(
        geometry, (geometry.handle("face", first_face),)
    )
    work = result.working_model
    mapped_part = result.source_to_work[geometry.handle("part", part)]
    mapped_sheet = result.source_to_work[geometry.handle("sheet", first_sheet)]
    mapped_member = result.source_to_work[geometry.handle("member", first_member)]

    assert geometry.handle("sheet", second_sheet) not in result.source_to_work
    assert geometry.handle("member", second_member) not in result.source_to_work
    assert geometry.handle("face", second_face) not in result.source_to_work
    assert geometry.handle("edge", second_axis) not in result.source_to_work
    assert work.parts[mapped_part.id].sheet_ids == (mapped_sheet.id,)
    assert work.parts[mapped_part.id].member_ids == (mapped_member.id,)
    assert len(work.sheets) == len(work.members) == 1
    assert work.validate_topology() == ()
    assert work._validate_structural() == ()  # noqa: SLF001


def test_extract_model_closure_rejects_wrong_model_and_feature_request() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)
    other = GeometryModel()
    foreign = other.handle("vertex", other.add_point(0.0, 0.0, 0.0))

    with pytest.raises(GeometryError, match="wrong-model"):
        extract_model_closure(geometry, (foreign,))
    with pytest.raises(GeometryError, match="feature-history closure"):
        extract_model_closure(
            geometry, (geometry.handle("vertex", vertex),), include_features=True
        )
    with pytest.raises(GeometryError, match="positive integer"):
        extract_model_closure(geometry, (("vertex", True),))
    with pytest.raises(GeometryError, match="positive integer"):
        extract_model_closure(geometry, (("vertex", 1.5),))


def test_non_structural_closure_rejects_structural_selection_clearly() -> None:
    geometry = GeometryModel()
    start, end = geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    member = geometry.add_member((geometry.add_line(start, end),))

    with pytest.raises(GeometryError, match="structural selections require"):
        extract_model_closure(
            geometry,
            (geometry.handle("member", member),),
            include_structural_closure=False,
        )


def test_vectorized_edge_contract_shapes_and_values() -> None:
    geometry = GeometryModel()
    start, end = geometry.add_points(((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)))
    edge = geometry.add_line(start, end)
    parameters = np.asarray((0.0, 0.25, 1.0))

    assert evaluate_edge_many(geometry, edge, parameters) == pytest.approx(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (4.0, 0.0, 0.0)))
    )
    assert edge_tangent_many(geometry, edge, parameters) == pytest.approx(
        np.tile((1.0, 0.0, 0.0), (3, 1))
    )
    assert geometry.evaluate_edge_many(edge, np.empty(0)).shape == (0, 3)


def test_vectorized_edge_contract_covers_arc_spline_and_strict_ids() -> None:
    geometry = GeometryModel()
    arc_points = geometry.add_points(((1, 0, 0), (1, 1, 0), (0, 1, 0)))
    arc = geometry.add_arc(*arc_points)
    spline_points = geometry.add_points(((0, 0, 1), (1, 1, 1), (2, 0, 1)))
    spline = geometry.add_spline(
        spline_points[0], (spline_points[1],), spline_points[2]
    )
    parameters = np.asarray((0.0, 0.25, 0.75, 1.0))

    for edge_id in (arc, spline):
        assert geometry.evaluate_edge_many(edge_id, parameters) == pytest.approx(
            geometry.sample_edge(edge_id, parameters)
        )
        tangents = geometry.edge_tangent_many(edge_id, parameters)
        assert np.linalg.norm(tangents, axis=1) == pytest.approx(np.ones(4))
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.evaluate_edge_many(True, parameters)
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.edge_tangent_many(1.0, parameters)


@pytest.mark.parametrize(
    "surface",
    (
        Plane(np.zeros(3), np.asarray((2.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))),
        Cylinder(
            np.zeros(3), np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)), 2.0, 3.0, sweep_angle=1.2,
        ),
        Cone(
            np.zeros(3), np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)), 2.0, 1.0, 3.0, sweep_angle=1.2,
        ),
        RuledSurface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            np.asarray(((0.0, 1.0, 0.5), (2.0, 1.0, 0.5))),
        ),
        CoonsSurface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            np.asarray(((2.0, 0.0, 0.0), (2.0, 1.0, 0.5))),
            np.asarray(((0.0, 1.0, 0.5), (2.0, 1.0, 0.5))),
            np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.5))),
        ),
    ),
    ids=("plane", "cylinder", "cone", "ruled", "coons"),
)
def test_vectorized_face_contract_for_supported_surfaces(surface: object) -> None:
    geometry = GeometryModel()
    face = _surface_face(geometry, surface)
    uv = np.asarray(((0.0, 0.0), (0.27, 0.41), (1.0, 1.0)))

    points = evaluate_face_many(geometry, face, uv)
    expected = np.asarray([surface.evaluate(*value) for value in uv])
    assert points.shape == (3, 3)
    assert points == pytest.approx(expected, abs=2.0e-10)
    du, dv = face_derivatives_many(geometry, face, uv)
    normals = face_normal_many(geometry, face, uv)
    assert du.shape == dv.shape == normals.shape == (3, 3)
    assert np.linalg.norm(normals, axis=1) == pytest.approx(np.ones(3))
    projected, made_uv, distances = project_to_face_many(
        geometry, face, points, initial_uv=uv
    )
    assert projected == pytest.approx(points, abs=2.0e-7)
    assert made_uv == pytest.approx(uv, abs=2.0e-5)
    assert distances == pytest.approx(np.zeros(3), abs=2.0e-7)


def test_parameterization_is_distinct_from_authoritative_support() -> None:
    geometry = GeometryModel()
    support = Plane(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
    )
    mapping = Plane(
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
    )
    face = _surface_face(geometry, support)
    geometry.set_face_parameterization(face, mapping)

    assert geometry.faces[face].support_surface is support
    assert geometry.face_support_point(face, 0.25, 0.5) == pytest.approx(
        (0.5, 0.5, 0.0)
    )
    assert geometry.evaluate_face_many(face, ((0.25, 0.5),))[0] == pytest.approx(
        (0.5, 0.5, 1.0)
    )
    projected, uv, distance = geometry.project_to_face_many(
        face, ((0.5, 0.5, 2.0),)
    )
    assert projected[0] == pytest.approx((0.5, 0.5, 1.0))
    assert uv[0] == pytest.approx((0.25, 0.5))
    assert distance == pytest.approx((1.0,))


def test_projection_seed_cannot_worsen_half_cylinder_solution() -> None:
    geometry = GeometryModel()
    surface = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        3.0,
        sweep_angle=np.pi,
    )
    face = _surface_face(geometry, surface)
    target = surface.evaluate(0.0, 0.5)

    projected, uv, distances = geometry.project_to_face_many(
        face, (target,), initial_uv=((1.0, 0.5),)
    )

    assert projected[0] == pytest.approx(target)
    assert uv[0] == pytest.approx((0.0, 0.5))
    assert distances == pytest.approx((0.0,))


def test_cone_projection_is_orthogonal_and_seed_independent() -> None:
    geometry = GeometryModel()
    surface = Cone(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        1.0,
        3.0,
        sweep_angle=np.pi,
    )
    face = _surface_face(geometry, surface)
    target = np.asarray((3.0, 0.0, 1.5))

    unseeded = geometry.project_to_face_many(face, (target,))
    seeded = geometry.project_to_face_many(
        face, (target,), initial_uv=((0.8, 0.9),)
    )

    assert unseeded[1][0] == pytest.approx((0.0, 0.35), abs=1.0e-12)
    assert unseeded[2][0] == pytest.approx(np.sqrt(2.025), abs=1.0e-12)
    assert seeded[0] == pytest.approx(unseeded[0])
    assert seeded[1] == pytest.approx(unseeded[1])
    assert seeded[2] == pytest.approx(unseeded[2])


def test_topology_coons_derivatives_survive_large_translation() -> None:
    geometry = GeometryModel()
    offset = 1.0e12
    face = geometry.add_plate(
        geometry.add_points(
            (
                (offset, offset, 0.0),
                (offset + 1.0, offset, 0.0),
                (offset + 1.0, offset + 1.0, 0.0),
                (offset, offset + 1.0, 0.0),
            )
        )
    )
    geometry.set_face_surface(face, CoonsSurface())

    uv = np.asarray(((0.25, 0.75),))
    du, dv = geometry.face_derivatives_many(face, uv)
    normals = geometry.face_normal_many(face, uv)

    assert du[0] == pytest.approx((1.0, 0.0, 0.0))
    assert dv[0] == pytest.approx((0.0, 1.0, 0.0))
    assert normals[0] == pytest.approx((0.0, 0.0, 1.0))


def test_face_batch_contract_accepts_empty_and_rejects_noninteger_id() -> None:
    geometry = GeometryModel()
    face = _surface_face(
        geometry,
        Plane(
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )

    assert geometry.evaluate_face_many(face, []).shape == (0, 3)
    assert geometry.face_normal_many(face, []).shape == (0, 3)
    projected, uv, distances = geometry.project_to_face_many(face, [])
    assert projected.shape == (0, 3)
    assert uv.shape == (0, 2)
    assert distances.shape == (0,)
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.evaluate_face_many(True, ((0.5, 0.5),))
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.project_to_face_many(1.0, ((0.0, 0.0, 0.0),))


@pytest.mark.parametrize(
    "surface, uv",
    (
        (
            Plane(
                np.asarray((2.0, -1.0, 0.5)),
                np.asarray((3.0, 0.0, 0.0)),
                np.asarray((0.5, 2.0, 0.0)),
            ),
            np.asarray(((0.1, 0.2), (0.45, 0.7), (0.9, 0.35))),
        ),
        (
            Cylinder(
                np.asarray((0.5, -0.25, 1.0)),
                np.asarray((0.0, 0.0, 1.0)),
                np.asarray((1.0, 0.0, 0.0)),
                2.0,
                4.0,
                start_angle=-0.4,
                sweep_angle=1.8,
            ),
            np.asarray(((0.05, 0.2), (0.5, 0.65), (0.95, 0.4))),
        ),
        (
            Cone(
                np.asarray((-0.5, 0.25, -1.0)),
                np.asarray((0.0, 0.0, 1.0)),
                np.asarray((1.0, 0.0, 0.0)),
                2.5,
                0.75,
                5.0,
                start_angle=0.2,
                sweep_angle=1.5,
            ),
            np.asarray(((0.1, 0.25), (0.55, 0.7), (0.9, 0.45))),
        ),
    ),
    ids=("plane", "cylinder", "cone"),
)
def test_builtin_projection_is_batched_and_builds_trim_once(
    monkeypatch: pytest.MonkeyPatch,
    surface: Plane | Cylinder | Cone,
    uv: np.ndarray,
) -> None:
    geometry = GeometryModel()
    face = _surface_face(geometry, surface)
    points = np.asarray([surface.evaluate(*value) for value in uv])
    points += np.asarray((0.2, -0.15, 0.35))
    expected = tuple(geometry.project_to_face(face, point) for point in points)

    calls = 0
    original = GeometryModel.face_trim_loops_uv

    def counted(self: GeometryModel, face_id: int, **kwargs: object) -> tuple[np.ndarray, ...]:
        nonlocal calls
        calls += 1
        return original(self, face_id, **kwargs)

    monkeypatch.setattr(GeometryModel, "face_trim_loops_uv", counted)
    projected, made_uv, distances = geometry.project_to_face_many(
        face,
        points,
        initial_uv=np.full((len(points), 2), 0.95),
    )

    assert calls == 1
    assert projected == pytest.approx(np.asarray([item[0] for item in expected]))
    assert made_uv == pytest.approx(np.asarray([item[1] for item in expected]))
    assert distances == pytest.approx(np.asarray([item[2] for item in expected]))


@pytest.mark.parametrize(
    ("sweep", "outside_angle"),
    ((0.5, -0.1), (-0.5, 0.1)),
)
def test_partial_cylinder_projection_selects_nearest_angular_endpoint(
    sweep: float, outside_angle: float
) -> None:
    geometry = GeometryModel()
    surface = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        3.0,
        start_angle=0.0,
        sweep_angle=sweep,
    )
    face = _surface_face(geometry, surface)
    target = np.asarray(
        (2.0 * np.cos(outside_angle), 2.0 * np.sin(outside_angle), 1.5)
    )

    projected, uv, distances = geometry.project_to_face_many(face, (target,))

    assert uv[0] == pytest.approx((0.0, 0.5))
    assert projected[0] == pytest.approx(surface.evaluate(0.0, 0.5))
    assert distances[0] == pytest.approx(
        np.linalg.norm(target - surface.evaluate(0.0, 0.5))
    )


def test_batched_plane_projection_survives_large_translation() -> None:
    offset = 1.0e12
    surface = Plane(
        np.asarray((offset, offset, offset)),
        np.asarray((4.0, 0.0, 0.0)),
        np.asarray((1.0, 2.0, 0.0)),
    )
    geometry = GeometryModel()
    face = _surface_face(geometry, surface)
    expected_uv = np.asarray(((0.2, 0.3), (0.6, 0.8), (0.9, 0.1)))
    expected = np.asarray([surface.evaluate(*value) for value in expected_uv])
    targets = expected + np.asarray((0.0, 0.0, 7.0))

    projected, uv, distances = geometry.project_to_face_many(
        face, targets, initial_uv=np.zeros_like(expected_uv)
    )

    assert projected == pytest.approx(expected, abs=2.0e-4)
    assert uv == pytest.approx(expected_uv, abs=5.0e-5)
    assert distances == pytest.approx(np.full(3, 7.0), abs=2.0e-4)


def test_bulk_bounds_reject_non_integer_identity() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)
    assert geometry.entity_bounds_many((("vertex", vertex),)) == (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.entity_bounds_many((("vertex", True),))
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.entity_bounds_many((("vertex", 1.5),))

    second = geometry.add_point(1.0, 0.0, 0.0)
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.add_lines(((True, second),))
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.add_lines(((1.9, second),))
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.add_member((True,))
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.add_faces(((True, 1, 2),))


def test_direct_sheet_attachment_qualifies_member_sheet_junction() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    part = geometry.add_part()
    sheet = geometry.add_sheet((face,), part_id=part)
    member = geometry.add_member((geometry.faces[face].loop[0].edge,), part_id=part)
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_SHEET,
        AttachmentTargetKind.SHEET,
        sheet,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange.point(0.0)),
        sheet_id=sheet,
    )
    geometry.add_junction(
        JunctionKind.OVERLAP,
        (JunctionMemberUse(member, ParameterRange(0.0, 1.0)),),
        sheet_ids=(sheet,),
        attachment_ids=(attachment,),
    )
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_topology_snapshot_restores_new_semantic_state_without_identity_loss() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    mapping = Plane(
        np.asarray((0.0, 0.0, 0.25)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
    )
    geometry.set_face_parameterization(face, mapping)
    part = geometry.add_part()
    geometry.mark_construction_vertices((vertices[3],), part_id=part)
    edge = geometry.faces[face].loop[0].edge
    old_member = geometry.add_member((edge,), part_id=part)
    stale = geometry.handle("member", old_member)
    descendants = geometry.split_member(old_member, 0.5)
    snapshot = geometry.topology_snapshot()

    geometry.set_face_parameterization(face, None)
    geometry.unmark_construction_vertices((vertices[3],))
    geometry._structural_replacement_history.clear()  # noqa: SLF001 - hostile undo case
    revision = geometry.revision
    geometry.restore_topology(snapshot)

    assert geometry.revision == revision + 1
    restored_mapping = geometry.faces[face].parameterization
    assert isinstance(restored_mapping, Plane)
    assert restored_mapping.origin == pytest.approx(mapping.origin)
    assert restored_mapping.u_vector == pytest.approx(mapping.u_vector)
    assert restored_mapping.v_vector == pytest.approx(mapping.v_vector)
    assert geometry.construction_owner(vertices[3]) == part
    assert geometry.resolve_handle(stale).status.value == "replaced"
    assert tuple(value.id for value in geometry.resolve_handle(stale).resolved) == descendants
    assert geometry.structural_replacement_history() == {
        ("member", old_member): tuple(("member", value) for value in descendants)
    }
