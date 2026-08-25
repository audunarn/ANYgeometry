"""Model-integrated strict audit qualification tests."""

from __future__ import annotations

from anygeometry.audit import AuditCode
from anygeometry.model import GeometryModel
from anygeometry.strict_audit import strict_audit
from anygeometry.generators import cylinder
from anygeometry.tolerance import TolerancePolicy
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
)


def _codes(model: GeometryModel) -> set[AuditCode]:
    return {issue.code for issue in strict_audit(model).issues}


def test_clean_plate_is_deterministic_certifiable_and_fully_accounted() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    geometry.add_plate(vertices)

    first = strict_audit(geometry)
    second = strict_audit(geometry)

    assert first.clean and first.certifiable
    assert first.metrics.classification_complete
    assert first.metrics.narrow_phase_tests == first.metrics.candidate_count
    assert first.to_dict() == second.to_dict()


def test_duplicate_vertex_crossing_t_junction_and_overlap_are_detected() -> None:
    duplicate = GeometryModel()
    duplicate.add_points(((3.0, 2.0, 1.0), (3.0, 2.0, 1.0)))
    assert AuditCode.VERTEX_COINCIDENCE in _codes(duplicate)

    crossing = GeometryModel()
    vertices = crossing.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))
    )
    crossing.add_line(vertices[0], vertices[1])
    crossing.add_line(vertices[2], vertices[3])
    assert AuditCode.EDGE_CROSSING in _codes(crossing)

    t_junction = GeometryModel()
    vertices = t_junction.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    )
    t_junction.add_line(vertices[0], vertices[1])
    t_junction.add_line(vertices[2], vertices[3])
    assert AuditCode.VERTEX_EDGE_T_JUNCTION in _codes(t_junction)

    overlap = GeometryModel()
    vertices = overlap.add_points(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    overlap.add_line(vertices[0], vertices[1])
    overlap.add_line(vertices[2], vertices[3])
    assert AuditCode.EDGE_COLLINEAR_OVERLAP in _codes(overlap)


def test_classification_is_unchanged_by_a_large_translation() -> None:
    def made(offset: float) -> GeometryModel:
        geometry = GeometryModel()
        vertices = geometry.add_points(
            (
                (offset + 0.0, offset + 0.0, offset),
                (offset + 2.0, offset + 0.0, offset),
                (offset + 1.0, offset - 1.0, offset),
                (offset + 1.0, offset + 1.0, offset),
            )
        )
        geometry.add_line(vertices[0], vertices[1])
        geometry.add_line(vertices[2], vertices[3])
        return geometry

    origin = strict_audit(made(0.0))
    translated = strict_audit(made(1.0e12))

    assert origin.issue_counts == translated.issue_counts == {"edge_crossing": 1}
    assert origin.metrics == translated.metrics


def test_classification_is_equivalent_in_metres_and_millimetres() -> None:
    def made(scale: float, units: str) -> GeometryModel:
        geometry = GeometryModel(tolerance=TolerancePolicy().scaled(scale))
        geometry.set_document_settings(units=units)
        vertices = geometry.add_points(
            (
                (0.0 * scale, 0.0 * scale, 0.0),
                (2.0 * scale, 0.0 * scale, 0.0),
                (1.0 * scale, -1.0 * scale, 0.0),
                (1.0 * scale, 1.0 * scale, 0.0),
            )
        )
        geometry.add_line(vertices[0], vertices[1])
        geometry.add_line(vertices[2], vertices[3])
        return geometry

    metres = strict_audit(made(1.0, "m"))
    millimetres = strict_audit(made(1000.0, "mm"))

    assert metres.issue_counts == millimetres.issue_counts == {"edge_crossing": 1}
    assert metres.metrics == millimetres.metrics


def test_member_overlap_needs_junction_but_declared_or_separate_is_intentional() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    first_edge = geometry.add_line(vertices[0], vertices[1])
    second_edge = geometry.add_line(vertices[2], vertices[3])
    part = geometry.add_part(name="frame")
    first_member = geometry.add_member((first_edge,), part_id=part)
    second_member = geometry.add_member((second_edge,), part_id=part)

    missing = strict_audit(geometry)
    assert AuditCode.MEMBER_MEMBER_OVERLAP in {issue.code for issue in missing.issues}
    assert not missing.clean

    geometry.add_junction(
        JunctionKind.OVERLAP,
        (
            JunctionMemberUse(first_member, ParameterRange(0.0, 1.0)),
            JunctionMemberUse(second_member, ParameterRange(0.0, 1.0)),
        ),
    )
    declared = strict_audit(geometry)
    assert declared.clean and declared.certifiable
    assert declared.issue_counts == {"intentional_coincidence": 3}

    separate = GeometryModel()
    vertices = separate.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    first_edge = separate.add_line(vertices[0], vertices[1])
    second_edge = separate.add_line(vertices[2], vertices[3])
    separate.add_member((first_edge,), part_id=separate.add_part())
    separate.add_member((second_edge,), part_id=separate.add_part())
    report = strict_audit(separate)
    assert report.clean and report.certifiable
    assert set(report.issue_counts) == {"intentional_coincidence"}


def test_member_face_embedding_requires_a_qualified_attachment() -> None:
    geometry = GeometryModel()
    plate_vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    face = geometry.add_plate(plate_vertices)
    axis_vertices = geometry.add_points(((0.2, 1.0, 0.0), (1.8, 1.0, 0.0)))
    edge = geometry.add_line(*axis_vertices)
    member = geometry.add_member((edge,))

    report = strict_audit(geometry)
    assert AuditCode.MEMBER_FACE_EMBEDDED in {issue.code for issue in report.issues}

    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.1, 0.9), ParameterRange.point(0.5)),
    )
    attached = strict_audit(geometry)
    assert AuditCode.MEMBER_FACE_EMBEDDED not in {issue.code for issue in attached.issues}
    assert AuditCode.INTENTIONAL_COINCIDENCE in {issue.code for issue in attached.issues}


def test_coplanar_overlap_and_transverse_face_crossing_are_qualified() -> None:
    coplanar = GeometryModel()
    first = coplanar.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    second = coplanar.add_points(
        ((1.0, 1.0, 0.0), (3.0, 1.0, 0.0), (3.0, 3.0, 0.0), (1.0, 3.0, 0.0))
    )
    coplanar.add_plate(first)
    coplanar.add_plate(second)
    assert AuditCode.FACE_COPLANAR_OVERLAP in _codes(coplanar)

    crossing = GeometryModel()
    horizontal = crossing.add_points(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
    )
    vertical = crossing.add_points(
        ((0.0, -1.0, -1.0), (0.0, 1.0, -1.0), (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))
    )
    crossing.add_plate(horizontal)
    crossing.add_plate(vertical)
    assert AuditCode.FACE_FACE_CROSSING in _codes(crossing)


def test_non_manifold_and_corrupt_structural_ownership_are_not_clean() -> None:
    non_manifold = GeometryModel()
    first, second, upper, lower, transverse = non_manifold.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    )
    common = non_manifold.add_line(first, second)
    for third in (upper, lower, transverse):
        second_side = non_manifold.add_line(second, third)
        third_side = non_manifold.add_line(third, first)
        non_manifold.add_face((common, second_side, third_side))
    assert AuditCode.SHEET_NON_MANIFOLD in _codes(non_manifold)

    corrupt = GeometryModel()
    first, second = corrupt.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    edge = corrupt.add_line(first, second)
    member = corrupt.add_member((edge,))
    use_id = corrupt.members[member].edge_use_ids[0]
    # Simulate a corrupted document below the public mutation layer.
    corrupt._structural_store("member_edge_use").pop(use_id)  # noqa: SLF001
    report = strict_audit(corrupt)
    assert not report.clean
    assert AuditCode.UNOWNED_STRUCTURAL_USE in {issue.code for issue in report.issues}


def test_spline_candidate_uses_the_shared_qualified_predicate() -> None:
    geometry = GeometryModel()
    (
        first_start,
        first_control,
        first_end,
        second_start,
        second_control,
        second_end,
    ) = geometry.add_points(
        (
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (1.0, 0.5, 0.0),
        )
    )
    geometry.add_spline(first_start, (first_control,), first_end)
    geometry.add_spline(second_start, (second_control,), second_end)

    report = strict_audit(geometry)

    assert not report.clean and not report.certifiable
    assert report.metrics.classification_complete
    assert report.metrics.unclassified_count == 0
    assert AuditCode.EDGE_CROSSING in {issue.code for issue in report.issues}


def test_declared_junction_and_attachment_geometry_are_verified() -> None:
    junction_model = GeometryModel()
    vertices = junction_model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 1.0, 0.0))
    )
    first_edge = junction_model.add_line(vertices[0], vertices[1])
    second_edge = junction_model.add_line(vertices[2], vertices[3])
    part = junction_model.add_part()
    first_member = junction_model.add_member((first_edge,), part_id=part)
    second_member = junction_model.add_member((second_edge,), part_id=part)
    junction_model.add_junction(
        JunctionKind.CROSSING,
        (
            JunctionMemberUse(first_member, ParameterRange.point(0.5)),
            JunctionMemberUse(second_member, ParameterRange.point(0.5)),
        ),
    )
    junction_report = strict_audit(junction_model)
    assert AuditCode.JUNCTION_INCONSISTENT in {
        issue.code for issue in junction_report.issues
    }

    attached = GeometryModel()
    plate_vertices = attached.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    face = attached.add_plate(plate_vertices)
    axis_vertices = attached.add_points(((0.2, 1.0, 0.0), (1.8, 1.0, 0.0)))
    edge = attached.add_line(*axis_vertices)
    member = attached.add_member((edge,))
    attachment = attached.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )
    attachment_report = strict_audit(attached)
    assert any(
        issue.code is AuditCode.ATTACHMENT_INCONSISTENT
        and any(
            entity.kind == "attachment" and entity.id == attachment
            for entity in issue.entities
        )
        for issue in attachment_report.issues
    )


def test_declared_junction_applies_only_at_its_member_parameters() -> None:
    geometry = GeometryModel()
    horizontal_vertices = geometry.add_points(((0.0, 0.0, 0.0), (6.0, 0.0, 0.0)))
    horizontal_edge = geometry.add_line(*horizontal_vertices)
    weaving_vertices = geometry.add_points(
        (
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (3.0, 1.0, 0.0),
            (3.0, -1.0, 0.0),
            (5.0, -1.0, 0.0),
            (5.0, 1.0, 0.0),
        )
    )
    weaving_edges = geometry.add_polyline(weaving_vertices)
    part = geometry.add_part()
    horizontal = geometry.add_member((horizontal_edge,), part_id=part)
    weaving = geometry.add_member(weaving_edges, part_id=part)
    geometry.add_junction(
        JunctionKind.CROSSING,
        (
            JunctionMemberUse(horizontal, ParameterRange.point(0.5)),
            JunctionMemberUse(weaving, ParameterRange.point(0.5)),
        ),
    )

    report = strict_audit(geometry)
    crossings = [
        issue
        for issue in report.issues
        if issue.code is AuditCode.MEMBER_MEMBER_CROSSING
    ]
    intentional = [
        issue
        for issue in report.issues
        if issue.code is AuditCode.INTENTIONAL_COINCIDENCE
        and dict(issue.details).get("intent") == '"declared_junction"'
    ]

    assert not report.clean and not report.certifiable
    assert len(crossings) == 2
    assert {round(issue.witnesses[0].point[0]) for issue in crossings} == {1, 5}
    assert len(intentional) == 1
    assert round(intentional[0].witnesses[0].point[0]) == 3
    assert report.metrics.classification_complete


def test_overlap_junction_is_proved_at_every_piecewise_straight_breakpoint() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(
        ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0))
    )
    second_vertices = geometry.add_points(
        ((0, 0, 0), (1, 1, 0), (2, 0, 0), (3, -1, 0), (4, 0, 0))
    )
    first_edges = geometry.add_polyline(first_vertices)
    second_edges = geometry.add_polyline(second_vertices)
    part = geometry.add_part()
    first = geometry.add_member(first_edges, part_id=part)
    second = geometry.add_member(second_edges, part_id=part)
    junction = geometry.add_junction(
        JunctionKind.OVERLAP,
        (
            JunctionMemberUse(first, ParameterRange(0.0, 1.0)),
            JunctionMemberUse(second, ParameterRange(0.0, 1.0)),
        ),
    )

    report = strict_audit(geometry)

    assert not report.clean and not report.certifiable
    assert any(
        issue.code is AuditCode.JUNCTION_INCONSISTENT
        and any(entity.kind == "junction" and entity.id == junction for entity in issue.entities)
        for issue in report.issues
    )


def test_junction_tolerance_ignores_unrelated_distant_geometry() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (1.0, 0.5, 0.0),
            (1.0e12, 0.0, 0.0),
        )
    )
    first_edge = geometry.add_line(vertices[0], vertices[1])
    second_edge = geometry.add_line(vertices[2], vertices[3])
    part = geometry.add_part()
    first_member = geometry.add_member((first_edge,), part_id=part)
    second_member = geometry.add_member((second_edge,), part_id=part)
    junction = geometry.add_junction(
        JunctionKind.OVERLAP,
        (
            JunctionMemberUse(first_member, ParameterRange(0.0, 1.0)),
            JunctionMemberUse(second_member, ParameterRange(0.0, 1.0)),
        ),
    )

    report = strict_audit(geometry)

    assert not report.clean and not report.certifiable
    assert any(
        issue.code is AuditCode.JUNCTION_INCONSISTENT
        and any(
            entity.kind == "junction" and entity.id == junction
            for entity in issue.entities
        )
        for issue in report.issues
    )


def test_junction_mismatch_cannot_inflate_its_own_tolerance_scale() -> None:
    policy = TolerancePolicy(relative_length=1.0e-3)
    geometry = GeometryModel(tolerance=policy)
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0e6, 0.5, 0.0),
            (1.0e6 + 1.0, 0.5, 0.0),
        )
    )
    first_edge = geometry.add_line(vertices[0], vertices[1])
    second_edge = geometry.add_line(vertices[2], vertices[3])
    part = geometry.add_part()
    first_member = geometry.add_member((first_edge,), part_id=part)
    second_member = geometry.add_member((second_edge,), part_id=part)
    geometry.add_junction(
        JunctionKind.OVERLAP,
        (
            JunctionMemberUse(first_member, ParameterRange(0.0, 1.0)),
            JunctionMemberUse(second_member, ParameterRange(0.0, 1.0)),
        ),
    )

    report = strict_audit(geometry)

    assert not report.clean and not report.certifiable
    assert AuditCode.JUNCTION_INCONSISTENT in _codes(geometry)


def test_two_arc_crossing_components_accept_two_point_junctions() -> None:
    geometry = GeometryModel()
    first_points = geometry.add_points(((0, -2, 0), (2, 0, 0), (0, 2, 0)))
    second_points = geometry.add_points(((1, -2, 0), (-1, 0, 0), (1, 2, 0)))
    first_edge = geometry.add_arc(*first_points)
    second_edge = geometry.add_arc(*second_points)
    part = geometry.add_part()
    first_member = geometry.add_member((first_edge,), part_id=part)
    second_member = geometry.add_member((second_edge,), part_id=part)
    for y in (-float(15 ** 0.5) / 2.0, float(15 ** 0.5) / 2.0):
        witness = (0.5, y, 0.0)
        _first_point, first_parameter, first_distance = geometry.closest_edge_point(
            first_edge, witness
        )
        _second_point, second_parameter, second_distance = geometry.closest_edge_point(
            second_edge, witness
        )
        assert max(first_distance, second_distance) < 1.0e-9
        geometry.add_junction(
            JunctionKind.CROSSING,
            (
                JunctionMemberUse(
                    first_member, ParameterRange.point(first_parameter)
                ),
                JunctionMemberUse(
                    second_member, ParameterRange.point(second_parameter)
                ),
            ),
        )

    report = strict_audit(geometry)

    assert report.clean and report.certifiable
    assert report.issue_counts == {"intentional_coincidence": 2}
    assert report.metrics.classification_complete


def test_loose_edge_face_candidates_and_far_attachments_cannot_certify_clean() -> None:
    geometry = GeometryModel()
    plate_vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        )
    )
    face = geometry.add_plate(plate_vertices)
    crossing_vertices = geometry.add_points(
        ((1.0, 1.0, -1.0), (1.0, 1.0, 1.0))
    )
    geometry.add_line(*crossing_vertices)
    far_vertices = geometry.add_points(((10.0, 10.0, 0.0), (11.0, 10.0, 0.0)))
    far_edge = geometry.add_line(*far_vertices)
    far_member = geometry.add_member((far_edge,))
    geometry.add_attachment(
        far_member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange(0.0, 1.0)),
    )

    report = strict_audit(geometry)
    codes = {issue.code for issue in report.issues}

    assert not report.clean and not report.certifiable
    assert AuditCode.NONCONFORMAL_INTERFACE in codes
    assert AuditCode.ATTACHMENT_INCONSISTENT in codes
    assert report.metrics.classification_complete


def test_coons_interior_outside_boundary_aabb_is_qualified_by_shared_predicate() -> None:
    geometry = GeometryModel()
    boundary = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.5, 1.0),
            (1.0, 1.0, 0.0),
            (0.5, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.5, 1.0),
        )
    )
    edges = geometry.add_polyline(boundary, close=True)
    face = geometry.add_face(edges, corners=(0, 2, 4, 6))
    assert geometry.face_point(face, 0.5, 0.5)[2] == 2.0
    crossing = geometry.add_points(((0.5, 0.5, 1.5), (0.5, 0.5, 2.5)))
    crossing_edge = geometry.add_line(*crossing)

    report = strict_audit(geometry)

    assert not report.clean and not report.certifiable
    assert any(
        issue.code is AuditCode.NONCONFORMAL_INTERFACE
        and any(
            entity.kind == "edge" and entity.id == crossing_edge
            for entity in issue.entities
        )
        for issue in report.issues
    )
    assert report.metrics.classification_complete


def test_shared_endpoint_curves_and_generated_cylinders_are_conformal() -> None:
    curves = GeometryModel()
    start, first_via, first_end, second_via, second_end = curves.add_points(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (1.5, 0.5, 0.5),
            (2.0, 0.0, 1.0),
        )
    )
    curves.add_arc(start, first_via, first_end)
    curves.add_arc(first_end, second_via, second_end)
    curve_report = strict_audit(curves)
    assert curve_report.clean and curve_report.certifiable
    assert curve_report.metrics.classification_complete

    small = cylinder(
        2.0,
        3.0,
        circumferential_segments=8,
        ring_spacing=1.0,
    )
    large = cylinder(
        2.0,
        3.0,
        circumferential_segments=16,
        ring_spacing=1.0,
    )
    small_report = strict_audit(small)
    large_report = strict_audit(large)

    assert small_report.clean and small_report.certifiable
    assert large_report.clean and large_report.certifiable
    assert small_report.issue_counts == large_report.issue_counts == {}
    assert small_report.metrics.classification_complete
    assert large_report.metrics.classification_complete
    assert large_report.metrics.candidate_count <= 3 * small_report.metrics.candidate_count
    assert large_report.metrics.index_leaf_tests <= 3 * small_report.metrics.index_leaf_tests


def test_sparse_audit_candidates_are_materially_subquadratic() -> None:
    count = 128
    geometry = GeometryModel()
    vertices = geometry.add_points(
        tuple((4.0 * index, 0.0, 0.0) for index in range(2 * count))
    )
    for index in range(count):
        geometry.add_line(vertices[2 * index], vertices[2 * index + 1])

    report = strict_audit(geometry)
    entity_count = len(geometry.vertices) + len(geometry.edges)
    naive_pairs = entity_count * (entity_count - 1) // 2

    assert report.certifiable
    assert report.metrics.classification_complete
    assert report.metrics.candidate_count * 20 < naive_pairs
    assert report.metrics.index_leaf_tests * 20 < naive_pairs
