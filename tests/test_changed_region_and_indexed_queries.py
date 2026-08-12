"""Gap-closure qualification for local audit and indexed public queries."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import EntityRef, GeometryError, GeometryModel, Plane
from anygeometry.audit import AuditCode, AuditEvidenceQuality, AuditScope
from anygeometry.operations import closest_point, transform
from anygeometry.overlaps import find_coplanar_overlaps
from anygeometry.spatial import AABB, AABBTree
from anygeometry.strict_audit import audit_changed_region, strict_audit
from anygeometry.tolerance import TolerancePolicy
from anygeometry.structural import (
    AttachmentEvidence,
    AttachmentKind,
    AttachmentTargetKind,
    ParameterRange,
)


def _plate(
    geometry: GeometryModel,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z: float = 0.0,
) -> int:
    points = geometry.add_points(
        ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))
    )
    return geometry.add_plate(points)


def test_changed_region_is_local_and_never_certifies_full_model() -> None:
    geometry = GeometryModel()
    geometry.add_point(0.0, 0.0, 0.0)
    geometry.add_point(0.0, 0.0, 0.0)

    report = audit_changed_region(geometry, geometry.last_change_set)

    assert report.scope is AuditScope.CHANGED_REGION
    assert not report.certifiable
    assert AuditCode.VERTEX_COINCIDENCE in {issue.code for issue in report.issues}
    assert report.metrics.candidate_count == report.metrics.classified_count


def test_changed_region_vertex_face_relation_has_typed_evidence() -> None:
    geometry = GeometryModel()
    face = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    vertex = geometry.add_point(1.0, 1.0, 0.0)

    report = audit_changed_region(geometry, geometry.last_change_set)
    issue = next(
        item for item in report.issues if item.code is AuditCode.VERTEX_FACE_INTERIOR
    )

    assert {entity.kind for entity in issue.entities} == {"face", "vertex"}
    assert {entity.id for entity in issue.entities} == {face, vertex}
    assert issue.measured_gap == 0.0
    assert issue.tolerance_used is not None
    assert issue.evidence_quality is AuditEvidenceQuality.EXACT
    assert issue.blocks_strict_handoff


def test_stale_change_set_fails_closed_without_full_certification() -> None:
    geometry = GeometryModel()
    geometry.add_point(0.0, 0.0, 0.0)
    stale = geometry.last_change_set
    geometry.add_point(5.0, 0.0, 0.0)

    report = audit_changed_region(geometry, stale)

    assert not report.completed
    assert not report.clean
    assert not report.certifiable
    assert AuditCode.CHECK_FAILED in {issue.code for issue in report.issues}


def test_orphan_control_callback_is_qualified_fail_closed() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)
    geometry.mark_construction_vertices((vertex,))

    report = strict_audit(geometry)

    issue = next(
        item
        for item in report.issues
        if item.code is AuditCode.ORPHAN_CONTROL_GEOMETRY
    )
    assert issue.entities[0].id == vertex
    assert issue.blocks_strict_handoff


def test_part_owned_construction_vertex_is_not_an_orphan_control() -> None:
    geometry = GeometryModel()
    part = geometry.add_part(name="reference owner")
    vertex = geometry.add_point(0.0, 0.0, 0.0)
    geometry.mark_construction_vertices((vertex,), part_id=part)

    report = strict_audit(geometry)

    assert not any(
        issue.code is AuditCode.ORPHAN_CONTROL_GEOMETRY
        and any(entity.kind == "vertex" and entity.id == vertex for entity in issue.entities)
        for issue in report.issues
    )


def test_face_qualification_uses_support_not_optional_parameterization() -> None:
    geometry = GeometryModel()
    face = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    geometry.set_face_parameterization(
        face,
        Plane(
            np.asarray((100.0, 100.0, 10.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 2.0, 0.0)),
        ),
    )
    vertex = geometry.add_point(1.0, 1.0, 0.0)
    geometry.add_attachment(
        None,
        AttachmentKind.VERTEX_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        source_kind="vertex",
        source_id=vertex,
        evidence=AttachmentEvidence.EXACT,
    )

    assert geometry.face_point(face, 0.5, 0.5)[2] == pytest.approx(10.0)
    assert geometry.face_support_point(face, 0.5, 0.5) == pytest.approx(
        (1.0, 1.0, 0.0)
    )
    report = strict_audit(geometry)

    assert AuditCode.ATTACHMENT_INCONSISTENT not in {
        issue.code for issue in report.issues
    }
    assert any(
        issue.code is AuditCode.INTENTIONAL_COINCIDENCE
        and {entity.kind for entity in issue.entities} >= {"face", "vertex"}
        for issue in report.issues
    )


def test_qualified_vertex_face_attachment_suppresses_unexplained_relation() -> None:
    geometry = GeometryModel()
    face = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    vertex = geometry.add_point(1.0, 1.0, 0.0)
    geometry.add_attachment(
        None,
        AttachmentKind.VERTEX_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        source_kind="vertex",
        source_id=vertex,
        evidence=AttachmentEvidence.EXACT,
    )

    report = strict_audit(geometry)
    codes = {item.code for item in report.issues}

    assert AuditCode.VERTEX_FACE_INTERIOR not in codes
    assert AuditCode.ATTACHMENT_INCONSISTENT not in codes
    assert AuditCode.INTENTIONAL_COINCIDENCE in codes


def test_changed_source_vertex_rechecks_attachment_geometry() -> None:
    geometry = GeometryModel()
    face = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    vertex = geometry.add_point(1.0, 1.0, 0.0)
    geometry.add_attachment(
        None,
        AttachmentKind.VERTEX_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        source_kind="vertex",
        source_id=vertex,
        evidence=AttachmentEvidence.EXACT,
    )
    geometry.move_point(vertex, 1.0, 1.0, 1.0)

    report = audit_changed_region(geometry, geometry.last_change_set)

    issue = next(
        item
        for item in report.issues
        if item.code is AuditCode.ATTACHMENT_INCONSISTENT
    )
    assert issue.measured_gap == pytest.approx(1.0)
    assert issue.tolerance_used is not None


def test_removed_structural_relation_fails_closed_with_unrelated_aabb_change() -> None:
    geometry = GeometryModel()
    face = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    vertex = geometry.add_point(1.0, 1.0, 0.0)
    attachment = geometry.add_attachment(
        None,
        AttachmentKind.VERTEX_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        source_kind="vertex",
        source_id=vertex,
        evidence=AttachmentEvidence.EXACT,
    )
    unrelated = geometry.add_point(100.0, 100.0, 100.0)
    with geometry.transaction():
        geometry.remove_attachment(attachment)
        geometry.move_point(unrelated, 101.0, 100.0, 100.0)
    change_set = geometry.last_change_set

    assert change_set.affected_aabbs
    report = audit_changed_region(geometry, change_set)

    assert any(
        issue.code is AuditCode.CAPABILITY_MISSING
        and any(
            entity.kind == "attachment" and entity.id == attachment
            for entity in issue.entities
        )
        for issue in report.issues
    )
    assert not report.certifiable


def test_overlap_query_uses_equivalent_indexed_selectors() -> None:
    geometry = GeometryModel()
    first = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    second = _plate(geometry, 1.0, 1.0, 3.0, 3.0)
    third = _plate(geometry, 20.0, 20.0, 21.0, 21.0)

    expected = find_coplanar_overlaps(
        geometry,
        candidate_pairs=((("face", first), ("face", second)),),
    )
    selected = find_coplanar_overlaps(geometry, face_ids=(third, second, first))
    changed = find_coplanar_overlaps(
        geometry,
        changed_aabbs=(AABB((0.5, 0.5, -0.1), (2.5, 2.5, 0.1)),),
    )

    assert expected == selected == changed
    assert expected[0].area == 1.0


@pytest.mark.parametrize("invalid", (True, 1.0, "1"))
def test_overlap_selectors_reject_noncanonical_face_ids(invalid: object) -> None:
    geometry = GeometryModel()
    first = _plate(geometry, 0.0, 0.0, 2.0, 2.0)
    second = _plate(geometry, 1.0, 1.0, 3.0, 3.0)

    with pytest.raises(GeometryError, match="positive integer"):
        find_coplanar_overlaps(geometry, face_ids=(first, invalid))
    with pytest.raises(GeometryError, match="positive integer"):
        find_coplanar_overlaps(
            geometry, candidate_pairs=((first, invalid),)
        )


@pytest.mark.parametrize("origin", (0.0, 1.0e9))
@pytest.mark.parametrize("scale", (1.0, 1.0e3))
def test_overlap_default_tolerance_is_local_translation_and_scale_invariant(
    origin: float, scale: float
) -> None:
    length_tolerance = 1.0e-6 * scale
    policy = TolerancePolicy(
        length=length_tolerance,
        coincidence=length_tolerance,
        aabb_padding=length_tolerance,
        surface_residual=length_tolerance,
    )

    within = GeometryModel(tolerance=policy)
    _plate(within, origin, origin, origin + 2.0 * scale, origin + 2.0 * scale, origin)
    _plate(
        within,
        origin + scale,
        origin + scale,
        origin + 3.0 * scale,
        origin + 3.0 * scale,
        origin + 0.5 * length_tolerance,
    )
    outside = GeometryModel(tolerance=policy)
    _plate(outside, origin, origin, origin + 2.0 * scale, origin + 2.0 * scale, origin)
    _plate(
        outside,
        origin + scale,
        origin + scale,
        origin + 3.0 * scale,
        origin + 3.0 * scale,
        origin + 2.0 * length_tolerance,
    )

    assert len(find_coplanar_overlaps(within)) == 1
    assert find_coplanar_overlaps(outside) == ()


def test_overlap_explicit_tolerance_straddles_fixed_gap() -> None:
    geometry = GeometryModel()
    _plate(geometry, 0.0, 0.0, 2.0, 2.0, 0.0)
    _plate(geometry, 1.0, 1.0, 3.0, 3.0, 5.0e-7)

    assert len(find_coplanar_overlaps(geometry, tolerance=1.0e-6)) == 1
    assert find_coplanar_overlaps(geometry, tolerance=1.0e-7) == ()


def test_changed_region_uses_conservative_coons_interior_bounds() -> None:
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
    geometry.add_face(edges, corners=(0, 2, 4, 6))
    start, end = geometry.add_points(((0.5, 0.5, 10.0), (0.5, 0.5, 11.0)))
    crossing_edge = geometry.add_line(start, end)
    geometry.spatial_candidates((-1, -1, -1), (2, 2, 12))

    with geometry.transaction():
        geometry.move_point(start, 0.5, 0.5, 1.5)
        geometry.move_point(end, 0.5, 0.5, 2.5)
    report = audit_changed_region(geometry, geometry.last_change_set)

    assert not report.clean and not report.certifiable
    assert any(
        issue.code is AuditCode.UNCLASSIFIED_CANDIDATE
        and any(
            entity.kind == "edge" and entity.id == crossing_edge
            for entity in issue.entities
        )
        for issue in report.issues
    )


def test_nearest_tree_prunes_unrelated_exact_evaluations_deterministically() -> None:
    tree = AABBTree(
        (("vertex", index), AABB.around_point((10.0 * index, 0.0, 0.0)))
        for index in range(1, 21)
    )
    evaluated: list[tuple[str, int]] = []

    def distance(key: tuple[str, int]) -> float:
        evaluated.append(key)
        return abs(10.0 * key[1] - 10.1)

    result = tree.nearest((10.1, 0.0, 0.0), distance)

    assert result.key == ("vertex", 1)
    assert result.distance == pytest.approx(0.1)
    assert result.diagnostics.candidate_count == len(evaluated)
    assert len(evaluated) < len(tree)


def test_selected_transform_change_set_excludes_unrelated_geometry() -> None:
    geometry = GeometryModel()
    selected_start, selected_end, unrelated_start, unrelated_end = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (100.0, 0.0, 0.0), (101.0, 0.0, 0.0))
    )
    selected_edge = geometry.add_line(selected_start, selected_end)
    unrelated_edge = geometry.add_line(unrelated_start, unrelated_end)
    matrix = np.eye(4)
    matrix[1, 3] = 2.0

    transform(geometry, matrix, (EntityRef("vertex", selected_start),))

    changed = set(geometry.last_change_set.changed)
    assert ("vertex", selected_start) in changed
    assert ("edge", selected_edge) not in changed  # immutable definition; bounds update only
    assert ("vertex", unrelated_start) not in changed
    assert ("edge", unrelated_edge) not in geometry.last_change_set.spatial_updates
