"""Contract fixtures for deterministic review and conservative repair."""
from dataclasses import replace
import math

import numpy as np
import pytest

from anygeometry.cleaner import (CleanerOptions, RepairPlan, review_geometry,
                                 propose_repair, apply_repair)
from anygeometry.model import GeometryModel
from anygeometry.errors import GeometryError


def rectangle(model, width=1., height=1., offset=0.):
    vertices = [model.add_point(x + offset, y) for x, y in
                ((0, 0), (width, 0), (width, height), (0, height))]
    return vertices, model.add_plate(vertices)


def kinds(report):
    return {item.kind for item in report.findings}


def unused(report):
    return [item.id for item in report.findings if item.kind == 'unused_vertex']


@pytest.mark.parametrize('field', ['short_edge', 'small_face', 'near_vertex', 'sliver_quality'])
@pytest.mark.parametrize('value', [0, -1, math.nan, math.inf, True])
def test_invalid_options(field, value):
    with pytest.raises(ValueError):
        CleanerOptions(**{field: value})


def test_healthy_square_and_threshold_boundary():
    model = GeometryModel()
    rectangle(model)
    baseline = model.revision
    report = review_geometry(model, CleanerOptions(short_edge=1, small_face=1, sliver_quality=.1))
    assert len([f for f in report.findings if f.kind == 'short_edge']) == 4
    assert len([f for f in report.findings if f.kind == 'small_face']) == 1
    assert 'sliver_face' not in kinds(report)
    assert report.complete
    assert model.revision == baseline
    assert report.findings == review_geometry(model, report.options).findings
    assert not review_geometry(model).findings


def test_sliver_and_small_face_are_diagnostic_only():
    model = GeometryModel()
    rectangle(model, 1, 1e-5)
    report = review_geometry(model)
    assert 'sliver_face' in kinds(report)
    item = next(f for f in report.findings if f.kind == 'sliver_face')
    with pytest.raises(GeometryError, match='manual'):
        propose_repair(model, report, finding_ids=[item.id])


def test_nearby_distinct_and_intentional_regions_are_never_stitched():
    model = GeometryModel()
    a, b, c, d = [model.add_point(*xy) for xy in ((0, 0), (1, 0), (1 + 1e-6, 0), (2, 0))]
    model.add_line(a, b)
    model.add_line(c, d)
    before = model.revision
    report = review_geometry(model)
    assert 'near_vertices' in kinds(report)
    assert len([f for f in report.findings if f.kind == 'disconnected_region']) == 2
    assert model.revision == before
    assert model.edges[1].end != model.edges[2].start


def test_exact_duplicates_are_grouped_and_faces_are_detected():
    model = GeometryModel()
    rectangle(model)
    rectangle(model)
    report = review_geometry(model)
    assert {'duplicate_vertices', 'duplicate_edges', 'duplicate_faces'} <= kinds(report)


def test_curves_remain_explicitly_unsupported_and_controls_are_not_unused():
    model = GeometryModel()
    vertices = [model.add_point(*xy) for xy in ((0, 0), (.5, .5), (1, 0))]
    model.add_arc(*vertices)
    report = review_geometry(model)
    assert not report.complete
    assert 'unsupported_edge' in kinds(report)
    assert 'unused_vertex' not in kinds(report)


def test_remove_preview_is_nonmutating_and_plan_is_revision_model_bound():
    model = GeometryModel()
    vertex = model.add_point(.1, .2)
    report = review_geometry(model)
    baseline = model.revision
    plan = propose_repair(model, report, finding_ids=unused(report))
    assert vertex in model.vertices and model.revision == baseline
    assert plan.changes[0].before == (.1, .2, 0.)
    with pytest.raises(GeometryError, match='forged'):
        apply_repair(model, replace(plan))
    with pytest.raises(GeometryError):
        apply_repair(model.clone(preserve_identity=True), plan)
    model.add_point(2, 3)
    with pytest.raises(GeometryError, match='stale'):
        apply_repair(model, plan)
    with pytest.raises(GeometryError, match='stale'):
        propose_repair(model, report)


def test_remove_selected_only_and_noop():
    model = GeometryModel()
    a, b = model.add_point(0, 0), model.add_point(1, 0)
    report = review_geometry(model)
    before = model.revision
    empty = propose_repair(model, report)
    assert empty.is_noop and apply_repair(model, empty) == ()
    assert model.revision == before
    selected = next(f.id for f in report.findings if f.refs[0].id == a)
    plan = propose_repair(model, report, finding_ids=[selected])
    apply_repair(model, plan)
    assert a not in model.vertices and b in model.vertices


def test_snap_bounds_collisions_and_dry_run():
    model = GeometryModel()
    a = model.add_point(.11, .21)
    model.add_point(0, 0)
    report = review_geometry(model)
    with pytest.raises(GeometryError, match='displacement'):
        propose_repair(model, report, snap_vertices=[a], grid_size=.1, max_displacement=.001)
    plan = propose_repair(model, report, snap_vertices=[a], grid_size=.1, max_displacement=.02)
    assert np.array_equal(model.vertex_position(a), [.11, .21, 0])
    apply_repair(model, plan)
    assert np.allclose(model.vertex_position(a), [.1, .2, 0])
    report = review_geometry(model)
    with pytest.raises(GeometryError, match='coincident'):
        propose_repair(model, report, snap_vertices=[a], grid_size=1, max_displacement=2)


def test_failed_live_apply_rolls_back(monkeypatch):
    model = GeometryModel()
    model.add_point(0, 0)
    model.add_point(1, 0)
    report = review_geometry(model)
    plan = propose_repair(model, report, finding_ids=unused(report))
    before = model.revision
    original = model.remove_vertex
    calls = []
    def fail_second(identifier, **kwargs):
        calls.append(identifier)
        if len(calls) == 2:
            raise GeometryError('injected mutation failure')
        return original(identifier, **kwargs)
    monkeypatch.setattr(model, 'remove_vertex', fail_second)
    with pytest.raises(GeometryError, match='injected'):
        apply_repair(model, plan)
    assert len(model.vertices) == 2 and model.revision == before


def test_construction_and_annotated_points_are_protected():
    model = GeometryModel()
    a = model.add_point(0, 0)
    model.mark_construction_vertices([a])
    assert 'unused_vertex' not in kinds(review_geometry(model))


def test_near_pair_cap_is_explicit():
    model = GeometryModel()
    for x in range(5):
        model.add_point(x * 1e-6, 0)
    report = review_geometry(model, CleanerOptions(max_vertex_pairs=2))
    assert not report.complete
    assert len([f for f in report.findings if f.kind == 'near_vertices']) == 2
    assert any('truncated' in text for text in report.coverage)


def test_planar_grid_snap_includes_dependent_edges_and_face():
    model = GeometryModel()
    vertices, face = rectangle(model, 1.01, 1.01)
    report = review_geometry(model)
    plan = propose_repair(model, report, snap_vertices=vertices, grid_size=.1, max_displacement=.02)
    assert any(item.kind == 'face' and item.id == face for item in plan.affected_refs)
    apply_repair(model, plan)
    assert review_geometry(model).complete
    assert np.array_equal(model.vertex_position(vertices[2]), [1, 1, 0])


def test_snap_refuses_creating_nonplanar_quad():
    model = GeometryModel()
    vertices = [model.add_point(x, y, .04) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    model.add_plate(vertices)
    report = review_geometry(model)
    baseline = model.revision
    with pytest.raises(GeometryError, match='planar'):
        propose_repair(model, report, snap_vertices=[vertices[0]], grid_size=.1, max_displacement=.05)
    assert model.revision == baseline


def test_feature_baseline_without_records_refuses_direct_repair():
    model = GeometryModel()
    model.add_point(0, 0)
    model.features.capture_baseline(model)
    report = review_geometry(model)
    with pytest.raises(GeometryError, match='Feature'):
        propose_repair(model, report, finding_ids=unused(report))

@pytest.mark.parametrize('units', ['mm', 'cm', 'ft'])
def test_nonmetre_document_coordinates_are_refused(units):
    model = GeometryModel()
    model.add_point(1, 2)
    report = review_geometry(model)
    plan = propose_repair(model, report)
    model.set_document_settings(units=units)
    with pytest.raises(GeometryError, match='metre'):
        review_geometry(model)
    with pytest.raises(GeometryError, match='metre'):
        propose_repair(model, report)
    with pytest.raises(GeometryError, match='metre'):
        apply_repair(model, plan)


@pytest.mark.parametrize('kind', ['translation', 'scaling'])
def test_transformed_document_coordinates_are_refused(kind):
    model = GeometryModel()
    transform = np.eye(4)
    if kind == 'translation':
        transform[0, 3] = 10
    else:
        transform[0, 0] = 2
    model.set_document_settings(coordinate_transform=transform)
    with pytest.raises(GeometryError, match='transform'):
        review_geometry(model)


def test_explicit_identity_transform_is_supported():
    model = GeometryModel()
    model.set_document_settings(coordinate_transform=np.eye(4), local_origin=(10, 20, 30))
    model.add_point(.11, .21)
    report = review_geometry(model)
    plan = propose_repair(model, report, snap_vertices=[1], grid_size=.1, max_displacement=.02)
    apply_repair(model, plan)
    assert np.allclose(model.vertex_position(1), [.1, .2, 0])
