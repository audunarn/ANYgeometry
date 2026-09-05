"""0.4.3 fail-closed overlap qualification and transactional ownership."""

from dataclasses import replace
import math

import numpy as np
import pytest

from anygeometry import (
    EntityRef, GeometryError, GeometryModel, OrientedEdge, Plane,
    OverlapOwnershipPolicy, OverlapQualificationError,
    apply_coplanar_fragmentation, find_coplanar_overlaps,
    fragment_coplanar_overlaps, plan_coplanar_fragmentation, to_dict,
    TolerancePolicy, query_intersection, plan_imprint, apply_imprint,
)


POLICY = OverlapOwnershipPolicy.FIRST_SELECTED


def rectangle(g, x0=0, x1=2, *, offset=0, scale=1):
    points = g.add_points(tuple((offset + scale*x, offset + scale*y, offset)
                               for x, y in ((x0, 0), (x1, 0), (x1, 1), (x0, 1))))
    return g.add_plate(points)


def pair(*, identical=False, owned=False):
    g = GeometryModel()
    a = rectangle(g)
    b = rectangle(g, 0 if identical else 1, 2 if identical else 3)
    if owned:
        g.add_sheet((a,), name="deck A")
        g.add_sheet((b,), name="deck B")
    return g, a, b


def snapshot(g):
    return (to_dict(g), g.revision, dict(g._next_id), dict(g._next_structural_id),
            dict(g._arc_cache), dict(g._edge_length_cache), g._spatial_index)


@pytest.mark.parametrize("invalid", [True, False, 1.9, 1.0, "1", 0, -1, 999, EntityRef("edge", 1)])
def test_ids_rejected_before_allocation(invalid):
    g, a, b = pair()
    before = snapshot(g)
    with pytest.raises(GeometryError):
        g.add_sheet((invalid,))
    assert snapshot(g) == before
    with pytest.raises(GeometryError):
        plan_coplanar_fragmentation(g, (invalid, b), ownership_policy=POLICY)
    assert snapshot(g) == before


def test_explicit_policy_and_deeply_immutable_preview_result():
    g, a, b = pair(identical=True, owned=True)
    g.add_to_group("B", (EntityRef("face", b),))
    g.tag(EntityRef("face", b), "tag-B")
    g.set_face_metadata(a, {"section": "A"})
    g.set_face_metadata(b, {"section": "B"})
    before = snapshot(g)
    with pytest.raises(GeometryError, match="ownership_policy"):
        fragment_coplanar_overlaps(g, (a, b))
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    assert snapshot(g) == before
    assert plan.expected_fragment_counts == {a: 1, b: 0}
    assert plan.effects["sheets"]["2"]["removed"]
    assert plan.effects["faces"][str(b)]["tags"] == ("tag-B",)
    assert plan.effects["metadata_conflicts"] == ((a, b),)
    with pytest.raises(TypeError):
        plan.effects["sheets"]["2"]["name"] = "changed"
    events = []
    g.add_change_hook(events.append)
    result = apply_coplanar_fragmentation(g, plan)
    assert len(events) == 1
    assert g.revision == before[1] + 1
    assert tuple(s.name for s in g.sheets.values()) == ("deck A",)
    assert not g.groups["B"]
    assert result.descendants[b] == ()
    with pytest.raises(TypeError):
        result.descendants[a] = ()
    with pytest.raises(TypeError):
        result.outputs["fake"] = EntityRef("face", a)
    with pytest.raises(GeometryError, match="STALE_PLAN"):
        apply_coplanar_fragmentation(g, plan)
    assert len(events) == 1
    assert g.validate_topology() == g._validate_structural() == ()


def test_plan_tampering_wrong_model_and_live_drift_are_nonmutating():
    g, a, b = pair()
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    before = snapshot(g)
    with pytest.raises(GeometryError, match="digest"):
        apply_coplanar_fragmentation(g, replace(plan, expected_fragment_counts={a: 99}))
    assert snapshot(g) == before
    other, _, _ = pair()
    with pytest.raises(GeometryError, match="WRONG_MODEL"):
        apply_coplanar_fragmentation(other, plan)
    g.add_point(10, 10, 10)
    before = snapshot(g)
    with pytest.raises(GeometryError, match="STALE_PLAN"):
        apply_coplanar_fragmentation(g, plan)
    assert snapshot(g) == before


def test_dependency_blocker_is_in_preview_and_does_not_allocate():
    g, a, b = pair()
    g.add_member((g.faces[a].loop[0].edge,))
    before = snapshot(g)
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    assert plan.blockers and "member" in plan.blockers[0]
    with pytest.raises(GeometryError, match="remapping"):
        apply_coplanar_fragmentation(g, plan)
    assert snapshot(g) == before


def test_late_failure_rolls_back_without_reusing_provisional_ids(monkeypatch):
    g, a, b = pair(owned=True)
    tree = g._spatial()
    spatial_before = tuple((key, tree.bounds(key)) for key in tree.keys)
    for edge in g.edges:
        g.edge_length(edge)
    cache_before = dict(g._edge_length_cache)
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    before_doc, revision = to_dict(g), g.revision
    first_free = g._next_id["face"]
    original = g._validate_incremental
    def fail(journal):
        assert original(journal) == ()
        return ("injected late failure",)
    monkeypatch.setattr(g, "_validate_incremental", fail)
    with pytest.raises(GeometryError, match="injected late failure"):
        apply_coplanar_fragmentation(g, plan)
    assert g.revision == revision
    after_doc = to_dict(g)
    # Allocator high-water is intentionally monotonic; live stores are exact.
    assert g.faces.keys() == {a, b}
    assert tuple(s.name for s in g.sheets.values()) == ("deck A", "deck B")
    assert g._next_id["face"] > first_free
    assert g._spatial_index is tree
    assert tuple((key, tree.bounds(key)) for key in tree.keys) == spatial_before
    assert g._edge_length_cache == cache_before
    assert {k: v for k, v in before_doc.items() if k not in ("id_state", "checksum")} == {
        k: v for k, v in after_doc.items() if k not in ("id_state", "checksum")
    }
    monkeypatch.setattr(g, "_validate_incremental", original)
    result = apply_coplanar_fragmentation(g, plan)
    assert min(i for ids in result.descendants.values() for i in ids) > first_free


def disc(g, cx, cy):
    ids = [g.add_point(cx + math.cos(i*math.pi/4), cy + math.sin(i*math.pi/4), 0) for i in range(8)]
    edges = [g.add_arc(ids[i], ids[i+1], ids[(i+2) % 8]) for i in range(0, 8, 2)]
    return g.add_face_from_loop(tuple(OrientedEdge(e, True) for e in edges),
                               surface=Plane(np.array((cx, cy, 0.)), np.array((1., 0, 0)), np.array((0, 1., 0))))


def test_between_chords_circle_overlap_is_never_silent_disjoint():
    g = GeometryModel()
    a = disc(g, 0, 0)
    d, angle = 1.9999, math.pi/128
    b = disc(g, d*math.cos(angle), d*math.sin(angle))
    assert g.validate_topology() == ()
    assert 2*math.acos(d/2) - .5*d*math.sqrt(4-d*d) > 1e-6
    with pytest.raises(OverlapQualificationError) as caught:
        find_coplanar_overlaps(g, candidate_pairs=((a, b),))
    assert caught.value.candidate_pairs == ((g.handle("face", a), g.handle("face", b)),)
    assert caught.value.diagnostics
    with pytest.raises(AttributeError):
        caught.value.diagnostics = ()


def test_no_empty_tag_records_from_fragmentation():
    g, a, b = pair()
    fragment_coplanar_overlaps(g, (a, b), ownership_policy=POLICY)
    assert not g.tags
    assert not g.last_change_set.tag_changes


def test_new_overlap_features_require_recorded_policy():
    g, a, b = pair()
    with pytest.raises(GeometryError, match="ownership_policy"):
        g.features.append("geometry.fragment.overlaps")
    record = g.features.append("geometry.fragment.overlaps", parameters={"ownership_policy": POLICY})
    assert record.parameters["ownership_policy"] == "first_selected"


def test_boundary_label_consequences_are_in_preview():
    g, a, b = pair()
    edge = g.faces[a].loop[0].edge
    ref = EntityRef("edge", edge)
    g.add_to_group("edge load", (ref,))
    g.tag(ref, "loaded")
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    entry = plan.effects["boundary_labels"][f"edge/{edge}"]
    assert entry["groups"] == ("edge load",)
    assert entry["tags"] == ("loaded",)
    assert entry["descendant_count"] == 2
    apply_coplanar_fragmentation(g, plan)
    assert len(g.group("edge load")) == 2
    assert all(g.tags_for(ref) == ("loaded",) for ref in g.group("edge load"))


@pytest.mark.parametrize("identical", [False, True])
def test_connect_retains_exact_sheet_identities(identical):
    g, a, b = pair(identical=identical, owned=True)
    for face_id, name in ((a, "A"), (b, "B")):
        g.add_to_group(name, (EntityRef("face", face_id),))
        g.tag(EntityRef("face", face_id), name)
    sheets = {s.id: s.name for s in g.sheets.values()}
    events = []
    g.add_change_hook(events.append)
    query = query_intersection(g, g.handle("face", a), g.handle("face", b))
    plan = plan_imprint(g, query, policy="connect")
    apply_imprint(g, plan, policy="connect")
    assert len(events) == 1
    assert {s.id: s.name for s in g.sheets.values()} == sheets
    shared = [face for face in g.faces if len(g._face_structural_uses.get(face, ())) == 2]
    assert shared
    for face in shared:
        assert {g.face_uses[i].sheet_id for i in g._face_structural_uses[face]} == set(sheets)
        ref = EntityRef("face", face)
        assert ref in g.group("A") and ref in g.group("B")
        assert g.tags_for(ref) == ("A", "B")
        assert ref in g.resolve_ref(EntityRef("face", a))
        assert ref in g.resolve_ref(EntityRef("face", b))
    assert g.validate_topology() == g._validate_structural() == ()


def test_corner_to_corner_diagonal_connect_preserves_corner_vertices():
    g = GeometryModel()
    vertices = g.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    base = g.add_plate(vertices)
    diagonal = g.add_line(vertices[0], vertices[2])
    wall = g.extrude((diagonal,), (0, 0, 1))[0]
    base_sheet = g.add_sheet((base,), name="deck")
    wall_sheet = g.add_sheet((wall,), name="wall")
    query = query_intersection(g, g.handle("face", base), g.handle("face", wall))
    apply_imprint(g, plan_imprint(g, query, policy="connect"), policy="connect")
    assert len(g.sheets[base_sheet].face_use_ids) == 2
    assert {g.edges[diagonal].start, g.edges[diagonal].end} == {vertices[0], vertices[2]}
    users = {g.face_uses[i].sheet_id for i in g.radial_face_uses(diagonal)}
    assert users == {base_sheet, wall_sheet}
    assert g.validate_topology() == g._validate_structural() == ()


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
@pytest.mark.parametrize("offset", [0., 1e6])
def test_fragmentation_translation_scale_invariance(scale, offset):
    # Keep offsets representable relative to the smallest geometry.
    translated = offset * scale
    g = GeometryModel(tolerance=TolerancePolicy().scaled(scale))
    a = rectangle(g, offset=translated, scale=scale)
    b = rectangle(g, 1, 3, offset=translated, scale=scale)
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    assert plan.expected_fragment_counts == {a: 2, b: 1}
    result = apply_coplanar_fragmentation(g, plan)
    assert result.overlap_area / scale**2 == pytest.approx(1., rel=1e-8)
    assert g.validate_topology() == ()


def test_fragmentation_work_and_validation_are_local(monkeypatch):
    def work(remote):
        g, a, b = pair(owned=True)
        with g.transaction():
            for i in range(remote):
                face = rectangle(g, 100 + i*4, 102 + i*4)
                g.add_sheet((face,), name=f"remote {i}")
        visited = []
        original = g._validate_face_geometry
        def counted(face_id):
            visited.append(face_id)
            return original(face_id)
        monkeypatch.setattr(g, "_validate_face_geometry", counted)
        result = fragment_coplanar_overlaps(g, (a, b), ownership_policy=POLICY)
        counts = (dict(result.work_counts), len(visited), g.last_structural_validation_diagnostics.visited_count)
        assert g.validate_topology() == g._validate_structural() == ()
        return counts
    assert work(0) == work(12)


def test_legacy_feature_without_policy_still_replays():
    from anygeometry.serialization import from_dict
    g, a, b = pair()
    g.features.capture_baseline(g)
    record = g.features.append("geometry.fragment.overlaps", parameters={"ownership_policy": POLICY},
                               inputs={"faces": (EntityRef("face", a), EntityRef("face", b))})
    assert g.regenerate_features().success
    # Simulate an already-authored legacy feature rather than allowing an
    # implicit policy through the new append API.
    g.features._get_record(record.feature_id).parameters = {}
    doc = to_dict(g)
    restored = from_dict(doc)
    assert restored.regenerate_features().success
    assert len(restored.faces) == 3


@pytest.mark.parametrize("distance", [0., 1., 2., 3.])
def test_curved_candidate_without_area_is_qualified_or_explicit(distance):
    g = GeometryModel()
    a, b = disc(g, 0, 0), disc(g, distance, 0)
    try:
        result = find_coplanar_overlaps(g, candidate_pairs=((a, b),))
    except OverlapQualificationError as error:
        assert error.candidate_pairs
    else:
        assert distance >= 2 and result == ()


def test_partial_answers_are_withheld_and_malformed_certificates_fail_closed(monkeypatch):
    from anygeometry import intersections
    g, a, b = pair()
    c, d = disc(g, 20, 0), disc(g, 21, 0)
    monkeypatch.setattr(intersections, "query_intersection", lambda *args: object())
    with pytest.raises(OverlapQualificationError) as error:
        find_coplanar_overlaps(g, candidate_pairs=((a, b), (c, d)))
    assert error.value.candidate_pairs == ((g.handle("face", c), g.handle("face", d)),)


def test_holed_plate_fragmentation_preserves_material_domain():
    g = GeometryModel()
    a = g.add_plate(g.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    b = g.add_plate(g.add_points(((2, 0, 0), (5, 0, 0), (5, 4, 0), (2, 4, 0))))
    with g.transaction():
        hole = g.add_polyline(g.add_points(((1, 1, 0), (1, 2, 0), (3, 2, 0), (3, 1, 0))), close=True)
        g._put_entity("face", replace(g.faces[a], holes=(tuple(OrientedEdge(i, True) for i in hole),)))
    assert find_coplanar_overlaps(g)[0].area == pytest.approx(7.)
    result = fragment_coplanar_overlaps(g, (a, b), ownership_policy=POLICY)
    assert result.overlap_area == pytest.approx(7.)
    assert find_coplanar_overlaps(g) == ()
    assert g.validate_topology() == ()


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_shared_planar_segment_predicate_rejects_cross_touch_overlap(scale):
    pred = GeometryModel._segments_intersect_2d
    p = lambda x, y: np.array((x*scale, y*scale))
    tol = 1e-9*scale
    assert not pred(p(0,0), p(2,0), p(0,1), p(2,1), tol)
    assert pred(p(0,0), p(2,2), p(0,2), p(2,0), tol)
    assert pred(p(0,0), p(2,0), p(2,0), p(2,1), tol)
    assert pred(p(0,0), p(2,0), p(1,0), p(3,0), tol)
    assert pred(p(0,0), p(0,0), p(1,0), p(2,0), tol)


def test_recomputed_digest_cannot_hide_modified_preview():
    from anygeometry.overlaps import _plan_digest
    g, a, b = pair()
    plan = plan_coplanar_fragmentation(g, (a, b), ownership_policy=POLICY)
    fake = replace(plan, expected_fragment_counts={a: 55, b: 1})
    fake = replace(fake, digest=_plan_digest(fake))
    before = snapshot(g)
    with pytest.raises(GeometryError, match="live preview"):
        apply_coplanar_fragmentation(g, fake)
    assert snapshot(g) == before


def test_nearly_parallel_unequal_segments_cannot_hide_a_crossing():
    p = GeometryModel._segments_intersect_2d
    short = (np.array((0., 0.)), np.array((1., 0.)))
    long = (np.array((-5e8, 5e-7)), np.array((5e8, -5e-7)))
    assert p(*short, *long, 1e-9)
    assert p(*long, *short, 1e-9)


@pytest.mark.parametrize("identical", [True, False])
def test_connect_retains_reversed_faceuse_orientation_and_metadata(identical):
    g, a, b = pair(identical=identical)
    g.add_sheet((a,), name="deck")
    sheet = g.add_sheet((b,), name="reverse", orientations=(-1,))
    original_id = g.sheets[sheet].face_use_ids[0]
    with g.transaction():
        g._put_structural("face_use", replace(g.face_uses[original_id], metadata={"owner.label": "reverse"}))
    query = query_intersection(g, g.handle("face", a), g.handle("face", b))
    apply_imprint(g, plan_imprint(g, query, policy="connect"), policy="connect")
    assert original_id in g.face_uses
    for use_id in g.sheets[sheet].face_use_ids:
        assert int(g.face_uses[use_id].orientation) == -1
        assert g.face_uses[use_id].metadata["owner.label"] == "reverse"
    assert g._validate_structural() == ()


@pytest.mark.parametrize("rollback", [False, True])
def test_face_id_never_invalidates_unrelated_edge_cache_with_same_integer(rollback):
    g = GeometryModel()
    vertices = g.add_points(((0, 0, 0), (1, 1, 0), (2, 0, 0)))
    arc = g.add_arc(*vertices)
    face = rectangle(g, 10, 12)
    assert arc == face
    g.edge_length(arc)
    frame, length = g._arc_cache[arc], g._edge_length_cache[arc]
    if rollback:
        with pytest.raises(GeometryError, match="rollback"):
            with g.transaction():
                g.remove_face(face)
                raise GeometryError("rollback")
        assert face in g.faces
    else:
        g.remove_face(face)
        assert face not in g.faces
    assert g._arc_cache[arc] is frame
    assert g._edge_length_cache[arc] == length
