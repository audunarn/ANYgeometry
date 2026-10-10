"""Box volume cut for submodelling (docs/SUBMODEL_VOLUME_CUT_CONTRACT.md, slice S2a)."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry.audit import AuditCode
from anygeometry.curves import Arc
from anygeometry.model import GeometryModel
from anygeometry.serialization import to_dict
from anygeometry.structural import AttachmentKind, ParameterRange
from anygeometry.surfaces import Cylinder, Plane
from anygeometry.volume_cut import INSIDE, OUTSIDE, BoxVolume, VolumeCutError, apply_volume_cut, plan_volume_cut


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def plate(model, corners, *, surface=True):
    ids = [model.add_point(*c) for c in corners]
    edges = [model.add_line(ids[i], ids[(i + 1) % 4]) for i in range(4)]
    plane = Plane(np.asarray(corners[0], float), np.subtract(corners[1], corners[0]),
                  np.subtract(corners[3], corners[0])) if surface else None
    return model.add_face(edges, surface=plane)


def unit_plate():
    model = GeometryModel()
    return model, plate(model, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])


def t_junction():
    model = GeometryModel()
    a = [model.add_point(*c) for c in [(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (1, 1, 0), (0.5, 1, 0), (0, 1, 0),
                                       (0.5, 0, 0.5), (0.5, 1, 0.5)]]
    line = model.add_line
    bottom_left, bottom_right = line(a[0], a[1]), line(a[1], a[2])
    right, top_right, top_left, left = line(a[2], a[3]), line(a[3], a[4]), line(a[4], a[5]), line(a[5], a[0])
    junction = line(a[1], a[4])
    web_bottom, web_top, web_far = line(a[1], a[6]), line(a[6], a[7]), line(a[7], a[4])
    faces = (model.add_face([bottom_left, junction, top_left, left], surface=Plane((0, 0, 0), (0.5, 0, 0), (0, 1, 0))),
             model.add_face([bottom_right, right, top_right, junction], surface=Plane((0.5, 0, 0), (0.5, 0, 0), (0, 1, 0))),
             model.add_face([junction, web_bottom, web_top, web_far], surface=Plane((0.5, 0, 0), (0, 1, 0), (0, 0, 0.5))))
    return model, faces, junction


def o_grid(radius=0.3, half=1.0):
    """A square plate with a circular hole: four faces around the hole."""

    model = GeometryModel()
    corners = [model.add_point(sx * half, sy * half, 0.0) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    rim = [model.add_point(radius * np.cos(a), radius * np.sin(a), 0.0) for a in np.pi / 4 + np.pi / 2 * np.arange(4)]
    mids = [model.add_point(radius * np.cos(a), radius * np.sin(a), 0.0) for a in np.pi / 2 * (np.arange(4) + 1)]
    outer = [model.add_line(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    spokes = [model.add_line(rim[k], corners[k]) for k in range(4)]
    arcs = [model.add_arc(rim[k], mids[k], rim[(k + 1) % 4]) for k in range(4)]
    plane = Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))
    faces = [model.add_face([spokes[k], outer[k], spokes[(k + 1) % 4], arcs[k]], surface=plane) for k in range(4)]
    return model, faces, arcs


HALF_STEP = (np.pi / 2) / 256  # half the spacing of 129 samples over a quarter arc


def disc(radius=1.0):
    """A disc of four quarter arcs; x = R lies half a 129-sample step off any sample."""

    model = GeometryModel()
    angles = -np.pi / 4 + HALF_STEP + np.pi / 2 * np.arange(4)
    rim = [model.add_point(radius * np.cos(a), radius * np.sin(a), 0.0) for a in angles]
    mids = [model.add_point(radius * np.cos(a + np.pi / 4), radius * np.sin(a + np.pi / 4), 0.0) for a in angles]
    arcs = [model.add_arc(rim[k], mids[k], rim[(k + 1) % 4]) for k in range(4)]
    return model, model.add_face(arcs, surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))


def cylinder_patches(radius=1.0, height=4.0):
    model = GeometryModel()

    def at(a, z):
        return model.add_point(radius * np.cos(a), radius * np.sin(a), z)

    bottom, top = [at(a, 0.0) for a in np.pi / 2 * np.arange(4)], [at(a, height) for a in np.pi / 2 * np.arange(4)]
    axial = [model.add_line(bottom[k], top[k]) for k in range(4)]
    faces = []
    for k in range(4):
        arc_b = model.add_arc(bottom[k], at(np.pi / 4 + np.pi / 2 * k, 0.0), bottom[(k + 1) % 4])
        arc_t = model.add_arc(top[k], at(np.pi / 4 + np.pi / 2 * k, height), top[(k + 1) % 4])
        surface = Cylinder(origin=(0, 0, 0), axis=(0, 0, 1), radial_direction=(1, 0, 0), radius=radius, height=height,
                           start_angle=np.pi / 2 * k, sweep_angle=np.pi / 2)
        faces.append(model.add_face([arc_b, axial[(k + 1) % 4], arc_t, axial[k]], surface=surface))
    return model, faces


# ----------------------------------------------------------------------
# independent checks (dense sampling with an explicit sagitta pad, not the module's extents)
# ----------------------------------------------------------------------
def dense(model, edge_id, count=4001):
    return model.sample_edge(edge_id, np.linspace(0.0, 1.0, count))


def extreme(model, edge_id, direction, count=2001):
    """Max of direction . x along an edge: a grid, then golden-section refinement (independent of the module)."""

    f = lambda t: float(model.sample_edge(edge_id, np.array([t]))[0] @ direction)  # noqa: E731
    grid = np.linspace(0.0, 1.0, count)
    values = model.sample_edge(edge_id, grid) @ direction
    k = int(np.argmax(values))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, count - 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        if f(c) > f(d):
            b = d
        else:
            a = c
    return max(float(values[k]), f(0.5 * (a + b)))


def assert_inside(result, tolerance=1e-9):
    assert result.unresolved == ()
    model, volume = result.closure.working_model, result.volume
    for edge_id in model.edges:
        for axis in range(3):
            direction = volume.rotation[:, axis]
            offset = float(volume.translation @ direction)
            assert extreme(model, edge_id, direction) - offset <= volume.upper[axis] + tolerance
            assert -extreme(model, edge_id, -direction) - offset >= volume.lower[axis] - tolerance


def area(model, face_id):
    face = model.faces[face_id]
    total = 0.0
    for loop in (face.loop,) + tuple(face.holes):
        points = []
        for use in loop:
            samples = dense(model, use.edge)
            points.extend((samples if use.forward else samples[::-1])[:-1])
        xy = np.asarray(points)[:, :2]
        signed = 0.5 * float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
        total += abs(signed) if not total else -abs(signed)
    return abs(total)


def findings(model):
    return {(issue.code, tuple(sorted((e.kind, e.id) for e in issue.entities))) for issue in model.audit().issues}


def assert_audit_provenance(source, result, *, imprint_via_vertices=False):
    """Every finding of the extracted model is present right after the imprint (mapped to
    the working model) and corresponds to a source finding of the same code through the
    lineage: entities the cut made on the box stand in for removed material, every other
    entity must lift to the source finding, and at least one must lift.  The exception is
    VERTEX_FACE_INTERIOR on a via vertex the imprint made, when that is expected."""

    source_findings = findings(source)
    imprint_findings = findings(result.working_model)
    extracted = result.closure.working_model
    to_work = {target.key: handle.key for handle, target in result.closure.source_to_work.items()}
    made_via = {("vertex", edge.curve.via_vertex) for edge in extracted.edges.values()
                if isinstance(edge.curve, Arc) and ("vertex", edge.curve.via_vertex) in result.created}
    excepted = 0
    for code, entities in findings(extracted):
        work_entities = tuple(sorted(to_work[e] for e in entities))
        assert (code, work_entities) in imprint_findings, (code, entities)
        if not entities:  # a model-level finding
            assert (code, ()) in source_findings, code
            continue
        if imprint_via_vertices and code is AuditCode.VERTEX_FACE_INTERIOR and set(entities) & made_via:
            excepted += 1
            continue
        lifted = [{(e[0], s) for s in result.lineage.get(e, ())} for e in entities if e not in result.created]
        assert lifted and all(lifted), (code, entities)
        assert any(c == code and all(options & set(src) for options in lifted) for c, src in source_findings), (
            code, entities)
    return excepted


def cut(model, volume):
    return apply_volume_cut(model, plan_volume_cut(model, volume))


# ----------------------------------------------------------------------
# cuts
# ----------------------------------------------------------------------
def test_a_plate_through_the_box():
    model, face = unit_plate()
    before = to_dict(model)["checksum"]["value"]
    result = cut(model, BoxVolume((0.3, 0.3, -0.1), (0.7, 0.7, 0.1)))
    extracted = result.closure.working_model
    assert to_dict(model)["checksum"]["value"] == before
    (inner,) = extracted.faces
    assert area(extracted, inner) == pytest.approx(0.16, abs=1e-12)
    assert result.source_faces == {inner: (face,)}
    assert set(result.cut_edges) == {"x0", "x1", "y0", "y1"}
    assert result.original_boundary_edges == () and result.unresolved == ()
    assert sorted(result.classification.values()) == [INSIDE, OUTSIDE]
    assert extracted.audit().certifiable
    assert_inside(result)


def test_a_rotated_box():
    model, _face = unit_plate()
    angle = np.pi / 7
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    transform[:3, 3] = (0.5, 0.5, 0.0)
    result = cut(model, BoxVolume((-0.2, -0.15, -0.1), (0.2, 0.15, 0.1), transform=transform))
    (inner,) = result.closure.working_model.faces
    assert area(result.closure.working_model, inner) == pytest.approx(0.12, abs=1e-10)
    assert set(result.cut_edges) == {"x0", "x1", "y0", "y1"}
    assert result.closure.working_model.audit().certifiable
    assert_inside(result)


def test_a_box_past_one_plate_edge_keeps_that_original_boundary():
    model, _face = unit_plate()
    result = cut(model, BoxVolume((0.6, 0.3, -0.1), (1.2, 0.7, 0.1)))
    assert set(result.cut_edges) == {"x0", "y0", "y1"}
    (boundary,) = result.original_boundary_edges
    assert boundary in result.source_edges


def test_a_plate_edge_on_a_box_side_is_not_an_artificial_cut():
    """Review round 1: the plate's own free edge at x = 0 lies on the box side x0."""

    model, _face = unit_plate()
    result = cut(model, BoxVolume((0.0, 0.3, -0.1), (0.7, 0.7, 0.1)))
    assert "x0" not in result.cut_edges
    assert set(result.cut_edges) == {"x1", "y0", "y1"}
    (boundary,) = result.original_boundary_edges
    assert boundary in result.source_edges


def test_a_t_junction_keeps_one_shared_inside_junction_edge():
    model, faces, junction = t_junction()
    result = cut(model, BoxVolume((0.3, 0.3, -0.1), (0.7, 0.7, 0.3)))
    extracted = result.closure.working_model
    assert sorted(result.source_faces.values()) == sorted((f,) for f in faces)
    shared = [e for e, sources in result.source_edges.items() if sources == (junction,)]
    assert len(shared) == 1
    users = [f for f, face in extracted.faces.items()
             if any(use.edge == shared[0] for loop in (face.loop,) + tuple(face.holes) for use in loop)]
    assert len(users) == 3
    assert all(shared[0] not in edges for edges in result.cut_edges.values())
    assert set(result.cut_edges) == {"x0", "x1", "y0", "y1", "z1"}
    assert assert_audit_provenance(model, result) == 0
    assert_inside(result)


@pytest.mark.parametrize("box, hole_inside", [(((-0.6, -0.6, -0.1), (0.6, 0.6, 0.1)), True),
                                               (((0.1, -0.5, -0.1), (0.8, 0.5, 0.1)), False)])
def test_a_plate_with_a_hole(box, hole_inside):
    model, faces, arcs = o_grid()
    result = cut(model, BoxVolume(*box))
    extracted = result.closure.working_model
    if hole_inside:
        total = sum(area(extracted, f) for f in extracted.faces)
        assert total == pytest.approx(1.2 * 1.2 - np.pi * 0.09, rel=1e-6)
        assert set(result.original_boundary_edges) == {e for e, s in result.source_edges.items() if s[0] in arcs}
        assert assert_audit_provenance(model, result) == 0
    else:
        assert assert_audit_provenance(model, result, imprint_via_vertices=True) >= 1
    assert_inside(result)


def test_a_cylinder_patch_across_a_seam():
    model, faces = cylinder_patches()
    result = cut(model, BoxVolume((0.5, -0.5, 1.5), (1.5, 0.5, 2.5)))
    extracted = result.closure.working_model
    assert len(extracted.faces) == 2
    assert sorted(s for v in result.source_faces.values() for s in v) == sorted([faces[0], faces[3]])
    assert all(isinstance(face.surface, Cylinder) for face in extracted.faces.values())
    assert set(result.cut_edges) == {"y0", "y1", "z0", "z1"}
    # Finding for the intersection owner: the plane/cylinder imprint leaves the new arcs'
    # via vertices flagged VERTEX_FACE_INTERIOR, already in the working model.
    assert assert_audit_provenance(model, result, imprint_via_vertices=True) >= 1
    assert_inside(result)


def test_an_extremum_between_samples_is_not_lost():
    """Review rounds 1-2: the disc's x-extreme lies half a sample step off the 129 samples,
    so a sampled extent would miss a box starting at x = 0.99999; the cut must find it."""

    model, face = disc()
    sampled = max(float(np.max(model.sample_edge(e, np.linspace(0.0, 1.0, 129))[:, 0])) for e in model.edges)
    assert sampled < 0.99999 < 1.0
    try:
        result = cut(model, BoxVolume((0.99999, -0.05, -0.1), (1.1, 0.05, 0.1)))
    except VolumeCutError as error:  # a sliver too thin to verify an interior point is a typed refusal
        assert error.code == "VOLUME_FACE_DEGENERATE"
        return
    (piece,) = result.closure.working_model.faces
    assert result.source_faces == {piece: (face,)}
    assert_inside(result)


def test_a_wider_cut_at_the_disc_extreme():
    model, face = disc()
    result = cut(model, BoxVolume((0.99, -0.05, -0.1), (1.1, 0.05, 0.1)))
    (piece,) = result.closure.working_model.faces
    assert result.source_faces == {piece: (face,)}
    assert_inside(result)


def test_structural_sheets_carry_lineage():
    model, face = unit_plate()
    part = model.add_part(name="deck")
    sheet = model.add_sheet([face], part_id=part)
    result = cut(model, BoxVolume((0.3, 0.3, -0.1), (0.7, 0.7, 0.1)))
    assert {"face", "sheet", "part"} <= {kind for kind, _ in result.lineage}
    assert [s for (k, _), s in result.lineage.items() if k == "sheet"] == [(sheet,)]
    assert [s for (k, _), s in result.lineage.items() if k == "part"] == [(part,)]
    assert result.unresolved == ()
    from anygeometry.closure import ModelClosure, _closure_keys

    closure = result.closure
    assert set(closure.work_to_source) == set(_closure_keys(closure.working_model))
    assert {h.kind for h in closure.work_to_source} >= {"face", "edge", "vertex", "sheet", "part", "face_use", "coedge"}
    restored = ModelClosure.from_transport(closure.to_transport())
    assert {(t.kind, t.id): (s.kind, s.id) for t, s in restored.work_to_source.items()} == {
        (t.kind, t.id): (s.kind, s.id) for t, s in closure.work_to_source.items()}
    assert assert_audit_provenance(model, result) == 0


# ----------------------------------------------------------------------
# refusals and binding
# ----------------------------------------------------------------------
def test_an_empty_volume_refuses():
    model, _face = unit_plate()
    with pytest.raises(VolumeCutError, match="VOLUME_EMPTY"):
        plan_volume_cut(model, BoxVolume((2, 2, 2), (3, 3, 3)))


def test_a_box_plane_on_a_plate_refuses():
    model, _face = unit_plate()
    with pytest.raises(VolumeCutError, match="VOLUME_COINCIDENT"):
        plan_volume_cut(model, BoxVolume((0.3, 0.3, 0.0), (0.7, 0.7, 0.2)))


def test_a_box_plane_inside_a_hole_is_not_coincident():
    """Review round 1: the O-grid lies in the box's bottom plane, but only its hole meets the box."""

    model, _faces, _arcs = o_grid()
    web = plate(model, [(0.0, -0.1, 0.05), (0.0, 0.1, 0.05), (0.0, 0.1, 0.15), (0.0, -0.1, 0.15)])
    result = cut(model, BoxVolume((-0.2, -0.2, 0.0), (0.2, 0.2, 0.2)))
    assert list(result.source_faces.values()) == [(web,)]


def test_crossing_unprepared_faces_refuse():
    model, _face = unit_plate()
    plate(model, [(0.5, 0.0, -0.5), (0.5, 1.0, -0.5), (0.5, 1.0, 0.5), (0.5, 0.0, 0.5)])
    with pytest.raises(VolumeCutError, match="VOLUME_SOURCE_INTERSECTION"):
        cut(model, BoxVolume((0.3, 0.3, -0.2), (0.7, 0.7, 0.2)))


def test_relations_and_members_near_the_box_refuse():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    axis = model.add_member((model.add_line(*model.add_points(((0, 2, 0), (1, 2, 0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, "face", face, ParameterRange(0, 1),
                         (ParameterRange(0, 0.25), ParameterRange(0.5, 0.5)), evidence="exact", tolerance_used=1e-8)
    with pytest.raises(VolumeCutError, match="VOLUME_RELATIONS_UNSUPPORTED"):
        plan_volume_cut(model, BoxVolume((2.5, 1.5, -0.1), (3.5, 2.5, 0.1)))
    with pytest.raises(VolumeCutError, match="VOLUME_MEMBERS_UNSUPPORTED"):
        plan_volume_cut(model, BoxVolume((0.5, 1.5, -0.1), (1.5, 2.5, 0.1)))


def test_a_non_planar_coons_face_refuses():
    model = GeometryModel()
    plate(model, [(0, 0, 0), (1, 0, 0.2), (1, 1, 0), (0, 1, 0.3)], surface=False)
    with pytest.raises(VolumeCutError, match="VOLUME_SURFACE_UNSUPPORTED|VOLUME_IMPRINT_FAILED"):
        cut(model, BoxVolume((0.3, 0.3, -0.5), (0.7, 0.7, 0.5)))


def test_a_stale_plan_refuses():
    model, _face = unit_plate()
    plan = plan_volume_cut(model, BoxVolume((0.3, 0.3, -0.1), (0.7, 0.7, 0.1)))
    model.add_point(5.0, 5.0, 5.0)
    with pytest.raises(VolumeCutError, match="VOLUME_STALE_PLAN"):
        apply_volume_cut(model, plan)


@pytest.mark.parametrize("lower, upper, transform", [
    ((0, 0, 0), (0, 1, 1), None),
    ((0, 0, 0), (1, 1, float("nan")), None),
    ((0, 0, 0), (1, 1, 1), np.diag([2.0, 1.0, 1.0, 1.0])),
])
def test_invalid_volumes_refuse(lower, upper, transform):
    with pytest.raises(VolumeCutError, match="VOLUME_INVALID"):
        BoxVolume(lower, upper, transform=transform)


def test_the_cut_is_deterministic():
    box = BoxVolume((0.3, 0.3, -0.1), (0.7, 0.7, 0.3))
    first, second = cut(t_junction()[0], box), cut(t_junction()[0], box)
    assert first.lineage == second.lineage and first.cut_edges == second.cut_edges


def test_a_cylinder_cut_by_a_tilted_box():
    """Review round 2: box sides oblique to the axis trace ellipses on the cylinder."""

    from anygeometry.exact_curves import EllipticArc

    model, faces = cylinder_patches()
    angle = np.radians(20.0)
    transform = np.eye(4)
    transform[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    transform[:3, 3] = (1.0, 0.0, 2.0)
    result = cut(model, BoxVolume((-0.5, -0.4, -0.5), (0.5, 0.4, 0.5), transform=transform))
    extracted = result.closure.working_model
    assert any(isinstance(edge.curve, EllipticArc) for edge in extracted.edges.values())
    assert set(result.cut_edges) == {"y0", "y1", "z0", "z1"}
    assert_inside(result)


def test_a_distant_unsupported_face_does_not_block_a_local_cut():
    """Review round 2: only the plan's faces are classified."""

    model, face = unit_plate()
    plate(model, [(5, 5, 0), (6, 5, 0.2), (6, 6, 0), (5, 6, 0.3)], surface=False)  # a non-planar Coons face, far away
    result = cut(model, BoxVolume((0.3, 0.3, -0.1), (0.7, 0.7, 0.1)))
    assert list(result.source_faces.values()) == [(face,)]
    assert_inside(result)
