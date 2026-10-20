"""Certified hydrostatics, slice CH1a: exact snapshot, admission, refusals, work budget (design sections 2, 6, 7)."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.certified_hydrostatics import (
    CertifiedHydrostaticsRefusal, PreparedCertifiedHydrostatics, RefusalCategory, WorkBudget, check_preparation,
    prepare_certified_hydrostatics)
from anygeometry.certified_hydrostatics._snapshot import collect_snapshot
from anygeometry.certified_hydrostatics._work import Meter
from anygeometry.curves import Straight
from anygeometry.entities import Edge, Face, OrientedEdge

C = RefusalCategory

# ------------------------------------------------------------------ builders


def build(points, faces, *, surfaces=None, raw=False):
    """Model from vertex coordinates and faces; a face is ``[outer, hole, ...]`` of vertex-index loops.

    The Plane support is taken from the outer loop's first three non-collinear vertices. ``raw`` writes edges and
    faces straight into the stores (the public API validates and would refuse the invalid fixtures)."""
    model = GeometryModel()
    ids = [model.add_point(*p) for p in points]
    lines: dict = {}

    def raw_edge(a, b):
        edge_id = model._next_id["edge"]
        model._next_id["edge"] = edge_id + 1
        model._put_entity("edge", Edge(edge_id, ids[a], ids[b], Straight()))
        return edge_id

    def items(loop):
        out = []
        for a, b in zip(loop, loop[1:] + loop[:1]):
            key = frozenset((a, b))
            if key not in lines:
                lines[key] = raw_edge(a, b) if raw else model.add_line(ids[a], ids[b])
            out.append(OrientedEdge(lines[key], model.edges[lines[key]].start == ids[a]))
        return out

    def support(loop):
        origin = np.array(points[loop[0]], dtype=float)
        u = np.array(points[loop[1]], dtype=float) - origin
        for k in loop[2:]:
            v = np.array(points[k], dtype=float) - origin
            if np.linalg.norm(np.cross(u, v)) > 1e-9:
                return Plane(origin, u, v)
        raise AssertionError("collinear loop")

    face_ids = []
    for index, (outer, *holes) in enumerate(faces):
        surface = support(outer) if surfaces is None or surfaces.get(index) is None else surfaces[index]
        if raw:
            face_id = model._next_id["face"]
            model._next_id["face"] = face_id + 1
            model._put_entity("face", Face(face_id, tuple(items(outer)), surface=surface,
                                           holes=tuple(tuple(items(h)) for h in holes)))
            face_ids.append(face_id)
            continue
        face_id = model.add_face_from_loop(items(outer), surface=surface)
        if holes:
            with model.transaction():
                model._put_entity("face", replace(model.faces[face_id], holes=tuple(tuple(items(h)) for h in holes)))
        face_ids.append(face_id)
    return model, face_ids


CUBE_QUADS = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3))   # outward
CUBE_POINTS = [(float(i), float(j), float(k)) for k in (0, 1) for j in (0, 1) for i in (0, 1)]


def unit_box(scale=1.0):
    return build([tuple(scale * c for c in p) for p in CUBE_POINTS], [[list(q)] for q in CUBE_QUADS])


def tetrahedron():
    return build([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                 [[[0, 2, 1]], [[0, 1, 3]], [[1, 2, 3]], [[0, 3, 2]]])


def extrusion(polygon, height=1.0, *, reverse=False):
    """Prism over the counter-clockwise ``polygon`` (x, y): outward faces, or all faces reversed."""
    n = len(polygon)
    points = [(x, y, 0.0) for x, y in polygon] + [(x, y, height) for x, y in polygon]
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    if reverse:
        faces = [loop[::-1] for loop in faces]
    return build(points, [[loop] for loop in faces])


def bored_box(*, wrong_winding=False):
    """4 x 4 x 4 box with a 2 x 2 square through-bore; ``wrong_winding``: hole loops wound like their outer loops
    and the bore walls reversed (consistent edges, wrong hole winding)."""
    outer, hole = [(0., 0.), (4., 0.), (4., 4.), (0., 4.)], [(1., 1.), (3., 1.), (3., 3.), (1., 3.)]
    points = ([(x, y, 0.) for x, y in outer] + [(x, y, 4.) for x, y in outer]
              + [(x, y, 0.) for x, y in hole] + [(x, y, 4.) for x, y in hole])
    ob, ot, hb, ht = range(0, 4), range(4, 8), range(8, 12), range(12, 16)
    top_hole = list(ht) if wrong_winding else list(ht)[::-1]
    bottom_hole = list(hb)[::-1] if wrong_winding else list(hb)
    faces = [[list(ot), top_hole], [list(ob)[::-1], bottom_hole]]
    faces += [[ob[i], ob[(i + 1) % 4], ot[(i + 1) % 4], ot[i]] for i in range(4)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([hb[i], hb[j], ht[j], ht[i]] if wrong_winding else [ht[i], ht[j], hb[j], hb[i]])
    return build(points, [[f] if isinstance(f[0], int) else f for f in faces])


def prepare(model, faces, **kwargs):
    return prepare_certified_hydrostatics(model, faces, **kwargs)


def category(model, faces, **kwargs):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, faces, **kwargs)
    return info.value.category


PLANE_Z = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
U_SHAPE = [(0., 0.), (3., 0.), (3., 3.), (2., 3.), (2., 1.), (1., 1.), (1., 3.), (0., 3.)]


# ------------------------------------------------------------------ admitted solids


@pytest.mark.parametrize("make", [
    unit_box, tetrahedron,
    lambda: extrusion([(0., 0.), (2., 0.), (0., 2.)], 1.5),
    bored_box,
    lambda: extrusion(U_SHAPE, 1.0),
], ids=["box", "tetrahedron", "triangular-prism", "bored-box", "extruded-U"])
def test_a_valid_solid_is_admitted(make):
    model, faces = make()
    prepared = prepare(model, faces)
    assert isinstance(prepared, PreparedCertifiedHydrostatics)
    assert prepared.face_ids == tuple(faces) and prepared.units == "m"
    assert prepared.model_id == str(model.model_id) and prepared.revision == model.revision
    assert len(prepared.snapshot_digest) == 64 and len(prepared.document_checksum) == 64
    assert prepared.work["operations"] > 0
    proof = prepared.proof_identity()
    assert set(proof) == {"artifact", "model"}
    assert proof["model"]["snapshot_digest"] == prepared.snapshot_digest
    assert {"distribution", "distribution_version", "package_version", "files", "dependencies", "python"} <= set(
        proof["artifact"])
    assert all(len(digest) == 64 for _path, digest in prepared.implementation.files)
    assert any(path == "certified_hydrostatics/_admission.py" for path, _d in prepared.implementation.files)
    assert not any("__pycache__" in path for path, _d in prepared.implementation.files)
    check_preparation(model, prepared)


def test_the_bored_box_really_has_holes():
    model, faces = bored_box()
    assert sum(1 for f in faces if model.faces[f].holes) == 2


def test_the_refusal_categories_are_the_design_list():
    assert {c.name for c in RefusalCategory} == {
        "UNSUPPORTED_SURFACE", "NOT_EXACTLY_PLANAR", "INVALID_LOOP", "INVALID_HOLES", "NOT_CLOSED", "NON_MANIFOLD",
        "SELF_INTERSECTING", "INVERTED_ORIENTATION", "NONFINITE_INPUT", "INVALID_QUERY", "ACCURACY_UNACHIEVABLE",
        "RESULT_OUT_OF_RANGE", "BIT_LENGTH_EXCEEDED", "BUDGET_EXHAUSTED", "CONCURRENT_MUTATION",
        "SYMBOLIC_DEGENERACY", "CANCELLED", "DEADLINE_EXCEEDED", "STALE_PREPARATION", "MALFORMED_PROOF",
        "VERIFICATION_FAILED"}


# ------------------------------------------------------------------ refusals


def test_a_curved_face_is_unsupported():
    import math
    from anygeometry import Cylinder
    model = GeometryModel()
    s = math.sqrt(.5)
    a, b, v = model.add_point(1, 0, 0), model.add_point(0, 1, 0), model.add_point(s, s, 0)
    a2, b2, v2 = model.add_point(1, 0, 1), model.add_point(0, 1, 1), model.add_point(s, s, 1)
    arcs = (model.add_arc(a, v, b), model.add_line(b, b2), model.add_arc(a2, v2, b2), model.add_line(a2, a))
    loop = [OrientedEdge(arcs[0], True), OrientedEdge(arcs[1], True), OrientedEdge(arcs[2], False),
            OrientedEdge(arcs[3], True)]
    face = model.add_face_from_loop(
        loop, surface=Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 0.0, math.pi / 2))
    assert category(model, [face]) is C.UNSUPPORTED_SURFACE


def test_a_curved_edge_is_unsupported():
    model = GeometryModel()
    a, b, via = model.add_point(0, 0, 0), model.add_point(2, 0, 0), model.add_point(1, -.5, 0)
    arc = model.add_arc(a, via, b)
    top = model.add_point(1, 1, 0)
    e2, e3 = model.add_line(b, top), model.add_line(top, a)
    loop = [OrientedEdge(arc, True), OrientedEdge(e2, True), OrientedEdge(e3, True)]
    face = model.add_face_from_loop(loop, surface=Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    assert category(model, [face]) is C.UNSUPPORTED_SURFACE


def test_a_face_without_plane_support_is_unsupported():
    model, faces = build(CUBE_POINTS[:4], [[[0, 1, 3, 2]]], surfaces={0: None}, raw=True)
    model._put_entity("face", replace(model.faces[faces[0]], surface=None))
    assert model.faces[faces[0]].surface is None
    assert category(model, faces) is C.UNSUPPORTED_SURFACE


def test_a_vertex_off_its_support_plane_is_not_exactly_planar():
    points = list(CUBE_POINTS)
    points[7] = (1., 1., 1. + 2. ** -40)
    model, faces = build(points, [[list(q)] for q in CUBE_QUADS])
    assert category(model, faces) is C.NOT_EXACTLY_PLANAR


def test_a_zero_length_edge_is_an_invalid_loop():
    points = [(0., 0., 0.), (1., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)]       # vertices 1 and 2 coincide
    model, faces = build(points, [[[0, 1, 2, 3, 4]]], surfaces={0: PLANE_Z}, raw=True)
    assert category(model, faces) is C.INVALID_LOOP


def test_a_self_touching_loop_is_an_invalid_loop():
    # figure eight touching at vertex 2 (visited twice)
    points = [(0., 0., 0.), (2., 0., 0.), (1., 1., 0.), (2., 2., 0.), (0., 2., 0.)]
    model, faces = build(points, [[[0, 1, 2, 3, 4, 2]]],
                         surfaces={0: PLANE_Z}, raw=True)
    assert category(model, faces) is C.INVALID_LOOP


def test_a_bowtie_loop_is_an_invalid_loop():
    points = [(0., 0., 0.), (3., 2., 0.), (3., 0., 0.), (0., 1., 0.)]
    model, faces = build(points, [[[0, 1, 2, 3]]], surfaces={0: PLANE_Z}, raw=True)
    assert category(model, faces) is C.INVALID_LOOP


def test_a_spike_in_a_loop_is_an_invalid_loop():
    points = [(0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (3., 2., 0.), (2., 2., 0.5), (0., 2., 0.)]
    points[4] = (1.5, 2., 0.)                           # doubles back along y = 2 towards the loop's own edge
    model, faces = build(points, [[[0, 1, 2, 3, 4, 5]]],
                         surfaces={0: PLANE_Z}, raw=True)
    assert category(model, faces) is C.INVALID_LOOP


def test_the_wrong_winding_bored_box_has_invalid_holes():
    model, faces = bored_box(wrong_winding=True)
    # the shell is closed and consistent: only the hole winding is wrong
    from anygeometry.shell_volume import closed_shell
    assert closed_shell(model, faces) == []
    assert category(model, faces) is C.INVALID_HOLES


def _holed_square(holes):
    points = [(0., 0., 0.), (6., 0., 0.), (6., 6., 0.), (0., 6., 0.)]
    loops = [[0, 1, 2, 3]]
    for hole in holes:
        base = len(points)
        points += [(x, y, 0.) for x, y in hole]
        loops.append(list(range(base, base + len(hole))))
    return build(points, [loops], surfaces={0: PLANE_Z}, raw=True)


def test_a_hole_touching_its_outer_loop_is_invalid():
    model, faces = _holed_square([[(0., 1.), (0., 2.), (1., 2.), (1., 1.)]])       # clockwise, on the edge x = 0
    assert category(model, faces) is C.INVALID_HOLES


def test_two_touching_holes_are_invalid():
    model, faces = _holed_square([[(1., 1.), (1., 2.), (2., 2.), (2., 1.)], [(2., 1.), (2., 2.), (3., 2.), (3., 1.)]])
    assert category(model, faces) is C.INVALID_HOLES


def test_a_hole_outside_the_outer_loop_is_invalid():
    model, faces = _holed_square([[(7., 1.), (7., 2.), (8., 2.), (8., 1.)]])
    assert category(model, faces) is C.INVALID_HOLES


def test_a_hole_inside_another_hole_is_invalid():
    model, faces = _holed_square([[(1., 1.), (1., 5.), (5., 5.), (5., 1.)], [(2., 2.), (2., 3.), (3., 3.), (3., 2.)]])
    assert category(model, faces) is C.INVALID_HOLES


def test_an_open_shell_is_not_closed():
    model, faces = unit_box()
    assert category(model, faces[:-1]) is C.NOT_CLOSED


def test_an_empty_face_set_is_not_closed():
    model, _faces = unit_box()
    assert category(model, []) is C.NOT_CLOSED


def test_two_cubes_sharing_only_a_vertex_are_not_manifold():
    points = list(CUBE_POINTS)
    index = {}
    second = []
    for n, (i, j, k) in enumerate((i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)):
        if (i, j, k) == (0, 0, 0):
            index[n] = 7                                         # the second cube's corner is the first cube's vertex 7
        else:
            index[n] = len(points)
            points.append((1. + i, 1. + j, 1. + k))
    faces = [[list(q)] for q in CUBE_QUADS] + [[[index[v] for v in q]] for q in CUBE_QUADS]
    model, ids = build(points, faces)
    assert category(model, ids) is C.NON_MANIFOLD


def test_a_face_used_by_three_faces_per_edge_is_not_manifold():
    model, faces = unit_box()
    assert category(model, faces + [faces[0]]) is C.NON_MANIFOLD


def test_two_separate_cubes_are_not_manifold():
    points = list(CUBE_POINTS) + [(x + 5., y, z) for x, y, z in CUBE_POINTS]
    faces = [[list(q)] for q in CUBE_QUADS] + [[[v + 8 for v in q]] for q in CUBE_QUADS]
    model, ids = build(points, faces)
    assert category(model, ids) is C.NON_MANIFOLD


def test_an_inconsistently_oriented_shell_is_not_manifold():
    model, faces = build(CUBE_POINTS, [[list(CUBE_QUADS[0][::-1])]] + [[list(q)] for q in CUBE_QUADS[1:]])
    assert category(model, faces) is C.NON_MANIFOLD


def test_overlapping_adjacent_faces_are_self_intersecting():
    # a closed tetrahedron whose apex lies in the plane of the base: faces ABC and ABD overlap beyond AB
    model, faces = build([(0., 0., 0.), (4., 0., 0.), (0., 4., 0.), (1., 1., 0.)],
                         [[[0, 2, 1]], [[0, 1, 3]], [[1, 2, 3]], [[0, 3, 2]]])
    assert category(model, faces) is C.SELF_INTERSECTING


def test_a_vertex_through_the_opposite_face_is_self_intersecting():
    points = list(CUBE_POINTS)
    points[7] = (.5, .5, -.5)                         # the top corner pierces the bottom face
    faces = []
    for a, b, c, d in CUBE_QUADS:
        faces += [[[a, b, c]], [[a, c, d]]]
    model, ids = build(points, faces)
    assert category(model, ids) is C.SELF_INTERSECTING


def test_an_inside_out_box_is_inverted():
    model, faces = build(CUBE_POINTS, [[list(q[::-1])] for q in CUBE_QUADS])
    assert category(model, faces) is C.INVERTED_ORIENTATION


def test_a_non_finite_coordinate_is_refused():
    model, faces = unit_box()
    position = model.vertices[next(iter(model.vertices))].position
    position.flags.writeable = True
    position[0] = np.nan
    assert category(model, faces) is C.NONFINITE_INPUT


def test_an_open_transaction_is_a_concurrent_mutation():
    model, faces = unit_box()
    with model.transaction():
        assert category(model, faces) is C.CONCURRENT_MUTATION


def test_a_revision_change_while_reading_is_a_concurrent_mutation():
    model, faces = unit_box()

    class Mutating(WorkBudget):
        done = False

        def reserve(self, kind, count=1, cancellation_check=None):
            if kind == "vertex_read" and not self.done:
                self.done = True
                model.add_point(9., 9., 9.)                 # another writer commits between the two revision reads
            super().reserve(kind, count, cancellation_check)
    assert category(model, faces, budget=Mutating(None, None)) is C.CONCURRENT_MUTATION


def test_an_unknown_face_id_is_an_invalid_request():
    model, faces = unit_box()
    assert category(model, faces + [10 ** 6]) is C.INVALID_QUERY


def test_a_refusal_names_its_phase_and_context():
    model, faces = unit_box()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, faces[:-1])
    refusal = info.value
    assert refusal.phase == "prepare" and refusal.work["operations"] > 0
    assert refusal.request["faces"] == faces[:-1] and refusal.identity["model_id"] == str(model.model_id)


# ------------------------------------------------------------------ digest and check_preparation


def _digest(model, faces):
    return collect_snapshot(model, faces, Meter(None)).digest(Meter(None))


def test_the_snapshot_digest_is_deterministic_and_independent_of_model_identity():
    first, faces1 = unit_box()
    second, faces2 = unit_box()
    assert faces1 == faces2 and first.model_id != second.model_id
    second.add_point(5., 5., 5.)                            # advances the revision, adds an unrelated vertex
    assert second.revision != first.revision
    a, b = prepare(first, faces1), prepare(second, faces2)
    assert a.snapshot_digest == b.snapshot_digest == _digest(first, faces1)
    assert a.model_id != b.model_id and a.revision != b.revision and a.document_checksum != b.document_checksum


@pytest.mark.parametrize("vertex_index", range(4))
@pytest.mark.parametrize("axis", range(3))
def test_the_digest_changes_when_a_vertex_moves_by_one_ulp(vertex_index, axis):
    model, faces = tetrahedron()
    before = _digest(model, faces)
    vertex = sorted(model.vertices)[vertex_index]
    position = np.array(model.vertices[vertex].position)
    position[axis] = np.nextafter(position[axis], np.inf)
    model.move_point(vertex, *position)
    assert _digest(model, faces) != before


def test_the_digest_depends_on_the_face_order():
    model, faces = unit_box()
    assert _digest(model, faces) != _digest(model, faces[::-1])


def test_check_preparation_passes_on_the_unchanged_model_and_refuses_after_a_mutation():
    model, faces = unit_box()
    prepared = prepare(model, faces)
    assert check_preparation(model, prepared)["operations"] > 0
    model.add_point(7., 7., 7.)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        check_preparation(model, prepared)
    assert info.value.category is C.STALE_PREPARATION and info.value.phase == "check"


def test_check_preparation_refuses_a_clone_with_moved_geometry_and_a_clone_with_a_new_identity():
    model, faces = unit_box()
    prepared = prepare(model, faces)
    clone = model.clone(preserve_identity=True)
    assert clone.model_id == model.model_id
    vertex = sorted(clone.vertices)[0]
    position = clone.vertices[vertex].position
    position.flags.writeable = True
    position[1] += 1e-3                                   # raw edit of the clone: same id, same revision counter
    assert model.vertices[vertex].position[1] == 0.0 and clone.revision == model.revision
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        check_preparation(clone, prepared)
    assert info.value.category is C.STALE_PREPARATION
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        check_preparation(model.clone(), prepared)       # a new model id
    assert info.value.category is C.STALE_PREPARATION


def test_check_preparation_detects_a_move_that_leaves_the_revision_unchanged():
    model, faces = unit_box()
    prepared = prepare(model, faces)
    position = model.vertices[sorted(model.vertices)[0]].position
    position.flags.writeable = True
    position[0] = np.nextafter(position[0], np.inf)          # raw edit: neither revision nor id changes
    assert model.revision == prepared.revision
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        check_preparation(model, prepared)
    assert info.value.category is C.STALE_PREPARATION


def test_check_preparation_refuses_an_open_transaction():
    model, faces = unit_box()
    prepared = prepare(model, faces)
    with model.transaction():
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            check_preparation(model, prepared)
    assert info.value.category is C.CONCURRENT_MUTATION


# ------------------------------------------------------------------ work budget


def _stopped(call):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        call()
    return info.value


def _each_stop(model, faces, run, phase):
    """The four budget refusals through ``run(budget, cancellation_check)``."""
    refusal = _stopped(lambda: run(WorkBudget(max_operations=25, max_bits=None), None))
    assert refusal.category is C.BUDGET_EXHAUSTED and refusal.phase == phase
    assert 0 < refusal.work["operations"] <= 25 and refusal.work["units"]
    refusal = _stopped(lambda: run(WorkBudget(None, None, deadline=time.monotonic() - 1.0), None))
    assert refusal.category is C.DEADLINE_EXCEEDED and refusal.phase == phase and "operations" in refusal.work
    state = {"calls": 0}

    def cancel():
        state["calls"] += 1
        return state["calls"] > 40
    budget = WorkBudget(None, None)
    refusal = _stopped(lambda: run(budget, cancel))
    assert refusal.category is C.CANCELLED and refusal.phase == phase
    assert refusal.work["operations"] == budget.operations == 40
    refusal = _stopped(lambda: run(WorkBudget(None, 4), None))
    assert refusal.category is C.BIT_LENGTH_EXCEEDED and refusal.phase == phase and refusal.work["max_bits_seen"] > 4


def test_every_budget_refusal_is_raised_during_preparation():
    model, faces = unit_box(0.1)                         # coordinates need 55-bit denominators
    _each_stop(model, faces, lambda budget, cancel: prepare(model, faces, budget=budget, cancellation_check=cancel),
               "prepare")


def test_every_budget_refusal_is_raised_during_check_preparation():
    model, faces = unit_box(0.1)
    prepared = prepare(model, faces)
    _each_stop(model, faces,
               lambda budget, cancel: check_preparation(model, prepared, budget=budget, cancellation_check=cancel),
               "check")


def test_a_budget_is_shared_across_calls_and_counts_accumulate():
    model, faces = unit_box()
    budget = WorkBudget(max_operations=10 ** 6, max_bits=10 ** 4)
    prepared = prepare(model, faces, budget=budget)
    after_prepare = budget.operations
    check_preparation(model, prepared, budget=budget)
    assert after_prepare == prepared.work["operations"] < budget.operations
    assert budget.counts()["max_bits_seen"] > 0 and "hash_chunk" in budget.counts()["units"]
    tight = WorkBudget(max_operations=budget.operations - 1, max_bits=None)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, faces, budget=tight)
        check_preparation(model, prepared, budget=tight)
    assert info.value.category is C.BUDGET_EXHAUSTED


def test_a_refused_unit_is_not_consumed():
    budget = WorkBudget(max_operations=3, max_bits=None)
    budget.reserve("a", 3)
    with pytest.raises(CertifiedHydrostaticsRefusal):
        budget.reserve("a", 1)
    assert budget.operations == 3


def test_work_budget_arguments_are_validated():
    with pytest.raises(ValueError):
        WorkBudget(max_operations=-1, max_bits=None)
    with pytest.raises(ValueError):
        WorkBudget(max_operations=True, max_bits=None)
    model, faces = unit_box()
    with pytest.raises(TypeError):
        prepare(model, faces, budget=object())


# ------------------------------------------------------------------ review round 1 regressions


def test_the_bit_cap_applies_to_derived_values_not_only_to_inputs():
    # support vectors scaled by 2**100: the inputs need 101 bits, the support normal 201
    model = GeometryModel()
    ids = [model.add_point(*p) for p in CUBE_POINTS]
    lines, faces = {}, []
    big = float(2 ** 100)
    for quad in CUBE_QUADS:
        loop = []
        for a, b in zip(quad, quad[1:] + quad[:1]):
            key = frozenset((a, b))
            if key not in lines:
                lines[key] = model.add_line(ids[a], ids[b])
            loop.append(OrientedEdge(lines[key], model.edges[lines[key]].start == ids[a]))
        o = np.array(CUBE_POINTS[quad[0]])
        u, v = np.array(CUBE_POINTS[quad[1]]) - o, np.array(CUBE_POINTS[quad[3]]) - o
        faces.append(model.add_face_from_loop(loop, surface=Plane(o, big * u, big * v)))
    assert prepare(model, faces).work["max_bits_seen"] > 101           # admitted without a cap
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, faces, budget=WorkBudget(None, 101))
    assert info.value.category is C.BIT_LENGTH_EXCEEDED


def test_an_exhausted_budget_stops_the_face_iterable_at_once():
    model, faces = unit_box()
    consumed = []

    def generator():
        for index in range(10_000):
            consumed.append(index)
            yield faces[0]
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, generator(), budget=WorkBudget(0, None))
    assert info.value.category is C.BUDGET_EXHAUSTED and len(consumed) <= 1
    consumed.clear()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, generator(), cancellation_check=lambda: True)
    assert info.value.category is C.CANCELLED and len(consumed) <= 1


def test_an_endless_face_iterable_is_stopped_by_the_budget():
    import itertools
    model, _faces = unit_box()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare(model, itertools.cycle([1]), budget=WorkBudget(1000, None))
    assert info.value.category is C.BUDGET_EXHAUSTED and info.value.phase == "prepare"


def test_check_preparation_verifies_the_stored_snapshot_against_its_digest():
    model, faces = unit_box()
    prepared = prepare(model, faces)
    bad = replace(prepared, snapshot=replace(prepared.snapshot, faces=prepared.snapshot.faces[:-1]))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        check_preparation(model, bad)
    assert info.value.category is C.STALE_PREPARATION


def _cube_with_identified_corners():
    """Triangulated cube whose opposite corners 0 and 7 are one vertex: connected, edge-consistent, pinched."""
    diagonals = {0: (1, 2), 1: (5, 6), 2: (2, 4), 3: (3, 5), 4: (1, 4), 5: (3, 6)}
    faces = []
    for index, quad in enumerate(CUBE_QUADS):
        x, y = diagonals[index]
        quad = list(quad)
        i, j = quad.index(x), quad.index(y)
        first = [quad[k % 4] for k in range(i, i + 1 + (j - i) % 4)]          # split along the diagonal x-y
        second = [quad[k % 4] for k in range(j, j + 1 + (i - j) % 4)]
        faces += [[[0 if v == 7 else v for v in first]], [[0 if v == 7 else v for v in second]]]
    return build(CUBE_POINTS, faces)


def test_a_pinched_connected_shell_fails_the_vertex_link_check():
    model, faces = _cube_with_identified_corners()
    refusal = _stopped(lambda: prepare(model, faces))
    assert refusal.category is C.NON_MANIFOLD
    assert "link_corner" in refusal.work["units"] and "bbox_pair" not in refusal.work["units"]


def test_disconnected_shells_fail_before_the_vertex_link_check():
    points = list(CUBE_POINTS) + [(x + 5., y, z) for x, y, z in CUBE_POINTS]
    faces = [[list(q)] for q in CUBE_QUADS] + [[[v + 8 for v in q]] for q in CUBE_QUADS]
    model, ids = build(points, faces)
    refusal = _stopped(lambda: prepare(model, ids))
    assert refusal.category is C.NON_MANIFOLD and "link_corner" not in refusal.work["units"]


class _Recording(WorkBudget):
    """Budget that notes, for each unit kind, the reserve call and operation count at its first appearance."""

    def __init__(self, **kwargs):
        super().__init__(None, None, **kwargs)
        self.calls, self.first = 0, {}

    def reserve(self, kind, count=1, cancellation_check=None):
        self.first.setdefault(kind, (self.calls, self.operations, count))
        super().reserve(kind, count, cancellation_check)
        self.calls += 1                                  # completed reserve calls


def _stops_inside(run, kind, phase):
    """Budget, cancellation and deadline stops that fire just after unit ``kind`` ran; the work record shows it."""
    probe = _Recording()
    run(probe, None)
    index, operations, count = probe.first[kind]
    refusal = _stopped(lambda: run(WorkBudget(operations + count + 1, None), None))
    assert refusal.category is C.BUDGET_EXHAUSTED and refusal.phase == phase and refusal.work["units"][kind] >= 1
    seen = {"n": 0}

    def cancel():
        seen["n"] += 1
        return seen["n"] > index + 2
    refusal = _stopped(lambda: run(WorkBudget(None, None), cancel))
    assert refusal.category is C.CANCELLED and refusal.phase == phase and refusal.work["units"][kind] >= 1
    clocked = _Recording(deadline=index + 2, clock=lambda: clocked.calls)
    refusal = _stopped(lambda: run(clocked, None))
    assert refusal.category is C.DEADLINE_EXCEEDED and refusal.phase == phase and refusal.work["units"][kind] >= 1


@pytest.mark.parametrize("kind", ["face_request", "serialize_element", "document_element", "plane_test", "loop_edge",
                                  "edge_pair", "point_edge_test", "link_corner", "bbox_pair", "triangle_sum"])
def test_stops_fire_inside_every_stage_of_preparation(kind):
    model, faces = bored_box()
    _stops_inside(lambda budget, cancel: prepare(model, faces, budget=budget, cancellation_check=cancel), kind,
                  "prepare")


@pytest.mark.parametrize("kind", ["serialize_element", "document_element", "vertex_read"])
def test_stops_fire_inside_check_preparation(kind):
    model, faces = unit_box()
    prepared = prepare(model, faces)
    _stops_inside(lambda budget, cancel: check_preparation(model, prepared, budget=budget,
                                                           cancellation_check=cancel), kind, "check")


def test_the_bit_cap_stops_admission_after_the_snapshot_and_its_serialization():
    model, faces = unit_box(0.1)                           # inputs need 56 bits; the support normals about 112
    refusal = _stopped(lambda: prepare(model, faces, budget=WorkBudget(None, 60)))
    assert refusal.category is C.BIT_LENGTH_EXCEEDED and refusal.work["max_bits_seen"] > 60
    assert refusal.work["units"]["serialize_element"] > 0 and "vertex_read" in refusal.work["units"]


def test_an_edge_pointing_to_a_missing_vertex_is_a_structured_refusal():
    model, faces = unit_box()
    edge = next(iter(model.edges))
    model._put_entity("edge", replace(model.edges[edge], end=10 ** 6))
    refusal = _stopped(lambda: prepare(model, faces))
    assert refusal.category is C.INVALID_LOOP and refusal.phase == "prepare"
    assert refusal.work["operations"] > 0 and isinstance(refusal.__cause__, KeyError)


def test_a_non_finite_unrelated_vertex_is_a_structured_refusal():
    model, faces = unit_box()
    stray = model.add_point(5., 5., 5.)
    position = model.vertices[stray].position
    position.flags.writeable = True
    position[2] = np.nan
    refusal = _stopped(lambda: prepare(model, faces))
    assert refusal.category is C.NONFINITE_INPUT and refusal.phase == "prepare"
    assert refusal.work["operations"] > 0 and refusal.__cause__ is not None
