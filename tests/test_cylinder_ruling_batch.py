"""Authenticated BATCH of exact Cylinder straight-ruling lifts.

Independent checks for the shared-scope batch producer: expected rays, windings
and ``v`` endpoints against the scalar evidence and a Decimal atan oracle,
constant document-hash cost, deterministic canonical ordering, whole-batch
refusals (malformed/duplicate/unsupported selections, aggregate budget
exhaustion where every scalar succeeds, cancellation without mutation, stale
revision), binding rederivation with forged/repaired receipts, final
replacement-surface and private-coefficient counterexamples, and the absence
of quadratic per-callback surface rescans. No partition, source-current,
reference, meshing or "1000" acceptance is claimed here.
"""
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import (
    Cylinder,
    GeometryError,
    GeometryModel,
    OrientedEdge,
    Plane,
    to_dict,
)
from anygeometry.cylinder_angular_lifts import (
    CylinderAngularLiftError,
    cylinder_ruling_lift_digest,
    exact_ray,
    query_cylinder_straight_ruling_lift,
    validate_cylinder_straight_ruling_lift_binding,
)
from anygeometry.cylinder_charts import CylinderAtlasPolicy
from anygeometry.cylinder_ruling_batch import (
    CylinderRulingBatchLift,
    cylinder_ruling_batch_digest,
    query_cylinder_ruling_batch,
    validate_cylinder_ruling_batch_binding,
)

import anygeometry.cylinder_angular_lifts as lifts
import anygeometry.cylinder_ruling_batch as batch_module


def panel(model, shift, foot, *, rounded=False, height=2.0):
    """One radius-5 canonical Z cylinder panel translated by ``shift``.

    The straight ruling sits at the exact rational footpoint ``foot`` (either
    (3, 4) or (4, 3) on the radius-5 circle), or at a rounded 45-degree
    footpoint that is off the exact carrier when ``rounded``.
    """
    surface = Cylinder(shift, (0.0, 0.0, 1.0), (7.0, 0.0, 0.0), 5.0, height, 0.0, 1.0)
    if rounded:
        foot = (5.0 / math.sqrt(2.0), 5.0 / math.sqrt(2.0))
    fx, fy = foot
    mid = (5.0 * math.cos(0.3), 5.0 * math.sin(0.3)) if foot == (4.0, 3.0) else (4.0, 3.0)
    points = {}

    def point(x, y, z):
        key = (x, y, z)
        if key not in points:
            points[key] = model.add_point(shift[0] + x, shift[1] + y, shift[2] + z)
        return points[key]

    a, b, c = point(5.0, 0.0, 0.0), point(mid[0], mid[1], 0.0), point(fx, fy, 0.0)
    top_a, top_b, top_c = (point(5.0, 0.0, height), point(mid[0], mid[1], height),
                          point(fx, fy, height))
    bottom = model.add_arc(a, b, c)
    ruling = model.add_line(c, top_c)
    top = model.add_arc(top_c, top_b, top_a)
    seam = model.add_line(top_a, a)
    return surface, (OrientedEdge(bottom, True), OrientedEdge(ruling, True),
                     OrientedEdge(top, True), OrientedEdge(seam, True))


def panels(spec, *, rounded=False):
    """Distinct translated panels authored in ONE transaction, one part each."""
    model = GeometryModel()
    with model.transaction():
        faces = []
        for index, (shift, foot) in enumerate(spec):
            surface, loop = panel(model, shift, foot, rounded=rounded)
            face = model.add_face_from_loop(loop, surface=surface)
            part = model.add_part(name=f"part{index}")
            model.add_sheet((face,), part_id=part)
            faces.append(face)
    assert model.validate_topology() == ()
    rulings, seams, arcs = {}, {}, {}
    for index, face in enumerate(faces):
        shift = spec[index][0]
        for use in model.faces[face].loop:
            edge = model.edges[use.edge]
            start = model.vertex_position(edge.start)
            if edge.curve.__class__.__name__ != "Straight":
                arcs[face] = use.edge
            elif abs(start[0] - shift[0] - 5.0) < 1e-9 and abs(start[1] - shift[1]) < 1e-9:
                seams[face] = use.edge
            else:
                rulings[face] = use.edge
    return model, faces, rulings, seams, arcs


def standard():
    """Three translated panels: rulings at (3,4), (4,3) and (3,4) again."""
    return panels((
        ((0.0, 0.0, 0.0), (3.0, 4.0)),
        ((10.0, 0.0, 0.0), (4.0, 3.0)),
        ((20.0, 0.0, 0.0), (3.0, 4.0)),
    ))


def decimal_atan(y, x):
    """Independent Decimal Taylor oracle for atan2(y, x) with x > 0."""
    with localcontext() as context:
        context.prec = 90
        value = Decimal(int(y)) / Decimal(int(x))
        halvings = 0
        while abs(value) > Decimal("1e-3"):
            value = value / (1 + (1 + value * value).sqrt())
            halvings += 1
        total, term, power, index = Decimal(0), value, value * value, 0
        while abs(term) > Decimal("1e-85"):
            total += term / (2 * index + 1) * (1 if index % 2 == 0 else -1)
            term *= power
            index += 1
        return total * (2 ** halvings)


def selection(model, faces, rulings):
    return [(face, rulings[face]) for face in faces]


def test_batch_lifts_match_scalar_evidence_and_independent_oracle():
    model, faces, rulings, seams, _ = standard()
    before = to_dict(model)
    pick = selection(model, faces, rulings)
    batch = query_cylinder_ruling_batch(model, pick)
    assert type(batch) is CylinderRulingBatchLift
    assert batch.model_id == model.model_id and batch.revision == model.revision
    assert batch.policy_limits == (256, 4096, 8192, 200000, 65536, 4096)
    # Complete membership in canonical order, nothing else.
    assert [(item.face_id, item.edge_id) for item in batch.items] == sorted(pick)
    expected_rays = [exact_ray(3, 4), exact_ray(4, 3), exact_ray(3, 4)]
    for item, face, ray in zip(batch.items, faces, expected_rays):
        scalar = query_cylinder_straight_ruling_lift(model, face, rulings[face])
        assert item.ray == ray == scalar.ray
        assert item.winding == scalar.winding == 0
        assert (item.v_start, item.v_end) == (F(0), F(1)) == (scalar.v_start, scalar.v_end)
        assert item.support_interval == scalar.support_interval == (F(0), F(1))
        assert item.angle_enclosure == scalar.angle_enclosure
        assert item.origin == scalar.origin and item.axis == scalar.axis
        assert item.radial_direction == scalar.radial_direction
        assert item.circumferential_direction == scalar.circumferential_direction
        assert item.frame_determinant == scalar.frame_determinant == F(50)
        assert item.radius == F(5) and item.height == F(2)
        assert item.start_angle == F(0) and item.sweep_angle == F(1)
        low, high = item.angle_enclosure
        oracle = decimal_atan(ray.y, ray.x)
        assert low <= oracle + Decimal("1e-50") and oracle - Decimal("1e-50") <= high
        assert high - low < F(1, 2**70)
    # Independent float identity along the whole analytic ruling line.
    surface = model.faces[faces[1]].surface
    item = batch.items[1]
    u = math.atan2(3.0, 4.0)
    start = np.array([10.0 + 4.0, 3.0, 0.0])
    end = np.array([10.0 + 4.0, 3.0, 2.0])
    for t in (-0.5, 0.0, 0.5, 1.0, 1.5):
        point = start + t * (end - start)
        v = float(item.v_start + F(t) * (item.v_end - item.v_start))
        assert np.allclose(surface.evaluate(u, v), point, atol=1e-12)
    # Clear batch budget/work provenance: cumulative per-item counts, and the
    # aggregate extends the final item by exactly the final cancellation check.
    counts = [dict(item.work_counts) for item in batch.items]
    assert all(counts[k + 1]["interval_operations"] > counts[k]["interval_operations"]
               for k in range(len(counts) - 1))
    aggregate = dict(batch.work_counts)
    assert aggregate["cancellation_checks"] == counts[-1]["cancellation_checks"] + 1
    assert all(aggregate[name] == counts[-1][name]
               for name in counts[-1] if name != "cancellation_checks")
    # The batch query never mutates the model, and the scalar contract still
    # holds on the same model afterwards.
    assert to_dict(model) == before
    scalar = query_cylinder_straight_ruling_lift(model, faces[0], rulings[faces[0]])
    validate_cylinder_straight_ruling_lift_binding(model, scalar)
    assert to_dict(model) == before


def test_batch_shares_one_face_frame_across_occurrences():
    model, faces, rulings, seams, _ = standard()
    face = faces[0]
    batch = query_cylinder_ruling_batch(model, [(face, rulings[face]), (face, seams[face])])
    assert [item.ray for item in batch.items] == [exact_ray(3, 4), exact_ray(1, 0)]
    assert batch.items[1].v_start == F(1) and batch.items[1].v_end == F(0)
    assert batch.items[0].angle_enclosure != batch.items[1].angle_enclosure


def test_document_hash_count_is_constant_in_operand_count(monkeypatch):
    model, faces, rulings, seams, _ = standard()
    calls = []
    original = batch_module._document_checksum

    def counting(model_instance):
        calls.append(1)
        return original(model_instance)

    monkeypatch.setattr(batch_module, "_document_checksum", counting)
    for pick in ([(faces[0], rulings[faces[0]])],
                 [(faces[0], rulings[faces[0]]), (faces[1], rulings[faces[1]])],
                 selection(model, faces, rulings),
                 sorted(selection(model, faces, rulings)
                        + [(face, seams[face]) for face in faces])):
        del calls[:]
        receipt = query_cylinder_ruling_batch(model, pick)
        assert len(calls) == 2
        del calls[:]
        validate_cylinder_ruling_batch_binding(model, receipt)
        assert len(calls) == 3


def test_reversed_and_repeated_selections_are_deterministic():
    model, faces, rulings, _, _ = standard()
    pick = selection(model, faces, rulings)
    batch = query_cylinder_ruling_batch(model, pick)
    digest = cylinder_ruling_batch_digest(batch)
    assert cylinder_ruling_batch_digest(
        query_cylinder_ruling_batch(model, list(reversed(pick)))) == digest
    assert cylinder_ruling_batch_digest(
        query_cylinder_ruling_batch(model, pick)) == digest
    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_ruling_batch(model, iter(pick))
    assert refused.value.code.value == "INVALID_REQUEST"
    # Order independence is canonicalization, not accident: a different set
    # of occurrences produces a different digest.
    other = query_cylinder_ruling_batch(model, pick[:2])
    assert cylinder_ruling_batch_digest(other) != digest


def test_malformed_and_duplicate_selections_refuse_typed():
    model, faces, rulings, _, _ = standard()
    face, edge = faces[0], rulings[faces[0]]
    with pytest.raises(TypeError):
        query_cylinder_ruling_batch("model", [(face, edge)])
    for bad in ("selection", 7, None, {"face": edge}, {},
                [(face,)], [(face, edge, face)], [(face, edge), "pair"],
                [(True, edge)], [(0, edge)], [(face, 0)], [(1.5, edge)],
                [(face, 2.5)], [None], [(face, edge), (face, edge)]):
        with pytest.raises(CylinderAngularLiftError) as request:
            query_cylinder_ruling_batch(model, bad)
        assert request.value.code.value == "INVALID_REQUEST"
    with pytest.raises(CylinderAngularLiftError) as callback:
        query_cylinder_ruling_batch(model, [(face, edge)], cancellation_check=7)
    assert callback.value.code.value == "INVALID_REQUEST"
    with pytest.raises(CylinderAngularLiftError) as policy:
        query_cylinder_ruling_batch(model, [(face, edge)], policy=object())
    assert policy.value.code.value == "INVALID_REQUEST"
    with pytest.raises(CylinderAngularLiftError) as revision:
        query_cylinder_ruling_batch(model, [(face, edge)],
                                    expected_revision=model.revision + 3)
    assert revision.value.code.value == "STALE_REVISION"
    with pytest.raises(CylinderAngularLiftError) as negative:
        query_cylinder_ruling_batch(model, [(face, edge)], expected_revision=-1)
    assert negative.value.code.value == "INVALID_REQUEST"
    with model.transaction():
        with pytest.raises(CylinderAngularLiftError) as busy:
            query_cylinder_ruling_batch(model, [(face, edge)])
        assert busy.value.code.value == "BUSY_MODEL"


def test_unsupported_mixed_selections_refuse_the_whole_batch():
    model, faces, rulings, seams, arcs = standard()
    before = to_dict(model)
    ruling = (faces[0], rulings[faces[0]])
    # Mixed with a curved boundary edge, a wrong-owner edge, an inactive
    # entity: the whole batch refuses, no partial receipts escape.
    for mixed, reason in ((((ruling, (faces[1], arcs[faces[1]])), "curved_boundary_edge"),
                           ([ruling, (faces[0], 999)], "inactive_edge"),
                           ([ruling, (999, 1)], "inactive_face"),
                           ([ruling, (faces[0], rulings[faces[1]])], "edge_not_on_face_boundary"))):
        with pytest.raises(CylinderAngularLiftError) as refused:
            query_cylinder_ruling_batch(model, mixed)
        assert refused.value.code.value == "UNQUALIFIED"
        assert reason in refused.value.diagnostics[0]
    # An off-carrier rounded ruling refuses during compilation.
    rounded, r_faces, r_rulings, _, _ = panels((
        ((0.0, 0.0, 0.0), (3.0, 4.0)), ((10.0, 0.0, 0.0), (3.0, 4.0))), rounded=True)
    with pytest.raises(CylinderAngularLiftError) as off:
        query_cylinder_ruling_batch(
            rounded, [(r_faces[0], r_rulings[r_faces[0]]),
                      (r_faces[1], r_rulings[r_faces[1]])])
    assert off.value.code.value == "UNQUALIFIED"
    assert "ruling_off_carrier" in off.value.diagnostics[0]
    # Mixed with a non-Cylinder face support.
    mixed_model = GeometryModel()
    with mixed_model.transaction():
        surface, loop = panel(mixed_model, (0.0, 0.0, 0.0), (3.0, 4.0))
        cylinder_face = mixed_model.add_face_from_loop(loop, surface=surface)
        a = mixed_model.add_point(30.0, 0.0, 0.0)
        b = mixed_model.add_point(31.0, 0.0, 0.0)
        c = mixed_model.add_point(31.0, 1.0, 0.0)
        d = mixed_model.add_point(30.0, 1.0, 0.0)
        ab, bc = mixed_model.add_line(a, b), mixed_model.add_line(b, c)
        cd, da = mixed_model.add_line(c, d), mixed_model.add_line(d, a)
        flat = mixed_model.add_face_from_loop(
            (OrientedEdge(ab, True), OrientedEdge(bc, True),
             OrientedEdge(cd, True), OrientedEdge(da, True)),
            surface=Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        part = mixed_model.add_part(name="mixed")
        mixed_model.add_sheet((cylinder_face,), part_id=part)
        flat_part = mixed_model.add_part(name="flat")
        mixed_model.add_sheet((flat,), part_id=flat_part)
    assert mixed_model.validate_topology() == ()
    cylinder_ruling = next(
        use.edge for use in mixed_model.faces[cylinder_face].loop
        if mixed_model.edges[use.edge].curve.__class__.__name__ == "Straight"
        and abs(mixed_model.vertex_position(
            mixed_model.edges[use.edge].start)[0] - 5.0) > 1e-9)
    with pytest.raises(CylinderAngularLiftError) as support:
        query_cylinder_ruling_batch(mixed_model,
                                    [(cylinder_face, cylinder_ruling), (flat, ab)])
    assert support.value.code.value == "UNQUALIFIED"
    assert "cylinder_support_required" in support.value.diagnostics[0]
    assert to_dict(model) == before


def test_aggregate_budget_exhaustion_where_every_scalar_succeeds():
    model, faces, rulings, _, _ = standard()
    before = to_dict(model)
    pick = selection(model, faces, rulings)
    # One aggregate budget spans all operands: each scalar lift needs about
    # 164 counted interval operations and succeeds under a 300 budget, but
    # the shared batch total (343) exhausts it mid-batch.
    policy = CylinderAtlasPolicy(max_interval_operations=300)
    scalars = [query_cylinder_straight_ruling_lift(model, face, rulings[face],
                                                   policy=policy) for face in faces]
    assert all(dict(s.work_counts)["interval_operations"] <= 300 for s in scalars)
    with pytest.raises(CylinderAngularLiftError) as exhausted:
        query_cylinder_ruling_batch(model, pick, policy=policy)
    assert exhausted.value.code.value == "BUDGET_EXHAUSTED"
    assert "qualification_budget_exhausted:interval_operations" in exhausted.value.diagnostics[0]
    assert to_dict(model) == before
    # The aggregate budget is never reset per edge: a generous budget covers
    # the whole batch, and the cached repeated-ray operand still pays
    # counted charges (its cumulative counts strictly increase).
    full = query_cylinder_ruling_batch(model, pick)
    counts = [dict(item.work_counts)["interval_operations"] for item in full.items]
    assert counts == sorted(counts) and all(
        counts[k + 1] > counts[k] for k in range(len(counts) - 1))
    assert dict(full.work_counts)["interval_operations"] == counts[-1]
    # Shared scope efficiency without laundering: the batch total is bounded
    # by the sum of the standalone scalar totals (the pi enclosure is derived
    # once per batch, not once per operand).
    scalar_totals = [dict(query_cylinder_straight_ruling_lift(
        model, face, rulings[face]).work_counts)["interval_operations"] for face in faces]
    assert counts[-1] <= sum(scalar_totals)
    assert counts[-1] > max(scalar_totals)


def test_cancellation_refuses_without_partial_receipt_or_mutation():
    model, faces, rulings, _, _ = standard()
    before = to_dict(model)
    pick = selection(model, faces, rulings)
    calls = []

    def cancel(phase):
        calls.append(phase)
        return True

    with pytest.raises(CylinderAngularLiftError) as cancelled:
        query_cylinder_ruling_batch(model, pick, cancellation_check=cancel)
    assert cancelled.value.code.value == "CANCELLED"
    assert calls and to_dict(model) == before
    sentinel = RuntimeError("callback failure")

    def raising(phase):
        raise sentinel

    with pytest.raises(RuntimeError) as propagated:
        query_cylinder_ruling_batch(model, pick, cancellation_check=raising)
    assert propagated.value is sentinel
    assert to_dict(model) == before
    # A benign callback observes the batch phases and still publishes the
    # genuine deterministic receipt.
    phases = []
    receipt = query_cylinder_ruling_batch(
        model, pick, cancellation_check=lambda phase: phases.append(phase) or False)
    assert "cylinder ruling batch: request" in phases
    assert "cylinder ruling batch: final" in phases
    assert any("cylinder ruling batch item" in phase for phase in phases)
    assert cylinder_ruling_batch_digest(receipt) == cylinder_ruling_batch_digest(
        query_cylinder_ruling_batch(model, pick))


def test_binding_rederives_the_complete_batch_and_refuses_forgery():
    model, faces, rulings, _, _ = standard()
    pick = selection(model, faces, rulings)
    batch = query_cylinder_ruling_batch(model, pick)
    validate_cylinder_ruling_batch_binding(model, batch)
    validate_cylinder_ruling_batch_binding(
        model, batch, cancellation_check=lambda phase: False)
    with pytest.raises(TypeError):
        validate_cylinder_ruling_batch_binding("model", batch)
    with pytest.raises(CylinderAngularLiftError) as shape:
        validate_cylinder_ruling_batch_binding(model, "receipt")
    assert shape.value.code.value == "INVALID_RESULT"
    with pytest.raises(CylinderAngularLiftError) as scalar_shape:
        validate_cylinder_ruling_batch_binding(model, batch.items[0])
    assert scalar_shape.value.code.value == "INVALID_RESULT"
    # Stale after any committed edit.
    model.add_point(1.0, 2.0, 3.0)
    with pytest.raises(CylinderAngularLiftError) as stale:
        validate_cylinder_ruling_batch_binding(model, batch)
    assert stale.value.code.value == "STALE_REVISION"
    # Foreign model identity.
    other_model, other_faces, other_rulings, _, _ = standard()
    with pytest.raises(CylinderAngularLiftError) as foreign:
        validate_cylinder_ruling_batch_binding(other_model, batch)
    assert foreign.value.code.value == "WRONG_MODEL"
    # Forged batch content: plausible but wrong receipts fail live
    # rederivation of the COMPLETE batch.
    fresh = query_cylinder_ruling_batch(other_model,
                                        selection(other_model, other_faces, other_rulings))
    forgeries = (
        replace(fresh, items=(replace(fresh.items[0], winding=1), *fresh.items[1:])),
        replace(fresh, items=fresh.items[:-1]),
        replace(fresh, items=tuple(reversed(fresh.items))),
        replace(fresh, items=(*fresh.items[1:], fresh.items[0])),
        replace(fresh, source_checksum="0" * 64),
        replace(fresh, policy_limits=(256, 4096, 8192, 300, 65536, 4096)),
        replace(fresh, work_counts=tuple(
            (name, value + 1) for name, value in fresh.work_counts)),
        replace(fresh, items=(replace(fresh.items[-1], v_end=F(1, 2)), *fresh.items[:-1])),
    )
    for forgery in forgeries:
        with pytest.raises(CylinderAngularLiftError) as forged:
            validate_cylinder_ruling_batch_binding(other_model, forgery)
        assert forged.value.code.value in ("CHANGED_MODEL", "INVALID_RESULT")
    # A standalone scalar receipt is not batch item evidence: its fresh
    # per-query work counts differ from the batch's cumulative counts.
    scalar = query_cylinder_straight_ruling_lift(
        other_model, other_faces[0], other_rulings[other_faces[0]])
    with pytest.raises(CylinderAngularLiftError) as embedded:
        validate_cylinder_ruling_batch_binding(
            other_model, replace(fresh, items=(scalar, *fresh.items[1:])))
    assert embedded.value.code.value == "CHANGED_MODEL"


def test_batch_items_never_masquerade_as_scalar_receipts():
    model, faces, rulings, _, _ = standard()
    pick = selection(model, faces, rulings)
    batch = query_cylinder_ruling_batch(model, pick)
    for item, face in zip(batch.items, faces):
        scalar = query_cylinder_straight_ruling_lift(model, face, rulings[face])
        # Identical mathematical evidence, but the batch item carries the
        # cumulative aggregate work provenance, so the digests differ and the
        # scalar binding refuses the batch item.
        assert item.ray == scalar.ray and item.winding == scalar.winding
        assert dict(item.work_counts)["cancellation_checks"] > (
            dict(scalar.work_counts)["cancellation_checks"])
        assert cylinder_ruling_lift_digest(item) != cylinder_ruling_lift_digest(scalar)
        with pytest.raises(CylinderAngularLiftError) as refused:
            validate_cylinder_straight_ruling_lift_binding(model, item)
        assert refused.value.code.value == "CHANGED_MODEL"
    # Even a single-operand batch item is distinguishable: the batch adds its
    # own request and final cancellation checks.
    single = query_cylinder_ruling_batch(model, pick[:1])
    scalar = query_cylinder_straight_ruling_lift(model, *pick[0])
    assert dict(single.items[0].work_counts)["cancellation_checks"] == (
        dict(scalar.work_counts)["cancellation_checks"] + 1)
    with pytest.raises(CylinderAngularLiftError):
        validate_cylinder_straight_ruling_lift_binding(model, single.items[0])


def test_forged_batch_repaired_in_flight_cannot_pass_binding():
    model, faces, rulings, _, _ = standard()
    pick = selection(model, faces, rulings)
    genuine = query_cylinder_ruling_batch(model, pick)
    forged = replace(genuine, items=tuple(
        replace(item, winding=item.winding + 1) for item in genuine.items))

    def repair(phase):
        if "final" in phase:
            object.__setattr__(forged.items[0], "winding", genuine.items[0].winding)
        return False

    # The entry digest was pinned before any callback, so repairing the
    # forged item inside the callback cannot make the receipt match.
    object.__setattr__(forged.items[0], "winding", genuine.items[0].winding + 1)
    with pytest.raises(CylinderAngularLiftError) as refused:
        validate_cylinder_ruling_batch_binding(model, forged, cancellation_check=repair)
    assert refused.value.code.value == "CHANGED_MODEL"


def test_final_replacement_surface_and_private_coefficient_refuse():
    model, faces, rulings, _, _ = standard()
    from anygeometry.serialization import _serialized_model_state
    before = to_dict(model)
    surface = model.faces[faces[0]].surface
    checksum = _serialized_model_state(model)["checksum"]["value"]
    replacement = Cylinder(surface.origin, surface.axis, surface.radial_direction,
                           surface.radius, surface.height, surface.start_angle,
                           surface.sweep_angle)
    forged = np.array([0.0, -1.0, 0.0])
    forged.flags.writeable = False
    object.__setattr__(replacement, "_circumferential", forged)

    def replace_surface(phase):
        if "final" in phase:
            object.__setattr__(model.faces[faces[0]], "surface", replacement)
        return False

    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_ruling_batch(
            model, selection(model, faces, rulings),
            cancellation_check=replace_surface)
    assert refused.value.code.value == "CHANGED_MODEL"
    # The serialized document is unchanged: the raw frame pin refused what
    # the checksum alone cannot see.  The adversary's installation persists,
    # so restore the authored surface before further queries.
    assert _serialized_model_state(model)["checksum"]["value"] == checksum
    object.__setattr__(model.faces[faces[0]], "surface", surface)
    assert to_dict(model) == before
    # The same counterexample refuses through the binding validator.
    genuine = query_cylinder_ruling_batch(model, selection(model, faces, rulings))
    with pytest.raises(CylinderAngularLiftError) as validation:
        validate_cylinder_ruling_batch_binding(
            model, genuine, cancellation_check=replace_surface)
    assert validation.value.code.value == "CHANGED_MODEL"
    object.__setattr__(model.faces[faces[0]], "surface", surface)
    assert to_dict(model) == before


def test_temporary_callback_mutations_publish_entry_evidence_only():
    model, faces, rulings, _, _ = standard()
    pick = selection(model, faces, rulings)
    genuine = query_cylinder_ruling_batch(model, pick)
    vertex = model.vertices[model.edges[rulings[faces[0]]].start]
    original = vertex.position

    def adversary(phase):
        if "request" in phase and "item" not in phase:
            moved = np.array([9.0, 9.0, 9.0])
            moved.flags.writeable = False
            object.__setattr__(vertex, "position", moved)
        if "final" in phase:
            object.__setattr__(vertex, "position", original)
        return False

    receipt = query_cylinder_ruling_batch(model, pick, cancellation_check=adversary)
    assert cylinder_ruling_batch_digest(receipt) == cylinder_ruling_batch_digest(genuine)
    # A private circumferential mutation that is fully restored before the
    # final pins also never enters the published evidence.
    surface = model.faces[faces[1]].surface
    stored = surface._circumferential

    def transient(phase):
        if "request" in phase and "item" not in phase:
            forged = np.array([0.0, -1.0, 0.0])
            forged.flags.writeable = False
            object.__setattr__(surface, "_circumferential", forged)
        if "final" in phase:
            object.__setattr__(surface, "_circumferential", stored)
        return False

    receipt = query_cylinder_ruling_batch(model, pick, cancellation_check=transient)
    assert cylinder_ruling_batch_digest(receipt) == cylinder_ruling_batch_digest(genuine)
    # A visible revision bump at any callback refuses at the cheap guard.
    def bumping(phase):
        if "request" in phase and "item" not in phase:
            model.add_point(11.0, 12.0, 13.0)
        return False

    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_ruling_batch(model, pick, cancellation_check=bumping)
    assert refused.value.code.value == "CHANGED_MODEL"
    # A finally installed private coefficient change refuses at the final pins.
    def installed(phase):
        if "final" in phase:
            forged = np.array([0.0, -1.0, 0.0])
            forged.flags.writeable = False
            object.__setattr__(surface, "_circumferential", forged)
        return False

    with pytest.raises(CylinderAngularLiftError) as late:
        query_cylinder_ruling_batch(model, pick, cancellation_check=installed)
    assert late.value.code.value == "CHANGED_MODEL"


def test_no_quadratic_surface_rescans_at_arithmetic_callbacks(monkeypatch):
    model, faces, rulings, _, _ = standard()
    pick = selection(model, faces, rulings)
    calls = {"pins": 0, "phases": 0}
    original = lifts._raw_pin

    def counting(surface):
        calls["pins"] += 1
        return original(surface)

    monkeypatch.setattr(lifts, "_raw_pin", counting)
    monkeypatch.setattr(batch_module, "_raw_pin", counting)

    def benign(phase):
        calls["phases"] += 1
        return False

    receipt = query_cylinder_ruling_batch(model, pick, cancellation_check=benign)
    # One capture pin and one final pin per DISTINCT selected surface, no
    # matter how many cancellation callbacks the counted arithmetic fires.
    assert calls["pins"] == 2 * len(faces)
    assert calls["phases"] > 2 * len(faces)
    validate_cylinder_ruling_batch_binding(model, receipt, cancellation_check=benign)
    assert calls["pins"] == 2 * len(faces) * 2
    assert calls["phases"] > 2 * len(faces) * 2


def test_batch_module_claims_no_broader_acceptance():
    import anygeometry.cylinder_ruling_batch as module
    assert not any("partition" in name for name in dir(module) if not name.startswith("_"))
    from anygeometry import validate_prepared_authored_face_partition
    model, faces, rulings, _, _ = standard()
    with pytest.raises(GeometryError):
        validate_prepared_authored_face_partition(model, None, {})


@pytest.mark.parametrize("inside_pair", [False, True])
def test_lazy_request_is_rejected_without_consumption(inside_pair):
    model, faces, rulings, _, _ = standard()

    class NeverConsume:
        def __iter__(self):
            raise AssertionError("unbounded request must not be consumed")

    request = [NeverConsume()] if inside_pair else NeverConsume()
    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_ruling_batch(model, request)
    assert refused.value.code.value == "INVALID_REQUEST"


def test_revision_edit_during_finite_id_normalization_refuses():
    model, faces, rulings, _, _ = standard()
    revision = model.revision

    class EditingId(int):
        def __int__(self):
            model.add_point(99.0, 99.0, 99.0)
            return super().__int__()

    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_ruling_batch(
            model, [(EditingId(faces[0]), rulings[faces[0]])],
            expected_revision=revision)
    assert refused.value.code.value in ("STALE_REVISION", "CHANGED_MODEL")


@pytest.mark.parametrize("change_pair", [False, True])
def test_request_snapshot_precedes_all_id_coercions(change_pair):
    model, faces, rulings, _, _ = standard()
    request = []

    class EditingId(int):
        def __int__(self):
            if change_pair:
                request[1][:] = [faces[2], rulings[faces[2]]]
            else:
                request.append([faces[1], rulings[faces[1]]])
            return super().__int__()

    request.append([EditingId(faces[0]), rulings[faces[0]]])
    expected = [(faces[0], rulings[faces[0]])]
    if change_pair:
        request.append([faces[1], rulings[faces[1]]])
        expected.append((faces[1], rulings[faces[1]]))
    receipt = query_cylinder_ruling_batch(model, request)
    assert [(item.face_id, item.edge_id) for item in receipt.items] == sorted(expected)


@pytest.mark.parametrize("kind", ["empty", "missing", "duplicate", "unknown"])
@pytest.mark.parametrize("item_counts", [False, True])
def test_incomplete_or_noncanonical_work_inventory_refuses_typed(kind, item_counts):
    model, faces, rulings, _, _ = standard()
    receipt = query_cylinder_ruling_batch(model, selection(model, faces, rulings))
    source = receipt.items[0].work_counts if item_counts else receipt.work_counts
    bad = {"empty": (), "missing": source[:-1],
           "duplicate": source + (source[0],),
           "unknown": source + (("invented", 0),)}[kind]
    if item_counts:
        forged = replace(receipt, items=(replace(receipt.items[0], work_counts=bad),
                                         *receipt.items[1:]))
    else:
        forged = replace(receipt, work_counts=bad)
    with pytest.raises(CylinderAngularLiftError) as refused:
        validate_cylinder_ruling_batch_binding(model, forged)
    assert refused.value.code.value == "INVALID_RESULT"
