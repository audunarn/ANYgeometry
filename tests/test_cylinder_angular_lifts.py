"""Exact ray ordering and authenticated Cylinder straight-ruling lift proofs.

Independent checks: Decimal-series atan oracle, float evaluate identity along
the whole analytic line, and refusal/binding tampering. No partition,
source-current, reference or meshing acceptance is claimed here.
"""
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError, GeometryModel, OrientedEdge, Plane
from anygeometry.cylinder_angular_lifts import (
    RAY_HALF_TURN,
    RAY_QUARTER_TURN,
    RAY_THREE_QUARTER_TURN,
    RAY_ZERO,
    CylinderAngularLiftError,
    CylinderRulingLift,
    compare_exact_rays,
    cylinder_ruling_lift_digest,
    exact_ray,
    query_cylinder_straight_ruling_lift,
    validate_cylinder_straight_ruling_lift_binding,
)
from anygeometry.cylinder_charts import CylinderAtlasPolicy


def panel(*, start=0.0, sweep=1.0, radius=5.0, height=2.0, rounded=False,
          axis=(0.0, 0.0, 1.0), radial=(7.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0)):
    """Radius-5 canonical Z panel: circle through (5,0),(4,3),(3,4) plus top
    and vertical sides. ``rounded`` moves the far ruling to a rounded 45
    degree footpoint that is off the exact carrier."""
    model = GeometryModel()
    surface = Cylinder(origin, axis, radial, radius, height, start, sweep)
    foot = (5.0 / math.sqrt(2.0), 5.0 / math.sqrt(2.0)) if rounded else (3.0, 4.0)
    base = [(5.0, 0.0), (4.0, 3.0), foot]
    points = {}
    for x, y in base:
        for z in (0.0, height):
            points[(x, y, z)] = model.add_point(origin[0] + x, origin[1] + y, origin[2] + z)
    bottom = model.add_arc(points[(5.0, 0.0, 0.0)], points[(4.0, 3.0, 0.0)], points[(foot[0], foot[1], 0.0)])
    ruling = model.add_line(points[(foot[0], foot[1], 0.0)], points[(foot[0], foot[1], height)])
    top = model.add_arc(points[(foot[0], foot[1], height)], points[(4.0, 3.0, height)], points[(5.0, 0.0, height)])
    seam = model.add_line(points[(5.0, 0.0, height)], points[(5.0, 0.0, 0.0)])
    with model.transaction():
        face = model.add_face_from_loop(
            (OrientedEdge(bottom, True), OrientedEdge(ruling, True),
             OrientedEdge(top, True), OrientedEdge(seam, True)),
            surface=surface)
        part = model.add_part(name="panel")
        model.add_sheet((face,), part_id=part)
    assert model.validate_topology() == ()
    return model, face, ruling, seam, bottom, surface


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


def test_exact_ray_canonicalization_and_cardinal_ordering():
    assert exact_ray(6, 8) == exact_ray(F(3), F(4)) == exact_ray(F(3, 5), F(4, 5))
    assert exact_ray(3, 4) != exact_ray(4, 3)
    assert exact_ray(-3, -4) == exact_ray(F(-9, 5), F(-12, 5))
    cardinals = (RAY_ZERO, RAY_QUARTER_TURN, RAY_HALF_TURN, RAY_THREE_QUARTER_TURN)
    assert cardinals == (exact_ray(1, 0), exact_ray(0, 1), exact_ray(-1, 0), exact_ray(0, -1))
    order = {RAY_ZERO: 0, RAY_QUARTER_TURN: 1, RAY_HALF_TURN: 2, RAY_THREE_QUARTER_TURN: 3}
    for first in cardinals:
        for second in cardinals:
            expected = (order[first] > order[second]) - (order[first] < order[second])
            assert compare_exact_rays(first, second) == expected
    # Non-cardinal exact order inside one half plane: atan2(3,4) < atan2(4,3).
    assert compare_exact_rays(exact_ray(4, 3), exact_ray(3, 4)) == -1
    assert compare_exact_rays(exact_ray(3, 4), exact_ray(4, 3)) == 1
    # Cross half-plane order is exact for irrational angles.
    assert compare_exact_rays(exact_ray(3, 4), RAY_HALF_TURN) == -1
    assert compare_exact_rays(RAY_THREE_QUARTER_TURN, exact_ray(3, 4)) == 1
    assert compare_exact_rays(exact_ray(3, -4), RAY_ZERO) == 1
    with pytest.raises(GeometryError):
        exact_ray(0, 0)
    with pytest.raises(GeometryError):
        exact_ray(0.5, 0.5)
    with pytest.raises(GeometryError):
        compare_exact_rays(RAY_ZERO, "ray")


def test_positive_lift_matches_decimal_oracle_and_full_line_identity():
    model, face, ruling, seam, _, surface = panel()
    before = model.features
    receipt = query_cylinder_straight_ruling_lift(model, face, ruling)
    assert receipt.ray == exact_ray(3, 4)
    assert receipt.winding == 0
    assert (receipt.v_start, receipt.v_end) == (F(0), F(1))
    assert receipt.support_interval == (F(0), F(1))
    assert receipt.start_angle == F(0) and receipt.sweep_angle == F(1)
    assert receipt.radius == F(5) and receipt.height == F(2)
    assert receipt.frame_determinant == F(50)
    assert receipt.loop_index == 0 and receipt.forward is True
    # The exact angle atan2(4,3) is irrational; its enclosure is an interval.
    low, high = receipt.angle_enclosure
    oracle = decimal_atan(4, 3)
    assert low <= oracle + Decimal("1e-50") and oracle - Decimal("1e-50") <= high
    assert high - low < F(1, 2**70)
    assert F(0) < low and high < F(1)
    # Whole analytic line identity: constant chart u, linear v, on the carrier.
    u = math.atan2(4.0, 3.0)
    start = np.array([3.0, 4.0, 0.0])
    end = np.array([3.0, 4.0, 2.0])
    for t in (-0.5, 0.0, 0.25, 0.5, 1.0, 1.5):
        point = start + t * (end - start)
        v = float(receipt.v_start + F(t) * (receipt.v_end - receipt.v_start))
        assert np.allclose(surface.evaluate(u, v), point, atol=1e-12)
        projected_u, projected_v = surface.local_uv(point)
        assert abs(projected_u - u) < 1e-12 and abs(projected_v - v) < 1e-12
    # The seam ruling at the exact cardinal angle zero certifies exactly.
    seam_receipt = query_cylinder_straight_ruling_lift(model, face, seam)
    assert seam_receipt.ray == RAY_ZERO
    assert seam_receipt.angle_enclosure == (F(0), F(0))
    assert seam_receipt.winding == 0
    # Queries never author or mutate geometry.
    from anygeometry import to_dict
    document = to_dict(model)
    query_cylinder_straight_ruling_lift(model, face, ruling)
    assert to_dict(model) == document


def test_negative_sweep_and_periodic_gauges_certify_without_clamping():
    # Same physical panel authored with a negative sweep.
    model, face, ruling, _, _, _ = panel(start=1.0, sweep=-1.0)
    receipt = query_cylinder_straight_ruling_lift(model, face, ruling)
    assert receipt.sweep_angle == F(-1)
    assert receipt.support_interval == (F(0), F(1))
    assert receipt.ray == exact_ray(3, 4) and receipt.winding == 0
    # Far periodic gauges certify through nonzero windings.
    forward, face_f, ruling_f, _, _, _ = panel(start=math.tau, sweep=1.0)
    lifted = query_cylinder_straight_ruling_lift(forward, face_f, ruling_f)
    assert lifted.winding == 1 and lifted.ray == exact_ray(3, 4)
    backward, face_b, ruling_b, _, _, _ = panel(start=-math.tau, sweep=1.0)
    lowered = query_cylinder_straight_ruling_lift(backward, face_b, ruling_b)
    assert lowered.winding == -1 and lowered.ray == exact_ray(3, 4)
    # A ruling outside the authored angular window refuses; no wrap or clamp.
    model, face, ruling, _, _, _ = panel()
    model.set_face_surface(face, Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 5.0, 2.0, 2.0, 1.0))
    with pytest.raises(CylinderAngularLiftError) as outside:
        query_cylinder_straight_ruling_lift(model, face, ruling)
    assert outside.value.code.value == "UNQUALIFIED"
    assert "ruling_angle_outside_support" in outside.value.diagnostics[0]


def test_off_carrier_rounded_ruling_and_curved_edges_refuse():
    model, face, ruling, seam, bottom, _ = panel(rounded=True)
    with pytest.raises(CylinderAngularLiftError) as rounded:
        query_cylinder_straight_ruling_lift(model, face, ruling)
    assert rounded.value.code.value == "UNQUALIFIED"
    assert "ruling_off_carrier" in rounded.value.diagnostics[0]
    # The circle-arc boundary edges are not straight rulings; no arc lift claimed.
    with pytest.raises(CylinderAngularLiftError) as curved:
        query_cylinder_straight_ruling_lift(model, face, bottom)
    assert curved.value.code.value == "UNQUALIFIED"
    assert "curved_boundary_edge" in curved.value.diagnostics[0]


def test_fullturn_parameterized_inactive_and_wrong_owner_refuse():
    model, face, ruling, seam, _, _ = panel()
    # Full-turn carrier is not a strict subturn chart.
    model.set_face_surface(face, Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 5.0, 2.0, 0.0, math.tau))
    with pytest.raises(CylinderAngularLiftError) as full:
        query_cylinder_straight_ruling_lift(model, face, ruling)
    assert "full_turn_sweep_refused" in full.value.diagnostics[0]
    model.set_face_surface(face, Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 5.0, 2.0, 0.0, 1.0))
    # Explicit parameterization refuses.
    model.set_face_parameterization(face, Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 5.0, 2.0, 0.0, 1.0))
    with pytest.raises(CylinderAngularLiftError) as mapped:
        query_cylinder_straight_ruling_lift(model, face, ruling)
    assert "explicit_parameterization_refused" in mapped.value.diagnostics[0]
    model.set_face_parameterization(face, None)
    # Inactive entities and wrong-owner occurrences refuse.
    for bad_face, bad_edge in ((999, 1), (face, 999)):
        with pytest.raises(CylinderAngularLiftError) as inactive:
            query_cylinder_straight_ruling_lift(model, bad_face, bad_edge)
        assert inactive.value.code.value == "UNQUALIFIED"
    stranger = model.add_point(50.0, 0.0, 0.0)
    other = model.add_line(stranger, model.add_point(50.0, 0.0, 2.0))
    with pytest.raises(CylinderAngularLiftError) as owner:
        query_cylinder_straight_ruling_lift(model, face, other)
    assert "edge_not_on_face_boundary" in owner.value.diagnostics[0]
    # Non-Cylinder support refuses: an authored planar face with an explicit
    # Plane surface (the model API refuses removing the surface from the
    # non-planar cylinder panel, so the wrong-support case is authored
    # directly instead).
    flat = GeometryModel()
    with flat.transaction():
        a = flat.add_point(0.0, 0.0, 0.0)
        b = flat.add_point(1.0, 0.0, 0.0)
        c = flat.add_point(1.0, 1.0, 0.0)
        d = flat.add_point(0.0, 1.0, 0.0)
        ab = flat.add_line(a, b)
        bc = flat.add_line(b, c)
        cd = flat.add_line(c, d)
        da = flat.add_line(d, a)
        planar = flat.add_face_from_loop(
            (OrientedEdge(ab, True), OrientedEdge(bc, True),
             OrientedEdge(cd, True), OrientedEdge(da, True)),
            surface=Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        part = flat.add_part(name="flat")
        flat.add_sheet((planar,), part_id=part)
    assert flat.validate_topology() == ()
    with pytest.raises(CylinderAngularLiftError) as support:
        query_cylinder_straight_ruling_lift(flat, planar, ab)
    assert "cylinder_support_required" in support.value.diagnostics[0]


def test_invalid_requests_refuse_typed():
    model, face, ruling, _, _, _ = panel()
    with pytest.raises(TypeError):
        query_cylinder_straight_ruling_lift("model", face, ruling)
    for bad_face, bad_edge in ((True, 1), (0, 1), ("1", 1), (face, 2.0)):
        with pytest.raises(CylinderAngularLiftError) as request:
            query_cylinder_straight_ruling_lift(model, bad_face, bad_edge)
        assert request.value.code.value == "INVALID_REQUEST"
    with pytest.raises(CylinderAngularLiftError) as revision:
        query_cylinder_straight_ruling_lift(model, face, ruling, expected_revision=model.revision + 5)
    assert revision.value.code.value == "STALE_REVISION"
    with pytest.raises(CylinderAngularLiftError) as callback:
        query_cylinder_straight_ruling_lift(model, face, ruling, cancellation_check=7)
    assert callback.value.code.value == "INVALID_REQUEST"
    with pytest.raises(CylinderAngularLiftError) as policy:
        query_cylinder_straight_ruling_lift(model, face, ruling, policy=object())
    assert policy.value.code.value == "INVALID_REQUEST"
    with model.transaction():
        with pytest.raises(CylinderAngularLiftError) as busy:
            query_cylinder_straight_ruling_lift(model, face, ruling)
        assert busy.value.code.value == "BUSY_MODEL"


def test_cancellation_and_budget_refuse_without_partial_receipt():
    model, face, ruling, _, _, _ = panel()
    from anygeometry import to_dict
    before = to_dict(model)
    calls = []

    def cancel(phase):
        calls.append(phase)
        return True

    with pytest.raises(CylinderAngularLiftError) as cancelled:
        query_cylinder_straight_ruling_lift(model, face, ruling, cancellation_check=cancel)
    assert cancelled.value.code.value == "CANCELLED"
    assert calls and to_dict(model) == before
    with pytest.raises(CylinderAngularLiftError) as budget:
        query_cylinder_straight_ruling_lift(
            model, face, ruling, policy=CylinderAtlasPolicy(max_interval_operations=1))
    assert budget.value.code.value == "BUDGET_EXHAUSTED"
    assert to_dict(model) == before
    # A callback exception propagates unchanged.
    sentinel = RuntimeError("callback failure")

    def raising(phase):
        raise sentinel

    with pytest.raises(RuntimeError) as propagated:
        query_cylinder_straight_ruling_lift(model, face, ruling, cancellation_check=raising)
    assert propagated.value is sentinel


def test_binding_validation_rederives_and_detects_tampering():
    model, face, ruling, _, _, _ = panel()
    receipt = query_cylinder_straight_ruling_lift(model, face, ruling)
    assert type(receipt) is CylinderRulingLift
    assert dict(receipt.work_counts)["cancellation_checks"] >= 1
    assert receipt.policy_limits == (256, 4096, 8192, 200000, 65536, 4096)
    validate_cylinder_straight_ruling_lift_binding(model, receipt)
    validate_cylinder_straight_ruling_lift_binding(model, receipt, cancellation_check=lambda phase: False)
    # Stale: any new committed edit bumps the revision.
    model.add_point(1.0, 2.0, 3.0)
    with pytest.raises(CylinderAngularLiftError) as stale:
        validate_cylinder_straight_ruling_lift_binding(model, receipt)
    assert stale.value.code.value == "STALE_REVISION"
    # Forged content: a plausible but wrong receipt fails live rederivation.
    fresh_model, fresh_face, fresh_ruling, _, _, _ = panel()
    fresh = query_cylinder_straight_ruling_lift(fresh_model, fresh_face, fresh_ruling)
    for forgery in (replace(fresh, winding=fresh.winding + 1),
                    replace(fresh, ray=exact_ray(4, 3)),
                    replace(fresh, v_end=F(1, 2)),
                    replace(fresh, source_checksum="0" * 64),
                    replace(fresh, sweep_angle=F(2))):
        with pytest.raises(CylinderAngularLiftError) as forged:
            validate_cylinder_straight_ruling_lift_binding(fresh_model, forgery)
        assert forged.value.code.value in ("CHANGED_MODEL", "INVALID_RESULT")
    # Foreign binding: another model's receipt refuses on identity.
    with pytest.raises(CylinderAngularLiftError) as foreign:
        validate_cylinder_straight_ruling_lift_binding(fresh_model, receipt)
    assert foreign.value.code.value == "WRONG_MODEL"
    with pytest.raises(CylinderAngularLiftError) as shape:
        validate_cylinder_straight_ruling_lift_binding(fresh_model, "receipt")
    assert shape.value.code.value == "INVALID_RESULT"
    # Same-revision content change without a revision bump refuses.
    tampered = query_cylinder_straight_ruling_lift(fresh_model, fresh_face, fresh_ruling)
    vertex = fresh_model.vertices[next(iter(fresh_model.vertices))]
    moved = np.array([9.0, 9.0, 9.0])
    moved.flags.writeable = False
    object.__setattr__(vertex, "position", moved)
    with pytest.raises(CylinderAngularLiftError) as changed:
        validate_cylinder_straight_ruling_lift_binding(fresh_model, tampered)
    assert changed.value.code.value == "CHANGED_MODEL"


def test_late_callback_mutation_refuses_even_without_revision():
    model, face, ruling, _, _, _ = panel()
    vertex_id = next(iter(model.vertices))

    def mutating(phase):
        if "final" in phase:
            moved = np.array([9.0, 9.0, 9.0])
            moved.flags.writeable = False
            object.__setattr__(model.vertices[vertex_id], "position", moved)
        return False

    with pytest.raises(CylinderAngularLiftError) as late:
        query_cylinder_straight_ruling_lift(model, face, ruling, cancellation_check=mutating)
    assert late.value.code.value == "CHANGED_MODEL"
    # A mid-query mutation is also caught; no partial receipt escapes.
    def early(phase):
        if "exact solve" in phase:
            moved = np.array([8.0, 8.0, 8.0])
            moved.flags.writeable = False
            object.__setattr__(model.vertices[vertex_id], "position", moved)
        return False

    with pytest.raises(CylinderAngularLiftError):
        query_cylinder_straight_ruling_lift(model, face, ruling, cancellation_check=early)


def test_benign_callback_genuine_positive_whole_ruling():
    model, face, ruling, _, _, surface = panel()
    genuine = query_cylinder_straight_ruling_lift(model, face, ruling)
    phases = []

    def benign(phase):
        phases.append(phase)
        return False

    receipt = query_cylinder_straight_ruling_lift(
        model, face, ruling, cancellation_check=benign)
    assert cylinder_ruling_lift_digest(receipt) == cylinder_ruling_lift_digest(genuine)
    assert "cylinder ruling lift: final" in phases
    # Genuine positive whole ruling under callbacks: the full analytic line.
    u = math.atan2(4.0, 3.0)
    start = np.array([3.0, 4.0, 0.0])
    end = np.array([3.0, 4.0, 2.0])
    for t in (-0.5, 0.0, 0.5, 1.0, 1.5):
        point = start + t * (end - start)
        v = float(receipt.v_start + F(t) * (receipt.v_end - receipt.v_start))
        assert np.allclose(surface.evaluate(u, v), point, atol=1e-12)


def test_mutate_first_restore_later_adversary_publishes_entry_evidence_only():
    model, face, ruling, _, _, _ = panel()
    genuine = query_cylinder_straight_ruling_lift(model, face, ruling)
    vertex = model.vertices[model.edges[ruling].start]
    original = vertex.position

    def adversary(phase):
        if "request" in phase:
            moved = np.array([9.0, 9.0, 9.0])
            moved.flags.writeable = False
            object.__setattr__(vertex, "position", moved)
        if "final" in phase:
            object.__setattr__(vertex, "position", original)
        return False

    receipt = query_cylinder_straight_ruling_lift(
        model, face, ruling, cancellation_check=adversary)
    # The published receipt is the genuine immutable entry evidence captured
    # before any callback, never the temporary geometry the adversary
    # installed mid-query and restored before the freshness check.
    assert cylinder_ruling_lift_digest(receipt) == cylinder_ruling_lift_digest(genuine)
    assert receipt.ray == exact_ray(3, 4)
    assert (receipt.v_start, receipt.v_end) == (F(0), F(1))


def test_visible_raw_frame_mutation_refuses_even_if_restore_planned():
    model, face, ruling, _, _, surface = panel()
    original = surface._circumferential

    def adversary(phase):
        if "request" in phase:
            forged = np.array([0.0, -1.0, 0.0])
            forged.flags.writeable = False
            object.__setattr__(surface, "_circumferential", forged)
        if "final" in phase:
            object.__setattr__(surface, "_circumferential", original)
        return False

    # The guard after the very first callback sees the raw frame mutation,
    # so the planned later restore is never reached.
    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_straight_ruling_lift(
            model, face, ruling, cancellation_check=adversary)
    assert refused.value.code.value == "CHANGED_MODEL"


def test_visible_callback_revision_mutation_refuses_at_guard():
    model, face, ruling, _, _, _ = panel()

    def bumping(phase):
        if "request" in phase:
            model.add_point(11.0, 12.0, 13.0)
        return False

    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_straight_ruling_lift(
            model, face, ruling, cancellation_check=bumping)
    assert refused.value.code.value == "CHANGED_MODEL"


def test_late_private_circumferential_mutation_typed_refuses():
    model, face, ruling, _, _, surface = panel()
    from anygeometry.serialization import _serialized_model_state
    checksum = _serialized_model_state(model)["checksum"]["value"]

    def late(phase):
        if "final" in phase:
            forged = np.array([0.0, -1.0, 0.0])
            forged.flags.writeable = False
            object.__setattr__(surface, "_circumferential", forged)
        return False

    with pytest.raises(CylinderAngularLiftError) as refused:
        query_cylinder_straight_ruling_lift(
            model, face, ruling, cancellation_check=late)
    assert refused.value.code.value == "CHANGED_MODEL"
    # Serialization excludes this coefficient: the document checksum alone
    # cannot see the mutation, so the raw frame pin is what refuses.
    assert _serialized_model_state(model)["checksum"]["value"] == checksum


@pytest.mark.parametrize("validate", [False, True])
def test_final_replacement_surface_raw_coefficient_refuses(validate):
    model, face, ruling, _, _, surface = panel()
    from anygeometry.serialization import _serialized_model_state
    receipt = query_cylinder_straight_ruling_lift(model, face, ruling)
    checksum = _serialized_model_state(model)["checksum"]["value"]
    replacement = Cylinder(surface.origin, surface.axis,
                           surface.radial_direction, surface.radius,
                           surface.height, surface.start_angle,
                           surface.sweep_angle)
    forged = np.array([0.0, -1.0, 0.0])
    forged.flags.writeable = False
    object.__setattr__(replacement, "_circumferential", forged)

    def replace_surface(phase):
        if "final" in phase:
            object.__setattr__(model.faces[face], "surface", replacement)
        return False

    with pytest.raises(CylinderAngularLiftError) as refused:
        if validate:
            validate_cylinder_straight_ruling_lift_binding(
                model, receipt, cancellation_check=replace_surface)
        else:
            query_cylinder_straight_ruling_lift(
                model, face, ruling, cancellation_check=replace_surface)
    assert refused.value.code.value == "CHANGED_MODEL"
    assert _serialized_model_state(model)["checksum"]["value"] == checksum


def test_forged_winding_repaired_inside_callback_refuses():
    model, face, ruling, _, _, _ = panel()
    genuine = query_cylinder_straight_ruling_lift(model, face, ruling)
    forged = replace(genuine, winding=genuine.winding + 1)

    def repair(phase):
        if "final" in phase:
            object.__setattr__(forged, "winding", genuine.winding)
        return False

    # The entry digest was pinned before any callback, so repairing the
    # forged winding inside the callback cannot make the receipt match the
    # fresh derivation.
    with pytest.raises(CylinderAngularLiftError) as refused:
        validate_cylinder_straight_ruling_lift_binding(
            model, forged, cancellation_check=repair)
    assert refused.value.code.value == "CHANGED_MODEL"


def test_noncardinal_radial_counterexample_has_no_decode_renormalization():
    # axisZ with radial (1,1,0): the stored radial is the normalized
    # irrational-direction float triple, so a decode/renormalization round
    # trip would drift by an ulp and a tolerance path would certify the
    # authored (3,4) footpoint.  The raw capture must keep the stored floats
    # exactly, and the exact proof must refuse the off-carrier ruling.
    model, face, ruling, _, _, surface = panel(
        axis=(0.0, 0.0, 1.0), radial=(1.0, 1.0, 0.0), start=-0.8, sweep=1.0)
    import anygeometry.cylinder_angular_lifts as lifts
    frame = lifts._capture_frame(surface)
    assert frame["radial"] == tuple(F(float(v)) for v in surface.radial_direction)
    assert frame["circum"] == tuple(F(float(v)) for v in surface._circumferential)
    assert frame["pin"] == (frame["origin"], frame["axis"], frame["radial"],
                            frame["circum"], frame["radius"], frame["height"],
                            frame["start"], frame["sweep"])
    with pytest.raises(CylinderAngularLiftError) as off:
        query_cylinder_straight_ruling_lift(model, face, ruling)
    assert off.value.code.value == "UNQUALIFIED"
    assert "ruling_off_carrier" in off.value.diagnostics[0]


def test_raw_capture_has_no_normalization_drift():
    model, face, ruling, _, _, surface = panel(
        axis=(0.0, 0.0, 2.0), radial=(7.0, 0.0, 0.0), origin=(0.5, -0.25, 0.125))
    receipt = query_cylinder_straight_ruling_lift(model, face, ruling)
    # Captured coefficients are the live stored floats, exactly.
    assert receipt.origin == tuple(F(float(v)) for v in surface.origin)
    assert receipt.axis == tuple(F(float(v)) for v in surface.axis)
    assert receipt.radial_direction == tuple(F(float(v)) for v in surface.radial_direction)
    assert receipt.circumferential_direction == tuple(
        F(float(v)) for v in surface.circumferential_direction)
    assert np.array_equal(np.cross(surface.axis, surface.radial_direction),
                          surface.circumferential_direction)
    # The exact solve is faithful to the stored carrier, not a rebuild.
    assert receipt.ray == exact_ray(3, 4)
    assert (receipt.v_start, receipt.v_end) == (F(0), F(1))
    u = math.atan2(4.0, 3.0)
    for t in (0.0, 0.5, 1.0):
        point = np.array([3.5, 3.75, 0.125]) + t * np.array([0.0, 0.0, 2.0])
        assert np.allclose(surface.evaluate(u, t), point, atol=1e-12)


def test_existing_public_partition_refusal_is_unchanged():
    from anygeometry import validate_prepared_authored_face_partition
    model, face, ruling, _, _, _ = panel()
    with pytest.raises(GeometryError):
        validate_prepared_authored_face_partition(model, None, {})
    # The narrow lift producer opens no partition acceptance path.
    import anygeometry.cylinder_angular_lifts as module
    assert not any("partition" in name for name in dir(module) if not name.startswith("_"))
