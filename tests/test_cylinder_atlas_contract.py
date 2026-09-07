"""Cylinder atlas gates on public, topology-owned sector models."""

from fractions import Fraction
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from copy import copy
import math

import numpy as np
import pytest

from anygeometry import (
    Cylinder, GeometryModel, OrientedEdge, TolerancePolicy,
    CylinderAtlasPolicy, CylinderAtlasResult, CylinderAtlasStatus as S,
    CylinderAtlasError, CylinderAtlasErrorCode as E, CylinderOccurrenceRequest,
    query_cylinder_atlas, validate_cylinder_atlas_binding,
    evaluate_cylinder_occurrences,
)
from anygeometry import cylinder_charts as charts


def sector_hole_model(*, hole=True, scale=1., offset=(0., 0., 0.),
                      sweep_sign=1, axis_sign=1, height_sign=1,
                      hole_sector=0, local_hole=False, use_orientation=1,
                      equivalent_turns=0, frame_perturbation=None):
    """Author a seam-crossing hole as two notched, distinct sector Faces.

    The registry is keyed by the authored angular fraction and axial level,
    never by rounded positions. No private stores or validation bypasses.
    """
    model = GeometryModel(tolerance=TolerancePolicy().scaled(scale))
    vertices = {}
    edges = {}
    faces = []

    def key(angle, z):
        return Fraction(angle) % 8, Fraction(z)

    def vertex(angle, z):
        identity = key(angle, z)
        if identity not in vertices:
            theta = sweep_sign * float(identity[0]) * math.pi / 4
            position = (
                offset[0]+scale*math.cos(theta), offset[1]+scale*math.sin(theta),
                offset[2]+height_sign*scale*float(identity[1])
            )
            if frame_perturbation and 3 <= identity[0] <= 4:
                x, y, z = position
                epsilon = 2.e-11
                position = ((x+epsilon, y, z) if frame_perturbation == "offset" else
                            (math.cos(epsilon)*x+math.sin(epsilon)*z, y,
                             -math.sin(epsilon)*x+math.cos(epsilon)*z))
            vertices[identity] = model.add_point(*position)
        return vertices[identity]

    def edge(a, b):
        a_key, b_key = key(*a), key(*b)
        kind = "arc" if a[1] == b[1] else "line"
        identity = (kind, *sorted((a_key, b_key)))
        start, end = vertex(*a), vertex(*b)
        if identity not in edges:
            if kind == "arc":
                via = vertex((a[0] + b[0]) / 2, a[1])
                identifier = model.add_arc(start, via, end)
            else:
                assert a[0] == b[0]
                identifier = model.add_line(start, end)
            edges[identity] = identifier
        identifier = edges[identity]
        return OrientedEdge(identifier, model.edges[identifier].start == start)

    rectangle = ((0, 0), (1, 0), (1, 2), (0, 2))
    first_notch = (
        (0, 0), (1, 0), (1, 2), (0, 2),
        (0, 1.25), (.25, 1.25), (.25, .75), (0, .75),
    )
    last_notch = (
        (0, 0), (1, 0), (1, .75), (.75, .75),
        (.75, 1.25), (1, 1.25), (1, 2), (0, 2),
    )
    with model.transaction():
        for index in range(8):
            local_loop = first_notch if hole and index == hole_sector else last_notch if hole and index == (hole_sector-1)%8 else rectangle
            points = tuple((Fraction(index) + Fraction(u), Fraction(z)) for u, z in local_loop)
            loop = tuple(edge(a, b) for a, b in zip(points, points[1:] + points[:1]))
            support = Cylinder(
                origin=offset, axis=(0., 0., float(axis_sign)),
                radial_direction=(1., 0., 0.), radius=scale,
                height=height_sign/axis_sign*2*scale,
                start_angle=sweep_sign/axis_sign*index*math.pi/4 + equivalent_turns*2*math.pi,
                sweep_angle=sweep_sign/axis_sign*math.pi/4,
            )
            if frame_perturbation and index == 3:
                epsilon = 2.e-11
                support = (replace(support, origin=(epsilon, 0., 0.)) if frame_perturbation == "offset" else
                           replace(support, axis=(math.sin(epsilon),0.,math.cos(epsilon)),
                                   radial_direction=(math.cos(epsilon),0.,-math.sin(epsilon))))
            faces.append(model.add_face_from_loop(loop, surface=support))
            if local_hole and index == 3:
                from anygeometry import trim_face
                inner = tuple((Fraction(index)+Fraction(u), Fraction(z)) for u, z in
                              ((.25,.75), (.25,1.25), (.75,1.25), (.75,.75)))
                inner_loop = tuple(edge(a, b) for a, b in zip(inner, inner[1:]+inner[:1]))
                trim_face(model, faces[-1], (inner_loop,))
        part = model.add_part(name="sector atlas authoring gate")
        sheet = model.add_sheet(faces, part_id=part, name="seam-hole cylinder",
                                orientations=(use_orientation,)*len(faces))
    return model, tuple(faces), sheet, vertices, edges


def test_public_sector_hole_authoring():
    model, faces, sheet, vertices, edges = sector_hole_model()
    assert model.validate_topology() == ()
    assert len(faces) == 8 and len(set(faces)) == 8
    assert len(model.parts) == 1 and len(model.sheets) == 1
    assert len(model.face_uses) == 8
    assert all(use.sheet_id == sheet for use in model.face_uses.values())
    assert len(model.faces[faces[0]].loop) == len(model.faces[faces[-1]].loop) == 8
    assert all(isinstance(model.faces[face].surface, Cylinder) for face in faces)
    seam = []
    for low, high in ((0, .75), (1.25, 2)):
        a, b = (Fraction(0), Fraction(low)), (Fraction(0), Fraction(high))
        identifier = edges[("line", *sorted((a, b)))]
        assert set(model.faces_using_edge(identifier)) == {faces[0], faces[-1]}
        uses = model.coedges_using_edge(identifier)
        assert len(uses) == 2
        assert model.coedges[uses[0]].orientation != model.coedges[uses[1]].orientation
        seam.append(identifier)
    void_endpoints = {vertices[(Fraction(0), Fraction(z))] for z in (.75, 1.25)}
    assert not any({item.start, item.end} == void_endpoints for item in model.edges.values())
    assert len(set(seam)) == 2


def selection(model, sheet):
    return tuple(model.handle("face_use", key) for key in model.sheets[sheet].face_use_ids)


def snapshot(model):
    return repr(model.__dict__), tuple((key, item.position.tobytes()) for key, item in model.vertices.items())


def query(model, selected, *, reference=None, **options):
    return query_cylinder_atlas(model, selected, reference_face_use=reference or selected[0],
                               expected_revision=model.revision, **options)


def validate(model, atlas, selected, *, reference=None, **options):
    return validate_cylinder_atlas_binding(model, atlas, selected,
        reference_face_use=reference or selected[0], expected_revision=model.revision, **options)


def evaluate(model, atlas, selected, requests, **options):
    return evaluate_cylinder_occurrences(model, atlas, requests, face_uses=selected,
        reference_face_use=selected[0], expected_revision=model.revision, **options)


@pytest.fixture(scope="module")
def qualified():
    model, faces, sheet, vertices, edges = sector_hole_model()
    selected = selection(model, sheet)
    before = snapshot(model)
    result = query(model, selected)
    assert result.status is S.QUALIFIED, result.diagnostics
    assert snapshot(model) == before
    return model, selected, result, vertices, edges


def test_sector_hole_correspondence_and_purity(qualified):
    model, selected, atlas, vertices, edges = qualified
    before = snapshot(model)
    validate(model, atlas, selected)
    assert snapshot(model) == before
    assert len(atlas.sectors) == 8
    assert sorted(c.winding for c in atlas.boundary_cycles) == [-1, 0, 1]
    seam = [item for item in atlas.interfaces if item.is_reference_seam]
    expected = {edges[("line", *sorted(((Fraction(0), Fraction(a)), (Fraction(0), Fraction(b)))))]
                for a, b in ((0, .75), (1.25, 2))}
    assert {item.edge.id for item in seam} == expected
    assert all(abs(item.lift_delta) == 1 for item in seam)
    assert atlas.certificate.complete
    assert atlas.certificate.max_residual <= atlas.certificate.tolerance_surface
    assert dict(atlas.certificate.work_counts)["faces"] == 8


def test_native_parameter_samples_and_identity(qualified):
    model, selected, atlas, _, _ = qualified
    seam = next(item for item in atlas.interfaces if item.is_reference_seam)
    parameters = (0., .125, .5, .875, 1.)
    requests = tuple(CylinderOccurrenceRequest(occurrence, t) for t in parameters for occurrence in seam.occurrences)
    before = snapshot(model)
    result = evaluate(model, atlas, selected, requests)
    assert snapshot(model) == before
    assert len(result.samples) == len(requests)
    for first, second in zip(result.samples[::2], result.samples[1::2]):
        assert first.parameter == second.parameter
        assert first.equivalence_key == second.equivalence_key
        assert first.point == second.point
        owner_point = model.sample_edge(first.edge.id, np.asarray([first.parameter]))[0]
        assert np.linalg.norm(np.asarray(first.point)-owner_point) <= atlas.certificate.tolerance_surface
        assert abs(abs(first.lifted_reference[0]-second.lifted_reference[0])-1) < 1.e-12
        assert first.endpoint_vertex is not None if first.parameter in (0, 1) else first.endpoint_vertex is None
    assert len({row.equivalence_key for row in result.samples}) == len(parameters)


@pytest.mark.parametrize("options", [
    {"hole": False}, {"sweep_sign": -1}, {"axis_sign": -1},
    {"height_sign": -1}, {"scale": 1.e-6}, {"scale": 1.e6},
    {"offset": (1.e6, -2.e6, 3.e6)},
])
def test_analytic_sector_covariance(options):
    model, _, sheet, _, _ = sector_hole_model(**options)
    selected = selection(model, sheet)
    before = snapshot(model)
    atlas = query(model, selected)
    assert atlas.status is S.QUALIFIED, atlas.diagnostics
    assert snapshot(model) == before
    expected = [-1, 1] if options.get("hole") is False else [-1, 0, 1]
    assert sorted(c.winding for c in atlas.boundary_cycles) == expected
    assert atlas.certificate.tolerance_surface == pytest.approx(1.e-8*options.get("scale", 1.))


def test_reference_and_order_binding(qualified):
    model, selected, atlas, _, _ = qualified
    before = snapshot(model)
    for arguments in ((selected[::-1], selected[0]), (selected, selected[1])):
        with pytest.raises(CylinderAtlasError) as caught:
            validate(model, atlas, arguments[0], reference=arguments[1])
        assert caught.value.code is E.INVALID_RESULT
    alternate = query(model, selected, reference=selected[3])
    assert alternate.status is S.QUALIFIED, alternate.diagnostics
    assert alternate.digest != atlas.digest
    assert snapshot(model) == before


def test_wrong_model_and_stale_binding():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    atlas = query(model, selected)
    other = GeometryModel()
    with pytest.raises(CylinderAtlasError) as caught:
        query_cylinder_atlas(other, selected, reference_face_use=selected[0], expected_revision=other.revision)
    assert caught.value.code is E.WRONG_MODEL
    model.add_point(9., 9., 9.)
    before = snapshot(model)
    with pytest.raises(CylinderAtlasError) as caught:
        validate(model, atlas, selected)
    assert caught.value.code is E.STALE_RESULT
    assert snapshot(model) == before


@pytest.mark.parametrize("revision", [True, 1.5, "1", -1])
def test_strict_revision_rejection(qualified, revision):
    model, selected, _, _, _ = qualified
    before = snapshot(model)
    with pytest.raises(CylinderAtlasError) as caught:
        query_cylinder_atlas(model, selected, reference_face_use=selected[0], expected_revision=revision)
    assert caught.value.code is E.INVALID_REQUEST
    assert snapshot(model) == before


@pytest.mark.parametrize("parameter", [True, float("nan"), float("inf"), -.1, 1.1])
def test_strict_parameter_rejection(parameter):
    with pytest.raises(CylinderAtlasError):
        CylinderOccurrenceRequest("coedge/1/whole", parameter)


def test_result_immutability_and_rehashed_tampering(qualified):
    model, selected, atlas, _, _ = qualified
    with pytest.raises(FrozenInstanceError):
        atlas.revision = 5
    with pytest.raises(TypeError):
        atlas.certificate.work_counts[0] = ("faces", 99)
    altered = replace(atlas, diagnostics=("forged success",))
    data = {field.name: getattr(altered, field.name) for field in charts.fields(altered) if field.name != "digest"}
    altered = replace(altered, digest=charts._digest(data))
    before = snapshot(model)
    with pytest.raises(CylinderAtlasError) as caught:
        validate(model, altered, selected)
    assert caught.value.code is E.INVALID_RESULT
    assert snapshot(model) == before


@pytest.mark.parametrize("api", ["query", "validate", "evaluate"])
def test_cancellation_exception_identity(qualified, api):
    model, selected, atlas, _, _ = qualified
    before = snapshot(model)
    marker = RuntimeError("cancel marker")
    def cancel(_phase):
        raise marker
    with pytest.raises(RuntimeError) as caught:
        if api == "query":
            query(model, selected, cancellation_check=cancel)
        elif api == "validate":
            validate(model, atlas, selected, cancellation_check=cancel)
        else:
            evaluate(model, atlas, selected, (), cancellation_check=cancel)
    assert caught.value is marker
    assert snapshot(model) == before


def test_owner_frame_operational_exception_is_not_swallowed(monkeypatch):
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    marker = RuntimeError("owner frame operational marker")
    def broken(*args):
        raise marker
    monkeypatch.setattr(charts, "arc_frame", broken)
    before = snapshot(model)
    with pytest.raises(RuntimeError) as caught:
        query(model, selected)
    assert caught.value is marker
    assert snapshot(model) == before


def test_tiny_budget_has_no_partial_atlas():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    before = snapshot(model)
    result = query(model, selected, policy=CylinderAtlasPolicy(max_interval_operations=1))
    assert result.status is S.UNRESOLVED and not result.certificate.complete
    assert not (result.sectors or result.occurrences or result.interfaces or result.boundary_cycles)
    assert snapshot(model) == before


def test_interval_arithmetic_independent_bounds():
    proof = charts._Proof(CylinderAtlasPolicy(), None)
    interval = proof.mul(proof.i(Fraction(-2, 3), Fraction(7, 5)), proof.i(Fraction(1, 7), Fraction(11, 13)))
    assert interval.lo <= Fraction(-22, 39)
    assert interval.hi >= Fraction(77, 65)
    root = proof.sqrt(2)
    assert root.lo**2 <= 2 <= root.hi**2
    assert root.hi-root.lo <= Fraction(1, 2**80)
    pi = proof.pi_bound()
    decimal_lower = Fraction("3.14159265358979323846264338327950288419716939937510")
    assert pi.lo <= decimal_lower and pi.hi >= decimal_lower+Fraction(1, 10**50)
    sine, cosine = proof.sincos(proof.div(pi, 2))
    assert sine.lo <= 1 <= sine.hi and cosine.lo <= 0 <= cosine.hi
    quarter = proof.atan(1)
    assert quarter.lo <= pi.mid/4 <= quarter.hi


def test_remote_geometry_does_not_change_qualification_work():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    first = query(model, selected)
    assert first.status is S.QUALIFIED, first.diagnostics
    with model.transaction():
        for i in range(32):
            x = 100.+3*i
            vertices = model.add_points(((x,0,0),(x+1,0,0),(x+1,1,0),(x,1,0)))
            model.add_plate(vertices)
    before = snapshot(model)
    second = query(model, selected)
    assert second.status is S.QUALIFIED, second.diagnostics
    assert first.certificate.work_counts == second.certificate.work_counts
    assert snapshot(model) == before


@pytest.mark.parametrize("failure", ["nonfinite", "shape", "wrong_type"])
def test_malformed_owner_frames_fail_closed(monkeypatch, failure):
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    original = charts.arc_frame
    def malformed(*args):
        frame = original(*args)
        if failure == "nonfinite":
            return replace(frame, radius=float("nan"))
        if failure == "shape":
            return replace(frame, e1=np.asarray((1., 0.)))
        return object()
    monkeypatch.setattr(charts, "arc_frame", malformed)
    before = snapshot(model)
    result = query(model, selected)
    assert result.status is S.UNRESOLVED
    assert result.diagnostics == ("malformed_owner_arc_frame",)
    assert not (result.sectors or result.occurrences or result.interfaces or result.boundary_cycles)
    assert snapshot(model) == before


def test_callback_matching_internal_refusal_propagates_unchanged():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    marker = charts._Refusal("callback marker, not numeric evidence")
    def cancel(phase):
        if phase == "cylinder atlas source face":
            raise marker
    before = snapshot(model)
    with pytest.raises(charts._Refusal) as caught:
        query(model, selected, cancellation_check=cancel)
    assert caught.value is marker
    assert snapshot(model) == before


def test_open_transaction_and_oversized_selection_are_preflight_errors():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    before = snapshot(model)
    consumed = []
    def oversized():
        for index in range(1000):
            consumed.append(index)
            yield selected[0]
    with pytest.raises(CylinderAtlasError) as caught:
        query(model, oversized(), reference=selected[0])
    assert caught.value.code is E.INVALID_REQUEST
    assert len(consumed) == 257
    assert snapshot(model) == before
    with model.transaction():
        during = snapshot(model)
        with pytest.raises(CylinderAtlasError) as caught:
            query(model, selected)
        assert caught.value.code is E.BUSY_MODEL
        assert snapshot(model) == during


def test_rational_resource_preflight_rejects_cross_endpoint_products():
    proof = charts._Proof(CylinderAtlasPolicy(), None)
    # Corresponding endpoint pairs fit; the cross endpoint does not. Refuse
    # before multiplying rather than allocating the oversized intermediate.
    huge = Fraction(1 << 5000)
    with pytest.raises(charts._Refusal, match="rational_bit_budget"):
        proof.mul(proof.i(-huge, 1), proof.i(-1, huge))
    with pytest.raises(charts._Refusal, match="rational_bit_budget"):
        proof.remainder_power(Fraction(1, (1 << 80)+1), 128, 1)


def test_trig_against_independent_high_precision_decimal_series():
    # Independently coded scalar series: each omitted alternating tail is less
    # than its first omitted term; the 1e-80 oracle envelope also dominates
    # accumulated rounding at 100 decimal digits for these <100-term sums.
    with localcontext() as context:
        context.prec = 100
        x = Decimal(1)/Decimal(3)
        sine, cosine = x, Decimal(1)
        sin_term, cos_term = x, Decimal(1)
        for n in range(1, 100):
            sin_term *= -x*x/Decimal((2*n)*(2*n+1))
            cos_term *= -x*x/Decimal((2*n-1)*(2*n))
            sine += sin_term
            cosine += cos_term
            if max(abs(sin_term), abs(cos_term)) < Decimal("1e-90"):
                break
        else:
            pytest.fail("independent oracle did not converge")
        proof = charts._Proof(CylinderAtlasPolicy(), None)
        actual = proof.sincos(Fraction(1, 3))
        for enclosure, oracle in zip(actual, (sine, cosine)):
            assert enclosure.lo <= Fraction(oracle)-Fraction(1, 10**80)
            assert enclosure.hi >= Fraction(oracle)+Fraction(1, 10**80)


@pytest.mark.parametrize("options", [
    {"hole_sector": 3}, {"hole": False, "local_hole": True}, {"use_orientation": -1},
    {"equivalent_turns": 1},
])
def test_additional_owned_material_domains(options):
    model, _, sheet, _, _ = sector_hole_model(**options)
    assert model.validate_topology() == ()
    selected = selection(model, sheet)
    before = snapshot(model)
    atlas = query(model, selected)
    assert atlas.status is S.QUALIFIED, atlas.diagnostics
    assert sorted(c.winding for c in atlas.boundary_cycles) == [-1, 0, 1]
    assert all(s.face_use_orientation == options.get("use_orientation", 1) for s in atlas.sectors)
    assert snapshot(model) == before


def test_distinct_parameterization_object_is_explicitly_out_of_scope():
    model, faces, sheet, _, _ = sector_hole_model()
    model.set_face_parameterization(faces[0], replace(model.faces[faces[0]].surface))
    before = snapshot(model)
    result = query(model, selection(model, sheet))
    assert result.status is S.CAPABILITY_MISSING
    assert result.diagnostics == ("support_or_parameterization_out_of_scope",)
    assert not result.occurrences and not result.certificate.complete
    assert snapshot(model) == before


def test_equal_coordinates_with_distinct_source_identity_never_weld():
    model, faces, sheet, _, _ = sector_hole_model(hole=False)
    source = model.faces[faces[-1]]
    selected = selection(model, sheet)
    vertices, edges = {}, {}
    with model.transaction():
        for oriented in source.loop:
            item = model.edges[oriented.edge]
            keys = (item.start, item.end, item.curve.via_vertex) if isinstance(item.curve, charts.Arc) else (item.start, item.end)
            for key in keys:
                if key not in vertices:
                    vertices[key] = model.add_point(*model.vertex_position(key))
            edges[item.id] = (model.add_arc(vertices[item.start], vertices[item.curve.via_vertex], vertices[item.end])
                              if isinstance(item.curve, charts.Arc) else model.add_line(vertices[item.start], vertices[item.end]))
        duplicate = model.add_face_from_loop(tuple(OrientedEdge(edges[o.edge], o.forward) for o in source.loop), surface=source.surface)
        duplicate_sheet = model.add_sheet((duplicate,))
    expected = (*selected[:-1], *selection(model, duplicate_sheet))
    assert model.validate_topology() == ()
    before = snapshot(model)
    result = query(model, expected)
    assert result.status is S.CAPABILITY_MISSING
    assert result.diagnostics == ("disconnected_or_unpartitioned_sectors",)
    assert not result.interfaces and not result.certificate.complete
    assert snapshot(model) == before


def test_public_transverse_split_replaces_cylinder_and_stales_atlas():
    from anygeometry import ResolutionStatus, split_face_at
    model, faces, sheet, _, _ = sector_hole_model(hole=False)
    for face in faces:
        model.set_face_corners(face, (0, 1, 2, 3))
    selected = selection(model, sheet)
    old = query(model, selected)
    assert old.status is S.QUALIFIED, old.diagnostics
    parent = model.handle("face", faces[3])
    original_surface = model.faces[parent.id].surface
    divider, children = split_face_at(model, parent.id, axis=1, fraction=.5)
    assert model.resolve_handle(parent).status is ResolutionStatus.REPLACED
    assert {h.id for h in model.resolve_handle(parent).resolved} == set(children)
    assert all(model.faces[key].surface is original_surface for key in children)
    assert isinstance(model.edges[divider].curve, charts.Arc)
    active = selection(model, sheet)
    assert len(active) == 9
    with pytest.raises(CylinderAtlasError) as caught:
        validate(model, old, active)
    assert caught.value.code is E.STALE_RESULT
    assert model.validate_topology() == ()
    before = snapshot(model)
    fresh = query(model, active)
    assert fresh.status is S.QUALIFIED, fresh.diagnostics
    assert divider in {item.edge.id for item in fresh.interfaces}
    assert set(children) <= {item.face.id for item in fresh.sectors}
    assert snapshot(model) == before


def test_public_recovered_plate_connect_retains_external_radial_incidence():
    from anygeometry import ConnectionIntent, apply_imprint, plan_imprint, query_intersection
    from anygeometry.generators import cylinder
    model = cylinder(radius=.5, height=2., circumferential_segments=12,
                     longitudinal_spacing=.5, ring_spacing=1.)
    sheet = next(iter(model.sheets))
    selected = selection(model, sheet)
    assert len(selected) == 24
    old = query(model, selected)
    assert old.status is S.QUALIFIED, old.diagnostics
    model.features.capture_baseline(model)
    feature = model.features.append("generator.plate", parameters={
        "length": 2., "width": 2., "origin": (-1.,-1.,1.),
        "u_direction": (1.,0.,0.), "v_direction": (0.,1.,0.), "semantic_group": "shell",
    })
    assert model.regenerate_features().success
    plate = model.features.get(feature.feature_id).outputs["face/1"]
    cylinder_face = model.face_uses[selected[0].id].face_id
    intersection = query_intersection(model, model.handle("face", plate.id), model.handle("face", cylinder_face))
    assert intersection.classified and intersection.certificate.complete
    plan = plan_imprint(model, intersection, policy=ConnectionIntent.CONNECT)
    hooks = []
    model.add_change_hook(hooks.append)
    applied = apply_imprint(model, plan, policy=ConnectionIntent.CONNECT)
    assert len(hooks) == 1 and applied.face_intersection is not None
    active = selection(model, sheet)
    with pytest.raises(CylinderAtlasError) as caught:
        validate(model, old, active)
    assert caught.value.code is E.STALE_RESULT
    before = snapshot(model)
    fresh = query(model, active)
    assert fresh.status is S.QUALIFIED, fresh.diagnostics
    ring = {r.id for r in model.group("ring_stiffeners")}
    interfaces = {item.edge.id: item for item in fresh.interfaces}
    assert len(ring) == 12 and ring <= interfaces.keys()
    selected_ids = {h.id for h in active}
    for edge in ring:
        expected = tuple(model.handle("coedge", key) for key in model.coedges_using_edge(edge)
                         if model.coedges[key].face_use_id not in selected_ids)
        assert len(expected) == 2
        assert interfaces[edge].external_coedges == expected
    assert snapshot(model) == before
    assert model.validate_topology() == ()


def test_cold_and_publicly_warmed_caches_are_not_changed_by_query():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    cold = snapshot(model)
    first = query(model, selected)
    assert first.status is S.QUALIFIED, first.diagnostics
    assert snapshot(model) == cold
    for edge in model.edges:
        model.sample_edge(edge, np.asarray((0., .5, 1.)))
        model.edge_length(edge)
    warm = snapshot(model)
    second = query(model, selected)
    assert second == first
    assert snapshot(model) == warm


def test_cancellation_at_every_observed_qualification_phase():
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    phases = []
    result = query(model, selected, cancellation_check=phases.append)
    assert result.status is S.QUALIFIED, result.diagnostics
    assert {"cylinder atlas source copy", "cylinder atlas face qualification", "cylinder atlas result"} <= set(phases)
    before = snapshot(model)
    for target in dict.fromkeys(phases):
        marker = RuntimeError(target)
        def cancel(phase):
            if phase == target:
                raise marker
        with pytest.raises(RuntimeError) as caught:
            query(model, selected, cancellation_check=cancel)
        assert caught.value is marker
        assert snapshot(model) == before


def test_inactive_handle_is_rejected_before_source_copy():
    from anygeometry import EntityHandle
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    absent = EntityHandle(model.model_id, "face_use", max(model.face_uses)+1)
    before = snapshot(model)
    with pytest.raises(CylinderAtlasError) as caught:
        query(model, (*selected[:-1], absent))
    assert caught.value.code is E.INACTIVE_ENTITY
    assert snapshot(model) == before


def test_ill_conditioned_detached_arc_fails_closed_without_live_mutation(monkeypatch):
    from anygeometry import Vertex
    model, _, sheet, _, _ = sector_hole_model()
    selected = selection(model, sheet)
    original = charts._Context.__init__
    def corrupt_detached_context(context, *args):
        original(context, *args)
        edge = next(item for item in context.edges.values() if isinstance(item.curve, charts.Arc))
        start, end = (context.vertices[key].position for key in (edge.start, edge.end))
        # Negative evidence injection ONLY into detached records. The live
        # publicly authored model is never changed or presented as malformed.
        key = edge.curve.via_vertex
        context.vertices[key] = Vertex(key, (start+end)/2)
    monkeypatch.setattr(charts._Context, "__init__", corrupt_detached_context)
    before = snapshot(model)
    result = query(model, selected)
    assert result.status is S.UNRESOLVED
    assert result.diagnostics == ("ill_conditioned_arc",)
    assert not result.certificate.complete and not result.occurrences
    assert snapshot(model) == before


def test_ambiguous_numeric_branch_and_degenerate_division_refuse():
    proof = charts._Proof(CylinderAtlasPolicy(), None)
    with pytest.raises(charts._Refusal, match="angular_branch_unqualified"):
        proof.atan2(proof.i(-1, 1), proof.i(-1, 1))
    with pytest.raises(charts._Refusal, match="uncertain_zero_divisor"):
        proof.div(1, proof.i(-1, 1))
    with pytest.raises(charts._Refusal, match="negative_sqrt_enclosure"):
        proof.sqrt(proof.i(-1, 1))


def test_evaluation_cancellation_after_output_rows_returns_no_partial_batch(qualified):
    model, selected, atlas, _, _ = qualified
    requests = tuple(CylinderOccurrenceRequest(atlas.occurrences[0].id, i/31) for i in range(32))
    marker = RuntimeError("after rows marker")
    def cancel(phase):
        if phase == "cylinder atlas evaluated result":
            raise marker
    before = snapshot(model)
    with pytest.raises(RuntimeError) as caught:
        evaluate(model, atlas, selected, requests, cancellation_check=cancel)
    assert caught.value is marker
    assert snapshot(model) == before


def test_same_endpoints_do_not_substitute_for_a_shared_source_edge():
    model, faces, sheet, _, _ = sector_hole_model(hole=False)
    selected = selection(model, sheet)
    source = model.faces[faces[-1]]
    # Replace just the seam identity in an otherwise connected selection.
    # Geometry and endpoint IDs agree; the source Edge identity does not.
    seam = next(o for o in source.loop if isinstance(model.edges[o.edge].curve, charts.Straight)
                and faces[0] in model.faces_using_edge(o.edge))
    item = model.edges[seam.edge]
    with model.transaction():
        duplicate = model.add_line(item.start, item.end)
        loop = tuple(OrientedEdge(duplicate if o.edge == seam.edge else o.edge, o.forward) for o in source.loop)
        face = model.add_face_from_loop(loop, surface=source.surface)
        extra_sheet = model.add_sheet((face,))
    active = (*selected[:-1], *selection(model, extra_sheet))
    assert model.validate_topology() == ()
    before = snapshot(model)
    atlas = query(model, active)
    assert atlas.status is S.CAPABILITY_MISSING
    assert not atlas.certificate.complete and not atlas.interfaces
    assert "source" in atlas.diagnostics[0]
    assert snapshot(model) == before


def test_aggregate_incidence_limit_and_copy_cancellation():
    model, _, sheet, _, _ = sector_hole_model(hole=False)
    selected = selection(model, sheet)
    edge = next(e for e in model.edges.values() if isinstance(e.curve, charts.Straight))
    start = model.vertex_position(edge.start)
    with model.transaction():
        third = model.add_point(2*start[0], 2*start[1], 1.)
        a, b = model.add_line(edge.end, third), model.add_line(third, edge.start)
        triangle = model.add_face_from_loop((OrientedEdge(edge.id, True), OrientedEdge(a, True), OrientedEdge(b, True)))
        model.add_sheet((triangle,))
    cap = sum(len(loop) for handle in selected for loop in model.face_uses[handle.id].loops)
    assert max(len(model.coedges_using_edge(e)) for e in model.edges) < cap
    before = snapshot(model)
    result = query(model, selected, policy=CylinderAtlasPolicy(max_occurrences=cap))
    assert result.status is S.UNRESOLVED
    assert result.diagnostics == ("qualification_budget_exhausted:external_incidences",)
    marker = RuntimeError("incidence cancellation")
    def cancel(phase):
        if phase == "cylinder atlas incidence copy":
            raise marker
    with pytest.raises(RuntimeError) as caught:
        query(model, selected, cancellation_check=cancel)
    assert caught.value is marker
    assert snapshot(model) == before


def test_material_seed_charges_candidate_and_clearance_work():
    proof = charts._Proof(CylinderAtlasPolicy(max_pair_tests=8), None)
    polygon = [(proof.i(x), proof.i(y)) for x, y in ((0,0),(1,0),(1,1),(0,1))]
    with pytest.raises(charts._Refusal, match="qualification_budget_exhausted:pair_tests"):
        charts._material_seed(proof, polygon, Fraction(1, 100))
    assert proof.counts["pair_tests"] == 8
    marker = RuntimeError("bounded seed construction")
    def cancel(_phase):
        raise marker
    proof = charts._Proof(CylinderAtlasPolicy(), cancel)
    polygon = [(proof.i(i), proof.i(i % 2)) for i in range(64)]
    with pytest.raises(RuntimeError) as caught:
        charts._material_seed(proof, polygon, Fraction(1,100))
    assert caught.value is marker and proof.counts["pair_tests"] == 32


def test_nested_binding_evidence_is_aggregate_bounded_and_cancellable(qualified):
    model, selected, atlas, _, _ = qualified
    forged = copy(atlas)
    sector = copy(atlas.sectors[0])
    # Each tuple is individually modest; their logical aggregate is not.
    object.__setattr__(sector, "loops", (("x",)*4096,)*64)
    object.__setattr__(forged, "sectors", (sector, *atlas.sectors[1:]))
    before = snapshot(model)
    with pytest.raises(CylinderAtlasError, match="aggregate budget"):
        validate(model, forged, selected)
    marker = CylinderAtlasError(E.INVALID_RESULT, ("caller cancellation",))
    def cancel(phase):
        if phase == "cylinder atlas bounded evidence":
            raise marker
    with pytest.raises(CylinderAtlasError) as caught:
        validate(model, forged, selected, cancellation_check=cancel)
    assert caught.value is marker
    assert charts._binding_budget.get() is None
    assert snapshot(model) == before


@pytest.mark.parametrize("perturbation", ["offset", "tilt"])
def test_reference_enclosure_composes_accepted_frame_error(perturbation):
    model, faces, sheet, _, _ = sector_hole_model(hole=False, frame_perturbation=perturbation)
    assert model.validate_topology() == ()
    selected = selection(model, sheet)
    before = snapshot(model)
    atlas = query(model, selected)
    assert atlas.status is S.QUALIFIED, atlas.diagnostics
    face_use = next(h for h in selected if model.face_uses[h.id].face_id == faces[3])
    occurrence = next(o for o in atlas.occurrences if o.face_use == face_use and o.carrier == "CIRCULAR")
    sample = evaluate(model, atlas, selected, (CylinderOccurrenceRequest(occurrence.id, .37),)).samples[0]
    # Independent high-precision inverse of the unchanged reference Cylinder:
    # theta=atan2(y,x), axial=z. No production chart or support inverse is used.
    with localcontext() as context:
        context.prec = 90
        x, y, z = (Decimal.from_float(v) for v in sample.point)
        assert x < 0 < y
        ratio = -y/x
        factor = 1
        while ratio > Decimal("0.25"):
            ratio /= 1+(1+ratio*ratio).sqrt()
            factor *= 2
        term, angle = ratio, ratio
        for n in range(1, 128):
            term *= -ratio*ratio
            contribution = term/Decimal(2*n+1)
            angle += contribution
            if abs(contribution) < Decimal("1e-80"):
                break
        else:
            pytest.fail("independent atan oracle did not converge")
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628")
        turns = (pi-factor*angle)/(2*pi)
        for expected, bounds in zip((turns, z), sample.reference_enclosure):
            low, high = (Decimal.from_float(v) for v in bounds)
            assert low <= expected-Decimal("1e-70") < expected+Decimal("1e-70") <= high
    assert snapshot(model) == before
