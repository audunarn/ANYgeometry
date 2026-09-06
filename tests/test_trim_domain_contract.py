"""Bounded owner trim evidence on real GeometryModel records, never mesh mocks."""

from dataclasses import replace
import math

import numpy as np
import pytest

from anygeometry import (
    EntityHandle, GeometryModel, OrientedEdge, Plane, TolerancePolicy,
    TrimBoundaryContact as B, TrimDomainError, TrimDomainErrorCode as E,
    TrimInteriorRelation as R, query_trim_domain_relation as query,
    validate_trim_domain_binding as validate,
)
from anygeometry import trim_domains as td
from anygeometry.entities import Vertex


def model_pair(*, radius=.5, scale=1., offset=(0., 0., 0.), rotate=False,
               reverse=False, straight=False, via_radius=None):
    g = GeometryModel(tolerance=TolerancePolicy(
        length=1.e-9 * scale, merge_length=1.e-7 * scale,
        area=1.e-18 * scale**2, surface_residual=1.e-8 * scale,
    ))
    rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))) if rotate else np.eye(3)
    origin = np.asarray(offset)

    def point(p): return tuple(origin + scale * (rotation @ np.asarray(p)))

    plane = Plane(origin, rotation[:, 0] * scale, rotation[:, 1] * scale)
    with g.transaction():
        vertices = g.add_points([point(p) for p in ((-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0))])
        outer_edges = [g.add_line(vertices[i], vertices[(i+1) % 4]) for i in range(4)]
        a = g.add_face_from_loop(tuple(OrientedEdge(e, True) for e in outer_edges), surface=plane)
        ring_points = g.add_points([point((radius*math.cos(i*math.pi/6), radius*math.sin(i*math.pi/6), 0)) for i in range(12)])
        rvia = radius if via_radius is None else via_radius
        vias = g.add_points([point((rvia*math.cos((i+.5)*math.pi/6), rvia*math.sin((i+.5)*math.pi/6), 0)) for i in range(12)])
        edges = [g.add_line(ring_points[i], ring_points[(i+1) % 12]) if straight else
                 g.add_arc(ring_points[i], vias[i], ring_points[(i+1) % 12]) for i in range(12)]
        loop = tuple(OrientedEdge(e, True) for e in edges)
        if reverse:
            loop = tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(loop))
        b = g.add_face_from_loop(loop, surface=plane)
        hole = tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(loop))
        g._put_entity("face", replace(g.faces[a], holes=(hole,)))
    return g, g.handle("face", a), g.handle("face", b)


def state(g):
    # Serialization validates (and warms caches), and correctly rejects hostile
    # records. A nonmutating raw snapshot is required for this query test.
    return repr(g.__dict__), tuple((k, v.position.tobytes()) for k, v in g.vertices.items())


def run(g, a, b):
    return query(g, a, b, expected_revision=g.revision)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("order", [False, True])
def test_twelve_arc_owner_fixture(reverse, rotate, order):
    g, a, b = model_pair(reverse=reverse, rotate=rotate, offset=(2.,-3.,4.) if rotate else (0.,0.,0.))
    if order:
        a, b = b, a
    assert g.validate_topology() == ()
    before = state(g)
    result = run(g, a, b)
    assert result.relation is R.DISJOINT_INTERIORS, result.diagnostics
    assert result.complementary and result.boundary is B.CURVE
    assert len(result.shared_boundary) == 12
    assert result.max_residual == 0
    assert result.algorithm == "exact_topology_star_ring_v1"
    validate(g, result, a, b, expected_revision=g.revision)
    assert state(g) == before
    assert result.digest == run(g, a, b).digest


@pytest.mark.parametrize("scale", [1.e-6, 1., 1.e6])
@pytest.mark.parametrize("offset", [(0.,0.,0.), (1.e6,-1.e6,1.e6)])
def test_supported_scale_translation(scale, offset):
    g, a, b = model_pair(scale=scale, offset=offset)
    result = run(g, a, b)
    assert result.complementary, result.diagnostics
    assert result.tolerance == pytest.approx(1.e-9*scale)


def test_noncircular_minor_arc_family():
    g, a, b = model_pair(via_radius=.501)
    assert run(g, a, b).complementary


def test_straight_complement_and_backend_outcomes():
    g, a, b = model_pair(straight=True)
    result = run(g, a, b)
    assert result.complementary
    assert result.work_counts["owner_queries"] == 1
    other = g.add_plate(g.add_points(((4,4,0), (5,4,0), (5,5,0), (4,5,0))))
    g.set_face_surface(other, g.faces[a.id].surface)
    assert run(g, a, g.handle("face", other)).relation is R.DISJOINT_INTERIORS
    assert run(g, b, g.handle("face", other)).boundary is B.NONE
    overlap = g.add_plate(g.add_points(((.1,.1,0), (.3,.1,0), (.3,.3,0), (.1,.3,0))))
    g.set_face_surface(overlap, g.faces[a.id].surface)
    assert run(g, b, g.handle("face", overlap)).relation is R.OVERLAPPING_INTERIORS


@pytest.mark.parametrize("defect", ["open", "duplicate", "bowtie", "nonplanar", "long_arc", "missing_hole", "extra_parent_hole", "extra_fill_hole", "parameterization", "two_turns", "degenerate"])
def test_hostile_trim_no_exemption(defect):
    g, a, b = model_pair()
    face = g.faces[b.id]
    # Deliberately inject invalid committed-like records to test fail-closed query
    # handling. Normal authoring rejects many of these earlier.
    edge = g.edges[face.loop[0].edge]
    if defect == "open":
        g._set_entity_unjournalled("edge", edge.id, replace(edge, end=g.edges[face.loop[2].edge].end))
    elif defect == "duplicate":
        g._set_entity_unjournalled("face", b.id, replace(face, loop=face.loop + face.loop[:1]))
    elif defect == "bowtie":
        ids = [g.oriented_start_vertex(e) for e in g.faces[a.id].loop]
        p, q = g.vertices[ids[1]].position, g.vertices[ids[2]].position
        g._set_entity_unjournalled("vertex", ids[1], Vertex(ids[1], q))
        g._set_entity_unjournalled("vertex", ids[2], Vertex(ids[2], p))
    elif defect in ("nonplanar", "long_arc", "degenerate"):
        key = edge.curve.via_vertex
        p = g.vertices[key].position.copy()
        if defect == "nonplanar": p[2] = .01
        elif defect == "long_arc": p = -p
        else: p = g.vertices[edge.start].position
        g._set_entity_unjournalled("vertex", key, Vertex(key, p))
    elif defect == "missing_hole":
        g._set_entity_unjournalled("face", a.id, replace(g.faces[a.id], holes=()))
    elif defect == "extra_parent_hole":
        g._set_entity_unjournalled("face", a.id, replace(g.faces[a.id], holes=g.faces[a.id].holes*2))
    elif defect == "extra_fill_hole":
        g._set_entity_unjournalled("face", b.id, replace(face, holes=(face.loop,)))
    elif defect == "parameterization":
        g._set_entity_unjournalled("face", b.id, replace(face, parameterization=face.surface))
    elif defect == "two_turns":
        # Distinct IDs and exact connectivity, but traverse the same geometry twice.
        new_loop = []
        for use in face.loop:
            old = g.edges[use.edge]
            key = max(g.edges)+1
            g._set_entity_unjournalled("edge", key, replace(old, id=key))
            new_loop.append(OrientedEdge(key, use.forward))
        loop = face.loop + tuple(new_loop)
        g._set_entity_unjournalled("face", b.id, replace(face, loop=loop))
        hole = tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(loop))
        g._set_entity_unjournalled("face", a.id, replace(g.faces[a.id], holes=(hole,)))
    before = state(g)
    result = run(g, a, b)
    assert result.relation is R.UNRESOLVED, (defect, result)
    assert not result.complementary
    assert state(g) == before


@pytest.mark.parametrize("radius", [1., 1.1])
def test_contact_outside_ring_never_exempt(radius):
    g, a, b = model_pair()
    # Keep valid baseline authoring; adversarially expand the shared ring.
    keys = {k for use in g.faces[b.id].loop for k in
            (g.edges[use.edge].start, g.edges[use.edge].end, g.edges[use.edge].curve.via_vertex)}
    for key in keys:
        g._set_entity_unjournalled("vertex", key, Vertex(key, g.vertices[key].position*radius/.5))
    assert run(g, a, b).relation is R.UNRESOLVED


@pytest.mark.parametrize("warm", [False, True])
def test_complete_source_state_unchanged(warm):
    g, a, b = model_pair()
    if warm:
        for use in g.faces[b.id].loop:
            g.edge_length(use.edge)
        g._spatial()
    else:
        g._arc_cache.clear()
        g._edge_length_cache.clear()
    before = state(g)
    result = run(g, a, b)
    validate(g, result, a, b, expected_revision=g.revision)
    assert state(g) == before
    assert result.work_counts["copied_faces"] == 2
    assert result.work_counts["copied_edges"] == 16


def test_request_and_retained_bindings():
    g, a, b = model_pair()
    result = run(g, a, b)
    before = state(g)
    for revision in (True, "1", 1., -1):
        with pytest.raises(TrimDomainError) as caught:
            query(g, a, b, expected_revision=revision)
        assert caught.value.code is E.INVALID_REQUEST
    foreign = EntityHandle(GeometryModel().model_id, "face", a.id)
    with pytest.raises(TrimDomainError) as caught:
        query(g, foreign, b, expected_revision=g.revision)
    assert caught.value.code is E.WRONG_MODEL
    with pytest.raises(TrimDomainError) as caught:
        validate(g, result, b, a, expected_revision=g.revision)
    assert caught.value.code is E.INVALID_RESULT
    with pytest.raises(TrimDomainError) as caught:
        query(g, a, EntityHandle(g.model_id, "face", 999), expected_revision=g.revision)
    assert caught.value.code is E.INACTIVE_ENTITY
    assert state(g) == before
    with g.transaction():
        busy_before = state(g)
        with pytest.raises(TrimDomainError) as caught:
            run(g, a, b)
        assert caught.value.code is E.BUSY_MODEL
        assert state(g) == busy_before
    g.add_point(8,8,8)
    with pytest.raises(TrimDomainError) as caught:
        validate(g, result, a, b, expected_revision=g.revision)
    assert caught.value.code is E.STALE_RESULT
    with pytest.raises(TrimDomainError) as caught:
        query(g, a, b, expected_revision=result.revision)
    assert caught.value.code is E.STALE_REVISION


def test_immutable_result_and_self_rehashed_forgery():
    g, a, b = model_pair()
    result = run(g, a, b)
    with pytest.raises(TypeError):
        result.work_counts["copied_faces"] = 42
    with pytest.raises(AttributeError):
        result.complementary = False
    data = td._result_data(result)
    data.update(algorithm="invented_proof")
    forged = td.TrimDomainResult(**data, digest=td._digest(data))
    with pytest.raises(TrimDomainError) as caught:
        validate(g, forged, a, b, expected_revision=g.revision)
    assert caught.value.code is E.INVALID_RESULT
    with pytest.raises(AttributeError):
        caught.value.code = E.WRONG_MODEL
    with pytest.raises(TrimDomainError):
        replace(result, tolerance=float("nan"))


@pytest.mark.parametrize("where", ["callback", "backend", "access"])
def test_operational_error_identity_and_nonmutation(monkeypatch, where):
    g, a, b = model_pair(straight=True)
    failure = RuntimeError("injected operational failure")

    def fail(*args, **kwargs): raise failure

    if where == "backend":
        monkeypatch.setattr(td, "query_intersection", fail)
    elif where == "access":
        monkeypatch.setattr(td, "deepcopy", fail)
    before = state(g)
    with pytest.raises(RuntimeError) as caught:
        query(g, a, b, expected_revision=g.revision, cancellation_check=fail if where == "callback" else None)
    assert caught.value is failure
    assert state(g) == before


def test_unrelated_geometry_not_copied():
    g, a, b = model_pair()
    before = run(g, a, b).work_counts
    for i in range(5):
        g.add_plate(g.add_points(((10+i,0,0),(11+i,0,0),(11+i,1,0),(10+i,1,0))))
    assert run(g, a, b).work_counts == before


def test_area_threshold_curve_never_certifies_no_overlap(monkeypatch):
    g, a, b = model_pair(straight=True)
    # Unrelated straight faces with a owner curve result cannot gain exemption.
    g._set_entity_unjournalled("face", a.id, replace(g.faces[a.id], holes=()))
    real = td.query_intersection

    def curve(*args, **kwargs):
        value = real(*args, **kwargs)
        return replace(value, dimension=td.IntersectionDimension.CURVE)

    monkeypatch.setattr(td, "query_intersection", curve)
    assert run(g, a, b).relation is R.UNRESOLVED


@pytest.mark.parametrize("factor", [1.e-12, 1., 1.e12, -1.])
def test_chart_rescaling_does_not_change_physical_tolerance(factor):
    g, a, b = model_pair()
    original = run(g, a, b)
    for key in (a.id, b.id):
        face = g.faces[key]
        surface = Plane(face.surface.origin, face.surface.u_vector*factor, face.surface.v_vector/factor)
        g._set_entity_unjournalled("face", key, replace(face, surface=surface))
    result = run(g, a, b)
    assert result.complementary and result.tolerance == original.tolerance


def test_full_arc_not_just_endpoints_must_be_contained():
    g, a, b = model_pair()
    edge = g.edges[g.faces[b.id].loop[0].edge]
    assert all(np.max(np.abs(g.vertices[key].position[:2])) < 1 for key in (edge.start, edge.end))
    key = edge.curve.via_vertex
    g._set_entity_unjournalled("vertex", key, Vertex(key, (1.2, .15, 0)))
    assert run(g, a, b).relation is R.UNRESOLVED


@pytest.mark.parametrize("limit", ["operations", "bits", "edges"])
def test_budget_exhaustion_is_model_level_unresolved(monkeypatch, limit):
    g, a, b = model_pair()
    if limit == "edges":
        face = g.faces[b.id]
        g._set_entity_unjournalled("face", b.id, replace(face, loop=face.loop*6))
    else:
        initialize = td._Proof.__init__

        def nearly_exhausted(self, callback):
            initialize(self, callback)
            if limit == "operations":
                self.counts["rational_operations"] = 200_000
            else:
                # The internal operation checks this operand before multiplication.
                self.q(td.Fraction(1 << 8192))

        if limit == "operations":
            monkeypatch.setattr(td._Proof, "__init__", nearly_exhausted)
        else:
            original = td._Proof.dot

            def huge_operand(self, lhs, rhs):
                self.q(td.Fraction(1 << 8192))
                return original(self, lhs, rhs)

            monkeypatch.setattr(td._Proof, "dot", huge_operand)
    before = state(g)
    result = run(g, a, b)
    assert result.relation is R.UNRESOLVED
    assert result.diagnostics == ("qualification_budget_exhausted",)
    assert state(g) == before


def test_typed_missing_backend_is_not_operational_exception(monkeypatch):
    from anygeometry import IntersectionResult, IntersectionKind
    g, a, b = model_pair(straight=True)
    monkeypatch.setattr(td, "query_intersection", lambda *args, **kwargs: IntersectionResult(
        IntersectionKind.CAPABILITY_MISSING, diagnostics=("planar_backend_unavailable",),
    ))
    result = run(g, a, b)
    assert result.relation is R.UNRESOLVED and not result.complementary
    assert "capability_missing" in result.diagnostics[0].lower()


def test_late_cancellation_leaves_cold_live_state_unchanged():
    g, a, b = model_pair()
    g._arc_cache.clear()
    before = state(g)
    failure = RuntimeError("cancel during rational proof")
    calls = []

    def cancel(phase):
        calls.append(phase)
        if len(calls) == 5:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        query(g, a, b, expected_revision=g.revision, cancellation_check=cancel)
    assert caught.value is failure
    assert calls == ["trim domain qualification"]*5
    assert state(g) == before


def test_wrong_model_retained_result_and_bad_digests():
    g, a, b = model_pair()
    result = run(g, a, b)
    other, x, y = model_pair()
    with pytest.raises(TrimDomainError) as caught:
        validate(other, result, x, y, expected_revision=other.revision)
    assert caught.value.code is E.WRONG_MODEL
    with pytest.raises(TrimDomainError):
        replace(result, digest="0"*64)
    data = td._result_data(result)
    counts = dict(result.work_counts)
    data["work_counts"] = counts
    copied = td.TrimDomainResult(**data, digest=td._digest(data))
    counts["owner_queries"] = 9
    assert copied.work_counts["owner_queries"] == 0


def test_unknown_protocol_types_rejected():
    g, a, b = model_pair()
    with pytest.raises(TypeError):
        query(object(), a, b, expected_revision=g.revision)
    with pytest.raises(TypeError):
        query(g, a, b, expected_revision=g.revision, cancellation_check="not callable")
    with pytest.raises(TrimDomainError) as caught:
        query(g, a.id, b.id, expected_revision=g.revision)
    assert caught.value.code is E.INVALID_REQUEST
    with pytest.raises(TrimDomainError):
        replace(run(g, a, b), relation="INVENTED_RELATION")


@pytest.mark.parametrize("family", ["disjoint", "region", "complement"])
@pytest.mark.parametrize("incomplete", ["result", "component"])
def test_incomplete_owner_evidence_never_qualifies(monkeypatch, family, incomplete):
    from anygeometry import IntersectionCertificate, IntersectionComponent, IntersectionResult, IntersectionKind, IntersectionDimension, IntersectionQuality
    g, a, b = model_pair(straight=True)
    if family != "complement":
        g._set_entity_unjournalled("face", a.id, replace(g.faces[a.id], holes=()))
    kind = {"disjoint": IntersectionKind.DISJOINT, "region": IntersectionKind.OVERLAP_REGION,
            "complement": IntersectionKind.OVERLAP_CURVE}[family]
    if family == "disjoint" and incomplete == "component":
        # DISJOINT has no legal components; test absent aggregate evidence too.
        owner = IntersectionResult(kind, certificate=IntersectionCertificate("test", 1.e-9, complete=False))
        object.__setattr__(owner, "certificate", None)
    else:
        components = () if family == "disjoint" else (IntersectionComponent(
            ((0.,0.,0.), (1.,0.,0.), (1.,1.,0.)), IntersectionQuality.EXACT,
            certificate=IntersectionCertificate("component", 1.e-9, complete=incomplete != "component"),
        ),)
        owner = IntersectionResult(
            kind, components=components,
            dimension=IntersectionDimension.NONE if family == "disjoint" else
                (IntersectionDimension.REGION if family == "region" else IntersectionDimension.CURVE),
            certificate=IntersectionCertificate("aggregate", 1.e-9, complete=incomplete != "result"),
        )
    monkeypatch.setattr(td, "query_intersection", lambda *args, **kwargs: owner)
    before = state(g)
    result = run(g, a, b)
    assert result.relation is R.UNRESOLVED and not result.complementary
    assert result.diagnostics == ("owner_incomplete_evidence",)
    assert state(g) == before


@pytest.mark.parametrize("ratio", [.975e-12, 1.025e-12])
def test_arc_owner_conditioning_boundary(ratio):
    g, a, b = model_pair()
    edge = g.edges[g.faces[b.id].loop[0].edge]
    start, end = g.vertices[edge.start].position, g.vertices[edge.end].position
    chord = end-start
    offset = np.cross((0.,0.,1.), chord) * (math.sqrt(ratio/(1-ratio))/2)
    via = (start+end)/2 + offset
    g._set_entity_unjournalled("vertex", edge.curve.via_vertex, Vertex(edge.curve.via_vertex, via))
    before = state(g)
    result = run(g, a, b)
    assert not result.complementary
    if ratio < 1.e-12:
        assert result.diagnostics == ("arc_conditioning_unqualified",)
        assert td.Fraction(1, 2**40) < td.Fraction.from_float(ratio) < td._ARC_CONDITIONING_MINIMUM
    else:
        assert result.diagnostics != ("arc_conditioning_unqualified",)
    assert state(g) == before
    assert td._ARC_CONDITIONING_MINIMUM == td.Fraction.from_float(1.e-12)


def test_arc_evaluator_exception_identity_and_live_immutability(monkeypatch):
    from anygeometry import DegenerateArcError
    g, a, b = model_pair()
    g._arc_cache.clear()
    before = state(g)
    failure = DegenerateArcError("injected evaluator refusal")

    def refuse(*args, **kwargs): raise failure

    monkeypatch.setattr(td, "arc_frame", refuse)
    with pytest.raises(DegenerateArcError) as caught:
        run(g, a, b)
    assert caught.value is failure
    assert state(g) == before


def test_recovered_cylinder_plate_public_reconstruction():
    from anygeometry import (
        Arc, ConnectionIntent, Cylinder, EntityRef, ResolutionStatus,
        apply_imprint, plan_imprint, query_intersection,
    )
    from anygeometry.generators import cylinder

    # Exact pre-imprint parameters handed off from the user's first two commands.
    # This is a fresh owner reconstruction, not a claim of GUI document byte identity.
    g = cylinder(
        radius=.5, height=2., circumferential_segments=12,
        origin=(0.,0.,0.), axis=(0.,0.,1.), radial_direction=(1.,0.,0.),
        longitudinal_spacing=.5, ring_spacing=1.,
    )
    cylinder_refs = tuple(g.group("shell"))
    ring_refs = tuple(g.group("ring_stiffeners"))
    assert len(cylinder_refs) == 24 and len(ring_refs) == 12
    assert all(isinstance(g.faces[r.id].surface, Cylinder) for r in cylinder_refs)
    g.features.capture_baseline(g)
    feature = g.features.append("generator.plate", parameters={
        "length":2., "width":2., "origin":(-1.,-1.,1.),
        "u_direction":(1.,0.,0.), "v_direction":(0.,1.,0.), "semantic_group":"shell",
    })
    assert g.regenerate_features().success
    authored = g.features.get(feature.feature_id).outputs["face/1"]
    plate_handle = g.handle("face", authored.id)
    cylinder_handle = g.handle("face", cylinder_refs[0].id)
    plane = g.faces[authored.id].surface
    assert isinstance(plane, Plane)
    assert np.array_equal(plane.origin, (-1.,-1.,1.))
    assert np.array_equal(plane.u_vector, (2.,0.,0.))
    assert np.array_equal(plane.v_vector, (0.,2.,0.))
    ring_ids = {r.id for r in ring_refs}
    assert all(isinstance(g.edges[key].curve, Arc) for key in ring_ids)
    control_ids = {key for identifier in ring_ids for key in (
        g.edges[identifier].start, g.edges[identifier].end, g.edges[identifier].curve.via_vertex,
    )}
    ring_definitions = {key: g.edges[key] for key in ring_ids}
    ring_bytes = {key: g.vertices[key].position.tobytes() for key in control_ids}
    intersection = query_intersection(g, plate_handle, cylinder_handle)
    assert intersection.classified and intersection.certificate.complete
    assert all(c.certificate is not None and c.certificate.complete for c in intersection.components)
    plan = plan_imprint(g, intersection, policy=ConnectionIntent.CONNECT)
    changes = []
    g.add_change_hook(changes.append)
    application = apply_imprint(g, plan, policy=ConnectionIntent.CONNECT)
    assert len(changes) == 1
    imprint = application.face_intersection
    assert imprint is not None and len(imprint.first_faces) == 2
    assert {r.id for r in imprint.edges} == ring_ids
    assert all(g.edges[key] == ring_definitions[key] for key in ring_ids)
    assert ring_bytes == {key: g.vertices[key].position.tobytes() for key in control_ids}
    resolution = g.resolve_handle(plate_handle)
    assert resolution.status is ResolutionStatus.REPLACED
    assert set(resolution.resolved) == {g.handle("face", r.id) for r in imprint.first_faces}
    assert set(g.resolve_ref(authored)) == set(imprint.first_faces)
    assert all(g.resolve_ref(ref) == (ref,) for ref in cylinder_refs)
    assert sorted(len(g.faces[r.id].holes) for r in imprint.first_faces) == [0,1]
    assert all(isinstance(g.faces[r.id].surface, Plane) for r in imprint.first_faces)
    assert all(len(g.faces_using_edge(key)) == 4 for key in ring_ids)
    assert all(len(g.coedges_using_edge(key)) == 4 for key in ring_ids)
    assert all(len(g.face_uses_using_edge(key)) == 4 for key in ring_ids)
    assert all(len(g.sheets_using_edge(key)) == 2 for key in ring_ids)
    assert g.validate_topology() == ()
    before = state(g)
    parents = tuple(g.handle("face", r.id) for r in imprint.first_faces)
    for first, second in (parents, parents[::-1]):
        result = run(g, first, second)
        assert result.complementary, result.diagnostics
        assert result.relation is R.DISJOINT_INTERIORS and result.boundary is B.CURVE
        assert {h.id for h in result.shared_boundary} == ring_ids
        validate(g, result, first, second, expected_revision=g.revision)
    assert state(g) == before
    # These IDs are observations asserted on this deterministic public replay.
    assert tuple(r.id for r in cylinder_refs) == tuple(range(1,25))
    assert authored == EntityRef("face",25)
    assert tuple(r.id for r in imprint.first_faces) == (26,27)
    assert ring_ids == set(range(13,25))


def test_nonfinite_owner_arc_frame_is_not_certified(monkeypatch):
    g, a, b = model_pair()
    evaluate = td.arc_frame

    def bad_frame(*args):
        return replace(evaluate(*args), radius=float("nan"))

    monkeypatch.setattr(td, "arc_frame", bad_frame)
    before = state(g)
    result = run(g, a, b)
    assert result.relation is R.UNRESOLVED
    assert result.diagnostics == ("owner_arc_evaluation_unqualified",)
    assert state(g) == before
