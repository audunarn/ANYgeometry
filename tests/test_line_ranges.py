"""Retained carrier regression definitions. UNRUN: root owns check authority."""
from dataclasses import replace
from fractions import Fraction as Q
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, LineRange, Straight, Plane
from anygeometry.definition_binding import definition_checksum, definition_value
from anygeometry.planar_exact import ExactPlane
from anygeometry.serialization import from_dict, to_dict, _checksum
from anygeometry._canonical_plate_joint import choose_station, JointRefused


def carrier():
    return LineRange((.123456789, -.78349127, .5), (1.234567891, .987654321, .5))


def test_carrier_retains_nonbinary_station_and_reversed_exact_partition():
    curve = carrier().subcurve(Q(1, 7), Q(6, 7)).subcurve(1, 0)
    left, right = curve.subcurve(0, Q(2, 3)), curve.subcurve(Q(2, 3), 1)
    assert left.a == right.a == curve.a
    assert left.b == right.b == curve.b
    assert left.domain[0] == curve.domain[0]
    assert left.domain[1] == right.domain[0]
    assert right.domain[1] == curve.domain[1]
    assert left.exact_point(1) == right.exact_point(0) == curve.exact_point(Q(2, 3))
    assert not isinstance(curve, Straight)


def test_evaluation_rounds_retained_carrier_without_refitting_proxy_chord():
    curve = carrier().subcurve(Q(1, 7), Q(2, 3))
    assert tuple(curve.evaluate(Q(3, 7))) == tuple(float(x) for x in curve.exact_point(Q(3, 7)))
    low, high = curve.bounds()
    for point in (curve.exact_point(0), curve.exact_point(1)):
        assert all(Q(float(a)) <= x <= Q(float(b)) for a, x, b in zip(low, point, high))


def test_repeated_reversal_preserves_complete_definition():
    curve = carrier().subcurve(Q(1, 11), Q(8, 11))
    assert curve.subcurve(1, 0).subcurve(1, 0) == curve


def test_transform_keeps_exact_directed_domains_and_authors_finite_carriers():
    curve = carrier().subcurve(Q(8, 11), Q(1, 11))
    matrix = np.eye(4)
    matrix[:3, 3] = [.123, -.17, 2.]
    transformed = curve.transformed(matrix)
    assert transformed.domain == curve.domain
    assert transformed.a == tuple(matrix[:3, :3] @ curve.a + matrix[:3, 3])
    assert transformed.b == tuple(matrix[:3, :3] @ curve.b + matrix[:3, 3])


def test_general_oblique_station_uses_raw_carrier_and_joint_union():
    curve = carrier()
    point = curve.exact_point(Q(2, 5))
    radius = Q(1, 10**12)
    low, high = tuple(x-radius for x in point), tuple(x+radius for x in point)
    q, lam = choose_station(tuple(map(Q, curve.a)), tuple(map(Q, curve.b)),
                            (Q(0), Q(0), Q(1)), Q(1, 2), low, high, 1.e-9, retained=True)
    assert lam == Q(2, 5)
    assert q == tuple(float(x) for x in point)


def test_joint_residual_is_independent_of_native_support_residual():
    curve = LineRange((0., 0., 5.e-9), (1., .7, 5.e-9))
    frame = ExactPlane.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    curve.certify_plane(frame, 1.e-8)
    point = tuple((a+b)/2 for a, b in zip(map(Q, curve.a), map(Q, curve.b)))
    projected = point[:2] + (Q(0),)
    with pytest.raises(JointRefused, match='joint tolerance'):
        choose_station(tuple(map(Q, curve.a)), tuple(map(Q, curve.b)), (Q(0), Q(0), Q(1)), Q(0),
                       projected, projected, 1.e-9, retained=True)


def test_whole_span_residual_checks_remote_end_not_only_station():
    frame = ExactPlane.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    curve = LineRange((0., 0., 0.), (1., 1., 2.e-8))
    with pytest.raises(GeometryError, match='native plane residual'):
        curve.certify_plane(frame, 1.e-8)
    with pytest.raises(GeometryError, match='native plane residual'):
        LineRange((0., 0., 0.), (1., 1., 2.e-9)).certify_plane(frame, 1.e-9)


def test_degenerate_projection_refuses():
    frame = ExactPlane.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    with pytest.raises(GeometryError, match='projection is degenerate'):
        LineRange((0., 0., 0.), (0., 0., 1.e-10)).certify_plane(frame)


def test_raw_same_revision_domain_edit_changes_binding_and_rejects_malformed_domain():
    curve = carrier()
    before = definition_checksum(curve)
    object.__setattr__(curve, 'domain', (Q(1, 3), Q(2, 3)))
    assert definition_checksum(curve) != before
    object.__setattr__(curve, 'domain', ({'fraction': [1, 3]}, Q(2, 3)))
    with pytest.raises(GeometryError, match='malformed'):
        curve.evaluate(0)


def test_existing_definition_encodings_remain_unchanged():
    value = {'a': [1., 2, True, None], 'b': ('x', 'y')}
    assert definition_value(value) == {'a': [1., 2, True, None], 'b': ['x', 'y']}
    assert definition_value(Q(2, 3)) == {'fraction': [2, 3]}


def test_model_split_reversal_and_fresh_json_round_trip():
    model = GeometryModel()
    curve = carrier().subcurve(Q(7, 9), Q(1, 9))
    start, end = (model.add_point(*curve.evaluate(t)) for t in (0, 1))
    edge = model.add_curve(start, end, curve)
    _, children = model.split_edge(edge, Q(1, 3))
    assert model.edges[children[0]].curve.domain[1] == model.edges[children[1]].curve.domain[0]
    document = json.loads(json.dumps(to_dict(model)))
    assert document['version'] == 8
    loaded = from_dict(document)
    assert loaded.edges[children[0]].curve == model.edges[children[0]].curve
    assert loaded.edges[children[1]].curve == model.edges[children[1]].curve


def test_loader_refuses_malformed_fraction_even_with_recomputed_checksum():
    model = GeometryModel()
    curve = carrier()
    start, end = (model.add_point(*curve.evaluate(t)) for t in (0, 1))
    model.add_curve(start, end, curve)
    document = to_dict(model)
    document['edges'][0]['curve']['domain'][0][1] = 0
    document['checksum'] = _checksum(document)
    with pytest.raises(GeometryError, match='fraction pairs'):
        from_dict(document)


@pytest.mark.parametrize('domain', [(Q(0), Q(0)), (Q(-1), Q(1)), (Q(0), Q(2))])
def test_invalid_retained_domains_refuse(domain):
    with pytest.raises(GeometryError):
        replace(carrier(), domain=domain)


@pytest.mark.parametrize('a,b,domain', [
    (None, (1., 1., 1.), (0, 1)),
    ((0., 0.), (1., 1., 1.), (0, 1)),
    ((0., 0., float('nan')), (1., 1., 1.), (0, 1)),
    ((0., 0., 0.), (1., 1., 1.), (0, float('inf'))),
    ((0., 0., 0.), (1., 1., 1.), None),
])
def test_malformed_public_carrier_input_has_typed_refusal(a, b, domain):
    with pytest.raises(GeometryError):
        LineRange(a, b, domain)


def _step_evidence(phase, event, **details):
    """Passive test-harness evidence, outside geometry callbacks and proof state."""
    path = os.environ.get('NURBS_STEP_EVIDENCE')
    if path:
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': phase, 'event': event,
                                     'monotonic_ns': time.monotonic_ns(), **details})+'\n')


@contextmanager
def _proof_phase(name):
    _step_evidence(name, 'begin')
    try:
        yield
    except BaseException as error:
        _step_evidence(name, 'error', exception=type(error).__name__)
        raise
    else:
        _step_evidence(name, 'end')


def _general_mantissa_public_cut(*, verify_plan_readonly=False):
    """Every caller runs the actual public plan/apply; no persisted receipt shortcut."""
    from anygeometry import plan_intersections, apply_intersections
    from anygeometry.nurbs import NurbsSurface
    from test_plane_split import bowl
    model, wall, plane = bowl()
    points = ((.3123456789, -1.123456789, .5), (1.5123456789, -1.123456789, .5),
              (1.5123456789, 1.123456789, .5), (.487654321, 1.123456789, .5))
    vertices = [model.add_point(*p) for p in points]
    edges = [model.add_line(vertices[i], vertices[(i+1) % 4]) for i in range(4)]
    plate = model.add_face(edges, surface=plane)
    wall_support = definition_checksum(model.faces[wall].surface)
    plate_support = definition_checksum(model.faces[plate].surface)
    oblique_carrier = (points[3], points[0])
    if verify_plan_readonly:
        with _proof_phase('serialize_source_before_plan'):
            before = to_dict(model)
    with _proof_phase('public_plan'):
        plan = plan_intersections(model, [('face', wall), ('face', plate)], policy='imprint')
    if verify_plan_readonly:
        with _proof_phase('serialize_source_after_plan'):
            assert to_dict(model) == before
    with _proof_phase('public_apply'):
        apply_intersections(model, plan, policy='imprint')
    wall_children = {f for f, face in model.faces.items() if type(face.surface) is NurbsSurface}
    plate_children = set(model.faces)-wall_children
    return (model, wall, plate, wall_support, plate_support, oblique_carrier,
            wall_children, plate_children, model._intersection_application_receipt[3])


def test_public_general_mantissa_oblique_plate_cut_material_topology_and_carriers():
    """Mandatory public split/material/topology contract, independent of native/load gates."""
    from anygeometry.entities import EntityRef
    from anygeometry.support_curves import PlaneNurbsCurve
    (model, wall, plate, wall_support, plate_support, oblique_carrier,
     wall_children, plate_children, traces) = _general_mantissa_public_cut(verify_plan_readonly=True)
    assert len(model.faces) == 4
    with _proof_phase('validate_cut_model'):
        assert not model.validate_topology()
    assert len(wall_children) == len(plate_children) == 2
    assert all(definition_checksum(model.faces[f].surface) == wall_support for f in wall_children)
    assert all(definition_checksum(model.faces[f].surface) == plate_support for f in plate_children)
    history = model.replacement_history()
    assert set(history[EntityRef('face', wall)]) == {EntityRef('face', f) for f in wall_children}
    assert set(history[EntityRef('face', plate)]) == {EntityRef('face', f) for f in plate_children}
    retained = {e: edge.curve for e, edge in model.edges.items() if type(edge.curve) is LineRange}
    assert retained
    assert all((curve.a, curve.b) == oblique_carrier for curve in retained.values())
    domains = sorted(tuple(sorted(curve.domain)) for curve in retained.values())
    assert domains[0][0] == 0 and domains[-1][1] == 1
    assert all(first[1] == second[0] for first, second in zip(domains, domains[1:]))
    assert traces and all(type(model.edges[e].curve) is PlaneNurbsCurve for e in traces)
    assert all(set(model.faces_using_edge(e)) == set(model.faces) for e in traces)


def _native_joint_symbols(current, traces, phase):
    from anygeometry._canonical_plate_joint import plate_joint
    symbols = {}
    with _proof_phase(phase):
        for edge_id in traces:
            edge = current.edges[edge_id]
            for vertex in (edge.start, edge.end):
                joint = plate_joint(current, vertex, edge_id)
                if joint is not None:
                    symbols.setdefault(vertex, set()).add(joint.identity)
    assert symbols and all(len(ids) == 1 for ids in symbols.values())
    return symbols


def _assert_native_face_admission(current, symbols, wall_children, plate_children, phase):
    from anygeometry.plane_source_reference import PlaneReference
    from anygeometry.regular_source_reference import NurbsReference
    assert len(wall_children) == len(plate_children) == 2
    assert wall_children | plate_children == set(current.faces)
    for face_id, face in current.faces.items():
        with _proof_phase(f'{phase}_construct_face_{face_id}'):
            reference = (NurbsReference(current, face) if face_id in wall_children
                         else PlaneReference(current, face))
        checked = 0
        for vertex, identities in symbols.items():
            if vertex not in reference.vertices:
                continue
            expected = next(iter(identities))
            if face_id in plate_children:
                expected = ('plane_root', reference.key, expected)
            assert reference.vertices[vertex].identity == expected
            checked += 1
        assert checked >= 2


def test_public_general_mantissa_oblique_plate_native_joint_and_all_face_admission():
    """Mandatory actual cut model native joint and all four incident face proofs."""
    model, _, _, _, _, _, wall_children, plate_children, traces = _general_mantissa_public_cut()
    symbols = _native_joint_symbols(model, traces, 'native_joint_cut')
    _assert_native_face_admission(model, symbols, wall_children, plate_children, 'native_cut')


def test_public_general_mantissa_oblique_plate_fresh_load_native_reconstruction():
    """Mandatory fresh JSON reconstruction; only joint identities are queried before load."""
    model, _, _, _, _, _, wall_children, plate_children, traces = _general_mantissa_public_cut()
    retained = {e: edge.curve for e, edge in model.edges.items() if type(edge.curve) is LineRange}
    symbols = _native_joint_symbols(model, traces, 'native_joint_before_load')
    with _proof_phase('serialize_cut_model'):
        document = json.loads(json.dumps(to_dict(model)))
    with _proof_phase('load_fresh_json'):
        loaded = from_dict(document)
    with _proof_phase('validate_loaded_model'):
        assert not loaded.validate_topology()
    assert {e: edge.curve for e, edge in loaded.edges.items() if type(edge.curve) is LineRange} == retained
    loaded_symbols = _native_joint_symbols(loaded, traces, 'native_joint_after_load')
    assert loaded_symbols == symbols
    _assert_native_face_admission(loaded, loaded_symbols, wall_children, plate_children, 'native_loaded')


@pytest.mark.parametrize('nonconvex', [False, True])
def test_complete_contact_ledger_classifies_only_proved_gaps(nonconvex):
    from test_bounded_nurbs_clipping import fixture, RECT, L_SHAPE
    from anygeometry.plane_section_plate import plate_loops
    from anygeometry.bounded_cut_graph import plate_roots, clip_path, _gap_inside, PlateContactRefused
    model, face, plate, plane, curve, frame = fixture(L_SHAPE if nonconvex else RECT)
    _, loops = plate_loops(model, model.faces[plate])
    ledger = plate_roots(model, plate, plane, frame, curve, 0, lambda: None, issue_ledger=True)
    roots = [root for root in ledger.rows if root.on_segment]
    decisions = clip_path(curve, frame, loops, roots, True, lambda: None, ledger=ledger)
    assert len(decisions) == len(roots)+1
    assert decisions[0][2] == decisions[-1][2]
    assert all(a[2] != b[2] for a,b in zip(decisions, decisions[1:]))
    with pytest.raises(PlateContactRefused, match='one certified contact-free'):
        _gap_inside(ledger, curve, frame, loops, 0, 1, lambda: None)


def test_contact_ledger_rejects_copy_forgery_stale_source_and_callback_mutation():
    from copy import copy
    from test_bounded_nurbs_clipping import fixture, RECT
    from anygeometry.plane_section_plate import plate_loops
    from anygeometry.bounded_cut_graph import plate_roots, _gap_inside, _ledger_gaps, PlateContactRefused
    model, face, plate, plane, curve, frame = fixture(RECT)
    _, loops = plate_loops(model, model.faces[plate])
    ledger = plate_roots(model, plate, plane, frame, curve, 0, lambda: None, issue_ledger=True)
    lo, hi = _ledger_gaps(ledger, curve, frame, loops)[0]
    for forged in (copy(ledger), replace(ledger)):
        with pytest.raises(PlateContactRefused, match='freshly owner-issued'):
            _gap_inside(forged, curve, frame, loops, lo, hi, lambda: None)
    old_domain = curve.domain
    def mutate():
        object.__setattr__(curve, 'domain', (.125, .875))
    try:
        with pytest.raises(PlateContactRefused, match='freshly owner-issued'):
            _gap_inside(ledger, curve, frame, loops, lo, hi, mutate)
    finally:
        object.__setattr__(curve, 'domain', old_domain)


def test_contact_gap_witness_cancellation_and_exhausted_work_do_not_mutate_source():
    from test_bounded_nurbs_clipping import fixture, RECT
    from anygeometry.plane_section_plate import plate_loops
    from anygeometry.bounded_cut_graph import plate_roots, _gap_inside, _ledger_gaps
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, plane, curve, frame = fixture(RECT)
    _, loops = plate_loops(model, model.faces[plate])
    ledger = plate_roots(model, plate, plane, frame, curve, 0, lambda: None, issue_ledger=True)
    lo, hi = _ledger_gaps(ledger, curve, frame, loops)[0]
    before = to_dict(model)
    def cancel():
        raise GeometryError('cancelled witness')
    with pytest.raises(GeometryError, match='cancelled witness'):
        _gap_inside(ledger, curve, frame, loops, lo, hi, cancel)
    token = CURRENT_WORK.set(Work(0, 32))
    try:
        with pytest.raises(GeometryError):
            _gap_inside(ledger, curve, frame, loops, lo, hi, lambda: None)
    finally:
        CURRENT_WORK.reset(token)
    assert to_dict(model) == before


def test_retained_boundary_service_uses_exact_projected_carrier_and_charges_proxy_transport():
    from anygeometry.entities import OrientedEdge
    from anygeometry.plane_section_plate import _line_endpoints
    model = GeometryModel()
    curve = carrier().subcurve(Q(1, 7), Q(6, 7))
    vertices = [model.add_point(*curve.evaluate(t)) for t in (0, 1)]
    edge = model.add_curve(*vertices, curve)
    frame = ExactPlane.from_plane(Plane((0.,0.,.5), (1.,0.,0.), (0.,1.,0.)))
    expected = tuple(frame.uv(curve.exact_point(t)) for t in (0, 1))
    assert _line_endpoints(model, OrientedEdge(edge, True), frame) == expected
    assert _line_endpoints(model, OrientedEdge(edge, False), frame) == expected[::-1]
    moved = np.asarray(model.vertex_position(vertices[0])) + [5.e-9, 0., 0.]
    object.__setattr__(model.vertices[vertices[0]], 'position', moved)
    with pytest.raises(GeometryError, match='proxy transport exceeds joint tolerance'):
        _line_endpoints(model, OrientedEdge(edge, True), frame)


def test_issued_contact_ledger_cannot_launder_dropped_roots_with_its_own_recomputed_hash():
    from test_bounded_nurbs_clipping import fixture, RECT
    from anygeometry.plane_section_plate import plate_loops
    from anygeometry.bounded_cut_graph import plate_roots, _gap_inside, _ledger_binding, PlateContactRefused
    model, face, plate, plane, curve, frame = fixture(RECT)
    _, loops = plate_loops(model, model.faces[plate])
    ledger = plate_roots(model, plate, plane, frame, curve, 0, lambda: None, issue_ledger=True)
    original_rows, original_binding = ledger.rows, ledger.binding
    calls = []
    try:
        object.__setattr__(ledger, 'rows', tuple(r for r in ledger.rows if not r.on_segment))
        object.__setattr__(ledger, 'binding', _ledger_binding(curve, frame, loops, ledger.rows))
        assert ledger.binding != original_binding
        with pytest.raises(PlateContactRefused, match='freshly owner-issued'):
            _gap_inside(ledger, curve, frame, loops, 0, 1, lambda: calls.append('witness'))
        assert not calls
    finally:
        object.__setattr__(ledger, 'rows', original_rows)
        object.__setattr__(ledger, 'binding', original_binding)


def test_contact_ledger_external_issuance_does_not_keep_the_ledger_alive():
    import gc
    import weakref
    from test_bounded_nurbs_clipping import fixture, RECT
    from anygeometry.bounded_cut_graph import plate_roots, _CONTACT_LEDGERS
    model, face, plate, plane, curve, frame = fixture(RECT)
    ledger = plate_roots(model, plate, plane, frame, curve, 0, lambda: None, issue_ledger=True)
    key, ref = id(ledger), weakref.ref(ledger)
    assert key in _CONTACT_LEDGERS
    del ledger
    gc.collect()
    assert ref() is None and key not in _CONTACT_LEDGERS


def test_shared_proxy_vertex_does_not_exempt_remote_retained_segment_crossing():
    from anygeometry.entities import OrientedEdge
    from anygeometry.section_faces import _retained_line_pair_verdict
    model = GeometryModel()
    eps = 2.**-34
    a = model.add_point(1., 3*eps, 0.)
    shared = model.add_point(0., 0., 0.)
    b = model.add_point(1., 5*eps, 0.)
    first = model.add_curve(a, shared, LineRange((1.,3*eps,0.), (0.,eps,0.)))
    second = model.add_curve(shared, b, LineRange((0.,-eps,0.), (1.,5*eps,0.)))
    plane = Plane((0.,0.,0.), (1.,0.,0.), (0.,1.,0.))
    assert _retained_line_pair_verdict(model, OrientedEdge(first, True), OrientedEdge(second, True), plane) is False


def test_same_retained_carrier_adjacent_children_share_one_exact_endpoint():
    from anygeometry.entities import OrientedEdge
    from anygeometry.section_faces import _retained_line_pair_verdict
    model = GeometryModel()
    curve = LineRange((0.,0.,0.), (1.,.7,0.))
    a, middle, b = (model.add_point(*curve.evaluate(t)) for t in (0, Q(1,3), 1))
    first = model.add_curve(a, middle, curve.subcurve(0,Q(1,3)))
    second = model.add_curve(middle, b, curve.subcurve(Q(1,3),1))
    plane = Plane((0.,0.,0.), (1.,0.,0.), (0.,1.,0.))
    assert _retained_line_pair_verdict(model, OrientedEdge(first, True), OrientedEdge(second, True), plane) is True
    assert _retained_line_pair_verdict(model, OrientedEdge(first, False), OrientedEdge(second, False), plane) is True
