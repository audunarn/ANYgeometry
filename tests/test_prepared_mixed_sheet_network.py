"""Current mixed Sheet network receipt: qualification and refusal adversaries."""
import dataclasses
import json

import pytest

from anygeometry import (
    GeometryModel, IntersectionBatchPolicy, OrientedEdge, plan_intersections,
    apply_intersections, trim_face, query_prepared_mixed_sheet_joint_network,
    validate_prepared_mixed_sheet_joint_network_binding,
    require_prepared_mixed_sheet_authored_material_coverage,
    require_prepared_mixed_sheet_reference_parameter_mapping,
)
from anygeometry.errors import GeometryError
from anygeometry.generators import cylinder
from anygeometry.prepared_mixed_sheet_network import _surface_correspondence


def _own_unowned(model, face_ids):
    owned = {model.face_uses[use].face_id
             for sheet in model.sheets.values() for use in sheet.face_use_ids}
    unowned = tuple(sorted(face for face in face_ids if face not in owned))
    if unowned:
        model.add_sheet(unowned)


def _mixed_bay(*, author_wall):
    """One connected-mixed bay with every authored face owned before planning."""
    model = GeometryModel()
    floor = model.add_plate(model.add_points(
        ((-3., -2., 0.), (7., -2., 0.), (7., 4., 0.), (-3., 4., 0.))))
    model.add_sheet((floor,))
    controls = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
    vector = (.25, 0., 1.5)
    vertices = model.add_points([(x, y, z) for x, y, z in controls])
    edge = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
    wall, = model.extrude((edge,), vector)
    if author_wall:
        model.add_sheet((wall,))
    generated = cylinder(.7, 8., origin=(-2., .4, .6), axis=(1., .2, .1),
                         radial_direction=(0., 1., 0.), circumferential_segments=8)
    for member in tuple(generated.members):
        generated.remove_member(member)
    model.insert_model(generated)
    if author_wall:
        _own_unowned(model, model.faces)
        owned = {model.face_uses[use].face_id
                 for sheet in model.sheets.values() for use in sheet.face_use_ids}
        assert owned == set(model.faces), 'mixed bay must be fully authored with owners'
    return model


def _plane_cylinder(*, hole=True, extra_member=False):
    """Small analytic plane/cylinder document with a seam, hole and joints."""
    model = GeometryModel()
    floor = model.add_plate(model.add_points(
        ((-3., -2., 0.), (7., -2., 0.), (7., 4., 0.), (-3., 4., 0.))))
    if hole:
        points = model.add_points(((1.75, .75, 0.), (2.25, .75, 0.),
                                   (2.25, 1.25, 0.), (1.75, 1.25, 0.)))
        edges = [model.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
        trim_face(model, floor, (tuple(OrientedEdge(edge, True) for edge in edges),))
    model.add_sheet((floor,))
    generated = cylinder(.7, 8., origin=(-2., 0., .6), axis=(1., 0., 0.),
                         radial_direction=(0., 1., 0.), circumferential_segments=8)
    for member in tuple(generated.members):
        generated.remove_member(member)
    model.insert_model(generated)
    _own_unowned(model, model.faces)
    if extra_member:
        start, end = model.add_points(((0., 2., 0.), (5., 2., 0.)))
        model.add_member((model.add_line(start, end),))
    return model


def _prepare(model):
    policy = IntersectionBatchPolicy()
    operands = [model.handle('face', face) for face in sorted(model.faces)]
    plan = plan_intersections(model, operands, policy=policy)
    apply_intersections(model, plan, policy=policy)
    return model


def _seed_joint_edge(model):
    for edge in sorted(model.edges):
        if len(list(model.sheets_using_edge(edge))) >= 2:
            return edge
    raise AssertionError('prepared model has no shared joint edge')


def _shared_edges(model):
    return {edge for edge in sorted(model.edges)
            if len(list(model.sheets_using_edge(edge))) >= 2}


@pytest.fixture(scope='module')
def mixed_model():
    return _prepare(_mixed_bay(author_wall=True))


@pytest.fixture(scope='module')
def mixed_receipt(mixed_model):
    return query_prepared_mixed_sheet_joint_network(
        mixed_model, _seed_joint_edge(mixed_model))


@pytest.fixture(scope='module')
def small_model():
    return _prepare(_plane_cylinder())


@pytest.fixture(scope='module')
def small_receipt(small_model):
    return query_prepared_mixed_sheet_joint_network(
        small_model, _seed_joint_edge(small_model))


def test_mixed_receipt_qualifies_current_document(mixed_model, mixed_receipt):
    receipt = mixed_receipt
    assert set(receipt.sheet_ids) == set(mixed_model.sheets)
    assert set(receipt.part_ids) == set(mixed_model.parts)
    assert len(receipt.authored_face_ids) == 10
    assert set(receipt.current_face_ids) == set(mixed_model.faces)
    assert set(receipt.junction_ids) == set(
        mixed_model.junctions)
    assert set(receipt.attachment_ids) == set(mixed_model.attachments)
    assert _shared_edges(mixed_model) == set(receipt.joint_edge_ids)
    # Root/source Sheet FaceUse occurrence correspondence is complete.
    assert {row[1] for row in receipt.occurrence_correspondence} == set(receipt.authored_face_ids)
    covered = {use for row in receipt.occurrence_correspondence for use in row[3]}
    assert covered == {use.id for use in mixed_model.face_uses.values()}
    # Complete current public trimmed charts, one per current face.
    charts = receipt.charts
    assert sorted(row['face'] for row in charts['faces']) == sorted(mixed_model.faces)
    assert {row['support'] for row in charts['faces']} <= {'Plane', 'Cylinder', 'ExtrudedSurface'}
    # Exact recorded surface definitions with carrier verdicts.
    rows = receipt.surface_correspondence
    assert sorted(row['face'] for row in rows) == sorted(mixed_model.faces)
    assert all(row['carrier_correspondence'] == 'exact_fields'
               for row in rows if row['source_surface']['type'] == 'cylinder')
    assert all(row['carrier_correspondence'] in ('exact_fields', 'recorded_only')
               for row in rows if row['source_surface']['type'] == 'plane')
    assert all(row['carrier_correspondence'] == 'recorded_only'
               for row in rows if row['source_surface']['type'] == 'coons')
    # Qualification flags cannot be inferred into coverage.
    assert receipt.occurrence_mapping_qualified is True
    for name in ('semantic_mapping_qualified', 'authored_material_coverage_qualified',
                 'reference_parameter_mapping_qualified', 'beam_discretization_qualified',
                 'load_transfer_qualified', 'solver_qualified', 'publication_qualified'):
        assert getattr(receipt, name) is False
    assert receipt.preserved_joint_attachment_ids == ()
    assert receipt.unqualified_semantics


def test_mixed_binding_is_deterministic(mixed_model, mixed_receipt):
    validate_prepared_mixed_sheet_joint_network_binding(mixed_model, mixed_receipt)
    again = query_prepared_mixed_sheet_joint_network(
        mixed_model, mixed_receipt.joint_edge_id,
        expected_revision=mixed_receipt.scope.face_preimages.revision)
    assert again == mixed_receipt


def test_mixed_typed_refusals(mixed_receipt):
    with pytest.raises(GeometryError, match='does not qualify authored material coverage'):
        require_prepared_mixed_sheet_authored_material_coverage(mixed_receipt)
    with pytest.raises(GeometryError, match='does not qualify reference parameter'):
        require_prepared_mixed_sheet_reference_parameter_mapping(mixed_receipt)


def test_small_plane_cylinder_receipt(small_model, small_receipt):
    receipt = small_receipt
    validate_prepared_mixed_sheet_joint_network_binding(small_model, receipt)
    assert _shared_edges(small_model) == set(receipt.joint_edge_ids)
    charts = receipt.charts
    assert sorted(row['face'] for row in charts['faces']) == sorted(small_model.faces)
    # The authored square hole survives as a complete two-loop chart.
    assert any(len(row['boundary_loops']) >= 2 for row in charts['faces'])
    # Seam/inner canonical edges stay single-Sheet occurrences; every shared
    # edge carries a complete qualified generated Sheet joint.
    for edge in sorted(small_model.edges):
        owners = list(small_model.sheets_using_edge(edge))
        if len(owners) >= 2:
            assert edge in receipt.joint_edge_ids
        else:
            assert edge not in receipt.joint_edge_ids


def test_forged_flags_refused(small_model, small_receipt):
    for name in ('authored_material_coverage_qualified', 'publication_qualified',
                 'reference_parameter_mapping_qualified', 'solver_qualified'):
        forged = dataclasses.replace(small_receipt, **{name: True})
        with pytest.raises(GeometryError, match='forged qualification flags'):
            validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)


def _reencoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)


def test_forged_inventory_and_records_refused(small_model, small_receipt):
    receipt = small_receipt
    forged = dataclasses.replace(
        receipt, sheet_ids=receipt.sheet_ids + (receipt.sheet_ids[-1],))
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    forged = dataclasses.replace(receipt, current_face_ids=receipt.current_face_ids[:-1])
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    forged = dataclasses.replace(
        receipt, occurrence_correspondence=receipt.occurrence_correspondence[:-1])
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    records = json.loads(receipt.current_records_json)
    records['faces'][0]['surface']['origin'] = [123., 456., 789.]
    forged = dataclasses.replace(receipt, current_records_json=_reencoded(records))
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    charts = json.loads(receipt.charts_json)
    charts['faces'][0]['material_area'] = charts['faces'][0]['material_area'] + 1.
    forged = dataclasses.replace(receipt, charts_json=_reencoded(charts))
    assert forged.charts_json != receipt.charts_json
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    surfaces = json.loads(receipt.surface_correspondence_json)
    # Overstating a tolerance-certified Plane carrier to exact_fields is a
    # genuinely different value only on a row that is not already exact.
    row = next(item for item in surfaces if item['carrier_correspondence'] != 'exact_fields')
    row['carrier_correspondence'] = 'exact_fields'
    forged = dataclasses.replace(receipt, surface_correspondence_json=_reencoded(surfaces))
    assert forged.surface_correspondence_json != receipt.surface_correspondence_json
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    # A genuinely different carrier field must also be refused.
    surfaces = json.loads(receipt.surface_correspondence_json)
    row = next(item for item in surfaces if item['current_surface']['type'] == 'plane')
    row['current_surface']['origin'] = [value + 1. for value in row['current_surface']['origin']]
    forged = dataclasses.replace(receipt, surface_correspondence_json=_reencoded(surfaces))
    assert forged.surface_correspondence_json != receipt.surface_correspondence_json
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)


def test_unowned_wall_source_less_owner_refused():
    model = _prepare(_mixed_bay(author_wall=False))
    with pytest.raises(GeometryError, match='source-less'):
        query_prepared_mixed_sheet_joint_network(model, _seed_joint_edge(model))


def test_disconnected_document_refused():
    model = _plane_cylinder(hole=False)
    far = model.add_plate(model.add_points(
        ((100., 0., 0.), (101., 0., 0.), (101., 1., 0.), (100., 1., 0.))))
    model.add_sheet((far,))
    _prepare(model)
    with pytest.raises(GeometryError, match='one connected whole-document Sheet component'):
        query_prepared_mixed_sheet_joint_network(model, _seed_joint_edge(model))


def test_member_document_refused():
    model = _prepare(_plane_cylinder(hole=False, extra_member=True))
    with pytest.raises(GeometryError, match='zero Members'):
        query_prepared_mixed_sheet_joint_network(model, _seed_joint_edge(model))


def test_cancellation_refused(small_model):
    with pytest.raises(GeometryError, match='cancelled'):
        query_prepared_mixed_sheet_joint_network(
            small_model, _seed_joint_edge(small_model),
            cancellation_check=lambda phase: True)


def test_callback_mutation_refused():
    model = _prepare(_plane_cylinder(hole=False))
    edge = _seed_joint_edge(model)
    calls = []

    def mutate(phase):
        if len(calls) == 1:
            model.add_point(50., 50., 50.)
        calls.append(phase)
        return False

    with pytest.raises(GeometryError):
        query_prepared_mixed_sheet_joint_network(model, edge, cancellation_check=mutate)


def test_stale_receipt_refused():
    model = _prepare(_plane_cylinder(hole=False))
    receipt = query_prepared_mixed_sheet_joint_network(model, _seed_joint_edge(model))
    model.add_point(25., 25., 25.)
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(model, receipt)


def test_forged_occurrence_correspondence_refused(small_model, small_receipt):
    receipt = small_receipt
    rows = receipt.occurrence_correspondence
    first, second = rows[0], rows[1]
    # Substituting another root's original FaceUse into a current occurrence
    # set forges the occurrence identity/orientation carrier; orientation is
    # rederived from the live document, never trusted from the tuple.
    forged_uses = first[3][:-1] + (second[2],)
    forged = dataclasses.replace(
        receipt,
        occurrence_correspondence=((first[0], first[1], first[2], forged_uses),) + rows[1:])
    assert forged.occurrence_correspondence != receipt.occurrence_correspondence
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    # Forging the original FaceUse column is refused the same way.
    forged = dataclasses.replace(
        receipt,
        occurrence_correspondence=((first[0], first[1], second[2], first[3]),) + rows[1:])
    assert forged.occurrence_correspondence != receipt.occurrence_correspondence
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)


def test_forged_charts_refused(small_model, small_receipt):
    receipt = small_receipt
    charts = json.loads(receipt.charts_json)
    row = charts['faces'][0]
    row['face_uses'] = list(row['face_uses']) + [row['face_uses'][0]]
    forged = dataclasses.replace(receipt, charts_json=_reencoded(charts))
    assert forged.charts_json != receipt.charts_json
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)
    charts = json.loads(receipt.charts_json)
    row = charts['faces'][0]
    row['boundary_loops'] = [count + 1 for count in row['boundary_loops']]
    forged = dataclasses.replace(receipt, charts_json=_reencoded(charts))
    assert forged.charts_json != receipt.charts_json
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(small_model, forged)


def test_callback_binding_mutation_refused():
    model = _prepare(_plane_cylinder(hole=False))
    receipt = query_prepared_mixed_sheet_joint_network(model, _seed_joint_edge(model))
    pinned = (receipt.source_records_json, receipt.current_records_json,
              receipt.charts_json, receipt.surface_correspondence_json,
              receipt.occurrence_correspondence, receipt.sheet_ids)
    # The owner receipt is immutable: no callback can mutate it in place.
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.sheet_ids = ()
    calls = []

    def mutate(phase):
        if len(calls) == 1:
            model.add_point(70., 70., 70.)
        calls.append(phase)
        return False

    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(
            model, receipt, cancellation_check=mutate)
    # The receipt survives the refused adversarial callback unchanged.
    assert (receipt.source_records_json, receipt.current_records_json,
            receipt.charts_json, receipt.surface_correspondence_json,
            receipt.occurrence_correspondence, receipt.sheet_ids) == pinned


_CONE = {'type': 'cone', 'origin': [0., 0., 0.], 'axis': [0., 0., 1.],
         'radial_direction': [1., 0., 0.], 'radius_start': 1., 'radius_end': 2.,
         'height': 2., 'start_angle': 0., 'sweep_angle': 6.283185307179586}
_CYLINDER = {'type': 'cylinder', 'origin': [0., 0., 0.], 'axis': [0., 0., 1.],
             'radial_direction': [1., 0., 0.], 'radius': .7, 'height': 2.,
             'start_angle': 0., 'sweep_angle': 6.283185307179586}


def _carrier_documents(source_surface, current_surface):
    source = {'faces': {1: {'surface': source_surface, 'parameterization': None}}}
    data = {'faces': {5: {'surface': current_surface, 'parameterization': None}}}
    return source, data, {5: 1}


def test_cone_height_is_not_a_restricted_field():
    # A Cone carrier must retain its serialized height exactly; only the
    # angular restriction fields may narrow on a qualified split.
    source, data, roots = _carrier_documents(_CONE, dict(_CONE, height=1.))
    with pytest.raises(GeometryError, match='surface carrier definition changed'):
        _surface_correspondence(source, data, roots, {}, lambda: None)
    source, data, roots = _carrier_documents(
        _CONE, dict(_CONE, start_angle=.5, sweep_angle=3.))
    rows = _surface_correspondence(source, data, roots, {}, lambda: None)
    assert rows[0]['carrier_correspondence'] == 'exact_fields'


def test_cylinder_height_restriction_is_type_specific():
    # Height may narrow only on a qualified Cylinder split; every other
    # serialized Cylinder field stays an invariant carrier definition.
    source, data, roots = _carrier_documents(_CYLINDER, dict(_CYLINDER, height=1.))
    rows = _surface_correspondence(source, data, roots, {}, lambda: None)
    assert rows[0]['carrier_correspondence'] == 'exact_fields'
    source, data, roots = _carrier_documents(_CYLINDER, dict(_CYLINDER, radius=.8))
    with pytest.raises(GeometryError, match='surface carrier definition changed'):
        _surface_correspondence(source, data, roots, {}, lambda: None)


@pytest.fixture(scope='module')
def reopened():
    from examples.reopened_sheet_joint_component_handoff import build
    _, closure, _, joint, _ = build()
    model = closure.working_model
    return model, query_prepared_mixed_sheet_joint_network(model, joint)


def test_preserved_joint_inventory_is_recorded_and_bound(reopened):
    model, receipt = reopened
    validate_prepared_mixed_sheet_joint_network_binding(model, receipt)
    # The reopened closure keeps its original joint unchanged; the mixed
    # receipt records that complete preserved inventory explicitly.
    assert receipt.preserved_joint_attachment_ids == (1, 2)
    assert receipt.preserved_joint_junction_ids == (1,)
    assert receipt.source_records['attachments'] == receipt.current_records['attachments']
    assert receipt.source_records['junctions'] == receipt.current_records['junctions']


@pytest.mark.parametrize('field', ('preserved_joint_attachment_ids',
                                    'preserved_joint_junction_ids'))
def test_forged_preserved_joint_inventory_refused(reopened, field):
    model, receipt = reopened
    forged = dataclasses.replace(receipt, **{field: ()})
    assert getattr(forged, field) != getattr(receipt, field)
    with pytest.raises(GeometryError):
        validate_prepared_mixed_sheet_joint_network_binding(model, forged)


def test_empty_part_refused():
    # An empty Part authored before the normal fixture preparation is never
    # selected by the connected Sheet component and must be typed-refused.
    model = _plane_cylinder(hole=False)
    model.add_part()
    _prepare(model)
    with pytest.raises(GeometryError, match='empty or unselected current Parts'):
        query_prepared_mixed_sheet_joint_network(model, _seed_joint_edge(model))


def test_plane_basis_residuals_do_not_certify_whole_material_distance():
    tolerance = 1e-8
    original = dict(type='plane', origin=[0., 0., 0.],
                    u_vector=[1., 0., 0.], v_vector=[0., 1., 0.])
    current = dict(original, origin=[0., 0., tolerance / 2],
                   u_vector=[1., 0., tolerance / 2],
                   v_vector=[0., 1., tolerance / 2])
    source, data, roots = _carrier_documents(original, current)
    row = _surface_correspondence(source, data, roots, {5: tolerance}, lambda: None)[0]
    assert all(value <= row['carrier_tolerance_bound'] for value in row['carrier_residuals'])
    # Current support at UV (1, 1) is farther than tolerance from z=0.
    assert sum(current[key][2] for key in ('origin', 'u_vector', 'v_vector')) > tolerance
    assert row['carrier_correspondence'] == 'recorded_only'
    source, data, roots = _carrier_documents(original, dict(original))
    assert _surface_correspondence(source, data, roots, {5: tolerance}, lambda: None)[0][
        'carrier_correspondence'] == 'exact_fields'
