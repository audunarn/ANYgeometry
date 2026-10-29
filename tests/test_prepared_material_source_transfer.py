"""Live clone/source authority; numerical execution intentionally deferred."""
from dataclasses import replace
import pytest
from anygeometry import (GeometryError, clone_prepared_geometry, query_material_surface_regions,
    query_prepared_material_boundary_chords, validate_prepared_material_curvilinear_cells,
    CurvilinearMaterialCell, query_prepared_material_source_transfer,
    validate_prepared_material_source_transfer_binding, to_dict, from_dict)
from test_prepared_material_curvilinear import square_cells


def prepared_square(*, unselected_cylinder=False):
    source, _, _, _ = square_cells()
    if unselected_cylinder:
        from test_cylinder_ruling_batch import panel
        support, loop = panel(source, (10., 0., 0.), (3., 4.))
        source.add_face_from_loop(loop, surface=support)
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    face = next(iter(prepared.faces))
    loop = prepared.faces[face].loop
    regions = query_material_surface_regions(prepared, [face])
    chords = query_prepared_material_boundary_chords(prepared, regions,
        tuple((prepared.handle('edge', use.edge), 0., 1.) for use in loop), chord_error=.01)
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    cell = CurvilinearMaterialCell(prepared.handle('face', face), (1, 2, 3, 4), uv,
        tuple((*p, 0.) for p in uv), tuple((i, i, True) for i in range(4)))
    receipt = validate_prepared_material_curvilinear_cells(prepared, regions, chords, (cell,), surface_error=.001)
    return source, prepared, regions, chords, receipt


def test_live_immediate_clone_transfers_literal_cells_and_exact_unchanged_edges():
    source, prepared, regions, chords, cell = prepared_square()
    original = to_dict(source)
    result = query_prepared_material_source_transfer(source, prepared, (cell,))
    validate_prepared_material_source_transfer_binding(source, prepared, result, cell_receipts=(cell,))
    assert result.prepared_cells == cell.cells
    assert result.source_cells[0].face.model_id == source.model_id
    assert result.source_cells[0].xyz == cell.cells[0].xyz
    assert len(result.edge_parameter_maps) == 4
    assert all(row[2:] == (0, 1) for row in result.edge_parameter_maps)
    assert not result.source_trace_ownership
    assert len(result.boundary_station_correspondence) == 8
    assert to_dict(source) == original


def test_transfer_copy_ordinary_clone_and_loaded_model_have_no_authority():
    source, prepared, regions, chords, cell = prepared_square()
    result = query_prepared_material_source_transfer(source, prepared, (cell,))
    with pytest.raises(GeometryError, match='not issued'):
        validate_prepared_material_source_transfer_binding(source, prepared, replace(result), cell_receipts=(cell,))
    for copied in (prepared.clone(), from_dict(to_dict(prepared))):
        with pytest.raises(GeometryError):
            query_prepared_material_source_transfer(source, copied, (cell,))


@pytest.mark.parametrize('operand', ('cells', 'chords', 'regions', 'source'))
@pytest.mark.parametrize('phase', ('source transfer qualification', 'source transfer qualification complete'))
def test_transfer_callback_pins_all_owner_operands_before_proof_reads(operand, phase):
    source, prepared, regions, chords, receipt = prepared_square()
    if operand == 'cells':
        target, name, bad = receipt.cells[0], 'xyz', ((2., 0., 0.), *receipt.cells[0].xyz[1:])
    elif operand == 'chords':
        target, name, bad = chords, 'chord_error', .9
    elif operand == 'regions':
        target, name, bad = regions.regions[0], 'material_area', 999.
    else:
        # Full public geometry edits at the same revision must invalidate transfer.
        target, name = source.vertices[next(iter(source.vertices))], 'position'
        bad = target.position.copy(); bad[0] += 9.
    original = getattr(target, name)
    def callback(current):
        if current == phase:
            object.__setattr__(target, name, bad)
        return False
    try:
        with pytest.raises(GeometryError):
            query_prepared_material_source_transfer(source, prepared, (receipt,), cancellation_check=callback)
    finally:
        object.__setattr__(target, name, original)


def test_transfer_final_cancellation_and_stale_receipt_refuse():
    source, prepared, _, _, cell = prepared_square()
    before = to_dict(prepared)
    with pytest.raises(GeometryError, match='cancelled'):
        query_prepared_material_source_transfer(source, prepared, (cell,),
            cancellation_check=lambda phase: phase == 'source transfer qualification complete')
    assert to_dict(prepared) == before
    result = query_prepared_material_source_transfer(source, prepared, (cell,))
    object.__setattr__(result, 'surface_errors', ())
    with pytest.raises(GeometryError, match='changed'):
        validate_prepared_material_source_transfer_binding(source, prepared, result, cell_receipts=(cell,))


def test_qualified_publication_restores_external_lineage_on_registration_failure(monkeypatch):
    from anygeometry import Plane, plan_plane_split, apply_plane_split
    from anygeometry import prepared_material_source_transfer as transport
    source, prepared, _, _, cell = prepared_square()
    before, lineage = to_dict(prepared), transport._LINKS[prepared]
    plan = plan_plane_split(prepared, list(prepared.faces), Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    original = transport._publish_source_lineage
    def fail(model, staged):
        original(model, staged)
        raise RuntimeError('injected source lineage registration failure')
    monkeypatch.setattr(transport, '_publish_source_lineage', fail)
    with pytest.raises(RuntimeError, match='injected'):
        apply_plane_split(prepared, plan, policy='imprint')
    assert to_dict(prepared) == before
    assert transport._LINKS[prepared] is lineage


def test_temporary_repair_restore_does_not_make_invalid_cell_receipt_authoritative():
    source, prepared, _, _, cell = prepared_square()
    original = cell.cells[0].xyz
    bad = ((3., 0., 0.), *original[1:])
    object.__setattr__(cell.cells[0], 'xyz', bad)
    def repair(phase):
        object.__setattr__(cell.cells[0], 'xyz', original if phase == 'source transfer qualification' else bad)
        return False
    try:
        with pytest.raises(GeometryError):
            query_prepared_material_source_transfer(source, prepared, (cell,), cancellation_check=repair)
    finally:
        object.__setattr__(cell.cells[0], 'xyz', original)


@pytest.mark.parametrize('owner', ('source', 'prepared'))
@pytest.mark.parametrize('attack', ('before_binding', 'restore_at_first_callback', 'final_callback'))
def test_existing_transfer_binds_full_issued_native_state(owner, attack):
    from anygeometry import Cylinder
    source, prepared, _, _, cell = prepared_square(unselected_cylinder=True)
    result = query_prepared_material_source_transfer(source, prepared, (cell,))
    model = source if owner == 'source' else prepared
    support = next(face.surface for face in model.faces.values() if isinstance(face.surface, Cylinder))
    stored = support._circumferential
    forged = stored.copy() * (1 + 2**-40)
    before = to_dict(model)
    calls = []
    if attack != 'final_callback':
        object.__setattr__(support, '_circumferential', forged)
        assert to_dict(model) == before  # Native-only, same serialized revision/checksum.
    def callback(phase):
        calls.append(phase)
        if attack == 'restore_at_first_callback':
            object.__setattr__(support, '_circumferential', stored)
        elif attack == 'final_callback' and phase == 'source transfer binding complete':
            object.__setattr__(support, '_circumferential', forged)
        return False
    try:
        with pytest.raises(GeometryError):
            validate_prepared_material_source_transfer_binding(source, prepared, result,
                cell_receipts=(cell,), cancellation_check=callback)
        if attack != 'final_callback':
            assert not calls  # Issued native identity is checked before user code.
    finally:
        object.__setattr__(support, '_circumferential', stored)

def test_public_clone_split_mixed_children_transfer_exact_source_edge_intervals():
    from anygeometry import Plane, plan_plane_split, apply_plane_split
    from test_mixed_source_junctions import fan_receipt
    source, _, _, _ = square_cells()
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    plan = plan_plane_split(prepared, list(prepared.faces),
        Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    result = apply_plane_split(prepared, plan, policy='imprint')
    descendants = dict(result.parent_children)[next(iter(source.faces))]
    assert len(descendants) == 2
    receipts = tuple(fan_receipt(prepared, prepared.faces[face]) for face in descendants)
    with pytest.raises(GeometryError, match='omits required material descendants'):
        query_prepared_material_source_transfer(source, prepared, receipts[:1])
    with pytest.raises(GeometryError, match='duplicate prepared material coverage'):
        query_prepared_material_source_transfer(source, prepared, (receipts[0], receipts[0]))
    transfer = query_prepared_material_source_transfer(source, prepared, receipts)
    validate_prepared_material_source_transfer_binding(source, prepared, transfer, cell_receipts=receipts)
    assert transfer.source_to_prepared_faces == ((source.handle('face', next(iter(source.faces))),
        tuple(prepared.handle('face', child) for child in descendants)),)
    assert len(transfer.source_trace_ownership) == len(result.trace_edges)
    assert all(owners == (source.handle('face', next(iter(source.faces))),)
               for trace, owners in transfer.source_trace_ownership)
    covers = {}
    for child, original, a, b in transfer.edge_parameter_maps:
        covers.setdefault(original.id, []).append((a, b))
    for intervals in covers.values():
        reach = 0
        for a, b in sorted(intervals):
            assert a == reach
            reach = b
        assert reach == 1
    maps = {child.id: (original, a, b) for child, original, a, b in transfer.edge_parameter_maps}
    for edge, t, original, source_t, owners in transfer.boundary_station_correspondence:
        if original is not None:
            parent, a, b = maps[edge.id]
            assert parent == original and source_t == a+(b-a)*t
        else:
            assert owners == (source.handle('face', next(iter(source.faces))),)

def test_repeated_public_axis_cuts_compose_original_edge_intervals_through_genuine_clone():
    from anygeometry import Plane, plan_plane_split, apply_plane_split
    from test_mixed_source_junctions import fan_receipt
    source, _, _, _ = square_cells()
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    for station in (.6, .3):
        plan = plan_plane_split(prepared, list(prepared.faces),
            Plane((station, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        apply_plane_split(prepared, plan, policy='imprint')
        if station == .6:
            prepared = clone_prepared_geometry(prepared, new_preparation_epoch=True)
    assert len(prepared.faces) == 3
    receipts = tuple(fan_receipt(prepared, face) for face in prepared.faces.values())
    transfer = query_prepared_material_source_transfer(source, prepared, receipts)
    validate_prepared_material_source_transfer_binding(source, prepared, transfer, cell_receipts=receipts)
    assert {face.id for _, children in transfer.source_to_prepared_faces for face in children} == set(prepared.faces)
    covers = {}
    for edge, original, a, b in transfer.edge_parameter_maps:
        covers.setdefault(original.id, []).append((a, b))
    assert set(covers) == set(source.edges)
    assert max(len(value) for value in covers.values()) == 3
    for intervals in covers.values():
        reach = 0
        for a, b in sorted(intervals):
            assert a == reach
            reach = b
        assert reach == 1
    owners = {edge.id: faces for edge, faces in transfer.source_trace_ownership}
    for edge, t, original, source_t, faces in transfer.boundary_station_correspondence:
        if original is None:
            assert faces == owners[edge.id]


def test_forged_producer_partition_refuses_before_publication_and_keeps_lineage(monkeypatch):
    from anygeometry import Plane, plan_plane_split, apply_plane_split
    import anygeometry.plane_split as split_owner
    from anygeometry import prepared_material_source_transfer as transport
    source, prepared, _, _, cell = prepared_square()
    before, old = to_dict(prepared), transport._LINKS[prepared]
    plan = plan_plane_split(prepared, list(prepared.faces),
        Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    row = split_owner._PLANS[plan]
    partition = row[-1]
    forged = tuple(list(partition))
    assert forged == partition and forged is not partition
    split_owner._PLANS[plan] = (*row[:-1], forged)
    with pytest.raises(GeometryError, match='not owner-issued'):
        apply_plane_split(prepared, plan, policy='imprint')
    assert to_dict(prepared) == before
    assert transport._LINKS[prepared] is old


def test_source_transfer_edge_map_literals_cannot_be_forged_after_issue():
    source, prepared, _, _, cell = prepared_square()
    receipt = query_prepared_material_source_transfer(source, prepared, (cell,))
    object.__setattr__(receipt, 'edge_parameter_maps', ())
    with pytest.raises(GeometryError, match='changed'):
        validate_prepared_material_source_transfer_binding(source, prepared, receipt, cell_receipts=(cell,))

@pytest.mark.parametrize('reverse', (False, True))
def test_source_edge_transport_retains_weighted_cropped_and_reversed_native_domains(reverse):
    from fractions import Fraction as F
    from anygeometry import Plane, plan_plane_split, apply_plane_split
    from anygeometry.nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
    from test_mixed_source_junctions import fan_receipt
    source, _, _, _ = square_cells()
    face = source.faces[next(iter(source.faces))]
    # Keep the physical unit square but place its trim in an inset chart.
    # The weighted, cropped carrier then has its ENTIRE control hull inside
    # the supported chart, including controls outside the active interval.
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((-1.5, -1.5, 0.), (-1.5, 2.5, 0.)), ((2.5, -1.5, 0.), (2.5, 2.5, 0.))))
    uv = ((.375, .375), (.625, .375), (.625, .625), (.375, .625))
    with source.transaction():
        source._put_entity('face', replace(face, surface=support))
        for index, use in enumerate(face.loop):
            first, last = uv[index], uv[(index+1)%4]
            controls = tuple(tuple(a+factor*(b-a) for a, b in zip(first, last)) for factor in (-1.25, 1.25))
            weights, domain = (1., 3.), (3., 5.)
            if reverse:
                controls, weights, domain = controls[::-1], weights[::-1], domain[::-1]
            edge = source.edges[use.edge]
            pcurve = NurbsCurve(1, (2., 2., 6., 6.), controls, weights, domain=domain)
            assert all(0 <= value <= 1 for point in controls for value in point)
            source._put_entity('edge', replace(edge, curve=CurveOnSurface(support, pcurve)))
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    plan = plan_plane_split(prepared, list(prepared.faces),
        Plane((.5, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    apply_plane_split(prepared, plan, policy='imprint')
    cells = tuple(fan_receipt(prepared, child) for child in prepared.faces.values())
    transfer = query_prepared_material_source_transfer(source, prepared, cells)
    validate_prepared_material_source_transfer_binding(source, prepared, transfer, cell_receipts=cells)
    for edge, original, a, b in transfer.edge_parameter_maps:
        domain = tuple(F(float(value)) for value in source.edges[original.id].curve.pcurve.domain)
        child = tuple(F(float(value)) for value in prepared.edges[edge.id].curve.pcurve.domain)
        assert child == tuple(domain[0]+(domain[1]-domain[0])*value for value in (a, b))
    assert any(a != 0 or b != 1 for edge, original, a, b in transfer.edge_parameter_maps)


@pytest.mark.parametrize('reverse', (False, True))
def test_source_edge_transport_crops_children_of_weighted_full_chart_nonunit_domain(reverse):
    """Separate full-source case; the inset pre-cropped-source case stays above."""
    from fractions import Fraction as F
    from anygeometry import Plane, plan_plane_split, apply_plane_split
    from anygeometry.nurbs import NurbsCurve
    from test_mixed_source_junctions import fan_receipt
    source, _, _, _ = square_cells()
    face = source.faces[next(iter(source.faces))]
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    with source.transaction():
        for index, use in enumerate(face.loop):
            controls, weights, domain = (uv[index], uv[(index+1)%4]), (1., 3.), (3., 5.)
            if reverse:
                controls, weights, domain = controls[::-1], weights[::-1], domain[::-1]
            edge = source.edges[use.edge]
            pcurve = NurbsCurve(1, (3., 3., 5., 5.), controls, weights, domain=domain)
            source._put_entity('edge', replace(edge, curve=replace(edge.curve, pcurve=pcurve)))
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    plan = plan_plane_split(prepared, list(prepared.faces),
        Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    apply_plane_split(prepared, plan, policy='imprint')
    cells = tuple(fan_receipt(prepared, child) for child in prepared.faces.values())
    transfer = query_prepared_material_source_transfer(source, prepared, cells)
    validate_prepared_material_source_transfer_binding(source, prepared, transfer, cell_receipts=cells)
    for edge, original, a, b in transfer.edge_parameter_maps:
        domain = tuple(F(float(value)) for value in source.edges[original.id].curve.pcurve.domain)
        child = tuple(F(float(value)) for value in prepared.edges[edge.id].curve.pcurve.domain)
        assert child == tuple(domain[0]+(domain[1]-domain[0])*value for value in (a, b))
    assert any(a != 0 or b != 1 for edge, original, a, b in transfer.edge_parameter_maps)

@pytest.mark.parametrize('poison', ('source', 'candidate'))
def test_plane_split_rejects_native_only_same_revision_source_or_candidate_edits(poison):
    from anygeometry import Cylinder, Plane, plan_plane_split, apply_plane_split
    from anygeometry import prepared_material_source_transfer as transport
    import anygeometry.plane_split as split_owner
    source, prepared, _, _, cell = prepared_square(unselected_cylinder=True)
    before = to_dict(prepared)
    old_lineage = transport._LINKS[prepared]
    plan = plan_plane_split(prepared, [cell.cells[0].face.id],
        Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    candidate = split_owner._PLANS[plan][2]
    target = prepared if poison == 'source' else candidate
    support = next(face.surface for face in target.faces.values() if isinstance(face.surface, Cylinder))
    original = support._circumferential
    changed = original.copy() * (1+2**-40)
    object.__setattr__(support, '_circumferential', changed)
    assert to_dict(prepared) == before  # Same revision/checksum despite native edit.
    try:
        with pytest.raises(GeometryError, match='stale|staged definition changed'):
            apply_plane_split(prepared, plan, policy='imprint')
        assert to_dict(prepared) == before
        assert transport._LINKS[prepared] is old_lineage
        assert support._circumferential is changed
    finally:
        object.__setattr__(support, '_circumferential', original)
