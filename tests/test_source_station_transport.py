"""Genuine source/cell/chord alias contracts. UNRUN; root owns execution."""
from dataclasses import replace
from fractions import Fraction as Q
from copy import copy
import pytest

from anygeometry import (GeometryError, Plane, clone_prepared_geometry,
    plan_plane_split, apply_plane_split, query_material_surface_regions,
    query_prepared_material_boundary_chords, validate_prepared_material_curvilinear_cells,
    CurvilinearMaterialCell, query_prepared_material_source_transfer,
    query_prepared_source_station_transport, validate_prepared_source_station_transport_binding,
    to_dict, CancellationSignal)


def genuine_station_inputs(*, split=True, midpoint=True):
    from test_prepared_material_curvilinear import square_cells
    source,_,_,_ = square_cells()
    prepared = clone_prepared_geometry(source,new_preparation_epoch=True)
    if split:
        plan = plan_plane_split(prepared,list(prepared.faces),Plane((.6,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        apply_plane_split(prepared,plan,policy='imprint')
    node_map, receipts, all_chords = {}, [], []
    next_node = max(prepared.vertices)+1
    for face in prepared.faces.values():
        regions = query_material_surface_regions(prepared,[face.id])
        parts = ((0.,.5),(.5,1.)) if midpoint else ((0.,1.),)
        intervals = tuple((prepared.handle('edge',use.edge),a,b) for use in face.loop for a,b in parts)
        chords = query_prepared_material_boundary_chords(prepared,regions,intervals,chord_error=.0025)
        ring = []
        for i,use in enumerate(face.loop):
            indices = list(range(i*len(parts),(i+1)*len(parts)))
            if not use.forward:
                indices.reverse()
            for index in indices:
                chord = chords.intervals[index]
                occurrence = next(row for row in chord.occurrences if row.face.id == face.id)
                direction = (0,1) if use.forward else (1,0)
                endpoints = []
                edge = prepared.edges[use.edge]
                for j in direction:
                    t = chord.parameter_range[j]
                    key = ('vertex',edge.start if t == 0 else edge.end) if t in (0,1) else ('edge',edge.id,t)
                    if key not in node_map:
                        node_map[key] = next_node; next_node += 1
                    endpoints.append((node_map[key],occurrence.uv[j],chord.xyz[j]))
                ring.append((index,use.forward,endpoints))
        centre = tuple(sum(endpoints[0][1][axis] for _,_,endpoints in ring)/len(ring) for axis in (0,1))
        xyz = tuple(float(x) for x in face.surface.evaluate(*centre))
        centre_id = next_node; next_node += 1
        cells = tuple(CurvilinearMaterialCell(prepared.handle('face',face.id),
            (ends[0][0],ends[1][0],centre_id),(ends[0][1],ends[1][1],centre),
            (ends[0][2],ends[1][2],xyz),((0,index,forward),)) for index,forward,ends in ring)
        receipts.append(validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,surface_error=.001))
        all_chords.append(chords)
    transfer = query_prepared_material_source_transfer(source,prepared,tuple(receipts))
    return source,prepared,transfer,tuple(receipts),tuple(all_chords)


def test_genuine_public_split_cell_chord_station_has_certified_nominal_alias_and_exact_ancestry():
    source,prepared,transfer,cells,chords = genuine_station_inputs()
    exact_before = transfer.boundary_station_correspondence
    result = query_prepared_source_station_transport(source,prepared,transfer)
    validate_prepared_source_station_transport_binding(source,prepared,transfer,result)
    nonbinary = [row for row in result.rows if row.source_parameter != Q(row.nominal_parameter)]
    assert nonbinary and transfer.boundary_station_correspondence == exact_before
    maps = {edge.id:(parent,a,b) for edge,parent,a,b in transfer.edge_parameter_maps}
    for row in result.rows:
        parent,a,b = maps[row.prepared_edge.id]
        assert row.source_edge == parent and row.source_parameter == a+(b-a)*row.prepared_parameter
        assert row.nominal_parameter == float(row.source_parameter)
        assert Q(row.combined_error_bound)**2 >= row.combined_error_squared_bound
        assert all(row.combined_error_squared_bound <= Q(limit)**2 for _,_,limit in row.limits)
        assert any(row.prepared_station_key in chord.station_keys and row.xyz == chord.xyz[chord.station_keys.index(row.prepared_station_key)]
                   for bundle in chords for chord in bundle.intervals)


def test_exact_binary_legacy_stations_need_no_alias_rows():
    source,prepared,transfer,_,_ = genuine_station_inputs(split=False,midpoint=False)
    result = query_prepared_source_station_transport(source,prepared,transfer)
    assert result.rows == ()
    validate_prepared_source_station_transport_binding(source,prepared,transfer,result)


def test_alias_receipt_copy_forgery_and_raw_output_edits_refuse():
    source,prepared,transfer,_,_ = genuine_station_inputs()
    result = query_prepared_source_station_transport(source,prepared,transfer)
    for forged in (copy(result),replace(result)):
        with pytest.raises(GeometryError,match='not issued'):
            validate_prepared_source_station_transport_binding(source,prepared,transfer,forged)
    row = result.rows[0]
    object.__setattr__(row,'nominal_parameter',.123)
    with pytest.raises(GeometryError,match='changed'):
        validate_prepared_source_station_transport_binding(source,prepared,transfer,result)


@pytest.mark.parametrize('operand',['source','prepared','transfer','chords'])
def test_alias_callback_raw_mutation_refuses_and_does_not_publish(operand):
    source,prepared,transfer,_,chords = genuine_station_inputs()
    if operand in ('source','prepared'):
        model = source if operand == 'source' else prepared
        target = model.vertices[next(iter(model.vertices))]
        name,original = 'position',target.position
        bad = original.copy(); bad[0] += 1.
    elif operand == 'transfer':
        target,name = transfer,'boundary_station_correspondence'
        original,bad = target.boundary_station_correspondence,()
    else:
        target,name = chords[0],'chord_error'
        original,bad = target.chord_error,.9
    def callback(phase):
        object.__setattr__(target,name,bad)
        return False
    try:
        with pytest.raises(GeometryError):
            query_prepared_source_station_transport(source,prepared,transfer,cancellation_check=callback)
    finally:
        object.__setattr__(target,name,original)


def test_alias_cancelled_or_callback_failure_keeps_geometry_unchanged():
    source,prepared,transfer,_,_ = genuine_station_inputs()
    before = to_dict(prepared)
    signal = CancellationSignal(); signal.cancel()
    with pytest.raises(GeometryError,match='cancelled'):
        query_prepared_source_station_transport(source,prepared,transfer,cancellation_check=signal)
    failure = KeyboardInterrupt('owned callback failure')
    def callback(phase):
        raise failure
    with pytest.raises(KeyboardInterrupt) as caught:
        query_prepared_source_station_transport(source,prepared,transfer,cancellation_check=callback)
    assert caught.value is failure and to_dict(prepared) == before


def test_alias_binding_refuses_raw_stale_source_and_each_output_callback_frontier():
    source,prepared,transfer,_,_ = genuine_station_inputs()
    result = query_prepared_source_station_transport(source,prepared,transfer)
    original = result.rows
    def callback(phase):
        object.__setattr__(result,'rows',() if phase.endswith('binding') else original)
        return False
    try:
        with pytest.raises(GeometryError,match='changed during callback'):
            validate_prepared_source_station_transport_binding(source,prepared,transfer,result,cancellation_check=callback)
    finally:
        object.__setattr__(result,'rows',original)
    vertex = source.vertices[next(iter(source.vertices))]
    original_position = vertex.position
    moved = original_position.copy(); moved[0] += 1.
    try:
        object.__setattr__(vertex,'position',moved)
        with pytest.raises(GeometryError):
            validate_prepared_source_station_transport_binding(source,prepared,transfer,result)
    finally:
        object.__setattr__(vertex,'position',original_position)


def test_unissued_transfer_and_alias_result_have_no_query_or_validation_authority():
    source,prepared,transfer,_,_ = genuine_station_inputs()
    with pytest.raises(GeometryError,match='live owner source transfer'):
        query_prepared_source_station_transport(source,prepared,replace(transfer))
    result = query_prepared_source_station_transport(source,prepared,transfer)
    with pytest.raises(GeometryError,match='other live operands'):
        validate_prepared_source_station_transport_binding(source,prepared,replace(transfer),result)


def test_alias_weak_registry_does_not_retain_its_key():
    import gc
    import weakref
    from anygeometry.source_station_transport import _ISSUED
    source,prepared,transfer,_,_ = genuine_station_inputs()
    result = query_prepared_source_station_transport(source,prepared,transfer)
    reference = weakref.ref(result)
    assert result in _ISSUED
    del result
    gc.collect()
    assert reference() is None


def test_alias_collision_checks_binary_siblings_and_endpoint_identity():
    from anygeometry.source_station_transport import _collision_check
    s = Q(1,2)+Q(1,2**60)
    with pytest.raises(GeometryError,match='collision'):
        _collision_check(((1,Q(1,2)),(1,s)))
    with pytest.raises(GeometryError,match='endpoint'):
        _collision_check(((1,Q(1)-Q(1,2**60)),(1,Q(1))))
    _collision_check(((1,s),(1,s)))


def test_direct_combined_affine_bound_includes_literal_target_error_and_chord_limit():
    from anygeometry import GeometryModel
    from anygeometry.source_station_transport import _combined_bound
    model = GeometryModel()
    a,b = model.add_point(0.,0.,0.),model.add_point(1.,0.,0.)
    edge = model.add_line(a,b)
    s = Q(1,2)+Q(1,2**60)
    target = (.5,2.e-10,0.)
    bound,error,limits = _combined_bound(model,model,edge,edge,s,float(s),target,.0025)
    assert bound >= Q(2.e-10)**2 and bound > (s-Q(float(s)))**2
    assert Q(error)**2 >= bound
    with pytest.raises(GeometryError,match='registered chord error'):
        _combined_bound(model,model,edge,edge,s,float(s),target,1.e-10)


def test_combined_alias_error_preserves_existing_joint_context_on_a_long_carrier():
    from anygeometry import GeometryModel
    from anygeometry.source_station_transport import _combined_bound
    model = GeometryModel()
    length = 3*2**26
    a,b = model.add_point(0.,0.,0.),model.add_point(float(length),0.,0.)
    edge = model.add_line(a,b)
    s = Q(1,3)
    target = (float(2**26),0.,0.)
    alias_error = abs(Q(length)*Q(float(s))-Q(2**26))
    assert alias_error == Q(1,2**28)
    assert Q(model.tolerance.effective_length(1.)) < alias_error < Q(model.tolerance.effective_length(length))
    assert alias_error < Q(1.e-8)
    with pytest.raises(GeometryError,match='existing joint length tolerance'):
        _combined_bound(model,model,edge,edge,s,float(s),target,.0025)


@pytest.mark.parametrize('family',['straight','line_range','nurbs'])
def test_registered_source_station_enclosure_honours_zero_box_work(family):
    from anygeometry import GeometryModel, LineRange
    from anygeometry.nurbs import NurbsCurve
    from anygeometry.source_station_transport import _world_enclosure
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model = GeometryModel()
    a,b = model.add_point(0.,0.,0.),model.add_point(1.,0.,0.)
    if family == 'straight':
        edge = model.add_line(a,b)
    else:
        curve = (LineRange((0.,0.,0.),(1.,0.,0.)) if family == 'line_range'
                 else NurbsCurve(1,(0.,0.,1.,1.),((0.,0.,0.),(1.,0.,0.))))
        edge = model.add_curve(a,b,curve)
    token = CURRENT_WORK.set(Work(0,32))
    try:
        with pytest.raises(GeometryError,match='budget exhausted'):
            _world_enclosure(model,edge,Q(1,3),Q(1,2))
    finally:
        CURRENT_WORK.reset(token)
