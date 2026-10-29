"""Spatial policy queries and real owner proof integration; no sample admission."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.local_approximation_field import (query_local_approximation_field,
    bind_local_approximation_field, prepare_local_cell_limit_query)
from test_local_approximation_field import fixture
from test_prepared_material_curvilinear import square_cells


def curved_source():
    model=owner.GeometryModel()
    support=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,1.))))
    corners=((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    curves=tuple(CurveOnSurface(support,NurbsCurve(1,(0.,0.,1.,1.),
        (corners[i],corners[(i+1)%4]))) for i in range(4))
    vertices=tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges=tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve) for i,curve in enumerate(curves))
    face=model.add_face(edges,surface=support);model.add_sheet((face,))
    return model,face


def test_certified_curved_field_has_distinct_limits_and_crossing_infimum():
    source,face=curved_source()
    field=query_local_approximation_field(source,(face,),relative_accuracy=1e-5,scale_error=.1,policy_floor=1e-12)
    tiles=field.face_tiles[0][2]
    assert len(tiles)>=16 and len({row[1] for row in tiles})>1
    prepared=owner.clone_prepared_geometry(source,new_preparation_epoch=True)
    regions=owner.query_material_surface_regions(prepared,(face,))
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    query=prepare_local_cell_limit_query(prepared,allocation)
    # For this authored tensor patch both owner maps are identity.
    seen=[]
    for box,limit,_ in tiles:
        inner=tuple((a+(b-a)/4,b-(b-a)/4) for a,b in box)
        actual,record=query(face,(inner,))
        assert actual==limit and record[0]=='intersected_tiles'
        seen.append(actual)
    assert len(set(seen))>1
    value,record=query(face,(((Q(0),Q(1)),(Q(0),Q(1))),))
    assert value==min(seen) and record[0]=='intersected_tiles'
    value,record=query(face,(((Q(-1),Q(2)),(Q(0),Q(1))),))
    assert value==min(seen) and record[0]=='whole_face_outside_enclosure'


def test_real_cell_receipt_binds_complete_fan_footprints_and_reuse():
    _,prepared,face,_,regions,allocation=fixture()
    _,_,_,template=square_cells()
    cells=tuple(replace(cell,face=prepared.handle('face',face)) for cell in template)
    edges=tuple(use.edge for use in prepared.faces[face].loop)
    chords=owner.query_prepared_material_boundary_chords(prepared,regions,
        tuple((prepared.handle('edge',edge),0.,1.) for edge in edges),chord_error=.01,local_approximation=allocation)
    receipt=owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,
        surface_error=.01,local_approximation=allocation)
    assert len(receipt.local_footprints_by_cell)==len(cells)
    assert all(len(record[1])==len(cell.node_ids) for record,cell in zip(receipt.local_footprints_by_cell,cells))
    reused=owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,
        surface_error=.01,local_approximation=allocation,previous=receipt)
    assert reused.local_footprints_by_cell==receipt.local_footprints_by_cell
    object.__setattr__(reused,'local_footprints_by_cell',())
    with pytest.raises(owner.GeometryError):
        owner.validate_prepared_material_curvilinear_cells_binding(prepared,regions,reused)


def test_spatial_failure_budget_is_issued_for_exact_operands(monkeypatch):
    import anygeometry.source_cell_complex as complex
    import anygeometry.prepared_material_curvilinear as receipts
    source,prepared,face,transport,regions,allocation=fixture()
    _,_,_,template=square_cells()
    cells=tuple(replace(cell,face=prepared.handle('face',face)) for cell in template)
    chords=owner.query_prepared_material_boundary_chords(prepared,regions,
        tuple((prepared.handle('edge',use.edge),0.,1.) for use in prepared.faces[face].loop),
        chord_error=.01,local_approximation=allocation)
    failure=owner.SourceCellRefinementRequired(face,cells[0].node_ids,Q(1,10000),Q(1,1000000),requested_error=Q(.01))
    def unresolved(*args,**kwargs):
        kwargs['local_allocation_records'][(face,cells[0].node_ids)]=(Q(1,1000),())
        raise failure
    monkeypatch.setattr(complex,'certify_source_cells',unresolved)
    with pytest.raises(owner.SourceCellRefinementRequired):
        owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,
            surface_error=.01,local_approximation=allocation)
    assert receipts._local_resolution_failure_limit(prepared,allocation,failure,cells,chords)==Q(1,1000)
    failure.error_limit=Q(1,10)
    with pytest.raises(owner.GeometryError,match='allocation'):
        receipts._local_resolution_failure_limit(prepared,allocation,failure,cells,chords)


def test_detached_numerical_view_composes_nonidentity_child_map(monkeypatch):
    # Isolated affine-map arithmetic test, deliberately not an authority test.
    import anygeometry.local_approximation_field as module
    _,prepared,face,field,_,allocation=fixture()
    stored=module._ALLOCATIONS[allocation]
    identity=((Q(1),Q(0),Q(0)),(Q(0),Q(1),Q(0)))
    swapped=((Q(0),Q(1),Q(0)),(Q(-1),Q(0),Q(1)))
    synthetic=replace(field,face_tiles=tuple((f,identity,rows) for f,m,rows in field.face_tiles))
    transport=replace(stored[3],footprints=tuple(replace(row,source_from_child=swapped) for row in stored[3].footprints))
    monkeypatch.setitem(module._ALLOCATIONS,allocation,(*stored[:2],synthetic,transport,stored[4]))
    monkeypatch.setattr(module,'validate_local_approximation_allocation',lambda *a,**k:None)
    query=module.prepare_local_cell_limit_query(prepared,allocation)
    _,record=query(face,(((Q(1,8),Q(1,4)),(Q(1,2),Q(3,4))),))
    assert record==('intersected_tiles',(((Q(1,2),Q(3,4)),(Q(3,4),Q(7,8))),))
