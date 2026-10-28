from dataclasses import replace
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.local_approximation_field import (query_local_approximation_field,
    query_local_approximation_limit,bind_local_approximation_field,
    validate_local_approximation_allocation,validate_local_approximation_field)
from anygeometry._local_curvature import LocalApproximationNotCertified
from test_prepared_material_curvilinear import square_cells

def fixture():
    source,_,_,_=square_cells()
    face=next(iter(source.faces))
    field=query_local_approximation_field(source,(face,),relative_accuracy=.001,scale_error=.001,policy_floor=1e-8)
    prepared=owner.clone_prepared_geometry(source,new_preparation_epoch=True)
    regions=owner.query_material_surface_regions(prepared,(face,))
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    return source,prepared,face,field,regions,allocation

def test_real_local_chord_cell_reuse_and_source_publication():
    source,prepared,face,field,regions,allocation=fixture()
    _,_,_,template=square_cells()
    cells=tuple(replace(cell,face=prepared.handle('face',face)) for cell in template)
    edges=tuple(use.edge for use in prepared.faces[face].loop)
    chords=owner.query_prepared_material_boundary_chords(prepared,regions,
        tuple((prepared.handle('edge',edge),0.,1.) for edge in edges),chord_error=.01,local_approximation=allocation)
    receipt=owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,
        surface_error=.01,local_approximation=allocation)
    assert receipt.local_field_fingerprint==allocation.fingerprint==chords.local_field_fingerprint
    assert receipt.surface_errors_by_cell==(Q(.001),)
    reused=owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,
        surface_error=.01,local_approximation=allocation,previous=receipt)
    assert reused.surface_errors_by_cell==receipt.surface_errors_by_cell
    transfer=owner.query_prepared_material_source_transfer(source,prepared,(reused,))
    assert transfer.local_field_fingerprints==(allocation.fingerprint,)
    assert all(row[2]<=.001 for row in transfer.surface_errors)
    with pytest.raises(owner.GeometryError,match='local'):
        owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,surface_error=.01)
    object.__setattr__(reused,'surface_errors_by_cell',(Q(.01),))
    with pytest.raises(owner.GeometryError):owner.validate_prepared_material_curvilinear_cells_binding(prepared,regions,reused)

def test_handle_queries_forgery_and_final_late_field_mutation(monkeypatch):
    source,prepared,face,field,regions,allocation=fixture()
    assert query_local_approximation_limit(source,field,source.handle('face',face))==Q(.001)
    with pytest.raises(owner.GeometryError):validate_local_approximation_field(source,replace(field))
    import anygeometry.local_field_transport as transport
    actual=transport.validate_local_field_transport
    def late(*args,**kwargs):
        result=actual(*args,**kwargs)
        if kwargs.get('cancellation_check') is not None:
            object.__setattr__(field,'scale_error',Q(1))
        return result
    monkeypatch.setattr(transport,'validate_local_field_transport',late)
    with pytest.raises(owner.GeometryError,match='changed'):
        validate_local_approximation_allocation(prepared,allocation,cancellation_check=lambda phase:False)

def test_cancellation_and_source_mutation_refuse():
    source,prepared,face,field,regions,allocation=fixture()
    with pytest.raises(owner.GeometryError):validate_local_approximation_allocation(prepared,allocation,cancellation_check=lambda phase:True)
    source.add_point(4,4,4)
    with pytest.raises(owner.GeometryError):validate_local_approximation_allocation(prepared,allocation)

def test_straight_corner_stars_keep_remote_hole_and_gap_competitors():
    from anygeometry._local_straight_clearance import clearance_of_straight_loops
    def loop(x,y,size):
        points=((Q(x),Q(y),Q(0)),(Q(x+size),Q(y),Q(0)),(Q(x+size),Q(y+size),Q(0)),(Q(x),Q(y+size),Q(0)))
        return tuple((points[i],points[(i+1)%4]) for i in range(4))
    wide=clearance_of_straight_loops((loop(0,0,1),))
    narrow=clearance_of_straight_loops((loop(0,0,1),loop(Q(1001,1000),0,1)))
    assert 0<narrow<=Q(1,1000)<wide
    with pytest.raises(LocalApproximationNotCertified):clearance_of_straight_loops((loop(0,0,1),loop(0,0,1)))

@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf'),'0.1'])
@pytest.mark.parametrize('name',['relative_accuracy','scale_error','policy_floor'])
def test_owner_policy_inputs_are_independently_typed(name,value):
    source,_,_,_=square_cells();face=next(iter(source.faces))
    values=dict(relative_accuracy=.001,scale_error=.001,policy_floor=1e-8);values[name]=value
    with pytest.raises(LocalApproximationNotCertified):query_local_approximation_field(source,(face,),**values)
