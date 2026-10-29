from dataclasses import replace
import pytest
from anygeometry import GeometryError, clone_prepared_geometry, to_dict, from_dict
from anygeometry._local_adaptation_lineage import query_local_adaptation_lineage, validate_local_adaptation_lineage
from test_prepared_material_curvilinear import square_cells

def test_owner_clone_has_preproof_identity_chart_transport():
    source,_,_,_=square_cells()
    prepared=clone_prepared_geometry(source,new_preparation_epoch=True)
    faces=tuple(prepared.faces)
    result=query_local_adaptation_lineage(source,prepared,faces)
    assert tuple(r[0] for r in result.face_maps)==faces
    assert all(row[0]==row[1] for row in result.face_maps)
    validate_local_adaptation_lineage(source,prepared,result)
    with pytest.raises(GeometryError,match='not issued'):
        validate_local_adaptation_lineage(source,prepared,replace(result))
    loaded=from_dict(to_dict(prepared))
    with pytest.raises(GeometryError,match='owner-issued'):
        query_local_adaptation_lineage(source,loaded,faces)

def test_stale_and_callback_mutation_refuse():
    source,_,_,_=square_cells();prepared=clone_prepared_geometry(source,new_preparation_epoch=True)
    faces=tuple(prepared.faces)
    result=query_local_adaptation_lineage(source,prepared,faces)
    source.add_point(10,10,10)
    with pytest.raises(GeometryError): validate_local_adaptation_lineage(source,prepared,result)
    source,_,_,_=square_cells();prepared=clone_prepared_geometry(source,new_preparation_epoch=True)
    def callback(phase):
        source.add_point(10,10,10)
        return False
    with pytest.raises(GeometryError):query_local_adaptation_lineage(source,prepared,tuple(prepared.faces),cancellation_check=callback)

def test_callback_receipt_mutation_is_checked_at_final_gate():
    from uuid import uuid4
    source,_,_,_=square_cells();prepared=clone_prepared_geometry(source,new_preparation_epoch=True)
    result=query_local_adaptation_lineage(source,prepared,tuple(prepared.faces))
    def callback(phase):
        object.__setattr__(result,'source_model_id',uuid4())
        return False
    with pytest.raises(GeometryError,match='receipt changed'):
        validate_local_adaptation_lineage(source,prepared,result,cancellation_check=callback)
