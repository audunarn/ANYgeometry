"""Original definitions are prospective immutable evidence, never coverage."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from anygeometry import (EntityRef, GeometryError, GeometryModel, Plane, apply_intersections,
    clone_prepared_geometry, from_dict, plan_intersections, query_prepared_face_preimages,
    set_prepared_face_corners, to_dict, validate_prepared_face_preimages_binding)
from anygeometry.entities import OrientedEdge
from anygeometry.prepared_face_preimages import (
    _binding_checksum, _capture_application_preimages,
    query_prepared_authored_face_definition as original_definition,
)


def authored(with_cutter=False):
    model=GeometryModel()
    points=model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0)))
    controls=model.add_points(((1,0,0),(3,0,0)))
    spline=model.add_spline(points[0],controls,points[1])
    edges=[spline,*(model.add_line(points[i],points[(i+1)%4]) for i in range(1,4))]
    face=model.add_face_from_loop(tuple(OrientedEdge(edge,True) for edge in edges),
        (0,1,2,3),surface=Plane((0,0,0),(1,0,0),(0,1,0)))
    holes=[]
    for x in (.5,1.5):
        vertices=model.add_points(((x,.5,0),(x,1,0),(x+.25,1,0),(x+.25,.5,0)))
        hole=model.add_polyline(vertices,close=True)
        holes.append(tuple(OrientedEdge(edge,True) for edge in hole))
    with model.transaction():
        model._put_entity('face',replace(model.faces[face],holes=tuple(holes)))
    model.set_face_metadata(face,{'material':'original','nested':{'value':[1,2]}})
    sheet=model.add_sheet((face,),name='authored sheet')
    use=model.sheets[sheet].face_use_ids[0]
    coedge=model.face_uses[use].loops[0][0]
    with model.transaction():
        model._put_structural('face_use',replace(model.face_uses[use],metadata={'role':'original use'}))
        model._put_structural('coedge',replace(model.coedges[coedge],metadata={'role':'original coedge'}))
    references=(EntityRef('face',face),EntityRef('edge',spline),EntityRef('vertex',controls[0]))
    model.add_to_group('source membership',references)
    for reference in references:model.tag(reference,'source-tag')
    if with_cutter:
        model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    return model,face,spline,controls


def prepare(model):
    plan=plan_intersections(model,tuple(model.faces),policy='connect')
    return apply_intersections(model,plan,policy='connect')


def assert_snapshot(record, document, face):
    payload=json.loads(record.definition_json)
    source_face=next(item for item in document['faces'] if item['id']==face)
    assert payload['face']==source_face
    edge_ids={edge for loop in (source_face['loop'],*source_face['holes']) for edge,_ in loop}
    selected_edges=[item for item in document['edges'] if item['id'] in edge_ids]
    assert payload['edges']==selected_edges
    vertex_ids={vertex for edge in selected_edges for vertex in
                (edge['start'],*edge['curve'].get('control_vertices',()),edge['end'])}
    assert payload['vertices']==[item for item in document['vertices'] if item['id'] in vertex_ids]
    assert len(payload['face']['holes'])==2
    assert record.source_checksum==document['checksum']['value']
    assert record.revision==document['revision']
    assert payload['occurrences']['face_uses']
    assert payload['occurrences']['coedges']
    assert payload['occurrences']['sheets'] and payload['occurrences']['parts']
    assert payload['groups']['source membership']
    assert any(item['entity'][0]=='face' for item in payload['tags'])
    assert payload['occurrences']['face_uses'][0]['metadata']=={'role':'original use'}
    assert payload['occurrences']['coedges'][0]['metadata']=={'role':'original coedge'}
    return payload


def test_capture_uses_one_source_snapshot_and_is_immutable(monkeypatch):
    import anygeometry.prepared_face_preimages as module
    model,face,_,controls=authored(); source=to_dict(model); calls=[]
    original=module.to_dict
    def counted(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(module,'to_dict',counted)
    binding=_capture_application_preimages(model,allow_seed=True)
    assert len(calls)==1
    record=binding.authored_face_definitions[0]
    payload=assert_snapshot(record,source,face)
    assert set(controls)<={item['id'] for item in payload['vertices']}
    payload['face']['metadata']['nested']['value'][0]=999
    assert json.loads(record.definition_json)['face']['metadata']['nested']['value']==[1,2]
    with pytest.raises(FrozenInstanceError):record.definition_json='{}'
    assert to_dict(model)==source


def test_original_holes_dependencies_occurrences_survive_fragmentation_and_copy():
    model,face,_,_=authored(with_cutter=True); source=to_dict(model)
    prepared_copy=model.clone(preserve_identity=True)
    prepare(prepared_copy)
    assert to_dict(model)==source
    binding=query_prepared_face_preimages(prepared_copy)
    assert len(dict(binding.face_descendants)[face])>1
    record=original_definition(prepared_copy,face)
    assert_snapshot(record,source,face)
    snapshot=to_dict(prepared_copy)
    clone=clone_prepared_geometry(prepared_copy)
    assert original_definition(clone,face)==record
    assert to_dict(prepared_copy)==snapshot
    current=next(identifier for identifier, face in prepared_copy.faces.items() if len(face.loop)==4)
    set_prepared_face_corners(prepared_copy,{current:(0,1,2,3)})
    assert original_definition(prepared_copy,face)==record


def test_definition_tamper_and_stale_source_refuse():
    model,face,_,_=authored();prepare(model)
    binding=query_prepared_face_preimages(model)
    record=original_definition(model,face)
    tampered=replace(binding,authored_face_definitions=(replace(record,definition_json='{}'),))
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate_prepared_face_preimages_binding(model,tampered)
    model.add_point(20,20,20)
    with pytest.raises(GeometryError):original_definition(model,face)
    with pytest.raises(GeometryError,match='stale'):
        original_definition(model,face,expected_revision=binding.revision)


def test_legacy_empty_records_keep_id_ancestry_but_definition_lookup_refuses():
    model,face,_,_=authored();prepare(model)
    binding=replace(query_prepared_face_preimages(model),authored_face_definitions=())
    model._prepared_face_preimages_receipt=(binding,_binding_checksum(binding))
    assert query_prepared_face_preimages(model)==binding
    validate_prepared_face_preimages_binding(model,binding)
    with pytest.raises(GeometryError,match='original authored face definition is unavailable'):
        original_definition(model,face)
    assert _capture_application_preimages(model,allow_seed=True).authored_face_definitions==()


def test_load_and_ordinary_clone_cannot_reconstruct_old_original_definitions():
    model,face,_,_=authored(with_cutter=True);prepare(model)
    for copy in (model.clone(preserve_identity=True),from_dict(to_dict(model))):
        with pytest.raises(GeometryError):original_definition(copy,face)
        assert _capture_application_preimages(copy,allow_seed=True) is None


def test_authorized_unfragmented_loaded_source_seeds_new_current_definitions():
    model,face,_,_=authored();loaded=from_dict(to_dict(model));source=to_dict(loaded)
    prepare(loaded)
    assert_snapshot(original_definition(loaded,face),source,face)
    with pytest.raises(GeometryError,match='positive authored face ID'):original_definition(loaded,True)
    with pytest.raises(GeometryError,match='not in the original source'):original_definition(loaded,999)
