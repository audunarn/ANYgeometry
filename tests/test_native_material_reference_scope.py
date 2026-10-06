"""Whole positive material and geometry-reference receipts through public callers."""
from dataclasses import replace
from fractions import Fraction as F
import json
import os
from pathlib import Path
import sys
from importlib.metadata import version

import numpy as np
import pytest

import anygeometry
from anygeometry import (GeometryModel, GeometryError, apply_intersections,
    plan_intersections, to_dict, query_prepared_native_material_reference_scope as query,
    validate_prepared_native_material_reference_scope_binding as validate)
from anygeometry.native_arc_parameter_maps import NativeArcParameterMapPolicy
from anygeometry.native_material_reference_scope import _positive_census, _issued
from anygeometry.definition_binding import definition_checksum
from anygeometry.prepared_face_preimages import _binding_checksum
from test_authored_domain_coverage import planar
from test_edge_subcurve_preimages_batch import crossing, prepare
from test_cylinder_boundary_correspondence import panel


def test_source_runtime():
    origin=Path(anygeometry.__file__).resolve()
    assert origin==Path(__file__).resolve().parents[1]/'src/anygeometry/__init__.py'
    assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    if os.environ.get('NATIVE_SCOPE_RUNTIME_EVIDENCE'):
        Path(os.environ['NATIVE_SCOPE_RUNTIME_EVIDENCE']).write_text(json.dumps(dict(
            executable=sys.executable,python_version=sys.version,source_origin=str(origin),
            numpy_version=version('numpy'),pytest_version=version('pytest'),
            threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2))


def test_production_plane_partition_positive_indicator_and_outside_references():
    model=crossing()
    roots=tuple(model.faces)
    prepare(model)
    before=to_dict(model)
    full=query(model,list(roots))
    assert full.document_material_qualified
    assert all(r.classification=='exact_document_material' for r in full.material_rows)
    assert full.geometry_native_reference_maps_qualified,[
        (r.get('kind'),r.get('id'),r.get('disposition'),r.get('geometry_semantic_disposition'))
        for field in ('record_dispositions','part_sheet_dispositions','current_extra_face_use_dispositions','coedge_dispositions')
        for r in full.inventory[field]
        if r.get('disposition','').startswith('refused') or r.get('geometry_semantic_disposition','').startswith('refused')]
    partial=query(model,[roots[0]])
    assert partial.outside_root_ids==(roots[1],)
    assert partial.inventory['traces']
    assert any(t['interior_constraint'] for t in partial.inventory['traces'])
    validate(model,full)
    assert to_dict(model)==before
    assert not full.meshing_permitted and not full.external_reference_scope_qualified
    assert not full.floating_evaluation_preservation_qualified


@pytest.mark.parametrize('kind',('concave','holes'))
def test_all_loops_participate_in_plane_material(kind):
    model,binding=planar(concave=kind=='concave',holes=kind=='holes',fragment=False)
    receipt=query(model,[binding.authored_definition.face_id])
    assert receipt.document_material_qualified
    evidence=json.loads(receipt.material_rows[0].domain_evidence_json)
    assert len(evidence['all_loops'][0])==(3 if kind=='holes' else 1)
    validate(model,receipt)


def test_duplicate_opposite_oriented_interior_cells_do_not_cancel_material():
    outer=tuple(map(lambda p:tuple(map(F,p)),((0,0),(4,0),(4,4),(0,4))))
    interior=tuple(map(lambda p:tuple(map(F,p)),((1,1),(2,1),(2,2),(1,2))))
    with pytest.raises(GeometryError,match='positive material indicator'):
        _positive_census(((outer,),(outer,),(interior,),(interior[::-1],)),lambda:None)
    # Exact positive partition independent of either stored loop traversal.
    left=tuple((F(x),F(y)) for x,y in ((0,0),(2,0),(2,4),(0,4)))
    right=tuple((F(x),F(y)) for x,y in ((2,0),(4,0),(4,4),(2,4)))
    assert _positive_census(((outer[::-1],),(left,),(right[::-1],)),lambda:None)>0


def test_actual_duplicate_descendant_inventory_is_rejected():
    model=GeometryModel()
    root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    first=model.add_plate(model.add_points(((1,1,0),(2,1,0),(2,2,0),(1,2,0))))
    second=model.add_plate(model.add_points(((1,2,0),(2,2,0),(2,1,0),(1,1,0))))
    # The normal preparation already refuses positive overlap. Independently
    # test the receipt core against the actual opposite-oriented document loops.
    from anygeometry.native_material_reference_scope import _material,_compile_document
    from anygeometry.native_support_snapshots import capture_native_supports
    from anygeometry.cylinder_charts import _Proof
    document=to_dict(model);index=_compile_document(document)
    supports={key:(kind,data) for key,kind,data in capture_native_supports(model)}
    row=_material(root,(root,first,second),index,index,supports,supports,_Proof(NativeArcParameterMapPolicy(),None))
    assert not row.document_material_qualified
    assert 'positive material indicator' in row.refusal


def test_cylinder_exact_document_chart_material_is_separate_from_native_evaluation():
    model,root=panel(0.)
    receipt=query(model,[root])
    assert receipt.document_material_qualified
    row=receipt.material_rows[0]
    evidence=json.loads(row.domain_evidence_json)
    assert row.family=='cylinder' and evidence['native_projection']
    assert evidence['rank_coordinates_are_area_units'] is False
    assert not receipt.floating_evaluation_preservation_qualified
    validate(model,receipt)


def test_public_cylinder_split_refits_can_preserve_exact_chart_material():
    from anygeometry.generators.structural import cylinder
    model=cylinder(1.,1.,circumferential_segments=3)
    roots=tuple(model.faces)
    model.add_plate(model.add_points(((2.,.5,-.5),(2.,.5,1.5),(-2.,.5,1.5),(-2.,.5,-.5))))
    prepare(model)
    receipt=query(model,[roots[0]])
    assert len(receipt.material_rows[0].current_face_ids)>1
    assert receipt.document_material_qualified,receipt.material_rows[0].refusal
    assert receipt.native_arc_maps
    assert any(r.classification=='bounded' for r in receipt.native_arc_maps)
    assert not receipt.geometry_native_reference_maps_qualified or not receipt.external_reference_scope_qualified
    validate(model,receipt)


def test_actual_raw_native_support_capture_and_same_revision_mutation():
    model,root=panel(0.)
    receipt=query(model,[root])
    before=to_dict(model)
    support=model.faces[root].surface
    raw=support._circumferential.copy()
    object.__setattr__(support,'_circumferential',raw+np.asarray((0.,1e-12,0.)))
    assert to_dict(model)==before # stored derived coefficient is nonserialized
    with pytest.raises(GeometryError,match='native support'):
        validate(model,receipt)
    with pytest.raises(GeometryError,match='native support'):
        query(model,[root])
    object.__setattr__(support,'_circumferential',raw)
    validate(model,receipt)


def test_typed_members_maps_isolated_and_opaque_external_obligations():
    model=GeometryModel()
    root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    edge=model.faces[root].loop[0].edge
    member=model.add_member((edge,),name='edge beam')
    isolated=model.add_point(20.,30.,40.)
    model._serialization_extensions={'external:test':{'vertex':isolated}}
    prepare(model)
    receipt=query(model,[root])
    inventory=receipt.inventory
    assert inventory['member_native_maps']
    assert inventory['member_native_maps'][0]['disposition']=='exact_native_map'
    assert inventory['source_use_coverage']
    assert any(r['vertex_id']==isolated for r in inventory['isolated_vertex_dispositions'])
    assert inventory['original']['opaque_unqualified']['extensions']
    assert receipt.geometry_native_reference_maps_qualified
    assert not receipt.opaque_semantics_qualified
    assert 'consumer' in inventory['external_reference_obligation']
    validate(model,receipt)


def test_full_typed_incoming_attachments_and_junctions_have_explicit_disposition():
    from test_prepared_model_scope import fixture,prepare as original_prepare
    model,(roots,sheet,member,attachments,isolated)=fixture()
    original_prepare(model)
    receipt=query(model,[roots[0]])
    inventory=receipt.inventory
    assert {r['id'] for r in inventory['original']['records']['attachments']}==set(attachments)
    assert any(r['kind']=='attachments' and not r['semantic_mapping_qualified']
               for r in inventory['record_dispositions'])
    assert inventory['all_face_use_occurrences']['original']
    assert not receipt.geometry_native_reference_maps_qualified
    assert not receipt.meshing_permitted


@pytest.mark.parametrize('validator',(False,True))
def test_atomic_cancellation_forgery_and_mutate_restore(validator):
    model,root=panel(0.)
    receipt=query(model,[root])
    action=(lambda cb:validate(model,receipt,cancellation_check=cb)) if validator else (
        lambda cb:query(model,[root],cancellation_check=cb))
    with pytest.raises(GeometryError,match='cancelled'):
        action(lambda _:True)
    marker=GeometryError('caller marker')
    def stop(_):
        raise marker
    with pytest.raises(GeometryError) as caught:
        action(stop)
    assert caught.value is marker
    support=model.faces[root].surface
    raw=support._circumferential.copy()
    def restore(_):
        object.__setattr__(support,'_circumferential',np.ones(3))
        object.__setattr__(support,'_circumferential',raw)
        return False
    action(restore)
    with pytest.raises(GeometryError,match='not issued'):
        validate(model,replace(receipt))


def test_aggregate_budget_has_no_accepted_partial_and_legacy_capture_refuses():
    model,root=panel(0.)
    with pytest.raises(GeometryError,match='budget'):
        query(model,[root],policy=NativeArcParameterMapPolicy(1))
    assert not _issued.get(model)
    before=model._prepared_face_preimages_receipt[0]
    legacy=replace(before,authored_native_supports=None,current_native_supports=None)
    model._prepared_face_preimages_receipt=(legacy,_binding_checksum(legacy))
    with pytest.raises(GeometryError,match='prospective'):
        query(model,[root])


def test_document_fingerprint_count_is_constant_per_batch(monkeypatch):
    import anygeometry.serialization as serialization
    before=serialization._checksum
    counts=[]
    def counted(*args,**kwargs):
        counts.append(1)
        return before(*args,**kwargs)
    monkeypatch.setattr(serialization,'_checksum',counted)
    model=crossing()
    roots=list(model.faces)
    prepare(model)
    counts.clear();query(model,[roots[0]]);single=len(counts)
    counts.clear();query(model,roots)
    assert len(counts)==single


def test_principal_seam_is_handled_by_relative_exact_ray_cut():
    import math
    model,root=panel(math.pi)
    receipt=query(model,[root])
    assert receipt.document_material_qualified,receipt.material_rows[0].refusal
    evidence=json.loads(receipt.material_rows[0].domain_evidence_json)
    assert evidence['relative_ray_anchor']
    validate(model,receipt)


def test_bbox_candidates_match_exhaustive_small_segment_oracle():
    from anygeometry.native_material_reference_scope import _segment_candidates
    segments=tuple(((F(i%3),F(i//3)),(F((i+2)%3),F((i+3)//3))) for i in range(8))
    actual={frozenset(pair) for pair in _segment_candidates(segments,lambda:None)}
    expected=set()
    for i,(a,b) in enumerate(segments):
        for j,(c,d) in enumerate(segments[i+1:],i+1):
            if all(max(min(a[k],b[k]),min(c[k],d[k]))<=min(max(a[k],b[k]),max(c[k],d[k])) for k in (0,1)):
                expected.add(frozenset((i,j)))
    assert actual==expected


def _inventory_inputs(model,receipt):
    from copy import deepcopy
    p=receipt.scope.face_preimages
    return [receipt.scope.authored_document,receipt.scope.current_document,p.face_descendants,
        set(receipt.selected_root_ids),deepcopy(receipt.inventory['native_edge_maps']),lambda:None,
        {key:(kind,data) for key,kind,data in p.authored_native_supports},
        {key:(kind,data) for key,kind,data in p.current_native_supports},
        {(root,child):True for root,children in p.face_descendants for child in children},
        {child for row in receipt.material_rows if row.document_material_qualified for child in row.current_face_ids}]


@pytest.mark.parametrize('adversary',('missing','extra','wrong_sheet'))
def test_generated_joint_requires_exact_complete_owner_collection(adversary):
    from copy import deepcopy
    from anygeometry.authored_constraint_scope import _complete_native_reference_inventory as census
    model=crossing();roots=tuple(model.faces);prepare(model)
    receipt=query(model,roots);args=_inventory_inputs(model,receipt)
    records=args[1]['structural']
    if adversary=='missing':
        records['junctions'][0]['attachment_ids'].pop()
    elif adversary=='extra':
        extra=deepcopy(records['attachments'][0]);extra['id']=10000
        records['attachments'].append(extra)
    else:
        records['attachments'][0]['source_id']=10000
    inventory,_,qualified=census(*args)
    assert not qualified
    assert any(r['kind']=='junctions' and not r['semantic_mapping_qualified'] for r in inventory['record_dispositions'])


def _owned_plane(prepared=True):
    model=GeometryModel()
    root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    model.add_sheet((root,),name='owner')
    if prepared:
        prepare(model)
    return model,root


@pytest.mark.parametrize('adversary',('refused_map','removed_coedge','reverse_occurrence','reassign_owner','mirror_native'))
def test_reference_census_checks_maps_coverage_owner_and_native_orientation(adversary):
    from anygeometry.authored_constraint_scope import _complete_native_reference_inventory as census
    model,root=_owned_plane();receipt=query(model,[root]);args=_inventory_inputs(model,receipt)
    assert receipt.geometry_native_reference_maps_qualified
    if adversary=='refused_map':
        for mapping in args[4]:
            mapping['classification']='refused'
    elif adversary=='removed_coedge':
        args[1]['structural']['coedges'].pop()
    elif adversary=='reverse_occurrence':
        use=args[1]['structural']['face_uses'][0]
        use['orientation']='reversed' if use['orientation']=='forward' else 'forward'
    elif adversary=='reassign_owner':
        args[1]['structural']['sheets'][0]['name']='new owner'
    else:
        args[8][root,root]=False
    _,_,qualified=census(*args)
    assert not qualified


def test_nonserialized_source_support_is_required_for_literal_face_reference():
    from anygeometry.authored_constraint_scope import _complete_native_reference_inventory as census
    model,root=_owned_plane(prepared=False)
    # Actual public owner identity includes a face orientation reference.
    edge=model.faces[root].loop[0].edge
    model.add_member((edge,),orientation_reference=('face',root))
    prepare(model);receipt=query(model,[root]);args=_inventory_inputs(model,receipt)
    assert census(*args)[2]
    kind,data=args[7][root]
    args[7][root]=(kind,(data[0],tuple(-x for x in data[1]),data[2]))
    assert not census(*args)[2]
    args[6][root]=args[7][root]=('unsupported',())
    assert not census(*args)[2]


def test_dependency_closure_includes_shared_via_vertex_outside_face():
    from anygeometry.authored_constraint_scope import _complete_native_reference_inventory as census
    model=crossing();roots=tuple(model.faces);prepare(model);receipt=query(model,roots)
    args=_inventory_inputs(model,receipt);args[3]={roots[0]}
    source_faces=dict(args[2]);selected_face=source_faces[roots[0]][0];outside_face=source_faces[roots[1]][0]
    faces={f['id']:f for f in args[1]['faces']};edges={e['id']:e for e in args[1]['edges']}
    first=edges[faces[selected_face]['loop'][0][0]];second=edges[faces[outside_face]['loop'][0][0]]
    first['curve']['via_vertex']=99999;second['curve']['control_vertices']=[99999]
    inventory,outside,_=census(*args)
    assert roots[1] in outside
    assert 99999 in inventory['selected_dependency_vertex_ids']
    assert ('vertex',99999) in inventory['dependency_closure']


def test_harmonic_density_bounds_do_not_claim_stored_dot_projection_density():
    from anygeometry.native_material_reference_scope import _cylinder_frame
    from anygeometry.cylinder_charts import _Proof
    support=((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),(0.,2.,0.),1.,1.,0.,1.)
    frame=_cylinder_frame(support,_Proof(NativeArcParameterMapPolicy(),None))
    # dA/dphi at alpha=0 equals 2/4=.5; harmonic x=R*alpha lower is sqrt(4/5).
    assert frame[5][0]>F(1,2)
    model,root=panel(0.);evidence=json.loads(query(model,[root]).material_rows[0].domain_evidence_json)
    assert 'not stored-dot phi' in evidence['harmonic_density_coordinates']
    assert 'density_bounds' not in evidence


def test_unchanged_standard_member_attachment_preserves_source_owner():
    from anygeometry import ParameterRange
    model,root=_owned_plane(prepared=False)
    edge=model.faces[root].loop[0].edge
    member=model.add_member((edge,))
    model.add_attachment(member,'member_on_face','face',root,ParameterRange(0.,1.),
                         (ParameterRange(0.,1.),ParameterRange.point(0.)),evidence='exact',tolerance_used=1e-9)
    prepare(model);receipt=query(model,[root])
    assert receipt.geometry_native_reference_maps_qualified,receipt.inventory['record_dispositions']
    attachments=[r for r in receipt.inventory['record_dispositions'] if r['kind']=='attachments']
    assert attachments and all(r['semantic_mapping_qualified'] for r in attachments)


def test_authored_sheet_joint_identity_requires_complete_oriented_sheet_maps():
    from anygeometry.joint_edges import declare_joint_edge
    model=GeometryModel()
    points=model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0),(1,0,1),(0,0,1)))
    first=model.add_plate(points[:4]);second=model.add_plate((points[1],points[0],points[5],points[4]))
    model.add_sheet((first,),name='first');model.add_sheet((second,),name='second')
    edge=model.faces[first].loop[0].edge
    from anygeometry.entities import OrientedEdge
    with model.transaction():
        face=model.faces[second]
        model._put_entity('face',replace(face,loop=(OrientedEdge(edge,False),*face.loop[1:])))
    declare_joint_edge(model,edge,1e-9)
    assert model.junctions
    prepare(model);receipt=query(model,[first,second])
    assert receipt.geometry_native_reference_maps_qualified,receipt.inventory['record_dispositions']
    assert all(r['geometry_semantic_disposition']=='qualified_native_owner_semantics'
        for r in receipt.inventory['record_dispositions'] if r['kind'] in ('attachments','junctions'))
