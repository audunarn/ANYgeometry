"""Small owned planar networks and adversarial receipt evidence."""
import importlib.util
from pathlib import Path
import pytest
from fractions import Fraction
from dataclasses import replace
from anygeometry import (plan_intersections, apply_intersections, to_dict, GeometryError,
    query_prepared_planar_member_sheet_network as query,
    validate_prepared_planar_member_sheet_network_binding as validate)

spec = importlib.util.spec_from_file_location('planar_small_fixtures',
    Path(__file__).parents[1]/'tools/general_intersections/large_connected_fixtures.py')
import sys
builders = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builders
spec.loader.exec_module(builders)

def build(family='connected_hub', count=4, splitter=False):
    fixture = getattr(builders, family)(count)
    if splitter:
        fixture.plate(((0.,5.,0.),(10.,5.,0.),(10.,5.,1.),(0.,5.,1.)),10.)
    model = fixture.model
    plan = plan_intersections(model, fixture.operands, policy='connect')
    apply_intersections(model, plan, policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind == 'sheet_on_joint')
    return model, joint

@pytest.mark.parametrize('family,count,points', [('connected_strip',10,0),
    ('connected_hub',4,4), ('connected_hub',10,25)])
def test_complete_small_network(family,count,points):
    model,joint = build(family,count)
    before = to_dict(model)
    receipt = query(model,joint)
    assert len(receipt.relations['point_contacts']) == points
    assert set(receipt.current_face_ids) == set(model.faces)
    assert not receipt.beam_discretization_qualified
    assert not receipt.load_transfer_qualified
    assert not receipt.external_reference_transfer_qualified
    assert not receipt.solver_qualified
    assert not receipt.publication_qualified
    validate(model,receipt)
    assert to_dict(model) == before

def snapshots(model,joint):
    from anygeometry.prepared_sheet_joint_component import _index
    scope = query(model,joint).scope
    return scope,_index(scope.authored_document),_index(scope.current_document)

def qualify(model,scope,source,current):
    from anygeometry.prepared_planar_member_sheet_network import _qualify
    return _qualify(model,scope,source,current,None)

@pytest.mark.parametrize('corruption', ['uv','station','sheet','child','omission','common_vertex',
                                       'junction_sheets','junction_members'])
def test_point_contact_adversaries(corruption):
    model,joint = build('connected_hub',4,splitter=corruption=='child')
    before = to_dict(model)
    scope,source,current = snapshots(model,joint)
    points = [row for row in current['attachments'].values() if row['kind']=='member_through_face']
    row = points[0]
    junction = next(j for j in current['junctions'].values() if row['id'] in j['attachment_ids'])
    if corruption=='uv':
        row['target_parameters'][0] = [100.,100.]
    elif corruption=='station':
        row['member_range'] = [.123,.123]
    elif corruption=='sheet':
        row['sheet_id'] = next(s for s in current['sheets'] if s!=row['sheet_id'])
    elif corruption=='child':
        root = next(root for root,children in scope.face_preimages.face_descendants
                    if row['target_id'] in children)
        row['target_id'] = next(child for old,children in scope.face_preimages.face_descendants
                               if old==root for child in children if child!=row['target_id'])
    elif corruption=='omission':
        del current['attachments'][row['id']]
        del current['junctions'][junction['id']]
    elif corruption=='common_vertex':
        other = next(p for p in points if p['id']!=row['id'] and p['sheet_id']==row['sheet_id'])
        old = next(j for j in current['junctions'].values() if other['id'] in j['attachment_ids'])
        junction['attachment_ids'].append(other['id'])
        junction['member_uses'].extend(old['member_uses'])
        junction['sheet_ids'] = sorted(set(junction['sheet_ids']+old['sheet_ids']))
        junction['kind']='multi_way'
        del current['junctions'][old['id']]
    elif corruption=='junction_sheets':
        junction['sheet_ids']=[]
    else:
        junction['member_uses']=[]
    with pytest.raises(GeometryError):
        qualify(model,scope,source,current)
    assert to_dict(model)==before

def test_collinear_finite_segment_and_loop():
    from anygeometry.prepared_planar_member_sheet_network import _on_segment,_qualified_loops
    point = lambda a,b: (Fraction(a),Fraction(b))
    assert not _on_segment(point(0,5),point(0,0),point(0,1))
    vertices = [point(0,0),point(1,0),point(2,0),point(2,1),point(0,1)]
    loop = tuple(zip(vertices,vertices[1:]+vertices[:1]))
    _qualified_loops((loop,),lambda:None)
    backtrack = [point(0,0),point(2,0),point(1,0),point(2,1),point(0,1)]
    with pytest.raises(GeometryError,match='backtracking'):
        _qualified_loops((tuple(zip(backtrack,backtrack[1:]+backtrack[:1])),),lambda:None)

def test_orphan_face_cannot_certify_material():
    model,joint=build('connected_strip',10)
    scope,source,current=snapshots(model,joint)
    member = next(iter(source['members'].values()))
    carrier = source['member_edge_uses'][member['edge_use_ids'][0]]['edge_id']
    from anygeometry.prepared_planar_member_sheet_network import _planar_face_charts,_material_membership
    for sheet in source['sheets'].values():
        sheet['face_use_ids']=[]
    charts = _planar_face_charts(source,lambda:None)
    edge=source['edges'][carrier]
    controls=tuple(tuple((Fraction(float(x)).numerator,Fraction(float(x)).denominator)
                         for x in source['vertices'][vertex]['position'])
                   for vertex in (edge['start'],edge['end']))
    with pytest.raises(GeometryError,match='no original support plane'):
        _material_membership(controls,charts,lambda:None)

def test_multiowner_overlap_uses_complete_live_sheets():
    fixture=builders.connected_strip(10)
    model=fixture.model
    carrier=next(edge.id for edge in model.edges.values()
                 if all(model.vertex_position(v)[0]==1. and model.vertex_position(v)[2]==0.
                        for v in (edge.start,edge.end)) and
                    model.vertex_position(edge.start)[1]!=model.vertex_position(edge.end)[1])
    member=model.add_member((carrier,))
    fixture.operands.append(model.handle('member',member))
    apply_intersections(model,plan_intersections(model,fixture.operands,policy='connect'),policy='connect')
    joint=next(a.target_id for a in model.attachments.values() if a.kind=='sheet_on_joint')
    scope,source,current=snapshots(model,joint)
    multi = next(j for j in current['junctions'].values()
                 if j['kind']=='overlap' and len(j['sheet_ids'])>1)
    qualify(model,scope,source,current)
    multi['sheet_ids']=multi['sheet_ids'][:1]
    with pytest.raises(GeometryError,match='Sheet'):
        qualify(model,scope,source,current)

def test_stale_cancelled_and_callback_binding():
    model,joint=build('connected_strip',10)
    receipt=query(model,joint)
    before=to_dict(model)
    with pytest.raises(GeometryError,match='cancelled'):
        query(model,joint,cancellation_check=lambda phase:True)
    assert to_dict(model)==before
    with pytest.raises(GeometryError,match='stale'):
        query(model,joint,expected_revision=receipt.scope.face_preimages.revision+1)
    def corrupt(phase):
        object.__setattr__(receipt,'relations_json','{}')
        return False
    with pytest.raises(GeometryError,match='binding changed'):
        validate(model,receipt,cancellation_check=corrupt)
    assert to_dict(model)==before

def test_public_receipt_omitted_point_relations_refuse():
    import json
    model,joint=build('connected_hub',4)
    receipt=query(model,joint)
    payload=receipt.relations
    payload['point_contacts'].pop()
    with pytest.raises(GeometryError,match='binding changed'):
        validate(model,replace(receipt,relations_json=json.dumps(payload)))

@pytest.mark.parametrize('kind',['point_duplicate','overlap_omission','unrelated_vertex'])
def test_contact_inventory_and_literal_vertex_identity(kind):
    import copy
    model,joint=build('connected_hub',4)
    scope,source,current=snapshots(model,joint)
    if kind=='point_duplicate':
        row=next(row for row in current['attachments'].values() if row['kind']=='member_through_face')
        duplicate=copy.deepcopy(row)
        duplicate['id']=max(current['attachments'])+1
        current['attachments'][duplicate['id']]=duplicate
        junction=next(j for j in current['junctions'].values() if row['id'] in j['attachment_ids'])
        junction['attachment_ids'].append(duplicate['id'])
    elif kind=='overlap_omission':
        row=next(row for row in current['attachments'].values() if row['kind']=='member_on_face_boundary')
        junction=next(j for j in current['junctions'].values() if row['id'] in j['attachment_ids'])
        del current['attachments'][row['id']]
        del current['junctions'][junction['id']]
    else:
        row=next(row for row in current['attachments'].values() if row['kind']=='member_through_face')
        relation=query(model,joint).relations['point_contacts'][0]
        edge=current['edges'][relation['current_carrier']]
        p=Fraction(float(row['member_range'][0]))
        use=next(use for use in current['member_edge_uses'].values()
                 if use['member_id']==row['member_id'] and use['edge_id']==edge['id'])
        # Replace the canonical endpoint only in the named trim by an exactly
        # coincident, separate vertex; carrier ancestry and coordinates survive.
        vertex=edge['start'] if use['parent_range'][0]==float(p) else edge['end']
        fake=max(current['vertices'])+1
        current['vertices'][fake]=dict(current['vertices'][vertex],id=fake)
        target=current['faces'][row['target_id']]
        for loop in (target['loop'],*target['holes']):
            for target_edge,_ in loop:
                item=current['edges'][target_edge]
                if item['id']==edge['id']:
                    continue
                for endpoint in ('start','end'):
                    if item[endpoint]==vertex:
                        item[endpoint]=fake
        # This target is transverse to the Member, so no carrier edge is in its loop.
    with pytest.raises(GeometryError):
        qualify(model,scope,source,current)

def test_edge_local_source_point_and_reversed_interior_member():
    from anygeometry.structural import ParameterRange
    fixture=builders.connected_strip(10)
    model=fixture.model
    member=next(iter(model.members.values()))
    edge=model.edges[model.member_edge_uses[member.edge_use_ids[0]].edge_id]
    a,b=(model.vertex_position(vertex) for vertex in (edge.start,edge.end))
    point=model.add_point(*(a*.75+b*.25))
    attachment=model.add_attachment(None,'vertex_on_edge','edge',edge.id,
        ParameterRange.point(0.),(ParameterRange.point(.25),),source_kind='vertex',
        source_id=point,evidence='exact',tolerance_used=1e-9)
    model.reverse_member(member.id)
    apply_intersections(model,plan_intersections(model,fixture.operands,policy='connect'),policy='connect')
    joint=next(a.target_id for a in model.attachments.values() if a.kind=='sheet_on_joint')
    before=to_dict(model)
    receipt=query(model,joint)
    assert any(row['source_attachment']['id']==attachment
               for row in receipt.relations['attachments'])
    assert any(row['id']==point for row in receipt.source_records['vertices'])
    validate(model,receipt)
    assert to_dict(model)==before

def test_coherent_narrowed_overlap_cannot_omit_carrier_quarters():
    model,joint=build('connected_hub',4)
    scope,source,current=snapshots(model,joint)
    row=next(row for row in current['attachments'].values()
             if row['kind']=='member_on_face_boundary')
    junction=next(j for j in current['junctions'].values() if row['id'] in j['attachment_ids'])
    low,high=row['member_range']
    narrowed=[low+(high-low)*.25,low+(high-low)*.75]
    row['member_range']=narrowed
    row['target_parameters']=[[.25,.75]]
    junction['member_uses'][0]['member_range']=narrowed[:]
    with pytest.raises(GeometryError,match='full'):
        qualify(model,scope,source,current)

def test_point_source_uv_bound_cannot_accumulate_two_tolerances(monkeypatch):
    import anygeometry.prepared_planar_member_sheet_network as module
    captured={}
    original=module._qualify_member_point_contacts
    def capture(*args):
        captured['args']=args
        return original(*args)
    monkeypatch.setattr(module,'_qualify_member_point_contacts',capture)
    model,joint=build('connected_hub',4)
    scope,source,current=snapshots(model,joint)
    result=qualify(model,scope,source,current)
    contact=__import__('json').loads(result.payload)['point_contacts'][0]
    row=current['attachments'][contact['attachment']['id']]
    tolerance=Fraction(*contact['coordinate_tolerance'])
    member=next(r for r in captured['args'][4] if r['source_member']['id']==row['member_id'])
    station=Fraction(float(row['member_range'][0]))
    use=next(u for u in member['current_member_uses']
             if u['edge_id']==contact['current_carrier'])
    low,high=(Fraction(float(x)) for x in use['parent_range'])
    local=(station-low)/(high-low)
    if use['orientation']=='reversed':
        local=1-local
    edge=current['edges'][use['edge_id']]
    vertex=edge['start'] if local==0 else edge['end']
    frame=module._frame(module._decode_surface(current['faces'][row['target_id']]['surface'],
                         strict=True,schema_version=module._QUADRIC_VERSION))
    axis=frame['u']
    length=float(sum(x*x for x in axis))**.5
    position=current['vertices'][vertex]['position']
    current['vertices'][vertex]['position']=[float(x)+.75*float(tolerance)*float(a)/length
                                            for x,a in zip(position,axis)]
    uv=row['target_parameters'][0][0]+1.5*float(tolerance)/length
    row['target_parameters'][0]=[uv,uv]
    mapping=next(m for m in member['source_use_mappings']
                 if m['source_carrier']==contact['source_carrier'])
    a,b=(Fraction(*x) for x in mapping['source_span'])
    native=(station-a)/(b-a)
    if mapping['orientation']=='reversed':
        native=1-native
    controls=captured['args'][8][mapping['source_carrier']][2]
    source_point=module._point(tuple(tuple(Fraction(*x) for x in point) for point in controls),native)
    current_point=module._point(module._current_edge_controls(current,edge['id']),local)
    u,v=(Fraction(float(pair[0])) for pair in row['target_parameters'])
    uv_point=tuple(o+u*x+v*y for o,x,y in zip(frame['origin'],frame['u'],frame['v']))
    assert module._distance_squared(source_point,current_point)<=tolerance**2
    assert module._distance_squared(uv_point,current_point)<=tolerance**2
    assert module._distance_squared(source_point,uv_point)>tolerance**2
    with pytest.raises(GeometryError,match='source-to-UV'):
        original(*captured['args'])

@pytest.mark.parametrize('kind',['overlap_member','overlap_target','overlap_junction',
                               'point_member','point_uv','point_junction'])
def test_malformed_contact_range_lengths_are_typed(kind):
    model,joint=build('connected_hub',4)
    scope,source,current=snapshots(model,joint)
    attachment_kind='member_on_face_boundary' if kind.startswith('overlap') else 'member_through_face'
    row=next(r for r in current['attachments'].values() if r['kind']==attachment_kind)
    if kind.endswith('junction'):
        junction=next(j for j in current['junctions'].values() if row['id'] in j['attachment_ids'])
        junction['member_uses'][0]['member_range']=[.5]
    elif kind.endswith('member'):
        row['member_range']=[.5]
    else:
        row['target_parameters'][0]=[.5]
    with pytest.raises(GeometryError):
        qualify(model,scope,source,current)

def test_degenerate_projected_segment_is_typed():
    from anygeometry.prepared_planar_member_sheet_network import _boundary_events
    p=(Fraction(0),Fraction(0))
    with pytest.raises(GeometryError,match='degenerate projected segment'):
        _boundary_events(p,p,(((p,(Fraction(1),Fraction(0))),),))

@pytest.mark.parametrize('document',['source','current'])
def test_unrelated_orphan_face_cannot_be_dropped_from_whole_document(document):
    import copy
    model,joint=build('connected_strip',10)
    before=to_dict(model)
    scope,source,current=snapshots(model,joint)
    selected=source if document=='source' else current
    orphan=copy.deepcopy(next(iter(selected['faces'].values())))
    orphan['id']=max(selected['faces'])+1
    selected['faces'][orphan['id']]=orphan
    with pytest.raises(GeometryError,match='unowned faces refuse'):
        qualify(model,scope,source,current)
    assert to_dict(model)==before
