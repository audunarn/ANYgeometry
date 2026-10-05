"""Independent small exact perpendicular network and relation adversaries."""
from dataclasses import replace
import json
import pytest
from fractions import Fraction
import numpy as np
from anygeometry import GeometryModel, Plane, plan_intersections, apply_intersections, to_dict
from anygeometry.structural import ParameterRange
from anygeometry import GeometryError, query_trimmed_surface_charts
from anygeometry import (
    query_prepared_member_sheet_joint_network as query,
    validate_prepared_member_sheet_joint_network_binding as validate,
)


def build_network(operand_count=5, shared=False, extra_points=False, empty_part=False,
                  disconnected=False, station=Fraction(3,4), unsplit=False, central=False):
    if not 3 <= operand_count <= 10:
        raise ValueError('This disposable fixture covers 3-10 operands only, not a kernel cap.')
    model = GeometryModel()
    last = operand_count-2
    base = model.add_plate(model.add_points(((-1,-2,0),(last+1,-2,0),
                                              (last+1,2,0),(-1,2,0))))
    model.set_face_surface(base, Plane((0,0,0),(1,0,0),(0,1,0)))
    model.add_sheet((base,))
    expected = []
    half = 2 if central else 1
    for x in range(operand_count-1):
        face = model.add_plate(model.add_points(((x,-half,-1),(x,half,-1),
                                                  (x,half,1),(x,-half,1))))
        model.set_face_surface(face, Plane((x,0,0),(0,1,0),(0,0,1)))
        model.add_sheet((face,))
        edge = next(e for e in model.edges.values()
                    if all((model.vertex_position(v)[0] == x and
                            model.vertex_position(v)[2 if unsplit else 1] == (-1 if unsplit else -half))
                           for v in (e.start,e.end)))
        member = model.add_member((edge.id,))
        if x % 2:
            model.reverse_member(member)
        if shared and x == 0:
            second = model.add_member((edge.id,))
            model.reverse_member(second)
        # Independent original endpoint interpolation, not edge sample evidence.
        a, b = (tuple(Fraction(float(c)) for c in model.vertex_position(v))
                for v in (edge.start,edge.end))
        xyz = tuple((1-station)*u+station*v for u,v in zip(a,b))
        vertex = model.add_point(*(float(c) for c in xyz))
        attachment = model.add_attachment(None, 'vertex_on_edge', 'edge', edge.id,
            ParameterRange.point(0.), (ParameterRange.point(float(station)),),
            source_kind='vertex',source_id=vertex,evidence='exact',tolerance_used=1e-9)
        expected.append({'member':member,'carrier':edge.id,'attachment':attachment,
                         'point':vertex,'xyz':tuple(map(float,xyz)),
                         'station':(station.numerator,station.denominator),'reverse':bool(x%2)})
        if extra_points:
            xyz2 = tuple(Fraction(3,4)*u+Fraction(1,4)*v for u,v in zip(a,b))
            vertex2 = model.add_point(*(float(c) for c in xyz2))
            model.add_attachment(None, 'vertex_on_edge', 'edge', edge.id,
                ParameterRange.point(0.), (ParameterRange.point(.25),),
                source_kind='vertex',source_id=vertex2,evidence='exact',tolerance_used=1e-9)
    if empty_part:
        model.add_part(name='retained empty Part')
    if central:
        edge = next(e for e in model.edges.values()
                    if all(np.array_equal(model.vertex_position(v)[1:],(-2,0))
                           for v in (e.start,e.end)))
        member = model.add_member((edge.id,))
        model.reverse_member(member)
        a,b = (tuple(Fraction(float(c)) for c in model.vertex_position(v))
               for v in (edge.start,edge.end))
        xyz = tuple(Fraction(1,4)*u+Fraction(3,4)*v for u,v in zip(a,b))
        vertex = model.add_point(*(float(c) for c in xyz))
        model.add_attachment(None,'vertex_on_edge','edge',edge.id,ParameterRange.point(0.),
            (ParameterRange.point(.75),),source_kind='vertex',source_id=vertex,
            evidence='exact',tolerance_used=1e-9)
    if disconnected:
        face = model.add_plate(model.add_points(((30,30,30),(32,30,30),
                                                  (32,32,30),(30,32,30))))
        model.set_face_surface(face,Plane((0,0,30),(1,0,0),(0,1,0)))
        model.add_sheet((face,))
    original = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind=='sheet_on_joint')
    return model,joint,original,expected


@pytest.mark.parametrize('count', (3,5,10))
def test_connected_network_complete_exact_relations_and_material(count):
    model,joint,original,expected = build_network(count)
    before = to_dict(model)
    receipt = query(model,joint)
    payload = receipt.relations
    assert len(receipt.sheet_ids) == count
    assert len(payload['members']) == len(payload['attachments']) == count-1
    assert len(receipt.source_records['members']) == count-1
    assert len(receipt.current_records['member_edge_uses']) == 2*(count-1)
    assert set(receipt.current_face_ids) == set(model.faces)
    assert receipt.bounded_relation_mapping_qualified
    assert receipt.occurrence_mapping_qualified
    assert not any((receipt.semantic_mapping_qualified, receipt.beam_discretization_qualified,
                    receipt.external_reference_transfer_qualified,receipt.publication_qualified))
    for row, point in zip(payload['attachments'],expected):
        assert row['source_attachment']['id'] == point['attachment']
        assert row['current_attachment']['target_parameters'] == [[.5,.5]]
        assert row['point_squared_distance_bound'] == [0,1]
        assert tuple(row['current_vertex']['position']) == point['xyz']
    # Independently constructed rectangle totals: base4*count, other4*(count-1).
    charts = query_trimmed_surface_charts(model)
    assert sum(chart.material_area for chart in charts.charts) == pytest.approx(8*count-4,rel=0,abs=1e-12)
    validate(model,receipt)
    assert query(model,joint) == receipt
    assert to_dict(model) == before
    assert len(original['faces']) == count


def test_shared_original_carrier_and_multiple_points_are_fully_accounted():
    model,joint,_,_ = build_network(3,shared=True,extra_points=True)
    receipt = query(model,joint)
    assert len(receipt.relations['members']) == 3
    assert len(receipt.relations['attachments']) == 4
    assert len(receipt.current_records['member_edge_uses']) == 6
    validate(model,receipt)


def test_whole_network_inventory_retains_empty_parts():
    model,joint,original,_ = build_network(3,empty_part=True)
    receipt = query(model,joint)
    assert {row['id'] for row in receipt.source_records['parts']} == set(model.parts)
    assert set(receipt.part_ids) == set(model.parts)
    assert receipt.source_records['parts'] == receipt.current_records['parts']
    assert len(receipt.source_records['parts']) == len(original['structural']['parts'])
    validate(model,receipt)


@pytest.mark.parametrize('field', ('members','attachments'))
def test_omitted_network_relation_refuses(field):
    model,joint,_,_ = build_network(3)
    receipt = query(model,joint)
    payload = receipt.relations
    payload[field].pop()
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate(model,replace(receipt,relations_json=json.dumps(payload)))


def test_cancelled_network_query_leaves_document_unchanged():
    model,joint,_,_ = build_network(3)
    before = to_dict(model)
    with pytest.raises(GeometryError,match='cancelled'):
        query(model,joint,cancellation_check=lambda phase: True)
    assert to_dict(model) == before


def test_stale_network_receipt_refuses():
    model,joint,_,_ = build_network(3)
    receipt = query(model,joint)
    model.add_point(20,20,20)
    with pytest.raises(GeometryError):
        validate(model,receipt)


def test_boundary_members_do_not_enable_legacy_or_bounded_receipts():
    from anygeometry import query_prepared_sheet_joint_component, query_prepared_member_sheet_joint_component
    model,joint,_,_ = build_network(3)
    with pytest.raises(GeometryError,match='Member semantics'):
        query_prepared_sheet_joint_component(model,joint)
    with pytest.raises(GeometryError,match='two-Sheet'):
        query_prepared_member_sheet_joint_component(model,joint)


def test_source_serialization_cannot_repair_invalid_network_receipt():
    model,joint,_,_ = build_network(3)
    receipt = query(model,joint)
    forged = replace(receipt,relations_json='{}')
    triggered = []
    class Repair(dict):
        def __deepcopy__(self,memo):
            object.__setattr__(forged,'relations_json',receipt.relations_json)
            model._serialization_extensions = {}
            triggered.append(True)
            return {}
    model._serialization_extensions = Repair()
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate(model,forged)
    assert triggered


def test_network_unknown_class_hooks_do_not_execute():
    model,joint,_,_ = build_network(3)
    receipt = query(model,joint)
    triggered = []
    class Meta(type):
        def __hash__(cls):
            triggered.append('hash')
            return type.__hash__(cls)
        def __eq__(cls,other):
            triggered.append('eq')
            return type.__eq__(cls,other)
    class Behavioral(dict,metaclass=Meta):
        pass
    with pytest.raises(GeometryError,match='plain immutable owner fields'):
        validate(model,replace(receipt,scope=Behavioral()))
    assert not triggered


def test_network_point_coordinates_are_not_display_samples(monkeypatch):
    model,joint,_,_ = build_network(3)
    def display_unavailable(*args,**kwargs):
        raise AssertionError('display samples are not relation proofs')
    monkeypatch.setattr(model,'sample_edge',display_unavailable)
    receipt = query(model,joint)
    assert all(row['point_squared_distance_bound']==[0,1]
               for row in receipt.relations['attachments'])
    validate(model,receipt)


def test_disconnected_operands_refuse_without_exclusion_or_mutation():
    model,joint,_,_ = build_network(3,disconnected=True)
    before = to_dict(model)
    with pytest.raises(GeometryError,match='whole document'):
        query(model,joint)
    assert to_dict(model) == before


@pytest.mark.parametrize('station', (Fraction(0),Fraction(1,2),Fraction(1)))
def test_endpoint_attachment_source_identity_and_coordinates_retained(station):
    model,joint,_,expected = build_network(3,station=station)
    receipt = query(model,joint)
    for row, point in zip(receipt.relations['attachments'],expected):
        assert row['source_vertex']['id'] == row['current_vertex']['id'] == point['point']
        assert tuple(row['current_vertex']['position']) == point['xyz']
        assert row['point_squared_distance_bound'] == [0,1]
        parameter = row['current_attachment']['target_parameters'][0][0]
        assert parameter in (0.,1.)
        assert point['point'] in {v['id'] for v in receipt.current_records['vertices']}
    validate(model,receipt)


def test_unsplit_carriers_retain_original_attachment_lineage():
    model,joint,_,_ = build_network(3,unsplit=True)
    receipt = query(model,joint)
    assert all(len(row['current_member_uses'])==1 for row in receipt.relations['members'])
    for row in receipt.relations['attachments']:
        assert row['current_attachment']['lineage'] == row['source_attachment']['lineage']
        assert row['current_attachment']['target_parameters'] == [[.75,.75]]
    validate(model,receipt)


def test_many_cuts_through_one_reversed_boundary_member():
    model,joint,_,_ = build_network(5,central=True)
    receipt = query(model,joint)
    central = max(receipt.relations['members'],key=lambda row:len(row['current_member_uses']))
    assert len(central['current_member_uses']) == 5
    assert central['source_member_uses'][0]['orientation'] == 'reversed'
    ranges = [row['parent_range'] for row in central['current_member_uses']]
    assert ranges[0][0]==0 and ranges[-1][1]==1
    assert all(first[1]==second[0] for first,second in zip(ranges,ranges[1:]))
    certificates = central['station_certificates']
    assert any(Fraction(*row['ancestry_squared_distance_bound']) > 0 for row in certificates)
    assert all(Fraction(*row['member_station_squared_distance_bound']) <=
               Fraction(*row['coordinate_tolerance'])**2 for row in certificates)
    point = next(row for row in receipt.relations['attachments']
                 if row['source_carrier']==central['source_carrier'])
    assert len(point['current_attachment']['lineage']) == 4
    validate(model,receipt)


def test_member_parameter_error_cannot_exceed_unchanged_tolerance():
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,_ = build_network(5,central=True)
    scope = query(model,joint).scope
    source,current = _index(scope.authored_document),_index(scope.current_document)
    member = max(current['members'].values(),key=lambda row:len(row['edge_use_ids']))
    first,second = (current['member_edge_uses'][key] for key in member['edge_use_ids'][:2])
    first['parent_range'][1] += .001
    second['parent_range'][0] = first['parent_range'][1]
    with pytest.raises(GeometryError,match='station restriction exceeds'):
        _qualify(model,scope,source,current,None)


def test_long_replacement_chain_has_no_recursion_count_ceiling():
    from anygeometry.prepared_member_sheet_network import _split_path
    parents = {i:(i-1,) for i in range(2,2001)}
    assert _split_path(parents,1,2000,{},lambda:None) == tuple(range(1,2000))


@pytest.mark.parametrize('parents',({3:(2,),2:(3,)},{4:(2,3),2:(1,),3:(1,)}))
def test_unusable_replacement_lineage_refuses_typed(parents):
    from anygeometry.prepared_member_sheet_network import _split_path
    with pytest.raises(GeometryError,match='cyclic|ambiguous'):
        _split_path(parents,1,max(parents),{},lambda:None)


@pytest.mark.parametrize('corruption',('source_owner','missing_current_use','missing_carrier'))
def test_network_proof_refuses_invalid_use_ownership_and_dangling_refs(corruption):
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,_ = build_network(3)
    scope = query(model,joint).scope
    source,current = _index(scope.authored_document),_index(scope.current_document)
    member = next(iter(source['members']))
    source_use = source['member_edge_uses'][source['members'][member]['edge_use_ids'][0]]
    if corruption=='source_owner':
        source_use['member_id'] = -1
    elif corruption=='missing_current_use':
        del current['member_edge_uses'][current['members'][member]['edge_use_ids'][0]]
    else:
        del source['edges'][source_use['edge_id']]
    with pytest.raises(GeometryError,match='owner|unavailable'):
        _qualify(model,scope,source,current,None)


@pytest.mark.parametrize('corruption',('target','lineage'))
def test_point_target_or_replacement_lineage_forgery_refuses(corruption):
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,_ = build_network(5,central=True)
    scope = query(model,joint).scope
    source,current = _index(scope.authored_document),_index(scope.current_document)
    point = next(row for row in current['attachments'].values() if row['kind']=='vertex_on_edge')
    if corruption=='target':
        point['target_parameters'] = [[.5001,.5001]]
    else:
        point['lineage'].append(['edge',999999])
    with pytest.raises(GeometryError,match='target or station|retained fields'):
        _qualify(model,scope,source,current,None)
