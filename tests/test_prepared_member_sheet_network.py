"""Independent small exact perpendicular network and relation adversaries."""
from dataclasses import replace
import json
import pytest
from fractions import Fraction
import numpy as np
from anygeometry import GeometryModel, Plane, OrientedEdge, plan_intersections, apply_intersections, to_dict
from anygeometry.structural import ParameterRange
from anygeometry import GeometryError, query_trimmed_surface_charts
from anygeometry import (
    query_prepared_member_sheet_joint_network as query,
    validate_prepared_member_sheet_joint_network_binding as validate,
)


def build_network(operand_count=5, shared=False, extra_points=False, empty_part=False,
                  disconnected=False, station=Fraction(3,4), unsplit=False, central=False,
                  seeded_lineage=False):
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
        lineage = (('vertex', vertex), ('edge', edge.id), ('vertex', vertex)) if seeded_lineage else ()
        attachment = model.add_attachment(None, 'vertex_on_edge', 'edge', edge.id,
            ParameterRange.point(0.), (ParameterRange.point(float(station)),),
            source_kind='vertex',source_id=vertex,evidence='exact',tolerance_used=1e-9,
            lineage=lineage)
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
        xyz = tuple((1-station)*u+station*v for u,v in zip(a,b))
        vertex = model.add_point(*(float(c) for c in xyz))
        model.add_attachment(None,'vertex_on_edge','edge',edge.id,ParameterRange.point(0.),
            (ParameterRange.point(float(station)),),source_kind='vertex',source_id=vertex,
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


@pytest.mark.parametrize('seeded_lineage', [False, True])
def test_unsplit_carriers_retain_original_attachment_lineage(seeded_lineage):
    model,joint,_,_ = build_network(3,unsplit=True,seeded_lineage=seeded_lineage)
    receipt = query(model,joint)
    assert all(len(row['current_member_uses'])==1 for row in receipt.relations['members'])
    for row in receipt.relations['attachments']:
        assert row['current_attachment']['lineage'] == row['source_attachment']['lineage']
        assert row['current_attachment']['target_parameters'] == [[.75,.75]]
    validate(model,receipt)


@pytest.mark.parametrize('seeded_lineage', [False, True])
def test_split_point_lineage_matches_owner_order_and_stable_deduplication(seeded_lineage):
    model,joint,_,_ = build_network(3,seeded_lineage=seeded_lineage)
    before = to_dict(model)
    receipt = query(model,joint)
    for row in receipt.relations['attachments']:
        source, current = row['source_attachment'], row['current_attachment']
        if seeded_lineage:
            expected = [['vertex', source['source_id']], ['edge', source['target_id']],
                        ['attachment', source['id']]]
        else:
            expected = [['attachment', source['id']], ['edge', source['target_id']]]
        assert current['lineage'] == expected
        assert current['id'] == source['id']
    validate(model,receipt)
    assert to_dict(model) == before


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
    assert point['current_attachment']['lineage'][0] == ['attachment', point['source_attachment']['id']]
    assert point['current_attachment']['lineage'][1] == ['edge', point['source_carrier']]
    assert len(point['current_attachment']['lineage']) == 5
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


@pytest.mark.parametrize('station',(Fraction(1,5),Fraction(2,5),Fraction(3,5),Fraction(4,5)))
def test_nondyadic_split_endpoint_point_uses_certified_retained_target(station):
    model,joint,_,_ = build_network(5,central=True,station=station)
    receipt = query(model,joint)
    central = max(receipt.relations['members'],key=lambda row:len(row['current_member_uses']))
    point = next(row for row in receipt.relations['attachments']
                 if row['source_carrier']==central['source_carrier'])
    assert 0 <= point['current_attachment']['target_parameters'][0][0] <= 1
    assert all(Fraction(*point[key]) <= Fraction(*point['coordinate_tolerance'])**2
               for key in ('point_squared_distance_bound','point_target_squared_distance_bound',
                           'point_station_squared_distance_bound'))
    validate(model,receipt)


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


@pytest.mark.parametrize('corruption',('target','lineage','attachment_omitted',
                                     'attachment_identity','attachment_order','attachment_duplicate'))
def test_point_target_or_replacement_lineage_forgery_refuses(corruption):
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,_ = build_network(5,central=True)
    scope = query(model,joint).scope
    source,current = _index(scope.authored_document),_index(scope.current_document)
    point = next(row for row in current['attachments'].values() if row['kind']=='vertex_on_edge')
    if corruption=='target':
        point['target_parameters'] = [[.5001,.5001]]
    elif corruption=='lineage':
        point['lineage'].append(['edge',999999])
    elif corruption=='attachment_omitted':
        point['lineage'].pop(0)
    elif corruption=='attachment_identity':
        point['lineage'][0][1] = -1
    elif corruption=='attachment_order':
        point['lineage'][0],point['lineage'][1] = point['lineage'][1],point['lineage'][0]
    else:
        point['lineage'].append(point['lineage'][0])
    before = to_dict(model)
    with pytest.raises(GeometryError,match='target or station|retained fields'):
        _qualify(model,scope,source,current,None)
    assert to_dict(model) == before


def _edge_between(model, first, second):
    return next(edge for edge in model.edges.values()
                if {edge.start, edge.end} == {first, second})


def build_chain_network(operand_count=4, central=False, repeated_carrier=False,
                        station=Fraction(3,4), mixed=False):
    """Small connected network whose Members are multi-edge boundary chains.

    Each operand plate carries a two-use chain (split side edge plus unsplit
    bottom edge) and a three-use chain sharing the bottom carrier, with the
    two-use chain reversed on alternating operands. The base plate carries one
    two-use boundary chain: split by decomposition seams, or the repeatedly cut bottom
    boundary edge plus the uncut right edge when ``central`` is set.
    """
    if not 3 <= operand_count <= 10:
        raise ValueError('This disposable fixture covers 3-10 operands only, not a kernel cap.')
    model = GeometryModel()
    last = operand_count-2
    base_points = model.add_points(((-1,-2,0),(last+1,-2,0),(last+1,2,0),(-1,2,0)))
    base = model.add_plate(base_points)
    model.set_face_surface(base, Plane((0,0,0),(1,0,0),(0,1,0)))
    model.add_sheet((base,))
    half = 2 if central else Fraction(3,2)
    operands = []
    for x in range(operand_count-1):
        corners = model.add_points(((x,-half,-1),(x,half,-1),(x,half,1),(x,-half,1)))
        if mixed:
            edges = (model.add_line(corners[1],corners[0]),
                     model.add_line(corners[1],corners[2]),
                     model.add_line(corners[2],corners[3]),
                     model.add_line(corners[3],corners[0]))
            face = model.add_face(edges)
        else:
            face = model.add_plate(corners)
        model.set_face_surface(face, Plane((x,0,0),(0,1,0),(0,0,1)))
        model.add_sheet((face,))
        bottom = _edge_between(model, corners[0], corners[1])
        right = _edge_between(model, corners[1], corners[2])
        top = _edge_between(model, corners[2], corners[3])
        left = _edge_between(model, corners[3], corners[0])
        two = model.add_member((left.id, bottom.id))
        three = model.add_member((OrientedEdge(bottom.id,not mixed), right.id, top.id))
        if x % 2:
            model.reverse_member(two)
        operands.append({'two':two,'three':three,'bottom':bottom.id,'right':right.id,
                         'top':top.id,'left':left.id,'reversed':bool(x%2)})
    if central:
        first = _edge_between(model, base_points[0], base_points[1])
        second = _edge_between(model, base_points[1], base_points[2])
    else:
        first = _edge_between(model, base_points[1], base_points[2])
        second = _edge_between(model, base_points[2], base_points[3])
    base_member = model.add_member((first.id, second.id))
    if central:
        model.reverse_member(base_member)
    if repeated_carrier:
        model.add_member((second.id, second.id))
    points = []
    for edge, st in ((bottom, Fraction(0)), (left, station), (first, station)):
        a, b = (tuple(Fraction(float(c)) for c in model.vertex_position(v))
                for v in (edge.start, edge.end))
        xyz = tuple((1-st)*u+st*v for u, v in zip(a, b))
        vertex = model.add_point(*(float(c) for c in xyz))
        model.add_attachment(None, 'vertex_on_edge', 'edge', edge.id,
            ParameterRange.point(0.), (ParameterRange.point(float(st)),),
            source_kind='vertex', source_id=vertex, evidence='exact', tolerance_used=1e-9)
        points.append({'carrier':edge.id,'station':st,'xyz':tuple(map(float,xyz)),
                       'vertex':vertex})
    original = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind=='sheet_on_joint')
    return model,joint,original,{'operands':operands,'base':{'member':base_member,
                                 'first':first.id,'second':second.id},'points':points}


def _assert_use_mapping_invariants(row):
    mappings = row['source_use_mappings']
    assert len(mappings) == len(row['source_member_uses']) == len(row['source_carriers'])
    spans = [[Fraction(*value) for value in mapping['source_span']] for mapping in mappings]
    assert spans[0][0] == 0 and spans[-1][1] == 1
    assert all(a < b for a, b in spans)
    assert all(first[1] == second[0] for first, second in zip(spans, spans[1:]))
    for mapping, use, carrier in zip(mappings, row['source_member_uses'], row['source_carriers']):
        assert mapping['source_member_use'] == use
        assert mapping['source_carrier'] == carrier
        assert mapping['orientation'] == use['orientation']
        assert Fraction(*mapping['source_span'][0]) == Fraction(float(use['parent_range'][0]))
        assert Fraction(*mapping['source_span'][1]) == Fraction(float(use['parent_range'][1]))
        assert mapping['current_member_uses']
        assert len(mapping['current_member_uses']) == len(mapping['station_certificates'])
        assert mapping['current_carriers'] == [use2['edge_id']
            for use2 in mapping['current_member_uses']] or \
            mapping['current_carriers'] == list(reversed([use2['edge_id']
            for use2 in mapping['current_member_uses']]))
    assert row['source_carrier'] == (row['source_carriers'][0]
                                     if len(mappings) == 1 else None)
    assert len(row['current_member_uses']) == len(row['station_certificates'])
    assert all(Fraction(*cert['member_station_squared_distance_bound']) <=
               Fraction(*cert['coordinate_tolerance'])**2
               for cert in row['station_certificates'])


@pytest.mark.parametrize('count', (3,4))
def test_multi_use_chains_certify_ordered_mappings_and_spans(count):
    model,joint,original,expected = build_chain_network(count)
    before = to_dict(model)
    receipt = query(model,joint)
    payload = receipt.relations
    assert len(payload['members']) == 2*(count-1)+1
    assert len(payload['attachments']) == 3
    for row in payload['members']:
        _assert_use_mapping_invariants(row)
    for operand in expected['operands']:
        two = next(row for row in payload['members']
                   if row['source_member']['id'] == operand['two'])
        three = next(row for row in payload['members']
                     if row['source_member']['id'] == operand['three'])
        assert len(two['source_use_mappings']) == 2
        assert len(three['source_use_mappings']) == 3
        # The split side carrier doubles, the unsplit bottom carrier does not.
        assert len(two['current_member_uses']) == 3
        assert len(three['current_member_uses']) == 4
        assert two['source_carrier'] is None and three['source_carrier'] is None
        assert {operand['bottom'], operand['left']} <= set(two['source_carriers'])
        assert three['source_carriers'] == [operand['bottom'], operand['right'], operand['top']]
        assert two['source_carriers'] == ([operand['bottom'], operand['left']]
                                          if operand['reversed']
                                          else [operand['left'], operand['bottom']])
        assert all(mapping['orientation'] == ('reversed' if operand['reversed'] else 'forward')
                   for mapping in two['source_use_mappings'])
        shared = (set(two['source_carriers']) & set(three['source_carriers']))
        assert shared == {operand['bottom']}
    base = next(row for row in payload['members']
                if row['source_member']['id'] == expected['base']['member'])
    assert base['source_carriers'] == [expected['base']['first'], expected['base']['second']]
    # Side decomposition seams occur at y=+-3/2; the top is partitioned
    # at every independently constructed perpendicular plate's x coordinate.
    assert [len(m['current_member_uses']) for m in base['source_use_mappings']] == [3,count]
    assert len(base['current_member_uses']) == count+3
    for row, point in zip(payload['attachments'], expected['points']):
        assert row['source_carrier'] == point['carrier']
        assert tuple(row['current_vertex']['position']) == point['xyz']
        assert row['point_squared_distance_bound'] == [0,1]
    endpoint = next(row for row in payload['attachments'] if row['source_carrier'] in
                    {operand['bottom'] for operand in expected['operands']})
    assert endpoint['current_attachment']['target_parameters'] == [[0.,0.]]
    validate(model,receipt)
    assert query(model,joint) == receipt
    assert to_dict(model) == before
    assert len(original['structural']['members']) == 2*(count-1)+1


def test_repeated_cuts_through_reversed_multi_use_base_chain():
    model,joint,_,expected = build_chain_network(5,central=True)
    receipt = query(model,joint)
    base = next(row for row in receipt.relations['members']
                if row['source_member']['id'] == expected['base']['member'])
    _assert_use_mapping_invariants(base)
    assert len(base['source_use_mappings']) == 2
    assert all(mapping['orientation'] == 'reversed' for mapping in base['source_use_mappings'])
    # Four operand cuts split the bottom boundary carrier into five retained rows.
    assert len(base['current_member_uses']) == 6
    assert base['source_carrier'] is None
    certificates = base['station_certificates']
    assert any(Fraction(*row['ancestry_squared_distance_bound']) > 0 for row in certificates)
    ranges = [row['parent_range'] for row in base['current_member_uses']]
    assert ranges[0][0]==0 and ranges[-1][1]==1
    assert all(first[1]==second[0] for first,second in zip(ranges,ranges[1:]))
    point = next(row for row in receipt.relations['attachments']
                 if row['source_carrier']==expected['base']['first'])
    assert point['current_attachment']['lineage'][0] == ['attachment', point['source_attachment']['id']]
    assert point['current_attachment']['lineage'][1] == ['edge', point['source_carrier']]
    assert len(point['current_attachment']['lineage']) == 5
    assert 0 <= point['current_attachment']['target_parameters'][0][0] <= 1
    validate(model,receipt)


def test_mixed_native_orientations_use_global_spans_without_sampling(monkeypatch):
    model,joint,original,expected = build_chain_network(3,mixed=True)
    monkeypatch.setattr(model,'sample_edge',lambda *a,**kw: pytest.fail('sampling is not an oracle'))
    receipt = query(model,joint)
    vertices = {row['id']: row['position'] for row in original['vertices']}
    edges = {row['id']: row for row in original['edges']}
    for operand in expected['operands']:
        row = next(row for row in receipt.relations['members']
                   if row['source_member']['id']==operand['two'])
        assert {mapping['orientation'] for mapping in row['source_use_mappings']} == {'forward','reversed'}
        _assert_use_mapping_invariants(row)
        for mapping in row['source_use_mappings']:
            edge = edges[mapping['source_carrier']]
            a,b = (tuple(Fraction(float(x)) for x in vertices[edge[key]])
                   for key in ('start','end'))
            p,q = (Fraction(*span) for span in mapping['source_span'])
            for use in mapping['current_member_uses']:
                u,v = (Fraction(float(x)) for x in use['parent_range'])
                native = ((u-p)/(q-p),(v-p)/(q-p))
                if mapping['orientation']=='reversed':
                    native = (1-native[1],1-native[0])
                child = model.edges[use['edge_id']]
                for t,vertex in zip(native,(child.start,child.end)):
                    wanted = tuple((1-t)*x+t*y for x,y in zip(a,b))
                    actual = tuple(Fraction(float(x)) for x in model.vertex_position(vertex))
                    # Fixture coordinates are independent rational rectangles.
                    assert sum((x-y)**2 for x,y in zip(actual,wanted)) <= Fraction(1,10**18)
    validate(model,receipt)


def test_unsplit_use_in_multi_use_chain_retains_original_attachment_lineage():
    model,joint,_,expected = build_chain_network(3)
    receipt = query(model,joint)
    carrier = expected['operands'][-1]['bottom']
    mappings = [mapping for row in receipt.relations['members']
                for mapping in row['source_use_mappings']
                if mapping['source_carrier'] == carrier]
    assert len(mappings) == 2  # Shared by two independently owned Members.
    assert all(len(mapping['current_member_uses']) == 1 for mapping in mappings)
    point = next(row for row in receipt.relations['attachments']
                 if row['source_carrier']==carrier)
    assert point['current_attachment']['lineage'] == point['source_attachment']['lineage']
    assert point['current_attachment']['target_parameters'] == [[0.,0.]]
    validate(model,receipt)


def test_repeated_carrier_within_one_member_refuses_ambiguous_semantics():
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,_ = build_chain_network(3)
    before = to_dict(model)
    scope = query(model,joint).scope
    source,current = _index(scope.authored_document),_index(scope.current_document)
    member = next(row for row in source['members'].values() if len(row['edge_use_ids'])==2)
    first,second = (source['member_edge_uses'][key] for key in member['edge_use_ids'])
    second['edge_id'] = first['edge_id']
    with pytest.raises(GeometryError,match='repeats an original Member carrier'):
        _qualify(model,scope,source,current,None)
    assert to_dict(model) == before


@pytest.mark.parametrize('corruption',('source_span','source_owner','current_tiling',
                                       'current_order','missing_descendant','current_owner'))
def test_multi_use_tiling_and_ownership_refuse_typed(corruption):
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,_ = build_chain_network(4)
    scope = query(model,joint).scope
    source,current = _index(scope.authored_document),_index(scope.current_document)
    member = next(key for key,row in source['members'].items() if len(row['edge_use_ids'])==2)
    source_use = source['member_edge_uses'][source['members'][member]['edge_use_ids'][0]]
    current_member = current['members'][member]
    current_use = current['member_edge_uses'][current_member['edge_use_ids'][0]]
    if corruption=='source_span':
        source_use['parent_range'][1] = source_use['parent_range'][1]/2
    elif corruption=='source_owner':
        source_use['member_id'] = -1
    elif corruption=='current_tiling':
        current_use['parent_range'][1] += .001
    elif corruption=='current_order':
        current_member['edge_use_ids'] = list(reversed(current_member['edge_use_ids']))
    elif corruption=='missing_descendant':
        current_member['edge_use_ids'].pop()
    else:
        current_use['member_id'] = -1
    pattern = {'source_span':'spans do not tile','source_owner':'owner',
               'current_tiling':'do not tile the original use span',
               'current_order':'out of original use order',
               'missing_descendant':'coverage changed',
               'current_owner':'traversal, orientation or parent station'}[corruption]
    with pytest.raises(GeometryError,match=pattern):
        _qualify(model,scope,source,current,None)


def test_multi_use_network_cancelled_and_stale_receipts_refuse():
    model,joint,_,_ = build_chain_network(3)
    before = to_dict(model)
    with pytest.raises(GeometryError,match='cancelled'):
        query(model,joint,cancellation_check=lambda phase: True)
    assert to_dict(model) == before
    receipt = query(model,joint)
    model.add_point(20,20,20)
    with pytest.raises(GeometryError):
        validate(model,receipt)


@pytest.mark.parametrize('corruption',('metadata','shared_station'))
def test_multi_use_descendant_metadata_and_tiling_preserving_station_refuse(corruption):
    from anygeometry.prepared_sheet_joint_component import _index
    from anygeometry.prepared_member_sheet_network import _qualify
    model,joint,_,expected = build_chain_network(3)
    receipt = query(model,joint)
    source,current = _index(receipt.scope.authored_document),_index(receipt.scope.current_document)
    member_id = expected['operands'][1]['two']  # Reversed bottom then left.
    row = next(row for row in receipt.relations['members']
               if row['source_member']['id']==member_id)
    mapping = row['source_use_mappings'][1]
    assert Fraction(*mapping['source_span'][0]) > 0
    assert len(mapping['current_member_uses']) == 2
    first,second = (current['member_edge_uses'][use['id']]
                    for use in mapping['current_member_uses'])
    if corruption=='metadata':
        first['metadata']['forged'] = True
        message = 'Member traversal, orientation or parent station changed'
    else:
        station = first['parent_range'][1]+.001
        first['parent_range'][1] = second['parent_range'][0] = station
        assert first['parent_range'][0] < station < second['parent_range'][1]
        message = 'Member station restriction exceeds unchanged owner tolerance'
    with pytest.raises(GeometryError,match=message):
        _qualify(model,receipt.scope,source,current,None)
