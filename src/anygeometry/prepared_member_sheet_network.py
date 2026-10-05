"""Connected whole-document member/point relation proof for Sheet networks.

This is the additive large-connected-model owner slice. It generalizes the
bounded single-member contract without weakening it: arbitrary finite counts
of original boundary Members and exact vertex-on-edge point Attachments are
proved independently against both documents. It grants no beam discretization,
load transfer or publication authority.
"""
from dataclasses import dataclass, fields
from fractions import Fraction
import json
from uuid import UUID

import numpy as np

from .definition_binding import definition_checksum
from .edge_subcurve_preimages import (
    PreparedEdgeSubcurvePreimages,
    EdgeSubcurvePreimage, PolynomialEdgeAncestor, PolynomialEdgeDefinition,
    query_prepared_edge_subcurve_preimages,
    validate_prepared_edge_subcurve_preimages_binding,
)
from .errors import GeometryError
from .prepared_model_scope import validate_prepared_model_scope_binding
from .prepared_model_scope import PreparedModelScope
from .prepared_face_preimages import PreparedFacePreimages, AuthoredFaceDefinition
from .prepared_sheet_joint_component import PreparedSheetJointComponent, _query_component


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _require(condition, detail):
    if not condition:
        raise GeometryError('prepared member Sheet network ' + detail)


def _controls(definition):
    _require(len(definition.controls) == 2 and
             all(len(point) == 3 for point in definition.controls),
             'requires three-dimensional straight controls')
    return tuple(tuple(Fraction(*value) for value in point)
                 for point in definition.controls)


def _point(controls, parameter):
    return tuple((1-parameter)*a+parameter*b for a,b in zip(*controls))


def _distance_squared(first, second):
    return sum((a-b)**2 for a,b in zip(first,second))


def _carrier_tolerance(model, record):
    controls = _controls(record.current_definition)
    length = float(np.linalg.norm(np.asarray(controls[1],dtype=float)-
                                  np.asarray(controls[0],dtype=float)))
    tolerance = Fraction(model.tolerance.effective_length(length))
    if record.tolerance is not None:
        tolerance = min(tolerance,Fraction(*record.tolerance))
    _require(tolerance >= 0,'has unusable carrier tolerance')
    return tolerance


def _split_path(parents, root, target, cache, check):
    """Find the unique persisted replacement chain, excluding the live target."""
    key = (root,target)
    if key in cache:
        return cache[key]
    # Iterative traversal avoids an implicit recursion-depth ceiling on a long
    # sequence of valid cuts. Keep only path counts and predecessor pointers.
    active, ways, chosen = set(), {root:1}, {}
    stack = [(target,False)]
    while stack:
        check()
        edge, closing = stack.pop()
        if edge in ways:
            continue
        if closing:
            active.remove(edge)
            paths = sum(ways[parent] for parent in parents.get(edge,()))
            _require(paths <= 1,'ambiguous edge replacement lineage')
            ways[edge] = paths
            if paths:
                chosen[edge] = next(parent for parent in parents[edge] if ways[parent])
        else:
            _require(edge not in active,'cyclic edge replacement lineage')
            active.add(edge)
            stack.append((edge,True))
            stack.extend((parent,False) for parent in reversed(parents.get(edge,()))
                         if parent not in ways)
    _require(ways[target]==1,'point carrier replacement lineage is unavailable')
    path = []
    edge = target
    while edge != root:
        check()
        edge = chosen[edge]
        path.append(edge)
    cache[key] = tuple(reversed(path))
    return cache[key]


@dataclass(frozen=True, slots=True)
class PreparedMemberSheetJointNetwork(PreparedSheetJointComponent):
    """Every original boundary member and exact point relation of one document.

    Complete source/current records remain in ``scope``. The additional JSON
    accounts for all bounded relations separately from the Sheet-joint
    records. It grants no beam discretization, load transfer or publication
    authority.
    """
    relations_json: str = ''
    edge_preimages: object = None
    bounded_relation_mapping_qualified: bool = True
    beam_discretization_qualified: bool = False
    external_reference_transfer_qualified: bool = False

    @property
    def relations(self):
        return json.loads(self.relations_json)


@dataclass(frozen=True)
class _NetworkRelations:
    members: tuple
    attachment_ids: tuple
    source_edges: tuple
    current_edges: tuple
    source_vertices: tuple
    source_index: object
    payload: str
    edge_preimages: object

    def output_signature(self, result):
        return _receipt_signature(result)

    def part_ids(self, data):
        # Whole-document inventory includes even unchanged empty Parts. This
        # records their definitions without qualifying external semantics.
        return set(data['parts'])

    def complete_records(self, encoded, data):
        records = json.loads(encoded)
        # The engine calls this once per document after the component expansion
        # reached its fixpoint, so the captured records are the full selection.
        _require({row['id'] for row in records['sheets']} == set(data['sheets']),
                 'requires the whole document as one connected component')
        boundary = {row['id'] for row in records['edges']}
        carriers = self.source_edges if data is self.source_index else self.current_edges
        _require(set(carriers) <= boundary,
                 'member carrier is outside the selected component boundary')
        records['members'] = [data['members'][key] for key in self.members]
        use_keys = sorted({key for member in self.members
                           for key in data['members'][member]['edge_use_ids']})
        records['member_edge_uses'] = [data['member_edge_uses'][key] for key in use_keys]
        for kind, keys in (('vertices', set(self.source_vertices)),
                           ('parts', set(data['parts']))):
            by_id = {row['id']: row for row in records[kind]}
            by_id.update({key: data[kind][key] for key in keys})
            records[kind] = [by_id[key] for key in sorted(by_id)]
        return _encode(records)

    def validate_dependencies(self, data, deps, *, original):
        # Member carriers are checked against the completed component boundary
        # in complete_records; here every document Member/Part must be ours.
        _require(set(data['members']) == set(self.members), 'has unaccounted Members')
        for part in data['parts'].values():
            _require(set(part['member_ids']) <= set(self.members),
                     'has unaccounted Part Members')

    def validate_final(self, model, scope, *, cancellation_check):
        validate_prepared_edge_subcurve_preimages_binding(
            model, self.edge_preimages, cancellation_check=cancellation_check)
        validate_prepared_model_scope_binding(model, scope)


def _carrier_ancestry(ancestry_by_root, scope, source, root, carriers):
    """Reuse one public ancestry query; cache the per-carrier proof."""
    cached = carriers.get(root)
    if cached is not None:
        return cached
    records = sorted(ancestry_by_root.get(root, ()),
                     key=lambda row: Fraction(*row.interval[0]))
    _require(len(records) >= 1, 'required source-carrier ancestry is unavailable')
    intervals = [(Fraction(*r.interval[0]), Fraction(*r.interval[1])) for r in records]
    _require(intervals[0][0] == 0 and intervals[-1][1] == 1 and
             all(a < b for a, b in intervals) and
             all(first[1] == second[0] for first, second in zip(intervals, intervals[1:])),
             'source carrier ancestry does not tile [0,1]')
    original_controls = tuple(tuple((value.numerator, value.denominator)
        for value in map(lambda x: Fraction(float(x)), source['vertices'][vertex]['position']))
        for vertex in (source['edges'][root]['start'], source['edges'][root]['end']))
    _require(all(len(point) == 3 for point in original_controls),
             'requires three-dimensional original controls')
    _require(original_controls[0] != original_controls[1],
             'requires a nondegenerate source carrier')
    _require(all(r.ancestor.model_id == scope.face_preimages.authored_model_id
                 and r.ancestor.revision == scope.face_preimages.authored_revision
                 and r.ancestor.source_checksum == scope.face_preimages.authored_checksum
                 and r.ancestor.definition.start == source['edges'][root]['start']
                 and r.ancestor.definition.end == source['edges'][root]['end']
                 and r.ancestor.definition.controls == original_controls for r in records),
             'requires straight restrictions anchored to the original document')
    for record in records:
        bound = Fraction(*record.squared_distance_bound)
        _require(bound >= 0 and (bound == 0 if record.tolerance is None else
                 bound <= Fraction(*record.tolerance)**2),
                 'ancestry restriction exceeds its unchanged certified tolerance')
    cached = (tuple(records), tuple(intervals), original_controls)
    carriers[root] = cached
    return cached


def _qualify(model, scope, source, current, cancellation_check):
    """Independently prove every Member, use, point and Part; refuse partials."""
    _require(not source['junctions'], 'does not remap pre-existing Junctions')
    _require(set(source['sheets']) == set(current['sheets']),
             'Sheet set changed between documents')
    members = sorted(source['members'])
    _require(members, 'needs at least one source boundary Member')
    _require(set(current['members']) == set(members), 'lost or gained boundary Members')
    _require(source['parts'] == current['parts'], 'Part fields or membership changed')
    ancestry = query_prepared_edge_subcurve_preimages(
        model, expected_revision=scope.face_preimages.revision,
        cancellation_check=cancellation_check)
    ancestry_by_root = {}
    for row in ancestry.records:
        ancestry_by_root.setdefault(row.ancestor.definition.edge_id, []).append(row)
    parents = {}
    for row in scope.current_document.get('replacement_history',()):
        if row['old'][0]=='edge':
            for kind,child in row['new']:
                _require(kind=='edge','unsupported edge replacement kind')
                parents.setdefault(child,[]).append(row['old'][1])
    lineage_cache = {}
    def check():
        if cancellation_check is not None and cancellation_check('prepared member Sheet network'):
            raise GeometryError('prepared member Sheet network cancelled')
    carriers = {}
    member_relations = []
    source_edges, current_edges, old_use_ids, current_use_ids = [], [], [], []
    for member in members:
        check()
        old_member = source['members'][member]
        new_member = current['members'][member]
        _require({k: v for k, v in old_member.items() if k != 'edge_use_ids'} ==
                 {k: v for k, v in new_member.items() if k != 'edge_use_ids'},
                 'Member fields changed')
        part = old_member['part_id']
        _require(part is None or (part in source['parts'] and part in current['parts'] and
                                 source['parts'][part] == current['parts'][part]),
                 'Member Part fields or membership changed')
        _require(len(old_member['edge_use_ids']) == 1 and
                 set(old_member['edge_use_ids']) <= set(source['member_edge_uses']),
                 'needs one fully accounted original MemberEdgeUse')
        old_use = source['member_edge_uses'][old_member['edge_use_ids'][0]]
        _require(old_use['member_id'] == member, 'original MemberEdgeUse has a different owner')
        old_use_ids.extend(old_member['edge_use_ids'])
        root = old_use['edge_id']
        _require(root in source['edges'], 'original Member carrier is unavailable')
        _require(all(vertex in source['vertices'] for vertex in
                     (source['edges'][root]['start'], source['edges'][root]['end'])),
                 'original Member carrier vertex is unavailable')
        _require(old_use['parent_range'] == [0., 1.] and
                 old_use['orientation'] in ('forward', 'reversed'),
                 'has unsupported original Member station semantics')
        _require(old_member['orientation_reference'] is None,
                 'has unsupported orientation references')
        _require(source['edges'][root]['curve'] == {'type': 'straight'},
                 'requires a straight original carrier')
        records, intervals, original_controls = _carrier_ancestry(
            ancestry_by_root, scope, source, root, carriers)
        use_ids = new_member['edge_use_ids']
        _require(len(set(use_ids)) == len(use_ids), 'has duplicate current MemberEdgeUses')
        _require(set(use_ids) <= set(current['member_edge_uses']),
                 'current MemberEdgeUse is unavailable')
        uses = [current['member_edge_uses'][key] for key in use_ids]
        current_use_ids.extend(use_ids)
        reverse = old_use['orientation'] == 'reversed'
        traversal = list(reversed(records)) if reverse else records
        _require(len(uses) == len(traversal), 'Member use coverage changed')
        parameter_ranges = [tuple(Fraction(float(t)) for t in use['parent_range']) for use in uses]
        _require(all(len(pair)==2 and 0 <= pair[0] < pair[1] <= 1 for pair in parameter_ranges)
                 and parameter_ranges[0][0]==0 and parameter_ranges[-1][1]==1
                 and all(first[1]==second[0] for first,second in zip(parameter_ranges,parameter_ranges[1:])),
                 'Member parent ranges do not tile [0,1]')
        certificates = []
        for use, record in zip(uses, traversal):
            a, b = map(lambda value: Fraction(*value), record.interval)
            _require(use['member_id'] == member and use['edge_id'] == record.edge_id and
                     {k: v for k, v in use.items() if k not in ('id', 'edge_id', 'parent_range')} ==
                     {k: v for k, v in old_use.items() if k not in ('id', 'edge_id', 'parent_range')},
                     'Member traversal, orientation or parent station changed')
            u,v = (Fraction(float(t)) for t in use['parent_range'])
            native = (1-v,1-u) if reverse else (u,v)
            original = _controls(record.ancestor.definition)
            current_controls = _controls(record.current_definition)
            # The error is affine. Its squared norm is convex, so the maximum
            # of the two exact endpoint residuals bounds the WHOLE interval.
            parameter_bound = max(_distance_squared(_point(original,t),point)
                                  for t,point in zip(native,current_controls))
            tolerance = _carrier_tolerance(model,record)
            _require(parameter_bound <= tolerance**2,
                     'Member station restriction exceeds unchanged owner tolerance')
            certificates.append({'member_edge_use':use['id'],'edge':record.edge_id,
                'native_parent_interval':[[t.numerator,t.denominator] for t in native],
                'ancestry_squared_distance_bound':list(record.squared_distance_bound),
                'member_station_squared_distance_bound':[parameter_bound.numerator,parameter_bound.denominator],
                'coordinate_tolerance':[tolerance.numerator,tolerance.denominator]})
        source_edges.append(root)
        current_edges.extend(use['edge_id'] for use in uses)
        member_relations.append({'source_member': old_member, 'current_member': new_member,
                                 'source_member_uses': [old_use], 'current_member_uses': uses,
                                 'source_carrier': root,
                                 'current_carriers': [record.edge_id for record in records],
                                 'station_certificates':certificates})
    _require(len(old_use_ids) == len(set(old_use_ids)) and
             set(old_use_ids) == set(source['member_edge_uses']),
             'has unaccounted original MemberEdgeUses')
    _require(len(current_use_ids) == len(set(current_use_ids)) and
             set(current_use_ids) == set(current['member_edge_uses']),
             'has unaccounted current MemberEdgeUses')
    points, point_vertices, attachment_relations = [], [], []
    for attachment in sorted(source['attachments']):
        check()
        old = source['attachments'][attachment]
        if old['kind'] != 'vertex_on_edge':
            _require(old['kind'] == 'sheet_on_joint',
                     'has unsupported original Attachment semantics')
            continue
        points.append(attachment)
        _require(attachment in current['attachments'], 'lost its point Attachment')
        new = current['attachments'][attachment]
        _require(old['source_kind'] == 'vertex' and old['target_kind'] == 'edge'
                 and old['member_id'] is None and old['evidence'] == 'exact'
                 and len(old['target_parameters']) == 1
                 and len(old['target_parameters'][0]) == 2
                 and old['target_parameters'][0][0] == old['target_parameters'][0][1],
                 'has unsupported point Attachment semantics')
        root = old['target_id']
        _require(root in carriers, 'point Attachment target is not a member carrier')
        records, intervals, original_controls = carriers[root]
        station = Fraction(old['target_parameters'][0][0])
        _require(0 <= station <= 1, 'point Attachment station is outside the carrier')
        candidates = [(r, a, b) for r, (a, b) in zip(records, intervals)
                       if r.edge_id == new['target_id']]
        _require(len(candidates) == 1, 'point Attachment has no unique retained carrier')
        record, a, b = candidates[0]
        _require(len(new['target_parameters'])==1 and
                 len(new['target_parameters'][0])==2 and
                 new['target_parameters'][0][0]==new['target_parameters'][0][1],
                 'point Attachment no longer has one point station')
        local = Fraction(float(new['target_parameters'][0][0]))
        _require(0 <= local <= 1,'point Attachment current station is outside the carrier')
        split_path = _split_path(parents,root,record.edge_id,lineage_cache,check)
        expected_lineage = [list(value) for value in dict.fromkeys(
            (*map(tuple, old['lineage']), *(('edge',edge) for edge in split_path)))]
        _require(new['lineage'] == expected_lineage and
                 {k: v for k, v in old.items() if k not in ('target_id', 'target_parameters', 'lineage')} ==
                 {k: v for k, v in new.items() if k not in ('target_id', 'target_parameters', 'lineage')},
                 'point Attachment station or retained fields changed')
        vertex = old['source_id']
        _require(source['vertices'].get(vertex) == current['vertices'].get(vertex)
                 and vertex in source['vertices'], 'point Attachment source vertex changed')
        # The whole straight-curve restriction is certified above; verify the
        # persistent source vertex relation with the owner's original tolerance.
        actual = model.vertex_position(vertex)
        _require(np.asarray(actual).shape == (3,) and np.isfinite(actual).all(),
                 'requires a finite three-dimensional source vertex')
        tolerance = min(Fraction(float(old['tolerance_used'])),_carrier_tolerance(model,record))
        expected = tuple((1-station)*Fraction(*a) + station*Fraction(*b)
                         for a, b in zip(*original_controls))
        squared_distance = sum((Fraction(float(x))-y)**2 for x, y in zip(actual, expected))
        _require(squared_distance <= tolerance**2,
                 'point Attachment exceeds its unchanged owner tolerance')
        current_point = _point(_controls(record.current_definition),local)
        target_bound = _distance_squared(tuple(Fraction(float(x)) for x in actual),current_point)
        parameter_bound = _distance_squared(expected,current_point)
        _require(target_bound <= tolerance**2 and parameter_bound <= tolerance**2,
                 'point Attachment target or station exceeds unchanged owner tolerance')
        point_vertices.append(vertex)
        ancestry_station = (station-a)/(b-a)
        attachment_relations.append({'source_attachment': old, 'current_attachment': new,
                                     'source_vertex': source['vertices'][vertex],
                                     'current_vertex': current['vertices'][vertex],
                                     'source_carrier': root,
                                     'point_squared_distance_bound':
                                         [squared_distance.numerator, squared_distance.denominator],
                                     'point_target_squared_distance_bound':[target_bound.numerator,target_bound.denominator],
                                     'point_station_squared_distance_bound':[parameter_bound.numerator,parameter_bound.denominator],
                                     'ancestry_station':[ancestry_station.numerator,ancestry_station.denominator],
                                     'coordinate_tolerance':[tolerance.numerator,tolerance.denominator]})
    _require(all(row['kind'] == 'sheet_on_joint'
                 for key, row in current['attachments'].items() if key not in points),
             'has unaccounted current Attachments')
    _require(all(row['kind'] == 'sheet_joint' and not row['member_uses']
                 and not (set(row['attachment_ids']) & set(points))
                 for row in current['junctions'].values()),
             'has unsupported Junction/member semantics')
    payload = _encode({'members': member_relations, 'attachments': attachment_relations})
    result = _NetworkRelations(tuple(members), tuple(points), tuple(source_edges),
                               tuple(current_edges), tuple(point_vertices), source,
                               payload, ancestry)
    result.validate_final(model, scope, cancellation_check=cancellation_check)
    return result


def query_prepared_member_sheet_joint_network(model, current_joint_edge_id, *,
        expected_revision=None, cancellation_check=None):
    """Qualify every bounded relation of the whole connected document."""
    holder = []

    def qualify(*args):
        relation = _qualify(*args)
        holder.append(relation)
        return relation

    def receipt(*args, **kwargs):
        relation = holder[0]
        return PreparedMemberSheetJointNetwork(*args, **kwargs,
            relations_json=relation.payload, edge_preimages=relation.edge_preimages)

    return _query_component(model, current_joint_edge_id,
        expected_revision=expected_revision, cancellation_check=cancellation_check,
        relation_factory=qualify, result_factory=receipt)


def _receipt_signature(receipt):
    """Pin supplied content without model serialization or copying hooks."""
    _require(type(receipt) is PreparedMemberSheetJointNetwork,
             'needs its distinct owner receipt')
    _require(all(type(getattr(receipt, name)) is str for name in
                 ('relations_json', 'source_records_json', 'current_records_json')) and
             type(receipt.edge_preimages) is PreparedEdgeSubcurvePreimages,
             'needs plain immutable relation and ancestry fields')
    # Definition fingerprints alone are not a type certificate: a Mapping can
    # imitate the encoded shape of a dataclass and execute code while traversed.
    # Reject every non-plain graph node BEFORE computing the signature.
    allowed = (PreparedMemberSheetJointNetwork, PreparedModelScope,
               PreparedFacePreimages, AuthoredFaceDefinition,
               PreparedEdgeSubcurvePreimages, EdgeSubcurvePreimage,
               PolynomialEdgeAncestor, PolynomialEdgeDefinition)
    active = set()
    def plain(value):
        kind = type(value)
        if any(kind is scalar for scalar in (type(None), bool, int, float, str)):
            return
        if kind is UUID:
            _require(type(object.__getattribute__(value, 'int')) is int,
                     'needs a plain immutable UUID')
            return
        _require(kind is tuple or any(kind is cls for cls in allowed),
                 'needs plain immutable owner fields')
        identifier = id(value)
        _require(identifier not in active, 'has cyclic receipt fields')
        active.add(identifier)
        try:
            if kind is tuple:
                for item in value:
                    plain(item)
            else:
                for field in fields(kind):
                    plain(object.__getattribute__(value, field.name))
        finally:
            active.remove(identifier)
    try:
        plain(receipt)
    except RecursionError as error:
        raise GeometryError('prepared member Sheet network invalid receipt nesting') from error
    try:
        return definition_checksum(receipt)
    except RecursionError as error:
        raise GeometryError('prepared member Sheet network invalid receipt nesting') from error
    except (TypeError, ValueError) as error:
        raise GeometryError('prepared member Sheet network invalid receipt definition') from error


def validate_prepared_member_sheet_joint_network_binding(model, receipt, *,
        cancellation_check=None):
    """Rederive all bounded relations; omissions cannot qualify."""
    # Even callback-free source serialization can execute user copy hooks.
    # Pin evidence before owner/model work, and inspect it after the LAST guard.
    signature = _receipt_signature(receipt)
    validate_prepared_model_scope_binding(model, receipt.scope)
    expected = query_prepared_member_sheet_joint_network(model, receipt.joint_edge_id,
        expected_revision=receipt.scope.face_preimages.revision,
        cancellation_check=cancellation_check)
    _require(_receipt_signature(expected) == signature and
             _receipt_signature(receipt) == signature, 'definition binding changed')
    validate_prepared_model_scope_binding(model, receipt.scope)
    _require(_receipt_signature(receipt) == signature, 'definition binding changed')
