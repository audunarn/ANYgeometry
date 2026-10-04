"""Bounded member/point relation proof beside the legacy Sheet-only contract."""
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
        raise GeometryError('prepared member Sheet component ' + detail)


@dataclass(frozen=True, slots=True)
class PreparedMemberSheetJointComponent(PreparedSheetJointComponent):
    """Only one straight boundary member and one vertex-on-edge point relation.

    Complete source/current records remain in ``scope``. The additional JSON
    accounts for the bounded relation separately from the Sheet-joint records.
    It grants no beam discretization, load transfer or publication authority.
    """
    member_relation_json: str = ''
    edge_preimages: object = None
    bounded_relation_mapping_qualified: bool = True
    beam_discretization_qualified: bool = False
    external_reference_transfer_qualified: bool = False

    @property
    def member_relation(self):
        return json.loads(self.member_relation_json)


@dataclass(frozen=True)
class _Relations:
    member: int
    attachment_ids: tuple
    source_edges: tuple
    current_edges: tuple
    source_vertex: int
    payload: str
    edge_preimages: object

    def output_signature(self, result):
        return _receipt_signature(result)

    def part_ids(self, data):
        part = data['members'][self.member]['part_id']
        return set() if part is None else {part}

    def complete_records(self, encoded, data):
        records = json.loads(encoded)
        records['members'] = [data['members'][self.member]]
        records['member_edge_uses'] = [data['member_edge_uses'][key]
            for key in data['members'][self.member]['edge_use_ids']]
        for kind, keys in (('vertices', {self.source_vertex}), ('parts', self.part_ids(data))):
            by_id = {row['id']: row for row in records[kind]}
            by_id.update({key: data[kind][key] for key in keys})
            records[kind] = [by_id[key] for key in sorted(by_id)]
        return _encode(records)

    def validate_dependencies(self, data, deps, *, original):
        edges = self.source_edges if original else self.current_edges
        _require(set(edges) <= deps['edges'], 'member is not on this component boundary')
        _require(set(data['members']) == {self.member}, 'has unaccounted Members')
        # Parts may carry this member, but no undeclared additional membership.
        for part in data['parts'].values():
            _require(set(part['member_ids']) <= {self.member}, 'has unaccounted Part Members')

    def validate_final(self, model, scope, *, cancellation_check):
        validate_prepared_edge_subcurve_preimages_binding(
            model, self.edge_preimages, cancellation_check=cancellation_check)
        validate_prepared_model_scope_binding(model, scope)


def _qualify(model, scope, source, current, cancellation_check):
    _require(len(source['sheets']) == len(current['sheets']) == 2,
             'supports the qualified two-Sheet case only')
    _require(len(source['members']) == len(current['members']) == 1,
             'needs exactly one source boundary Member')
    _require(not source['junctions'], 'does not remap pre-existing Junctions')
    _require(len(source['attachments']) == 1, 'needs one original point Attachment')
    member, old_member = next(iter(source['members'].items()))
    _require(member in current['members'], 'lost its original Member')
    new_member = current['members'][member]
    _require({k: v for k, v in old_member.items() if k != 'edge_use_ids'} ==
             {k: v for k, v in new_member.items() if k != 'edge_use_ids'},
             'Member fields changed')
    part = old_member['part_id']
    _require(part is None or (part in source['parts'] and part in current['parts'] and
                             source['parts'][part] == current['parts'][part]),
             'Member Part fields or membership changed')
    _require(len(old_member['edge_use_ids']) == 1 and
             set(source['member_edge_uses']) == set(old_member['edge_use_ids']),
             'needs one fully accounted original MemberEdgeUse')
    old_use = source['member_edge_uses'][old_member['edge_use_ids'][0]]
    root = old_use['edge_id']
    _require(old_use['parent_range'] == [0., 1.] and
             old_use['orientation'] in ('forward', 'reversed'),
             'has unsupported original Member station semantics')
    _require(old_member['orientation_reference'] is None,
             'has unsupported orientation references')
    _require(source['edges'][root]['curve'] == {'type': 'straight'},
             'requires a straight original carrier')
    ancestry = query_prepared_edge_subcurve_preimages(
        model, expected_revision=scope.face_preimages.revision,
        cancellation_check=cancellation_check)
    records = sorted((row for row in ancestry.records
                      if row.ancestor.definition.edge_id == root),
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
    _require(all(r.ancestor.model_id == scope.face_preimages.authored_model_id
                 and r.ancestor.revision == scope.face_preimages.authored_revision
                 and r.ancestor.source_checksum == scope.face_preimages.authored_checksum
                 and r.ancestor.definition.start == source['edges'][root]['start']
                 and r.ancestor.definition.end == source['edges'][root]['end']
                 and r.ancestor.definition.controls == original_controls
                 and Fraction(*r.squared_distance_bound) == 0 for r in records),
             'requires exact straight restrictions anchored to the original document')
    use_ids = new_member['edge_use_ids']
    _require(len(set(use_ids)) == len(use_ids) and
             set(use_ids) == set(current['member_edge_uses']),
             'has unaccounted current MemberEdgeUses')
    uses = [current['member_edge_uses'][key] for key in use_ids]
    reverse = old_use['orientation'] == 'reversed'
    traversal = list(reversed(records)) if reverse else records
    _require(len(uses) == len(traversal), 'Member use coverage changed')
    for use, record in zip(uses, traversal):
        a, b = map(lambda value: Fraction(*value), record.interval)
        expected_range = [float(1-b), float(1-a)] if reverse else [float(a), float(b)]
        _require(use['edge_id'] == record.edge_id and
                 use['parent_range'] == expected_range and
                 {k: v for k, v in use.items() if k not in ('id', 'edge_id', 'parent_range')} ==
                 {k: v for k, v in old_use.items() if k not in ('id', 'edge_id', 'parent_range')},
                 'Member traversal, orientation or parent station changed')
    attachment, old = next(iter(source['attachments'].items()))
    _require(attachment in current['attachments'], 'lost its point Attachment')
    new = current['attachments'][attachment]
    _require(old['kind'] == 'vertex_on_edge' and old['source_kind'] == 'vertex'
             and old['target_kind'] == 'edge' and old['target_id'] == root
             and old['member_id'] is None and old['evidence'] == 'exact'
             and len(old['target_parameters']) == 1
             and old['target_parameters'][0][0] == old['target_parameters'][0][1],
             'has unsupported point Attachment semantics')
    station = Fraction(old['target_parameters'][0][0])
    _require(0 <= station <= 1, 'point Attachment station is outside the carrier')
    candidates = [(r, a, b) for r, (a, b) in zip(records, intervals)
                  if a <= station <= b and r.edge_id == new['target_id']]
    _require(len(candidates) == 1, 'point Attachment has no unique retained carrier')
    record, a, b = candidates[0]
    local = float((station-a)/(b-a))
    expected_lineage = [list(value) for value in dict.fromkeys(
        (*map(tuple, old['lineage']), ('edge', root)))]
    if len(records) == 1 and record.edge_id == root:
        expected_lineage = old['lineage']
    _require(new['target_parameters'] == [[local, local]] and
             new['lineage'] == expected_lineage and
             {k: v for k, v in old.items() if k not in ('target_id', 'target_parameters', 'lineage')} ==
             {k: v for k, v in new.items() if k not in ('target_id', 'target_parameters', 'lineage')},
             'point Attachment station or retained fields changed')
    vertex = old['source_id']
    _require(source['vertices'].get(vertex) == current['vertices'].get(vertex)
             and vertex in source['vertices'], 'point Attachment source vertex changed')
    # Whole straight-curve restriction is certified above; verify the persistent
    # source vertex relation with the owner's original, unchanged tolerance.
    controls = np.array([[float(Fraction(*x)) for x in point]
                         for point in record.ancestor.definition.controls])
    actual = model.vertex_position(vertex)
    _require(np.asarray(actual).shape == (3,) and np.isfinite(actual).all(),
             'requires a finite three-dimensional source vertex')
    tolerance = min(old['tolerance_used'], model.tolerance.effective_length(
        float(np.linalg.norm(controls[1]-controls[0]))))
    expected = tuple((1-station)*Fraction(*a) + station*Fraction(*b)
                     for a, b in zip(*original_controls))
    squared_distance = sum((Fraction(float(x))-y)**2 for x, y in zip(actual, expected))
    _require(squared_distance <= Fraction(tolerance)**2,
             'point Attachment exceeds its unchanged owner tolerance')
    _require(all(row['kind'] == 'sheet_on_joint' for key, row in current['attachments'].items()
                 if key != attachment), 'has unaccounted current Attachments')
    _require(all(row['kind'] == 'sheet_joint' and not row['member_uses']
                 and attachment not in row['attachment_ids']
                 for row in current['junctions'].values()),
             'has unsupported Junction/member semantics')
    payload = _encode({'source_member': old_member, 'current_member': new_member,
                       'source_member_uses': [old_use], 'current_member_uses': uses,
                       'source_attachment': old, 'current_attachment': new,
                       'source_vertex': source['vertices'][vertex],
                       'current_vertex': current['vertices'][vertex],
                       'point_squared_distance_bound': [squared_distance.numerator,
                                                        squared_distance.denominator],
                       'coordinate_tolerance': tolerance})
    result = _Relations(member, (attachment,), (root,),
                        tuple(r.edge_id for r in records), vertex, payload, ancestry)
    result.validate_final(model, scope, cancellation_check=cancellation_check)
    return result


def query_prepared_member_sheet_joint_component(model, current_joint_edge_id, *,
        expected_revision=None, cancellation_check=None):
    """Qualify the bounded boundary-member case without weakening Sheet-only queries."""
    holder = []

    def qualify(*args):
        relation = _qualify(*args)
        holder.append(relation)
        return relation

    def receipt(*args, **kwargs):
        relation = holder[0]
        return PreparedMemberSheetJointComponent(*args, **kwargs,
            member_relation_json=relation.payload, edge_preimages=relation.edge_preimages)

    return _query_component(model, current_joint_edge_id,
        expected_revision=expected_revision, cancellation_check=cancellation_check,
        relation_factory=qualify, result_factory=receipt)


def _receipt_signature(receipt):
    """Pin supplied content without model serialization or copying hooks."""
    _require(type(receipt) is PreparedMemberSheetJointComponent, 'needs its distinct owner receipt')
    _require(all(type(getattr(receipt, name)) is str for name in
                 ('member_relation_json', 'source_records_json', 'current_records_json')) and
             type(receipt.edge_preimages) is PreparedEdgeSubcurvePreimages,
             'needs plain immutable relation and ancestry fields')
    # Definition fingerprints alone are not a type certificate: a Mapping can
    # imitate the encoded shape of a dataclass and execute code while traversed.
    # Reject every non-plain graph node BEFORE computing the signature.
    allowed = (PreparedMemberSheetJointComponent, PreparedModelScope,
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
        raise GeometryError('prepared member Sheet component invalid receipt nesting') from error
    try:
        return definition_checksum(receipt)
    except (TypeError, ValueError) as error:
        raise GeometryError('prepared member Sheet component invalid receipt definition') from error


def validate_prepared_member_sheet_joint_component_binding(model, receipt, *,
        cancellation_check=None):
    """Rederive bounded relations and all component content; omissions cannot qualify."""
    # Even callback-free source serialization can execute user copy hooks.
    # Pin evidence before owner/model work, and inspect it after the LAST guard.
    signature = _receipt_signature(receipt)
    validate_prepared_model_scope_binding(model, receipt.scope)
    expected = query_prepared_member_sheet_joint_component(model, receipt.joint_edge_id,
        expected_revision=receipt.scope.face_preimages.revision,
        cancellation_check=cancellation_check)
    _require(_receipt_signature(expected) == signature and
             _receipt_signature(receipt) == signature, 'definition binding changed')
    validate_prepared_model_scope_binding(model, receipt.scope)
    _require(_receipt_signature(receipt) == signature, 'definition binding changed')
