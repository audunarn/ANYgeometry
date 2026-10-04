"""Complete typed owner inventory for selected authored-root traces.

Literal record equality is not semantic preservation, a parameter remap,
material coverage, external-reference completeness or mesh admission.
"""
from dataclasses import dataclass
import json
from numbers import Integral

import numpy as np

from .authored_boundary_correspondence import (
    AuthoredBoundaryCorrespondence,
    query_prepared_authored_boundary_correspondence,
    validate_prepared_authored_boundary_correspondence_binding,
)
from .definition_binding import definition_checksum
from .errors import GeometryError
from .prepared_model_scope import (
    PreparedModelScope, query_prepared_model_scope, validate_prepared_model_scope_binding,
)


_RECORD_KINDS = ('members', 'member_edge_uses', 'attachments', 'junctions')


@dataclass(frozen=True, slots=True)
class PreparedAuthoredConstraintScope:
    scope: PreparedModelScope
    selected_root_ids: tuple
    current_face_ids: tuple
    outside_root_ids: tuple
    boundary_correspondences: tuple
    inventory_json: str
    typed_inventory_complete: bool = True
    semantic_mapping_qualified: bool = False
    parameter_remapping_qualified: bool = False
    material_qualified: bool = False
    publication_qualified: bool = False
    external_reference_scope_qualified: bool = False

    @property
    def inventory(self):
        """Fresh detached data; original/current IDs retain distinct meanings."""
        return json.loads(self.inventory_json)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _signature(receipt):
    try:
        return definition_checksum(receipt)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored constraint scope receipt is malformed') from error


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GeometryError(f'authored constraint scope requires integer {label}')
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f'authored constraint scope requires usable integer {label}') from error


def _check(callback):
    if callback is not None and callback('authored constraint scope'):
        raise GeometryError('authored constraint scope cancelled')


def _isolated_vertices(document):
    referenced = set()
    for edge in document['edges']:
        referenced.update((edge['start'], edge['end']))
        referenced.update(edge['curve'].get('control_vertices', ()))
        if 'via_vertex' in edge['curve']:
            referenced.add(edge['curve']['via_vertex'])
    # Face.corners are LOOP OFFSETS. Endpoints cover their derived vertices.
    return sorted({row['id'] for row in document['vertices']} - referenced)


def _document_inventory(document):
    metadata = {}
    tables = {kind: document[kind] for kind in ('vertices', 'edges', 'faces')}
    tables.update(document['structural'])
    for kind, rows in tables.items():
        metadata[kind] = [{'id': row['id'], 'metadata': row['metadata']}
                          for row in rows if 'metadata' in row]
    return {
        'records': {kind: document['structural'][kind] for kind in _RECORD_KINDS},
        'isolated_vertex_ids': _isolated_vertices(document),
        'opaque_unqualified': {
            'groups': document['groups'], 'tags': document['tags'],
            'features': document.get('features', {}), 'extensions': document['extensions'],
            'metadata': metadata, 'construction_vertices': document['construction_vertices'],
            'coordinates': document['coordinates'],
        },
    }


def _dispositions(original, current):
    result = []
    for kind in _RECORD_KINDS:
        before = {row['id']: row for row in original['structural'][kind]}
        after = {row['id']: row for row in current['structural'][kind]}
        for key in sorted(before.keys() | after.keys()):
            if key not in before:
                status = 'current_only'
            elif key not in after:
                status = 'source_only'
            elif _encode(before[key]) == _encode(after[key]):
                status = 'literal_unchanged'
            else:
                status = 'literal_changed'
            result.append({'kind': kind, 'id': key, 'literal_disposition': status,
                           'semantic_mapping_qualified': False,
                           'parameter_remapping_qualified': False})
    return result


def _trace_inventory(current, correspondences, roots_by_face, selected, check):
    faces = {row['id']: row for row in current['faces']}
    edges = {row['id']: row for row in current['edges']}
    vertices = {row['id']: row for row in current['vertices']}
    structural = current['structural']
    uses = {row['id']: row for row in structural['face_uses']}
    sheets = {row['id']: row for row in structural['sheets']}
    coedges = {row['id']: row for row in structural['coedges']}
    face_incidence, coedge_incidence, uses_by_face = {}, {}, {}
    for face in faces.values():
        check()
        for loop in (face['loop'], *face['holes']):
            for edge, forward in loop:
                face_incidence.setdefault(edge, []).append((face['id'], forward))
    for row in coedges.values():
        check()
        coedge_incidence.setdefault(row['edge_id'], []).append(row['id'])
    for key, use in uses.items():
        check()
        uses_by_face.setdefault(use['face_id'], []).append(key)
    result, outside = [], set()
    for correspondence in correspondences:
        root = correspondence.authored_definition.face_id
        exterior = {edge for loop in correspondence.exterior_loops
                    for _, _, current_edges in loop for edge in current_edges}
        interior = dict(correspondence.interior_incidence)
        if exterior & interior.keys():
            raise GeometryError('authored constraint scope has ambiguous root trace incidence')
        for edge_id in sorted(exterior | interior.keys()):
            check()
            adjacent = tuple(sorted(face_incidence.get(edge_id, ())))
            if not adjacent:
                raise GeometryError('authored constraint scope trace has no literal adjacent faces')
            root_incidence = tuple(row for row in adjacent if row[0] in correspondence.descendants)
            if edge_id in interior and root_incidence != tuple(sorted(interior[edge_id])):
                raise GeometryError('authored constraint scope interior incidence changed')
            if edge_id in exterior and len(root_incidence) != 1:
                raise GeometryError('authored constraint scope exterior incidence changed')
            adjacent_faces = sorted({face for face, _ in adjacent})
            neighbour_roots = set()
            for face in adjacent_faces:
                if face not in roots_by_face:
                    raise GeometryError('authored constraint scope has an unmapped adjacent face')
                neighbour_roots.update(roots_by_face[face])
            outside_roots = neighbour_roots - selected
            outside.update(outside_roots)
            adjacent_uses = sorted(key for face in adjacent_faces
                                   for key in uses_by_face.get(face, ()))
            adjacent_coedges = sorted(coedge_incidence.get(edge_id, ()))
            adjacent_sheets = sorted({uses[key]['sheet_id'] for key in adjacent_uses})
            edge = edges[edge_id]
            result.append({
                'authored_root_id': root, 'current_edge_id': edge_id,
                'trace_kind': 'exterior' if edge_id in exterior else 'paired_interior',
                'root_child_incidence': root_incidence, 'adjacent_face_incidence': adjacent,
                'endpoint_ids': (edge['start'], edge['end']),
                'endpoint_records': [vertices[edge['start']], vertices[edge['end']]],
                'edge_definition': edge,
                'adjacent_face_ids': adjacent_faces,
                'adjacent_faces': [faces[key] for key in adjacent_faces],
                'adjacent_face_use_ids': adjacent_uses,
                'adjacent_face_uses': [uses[key] for key in adjacent_uses],
                'adjacent_sheet_ids': adjacent_sheets,
                'adjacent_sheets': [sheets[key] for key in adjacent_sheets],
                'adjacent_coedge_ids': adjacent_coedges,
                'adjacent_coedges': [coedges[key] for key in adjacent_coedges],
                'adjacent_authored_root_ids': sorted(neighbour_roots),
                'outside_authored_root_ids': sorted(outside_roots),
                'semantic_mapping_qualified': False, 'parameter_remapping_qualified': False,
            })
    return result, tuple(sorted(outside))


def query_prepared_authored_constraint_scope(model, authored_face_ids, *, expected_revision=None,
                                             cancellation_check=None):
    """Inventory all typed records and literal selected-root trace neighbours.

    Whole original/current Member, MemberEdgeUse, Attachment and Junction tables
    prevent incoming-reference omission. Dispositions compare literal payloads,
    not semantic meaning or parameter carriers. Outside roots are reported,
    never silently selected or granted shared-publication permission.
    """
    scope = query_prepared_model_scope(model)
    try:
        identifiers = tuple(_integer(value, 'authored face IDs') for value in tuple(authored_face_ids))
    except TypeError as error:
        raise GeometryError('authored constraint scope requires iterable authored face IDs') from error
    revision = scope.face_preimages.revision
    if expected_revision is not None and _integer(expected_revision, 'revision') != revision:
        raise GeometryError('authored constraint scope revision is stale')
    validate_prepared_model_scope_binding(model, scope)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise GeometryError('authored constraint scope requires nonempty unique authored face IDs')
    selected = tuple(sorted(identifiers))
    if not set(selected) <= set(scope.face_preimages.authored_face_ids):
        raise GeometryError('authored constraint scope has missing authored face IDs')
    original, current = scope.authored_document, scope.current_document

    def check():
        _check(cancellation_check)

    check()
    correspondences = []
    for root in selected:
        check()
        validate_prepared_model_scope_binding(model, scope)
        correspondence = query_prepared_authored_boundary_correspondence(model, root,
            expected_revision=revision, cancellation_check=cancellation_check)
        if correspondence.face_preimages != scope.face_preimages:
            raise GeometryError('authored constraint scope boundaries bind a different preparation')
        correspondences.append(correspondence)
    roots_by_face = {}
    for root, descendants in scope.face_preimages.face_descendants:
        for face in descendants:
            roots_by_face.setdefault(face, set()).add(root)
    traces, outside = _trace_inventory(current, correspondences, roots_by_face, set(selected), check)
    current_faces = tuple(sorted({face for item in correspondences for face in item.descendants}))
    inventory = {'original': _document_inventory(original), 'current': _document_inventory(current),
                 'record_dispositions': _dispositions(original, current), 'traces': traces,
                 'external_reference_scope_qualified': False,
                 'semantic_mapping_qualified': False, 'parameter_remapping_qualified': False}
    result = PreparedAuthoredConstraintScope(scope, selected, current_faces, outside,
                                            tuple(correspondences), _encode(inventory))
    check()
    # Final owner guards are callback-free; no caller action follows them.
    for correspondence in correspondences:
        validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    validate_prepared_model_scope_binding(model, scope)
    return result


def validate_prepared_authored_constraint_scope_binding(model, receipt, *, cancellation_check=None):
    """Recompute the owner inventory; forged omissions cannot inherit freshness."""
    if (type(receipt) is not PreparedAuthoredConstraintScope or
            type(receipt.scope) is not PreparedModelScope or type(receipt.inventory_json) is not str or
            any(type(getattr(receipt, field)) is not tuple or
                any(type(key) is not int for key in getattr(receipt, field))
                for field in ('selected_root_ids', 'current_face_ids', 'outside_root_ids')) or
            type(receipt.boundary_correspondences) is not tuple or
            any(type(item) is not AuthoredBoundaryCorrespondence
                for item in receipt.boundary_correspondences)):
        raise GeometryError('authored constraint scope requires a plain immutable owner receipt')
    validate_prepared_model_scope_binding(model, receipt.scope)
    signature = _signature(receipt)
    expected = query_prepared_authored_constraint_scope(model, receipt.selected_root_ids,
        expected_revision=receipt.scope.face_preimages.revision, cancellation_check=cancellation_check)
    if _signature(expected) != signature or _signature(receipt) != signature:
        raise GeometryError('authored constraint scope definition binding changed')
    validate_prepared_model_scope_binding(model, receipt.scope)
