"""Exact authored/current polynomial exterior correspondence, not a domain.

Oriented boundary tiling does not establish an embedded material partition,
interior curve completeness, station correspondence or meshing permission.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import json

from .definition_binding import definition_checksum
from .edge_subcurve_preimages import (
    query_prepared_edge_subcurve_preimages, validate_prepared_edge_subcurve_preimages_binding)
from .errors import GeometryError
from .prepared_face_preimages import (
    query_prepared_face_preimages, query_prepared_authored_face_definition,
    validate_prepared_face_preimages_binding)


@dataclass(frozen=True, slots=True)
class AuthoredBoundaryCorrespondence:
    face_preimages: object
    edge_preimages: object
    authored_definition: object
    descendants: tuple[int, ...]
    # Original loop order: (root edge ID, direction, current edge IDs).
    exterior_loops: tuple
    # All paired internal edges remain explicit, including physical joints.
    interior_incidence: tuple


def _check(callback):
    if callback is not None and callback('authored boundary correspondence'):
        raise GeometryError('authored boundary correspondence cancelled')


def _polynomial_definition(edge, vertices):
    curve = edge['curve']
    if curve['type'] not in ('straight', 'spline'):
        raise GeometryError('authored boundary correspondence needs polynomial original edges')
    ids = (edge['start'], *curve.get('control_vertices', ()), edge['end'])
    return tuple(tuple(F(float(x)) for x in vertices[i]['position']) for i in ids)


def _tile(rows, forward):
    """Exact directed source intervals; no tolerance weld or gap bridging."""
    directed = []
    for record, direction in rows:
        first, second = (F(*value) for value in record.interval)
        if not direction:
            first, second = second, first
        if not forward:
            first, second = 1-first, 1-second
        if not 0 <= first < second <= 1:
            raise GeometryError('authored boundary correspondence has reversed or invalid interval')
        directed.append((first, second, record.edge_id))
    directed.sort()
    previous = F(0)
    for first, second, _ in directed:
        if first != previous:
            raise GeometryError('authored boundary correspondence has a gap or duplicate coverage')
        previous = second
    if previous != 1:
        raise GeometryError('authored boundary correspondence has incomplete root coverage')
    return tuple(edge for _, _, edge in directed)


def query_prepared_authored_boundary_correspondence(model, authored_face_id, *,
        expected_revision=None, cancellation_check=None):
    """Prove exact exterior tiling for ALL descendants of one authored face.

    The original boundary, including holes, is read from prospective owner
    evidence. Current edges must carry authenticated polynomial split ancestry.
    Internal paired incidence is returned without cancelling physical joints.
    This is neither current material containment nor partition equivalence.
    """
    _check(cancellation_check)
    original = query_prepared_authored_face_definition(model, authored_face_id,
        expected_revision=expected_revision)
    faces = query_prepared_face_preimages(model, expected_revision=expected_revision)
    edges = query_prepared_edge_subcurve_preimages(model, expected_revision=expected_revision,
        cancellation_check=cancellation_check)
    descendants = dict(faces.face_descendants)[int(authored_face_id)]
    payload = json.loads(original.definition_json)
    source_face = payload['face']
    original_loops = (source_face['loop'], *source_face['holes'])
    original_edges = {row['id']: row for row in payload['edges']}
    vertices = {row['id']: row for row in payload['vertices']}
    root_ids = [edge for loop in original_loops for edge, _ in loop]
    if len(set(root_ids)) != len(root_ids):
        raise GeometryError('authored boundary correspondence has repeated original edge occurrence')
    controls = {edge: _polynomial_definition(original_edges[edge], vertices) for edge in root_ids}
    incidence = {}
    for face in descendants:
        _check(cancellation_check)
        current = model.faces[face]
        for loop in (current.loop, *current.holes):
            for use in loop:
                incidence.setdefault(use.edge, []).append((face, use.forward))
    exterior, interior = {}, []
    known = {row.edge_id: row for row in edges.records}
    for edge, uses in sorted(incidence.items()):
        _check(cancellation_check)
        if len(uses) == 2 and uses[0][1] != uses[1][1] and uses[0][0] != uses[1][0]:
            interior.append((edge, tuple(uses)))
            continue
        if len(uses) != 1:
            raise GeometryError('authored boundary correspondence has ambiguous internal incidence')
        record = known.get(edge)
        if record is None:
            raise GeometryError('authored boundary correspondence exterior ancestry is unavailable')
        ancestor = record.ancestor
        root = ancestor.definition.edge_id
        if (ancestor.model_id != original.model_id or ancestor.revision != original.revision
                or ancestor.source_checksum != original.source_checksum or root not in controls):
            raise GeometryError('authored boundary correspondence has a different original anchor')
        definition = original_edges[root]
        original_controls = tuple(tuple(F(*x) for x in point) for point in ancestor.definition.controls)
        if (original_controls != controls[root] or ancestor.definition.start != definition['start']
                or ancestor.definition.end != definition['end']):
            raise GeometryError('authored boundary correspondence original polynomial changed')
        exterior.setdefault(root, []).append((record, uses[0][1]))
    loops = []
    for loop in original_loops:
        _check(cancellation_check)
        if not loop:
            raise GeometryError('authored boundary correspondence has an empty original loop')
        endpoints, rows = [], []
        for root, forward in loop:
            points = controls[root] if forward else controls[root][::-1]
            endpoints.append((points[0], points[-1]))
            rows.append((root, forward, _tile(exterior.get(root, ()), forward)))
        if any(first[1] != second[0] for first, second in zip(endpoints, (*endpoints[1:], endpoints[0]))):
            raise GeometryError('authored boundary correspondence original loop is not exactly closed')
        loops.append(tuple(rows))
    result = AuthoredBoundaryCorrespondence(faces, edges, original, descendants,
        tuple(loops), tuple(interior))
    _check(cancellation_check)
    validate_prepared_edge_subcurve_preimages_binding(model, edges, cancellation_check=cancellation_check)
    # Check face lineage after the last callback-bearing edge validation.
    validate_prepared_face_preimages_binding(model, faces)
    return result


def validate_prepared_authored_boundary_correspondence_binding(model, result, *, cancellation_check=None):
    if not isinstance(result, AuthoredBoundaryCorrespondence):
        raise GeometryError('authored boundary correspondence needs an owner result')
    expected = query_prepared_authored_boundary_correspondence(model, result.authored_definition.face_id,
        expected_revision=result.face_preimages.revision, cancellation_check=cancellation_check)
    if definition_checksum(expected) != definition_checksum(result):
        raise GeometryError('authored boundary correspondence definition binding changed')
