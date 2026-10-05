"""Whole-document CURRENT topology, occurrence and chart receipt for mixed Sheets.

This additive owner slice captures one complete connected mixed Sheet-only
document (Plane, Cylinder/Cone panels and supported extruded walls) after a
complete current intersection preparation. It proves CURRENT structural truth
only: literal incidence/index correspondence, complete generated Sheet joints
over shared canonical edges, root/source Sheet FaceUse occurrence
correspondence, complete public trimmed charts and exact recorded original
and current surface definitions.

It is not authored material coverage, reference parameter mapping,
beam/load/solver equivalence or publication permission. Authored-root curved
material coverage (exact arrangement/domain inclusion and nonoverlap) remains
future work; callers requesting it receive a typed refusal, never coverage
inferred from area totals, retained root IDs or samples. Supports use public
chart qualification, not copied curve classifications.
"""
from dataclasses import dataclass, fields
import json
from uuid import UUID

import numpy as np

from .definition_binding import definition_checksum
from .errors import GeometryError
from .prepared_face_preimages import AuthoredFaceDefinition, PreparedFacePreimages
from .prepared_model_scope import (
    PreparedModelScope, query_prepared_model_scope, validate_prepared_model_scope_binding,
)
from .prepared_sheet_joint_component import (
    _crosscheck, _dependencies, _index, _integer, _live_incidence, _live_joint,
    _qualify_joint, _qualify_preserved_source_joints, _records, _refuse_members,
    _source_relations, _touch_attachment,
)
from .serialization import _decode_surface, _QUADRIC_VERSION
from .surfaces import Plane
from .trimmed_charts import (
    TrimmedSurfaceCharts, query_trimmed_surface_charts,
    validate_trimmed_surface_charts_binding,
)

# Parametric restriction fields may narrow on a qualified split; every other
# serialized field is an invariant carrier definition and must not change.
# Height may narrow only on a qualified Cylinder split: a Cone carrier must
# retain its serialized height exactly.
def _restricted_surface_fields(surface_type):
    if surface_type == 'cylinder':
        return frozenset({'start_angle', 'sweep_angle', 'height'})
    return frozenset({'start_angle', 'sweep_angle'})
_SURFACE_FAMILIES = {
    'plane': frozenset({'plane'}),
    'cylinder': frozenset({'cylinder'}),
    'cone': frozenset({'cone'}),
    'coons': frozenset({'coons', 'extruded'}),
    'extruded': frozenset({'extruded'}),
}
_FALSE_FLAGS = (
    'semantic_mapping_qualified',
    'authored_material_coverage_qualified',
    'reference_parameter_mapping_qualified',
    'beam_discretization_qualified',
    'load_transfer_qualified',
    'solver_qualified',
    'publication_qualified',
)


@dataclass(frozen=True, slots=True)
class PreparedMixedSheetJointNetwork:
    """Complete CURRENT mixed Sheet document receipt; no material authority.

    ``occurrence_correspondence`` rows are
    ``(Sheet ID, authored root ID, original FaceUse ID, current FaceUse IDs)``.
    ``charts_json`` records the complete public trimmed-chart qualification
    per current face; ``surface_correspondence_json`` records the exact
    original/current surface definitions per current face with the carrier
    preservation verdict. ``preserved_joint_attachment_ids`` and
    ``preserved_joint_junction_ids`` record the complete unchanged original
    joint inventory. Neither record qualifies authored material
    coverage or any parameter remap.
    """
    scope: object
    joint_edge_id: int
    sheet_ids: tuple
    part_ids: tuple
    authored_face_ids: tuple
    current_face_ids: tuple
    joint_edge_ids: tuple
    junction_ids: tuple
    attachment_ids: tuple
    occurrence_correspondence: tuple
    source_records_json: str
    current_records_json: str
    charts_json: str
    surface_correspondence_json: str
    unqualified_semantics: tuple
    occurrence_mapping_qualified: bool = True
    semantic_mapping_qualified: bool = False
    authored_material_coverage_qualified: bool = False
    reference_parameter_mapping_qualified: bool = False
    beam_discretization_qualified: bool = False
    load_transfer_qualified: bool = False
    solver_qualified: bool = False
    publication_qualified: bool = False
    preserved_joint_attachment_ids: tuple = ()
    preserved_joint_junction_ids: tuple = ()

    @property
    def source_records(self):
        return json.loads(self.source_records_json)

    @property
    def current_records(self):
        return json.loads(self.current_records_json)

    @property
    def charts(self):
        return json.loads(self.charts_json)

    @property
    def surface_correspondence(self):
        return json.loads(self.surface_correspondence_json)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _require(condition, detail):
    if not condition:
        raise GeometryError('prepared mixed Sheet network ' + detail)


def _surface_correspondence(source, data, roots, tolerances, check):
    """Record exact definitions; prove only carrier preservation.

    Cylinder and Cone carriers must retain every non-restricted serialized
    field exactly; height is a restricted field for Cylinder carriers only,
    so a Cone carrier must retain its serialized height exactly. Plane
    carriers are proved against the face's certified
    chart tolerance with recorded residuals; ``exact_fields`` is labeled only
    when the serialized carrier field dictionaries are literally equal,
    because zero floating residuals alone do not establish exact serialized
    fields. Coons/extruded wall transitions are recorded verbatim
    without a carrier claim. No row is material coverage evidence.
    """
    rows = []
    for face_id in sorted(data['faces']):
        check()
        current_face = data['faces'][face_id]
        if current_face['parameterization'] is not None:
            raise GeometryError('prepared mixed Sheet network does not qualify parameterizations')
        _require(face_id in roots and roots[face_id] in source['faces'],
                 'has a current face without an authored root')
        root = roots[face_id]
        original_face = source['faces'][root]
        if original_face['parameterization'] is not None:
            raise GeometryError('prepared mixed Sheet network does not qualify original parameterizations')
        old_surface, new_surface = original_face['surface'], current_face['surface']
        old_type = old_surface.get('type') if isinstance(old_surface, dict) else None
        new_type = new_surface.get('type') if isinstance(new_surface, dict) else None
        _require(old_type in _SURFACE_FAMILIES and new_type in _SURFACE_FAMILIES[old_type],
                 'has an unsupported surface-family transition')
        row = {'face': face_id, 'root': root,
               'source_surface': old_surface, 'current_surface': new_surface,
               'carrier_residuals': []}
        if old_type == 'plane':
            old_support = _decode_surface(old_surface, strict=True, schema_version=_QUADRIC_VERSION)
            new_support = _decode_surface(new_surface, strict=True, schema_version=_QUADRIC_VERSION)
            if type(old_support) is not Plane or type(new_support) is not Plane:
                raise GeometryError('prepared mixed Sheet network requires literal Plane definitions')
            normal = np.cross(np.asarray(old_support.u_vector, dtype=float),
                              np.asarray(old_support.v_vector, dtype=float))
            scale = float(np.linalg.norm(normal))
            _require(scale > 0. and bool(np.isfinite(scale)),
                     'has a singular original Plane carrier')
            residual = [abs(float(np.dot(normal, row_vector))) for row_vector in (
                np.asarray(new_support.origin, dtype=float)
                - np.asarray(old_support.origin, dtype=float),
                np.asarray(new_support.u_vector, dtype=float),
                np.asarray(new_support.v_vector, dtype=float))]
            bound = float(tolerances[face_id]) * scale
            _require(all(bool(np.isfinite(value)) for value in residual)
                     and bool(np.isfinite(bound)) and bound >= 0.,
                     'Plane carrier proof is not finite')
            _require(all(value <= bound for value in residual),
                     'Plane carrier exceeds its certified owner tolerance')
            row['carrier_residuals'] = residual
            row['carrier_tolerance_bound'] = bound
            # Zero floating residuals do NOT establish exact serialized fields:
            # exact_fields is reserved for literally equal carrier field
            # dictionaries; every other Plane carrier is certified only
            # against its owner tolerance with recorded residuals.
            carrier = {key: value for key, value in old_surface.items()
                       if key not in _restricted_surface_fields('plane')}
            current_carrier = {key: value for key, value in new_surface.items()
                               if key not in _restricted_surface_fields('plane')}
            row['carrier_correspondence'] = ('exact_fields' if carrier == current_carrier
                                             else 'plane_owner_tolerance')
        elif old_type == new_type:
            carrier = {key: value for key, value in old_surface.items()
                       if key not in _restricted_surface_fields(old_type)}
            current_carrier = {key: value for key, value in new_surface.items()
                               if key not in _restricted_surface_fields(old_type)}
            _require(carrier == current_carrier, 'surface carrier definition changed')
            row['carrier_correspondence'] = 'exact_fields'
        else:
            # Coons/extruded wall transition: definitions are recorded
            # verbatim; no carrier claim is made for the wall family.
            row['carrier_correspondence'] = 'recorded_only'
        rows.append(row)
    return rows


def _chart_records(model, data, faces, scope, cancellation_check, check):
    """Complete public trimmed-chart qualification; no copied classifications."""
    def chart_check(phase):
        return cancellation_check is not None and bool(
            cancellation_check('prepared mixed Sheet network charts: ' + phase))

    charts = query_trimmed_surface_charts(
        model, expected_revision=scope.face_preimages.revision,
        cancellation_check=chart_check)
    if not isinstance(charts, TrimmedSurfaceCharts):
        raise GeometryError('prepared mixed Sheet network requires public trimmed charts')
    validate_trimmed_surface_charts_binding(
        model, charts, expected_revision=scope.face_preimages.revision,
        cancellation_check=chart_check)
    _require(sorted(chart.face.id for chart in charts.charts) == sorted(faces),
             'requires complete current trimmed charts')
    uses = {}
    for key, row in data['face_uses'].items():
        uses.setdefault(row['face_id'], []).append(key)
    rows = []
    for chart in charts.charts:
        check()
        _require(sorted(use.id for use in chart.face_uses) == sorted(uses[chart.face.id]),
                 'chart occurrence binding mismatch')
        rows.append({'face': chart.face.id,
                     'face_uses': sorted(use.id for use in chart.face_uses),
                     'support': type(chart.support).__name__,
                     'world_tolerance': chart.world_tolerance,
                     'material_area': chart.material_area,
                     'boundary_loops': [len(loop) for loop in chart.boundaries]})
    return charts, {'model_id': str(charts.model_id), 'revision': charts.revision,
                    'source_checksum': charts.source_checksum, 'faces': rows}


def query_prepared_mixed_sheet_joint_network(model, current_joint_edge_id, *,
                                             expected_revision=None,
                                             cancellation_check=None):
    """Capture the complete CURRENT mixed Sheet document as one owner receipt.

    Requires a completely prepared, connected, fully authored Sheet-only
    document (zero Members). Every current face must be owned by an authored
    source Sheet; every shared canonical edge must carry a complete qualified
    generated Sheet joint; every current face must have a public trimmed
    chart. Refuses unowned/disconnected faces, source-less owners, original
    attachment/junction remaps, orientation references, parameterized faces
    and unsupported surface-family transitions. Nonmutating and
    deterministic.
    """
    def check():
        if cancellation_check is not None and cancellation_check('prepared mixed Sheet network'):
            raise GeometryError('prepared mixed Sheet network cancelled')

    scope = query_prepared_model_scope(model)
    identifier = _integer(current_joint_edge_id, 'joint edge ID')
    if expected_revision is not None and _integer(expected_revision, 'expected revision') != scope.face_preimages.revision:
        raise GeometryError('prepared mixed Sheet network revision is stale')
    validate_prepared_model_scope_binding(model, scope)
    original, current = scope.authored_document, scope.current_document
    source, data = _index(original), _index(current)
    for document in (source, data):
        if document['members'] or document['member_edge_uses']:
            raise GeometryError('prepared mixed Sheet network requires a Sheet-only document (zero Members)')
        for row in document['face_uses'].values():
            if row['orientation'] not in ('forward', 'reversed'):
                raise GeometryError('prepared mixed Sheet network has unsupported occurrence orientation')
    _require(set(source['sheets']) == set(data['sheets']),
             'has a source-less or lost Sheet owner')
    _require(set(source['parts']) == set(data['parts']),
             'has a source-less or lost Part owner')
    for key in sorted(data['sheets']):
        check()
        if key not in source['sheets']:
            raise GeometryError('prepared mixed Sheet network has a batch-created source-less Sheet')
        if {k: v for k, v in data['sheets'][key].items() if k != 'face_use_ids'} != \
                {k: v for k, v in source['sheets'][key].items() if k != 'face_use_ids'}:
            raise GeometryError('prepared mixed Sheet network Sheet semantic fields changed')
    for key in sorted(data['parts']):
        if key not in source['parts']:
            raise GeometryError('prepared mixed Sheet network has a source-less Part')
        if {k: v for k, v in data['parts'][key].items() if k not in ('sheet_ids', 'member_ids')} != \
                {k: v for k, v in source['parts'][key].items() if k not in ('sheet_ids', 'member_ids')}:
            raise GeometryError('prepared mixed Sheet network Part semantic fields changed')

    occurrence_edges = {row['edge_id'] for row in data['coedges'].values()}
    incidences = {edge: _live_incidence(model, data, edge) for edge in sorted(occurrence_edges)}
    candidates = {identifier} | {edge for edge, (literal, _) in incidences.items()
                                 if len(literal[2]) >= 2} | \
        {row['target_id'] for row in data['attachments'].values()
         if row['kind'] == 'sheet_on_joint' and row['target_kind'] == 'edge'}
    observed = {key: _live_joint(model, data, key) for key in sorted(candidates)}
    validate_prepared_model_scope_binding(model, scope)
    sheets = _crosscheck(observed[identifier])

    joint_edges, junctions, attachments, inspected = set(), set(), set(), set()
    deps = None
    while True:
        check()
        deps = _dependencies(data, sheets)
        incidence_junctions = set()
        for edge in sorted(deps['edges']):
            check()
            inspected.add(edge)
            literal, derived = incidences[edge]
            if literal != derived:
                raise GeometryError('prepared mixed Sheet network literal occurrence/index mismatch')
            if len(literal[2]) >= 2:
                # A shared canonical edge is an explicit incoming occurrence
                # and must carry a complete qualified joint declaration.
                _crosscheck(observed[edge])
                incidence_junctions.update(observed[edge][3])
        near_edges = {key for key, row in data['edges'].items()
                      if key in deps['edges'] or row['start'] in deps['vertices']
                      or row['end'] in deps['vertices']}
        _refuse_members(data, deps, near_edges)
        touching = {key for key, row in data['attachments'].items()
                    if _touch_attachment(row, deps, near_edges, attachments, junctions)}
        selected = {key for key, row in data['junctions'].items()
                    if set(row['sheet_ids']) & sheets or set(row['attachment_ids']) & touching} | \
            incidence_junctions
        expanded = set(sheets)
        for key in sorted(selected):
            check()
            row = data['junctions'][key]
            if row['kind'] != 'sheet_joint':
                raise GeometryError('prepared mixed Sheet network has unsupported Junction semantics')
            edge, owners, links = _qualify_joint(data, row, observed)
            joint_edges.add(edge)
            attachments.update(links)
            junctions.add(key)
            expanded.update(owners)
        touching.update(key for key, row in data['attachments'].items()
                        if _touch_attachment(row, deps, near_edges, attachments, junctions))
        if touching - attachments:
            raise GeometryError('prepared mixed Sheet network touches unsupported/uncontained Attachments')
        if expanded == sheets:
            break
        sheets = expanded
    if identifier not in joint_edges:
        raise GeometryError('prepared mixed Sheet network lacks its requested declaration')
    _require(sheets == set(data['sheets']),
             'requires one connected whole-document Sheet component')
    _require(deps['faces'] == set(data['faces']),
             'has unowned or disconnected current faces')
    _require(junctions == set(data['junctions']),
             'has unqualified current Junctions')
    _require(attachments == set(data['attachments']),
             'has unqualified current Attachments')
    _require(set(data['parts']) and deps['parts'] == set(data['parts']),
             'has empty or unselected current Parts')

    descendants = dict(scope.face_preimages.face_descendants)
    roots = {child: root for root, children in descendants.items() for child in children}
    occurrence, authored = [], set()
    for sheet in sorted(sheets):
        old_uses = [source['face_uses'][key] for key in source['sheets'][sheet]['face_use_ids']]
        new_uses = [data['face_uses'][key] for key in data['sheets'][sheet]['face_use_ids']]
        _require(all(row['face_id'] in roots for row in new_uses),
                 'has a current face without an authored root')
        old_roots = {row['face_id'] for row in old_uses}
        if old_roots != {roots[row['face_id']] for row in new_uses}:
            raise GeometryError('prepared mixed Sheet network Sheet root membership changed')
        for root in sorted(old_roots):
            check()
            originals = [row for row in old_uses if row['face_id'] == root]
            uses = [row for row in new_uses if roots[row['face_id']] == root]
            if len(originals) != 1 or sorted(row['face_id'] for row in uses) != list(descendants[root]):
                raise GeometryError('prepared mixed Sheet network lacks unique complete root/Sheet occurrences')
            if any(row['orientation'] != originals[0]['orientation']
                   or row['metadata'] != originals[0]['metadata'] for row in uses):
                raise GeometryError('prepared mixed Sheet network occurrence properties changed')
            occurrence.append((sheet, root, originals[0]['id'],
                               tuple(sorted(row['id'] for row in uses))))
            authored.add(root)
    _require(authored == set(source['faces']),
             'requires complete authored face ownership')

    for root, children in sorted(descendants.items()):
        check()
        if children != (root,):
            continue
        if source['faces'][root] != data['faces'][root]:
            raise GeometryError('prepared mixed Sheet network untouched source face payload changed')
        for loop in (source['faces'][root]['loop'], *source['faces'][root]['holes']):
            for edge_id, _ in loop:
                if source['edges'][edge_id] != data['edges'].get(edge_id):
                    raise GeometryError('prepared mixed Sheet network untouched source edge payload changed')
                for vertex in (source['edges'][edge_id]['start'], source['edges'][edge_id]['end']):
                    if source['vertices'][vertex] != data['vertices'].get(vertex):
                        raise GeometryError('prepared mixed Sheet network untouched source vertex payload changed')

    source_deps = _dependencies(source, sheets)
    _require(source_deps['parts'] == set(source['parts']),
             'has empty or unselected source Parts')
    source_near = {key for key, row in source['edges'].items()
                    if key in source_deps['edges'] or row['start'] in source_deps['vertices']
                    or row['end'] in source_deps['vertices']}
    _refuse_members(source, source_deps, source_near)
    source_attachments, source_junctions = _source_relations(
        source, source_deps, source_near, check)
    _qualify_preserved_source_joints(source, data, source_attachments,
                                     source_junctions, attachments, junctions, check)

    charts, chart_rows = _chart_records(model, data, deps['faces'], scope,
                                        cancellation_check, check)
    tolerances = {row['face']: row['world_tolerance'] for row in chart_rows['faces']}
    correspondence = _surface_correspondence(source, data, roots, tolerances, check)

    original_records = _records(original, source, source_deps,
                                source_attachments, source_junctions)
    current_records = _records(current, data, deps, attachments, junctions)
    result = PreparedMixedSheetJointNetwork(
        scope, identifier, tuple(sorted(sheets)), tuple(sorted(deps['parts'])),
        tuple(sorted(authored)), tuple(sorted(deps['faces'])),
        tuple(sorted(joint_edges)), tuple(sorted(junctions)),
        tuple(sorted(attachments)), tuple(occurrence),
        original_records, current_records,
        _encode(chart_rows), _encode(correspondence),
        ('raw metadata/group/tag/feature/extension semantics and parameter remapping',
         'authored curved material coverage (exact arrangement/domain inclusion and nonoverlap)',
         'reference parameter mapping',
         'coons/extruded wall surface-definition correspondence beyond recorded definitions',
         'beam/load/solver/publication authority'),
        preserved_joint_attachment_ids=tuple(sorted(source_attachments)),
        preserved_joint_junction_ids=tuple(sorted(source_junctions)))
    signature = _receipt_signature(result)
    check()
    validate_prepared_model_scope_binding(model, scope)
    for edge in result.joint_edge_ids:
        if _live_joint(model, data, edge) != observed[edge]:
            raise GeometryError('prepared mixed Sheet network derived declaration changed')
    for edge in sorted(inspected):
        if _live_incidence(model, data, edge) != incidences[edge]:
            raise GeometryError('prepared mixed Sheet network derived occurrence changed')
    validate_trimmed_surface_charts_binding(
        model, charts, expected_revision=scope.face_preimages.revision,
        cancellation_check=None)
    if _receipt_signature(result) != signature:
        raise GeometryError('prepared mixed Sheet network output definition changed')
    return result


def _receipt_signature(receipt):
    """Pin supplied content without model serialization or copying hooks."""
    _require(type(receipt) is PreparedMixedSheetJointNetwork,
             'needs its distinct owner receipt')
    _require(all(type(getattr(receipt, name)) is str for name in
                 ('source_records_json', 'current_records_json', 'charts_json',
                  'surface_correspondence_json')) and
             type(receipt.scope) is PreparedModelScope,
             'needs plain immutable owner fields')
    # Definition fingerprints alone are not a type certificate: a Mapping can
    # imitate the encoded shape of a dataclass and execute code while traversed.
    # Reject every non-plain graph node BEFORE computing the signature.
    allowed = (PreparedMixedSheetJointNetwork, PreparedModelScope,
               PreparedFacePreimages, AuthoredFaceDefinition)
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
        raise GeometryError('prepared mixed Sheet network invalid receipt nesting') from error
    try:
        return definition_checksum(receipt)
    except RecursionError as error:
        raise GeometryError('prepared mixed Sheet network invalid receipt nesting') from error
    except (TypeError, ValueError) as error:
        raise GeometryError('prepared mixed Sheet network invalid receipt definition') from error


def validate_prepared_mixed_sheet_joint_network_binding(model, receipt, *,
                                                        cancellation_check=None):
    """Rederive immutable current coverage; no omitted/forged record may pass."""
    if type(receipt) is not PreparedMixedSheetJointNetwork:
        raise GeometryError('prepared mixed Sheet network requires an owner receipt')
    if type(receipt.scope) is not PreparedModelScope:
        raise GeometryError('prepared mixed Sheet network requires an owner scope')
    if receipt.occurrence_mapping_qualified is not True:
        raise GeometryError('prepared mixed Sheet network occurrence mapping is not qualified')
    for name in _FALSE_FLAGS:
        if getattr(receipt, name) is not False:
            raise GeometryError('prepared mixed Sheet network refuses forged qualification flags: ' + name)
    # Even callback-free source serialization can execute user copy hooks.
    # Pin evidence before owner/model work, and inspect it after the LAST guard.
    signature = _receipt_signature(receipt)
    validate_prepared_model_scope_binding(model, receipt.scope,
                                         cancellation_check=cancellation_check)
    expected = query_prepared_mixed_sheet_joint_network(
        model, receipt.joint_edge_id,
        expected_revision=receipt.scope.face_preimages.revision,
        cancellation_check=cancellation_check)
    if _receipt_signature(expected) != signature or _receipt_signature(receipt) != signature:
        raise GeometryError('prepared mixed Sheet network definition binding changed')
    validate_prepared_model_scope_binding(model, receipt.scope,
                                          cancellation_check=cancellation_check)
    if _receipt_signature(receipt) != signature:
        raise GeometryError('prepared mixed Sheet network definition binding changed')


def require_prepared_mixed_sheet_authored_material_coverage(receipt):
    """Typed refusal: authored curved material coverage is not qualified here.

    No area total, retained root ID, chart record or sample set can promote
    this receipt to authored material coverage. Exact arrangement/domain
    inclusion and nonoverlap certificates for generic curved source material
    remain future work.
    """
    raise GeometryError(
        'prepared mixed Sheet network does not qualify authored material coverage: '
        'exact curved source arrangement/domain inclusion and nonoverlap remain '
        'future work; this receipt proves current topology/occurrence/chart binding only')


def require_prepared_mixed_sheet_reference_parameter_mapping(receipt):
    """Typed refusal: no original-to-current parameter remap is certified."""
    raise GeometryError(
        'prepared mixed Sheet network does not qualify reference parameter '
        'mapping: no original-to-current parameter correspondence is certified; '
        'surface-definition records and chart bindings are visibility, not a remap')
