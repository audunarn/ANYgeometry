"""Transient authored-face provenance for qualified intersection preparation.

This proof is separate from classification and document replacement history.
Only exact owner application deltas compose its map. It is not serialized.
"""
from dataclasses import dataclass, replace
import json
from numbers import Integral
from types import MappingProxyType
from uuid import UUID

from .definition_binding import definition_checksum
from .errors import GeometryError
from .serialization import to_dict, _serialized_model_state


@dataclass(frozen=True, slots=True)
class AuthoredFaceDefinition:
    """Original dependency snapshot, not a material/partition certificate.

    ``definition_json`` is canonical immutable JSON. It contains complete face
    and edge/vertex definitions, original occurrence records, and scoped group
    memberships/tags. Parsing it gives detached data, never live model aliases.
    """
    face_id: int
    model_id: UUID
    revision: int
    source_checksum: str
    definition_json: str


@dataclass(frozen=True, slots=True)
class PreparedFacePreimages:
    authored_model_id: UUID
    authored_revision: int
    authored_checksum: str
    authored_face_ids: tuple[int, ...]
    model_id: UUID
    revision: int
    source_checksum: str
    face_descendants: tuple[tuple[int, tuple[int, ...]], ...]
    coverage: tuple[int, ...]
    authored_face_definitions: tuple[AuthoredFaceDefinition, ...] = ()
    # Complete prospective persisted owner input, not a meshing certificate.
    # None denotes legacy ID/per-face-only receipts and must not prove absence.
    authored_document_json: str | None = None
    # Prospective actual stored support coefficients, including Cylinder's raw
    # nonserialized circumferential vector. Legacy evidence cannot invent them.
    authored_native_supports: tuple | None = None
    current_native_supports: tuple | None = None
    # Explicit new-epoch sources, propagated by actual owner remappers.
    # None preserves historical provenance behavior; lineage is never a selector here.
    attachment_source_ids: tuple | None = None
    epoch_derived_attachment_ids: tuple | None = None

    @property
    def source_to_current_faces(self):
        """Read-only map; authored IDs need not remain active entities."""
        return MappingProxyType(dict(self.face_descendants))


def _binding_checksum(binding):
    try:
        return definition_checksum(binding)
    except (TypeError, ValueError) as exc:
        raise GeometryError('prepared face provenance definition binding changed') from exc


def _validate_shape(model, binding):
    if binding.authored_document_json is not None and type(binding.authored_document_json) is not str:
        raise GeometryError('prepared original model snapshot has invalid content')
    sources=binding.attachment_source_ids
    if sources is not None:
        if (type(sources) is not tuple or any(type(row) is not tuple or len(row)!=2 or
            any(type(key) is not int or key<=0 for key in row) for row in sources) or
            tuple(sorted(sources))!=sources or len({key for key,_ in sources})!=len(sources) or
            not {key for key,_ in sources}<=set(model.attachments) or binding.authored_document_json is None):
            raise GeometryError('prepared epoch attachment source map is invalid')
        derived=binding.epoch_derived_attachment_ids
        if (type(derived) is not tuple or any(type(key) is not int or key<=0 for key in derived) or
                tuple(sorted(set(derived)))!=derived or set(derived)&{key for key,_ in sources} or
                set(derived)|{key for key,_ in sources}!=set(model.attachments)):
            raise GeometryError('epoch attachment creation coverage is invalid')
        source_ids={row['id'] for row in json.loads(binding.authored_document_json)['structural']['attachments']}
        if ({key for _,key in sources}!=source_ids or
                not (source_ids & set(model.attachments))<={key for key,_ in sources}):
            raise GeometryError('prepared epoch attachment source map is incomplete or has unknown source IDs')
    authored = binding.authored_face_ids
    if (not isinstance(authored, tuple) or any(type(face) is not int or face <= 0 for face in authored)
            or tuple(sorted(set(authored))) != authored):
        raise GeometryError('prepared face provenance has invalid authored identities')
    entries = binding.face_descendants
    if (not isinstance(entries, tuple) or any(not isinstance(entry, tuple) or len(entry) != 2
                                            for entry in entries)
            or tuple(entry[0] for entry in entries) != authored):
        raise GeometryError('prepared face provenance has invalid mapping keys')
    leaves = []
    for _, descendants in entries:
        if (not isinstance(descendants, tuple) or not descendants
                or any(type(face) is not int or face <= 0 for face in descendants)
                or tuple(sorted(set(descendants))) != descendants):
            raise GeometryError('prepared face provenance has invalid descendants')
        leaves.extend(descendants)
    if len(set(leaves)) != len(leaves) or set(leaves) != set(model.faces):
        raise GeometryError('prepared face provenance has missing or conflicting preimages')
    if binding.coverage not in ((), tuple(sorted(model.faces))):
        raise GeometryError('prepared face provenance has invalid coverage')
    definitions = binding.authored_face_definitions
    if (not isinstance(definitions, tuple)
            or any(not isinstance(item, AuthoredFaceDefinition) for item in definitions)
            or (definitions and tuple(item.face_id for item in definitions) != authored)
            or any(item.model_id != binding.authored_model_id
                   or item.revision != binding.authored_revision
                   or item.source_checksum != binding.authored_checksum
                   or not isinstance(item.definition_json, str) for item in definitions)):
        raise GeometryError('prepared authored-face definitions have invalid source binding')


def _current_receipt(model):
    receipt = getattr(model, '_prepared_face_preimages_receipt', None)
    if receipt is None:
        raise GeometryError('prepared authored-face provenance is unavailable')
    try:
        binding, checksum = receipt
    except (TypeError, ValueError) as exc:
        raise GeometryError('prepared face provenance receipt is invalid') from exc
    if not isinstance(binding, PreparedFacePreimages):
        raise GeometryError('prepared face provenance receipt is invalid')
    if binding.model_id != model.model_id or binding.authored_model_id != model.model_id:
        raise GeometryError('prepared face provenance belongs to another model')
    if binding.revision != model.revision:
        raise GeometryError('prepared face provenance is stale')
    if checksum != _binding_checksum(binding):
        raise GeometryError('prepared face provenance definition binding changed')
    # A current local preparation receipt supplies prior qualification; this
    # complete fingerprint supplies freshness, including same-revision edits.
    if _serialized_model_state(model)['checksum']['value'] != binding.source_checksum:
        raise GeometryError('prepared face provenance source binding changed')
    _validate_shape(model, binding)
    from .preparation_epochs import _validate_sources
    _validate_sources(model,binding)
    return binding


def query_prepared_face_preimages(model, *, face_ids=None, expected_revision=None):
    """Return exact authored preimages for a complete current preparation.

    The full immutable map is returned; ``face_ids`` validates requested active
    face scope. Classification without known provenance is insufficient. Loads
    and ordinary clones do not carry this local proof.
    """
    if model._transaction_journal is not None or model._notifying_hooks:
        raise GeometryError('prepared face provenance requires a committed model')
    if expected_revision is not None and expected_revision != model.revision:
        raise GeometryError('prepared face provenance query revision is stale')
    from .batch_intersections import has_current_intersection_preparation
    if not has_current_intersection_preparation(model, face_ids=face_ids):
        raise GeometryError('prepared face provenance requires a complete current preparation')
    binding = _current_receipt(model)
    if binding.coverage != tuple(sorted(model.faces)):
        raise GeometryError('prepared face provenance requires complete coverage')
    return binding


def validate_prepared_face_preimages_binding(model, binding):
    """Validate immutable provenance against the owner's current local proof."""
    if not isinstance(binding, PreparedFacePreimages):
        raise GeometryError('prepared face preimages need a PreparedFacePreimages binding')
    if binding.model_id != model.model_id:
        raise GeometryError('prepared face provenance belongs to another model')
    if binding.revision != model.revision:
        raise GeometryError('prepared face provenance is stale')
    current = query_prepared_face_preimages(model)
    if _binding_checksum(binding) != _binding_checksum(current):
        raise GeometryError('prepared face provenance definition binding changed')


def query_prepared_authored_face_definition(model, authored_face_id, *, expected_revision=None):
    """Look up an original face snapshot through a current local owner receipt.

    IDs address authored faces, which may no longer exist. Older receipts may
    retain valid ID ancestry without these prospectively captured definitions;
    this lookup refuses rather than reconstructing them from current geometry.
    """
    if isinstance(authored_face_id, bool) or not isinstance(authored_face_id, Integral) or authored_face_id <= 0:
        raise GeometryError('authored face definition needs a positive authored face ID')
    binding = query_prepared_face_preimages(model, expected_revision=expected_revision)
    if authored_face_id not in binding.authored_face_ids:
        raise GeometryError('authored face definition ID is not in the original source')
    result = next((item for item in binding.authored_face_definitions if item.face_id == authored_face_id), None)
    if result is None:
        raise GeometryError('original authored face definition is unavailable')
    return result


def _original_face_definitions(document, model_id):
    """Index ONE serialized source snapshot; never serialize per face."""
    edges = {item['id']: item for item in document['edges']}
    vertices = {item['id']: item for item in document['vertices']}
    structural = document['structural']
    sheets = {item['id']: item for item in structural['sheets']}
    parts = {item['id']: item for item in structural['parts']}
    occurrences = {}
    for item in structural['face_uses']:
        occurrences.setdefault(item['face_id'], []).append(item)
    coedges = {}
    for item in structural['coedges']:
        coedges.setdefault(item['face_use_id'], []).append(item)
    groups = {}
    for name, references in document['groups'].items():
        for ref in references:
            groups.setdefault(tuple(ref), []).append(name)
    tags = {tuple(item['entity']): item for item in document['tags']}
    result = []
    for face in document['faces']:
        face_id = face['id']
        edge_ids = sorted({edge for loop in (face['loop'], *face['holes']) for edge, _ in loop})
        selected_edges = [edges[edge] for edge in edge_ids]
        vertex_ids = set()
        for edge in selected_edges:
            vertex_ids.update((edge['start'], edge['end']))
            vertex_ids.update(edge['curve'].get('control_vertices', ()))
            if 'via_vertex' in edge['curve']:
                vertex_ids.add(edge['curve']['via_vertex'])
        uses = occurrences.get(face_id, [])
        selected_coedges = sorted((edge for use in uses for edge in coedges.get(use['id'], ())),
                                  key=lambda item: item['id'])
        selected_sheets = [sheets[i] for i in sorted({use['sheet_id'] for use in uses})]
        selected_parts = [parts[i] for i in sorted({sheet['part_id'] for sheet in selected_sheets})]
        scoped = {('face', face_id)} | {('edge', i) for i in edge_ids} | {('vertex', i) for i in vertex_ids}
        for kind, rows in (('face_use', uses), ('coedge', selected_coedges),
                           ('sheet', selected_sheets), ('part', selected_parts)):
            scoped.update((kind, item['id']) for item in rows)
        memberships = {}
        for ref in sorted(scoped):
            for name in groups.get(ref, ()):
                memberships.setdefault(name, []).append(list(ref))
        payload = dict(face=face, edges=selected_edges,
                       vertices=[vertices[i] for i in sorted(vertex_ids)],
                       occurrences=dict(face_uses=uses, coedges=selected_coedges,
                                        sheets=selected_sheets, parts=selected_parts),
                       groups=memberships, tags=[tags[ref] for ref in sorted(scoped) if ref in tags],
                       coordinates=document['coordinates'], tolerance=document['tolerance'])
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        result.append(AuthoredFaceDefinition(face_id, model_id, document['revision'],
                                            document['checksum']['value'], encoded))
    return tuple(result)


def _capture_application_preimages(model, *, allow_seed):
    """Capture before application; stale provenance never becomes new roots."""
    if hasattr(model, '_prepared_face_preimages_receipt'):
        try:
            return _current_receipt(model)
        except GeometryError:
            return None
    from .preparation_epochs import _has_epoch_permit
    epoch=_has_epoch_permit(model)
    if not allow_seed or (not epoch and any(old.kind == 'face' for old in model.replacement_history())):
        return None
    faces = tuple(sorted(model.faces))
    document = to_dict(model)
    checksum = document['checksum']['value']
    from .native_support_snapshots import capture_native_supports
    native = capture_native_supports(model)
    return PreparedFacePreimages(model.model_id, model.revision, checksum, faces,
                                 model.model_id, model.revision, checksum,
                                 tuple((face, (face,)) for face in faces), (),
                                 _original_face_definitions(document, model.model_id),
                                 json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False),
                                 native, native,
                                 tuple((key,key) for key in sorted(model.attachments)) if epoch else None,
                                 () if epoch else None)


def _compose_application_preimages(candidate, previous, changes, *, attachment_source_ids=None,epoch_derived_attachment_ids=None):
    if previous is None:
        return None
    replacements = {old.id: tuple(child.id for child in children)
                    for old, children in changes if old.kind == 'face'}

    def expand(face, visiting):
        if face in visiting:
            raise GeometryError('prepared face provenance replacement cycle')
        if face not in replacements:
            return (face,)
        return tuple(leaf for child in replacements[face]
                     for leaf in expand(child, visiting | {face}))

    entries = tuple((authored, tuple(sorted(leaf for face in descendants
                                          for leaf in expand(face, set()))))
                    for authored, descendants in previous.face_descendants)
    binding = replace(previous, model_id=candidate.model_id, revision=candidate.revision,
                      source_checksum=to_dict(candidate)['checksum']['value'],
                      face_descendants=entries, coverage=(),
                      attachment_source_ids=attachment_source_ids if attachment_source_ids is not None else previous.attachment_source_ids,
                      epoch_derived_attachment_ids=epoch_derived_attachment_ids if epoch_derived_attachment_ids is not None else previous.epoch_derived_attachment_ids)
    _validate_shape(candidate, binding)
    return binding


def _publish_application_preimages(model, binding, coverage, checksum):
    if binding is not None:
        from .native_support_snapshots import capture_native_supports
        binding = replace(binding, model_id=model.model_id, revision=model.revision,
                          source_checksum=checksum, coverage=tuple(coverage),
                          current_native_supports=capture_native_supports(model))
        model._prepared_face_preimages_receipt = (binding, _binding_checksum(binding))
        from .preparation_epochs import _seal_sources
        _seal_sources(model,binding)


def _copy_current_preimages(source, target):
    try:
        binding = _current_receipt(source)
    except GeometryError:
        return
    if target.model_id != binding.model_id or to_dict(target)['checksum']['value'] != binding.source_checksum:
        raise GeometryError('prepared face provenance changed during detached copying')
    from .native_support_snapshots import capture_native_supports
    # The qualified clone is another actual native map. Preserve original
    # capture and prospectively bind its current stored coefficients.
    binding = replace(binding, current_native_supports=capture_native_supports(target))
    target._prepared_face_preimages_receipt = (binding, _binding_checksum(binding))
    from .preparation_epochs import _seal_sources
    _seal_sources(target,binding)
