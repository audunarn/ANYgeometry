"""Transient authored-face provenance for qualified intersection preparation.

This proof is separate from classification and document replacement history.
Only exact owner application deltas compose its map. It is not serialized.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
from uuid import UUID

from .definition_binding import definition_checksum
from .errors import GeometryError
from .serialization import to_dict


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
    if to_dict(model)['checksum']['value'] != binding.source_checksum:
        raise GeometryError('prepared face provenance source binding changed')
    _validate_shape(model, binding)
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


def _capture_application_preimages(model, *, allow_seed):
    """Capture before application; stale provenance never becomes new roots."""
    if hasattr(model, '_prepared_face_preimages_receipt'):
        try:
            return _current_receipt(model)
        except GeometryError:
            return None
    if not allow_seed or any(old.kind == 'face' for old in model.replacement_history()):
        return None
    faces = tuple(sorted(model.faces))
    checksum = to_dict(model)['checksum']['value']
    return PreparedFacePreimages(model.model_id, model.revision, checksum, faces,
                                 model.model_id, model.revision, checksum,
                                 tuple((face, (face,)) for face in faces), ())


def _compose_application_preimages(candidate, previous, changes):
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
                      face_descendants=entries, coverage=())
    _validate_shape(candidate, binding)
    return binding


def _publish_application_preimages(model, binding, coverage, checksum):
    if binding is not None:
        binding = replace(binding, model_id=model.model_id, revision=model.revision,
                          source_checksum=checksum, coverage=tuple(coverage))
        model._prepared_face_preimages_receipt = (binding, _binding_checksum(binding))


def _copy_current_preimages(source, target):
    try:
        binding = _current_receipt(source)
    except GeometryError:
        return
    if target.model_id != binding.model_id or to_dict(target)['checksum']['value'] != binding.source_checksum:
        raise GeometryError('prepared face provenance changed during detached copying')
    target._prepared_face_preimages_receipt = (binding, _binding_checksum(binding))
