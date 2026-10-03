"""Complete persisted owner inputs bound to local intersection preparation.

This supplies visibility, not original/current reference remapping, geometric
membership, constraint completeness outside the owner document, or meshing
permission. Unknown external application references remain the consumer's scope.
"""
from dataclasses import dataclass
import json

from .errors import GeometryError
from .prepared_face_preimages import (
    PreparedFacePreimages, query_prepared_face_preimages,
    validate_prepared_face_preimages_binding,
)
from .serialization import _serialized_model_state


@dataclass(frozen=True, slots=True)
class PreparedModelScope:
    """Immutable complete original/current geometry documents and ancestry.

    Document properties parse fresh detached dictionaries on each access. The
    documents include ALL persisted owner entities/properties, not just those
    incident on one face: occurrences, members, attachments, isolated vertices,
    groups, tags, features, extensions, coordinates and replacement history.
    Original IDs/parameters retain original meaning. Current IDs/parameters
    retain current meaning. Their presence does not establish a valid remap.
    """
    face_preimages: PreparedFacePreimages
    authored_document_json: str
    current_document_json: str

    @property
    def authored_document(self):
        return json.loads(self.authored_document_json)

    @property
    def current_document(self):
        return json.loads(self.current_document_json)


def _encode(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _check(callback):
    if callback is not None and callback('prepared model scope'):
        raise GeometryError('prepared model scope cancelled')


def _validate(model, scope):
    if type(scope) is not PreparedModelScope:
        raise GeometryError('prepared model scope needs a PreparedModelScope binding')
    if (type(scope.face_preimages) is not PreparedFacePreimages or
            type(scope.authored_document_json) is not str or
            type(scope.current_document_json) is not str):
        raise GeometryError('prepared model scope needs plain immutable snapshot fields')
    validate_prepared_face_preimages_binding(model, scope.face_preimages)
    original = scope.face_preimages.authored_document_json
    if original is None:
        raise GeometryError('complete original model snapshot is unavailable')
    if scope.authored_document_json != original:
        raise GeometryError('prepared model scope original binding changed')
    if scope.current_document_json != _encode(_serialized_model_state(model)):
        raise GeometryError('prepared model scope current binding changed')


def query_prepared_model_scope(model, *, expected_revision=None, cancellation_check=None):
    """Return complete persisted inputs from a current ALL-face preparation.

    The original document was validated/captured prospectively before the first
    owner application, and is preserved through qualified preparation updates.
    Legacy receipts, ordinary clones and document reloads cannot invent it.
    Complete current content is checked against the preparation fingerprint.
    No geometric/reference interpretation, external-load snapshot, negative
    proof about external data, or automatic publication authority is supplied.
    """
    # Bind before caller-controlled callbacks, then guard against their edits.
    preimages = query_prepared_face_preimages(model, expected_revision=expected_revision)
    if preimages.authored_document_json is None:
        raise GeometryError('complete original model snapshot is unavailable')
    _check(cancellation_check)
    validate_prepared_face_preimages_binding(model, preimages)
    result = PreparedModelScope(preimages, preimages.authored_document_json,
                                _encode(_serialized_model_state(model)))
    _check(cancellation_check)
    _validate(model, result)
    return result


def validate_prepared_model_scope_binding(model, scope, *, cancellation_check=None):
    """Guard all persisted original/current input visibility before publication.

    Detects changed model/revision, raw same-revision edits, altered/omitted
    snapshot records and caller callback edits. Revalidate after consumer work.
    """
    _validate(model, scope)
    _check(cancellation_check)
    _validate(model, scope)
