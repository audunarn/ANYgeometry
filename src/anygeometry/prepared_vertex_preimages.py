"""Complete original/current vertex identity ancestry, without coordinate inference."""
from dataclasses import dataclass

from .definition_binding import definition_checksum
from .errors import GeometryError
from .identity import EntityHandle, ResolutionStatus
from .prepared_model_scope import query_prepared_model_scope, validate_prepared_model_scope_binding


@dataclass(frozen=True, slots=True)
class PreparedVertexPreimages:
    scope: object
    # Every original active vertex: (ID, resolution status, current vertex IDs).
    authored_resolutions: tuple
    # Every current active vertex: (ID, original active vertex IDs), possibly empty.
    current_to_authored: tuple
    without_authored_vertex: tuple
    authored_positions: tuple
    current_positions: tuple


def _check(callback):
    if callback is not None and callback('prepared vertex preimages'):
        raise GeometryError('prepared vertex preimages cancelled')


def query_prepared_vertex_preimages(model, *, expected_revision=None, cancellation_check=None):
    """Bind ALL current vertices to original active vertex identities.

    Owner handle resolution supplies ancestry, never XYZ proximity. ID tuples
    are sets in deterministic order, NOT edge traversal. An empty preimage
    means no original VERTEX ancestor: it does not exclude a source edge,
    member, attachment or feature reference. Positions are snapshots, not a
    coordinate equivalence or remapping certificate.
    """
    scope = query_prepared_model_scope(model, expected_revision=expected_revision)
    original = scope.authored_document
    current = scope.current_document
    originals = sorted(original['vertices'], key=lambda row: row['id'])
    currents = sorted(current['vertices'], key=lambda row: row['id'])
    reverse = {row['id']: [] for row in currents}
    resolutions = []
    # Snapshot all resolution truth before the first caller-controlled callback.
    for row in originals:
        result = model.resolve_handle(EntityHandle(model.model_id, 'vertex', row['id']))
        if result.status not in (ResolutionStatus.ACTIVE, ResolutionStatus.REPLACED, ResolutionStatus.DELETED):
            raise GeometryError('prepared vertex ancestry has an unresolved original vertex')
        leaves = tuple(sorted(handle.id for handle in result.resolved))
        if (len(set(leaves)) != len(leaves) or
                any(handle.kind != 'vertex' or handle.model_id != model.model_id for handle in result.resolved) or
                any(identifier not in reverse for identifier in leaves)):
            raise GeometryError('prepared vertex ancestry has invalid current descendants')
        resolutions.append((row['id'], str(result.status), leaves))
        for identifier in leaves:
            reverse[identifier].append(row['id'])
    receipt = PreparedVertexPreimages(scope, tuple(resolutions),
        tuple((identifier, tuple(ancestors)) for identifier, ancestors in sorted(reverse.items())),
        tuple(identifier for identifier, ancestors in sorted(reverse.items()) if not ancestors),
        tuple((row['id'], tuple(row['position'])) for row in originals),
        tuple((row['id'], tuple(row['position'])) for row in currents))
    _check(cancellation_check)
    validate_prepared_model_scope_binding(model, scope, cancellation_check=cancellation_check)
    return receipt


def validate_prepared_vertex_preimages_binding(model, receipt, *, cancellation_check=None):
    """Authenticate complete vertex ancestry and positions at the bound revision."""
    if type(receipt) is not PreparedVertexPreimages:
        raise GeometryError('prepared vertex ancestry needs owner preimages')
    try:
        signature = definition_checksum(receipt)
    except (TypeError, ValueError) as error:
        raise GeometryError('prepared vertex ancestry definition binding changed') from error
    expected = query_prepared_vertex_preimages(model, cancellation_check=cancellation_check)
    if definition_checksum(expected) != signature or definition_checksum(receipt) != signature:
        raise GeometryError('prepared vertex ancestry definition binding changed')
    validate_prepared_model_scope_binding(model, receipt.scope, cancellation_check=cancellation_check)
    if definition_checksum(receipt) != signature:
        raise GeometryError('prepared vertex ancestry definition binding changed')
