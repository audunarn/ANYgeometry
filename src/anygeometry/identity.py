"""Model-bound public identity and explicit replacement resolution.

The geometry model stores compact ``(kind, integer_id)`` keys internally.
``EntityHandle`` adds the model UUID only at API and package boundaries, which
prevents an otherwise valid local identifier from silently resolving against
the wrong model.  This module deliberately has no dependency on
``GeometryModel`` so it can also be used by serializers and downstream
packages without introducing an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import total_ordering
from operator import index
from typing import Final, Iterable, TypeAlias
from uuid import UUID

from .errors import GeometryError

__all__ = [
    "ENTITY_KINDS",
    "GEOMETRY_ENTITY_KINDS",
    "STRUCTURAL_ENTITY_KINDS",
    "EntityHandle",
    "EntityKey",
    "Resolution",
    "ResolutionStatus",
    "canonical_model_id",
    "validate_entity_kind",
    "validate_local_id",
]


GEOMETRY_ENTITY_KINDS: Final[tuple[str, ...]] = ("vertex", "edge", "face")
STRUCTURAL_ENTITY_KINDS: Final[tuple[str, ...]] = (
    "part",
    "sheet",
    "face_use",
    "coedge",
    "member",
    "member_edge_use",
    "attachment",
    "junction",
)
ENTITY_KINDS: Final[tuple[str, ...]] = (
    *GEOMETRY_ENTITY_KINDS,
    *STRUCTURAL_ENTITY_KINDS,
)
_ENTITY_KIND_ORDER: Final[dict[str, int]] = {
    kind: position for position, kind in enumerate(ENTITY_KINDS)
}

EntityKey: TypeAlias = tuple[str, int]


def canonical_model_id(value: UUID | str) -> UUID:
    """Return a non-nil UUID suitable for persistent model identity."""

    if isinstance(value, UUID):
        result = value
    elif isinstance(value, str):
        try:
            result = UUID(value)
        except (AttributeError, ValueError) as error:
            raise GeometryError("model_id must be a valid UUID") from error
    else:
        raise GeometryError("model_id must be a UUID or UUID string")
    if result.int == 0:
        raise GeometryError("model_id cannot be the nil UUID")
    return result


def validate_entity_kind(kind: object) -> str:
    """Return a supported canonical entity kind or fail closed."""

    if not isinstance(kind, str) or kind not in _ENTITY_KIND_ORDER:
        raise GeometryError(f"unknown entity kind {kind!r}")
    return kind


def validate_local_id(value: object, *, name: str = "entity ID") -> int:
    """Return a positive integer identifier, rejecting booleans and floats."""

    if isinstance(value, bool):
        raise GeometryError(f"{name} must be a positive integer")
    try:
        result = index(value)  # Accept integer scalar types without NumPy coupling.
    except TypeError as error:
        raise GeometryError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise GeometryError(f"{name} must be a positive integer")
    return int(result)


@total_ordering
@dataclass(frozen=True, slots=True)
class EntityHandle:
    """A stable public entity reference bound to one persistent model UUID."""

    model_id: UUID | str
    kind: str
    id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", canonical_model_id(self.model_id))
        object.__setattr__(self, "kind", validate_entity_kind(self.kind))
        object.__setattr__(self, "id", validate_local_id(self.id))

    @property
    def key(self) -> EntityKey:
        """Return the compact key used by internal model stores."""

        return (self.kind, self.id)

    @classmethod
    def from_key(
        cls, model_id: UUID | str, key: tuple[object, object]
    ) -> EntityHandle:
        """Bind one compact internal key to a persistent model UUID."""

        if not isinstance(key, tuple) or len(key) != 2:
            raise GeometryError("entity key must be a (kind, ID) tuple")
        return cls(model_id, validate_entity_kind(key[0]), validate_local_id(key[1]))

    @property
    def sort_key(self) -> tuple[str, int, int]:
        """Return the canonical cross-platform ordering key."""

        assert isinstance(self.model_id, UUID)  # Normalized in ``__post_init__``.
        return (self.model_id.hex, _ENTITY_KIND_ORDER[self.kind], self.id)

    def belongs_to(self, model_id: UUID | str) -> bool:
        """Whether this handle belongs to ``model_id``."""

        return self.model_id == canonical_model_id(model_id)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EntityHandle):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __str__(self) -> str:  # pragma: no cover - display convenience
        return f"{self.model_id}:{self.kind}{self.id}"


class ResolutionStatus(StrEnum):
    """Complete, non-overlapping outcomes when resolving an entity handle."""

    ACTIVE = "active"
    REPLACED = "replaced"
    DELETED = "deleted"
    UNKNOWN = "unknown"
    WRONG_MODEL = "wrong_model"
    SUPPRESSED = "suppressed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Resolution:
    """Typed result of resolving one handle against one model.

    ``model_id`` identifies the resolving model.  It intentionally remains
    separate from ``requested.model_id`` so ``WRONG_MODEL`` is representable
    without an untyped exception or a sentinel handle.
    """

    model_id: UUID | str
    requested: EntityHandle
    status: ResolutionStatus | str
    resolved: tuple[EntityHandle, ...] = ()
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        owner = canonical_model_id(self.model_id)
        if not isinstance(self.requested, EntityHandle):
            raise GeometryError("resolution request must be an EntityHandle")
        try:
            status = ResolutionStatus(self.status)
        except (TypeError, ValueError) as error:
            raise GeometryError(f"unknown resolution status {self.status!r}") from error

        try:
            handles = tuple(self.resolved)
        except TypeError as error:
            raise GeometryError("resolved entities must be an iterable of handles") from error
        if any(not isinstance(handle, EntityHandle) for handle in handles):
            raise GeometryError("resolved entities must all be EntityHandle values")
        if len(set(handles)) != len(handles):
            raise GeometryError("resolution contains duplicate handles")
        handles = tuple(sorted(handles))

        if self.diagnostic is not None:
            if not isinstance(self.diagnostic, str) or not self.diagnostic.strip():
                raise GeometryError("resolution diagnostic must be a non-empty string")

        same_model = self.requested.model_id == owner
        if status is ResolutionStatus.WRONG_MODEL:
            if same_model:
                raise GeometryError("WRONG_MODEL requires a handle from another model")
            if handles:
                raise GeometryError("WRONG_MODEL resolution cannot contain entities")
        else:
            if not same_model:
                raise GeometryError(
                    "only WRONG_MODEL can resolve a handle from another model"
                )
            if any(handle.model_id != owner for handle in handles):
                raise GeometryError("resolved handle belongs to another model")
            if any(handle.kind != self.requested.kind for handle in handles):
                raise GeometryError("replacement resolution cannot change entity kind")

        if status is ResolutionStatus.ACTIVE:
            if not handles:
                handles = (self.requested,)
            if handles != (self.requested,):
                raise GeometryError("ACTIVE must resolve only to the requested handle")
        elif status is ResolutionStatus.REPLACED:
            if not handles:
                raise GeometryError("REPLACED requires at least one active descendant")
            if self.requested in handles:
                raise GeometryError("a replaced entity cannot resolve to itself")
        elif handles:
            raise GeometryError(f"{status.name} resolution cannot contain entities")

        object.__setattr__(self, "model_id", owner)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "resolved", handles)

    @classmethod
    def active(cls, requested: EntityHandle) -> Resolution:
        """Construct the successful active-entity case."""

        return cls(requested.model_id, requested, ResolutionStatus.ACTIVE)

    @classmethod
    def replaced(
        cls,
        requested: EntityHandle,
        descendants: Iterable[EntityHandle],
    ) -> Resolution:
        """Construct a deterministic replacement resolution."""

        return cls(
            requested.model_id,
            requested,
            ResolutionStatus.REPLACED,
            tuple(descendants),
        )

    @classmethod
    def terminal(
        cls,
        requested: EntityHandle,
        status: ResolutionStatus | str,
        *,
        model_id: UUID | str | None = None,
        diagnostic: str | None = None,
    ) -> Resolution:
        """Construct an outcome without active descendants."""

        try:
            made_status = ResolutionStatus(status)
        except (TypeError, ValueError) as error:
            raise GeometryError(f"unknown resolution status {status!r}") from error
        if made_status in (ResolutionStatus.ACTIVE, ResolutionStatus.REPLACED):
            raise GeometryError("terminal resolution status cannot be successful")
        return cls(
            requested.model_id if model_id is None else model_id,
            requested,
            made_status,
            diagnostic=diagnostic,
        )

    @property
    def is_resolved(self) -> bool:
        """Whether the request resolves to one or more active handles."""

        return self.status in (ResolutionStatus.ACTIVE, ResolutionStatus.REPLACED)

    def require(self) -> tuple[EntityHandle, ...]:
        """Return active descendants or raise a domain error for this status."""

        if self.is_resolved:
            return self.resolved
        detail = f": {self.diagnostic}" if self.diagnostic else ""
        raise GeometryError(
            f"cannot resolve {self.requested}: {self.status.value}{detail}"
        )
