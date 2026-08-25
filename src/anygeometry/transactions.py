"""Delta transactions and deterministic changed-region descriptions.

The public records in this module are intentionally independent of meshing or
analysis packages.  Consumers may subscribe to committed ``ChangeSet`` values
to invalidate their own caches without introducing a reverse dependency.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Hashable, Mapping

if TYPE_CHECKING:
    from .entities import EntityRef
    from .model import GeometryModel

__all__ = [
    "AABBChange",
    "ChangeSet",
    "EntityKey",
    "TopologyTransaction",
]

EntityKey = tuple[str, int]
Bounds = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class AABBChange:
    """Bounds before and after one committed entity change."""

    entity: EntityKey
    before: Bounds | None
    after: Bounds | None


@dataclass(frozen=True, slots=True)
class ChangeSet:
    """One successful outer transaction, in deterministic key order."""

    revision_before: int
    revision_after: int
    added: tuple[EntityKey, ...] = ()
    removed: tuple[EntityKey, ...] = ()
    modified: tuple[EntityKey, ...] = ()
    replacements: tuple[tuple["EntityRef", tuple["EntityRef", ...]], ...] = ()
    ownership_changes: tuple[EntityKey, ...] = ()
    member_changes: tuple[EntityKey, ...] = ()
    attachment_changes: tuple[EntityKey, ...] = ()
    group_changes: tuple[str, ...] = ()
    tag_changes: tuple[EntityKey, ...] = ()
    affected_aabbs: tuple[AABBChange, ...] = ()
    invalidated_caches: tuple[EntityKey, ...] = ()
    spatial_updates: tuple[EntityKey, ...] = ()
    feature_history_changed: bool = False
    document_settings_changed: bool = False

    @property
    def changed(self) -> tuple[EntityKey, ...]:
        """All changed keys without duplicates, in stable order."""

        return tuple(sorted({*self.added, *self.removed, *self.modified}))

    @property
    def is_empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.modified
            or self.replacements
            or self.ownership_changes
            or self.member_changes
            or self.attachment_changes
            or self.group_changes
            or self.tag_changes
            or self.feature_history_changed
            or self.document_settings_changed
        )


_MISSING = object()


@dataclass(slots=True)
class _TransactionJournal:
    """First-write journal owned by one outer model transaction."""

    revision_before: int
    depth: int = 0
    entity_before: dict[EntityKey, object] = field(default_factory=dict)
    owner_writes: set[EntityKey] = field(default_factory=set)
    mapping_before: dict[tuple[str, Hashable], object] = field(default_factory=dict)
    structural_before: dict[EntityKey, object] = field(default_factory=dict)
    versions_before: dict[EntityKey, object] = field(default_factory=dict)
    changed: set[EntityKey] = field(default_factory=set)
    ownership_changes: set[EntityKey] = field(default_factory=set)
    member_changes: set[EntityKey] = field(default_factory=set)
    attachment_changes: set[EntityKey] = field(default_factory=set)
    group_changes: set[str] = field(default_factory=set)
    tag_changes: set[EntityKey] = field(default_factory=set)
    invalidated_caches: set[EntityKey] = field(default_factory=set)
    arc_cache_before: dict[int, object] = field(default_factory=dict)
    edge_length_cache_before: dict[int, object] = field(default_factory=dict)
    spatial_index_before: object = _MISSING
    spatial_updates: set[EntityKey] = field(default_factory=set)
    bounds_before: dict[EntityKey, Bounds | None] = field(default_factory=dict)
    replacement_log_start: int = 0
    rolling_back: bool = False
    failure: BaseException | None = None

    def capture_entity(self, key: EntityKey, value: object = _MISSING) -> None:
        self.entity_before.setdefault(key, value)
        self.changed.add(key)

    def capture_mapping(
        self, namespace: str, key: Hashable, value: object = _MISSING
    ) -> None:
        self.mapping_before.setdefault((namespace, key), value)


class TopologyTransaction(AbstractContextManager["TopologyTransaction"]):
    """Nested model transaction.

    Instances are normally created with ``GeometryModel.transaction()``.  A
    nested context joins the outer journal; only the outermost successful exit
    validates, increments the model revision, and emits a ``ChangeSet``.
    """

    __slots__ = ("_model", "_entered")

    def __init__(self, model: "GeometryModel") -> None:
        self._model = model
        self._entered = False

    def __enter__(self) -> "TopologyTransaction":
        if self._entered:
            raise RuntimeError("a transaction context cannot be re-entered")
        self._entered = True
        self._model._enter_transaction()  # noqa: SLF001
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._entered:
            return False
        self._entered = False
        self._model._exit_transaction(exc)  # noqa: SLF001
        return False


ChangeHook = Callable[[ChangeSet], None]
