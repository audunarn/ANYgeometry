"""The neutral structural-surface topology and geometry model.

Modelling is bottom-up and point-driven, which is the paradigm the mapped
mesher wants anyway:

    points  ->  lines between points  ->  faces bounded by line loops
                                      ->  beams carried on lines

Faces may carry explicit planar, cylindrical, conical or ruled surfaces, or a
four-boundary Coons patch.  Four mapped corners remain optional compatibility
metadata; they are not a restriction on neutral face topology.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from functools import wraps
from types import MappingProxyType
from typing import Dict, Hashable, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple
from uuid import UUID, uuid4

import numpy as np

from .curves import (
    Arc,
    ArcFrame,
    CurveShape,
    Spline,
    Straight,
    arc_frame,
    arc_tangent,
    sample_arc,
    sample_straight,
    sample_spline,
    spline_tangent,
    straight_tangent,
)
from .errors import GeometryError
from .entities import Edge, EntityRef, Face, OrientedEdge, Vertex
from .features import FeatureHistory, FeatureRegistry, RegenerationReport
from .surfaces import CoonsSurface, Plane, Surface, closest_uv
from .transactions import (
    AABBChange,
    ChangeHook,
    ChangeSet,
    EntityKey,
    TopologyTransaction,
    _MISSING,
    _TransactionJournal,
)
from .identity import (
    EntityHandle,
    Resolution,
    ResolutionStatus,
    canonical_model_id,
    validate_entity_kind,
    validate_local_id,
)
from .tolerance import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from .structural import (
    Attachment,
    AttachmentKind,
    AttachmentTargetKind,
    Coedge,
    FaceUse,
    Junction,
    JunctionMemberUse,
    Member,
    MemberEdgeUse,
    Orientation,
    ParameterRange,
    Part,
    Sheet,
    SheetTopologyPolicy,
    replace_member_edge_use,
    validate_structural_topology,
)

# GeometryError is re-exported for temporary compatibility imports.
__all__ = ["GeometryError", "GeometryModel"]


def _rotate_about_axis(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate a point about an arbitrary axis (Rodrigues' formula)."""

    offset = np.asarray(point, dtype=float) - origin
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return (
        origin
        + offset * cosine
        + np.cross(direction, offset) * sine
        + direction * float(direction @ offset) * (1.0 - cosine)
    )


def _records_equal(first: object, second: object) -> bool:
    """NumPy-safe equality for the small records captured by a delta journal."""

    if first is second:
        return True
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return bool(np.array_equal(first, second))
        except (TypeError, ValueError):
            return False
    if type(first) is not type(second):
        return False
    if is_dataclass(first) and not isinstance(first, type):
        return all(
            _records_equal(getattr(first, item.name), getattr(second, item.name))
            for item in fields(first)
        )
    if isinstance(first, Mapping):
        return (
            first.keys() == second.keys()  # type: ignore[union-attr]
            and all(_records_equal(first[key], second[key]) for key in first)  # type: ignore[index]
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(  # type: ignore[arg-type]
            _records_equal(left, right)
            for left, right in zip(first, second)  # type: ignore[arg-type]
        )
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _transactional(method):
    """Join the current model transaction or create one for a public edit."""

    @wraps(method)
    def wrapped(self: "GeometryModel", *args, **kwargs):
        with self.transaction():
            return method(self, *args, **kwargs)

    return wrapped


class _ReadOnlySetMapping(Mapping):
    """Live mapping view whose set values cannot mutate owner state."""

    __slots__ = ("_source",)

    def __init__(self, source: Mapping[Hashable, Set[object]]) -> None:
        self._source = source

    def __getitem__(self, key: Hashable) -> frozenset[object]:
        return frozenset(self._source[key])

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)


class GeometryModel:
    """A container of vertices, edges and faces with persistent IDs."""

    def __init__(
        self,
        *,
        model_id: UUID | str | None = None,
        tolerance: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    ) -> None:
        self._model_id = canonical_model_id(uuid4() if model_id is None else model_id)
        if not isinstance(tolerance, TolerancePolicy):
            raise TypeError("tolerance must be a TolerancePolicy")
        self._tolerance = tolerance
        self._revision = 0
        self._units = "m"
        self._local_origin = np.zeros(3, dtype=float)
        self._local_origin.flags.writeable = False
        self._coordinate_transform: np.ndarray | None = None

        # Private stores are the only mutation authority. Public access is a
        # live read-only view over immutable entity records.
        self._vertices: Dict[int, Vertex] = {}
        self._edges: Dict[int, Edge] = {}
        self._faces: Dict[int, Face] = {}
        self.vertices: Mapping[int, Vertex] = MappingProxyType(self._vertices)
        self.edges: Mapping[int, Edge] = MappingProxyType(self._edges)
        self.faces: Mapping[int, Face] = MappingProxyType(self._faces)
        self._next_id: Dict[str, int] = {"vertex": 1, "edge": 1, "face": 1}
        self._arc_cache: Dict[int, Tuple[int, ArcFrame]] = {}
        self._edge_length_cache: Dict[int, Tuple[int, float]] = {}
        self._entity_versions: Dict[EntityKey, int] = {}
        self._vertex_edges: Dict[int, Set[int]] = {}
        self._edge_faces: Dict[int, Set[int]] = {}
        self._spatial_index = None
        self._parts: Dict[int, Part] = {}
        self._sheets: Dict[int, Sheet] = {}
        self._face_uses: Dict[int, FaceUse] = {}
        self._coedges: Dict[int, Coedge] = {}
        self._members: Dict[int, Member] = {}
        self._member_edge_uses: Dict[int, MemberEdgeUse] = {}
        self._attachments: Dict[int, Attachment] = {}
        self._junctions: Dict[int, Junction] = {}
        self.parts: Mapping[int, Part] = MappingProxyType(self._parts)
        self.sheets: Mapping[int, Sheet] = MappingProxyType(self._sheets)
        self.face_uses: Mapping[int, FaceUse] = MappingProxyType(self._face_uses)
        self.coedges: Mapping[int, Coedge] = MappingProxyType(self._coedges)
        self.members: Mapping[int, Member] = MappingProxyType(self._members)
        self.member_edge_uses: Mapping[int, MemberEdgeUse] = MappingProxyType(
            self._member_edge_uses
        )
        self.attachments: Mapping[int, Attachment] = MappingProxyType(
            self._attachments
        )
        self.junctions: Mapping[int, Junction] = MappingProxyType(self._junctions)
        self._next_structural_id: Dict[str, int] = {
            "part": 1,
            "sheet": 1,
            "face_use": 1,
            "coedge": 1,
            "member": 1,
            "member_edge_use": 1,
            "attachment": 1,
            "junction": 1,
        }
        self._edge_member_uses: Dict[int, Set[int]] = {}
        self._face_structural_uses: Dict[int, Set[int]] = {}
        self._transaction_journal: _TransactionJournal | None = None
        self._notifying_hooks = False
        self._last_change_set = ChangeSet(0, 0)
        self._change_hooks: List[ChangeHook] = []
        # What each removed entity was replaced by, so attributes attached to
        # it can follow.  Splitting a line that carries a load must not throw
        # the load away.
        self._replacements: List[Tuple[EntityRef, Tuple[EntityRef, ...]]] = []
        self._replacement_history: Dict[EntityRef, Tuple[EntityRef, ...]] = {}
        self._groups: Dict[str, Set[EntityRef]] = {}
        self._tags: Dict[EntityRef, Set[str]] = {}
        self._group_view = _ReadOnlySetMapping(self._groups)
        self._tag_view = _ReadOnlySetMapping(self._tags)
        # Persistent design intent is separate from the materialized topology,
        # but travels with its owner so a geometry document cannot silently
        # lose its editable feature tree.
        self._features = FeatureHistory()
        self._features._bind_owner(self)  # noqa: SLF001

    @property
    def model_id(self) -> UUID:
        """Stable document identity; it cannot change while handles exist."""

        return self._model_id

    @property
    def revision(self) -> int:
        """Monotonic committed document revision."""

        return self._revision

    @property
    def tolerance(self) -> TolerancePolicy:
        return self._tolerance

    @property
    def units(self) -> str:
        return self._units

    @property
    def local_origin(self) -> np.ndarray:
        return self._local_origin

    @property
    def coordinate_transform(self) -> np.ndarray | None:
        return self._coordinate_transform

    @property
    def features(self) -> FeatureHistory:
        """Owner-bound persistent feature history."""

        return self._features

    def _install_feature_history(self, history: FeatureHistory) -> None:
        """Install decoded history on an unpublished/staged model."""

        if not isinstance(history, FeatureHistory):
            raise TypeError("features must be a FeatureHistory")
        history.validate()
        history._bind_owner(self)  # noqa: SLF001
        self._features = history

    def _feature_history_will_change(self, history: FeatureHistory) -> None:
        """Guard the start of one owner-aware history edit."""

        if history is not self._features:
            raise GeometryError("feature history is not owned by this model")
        if self._transaction_journal is not None:
            raise GeometryError("feature history cannot change inside a topology transaction")
        if self._notifying_hooks:
            raise GeometryError("change hooks are read-only observers")

    def _feature_history_did_change(self, history: FeatureHistory) -> None:
        """Publish one successfully validated feature-history edit."""

        if history is not self._features:
            raise GeometryError("feature history is not owned by this model")
        history.validate()
        revision_before = self._revision
        self._revision += 1
        change_set = ChangeSet(
            revision_before,
            self._revision,
            feature_history_changed=True,
        )
        self._last_change_set = change_set
        self._notifying_hooks = True
        try:
            for hook in tuple(self._change_hooks):
                try:
                    hook(change_set)
                except Exception:
                    continue
        finally:
            self._notifying_hooks = False

    def set_document_settings(
        self,
        *,
        tolerance: TolerancePolicy | None = None,
        units: str | None = None,
        local_origin: Sequence[float] | None = None,
        coordinate_transform: object = _MISSING,
    ) -> None:
        """Atomically update revisioned coordinate/tolerance document state."""

        if self._transaction_journal is not None or self._notifying_hooks:
            raise GeometryError("document settings cannot change during a transaction or hook")
        made_tolerance = self._tolerance if tolerance is None else tolerance
        if not isinstance(made_tolerance, TolerancePolicy):
            raise TypeError("tolerance must be a TolerancePolicy")
        made_units = self._units if units is None else str(units)
        if not made_units or "\x00" in made_units:
            raise GeometryError("units must be a non-empty string without NUL")
        made_origin = (
            self._local_origin
            if local_origin is None
            else np.asarray(local_origin, dtype=float)
        )
        if made_origin.shape != (3,) or not np.all(np.isfinite(made_origin)):
            raise GeometryError("local_origin must be a finite 3-vector")
        made_origin = np.array(made_origin, dtype=float, copy=True)
        made_origin.flags.writeable = False
        if coordinate_transform is _MISSING:
            made_transform = self._coordinate_transform
        elif coordinate_transform is None:
            made_transform = None
        else:
            made_transform = np.asarray(coordinate_transform, dtype=float)
            if (
                made_transform.shape != (4, 4)
                or not np.all(np.isfinite(made_transform))
                or not np.allclose(made_transform[3], (0.0, 0.0, 0.0, 1.0))
                or abs(float(np.linalg.det(made_transform[:3, :3])))
                <= np.finfo(float).eps
            ):
                raise GeometryError("coordinate_transform must be a finite invertible affine 4x4 matrix")
            made_transform = np.array(made_transform, dtype=float, copy=True)
            made_transform.flags.writeable = False
        same_transform = (
            (made_transform is None and self._coordinate_transform is None)
            or (
                made_transform is not None
                and self._coordinate_transform is not None
                and np.array_equal(made_transform, self._coordinate_transform)
            )
        )
        if (
            made_tolerance == self._tolerance
            and made_units == self._units
            and np.array_equal(made_origin, self._local_origin)
            and same_transform
        ):
            return
        revision_before = self._revision
        self._tolerance = made_tolerance
        self._units = made_units
        self._local_origin = made_origin
        self._coordinate_transform = made_transform
        self._arc_cache.clear()
        self._edge_length_cache.clear()
        self._spatial_index = None
        self._revision += 1
        change_set = ChangeSet(
            revision_before,
            self._revision,
            document_settings_changed=True,
        )
        self._last_change_set = change_set
        self._notifying_hooks = True
        try:
            for hook in tuple(self._change_hooks):
                try:
                    hook(change_set)
                except Exception:
                    continue
        finally:
            self._notifying_hooks = False

    @property
    def groups(self) -> Mapping[str, frozenset[EntityRef]]:
        """Live read-only semantic groups."""

        return self._group_view  # type: ignore[return-value]

    @property
    def tags(self) -> Mapping[EntityRef, frozenset[str]]:
        """Live read-only entity tags."""

        return self._tag_view  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # delta transactions and changed-region state
    # ------------------------------------------------------------------
    def transaction(self) -> TopologyTransaction:
        """Return a nested delta transaction context for public mutation."""

        return TopologyTransaction(self)

    def handle(self, kind: str, identifier: int) -> EntityHandle:
        """Return a model-bound public handle after local existence checking."""

        key = (validate_entity_kind(kind), validate_local_id(identifier))
        if not self._contains_entity(*key):
            raise GeometryError(f"no {kind} {identifier}")
        return EntityHandle(self.model_id, key[0], key[1])

    def resolve_handle(self, handle: EntityHandle) -> Resolution:
        """Resolve a public handle with an explicit status."""

        if not isinstance(handle, EntityHandle):
            raise TypeError("resolve_handle needs an EntityHandle")
        if handle.model_id != self.model_id:
            return Resolution.terminal(
                handle,
                ResolutionStatus.WRONG_MODEL,
                model_id=self.model_id,
                diagnostic="handle belongs to another geometry model",
            )
        if handle.kind in ("vertex", "edge", "face"):
            local = EntityRef(handle.kind, handle.id)  # type: ignore[arg-type]
            current = self.resolve_ref(local)
            if self._contains_entity(handle.kind, handle.id):
                return Resolution.active(handle)
            if current:
                return Resolution.replaced(
                    handle,
                    (
                        EntityHandle(self.model_id, item.kind, item.id)
                        for item in current
                    ),
                )
            if local in self._replacement_history:
                return Resolution.terminal(handle, ResolutionStatus.DELETED)
        elif self._contains_entity(handle.kind, handle.id):
            return Resolution.active(handle)
        elif (
            handle.kind in self._next_structural_id
            and handle.id < self._next_structural_id[handle.kind]
        ):
            return Resolution.terminal(handle, ResolutionStatus.DELETED)
        return Resolution.terminal(handle, ResolutionStatus.UNKNOWN)

    @property
    def last_change_set(self) -> ChangeSet:
        return self._last_change_set

    def add_change_hook(self, hook: ChangeHook) -> None:
        """Subscribe to committed changes without creating a package dependency."""

        if not callable(hook):
            raise TypeError("a change hook must be callable")
        if hook not in self._change_hooks:
            self._change_hooks.append(hook)

    def remove_change_hook(self, hook: ChangeHook) -> None:
        try:
            self._change_hooks.remove(hook)
        except ValueError:
            pass

    def _enter_transaction(self) -> None:
        if self._notifying_hooks:
            raise GeometryError("change hooks are read-only observers")
        journal = self._transaction_journal
        if journal is None:
            journal = _TransactionJournal(
                revision_before=self.revision,
                replacement_log_start=len(self._replacements),
            )
            self._transaction_journal = journal
        journal.depth += 1

    def _exit_transaction(self, error: BaseException | None) -> None:
        journal = self._transaction_journal
        if journal is None or journal.depth <= 0:
            raise RuntimeError("transaction exit without a matching entry")
        journal.depth -= 1
        if journal.depth:
            if error is not None:
                # The exception continues through the outer context, which
                # owns rollback.  No nested savepoint is implied.
                journal.failure = error
                return
            return
        try:
            failure = error if error is not None else journal.failure
            if failure is not None:
                self._rollback_transaction(journal)
                if error is None:
                    raise GeometryError(
                        "transaction cannot commit after a nested edit failed"
                    ) from failure
                return
            try:
                self._capture_direct_mutations(journal)
                # Every public edit, including a pure addition, must publish a
                # valid committed state.  Standalone vertices and edges are
                # valid neutral geometry; new faces additionally receive the
                # same changed-closure trim/surface checks as edited faces.
                validates_commit = bool(journal.entity_before) or bool(
                    journal.structural_before
                    or journal.group_changes
                    or journal.tag_changes
                    or journal.ownership_changes
                    or journal.member_changes
                    or journal.attachment_changes
                )
                if validates_commit:
                    problems = self._validate_incremental(journal)
                    if problems:
                        raise GeometryError("; ".join(problems))
                self._commit_transaction(journal)
            except BaseException:
                self._rollback_transaction(journal)
                raise
        finally:
            self._transaction_journal = None

    def _entity_store(self, kind: str) -> Dict[int, object]:
        try:
            return {
                "vertex": self._vertices,
                "edge": self._edges,
                "face": self._faces,
            }[kind]
        except KeyError:
            raise GeometryError(f"unknown entity kind {kind!r}") from None

    def _contains_entity(self, kind: str, identifier: int) -> bool:
        """Check one compact key without materializing the whole key set."""

        if kind in ("vertex", "edge", "face"):
            return int(identifier) in self._entity_store(kind)
        try:
            return int(identifier) in self._structural_store(kind)
        except KeyError:
            return False

    def _entity_bounds(self, key: EntityKey) -> tuple[float, ...] | None:
        kind, identifier = key
        if kind == "vertex":
            vertex = self._vertices.get(identifier)
            if vertex is None:
                return None
            point = np.asarray(vertex.position, dtype=float)
            return (*point, *point)
        if kind == "edge":
            edge = self._edges.get(identifier)
            if edge is None:
                return None
            if isinstance(edge.curve, Arc):
                frame = self._arc_frame(edge)
                angles = [0.0, frame.sweep]
                for coordinate in range(3):
                    phase = float(
                        np.arctan2(frame.e2[coordinate], frame.e1[coordinate])
                    )
                    for candidate in (phase, phase + np.pi):
                        if self._angle_on_arc_sweep(candidate, frame.sweep):
                            angles.append(candidate)
                samples = np.asarray(
                    [
                        frame.center
                        + frame.radius
                        * (
                            np.cos(angle) * frame.e1
                            + np.sin(angle) * frame.e2
                        )
                        for angle in angles
                    ]
                )
            elif isinstance(edge.curve, Spline):
                # A Bezier curve is contained in the convex hull of its
                # control polygon, so this bound is analytic and conservative.
                samples = self._spline_points(edge)
            else:
                samples = np.asarray(
                    (
                        self._vertices[edge.start].position,
                        self._vertices[edge.end].position,
                    )
                )
            lower, upper = np.min(samples, axis=0), np.max(samples, axis=0)
            return (*lower, *upper)
        if kind == "face":
            face = self._faces.get(identifier)
            if face is None:
                return None
            bounds = [
                self._entity_bounds(("edge", item.edge))
                for loop in (face.loop,) + face.holes
                for item in loop
                if item.edge in self._edges
            ]
            if not bounds:
                return None
            lower = np.min(np.asarray([item[:3] for item in bounds]), axis=0)
            upper = np.max(np.asarray([item[3:] for item in bounds]), axis=0)
            return (*lower, *upper)
        return None

    @staticmethod
    def _angle_on_arc_sweep(angle: float, sweep: float) -> bool:
        period = 2.0 * np.pi
        if sweep >= 0.0:
            delta = float(angle % period)
            return delta <= sweep + 1.0e-14
        delta = float((-angle) % period)
        return delta <= -sweep + 1.0e-14

    def _spatial(self):
        """Return the lazily materialized deterministic entity AABB tree."""

        if self._spatial_index is None:
            from .spatial import AABB, AABBTree

            items = []
            for key in sorted(self.entity_keys()):
                bounds = self._entity_bounds(key)
                if bounds is not None:
                    items.append((key, AABB(bounds[:3], bounds[3:])))
            self._spatial_index = AABBTree(items)
        return self._spatial_index

    def spatial_candidates(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        kinds: Iterable[str] | None = None,
    ) -> tuple[EntityKey, ...]:
        """Return stable candidate keys intersecting a world-space box."""

        from .spatial import AABB

        allowed = None if kinds is None else frozenset(str(item) for item in kinds)
        region = AABB(tuple(lower), tuple(upper))
        journal = self._transaction_journal
        if journal is not None and journal.changed:
            # Queries made by a compound edit must observe provisional owner
            # writes.  Use a changed-key overlay on the committed tree: stale
            # hits are rechecked against current exact bounds and provisional
            # additions/moves are tested explicitly.  This remains O(log n+k+c)
            # for a local edit with c changed records.
            committed = set(self._spatial().query(region).keys)
            changed = set(journal.changed) | set(journal.spatial_updates)
            committed.difference_update(changed)
            for key in changed:
                bounds = self._entity_bounds(key)
                if bounds is not None and region.intersects(
                    AABB(bounds[:3], bounds[3:])
                ):
                    committed.add(key)
            made = tuple(sorted(committed))
        else:
            made = self._spatial().query(region).keys
        return tuple(
            key for key in made if allowed is None or key[0] in allowed
        )

    def strict_audit(self, *, policy=None):
        """Run deterministic, fail-closed full-model qualification."""

        from .strict_audit import strict_audit

        return strict_audit(self, policy=policy)

    def _capture_entity(self, kind: str, identifier: int) -> None:
        journal = self._transaction_journal
        if journal is None or journal.rolling_back:
            return
        key = (kind, int(identifier))
        store = self._entity_store(kind)
        current = store.get(int(identifier), _MISSING)
        if key not in journal.entity_before:
            journal.bounds_before[key] = self._entity_bounds(key)
            journal.versions_before[key] = self._entity_versions.get(key, _MISSING)
            captured = _MISSING if current is _MISSING else deepcopy(current)
            journal.capture_entity(key, captured)
        else:
            journal.changed.add(key)
        journal.invalidated_caches.add(key)

    def _detach_entity(self, kind: str, value: object) -> None:
        if kind == "edge":
            edge = value
            assert isinstance(edge, Edge)
            for vertex_id in {edge.start, edge.end, *(
                (edge.curve.via_vertex,)
                if isinstance(edge.curve, Arc)
                else edge.curve.control_vertices
                if isinstance(edge.curve, Spline)
                else ()
            )}:
                uses = self._vertex_edges.get(vertex_id)
                if uses is not None:
                    uses.discard(edge.id)
                    if not uses:
                        self._vertex_edges.pop(vertex_id, None)
        elif kind == "face":
            face = value
            assert isinstance(face, Face)
            for edge_id in {item.edge for loop in (face.loop,) + face.holes for item in loop}:
                uses = self._edge_faces.get(edge_id)
                if uses is not None:
                    uses.discard(face.id)
                    if not uses:
                        self._edge_faces.pop(edge_id, None)

    def _attach_entity(self, kind: str, value: object) -> None:
        if kind == "edge":
            edge = value
            assert isinstance(edge, Edge)
            vertices = {edge.start, edge.end}
            if isinstance(edge.curve, Arc):
                vertices.add(edge.curve.via_vertex)
            elif isinstance(edge.curve, Spline):
                vertices.update(edge.curve.control_vertices)
            for vertex_id in vertices:
                self._vertex_edges.setdefault(vertex_id, set()).add(edge.id)
        elif kind == "face":
            face = value
            assert isinstance(face, Face)
            for edge_id in {item.edge for loop in (face.loop,) + face.holes for item in loop}:
                self._edge_faces.setdefault(edge_id, set()).add(face.id)

    def _rebuild_incidence(self) -> None:
        """Rebuild reverse incidence after trusted document materialization."""

        self._vertex_edges.clear()
        self._edge_faces.clear()
        for edge in self._edges.values():
            self._attach_entity("edge", edge)
        for face in self._faces.values():
            self._attach_entity("face", face)

    def _put_entity(self, kind: str, value: Vertex | Edge | Face) -> None:
        identifier = int(value.id)
        self._capture_entity(kind, identifier)
        if self._transaction_journal is not None:
            self._transaction_journal.owner_writes.add((kind, identifier))
        store = self._entity_store(kind)
        previous = store.get(identifier)
        if previous is not None:
            self._detach_entity(kind, previous)
        store[identifier] = value
        self._attach_entity(kind, value)
        if kind == "face" and isinstance(value, Face):
            self._synchronize_face_uses(value)
        key = (kind, identifier)
        self._entity_versions[key] = self._entity_versions.get(key, 0) + 1
        if kind == "vertex":
            for edge_id in self._vertex_edges.get(identifier, ()):
                self._arc_cache.pop(edge_id, None)
                self._edge_length_cache.pop(edge_id, None)
                if self._transaction_journal is not None:
                    self._transaction_journal.invalidated_caches.add(("edge", edge_id))
        elif kind == "edge":
            self._arc_cache.pop(identifier, None)
            self._edge_length_cache.pop(identifier, None)

    def _capture_direct_mutations(self, journal: _TransactionJournal) -> None:
        """Discover legacy direct field/store writes before commit.

        Core operations are migrating to ``_put_entity``.  During that
        migration, older qualified edit code still assigns dataclass fields
        directly.  Comparing the first-write baseline lets the transaction
        maintain incidence and rollback correctness without a full-model
        snapshot; only records already touched by the operation are checked.
        """

        for key, before in tuple(journal.entity_before.items()):
            if before is _MISSING or key in journal.owner_writes:
                continue
            after = self._entity_store(key[0]).get(key[1], _MISSING)
            if after is _MISSING:
                continue
            changed = not _records_equal(before, after)
            if changed:
                self._detach_entity(key[0], before)
                self._attach_entity(key[0], after)
                journal.changed.add(key)

    def _delete_entity(self, kind: str, identifier: int) -> object:
        store = self._entity_store(kind)
        try:
            previous = store[int(identifier)]
        except KeyError:
            raise GeometryError(f"no {kind} {identifier}") from None
        self._capture_entity(kind, int(identifier))
        if self._transaction_journal is not None:
            self._transaction_journal.owner_writes.add((kind, int(identifier)))
        self._detach_entity(kind, previous)
        del store[int(identifier)]
        key = (kind, int(identifier))
        self._entity_versions[key] = self._entity_versions.get(key, 0) + 1
        self._arc_cache.pop(int(identifier), None)
        self._edge_length_cache.pop(int(identifier), None)
        return previous

    def _set_entity_unjournalled(
        self, kind: str, identifier: int, value: object
    ) -> None:
        store = self._entity_store(kind)
        current = store.pop(identifier, None)
        if current is not None:
            self._detach_entity(kind, current)
        if value is not _MISSING:
            store[identifier] = value
            self._attach_entity(kind, value)
        self._arc_cache.pop(identifier, None)
        self._edge_length_cache.pop(identifier, None)

    def _capture_mapping(self, namespace: str, key: object, value: object) -> None:
        journal = self._transaction_journal
        if journal is not None and not journal.rolling_back:
            journal.capture_mapping(namespace, key, value)

    def _rollback_transaction(self, journal: _TransactionJournal) -> None:
        journal.rolling_back = True
        # Remove current changed records from dependency-rich to foundational
        # order, then reattach originals in the inverse order.
        for kind in ("face", "edge", "vertex"):
            for current_kind, identifier in sorted(journal.entity_before, reverse=True):
                if current_kind == kind:
                    self._set_entity_unjournalled(kind, identifier, _MISSING)
        for kind in ("vertex", "edge", "face"):
            for (current_kind, identifier), original in sorted(journal.entity_before.items()):
                if current_kind == kind and original is not _MISSING:
                    self._set_entity_unjournalled(kind, identifier, original)
        for key, original in journal.versions_before.items():
            if original is _MISSING:
                self._entity_versions.pop(key, None)
            else:
                self._entity_versions[key] = int(original)

        for (namespace, key), original in reversed(tuple(journal.mapping_before.items())):
            mapping = {
                "group": self._groups,
                "tag": self._tags,
                "replacement": self._replacement_history,
            }.get(namespace)
            if mapping is None:
                continue
            if original is _MISSING:
                mapping.pop(key, None)
            else:
                if isinstance(original, (set, dict, list)):
                    original = deepcopy(original)
                mapping[key] = original
        for (kind, identifier), original in reversed(
            tuple(journal.structural_before.items())
        ):
            store = self._structural_store(kind)
            if original is _MISSING:
                store.pop(identifier, None)
            else:
                store[identifier] = original
        self._rebuild_member_incidence()
        del self._replacements[journal.replacement_log_start :]
        # Values may have been queried while provisional geometry was live.
        # Rollback is exceptional, so discard the small derived caches and the
        # lazily rebuilt tree instead of attempting a fragile inverse update.
        self._arc_cache.clear()
        self._edge_length_cache.clear()
        self._spatial_index = None
        # High-water marks deliberately remain at their highest allocation.
        self._revision = journal.revision_before
        journal.rolling_back = False

    def _validate_incremental(self, journal: _TransactionJournal) -> tuple[str, ...]:
        keys = set(journal.changed) | set(journal.invalidated_caches)
        for kind, identifier in tuple(keys):
            if kind == "vertex":
                keys.update(("edge", edge) for edge in self._vertex_edges.get(identifier, ()))
            if kind == "edge":
                keys.update(("face", face) for face in self._edge_faces.get(identifier, ()))
        for kind, identifier in tuple(keys):
            if kind == "edge":
                keys.update(("face", face) for face in self._edge_faces.get(identifier, ()))

        errors: List[str] = []
        for kind, identifier in sorted(keys):
            if kind == "vertex":
                vertex = self._vertices.get(identifier)
                if vertex is not None and (
                    vertex.id != identifier
                    or np.asarray(vertex.position).shape != (3,)
                    or not np.all(np.isfinite(vertex.position))
                ):
                    errors.append(f"vertex {identifier} has invalid coordinates or identity")
            elif kind == "edge":
                edge = self._edges.get(identifier)
                if edge is None:
                    continue
                if edge.id != identifier:
                    errors.append(f"edge key {identifier} does not match ID {edge.id}")
                    continue
                if edge.start not in self._vertices or edge.end not in self._vertices:
                    errors.append(f"edge {identifier} references a missing endpoint")
                    continue
                if edge.start == edge.end:
                    errors.append(f"edge {identifier} has coincident topology endpoints")
                    continue
                start = self._vertices[edge.start].position
                end = self._vertices[edge.end].position
                length = float(np.linalg.norm(end - start))
                if length <= self.tolerance.effective_length(length):
                    errors.append(f"edge {identifier} has zero geometric length")
                    continue
                if isinstance(edge.curve, Arc):
                    if edge.curve.via_vertex not in self._vertices:
                        errors.append(f"arc edge {identifier} references a missing via vertex")
                    else:
                        try:
                            self._arc_frame(edge)
                        except (ValueError, GeometryError) as exc:
                            errors.append(f"arc edge {identifier} is invalid: {exc}")
                elif isinstance(edge.curve, Spline):
                    missing = [item for item in edge.curve.control_vertices if item not in self._vertices]
                    if missing:
                        errors.append(f"spline edge {identifier} has missing control vertices {missing}")
            elif kind == "face":
                face = self._faces.get(identifier)
                if face is None:
                    continue
                if face.id != identifier:
                    errors.append(f"face key {identifier} does not match ID {face.id}")
                    continue
                loops = (face.loop,) + face.holes
                for loop in loops:
                    if len(loop) < 3:
                        errors.append(f"face {identifier} has a loop with fewer than three edges")
                        continue
                    if any(item.edge not in self._edges for item in loop):
                        errors.append(f"face {identifier} references a missing edge")
                        continue
                    for current, following in zip(loop, loop[1:] + loop[:1]):
                        if self.oriented_end_vertex(current) != self.oriented_start_vertex(following):
                            errors.append(f"face {identifier} has a discontinuous loop")
                            break
                if not errors:
                    errors.extend(self._validate_face_geometry(identifier))

        for name in sorted(journal.group_changes):
            for reference in self._groups.get(name, ()):
                if not self._contains_entity(reference.kind, reference.id):
                    errors.append(f"group {name!r} references missing entity {reference}")
        for kind, identifier in sorted(journal.tag_changes):
            reference = EntityRef(kind, identifier)  # type: ignore[arg-type]
            if reference in self._tags and not self._contains_entity(kind, identifier):
                errors.append(f"tags reference missing entity {kind}{identifier}")
        if (
            journal.ownership_changes
            or journal.member_changes
            or journal.attachment_changes
            or any(
                key[0] in ("edge", "face")
                and self._entity_store(key[0]).get(key[1]) is None
                for key in journal.entity_before
            )
        ):
            errors.extend(self._validate_structural())
        return tuple(dict.fromkeys(errors))

    def _commit_transaction(self, journal: _TransactionJournal) -> None:
        from .spatial import AABB

        added: List[EntityKey] = []
        removed: List[EntityKey] = []
        modified: List[EntityKey] = []
        aabbs: List[AABBChange] = []
        for key, before in sorted(journal.entity_before.items()):
            after = self._entity_store(key[0]).get(key[1], _MISSING)
            if before is _MISSING and after is not _MISSING:
                added.append(key)
            elif before is not _MISSING and after is _MISSING:
                removed.append(key)
            elif before is not _MISSING and after is not _MISSING:
                modified.append(key)
            before_bounds = journal.bounds_before.get(key)
            after_bounds = self._entity_bounds(key)
            if before_bounds != after_bounds:
                aabbs.append(AABBChange(key, before_bounds, after_bounds))

        spatial_updates: Set[EntityKey] = set(journal.spatial_updates)
        if self._spatial_index is not None:
            # The lazy tree may have been materialized from provisional state
            # by a read-your-writes query.  Reconcile every locally touched
            # geometry key, including net-zero add/remove or move/move-back
            # edits whose committed before/after bounds compare equal.
            reconcile = {
                *(change.entity for change in aabbs),
                *journal.changed,
                *journal.spatial_updates,
            }
            for key in sorted(reconcile):
                after_bounds = self._entity_bounds(key)
                if after_bounds is None:
                    self._spatial_index.discard(key)
                else:
                    self._spatial_index.upsert(
                        key, AABB(after_bounds[:3], after_bounds[3:])
                    )
                spatial_updates.add(key)

        semantic_change = bool(
            journal.structural_before
            or journal.group_changes
            or journal.tag_changes
            or journal.ownership_changes
            or journal.member_changes
            or journal.attachment_changes
            or len(self._replacements) != journal.replacement_log_start
        )
        changed = bool(added or removed or modified or semantic_change)
        revision_after = self.revision + (1 if changed else 0)
        change_set = ChangeSet(
            revision_before=journal.revision_before,
            revision_after=revision_after,
            added=tuple(added),
            removed=tuple(removed),
            modified=tuple(modified),
            replacements=tuple(self._replacements[journal.replacement_log_start :]),
            ownership_changes=tuple(sorted(journal.ownership_changes)),
            member_changes=tuple(sorted(journal.member_changes)),
            attachment_changes=tuple(sorted(journal.attachment_changes)),
            group_changes=tuple(sorted(journal.group_changes)),
            tag_changes=tuple(sorted(journal.tag_changes)),
            affected_aabbs=tuple(aabbs),
            invalidated_caches=tuple(sorted(journal.invalidated_caches)),
            spatial_updates=tuple(sorted(spatial_updates)),
        )
        self._revision = revision_after
        self._last_change_set = change_set
        if changed:
            # Detach the committed journal before notifying downstream
            # observers.  Hooks are deliberately read-only: allowing one to
            # join the just-committed journal recursively corrupts revision
            # and ChangeSet boundaries.
            self._transaction_journal = None
            self._notifying_hooks = True
            try:
                for hook in tuple(self._change_hooks):
                    try:
                        hook(change_set)
                    except Exception:
                        # A consumer callback cannot make a valid commit
                        # partial or prevent other observers from running.
                        continue
            finally:
                self._notifying_hooks = False

    # ------------------------------------------------------------------
    # replacement log
    # ------------------------------------------------------------------
    def begin_replacement_log(self) -> None:
        """Start recording what replaces what, for the duration of an edit."""

        self._replacements = []

    def replacement_log(self) -> List[Tuple[EntityRef, Tuple[EntityRef, ...]]]:
        """Entities removed during the current edit, and what took their place."""

        return list(self._replacements)

    @_transactional
    def record_replacement(
        self, old: EntityRef, new: Sequence[EntityRef]
    ) -> None:
        """Note that one entity has been superseded by others."""

        replacements = tuple(new)
        if (
            old.kind not in self._next_id
            or old.id <= 0
            or old.id >= self._next_id[old.kind]
        ):
            raise GeometryError(f"replacement history references missing entity {old}")
        if self._contains_entity(old.kind, old.id):
            raise GeometryError(
                f"cannot record replacement for surviving entity {old}"
            )
        if any(reference.kind not in self._next_id for reference in replacements):
            raise GeometryError("replacement history contains an invalid entity kind")
        if any(reference.kind != old.kind for reference in replacements):
            raise GeometryError("replacement history cannot change entity kind")
        if old in replacements:
            raise GeometryError("an entity cannot replace itself")
        if old in self._replacement_history:
            raise GeometryError(f"replacement history already exists for {old}")
        for reference in replacements:
            if reference.id <= 0 or reference.id >= self._next_id[reference.kind]:
                raise GeometryError(
                    f"replacement history references missing entity {reference}"
                )
            if (
                not self._contains_entity(reference.kind, reference.id)
                and reference not in self._replacement_history
            ):
                raise GeometryError(
                    f"replacement history has an unresolved descendant {reference}"
                )

        def reaches_old(reference: EntityRef, seen: Set[EntityRef]) -> bool:
            if reference == old:
                return True
            if reference in seen:
                return False
            seen.add(reference)
            return any(
                reaches_old(descendant, seen)
                for descendant in self._replacement_history.get(reference, ())
            )

        if any(reaches_old(reference, set()) for reference in replacements):
            raise GeometryError(f"replacement history contains a cycle at {old}")
        self._capture_mapping(
            "replacement",
            old,
            self._replacement_history.get(old, _MISSING),
        )
        self._replacements.append((old, replacements))
        self._replacement_history[old] = replacements
        if old.kind == "face":
            self._replace_structural_face_ownership(
                old.id,
                tuple(reference.id for reference in replacements),
            )
        for name, members in self._groups.items():
            if old in members:
                self._capture_mapping("group", name, set(members))
                assert self._transaction_journal is not None
                self._transaction_journal.group_changes.add(name)
                members.discard(old)
                members.update(replacements)
        self._capture_mapping("tag", old, set(self._tags.get(old, ())) if old in self._tags else _MISSING)
        inherited = self._tags.pop(old, set())
        if self._transaction_journal is not None:
            self._transaction_journal.tag_changes.add((old.kind, old.id))
        for replacement in replacements:
            self._capture_mapping(
                "tag",
                replacement,
                set(self._tags.get(replacement, ())) if replacement in self._tags else _MISSING,
            )
            self._tags.setdefault(replacement, set()).update(inherited)
            if self._transaction_journal is not None:
                self._transaction_journal.tag_changes.add(
                    (replacement.kind, replacement.id)
                )

    @_transactional
    def record_replacements_atomic(
        self,
        entries: Iterable[Tuple[EntityRef, Sequence[EntityRef]]],
    ) -> None:
        """Append a complete replacement graph atomically.

        Regeneration may need to reconnect a retained historical graph and a
        set of old-materialization to new-materialization transitions in one
        operation.  Adding those arcs one at a time can temporarily leave a
        descendant unresolved even though the final batch is valid, so this
        method validates and commits the combined graph as a unit.
        """

        normalized: List[Tuple[EntityRef, Tuple[EntityRef, ...]]] = []
        supplied: Dict[EntityRef, Tuple[EntityRef, ...]] = {}
        for old, descendants in entries:
            made = tuple(descendants)
            previous = supplied.get(old)
            if previous is not None and previous != made:
                raise GeometryError(f"conflicting replacement entries for {old}")
            supplied[old] = made
        for old, descendants in supplied.items():
            current = self._replacement_history.get(old)
            if current is not None:
                if current != descendants:
                    raise GeometryError(
                        f"replacement history already exists for {old}"
                    )
                continue
            if self._contains_entity(old.kind, old.id):
                raise GeometryError(
                    f"cannot record replacement for surviving entity {old}"
                )
            if (
                old.kind not in self._next_id
                or old.id <= 0
                or old.id >= self._next_id[old.kind]
            ):
                raise GeometryError(
                    f"replacement history references missing entity {old}"
                )
            if any(item.kind != old.kind for item in descendants):
                raise GeometryError("replacement history cannot change entity kind")
            if old in descendants:
                raise GeometryError("an entity cannot replace itself")
            normalized.append((old, descendants))

        previous_history = dict(self._replacement_history)
        previous_log = list(self._replacements)
        previous_groups = {
            name: set(items) for name, items in self._groups.items()
        }
        previous_tags = {
            reference: set(values) for reference, values in self._tags.items()
        }
        try:
            for old, _descendants in normalized:
                self._capture_mapping(
                    "replacement",
                    old,
                    self._replacement_history.get(old, _MISSING),
                )
            self._replacement_history.update(normalized)
            errors = self._validate_replacement_history()
            if errors:
                raise GeometryError("; ".join(errors))
            for old, _descendants in normalized:
                if old.kind == "face" and self._face_structural_uses.get(old.id):
                    resolved = tuple(
                        reference.id
                        for reference in self.resolve_ref(old)
                        if reference.kind == "face"
                    )
                    self._replace_structural_face_ownership(old.id, resolved)
            for old, descendants in normalized:
                self._replacements.append((old, descendants))
                for name, members in self._groups.items():
                    if old in members:
                        self._capture_mapping("group", name, set(members))
                        if self._transaction_journal is not None:
                            self._transaction_journal.group_changes.add(name)
                        members.discard(old)
                        members.update(descendants)
                self._capture_mapping(
                    "tag",
                    old,
                    set(self._tags.get(old, ())) if old in self._tags else _MISSING,
                )
                inherited = self._tags.pop(old, set())
                if self._transaction_journal is not None:
                    self._transaction_journal.tag_changes.add((old.kind, old.id))
                for descendant in descendants:
                    self._capture_mapping(
                        "tag",
                        descendant,
                        set(self._tags.get(descendant, ()))
                        if descendant in self._tags
                        else _MISSING,
                    )
                    self._tags.setdefault(descendant, set()).update(inherited)
                    if self._transaction_journal is not None:
                        self._transaction_journal.tag_changes.add(
                            (descendant.kind, descendant.id)
                        )
        except Exception:
            self._replacement_history = previous_history
            self._replacements = previous_log
            self._groups.clear()
            self._groups.update(previous_groups)
            self._tags.clear()
            self._tags.update(previous_tags)
            raise

    def replacement_history(self) -> Dict[EntityRef, Tuple[EntityRef, ...]]:
        """Complete supersession map retained across edit transactions."""

        return dict(self._replacement_history)

    def resolve_ref(self, reference: EntityRef) -> Tuple[EntityRef, ...]:
        """Resolve a stale reference to its current surviving descendants."""

        pending = [reference]
        resolved: List[EntityRef] = []
        seen: Set[EntityRef] = set()
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            replacements = self._replacement_history.get(current)
            if replacements is None:
                if self._contains_entity(current.kind, current.id):
                    resolved.append(current)
            else:
                pending.extend(replacements)
        return tuple(resolved)

    @_transactional
    def add_to_group(self, name: str, references: Iterable[EntityRef]) -> None:
        """Add checked entities to a persistent semantic group."""

        key = str(name)
        self._capture_mapping(
            "group", key, set(self._groups[key]) if key in self._groups else _MISSING
        )
        group = self._groups.setdefault(key, set())
        for reference in references:
            self.entity_ref(reference.kind, reference.id)
            group.add(reference)
        if self._transaction_journal is not None:
            self._transaction_journal.group_changes.add(key)

    @_transactional
    def remove_group(self, name: str) -> None:
        """Remove a semantic group without exposing its mutable backing set."""

        key = str(name)
        if key not in self._groups:
            return
        self._capture_mapping("group", key, set(self._groups[key]))
        self._groups.pop(key)
        assert self._transaction_journal is not None
        self._transaction_journal.group_changes.add(key)

    @_transactional
    def untag(self, reference: EntityRef, *values: str) -> None:
        """Remove selected tags, or every tag when no values are supplied."""

        current = self._tags.get(reference)
        if current is None:
            return
        self._capture_mapping("tag", reference, set(current))
        if values:
            current.difference_update(str(value) for value in values)
            if not current:
                self._tags.pop(reference, None)
        else:
            self._tags.pop(reference, None)
        assert self._transaction_journal is not None
        self._transaction_journal.tag_changes.add((reference.kind, reference.id))

    def group(self, name: str, *, resolve: bool = True) -> Tuple[EntityRef, ...]:
        members = self._groups.get(str(name), set())
        if not resolve:
            return tuple(sorted(members, key=lambda ref: (ref.kind, ref.id)))
        current = {item for member in members for item in self.resolve_ref(member)}
        return tuple(sorted(current, key=lambda ref: (ref.kind, ref.id)))

    @_transactional
    def tag(self, reference: EntityRef, *values: str) -> None:
        self.entity_ref(reference.kind, reference.id)
        self._capture_mapping(
            "tag",
            reference,
            set(self._tags[reference]) if reference in self._tags else _MISSING,
        )
        self._tags.setdefault(reference, set()).update(str(value) for value in values)
        if self._transaction_journal is not None:
            self._transaction_journal.tag_changes.add((reference.kind, reference.id))

    def tags_for(self, reference: EntityRef) -> Tuple[str, ...]:
        return tuple(sorted(self._tags.get(reference, set())))

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    def _allocate(self, kind: str) -> int:
        entity_id = self._next_id[kind]
        store = {"vertex": self.vertices, "edge": self.edges, "face": self.faces}[kind]
        if entity_id in store:
            raise GeometryError(
                f"{kind} ID {entity_id} is already in use; the ID counter and "
                "the model have gone out of step"
            )
        self._next_id[kind] = entity_id + 1
        return entity_id

    def id_state(self) -> Dict[str, int]:
        """Return allocator high-water marks; gaps are intentional and valid."""

        return dict(self._next_id)

    def restore_id_state(self, state: Mapping[str, int]) -> None:
        """Raise allocator floors without permitting public identity rewind.

        The historical method name is retained for schema-1/2 loaders.  It no
        longer means that public undo may reuse a committed identifier.
        """

        normalized: Dict[str, int] = {}
        for kind in ("vertex", "edge", "face"):
            if kind not in state:
                raise GeometryError(f"missing {kind} ID counter")
            value = int(state[kind])
            if value < self._next_id[kind]:
                raise GeometryError(
                    f"cannot rewind committed {kind} ID high-water mark"
                )
            normalized[kind] = value
        self._next_id.update(normalized)

    def reserve_id_state(self, state: Mapping[str, int]) -> None:
        """Raise allocator floors without ever moving a counter backwards."""

        for kind in ("vertex", "edge", "face"):
            if kind not in state:
                raise GeometryError(f"missing {kind} ID counter")
            value = int(state[kind])
            if value < 1:
                raise GeometryError(f"{kind} ID counter must be positive")
            self._next_id[kind] = max(self._next_id[kind], value)

    def topology_snapshot(self) -> Dict[str, object]:
        """Return an explicit, expensive compatibility snapshot for undo.

        New kernel edits use delta transactions. This full-model form remains
        temporarily for feature/history compatibility while those callers
        migrate.
        """

        return {
            "vertices": dict(self.vertices),
            "edges": dict(self.edges),
            "faces": dict(self.faces),
            "vertex_state": {
                vertex_id: vertex.position.copy()
                for vertex_id, vertex in self.vertices.items()
            },
            "edge_state": {
                edge_id: (edge.start, edge.end, edge.curve)
                for edge_id, edge in self.edges.items()
            },
            "face_state": {
                face_id: (
                    face.loop,
                    face.corners,
                    deepcopy(face.metadata),
                    face.holes,
                    deepcopy(face.surface),
                )
                for face_id, face in self.faces.items()
            },
            "ids": dict(self._next_id),
            "groups": {name: set(members) for name, members in self.groups.items()},
            "tags": {reference: set(values) for reference, values in self.tags.items()},
            "replacement_history": dict(self._replacement_history),
            "replacements": list(self._replacements),
            "structural": {
                "parts": dict(self.parts),
                "sheets": dict(self.sheets),
                "face_uses": dict(self.face_uses),
                "coedges": dict(self.coedges),
                "members": dict(self.members),
                "member_edge_uses": dict(self.member_edge_uses),
                "attachments": dict(self.attachments),
                "junctions": dict(self.junctions),
            },
            "structural_ids": dict(self._next_structural_id),
        }

    def design_snapshot(self) -> Dict[str, object]:
        """Snapshot topology and persistent feature definitions for undo."""

        return {
            "topology": self.topology_snapshot(),
            "features": self.features.snapshot(),
        }

    def restore_topology(self, snapshot: Mapping[str, object]) -> None:
        """Atomically restore a compatibility snapshot and publish the change.

        Decoding happens in an unpublished staging model.  A malformed or
        stale snapshot therefore cannot partially clear the live topology.
        Allocator high-water marks remain monotonic, even when the snapshot is
        older than the current model.
        """

        if self._transaction_journal is not None or self._notifying_hooks:
            raise GeometryError(
                "topology snapshots cannot be restored inside a transaction or change hook"
            )
        candidate = GeometryModel(model_id=self.model_id, tolerance=self.tolerance)
        candidate._next_id.update(self._next_id)
        candidate._next_structural_id.update(self._next_structural_id)
        try:
            candidate._restore_topology_unchecked(snapshot)
        except GeometryError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise GeometryError(f"invalid topology snapshot: {error}") from error
        problems = (*candidate.validate_topology(), *candidate._validate_structural())
        if problems:
            raise GeometryError("invalid topology snapshot: " + "; ".join(problems))

        geometry_kinds = ("vertex", "edge", "face")
        old_records = {
            (kind, identifier): value
            for kind in geometry_kinds
            for identifier, value in self._entity_store(kind).items()
        }
        new_records = {
            (kind, identifier): value
            for kind in geometry_kinds
            for identifier, value in candidate._entity_store(kind).items()
        }
        old_keys, new_keys = set(old_records), set(new_records)
        added = tuple(sorted(new_keys - old_keys))
        removed = tuple(sorted(old_keys - new_keys))
        modified = tuple(
            sorted(
                key
                for key in old_keys & new_keys
                if not _records_equal(old_records[key], new_records[key])
            )
        )
        changed_geometry = set(added) | set(removed) | set(modified)
        bounds_before = {
            key: self._entity_bounds(key) for key in changed_geometry
        }

        structural_changes: Dict[str, Set[EntityKey]] = {
            "ownership": set(),
            "member": set(),
            "attachment": set(),
        }
        for kind in self._next_structural_id:
            before = self._structural_store(kind)
            after = candidate._structural_store(kind)
            category = (
                "member"
                if kind in ("member", "member_edge_use")
                else "attachment"
                if kind == "attachment"
                else "ownership"
            )
            for identifier in set(before) | set(after):
                if (
                    identifier not in before
                    or identifier not in after
                    or not _records_equal(before[identifier], after[identifier])
                ):
                    structural_changes[category].add((kind, identifier))

        group_changes = tuple(
            sorted(
                name
                for name in set(self._groups) | set(candidate._groups)
                if self._groups.get(name) != candidate._groups.get(name)
            )
        )
        tag_changes = tuple(
            sorted(
                ((reference.kind, reference.id) for reference in set(self._tags) | set(candidate._tags)
                 if self._tags.get(reference) != candidate._tags.get(reference)),
            )
        )
        semantic_change = bool(
            structural_changes["ownership"]
            or structural_changes["member"]
            or structural_changes["attachment"]
            or group_changes
            or tag_changes
            or self._replacement_history != candidate._replacement_history
        )
        if not changed_geometry and not semantic_change:
            return

        for kind in geometry_kinds:
            destination = self._entity_store(kind)
            destination.clear()
            destination.update(candidate._entity_store(kind))
        for kind in self._next_structural_id:
            destination = self._structural_store(kind)
            destination.clear()
            destination.update(candidate._structural_store(kind))
        self._groups.clear()
        self._groups.update(candidate._groups)
        self._tags.clear()
        self._tags.update(candidate._tags)
        self._replacement_history = dict(candidate._replacement_history)
        self._replacements = list(candidate._replacements)
        self._next_id.update(candidate._next_id)
        self._next_structural_id.update(candidate._next_structural_id)
        self._rebuild_incidence()
        self._rebuild_structural_incidence()
        self._arc_cache.clear()
        self._edge_length_cache.clear()
        self._spatial_index = None
        for key in changed_geometry:
            self._entity_versions[key] = self._entity_versions.get(key, 0) + 1

        aabbs = []
        for key in sorted(changed_geometry):
            before = bounds_before[key]
            after = self._entity_bounds(key)
            if before != after:
                aabbs.append(AABBChange(key, before, after))
        revision_before = self.revision
        self._revision += 1
        change_set = ChangeSet(
            revision_before,
            self._revision,
            added=added,
            removed=removed,
            modified=modified,
            ownership_changes=tuple(sorted(structural_changes["ownership"])),
            member_changes=tuple(sorted(structural_changes["member"])),
            attachment_changes=tuple(sorted(structural_changes["attachment"])),
            group_changes=group_changes,
            tag_changes=tag_changes,
            affected_aabbs=tuple(aabbs),
            invalidated_caches=tuple(sorted(changed_geometry)),
            spatial_updates=tuple(sorted(changed_geometry)),
        )
        self._last_change_set = change_set
        self._notifying_hooks = True
        try:
            for hook in tuple(self._change_hooks):
                try:
                    hook(change_set)
                except Exception:
                    continue
        finally:
            self._notifying_hooks = False

    def _restore_topology_unchecked(self, snapshot: Mapping[str, object]) -> None:
        """Decode into an unpublished staging model."""

        vertex_state = snapshot.get("vertex_state", {})
        self._vertices.clear()
        for vertex_id, vertex in snapshot["vertices"].items():  # type: ignore[union-attr]
            position = vertex_state.get(vertex_id, vertex.position)  # type: ignore[union-attr]
            self._vertices[int(vertex_id)] = replace(vertex, position=position)
        edge_state = snapshot.get("edge_state", {})
        self._edges.clear()
        for edge_id, edge in snapshot["edges"].items():  # type: ignore[union-attr]
            start, end, curve = edge_state.get(  # type: ignore[union-attr]
                edge_id, (edge.start, edge.end, edge.curve)
            )
            self._edges[int(edge_id)] = replace(
                edge, start=start, end=end, curve=deepcopy(curve)
            )
        face_state = snapshot.get("face_state", {})
        self._faces.clear()
        for face_id, face in snapshot["faces"].items():  # type: ignore[union-attr]
            loop, corners, metadata, holes, surface = face_state.get(  # type: ignore[union-attr]
                face_id,
                (face.loop, face.corners, face.metadata, face.holes, face.surface),
            )
            self._faces[int(face_id)] = replace(
                face,
                loop=loop,
                corners=corners,
                metadata=metadata,
                holes=holes,
                surface=deepcopy(surface),
            )
        saved_ids = dict(snapshot["ids"])  # type: ignore[arg-type]
        for kind in self._next_id:
            self._next_id[kind] = max(self._next_id[kind], int(saved_ids[kind]))
        self._groups.clear()
        self._groups.update(
            {name: set(members) for name, members in snapshot.get("groups", {}).items()}  # type: ignore[union-attr]
        )
        self._tags.clear()
        self._tags.update(
            {reference: set(values) for reference, values in snapshot.get("tags", {}).items()}  # type: ignore[union-attr]
        )
        self._replacement_history = dict(snapshot.get("replacement_history", {}))  # type: ignore[arg-type]
        self._replacements = list(snapshot.get("replacements", []))  # type: ignore[arg-type]
        structural = snapshot.get("structural", {})
        if isinstance(structural, Mapping):
            for name, store in (
                ("parts", self._parts),
                ("sheets", self._sheets),
                ("face_uses", self._face_uses),
                ("coedges", self._coedges),
                ("members", self._members),
                ("member_edge_uses", self._member_edge_uses),
                ("attachments", self._attachments),
                ("junctions", self._junctions),
            ):
                store.clear()
                store.update(structural.get(name, {}))  # type: ignore[arg-type]
        saved_structural_ids = snapshot.get("structural_ids", {})
        if isinstance(saved_structural_ids, Mapping):
            for kind in self._next_structural_id:
                if kind in saved_structural_ids:
                    self._next_structural_id[kind] = max(
                        self._next_structural_id[kind],
                        int(saved_structural_ids[kind]),
                    )
        self._arc_cache.clear()
        self._edge_length_cache.clear()
        self._rebuild_incidence()
        self._rebuild_member_incidence()
        self._spatial_index = None

    def restore_design(self, snapshot: Mapping[str, object]) -> None:
        """Restore topology and feature intent as one staged publication."""

        if self._transaction_journal is not None or self._notifying_hooks:
            raise GeometryError(
                "design snapshots cannot be restored inside a transaction or change hook"
            )
        if not isinstance(snapshot, Mapping):
            raise GeometryError("design snapshot must be a mapping")
        try:
            topology = snapshot["topology"]
            feature_snapshot = snapshot["features"]
        except KeyError as error:
            raise GeometryError(f"design snapshot is missing {error.args[0]!r}") from error
        if not isinstance(topology, Mapping) or not isinstance(feature_snapshot, Mapping):
            raise GeometryError("design snapshot topology/features must be mappings")

        staged_features = FeatureHistory()
        try:
            staged_features.restore(feature_snapshot)
            staged_features.validate()
        except (GeometryError, TypeError, ValueError, AttributeError) as error:
            raise GeometryError(f"invalid feature snapshot: {error}") from error

        feature_changed = not _records_equal(
            self.features.snapshot(), staged_features.snapshot()
        )
        events: list[ChangeSet] = []
        original_hooks = self._change_hooks
        self._change_hooks = [events.append]
        try:
            self.restore_topology(topology)
        finally:
            self._change_hooks = original_hooks
        # Keep the owner-bound history object stable so previously acquired
        # read handles cannot become a detached mutable design history.
        self._features._restore_unchecked(staged_features.snapshot())  # noqa: SLF001
        if events or feature_changed:
            if events:
                change_set = replace(events[0], feature_history_changed=feature_changed)
            else:
                revision_before = self.revision
                self._revision += 1
                change_set = ChangeSet(
                    revision_before,
                    self._revision,
                    feature_history_changed=True,
                )
            self._last_change_set = change_set
            # Feature intent is outside compact topology key space, but it is
            # still one document mutation.  Publish the already-committed
            # ChangeSet exactly once after both halves are installed.
            self._notifying_hooks = True
            try:
                for hook in tuple(original_hooks):
                    try:
                        hook(change_set)
                    except Exception:
                        continue
            finally:
                self._notifying_hooks = False

    def clone(self, *, include_features: bool = True) -> "GeometryModel":
        """Return a deep, independently mutable geometry copy."""

        from .serialization import from_dict, to_dict

        made = from_dict(to_dict(self, include_features=include_features))
        made._model_id = canonical_model_id(uuid4())
        return made

    def insert_model(
        self,
        source: "GeometryModel",
        *,
        matrix: Sequence[Sequence[float]] | None = None,
        group_prefix: str | None = None,
    ):
        """Insert a flattened topology copy with fresh destination IDs."""

        from .editing import insert_model

        return insert_model(
            self, source, matrix=matrix, group_prefix=group_prefix
        )

    def regenerate_features(
        self, registry: FeatureRegistry | None = None
    ) -> RegenerationReport:
        """Replay persistent features with neutral executors by default."""

        if registry is None:
            from .features import builtin_feature_registry

            registry = builtin_feature_registry()
        return self.features.regenerate(self, registry)

    def entity_keys(self) -> Set[Tuple[str, int]]:
        """Every entity in the model, as ``(kind, id)`` pairs."""

        return (
            {("vertex", key) for key in self.vertices}
            | {("edge", key) for key in self.edges}
            | {("face", key) for key in self.faces}
            | {("part", key) for key in self.parts}
            | {("sheet", key) for key in self.sheets}
            | {("face_use", key) for key in self.face_uses}
            | {("coedge", key) for key in self.coedges}
            | {("member", key) for key in self.members}
            | {("member_edge_use", key) for key in self.member_edge_uses}
            | {("attachment", key) for key in self.attachments}
            | {("junction", key) for key in self.junctions}
        )

    def _allocate_structural(self, kind: str) -> int:
        identifier = self._next_structural_id[kind]
        self._next_structural_id[kind] = identifier + 1
        return identifier

    def _structural_store(self, kind: str) -> Dict[int, object]:
        return {
            "part": self._parts,
            "sheet": self._sheets,
            "face_use": self._face_uses,
            "coedge": self._coedges,
            "member": self._members,
            "member_edge_use": self._member_edge_uses,
            "attachment": self._attachments,
            "junction": self._junctions,
        }[kind]

    def _put_structural(self, kind: str, value: object) -> None:
        identifier = int(value.id)  # type: ignore[attr-defined]
        store = self._structural_store(kind)
        key = (kind, identifier)
        journal = self._transaction_journal
        if journal is not None and key not in journal.structural_before:
            journal.structural_before[key] = store.get(identifier, _MISSING)
            if kind in ("member", "member_edge_use"):
                journal.member_changes.add(key)
            elif kind == "attachment":
                journal.attachment_changes.add(key)
            else:
                journal.ownership_changes.add(key)
        previous = store.get(identifier)
        if previous is not None:
            self._detach_structural_incidence(kind, previous)
        store[identifier] = value
        self._attach_structural_incidence(kind, value)

    def _delete_structural(self, kind: str, identifier: int) -> object:
        store = self._structural_store(kind)
        if identifier not in store:
            raise GeometryError(f"no {kind} {identifier}")
        journal = self._transaction_journal
        key = (kind, int(identifier))
        if journal is not None and key not in journal.structural_before:
            journal.structural_before[key] = store[identifier]
            if kind in ("member", "member_edge_use"):
                journal.member_changes.add(key)
            elif kind == "attachment":
                journal.attachment_changes.add(key)
            else:
                journal.ownership_changes.add(key)
        previous = store.pop(identifier)
        self._detach_structural_incidence(kind, previous)
        return previous

    def _detach_structural_incidence(self, kind: str, value: object) -> None:
        """Remove one immutable structural record from maintained reverse incidence."""

        if kind == "face_use":
            assert isinstance(value, FaceUse)
            uses = self._face_structural_uses.get(value.face_id)
            if uses is not None:
                uses.discard(value.id)
                if not uses:
                    self._face_structural_uses.pop(value.face_id, None)
        elif kind == "member_edge_use":
            assert isinstance(value, MemberEdgeUse)
            uses = self._edge_member_uses.get(value.edge_id)
            if uses is not None:
                uses.discard(value.id)
                if not uses:
                    self._edge_member_uses.pop(value.edge_id, None)

    def _attach_structural_incidence(self, kind: str, value: object) -> None:
        """Add one immutable structural record to maintained reverse incidence."""

        if kind == "face_use":
            assert isinstance(value, FaceUse)
            self._face_structural_uses.setdefault(value.face_id, set()).add(value.id)
        elif kind == "member_edge_use":
            assert isinstance(value, MemberEdgeUse)
            self._edge_member_uses.setdefault(value.edge_id, set()).add(value.id)

    def _validate_structural(self) -> tuple[str, ...]:
        errors = list(validate_structural_topology(
            parts=self.parts,
            sheets=self.sheets,
            face_uses=self.face_uses,
            coedges=self.coedges,
            members=self.members,
            member_edge_uses=self.member_edge_uses,
            attachments=self.attachments,
            junctions=self.junctions,
            edge_ids=tuple(self.edges),
            face_ids=tuple(self.faces),
            edge_vertices={
                edge.id: (edge.start, edge.end) for edge in self.edges.values()
            },
        ))
        for face_use in self.face_uses.values():
            face = self.faces.get(face_use.face_id)
            if face is None:
                continue
            expected = tuple(
                tuple(
                    (
                        item.edge,
                        Orientation.FORWARD
                        if item.forward
                        else Orientation.REVERSED,
                    )
                    for item in loop
                )
                for loop in (face.loop,) + face.holes
            )
            try:
                actual = tuple(
                    tuple(
                        (
                            self.coedges[coedge_id].edge_id,
                            self.coedges[coedge_id].orientation,
                        )
                        for coedge_id in loop
                    )
                    for loop in face_use.loops
                )
            except KeyError:
                continue
            if actual != expected:
                errors.append(
                    f"face use {face_use.id} coedges do not match face "
                    f"{face.id} trim loops"
                )
        expected_face_incidence: Dict[int, Set[int]] = {}
        for use in self.face_uses.values():
            expected_face_incidence.setdefault(use.face_id, set()).add(use.id)
        if expected_face_incidence != self._face_structural_uses:
            errors.append("face structural reverse incidence is stale")
        expected_member_incidence: Dict[int, Set[int]] = {}
        for use in self.member_edge_uses.values():
            expected_member_incidence.setdefault(use.edge_id, set()).add(use.id)
        if expected_member_incidence != self._edge_member_uses:
            errors.append("member-edge reverse incidence is stale")
        return tuple(sorted(set(errors)))

    def _rebuild_structural_incidence(self) -> None:
        self._face_structural_uses.clear()
        for use in self.face_uses.values():
            self._face_structural_uses.setdefault(use.face_id, set()).add(use.id)
        self._edge_member_uses.clear()
        for use in self.member_edge_uses.values():
            self._edge_member_uses.setdefault(use.edge_id, set()).add(use.id)

    def _rebuild_member_incidence(self) -> None:
        """Compatibility alias for internal callers predating face-use incidence."""

        self._rebuild_structural_incidence()

    def _synchronize_face_uses(self, face: Face) -> None:
        """Keep persistent coedges aligned with one replaced face trim."""

        for face_use_id in tuple(sorted(self._face_structural_uses.get(face.id, ()))):
            face_use = self.face_uses[face_use_id]
            available: Dict[int, List[int]] = {}
            for coedge_id in face_use.coedge_ids:
                coedge = self.coedges.get(coedge_id)
                if coedge is not None:
                    available.setdefault(coedge.edge_id, []).append(coedge_id)

            loops: List[tuple[int, ...]] = []
            reused: Set[int] = set()
            for loop in (face.loop,) + face.holes:
                made: List[int] = []
                for item in loop:
                    orientation = (
                        Orientation.FORWARD
                        if item.forward
                        else Orientation.REVERSED
                    )
                    candidates = [
                        identifier
                        for identifier in available.get(item.edge, ())
                        if identifier not in reused
                    ]
                    exact = [
                        identifier
                        for identifier in candidates
                        if self.coedges[identifier].orientation is orientation
                    ]
                    if len(exact) == 1:
                        coedge_id = exact[0]
                        reused.add(coedge_id)
                    elif len(candidates) == 1:
                        # Reversing an edge or a complete face changes only
                        # the use orientation/order.  The underlying edge ID
                        # still identifies the same persistent coedge
                        # unambiguously, so update that record in place rather
                        # than retiring its public identity.
                        coedge_id = candidates[0]
                        reused.add(coedge_id)
                        coedge = self.coedges[coedge_id]
                        if coedge.orientation is not orientation:
                            self._put_structural(
                                "coedge",
                                replace(coedge, orientation=orientation),
                            )
                    else:
                        coedge_id = self._allocate_structural("coedge")
                        self._put_structural(
                            "coedge",
                            Coedge(
                                coedge_id,
                                face_use.id,
                                item.edge,
                                orientation,
                            ),
                        )
                    made.append(coedge_id)
                loops.append(tuple(made))

            target = tuple(loops)
            if target != face_use.loops:
                self._put_structural(
                    "face_use", replace(face_use, loops=target)
                )
            for coedge_id in set(face_use.coedge_ids) - reused:
                if coedge_id in self.coedges and all(
                    coedge_id not in loop for loop in target
                ):
                    self._delete_structural("coedge", coedge_id)

    def _new_face_use(
        self,
        face_use_id: int,
        sheet_id: int,
        face_id: int,
        *,
        orientation: Orientation = Orientation.FORWARD,
        metadata: Mapping[str, object] | object | None = None,
    ) -> FaceUse:
        face = self._require_face(face_id)
        loops: List[tuple[int, ...]] = []
        for loop in (face.loop,) + face.holes:
            made: List[int] = []
            for item in loop:
                coedge_id = self._allocate_structural("coedge")
                self._put_structural(
                    "coedge",
                    Coedge(
                        coedge_id,
                        face_use_id,
                        item.edge,
                        Orientation.FORWARD
                        if item.forward
                        else Orientation.REVERSED,
                    ),
                )
                made.append(coedge_id)
            loops.append(tuple(made))
        return FaceUse(
            face_use_id,
            sheet_id,
            face_id,
            tuple(loops),
            orientation=orientation,
            metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
        )

    def _replace_structural_face_ownership(
        self, old_face_id: int, new_face_ids: Sequence[int]
    ) -> None:
        """Transfer a sheet face use across one geometry replacement event."""

        replacements = tuple(int(identifier) for identifier in new_face_ids)
        if len(set(replacements)) != len(replacements):
            raise GeometryError("face replacement contains duplicate descendants")
        for face_id in replacements:
            self._require_face(face_id)
        owners = tuple(
            self.face_uses[identifier]
            for identifier in sorted(
                self._face_structural_uses.get(old_face_id, ())
            )
        )
        if len(owners) > 1:
            raise GeometryError(
                f"face {old_face_id} has multiple structural owners"
            )
        attachments = sorted(
            attachment.id
            for attachment in self.attachments.values()
            if attachment.target_kind is AttachmentTargetKind.FACE
            and attachment.target_id == old_face_id
        )
        if attachments:
            raise GeometryError(
                f"cannot replace face {old_face_id}: attachments {attachments} "
                "require an explicit parameter remap"
            )
        if not owners:
            return
        conflicts = sorted(
            use_id
            for face_id in replacements
            for use_id in self._face_structural_uses.get(face_id, ())
        )
        if conflicts:
            raise GeometryError(
                "cannot transfer face ownership: replacement face(s) are "
                f"already owned by face uses {conflicts}"
            )
        for old_use in owners:
            sheet = self.sheets[old_use.sheet_id]
            replacement_use_ids: List[int] = []
            if replacements:
                first_face = replacements[0]
                self._put_structural(
                    "face_use", replace(old_use, face_id=first_face)
                )
                self._synchronize_face_uses(self._require_face(first_face))
                replacement_use_ids.append(old_use.id)
                for face_id in replacements[1:]:
                    use_id = self._allocate_structural("face_use")
                    made = self._new_face_use(
                        use_id,
                        sheet.id,
                        int(face_id),
                        orientation=old_use.orientation,
                        metadata=old_use.metadata,
                    )
                    self._put_structural("face_use", made)
                    replacement_use_ids.append(use_id)
            else:
                for coedge_id in old_use.coedge_ids:
                    self._delete_structural("coedge", coedge_id)
                self._delete_structural("face_use", old_use.id)

            retained = [
                identifier
                for identifier in sheet.face_use_ids
                if identifier != old_use.id
            ]
            retained.extend(replacement_use_ids)
            if retained:
                self._put_structural(
                    "sheet", replace(sheet, face_use_ids=tuple(retained))
                )
            else:
                part = self.parts[sheet.part_id]
                self._delete_structural("sheet", sheet.id)
                self._put_structural(
                    "part",
                    replace(
                        part,
                        sheet_ids=tuple(
                            value
                            for value in part.sheet_ids
                            if value != sheet.id
                        ),
                    ),
                )

    @_transactional
    def add_part(
        self,
        *,
        name: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> int:
        identifier = self._allocate_structural("part")
        self._put_structural(
            "part", Part(identifier, name=name, metadata=metadata or {})
        )
        return identifier

    @_transactional
    def add_member(
        self,
        edge_ids: Sequence[int | OrientedEdge],
        *,
        part_id: int | None = None,
        name: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> int:
        """Create one persistent physical member over an ordered edge chain."""

        if not edge_ids:
            raise GeometryError("a member needs at least one axis edge")
        if part_id is None:
            part_id = self.add_part()
        if part_id not in self.parts:
            raise GeometryError(f"no part {part_id}")
        member_id = self._stage_member(
            edge_ids,
            part_id=part_id,
            name=name,
            metadata=metadata,
        )
        part = self.parts[part_id]
        self._put_structural(
            "part",
            replace(part, member_ids=tuple((*part.member_ids, member_id))),
        )
        return member_id

    def _stage_member(
        self,
        edge_ids: Sequence[int | OrientedEdge],
        *,
        part_id: int,
        name: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> int:
        """Materialize one member without repeatedly copying its owning part."""

        if not edge_ids:
            raise GeometryError("a member needs at least one axis edge")
        oriented: List[tuple[int, Orientation]] = []
        previous_end: int | None = None
        for raw in edge_ids:
            if isinstance(raw, OrientedEdge):
                edge_id = raw.edge
                orientation = Orientation.FORWARD if raw.forward else Orientation.REVERSED
            else:
                edge_id = int(raw)
                edge = self._require_edge(edge_id)
                if previous_end is None or edge.start == previous_end:
                    orientation = Orientation.FORWARD
                elif edge.end == previous_end:
                    orientation = Orientation.REVERSED
                else:
                    raise GeometryError("member axis edges are not continuous")
            edge = self._require_edge(edge_id)
            start = edge.start if orientation is Orientation.FORWARD else edge.end
            end = edge.end if orientation is Orientation.FORWARD else edge.start
            if previous_end is not None and start != previous_end:
                raise GeometryError("member axis edges are not continuous")
            previous_end = end
            oriented.append((edge_id, orientation))

        member_id = self._allocate_structural("member")
        lengths = [self.edge_length(edge_id) for edge_id, _ in oriented]
        total = sum(lengths)
        if total <= 0.0:
            raise GeometryError("member axis has zero total length")
        uses: List[int] = []
        station = 0.0
        for index, ((edge_id, orientation), length) in enumerate(zip(oriented, lengths)):
            use_id = self._allocate_structural("member_edge_use")
            end = 1.0 if index == len(oriented) - 1 else station + length / total
            use = MemberEdgeUse(
                use_id,
                member_id,
                edge_id,
                ParameterRange(station, end),
                orientation,
            )
            self._put_structural("member_edge_use", use)
            uses.append(use_id)
            station = end
        self._put_structural(
            "member",
            Member(
                member_id,
                part_id,
                tuple(uses),
                name=name,
                metadata=metadata or {},
            ),
        )
        return member_id

    @_transactional
    def add_members(
        self,
        edge_chains: Iterable[Sequence[int | OrientedEdge]],
        *,
        part_id: int | None = None,
        names: Iterable[str] | None = None,
        metadata: Iterable[Mapping[str, object] | None] | None = None,
    ) -> List[int]:
        """Create many members with one part update and one outer validation.

        This is the construction path for large beam lattices.  It avoids the
        quadratic tuple copying and repeated whole-structure validation that
        would result from committing each member separately.
        """

        chains = tuple(tuple(chain) for chain in edge_chains)
        if not chains:
            return []
        made_names = (
            ("",) * len(chains)
            if names is None
            else tuple(str(value) for value in names)
        )
        made_metadata = (
            (None,) * len(chains) if metadata is None else tuple(metadata)
        )
        if len(made_names) != len(chains) or len(made_metadata) != len(chains):
            raise GeometryError("member names/metadata must match the number of chains")
        if part_id is None:
            part_id = self.add_part()
        if part_id not in self.parts:
            raise GeometryError(f"no part {part_id}")
        member_ids = [
            self._stage_member(
                chain,
                part_id=part_id,
                name=name,
                metadata=item_metadata,
            )
            for chain, name, item_metadata in zip(
                chains, made_names, made_metadata
            )
        ]
        part = self.parts[part_id]
        self._put_structural(
            "part",
            replace(part, member_ids=tuple((*part.member_ids, *member_ids))),
        )
        return member_ids

    @_transactional
    def add_sheet(
        self,
        face_ids: Sequence[int],
        *,
        part_id: int | None = None,
        name: str = "",
        policy: SheetTopologyPolicy = SheetTopologyPolicy(),
    ) -> int:
        """Create an oriented structural sheet and persistent loop coedges."""

        if not face_ids:
            raise GeometryError("a sheet needs at least one face")
        if part_id is None:
            part_id = self.add_part()
        if part_id not in self.parts:
            raise GeometryError(f"no part {part_id}")
        sheet_id = self._allocate_structural("sheet")
        use_ids: List[int] = []
        for face_id in face_ids:
            face = self._require_face(int(face_id))
            use_id = self._allocate_structural("face_use")
            loop_ids: List[tuple[int, ...]] = []
            for loop in (face.loop,) + face.holes:
                made: List[int] = []
                for item in loop:
                    coedge_id = self._allocate_structural("coedge")
                    self._put_structural(
                        "coedge",
                        Coedge(
                            coedge_id,
                            use_id,
                            item.edge,
                            Orientation.FORWARD if item.forward else Orientation.REVERSED,
                        ),
                    )
                    made.append(coedge_id)
                loop_ids.append(tuple(made))
            self._put_structural(
                "face_use", FaceUse(use_id, sheet_id, face.id, tuple(loop_ids))
            )
            use_ids.append(use_id)
        self._put_structural(
            "sheet", Sheet(sheet_id, part_id, tuple(use_ids), policy, name=name)
        )
        part = self.parts[part_id]
        self._put_structural(
            "part",
            replace(part, sheet_ids=tuple(sorted((*part.sheet_ids, sheet_id)))),
        )
        return sheet_id

    @_transactional
    def add_attachment(
        self,
        member_id: int,
        kind: AttachmentKind | str,
        target_kind: AttachmentTargetKind | str,
        target_id: int,
        member_range: ParameterRange,
        target_parameters: Sequence[ParameterRange],
    ) -> int:
        identifier = self._allocate_structural("attachment")
        self._put_structural(
            "attachment",
            Attachment(
                identifier,
                member_id,
                kind,
                target_kind,
                target_id,
                member_range,
                tuple(target_parameters),
            ),
        )
        return identifier

    @_transactional
    def add_junction(
        self,
        kind: str,
        member_uses: Sequence[JunctionMemberUse],
        *,
        sheet_ids: Sequence[int] = (),
        attachment_ids: Sequence[int] = (),
    ) -> int:
        identifier = self._allocate_structural("junction")
        self._put_structural(
            "junction",
            Junction(
                identifier,
                kind,
                tuple(member_uses),
                tuple(sheet_ids),
                tuple(attachment_ids),
            ),
        )
        return identifier

    @_transactional
    def remove_junction(self, junction_id: int) -> None:
        """Remove an explicit connection intent record."""

        if junction_id not in self.junctions:
            raise GeometryError(f"no junction {junction_id}")
        self._delete_structural("junction", junction_id)

    @_transactional
    def remove_attachment(self, attachment_id: int) -> None:
        """Remove an attachment that is not still used by a junction."""

        if attachment_id not in self.attachments:
            raise GeometryError(f"no attachment {attachment_id}")
        junctions = sorted(
            junction.id
            for junction in self.junctions.values()
            if attachment_id in junction.attachment_ids
        )
        if junctions:
            raise GeometryError(
                f"cannot remove attachment {attachment_id}: junction(s) "
                f"{junctions} still reference it"
            )
        self._delete_structural("attachment", attachment_id)

    @_transactional
    def remove_member(self, member_id: int) -> None:
        """Remove a member and its owned axis uses when externally unreferenced."""

        member = self.members.get(member_id)
        if member is None:
            raise GeometryError(f"no member {member_id}")
        attachments = sorted(
            attachment.id
            for attachment in self.attachments.values()
            if attachment.member_id == member_id
        )
        junctions = sorted(
            junction.id
            for junction in self.junctions.values()
            if member_id in junction.member_ids
        )
        if attachments or junctions:
            raise GeometryError(
                f"cannot remove member {member_id}: attachments {attachments} "
                f"or junctions {junctions} still reference it"
            )
        for use_id in member.edge_use_ids:
            self._delete_structural("member_edge_use", use_id)
        self._delete_structural("member", member_id)
        part = self.parts[member.part_id]
        self._put_structural(
            "part",
            replace(
                part,
                member_ids=tuple(
                    value for value in part.member_ids if value != member_id
                ),
            ),
        )

    @_transactional
    def remove_sheet(self, sheet_id: int) -> None:
        """Remove a sheet and its owned face uses/coedges when unreferenced."""

        sheet = self.sheets.get(sheet_id)
        if sheet is None:
            raise GeometryError(f"no sheet {sheet_id}")
        junctions = sorted(
            junction.id
            for junction in self.junctions.values()
            if sheet_id in junction.sheet_ids
        )
        if junctions:
            raise GeometryError(
                f"cannot remove sheet {sheet_id}: junction(s) {junctions} "
                "still reference it"
            )
        for face_use_id in sheet.face_use_ids:
            face_use = self.face_uses[face_use_id]
            for coedge_id in face_use.coedge_ids:
                self._delete_structural("coedge", coedge_id)
            self._delete_structural("face_use", face_use_id)
        self._delete_structural("sheet", sheet_id)
        part = self.parts[sheet.part_id]
        self._put_structural(
            "part",
            replace(
                part,
                sheet_ids=tuple(
                    value for value in part.sheet_ids if value != sheet_id
                ),
            ),
        )

    @_transactional
    def remove_part(self, part_id: int) -> None:
        """Remove an empty ownership boundary without cascading physical data."""

        part = self.parts.get(part_id)
        if part is None:
            raise GeometryError(f"no part {part_id}")
        if part.sheet_ids or part.member_ids:
            raise GeometryError(
                f"cannot remove non-empty part {part_id}: sheets "
                f"{list(part.sheet_ids)} or members {list(part.member_ids)} remain"
            )
        self._delete_structural("part", part_id)

    # ------------------------------------------------------------------
    # dependencies and removal
    # ------------------------------------------------------------------
    def edges_using_vertex(self, vertex_id: int) -> List[int]:
        """Edges that reference a vertex, as an end point or as an arc's via."""

        self._require_vertex(vertex_id)
        return sorted(self._vertex_edges.get(vertex_id, ()))

    def faces_using_edge(self, edge_id: int) -> List[int]:
        self._require_edge(edge_id)
        return sorted(self._edge_faces.get(edge_id, ()))

    @_transactional
    def remove_face(self, face_id: int, *, record: bool = True) -> None:
        self._require_face(face_id)
        structural_uses = sorted(self._face_structural_uses.get(face_id, ()))
        attachments = sorted(
            attachment.id
            for attachment in self.attachments.values()
            if attachment.target_kind is AttachmentTargetKind.FACE
            and attachment.target_id == face_id
        )
        # ``record=False`` is the internal retirement half of an atomic
        # replacement.  The enclosing operation creates the replacement
        # faces and calls ``record_replacement`` before commit, which moves
        # each persistent FaceUse to the descendants.  A public deletion
        # (``record=True``) must never leave either ownership or intent
        # dangling, and attachments are not yet geometrically remapped by a
        # face split, so they remain blocking in both modes.
        if (record and structural_uses) or attachments:
            raise GeometryError(
                f"cannot remove face {face_id}: structural face uses "
                f"{structural_uses} or attachments {attachments} still reference it"
            )
        self._delete_entity("face", face_id)
        if record:
            self.record_replacement(EntityRef("face", face_id), ())

    @_transactional
    def remove_edge(self, edge_id: int, *, record: bool = True) -> None:
        self._require_edge(edge_id)
        users = self.faces_using_edge(edge_id)
        if users:
            raise GeometryError(
                f"cannot remove edge {edge_id}: it bounds face(s) {sorted(users)}"
            )
        member_uses = sorted(self._edge_member_uses.get(edge_id, ()))
        attachments = sorted(
            attachment.id
            for attachment in self.attachments.values()
            if attachment.target_kind is AttachmentTargetKind.EDGE
            and attachment.target_id == edge_id
        )
        if member_uses or attachments:
            raise GeometryError(
                f"cannot remove edge {edge_id}: structural member uses "
                f"{member_uses} or attachments {attachments} still reference it"
            )
        self._delete_entity("edge", edge_id)
        self._arc_cache.pop(edge_id, None)
        if record:
            self.record_replacement(EntityRef("edge", edge_id), ())

    @_transactional
    def remove_vertex(self, vertex_id: int, *, record: bool = True) -> None:
        self._require_vertex(vertex_id)
        users = self.edges_using_vertex(vertex_id)
        if users:
            raise GeometryError(
                f"cannot remove point {vertex_id}: it is used by edge(s) "
                f"{sorted(users)}"
            )
        self._delete_entity("vertex", vertex_id)
        if record:
            self.record_replacement(EntityRef("vertex", vertex_id), ())

    @_transactional
    def remove_entities(self, keys: Iterable[Tuple[str, int]]) -> None:
        """Remove a set of entities, innermost dependency last."""

        remaining = list(keys)
        order = {"face": 0, "edge": 1, "vertex": 2}
        for kind, entity_id in sorted(remaining, key=lambda k: (order[k[0]], -k[1])):
            if kind == "face":
                self.remove_face(entity_id)
            elif kind == "edge":
                self.remove_edge(entity_id)
            else:
                self.remove_vertex(entity_id)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @_transactional
    def add_point(self, x: float, y: float, z: float = 0.0) -> int:
        """Place a point and return its vertex ID."""

        position = np.asarray((x, y, z), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise GeometryError("point coordinates must be a finite 3-vector")
        vertex_id = self._allocate("vertex")
        self._put_entity("vertex", Vertex(id=vertex_id, position=position))
        return vertex_id

    @_transactional
    def add_points(self, positions: Iterable[Sequence[float]]) -> List[int]:
        """Place several points at once."""

        return [self.add_point(*np.asarray(p, dtype=float)) for p in positions]

    @_transactional
    def add_line(self, start: int, end: int) -> int:
        """Connect two points with a straight line."""

        self._require_vertex(start)
        self._require_vertex(end)
        if start == end:
            raise GeometryError("a line needs two distinct points")
        if float(
            np.linalg.norm(
                self.vertices[end].position - self.vertices[start].position
            )
        ) <= 0.0:
            raise GeometryError("a line needs two spatially distinct points")
        return self._add_edge(start, end, Straight())

    @_transactional
    def add_arc(self, start: int, via: int, end: int) -> int:
        """Connect two points with a circular arc through a third point."""

        self._require_vertex(start)
        self._require_vertex(via)
        self._require_vertex(end)
        if len({start, via, end}) != 3:
            raise GeometryError("an arc needs three distinct points")
        # Resolve now so a bad arc is rejected at modelling time rather than
        # at mesh time, where the diagnostic would be far from the cause.
        arc_frame(
            self.vertices[start].position,
            self.vertices[via].position,
            self.vertices[end].position,
        )
        return self._add_edge(start, end, Arc(via_vertex=via))

    @_transactional
    def add_spline(
        self, start: int, control_vertices: Sequence[int], end: int
    ) -> int:
        """Connect two vertices with a lightweight Bezier spline."""

        self._require_vertex(start)
        self._require_vertex(end)
        controls = tuple(int(vertex) for vertex in control_vertices)
        for vertex in controls:
            self._require_vertex(vertex)
        if start == end or len({start, *controls, end}) < 2:
            raise GeometryError("a spline needs two distinct end points")
        return self._add_edge(start, end, Spline(controls))

    @_transactional
    def add_polyline(self, vertex_ids: Sequence[int], close: bool = False) -> List[int]:
        """Connect a run of points with straight lines."""

        ids = list(vertex_ids)
        if len(ids) < 2:
            raise GeometryError("a polyline needs at least two points")
        pairs = list(zip(ids, ids[1:]))
        if close:
            pairs.append((ids[-1], ids[0]))
        return [self.add_line(a, b) for a, b in pairs]

    def _add_edge(self, start: int, end: int, curve: CurveShape) -> int:
        edge_id = self._allocate("edge")
        self._put_entity(
            "edge", Edge(id=edge_id, start=start, end=end, curve=curve)
        )
        return edge_id

    @_transactional
    def add_face(
        self,
        edge_ids: Sequence[int],
        corners: Sequence[int] | None = None,
        *,
        surface: Surface | None = None,
    ) -> int:
        """Create a plate bounded by a closed loop of edges.

        The edges may be given in any order and any direction; the loop is
        ordered here.  ``corners`` optionally overrides the four loop indices
        where the sides begin, for faces whose corners are not obvious from
        boundary turn angle.
        """

        loop = self._order_loop(edge_ids)
        if corners is None:
            resolved = self._detect_corners(loop) if len(loop) >= 4 else ()
        else:
            resolved = self._validate_corners(tuple(int(c) for c in corners), len(loop))
        return self._add_face_from_loop(loop, resolved, surface=surface)

    def order_loop(self, edge_ids: Sequence[int]) -> Tuple[OrientedEdge, ...]:
        """Order an unordered edge set into a closed, oriented boundary loop."""

        return self._order_loop(edge_ids)

    @_transactional
    def add_face_from_loop(
        self,
        loop: Sequence[OrientedEdge],
        corners: Sequence[int] | None = None,
        *,
        surface: Surface | None = None,
    ) -> int:
        """Create a face from an explicit oriented loop and corner positions.

        Used by the decomposition tools, which know exactly where the corners
        belong and must not leave it to turn-angle detection.
        """

        ordered = tuple(loop)
        if len(ordered) < 3:
            raise GeometryError("a face needs at least three edges")
        for item in ordered:
            self._require_edge(item.edge)
        for current, following in zip(ordered, ordered[1:] + ordered[:1]):
            if self.oriented_end_vertex(current) != self.oriented_start_vertex(
                following
            ):
                raise GeometryError(
                    f"loop is not continuous at edge {following.edge}"
                )
        return self._add_face_from_loop(
            ordered,
            () if corners is None else self._validate_corners(
                tuple(int(c) for c in corners), len(ordered)
            ),
            surface=surface,
        )

    def _add_face_from_loop(
        self,
        loop: Tuple[OrientedEdge, ...],
        corners: Tuple[int, ...],
        *,
        surface: Surface | None = None,
    ) -> int:
        face_id = self._allocate("face")
        self._put_entity(
            "face",
            Face(
                id=face_id,
                loop=loop,
                corners=corners,
                surface=surface or (CoonsSurface() if len(corners) == 4 else None),
            ),
        )
        return face_id

    @_transactional
    def add_plate(self, vertex_ids: Sequence[int]) -> int:
        """Create a plate directly from an ordered ring of points.

        Convenience for the common case: the lines are created too.
        """

        edge_ids = self.add_polyline(vertex_ids, close=True)
        face_id = self.add_face(edge_ids)
        corners = [self.vertex_position(vertex_id) for vertex_id in vertex_ids[:4]]
        if len(corners) >= 3:
            origin = corners[0]
            u_vector = corners[1] - origin
            v_vector = corners[-1] - origin
            if float(np.linalg.norm(np.cross(u_vector, v_vector))) > 1.0e-14:
                self._put_entity(
                    "face",
                    replace(
                        self.faces[face_id],
                        surface=Plane(origin, u_vector, v_vector),
                    ),
                )
        return face_id

    def validate_topology(self) -> Tuple[str, ...]:
        """Return deterministic topology errors without mutating the model."""

        errors: List[str] = []
        for vertex_id, vertex in sorted(self.vertices.items()):
            if vertex.id != vertex_id:
                errors.append(f"vertex key {vertex_id} does not match ID {vertex.id}")
            if vertex.position.shape != (3,) or not np.all(np.isfinite(vertex.position)):
                errors.append(f"vertex {vertex_id} has invalid coordinates")
        for edge_id, edge in sorted(self.edges.items()):
            if edge.id != edge_id:
                errors.append(f"edge key {edge_id} does not match ID {edge.id}")
            for vertex_id in (edge.start, edge.end):
                if vertex_id not in self.vertices:
                    errors.append(f"edge {edge_id} references missing vertex {vertex_id}")
            if edge.start == edge.end:
                errors.append(f"edge {edge_id} has coincident topology endpoints")
            if edge.start in self.vertices and edge.end in self.vertices:
                start = self.vertices[edge.start].position
                end = self.vertices[edge.end].position
                length = float(np.linalg.norm(end - start))
                if length <= self.tolerance.effective_length(length):
                    errors.append(f"edge {edge_id} has zero geometric length")
            if isinstance(edge.curve, Arc) and edge.curve.via_vertex not in self.vertices:
                errors.append(
                    f"edge {edge_id} references missing arc via vertex {edge.curve.via_vertex}"
                )
            elif (
                isinstance(edge.curve, Arc)
                and edge.start in self.vertices
                and edge.end in self.vertices
                and edge.curve.via_vertex in self.vertices
            ):
                try:
                    arc_frame(
                        self.vertices[edge.start].position,
                        self.vertices[edge.curve.via_vertex].position,
                        self.vertices[edge.end].position,
                    )
                except (ValueError, GeometryError) as error:
                    errors.append(f"edge {edge_id} has invalid arc geometry: {error}")
            if isinstance(edge.curve, Spline):
                for vertex_id in edge.curve.control_vertices:
                    if vertex_id not in self.vertices:
                        errors.append(
                            f"edge {edge_id} references missing spline control vertex {vertex_id}"
                        )
        for face_id, face in sorted(self.faces.items()):
            if face.id != face_id:
                errors.append(f"face key {face_id} does not match ID {face.id}")
            if face.corners:
                if (
                    len(face.corners) != 4
                    or len(set(face.corners)) != 4
                    or tuple(sorted(face.corners)) != face.corners
                    or any(index < 0 or index >= len(face.loop) for index in face.corners)
                ):
                    errors.append(f"face {face_id} has invalid mapped corners")
            loops = (face.loop,) + tuple(face.holes)
            for loop_index, loop in enumerate(loops):
                if not loop:
                    errors.append(f"face {face_id} has an empty loop {loop_index}")
                    continue
                minimum = 3 if loop_index == 0 else 2
                if len(loop) < minimum:
                    errors.append(
                        f"face {face_id} loop {loop_index} needs at least "
                        f"{minimum} edges"
                    )
                for item in loop:
                    if item.edge not in self.edges:
                        errors.append(f"face {face_id} references missing edge {item.edge}")
                if len({item.edge for item in loop}) != len(loop):
                    errors.append(f"face {face_id} loop {loop_index} repeats an edge")
                existing = [item for item in loop if item.edge in self.edges]
                for current, following in zip(existing, existing[1:] + existing[:1]):
                    if self.oriented_end_vertex(current) != self.oriented_start_vertex(following):
                        errors.append(f"face {face_id} loop {loop_index} is discontinuous")
                        break
            all_edges = [item.edge for loop in loops for item in loop]
            if len(set(all_edges)) != len(all_edges):
                errors.append(
                    f"face {face_id} reuses an edge across outer and inner loops"
                )
            if not any(
                message.startswith(f"face {face_id} ") for message in errors
            ):
                errors.extend(self._validate_face_geometry(face_id))
        keys = self.entity_keys()
        for name, members in sorted(self.groups.items()):
            for reference in sorted(members, key=lambda item: (item.kind, item.id)):
                if (reference.kind, reference.id) not in keys and reference not in self._replacement_history:
                    errors.append(f"group {name!r} references missing entity {reference}")
        for reference in sorted(self.tags, key=lambda item: (item.kind, item.id)):
            if (reference.kind, reference.id) not in keys and reference not in self._replacement_history:
                errors.append(f"tags reference missing entity {reference}")
        errors.extend(self._validate_replacement_history())
        return tuple(errors)

    @staticmethod
    def _segments_intersect_2d(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
        tolerance: float,
    ) -> bool:
        """Whether two bounded planar segments touch or cross."""

        def cross(first: np.ndarray, second: np.ndarray) -> float:
            return float(first[0] * second[1] - first[1] * second[0])

        first = first_end - first_start
        second = second_end - second_start
        denominator = cross(first, second)
        offset = second_start - first_start
        if abs(denominator) <= tolerance:
            if abs(cross(offset, first)) > tolerance:
                return False
            axis = int(np.argmax(np.abs(first)))
            if abs(float(first[axis])) <= tolerance:
                return float(np.linalg.norm(first_start - second_start)) <= tolerance
            interval = sorted(
                (
                    float((second_start[axis] - first_start[axis]) / first[axis]),
                    float((second_end[axis] - first_start[axis]) / first[axis]),
                )
            )
            return max(0.0, interval[0]) <= min(1.0, interval[1]) + tolerance
        first_parameter = cross(offset, second) / denominator
        second_parameter = cross(offset, first) / denominator
        return (
            -tolerance <= first_parameter <= 1.0 + tolerance
            and -tolerance <= second_parameter <= 1.0 + tolerance
        )

    @classmethod
    def _polygon_self_intersects(
        cls, polygon: np.ndarray, tolerance: float = 1.0e-10
    ) -> bool:
        count = len(polygon)
        for first in range(count):
            first_next = (first + 1) % count
            for second in range(first + 1, count):
                second_next = (second + 1) % count
                if (
                    first == second
                    or first_next == second
                    or second_next == first
                ):
                    continue
                if cls._segments_intersect_2d(
                    polygon[first],
                    polygon[first_next],
                    polygon[second],
                    polygon[second_next],
                    tolerance,
                ):
                    return True
        return False

    def _validation_loop_points(
        self, loop: Sequence[OrientedEdge]
    ) -> np.ndarray:
        points: List[np.ndarray] = []
        for item in loop:
            edge = self.edges[item.edge]
            # Circular arcs are analytical and four quarter-parameter spans
            # are sufficient to qualify their trim winding and residual on an
            # analytical support.  Keep the denser fallback for splines,
            # whose shape is not fixed by three defining vertices.
            count = (
                3
                if isinstance(edge.curve, Straight)
                else 5
                if isinstance(edge.curve, Arc)
                else 17
            )
            samples = self.sample_edge(item.edge, np.linspace(0.0, 1.0, count))
            if not item.forward:
                samples = samples[::-1]
            points.extend(samples[:-1])
        return np.asarray(points, dtype=float)

    def _validate_face_geometry(self, face_id: int) -> List[str]:
        """Validate trim geometry after its topology references are known valid."""

        face = self.faces[face_id]
        loops = (face.loop,) + tuple(face.holes)
        if isinstance(face.surface, Plane) and all(
            isinstance(self.edges[item.edge].curve, Straight)
            for loop in loops
            for item in loop
        ):
            if len(loops) == 1 and len(face.loop) == 4:
                return self._validate_planar_straight_quad_geometry(face_id)
            return self._validate_planar_straight_face_geometry(face_id)
        points_3d = [self._validation_loop_points(loop) for loop in loops]
        if any(len(points) < 3 for points in points_3d):
            return []

        combined = np.vstack(points_3d)
        origin = combined.mean(axis=0)
        _values, singular, vectors = np.linalg.svd(combined - origin)
        extent = float(np.linalg.norm(np.ptp(combined, axis=0)))
        planar = len(singular) < 3 or float(singular[-1]) <= self.tolerance.effective_surface_residual(extent)
        surface = face.surface
        if surface is not None and not (
            isinstance(surface, CoonsSurface) and not surface.has_boundaries
        ):
            try:
                for point in combined:
                    uv = surface.local_uv(point)
                    projected = np.asarray(surface.evaluate(*uv), dtype=float)
                    if (
                        float(np.linalg.norm(projected - point))
                        > self.tolerance.effective_surface_residual(extent)
                    ):
                        return [
                            f"face {face_id} boundary is inconsistent with its explicit surface"
                        ]
            except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
                return [f"face {face_id} has invalid surface geometry: {error}"]
        if planar:
            polygons = [
                np.column_stack(
                    ((points - origin) @ vectors[0], (points - origin) @ vectors[1])
                )
                for points in points_3d
            ]
        elif surface is not None and not (
            isinstance(surface, CoonsSurface) and not surface.has_boundaries
        ):
            try:
                polygons = [
                    np.asarray([surface.local_uv(point) for point in points])
                    for points in points_3d
                ]
            except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
                return [f"face {face_id} has invalid surface geometry: {error}"]
        elif len(face.corners) == 4:
            try:
                polygons = [
                    np.asarray([self.face_local_uv(face_id, point) for point in points])
                    for points in points_3d
                ]
            except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
                return [f"face {face_id} has invalid Coons geometry: {error}"]
        else:
            return [f"face {face_id} is non-planar and has no explicit surface"]

        result: List[str] = []
        for index, polygon in enumerate(polygons):
            following = np.roll(polygon, -1, axis=0)
            area = 0.5 * abs(
                float(
                    np.sum(
                        polygon[:, 0] * following[:, 1]
                        - following[:, 0] * polygon[:, 1]
                    )
                )
            )
            extent = float(np.linalg.norm(np.ptp(polygon, axis=0)))
            if area <= self.tolerance.effective_area(extent):
                result.append(f"face {face_id} loop {index} has zero area")
            if self._polygon_self_intersects(polygon):
                result.append(f"face {face_id} loop {index} self-intersects")

        outer = polygons[0]
        for index, hole in enumerate(polygons[1:], start=1):
            if not all(
                self._point_in_polygon(point, outer, include_boundary=False)
                for point in hole
            ):
                result.append(
                    f"face {face_id} hole {index} is not strictly inside the outer loop"
                )
            if self._polygons_intersect(outer, hole):
                result.append(f"face {face_id} hole {index} intersects the outer loop")
        for first in range(1, len(polygons)):
            for second in range(first + 1, len(polygons)):
                if (
                    self._polygons_intersect(polygons[first], polygons[second])
                    or self._point_in_polygon(
                        polygons[first][0], polygons[second], include_boundary=True
                    )
                    or self._point_in_polygon(
                        polygons[second][0], polygons[first], include_boundary=True
                    )
                ):
                    result.append(
                        f"face {face_id} holes {first} and {second} overlap"
                    )
        return result

    @classmethod
    def _polygons_intersect(
        cls, first: np.ndarray, second: np.ndarray
    ) -> bool:
        for first_index in range(len(first)):
            for second_index in range(len(second)):
                if cls._segments_intersect_2d(
                    first[first_index],
                    first[(first_index + 1) % len(first)],
                    second[second_index],
                    second[(second_index + 1) % len(second)],
                    1.0e-10,
                ):
                    return True
        return False

    def _validate_replacement_history(self) -> List[str]:
        errors: List[str] = []
        keys = self.entity_keys()
        for old, replacements in sorted(
            self._replacement_history.items(),
            key=lambda item: (str(item[0].kind), item[0].id),
        ):
            if old.kind not in self._next_id or old.id <= 0 or old.id >= self._next_id[old.kind]:
                errors.append(f"replacement history references missing entity {old}")
                continue
            if (old.kind, old.id) in keys:
                errors.append(f"replacement history supersedes surviving entity {old}")
            for replacement in replacements:
                if replacement.kind != old.kind:
                    errors.append(
                        f"replacement history changes entity kind from {old} to {replacement}"
                    )
                    continue
                if (
                    replacement.id <= 0
                    or replacement.id >= self._next_id[replacement.kind]
                ):
                    errors.append(
                        f"replacement history references missing entity {replacement}"
                    )
                elif (
                    (replacement.kind, replacement.id) not in keys
                    and replacement not in self._replacement_history
                ):
                    errors.append(
                        "replacement history has an unresolved descendant "
                        f"{replacement}"
                    )

        visiting: Set[EntityRef] = set()
        visited: Set[EntityRef] = set()

        def visit(reference: EntityRef) -> None:
            if reference in visited or reference not in self._replacement_history:
                return
            if reference in visiting:
                errors.append(
                    f"replacement history contains a cycle at {reference}"
                )
                return
            visiting.add(reference)
            for descendant in self._replacement_history[reference]:
                visit(descendant)
            visiting.remove(reference)
            visited.add(reference)

        for reference in self._replacement_history:
            visit(reference)
        return errors

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    @_transactional
    def extrude(
        self, edge_ids: Sequence[int], vector: Sequence[float]
    ) -> List[int]:
        """Sweep edges along a vector, producing one face per edge.

        Shared points between consecutive edges produce shared swept lines, so
        extruding a chain gives a strip of faces that is conformal by
        construction rather than by coincident-node merging.
        """

        offset = np.asarray(vector, dtype=float)
        if offset.shape != (3,):
            raise GeometryError("extrusion vector must be a 3 component vector")
        if float(np.linalg.norm(offset)) <= 0.0:
            raise GeometryError("extrusion vector must be non-zero")

        swept_vertex: Dict[int, int] = {}
        swept_line: Dict[int, int] = {}

        def translated(vertex_id: int) -> int:
            if vertex_id not in swept_vertex:
                position = self.vertices[vertex_id].position + offset
                swept_vertex[vertex_id] = self.add_point(*position)
            return swept_vertex[vertex_id]

        def connector(vertex_id: int) -> int:
            if vertex_id not in swept_line:
                swept_line[vertex_id] = self.add_line(
                    vertex_id, translated(vertex_id)
                )
            return swept_line[vertex_id]

        face_ids: List[int] = []
        for edge_id in edge_ids:
            edge = self._require_edge(edge_id)
            start_top = translated(edge.start)
            end_top = translated(edge.end)
            if isinstance(edge.curve, Arc):
                via_top = translated(edge.curve.via_vertex)
                top_edge = self.add_arc(start_top, via_top, end_top)
            elif isinstance(edge.curve, Spline):
                controls = tuple(translated(vertex) for vertex in edge.curve.control_vertices)
                top_edge = self.add_spline(start_top, controls, end_top)
            else:
                top_edge = self.add_line(start_top, end_top)

            loop = (
                OrientedEdge(edge_id, True),
                OrientedEdge(connector(edge.end), True),
                OrientedEdge(top_edge, False),
                OrientedEdge(connector(edge.start), False),
            )
            face_ids.append(self._add_face_from_loop(loop, (0, 1, 2, 3)))
        return face_ids

    @_transactional
    def revolve(
        self,
        edge_ids: Sequence[int],
        axis_point: Sequence[float],
        axis_direction: Sequence[float],
        angle: float,
        segments: int | None = None,
    ) -> List[int]:
        """Sweep edges about an axis, producing one face per edge per segment.

        The swept boundaries are true arcs, so a revolved profile is exact
        rather than faceted.  The sweep is cut into segments of at most a
        quarter turn, which keeps every arc well conditioned.
        """

        origin = np.asarray(axis_point, dtype=float)
        direction = np.asarray(axis_direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if origin.shape != (3,) or direction.shape != (3,):
            raise GeometryError("the revolve axis needs a point and a direction")
        if norm <= 0.0:
            raise GeometryError("the revolve axis direction must be non-zero")
        direction = direction / norm
        if not np.isfinite(angle) or angle == 0.0:
            raise GeometryError("the revolve angle must be non-zero")

        profile = list(dict.fromkeys(int(e) for e in edge_ids))
        for edge_id in profile:
            self._require_edge(edge_id)
        self._reject_on_axis(profile, origin, direction)

        if segments is None:
            segments = max(1, int(np.ceil(abs(angle) / (0.5 * np.pi) - 1.0e-9)))
        segments = int(segments)
        if segments < 1:
            raise GeometryError("a revolve needs at least one segment")
        step = float(angle) / segments

        # A full turn must land back on the profile it started from, otherwise
        # the result is a slit cylinder with a seam of coincident-but-separate
        # points rather than a closed one.
        closes = abs(abs(float(angle)) - 2.0 * np.pi) <= 1.0e-9
        start_edges = list(profile)
        start_vertices = {edge_id: edge_id for edge_id in profile}
        edge_origin = {edge_id: edge_id for edge_id in profile}
        vertex_origin: Dict[int, int] = {}
        for edge_id in profile:
            edge = self.edges[edge_id]
            controls = (
                (edge.curve.via_vertex,)
                if isinstance(edge.curve, Arc)
                else edge.curve.control_vertices
                if isinstance(edge.curve, Spline)
                else ()
            )
            for vertex_id in (edge.start, edge.end, *controls):
                vertex_origin[vertex_id] = vertex_id
        del start_vertices

        face_ids: List[int] = []
        for index in range(segments):
            closing = closes and index == segments - 1
            profile, made, edge_origin, vertex_origin = self._revolve_once(
                profile,
                origin,
                direction,
                step,
                edge_origin=edge_origin,
                vertex_origin=vertex_origin,
                closing=closing,
            )
            face_ids.extend(made)
        del start_edges
        return face_ids

    def _revolve_once(
        self,
        profile: Sequence[int],
        origin: np.ndarray,
        direction: np.ndarray,
        step: float,
        *,
        edge_origin: Dict[int, int],
        vertex_origin: Dict[int, int],
        closing: bool = False,
    ) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
        swept_vertex: Dict[int, int] = {}
        swept_arc: Dict[int, int] = {}

        def rotated(vertex_id: int) -> int:
            if closing:
                # Land back on the point this one was swept from.
                return vertex_origin[vertex_id]
            if vertex_id not in swept_vertex:
                position = _rotate_about_axis(
                    self.vertices[vertex_id].position, origin, direction, step
                )
                swept_vertex[vertex_id] = self.add_point(*position)
            return swept_vertex[vertex_id]

        def connector(vertex_id: int) -> int:
            if vertex_id not in swept_arc:
                midpoint = _rotate_about_axis(
                    self.vertices[vertex_id].position,
                    origin,
                    direction,
                    0.5 * step,
                )
                via = self.add_point(*midpoint)
                swept_arc[vertex_id] = self.add_arc(
                    vertex_id, via, rotated(vertex_id)
                )
            return swept_arc[vertex_id]

        next_profile: List[int] = []
        face_ids: List[int] = []
        next_edge_origin: Dict[int, int] = {}
        next_vertex_origin: Dict[int, int] = {}

        for edge_id in profile:
            edge = self.edges[edge_id]
            start_top = rotated(edge.start)
            end_top = rotated(edge.end)
            if closing:
                top_edge = edge_origin[edge_id]
            elif isinstance(edge.curve, Arc):
                top_edge = self.add_arc(
                    start_top, rotated(edge.curve.via_vertex), end_top
                )
            elif isinstance(edge.curve, Spline):
                top_edge = self.add_spline(
                    start_top,
                    tuple(rotated(vertex) for vertex in edge.curve.control_vertices),
                    end_top,
                )
            else:
                top_edge = self.add_line(start_top, end_top)

            loop = (
                OrientedEdge(edge_id, True),
                OrientedEdge(connector(edge.end), True),
                OrientedEdge(top_edge, False),
                OrientedEdge(connector(edge.start), False),
            )
            face_ids.append(self._add_face_from_loop(loop, (0, 1, 2, 3)))
            next_profile.append(top_edge)

            next_edge_origin[top_edge] = edge_origin[edge_id]
            next_vertex_origin[start_top] = vertex_origin[edge.start]
            next_vertex_origin[end_top] = vertex_origin[edge.end]
            if isinstance(edge.curve, Arc) and not closing:
                via_top = rotated(edge.curve.via_vertex)
                next_vertex_origin[via_top] = vertex_origin[edge.curve.via_vertex]
            elif isinstance(edge.curve, Spline) and not closing:
                for control in edge.curve.control_vertices:
                    control_top = rotated(control)
                    next_vertex_origin[control_top] = vertex_origin[control]

        return next_profile, face_ids, next_edge_origin, next_vertex_origin

    def _reject_on_axis(
        self,
        edge_ids: Sequence[int],
        origin: np.ndarray,
        direction: np.ndarray,
        tolerance: float = 1.0e-9,
    ) -> None:
        """A point on the axis would sweep into itself, not into an arc."""

        checked: set[int] = set()
        for edge_id in edge_ids:
            edge = self.edges[edge_id]
            vertices = [edge.start, edge.end]
            if isinstance(edge.curve, Arc):
                vertices.append(edge.curve.via_vertex)
            elif isinstance(edge.curve, Spline):
                vertices.extend(edge.curve.control_vertices)
            for vertex_id in vertices:
                if vertex_id in checked:
                    continue
                checked.add(vertex_id)
                offset = self.vertices[vertex_id].position - origin
                radial = offset - float(offset @ direction) * direction
                if float(np.linalg.norm(radial)) <= tolerance:
                    raise GeometryError(
                        f"point {vertex_id} lies on the revolve axis, so it "
                        "would sweep into itself rather than into an arc. "
                        "Move it off the axis, or model the apex region "
                        "separately."
                    )

    # ------------------------------------------------------------------
    # splitting
    # ------------------------------------------------------------------
    @_transactional
    def split_edge(
        self, edge_id: int, t: float = 0.5
    ) -> Tuple[int, Tuple[int, int]]:
        """Split a line or arc at parameter ``t``, keeping every face valid.

        Returns the new point and the two replacement edges.  Faces that used
        the original edge have it swapped for the pair in traversal order, and
        their corner indices shift to match, so a side that was one edge simply
        becomes a chain of two.  This is the primitive behind imprinting.
        """

        edge = self._require_edge(edge_id)
        edge_attachments = sorted(
            attachment.id
            for attachment in self.attachments.values()
            if attachment.target_kind is AttachmentTargetKind.EDGE
            and attachment.target_id == edge_id
        )
        if edge_attachments:
            raise GeometryError(
                f"cannot split edge {edge_id}: attachments {edge_attachments} "
                "require an explicit parameter remap"
            )
        member_use_ids = tuple(sorted(self._edge_member_uses.get(edge_id, ())))
        if not 0.0 < float(t) < 1.0:
            raise GeometryError(
                f"split parameter must be strictly between 0 and 1, got {t}"
            )

        new_vertex = self.add_point(*self.sample_edge(edge_id, np.array([t]))[0])
        if isinstance(edge.curve, Arc):
            first_via = self.add_point(
                *self.sample_edge(edge_id, np.array([0.5 * t]))[0]
            )
            second_via = self.add_point(
                *self.sample_edge(edge_id, np.array([0.5 * (1.0 + t)]))[0]
            )
            first = self.add_arc(edge.start, first_via, new_vertex)
            second = self.add_arc(new_vertex, second_via, edge.end)
        elif isinstance(edge.curve, Spline):
            points = self._spline_points(edge)
            left, right = self._split_bezier(points, float(t))
            left_controls = tuple(self.add_point(*point) for point in left[1:-1])
            right_controls = tuple(self.add_point(*point) for point in right[1:-1])
            first = self.add_spline(edge.start, left_controls, new_vertex)
            second = self.add_spline(new_vertex, right_controls, edge.end)
        else:
            first = self.add_line(edge.start, new_vertex)
            second = self.add_line(new_vertex, edge.end)

        for face_id in self.faces_using_edge(edge_id):
            self._put_entity(
                "face",
                self._replace_edge_in_loop(
                    self.faces[face_id], edge_id, first, second
                ),
            )

        self._delete_entity("edge", edge_id)
        for use_id in member_use_ids:
            original_use = self.member_edge_uses[use_id]
            member = self.members[original_use.member_id]
            use_ids = (
                self._allocate_structural("member_edge_use"),
                self._allocate_structural("member_edge_use"),
            )
            span = original_use.parent_range.end - original_use.parent_range.start
            forward = original_use.orientation is Orientation.FORWARD
            split_fraction = float(t) if forward else 1.0 - float(t)
            split_parent = original_use.parent_range.start + split_fraction * span
            axes = (first, second) if forward else (second, first)
            children = (
                MemberEdgeUse(
                    use_ids[0],
                    member.id,
                    axes[0],
                    ParameterRange(original_use.parent_range.start, split_parent),
                    original_use.orientation,
                ),
                MemberEdgeUse(
                    use_ids[1],
                    member.id,
                    axes[1],
                    ParameterRange(split_parent, original_use.parent_range.end),
                    original_use.orientation,
                ),
            )
            self._put_structural("member", replace_member_edge_use(member, original_use, children))
            for child in children:
                self._put_structural("member_edge_use", child)
            self._delete_structural("member_edge_use", original_use.id)
        self.record_replacement(
            EntityRef("edge", edge_id),
            (EntityRef("edge", first), EntityRef("edge", second)),
        )
        return new_vertex, (first, second)

    @staticmethod
    def _replace_edge_in_loop(
        face: Face, edge_id: int, first: int, second: int
    ) -> Face:
        corners = face.corners

        def replaced(
            loop: Tuple[OrientedEdge, ...], *, update_corners: bool
        ) -> Tuple[OrientedEdge, ...]:
            nonlocal corners
            positions = [
                index for index, item in enumerate(loop) if item.edge == edge_id
            ]
            for position in reversed(positions):
                item = loop[position]
                if item.forward:
                    replacement = (
                        OrientedEdge(first, True),
                        OrientedEdge(second, True),
                    )
                else:
                    # Traversed backwards, the far half comes first.
                    replacement = (
                        OrientedEdge(second, False),
                        OrientedEdge(first, False),
                    )
                loop = loop[:position] + replacement + loop[position + 1 :]
                if update_corners:
                    # A corner sitting on the split edge still starts where it
                    # did; everything after it moves along by one.
                    corners = tuple(
                        corner + 1 if corner > position else corner
                        for corner in corners
                    )
            return loop

        loop = replaced(face.loop, update_corners=True)
        holes = tuple(replaced(item, update_corners=False) for item in face.holes)
        return replace(
            face,
            loop=loop,
            corners=corners,
            holes=holes,
        )

    @_transactional
    def set_face_corners(self, face_id: int, corners: Sequence[int]) -> None:
        """Override which loop positions begin each of the four sides."""

        face = self._require_face(face_id)
        self._put_entity(
            "face",
            replace(
                face,
                corners=self._validate_corners(
                    tuple(int(c) for c in corners), len(face.loop)
                ),
            ),
        )

    @_transactional
    def set_face_surface(self, face_id: int, surface: Surface | None) -> None:
        """Replace a face support surface through the transaction owner."""

        face = self._require_face(face_id)
        self._put_entity("face", replace(face, surface=surface))

    @_transactional
    def set_face_metadata(
        self, face_id: int, metadata: Mapping[str, object]
    ) -> None:
        """Replace immutable face metadata with a checked JSON mapping."""

        face = self._require_face(face_id)
        self._put_entity("face", replace(face, metadata=metadata))

    @_transactional
    def update_face_metadata(
        self, face_id: int, **values: object
    ) -> None:
        """Return a face to committed state with selected metadata updates."""

        face = self._require_face(face_id)
        metadata = face.metadata.to_dict()  # type: ignore[union-attr]
        metadata.update(values)
        self._put_entity("face", replace(face, metadata=metadata))

    def face_side_lengths(self, face_id: int) -> Tuple[float, float, float, float]:
        face = self._require_face(face_id)
        return tuple(  # type: ignore[return-value]
            self.side_length(side) for side in face.sides()
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def vertex_position(self, vertex_id: int) -> np.ndarray:
        return self._require_vertex(vertex_id).position

    def sample_edge(self, edge_id: int, t: np.ndarray) -> np.ndarray:
        """Sample points along an edge in its own direction.

        ``t`` runs from 0 at the start vertex to 1 at the end vertex.  Uniform
        ``t`` gives uniform arc length for both straight lines and arcs.
        """

        edge = self._require_edge(edge_id)
        start = self.vertices[edge.start].position
        end = self.vertices[edge.end].position
        if isinstance(edge.curve, Arc):
            return sample_arc(self._arc_frame(edge), t)
        if isinstance(edge.curve, Spline):
            return sample_spline(self._spline_points(edge), t)
        return sample_straight(start, end, t)

    def edge_length(self, edge_id: int) -> float:
        edge = self._require_edge(edge_id)
        version = self._entity_versions.get(("edge", edge_id), 0)
        cached = self._edge_length_cache.get(edge_id)
        if cached is not None and cached[0] == version:
            return cached[1]
        if isinstance(edge.curve, Arc):
            result = self._arc_frame(edge).length
        elif isinstance(edge.curve, Spline):
            samples = sample_spline(
                self._spline_points(edge), np.linspace(0.0, 1.0, 65)
            )
            result = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
        else:
            start = self.vertices[edge.start].position
            end = self.vertices[edge.end].position
            result = float(np.linalg.norm(end - start))
        self._edge_length_cache[edge_id] = (version, result)
        return result

    def _validate_planar_straight_quad_geometry(self, face_id: int) -> List[str]:
        """Exact allocation-light validation for the dominant plate cell."""

        face = self.faces[face_id]
        surface = face.surface
        assert isinstance(surface, Plane)
        world = tuple(
            self.vertices[self.oriented_start_vertex(item)].position
            for item in face.loop
        )
        minimum = np.minimum(np.minimum(world[0], world[1]), np.minimum(world[2], world[3]))
        maximum = np.maximum(np.maximum(world[0], world[1]), np.maximum(world[2], world[3]))
        extent = float(np.linalg.norm(maximum - minimum))
        normal_value = np.cross(surface.u_vector, surface.v_vector)
        normal_length = float(np.linalg.norm(normal_value))
        if not np.isfinite(normal_length) or normal_length <= 0.0:
            return [f"face {face_id} has invalid surface geometry"]
        normal = normal_value / normal_length
        residual_tolerance = self.tolerance.effective_surface_residual(extent)
        if any(
            abs(float((point - surface.origin) @ normal)) > residual_tolerance
            for point in world
        ):
            return [
                f"face {face_id} boundary is inconsistent with its explicit surface"
            ]

        first_axis = surface.u_vector / float(np.linalg.norm(surface.u_vector))
        second_axis = np.cross(normal, first_axis)
        points = tuple(
            (
                float((point - surface.origin) @ first_axis),
                float((point - surface.origin) @ second_axis),
            )
            for point in world
        )
        area_twice = abs(
            sum(
                points[index][0] * points[(index + 1) % 4][1]
                - points[(index + 1) % 4][0] * points[index][1]
                for index in range(4)
            )
        )
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        local_extent = float(np.hypot(max_x - min_x, max_y - min_y))
        result: List[str] = []
        if 0.5 * area_twice <= self.tolerance.effective_area(local_extent):
            result.append(f"face {face_id} loop 0 has zero area")

        def cross(first: tuple[float, float], second: tuple[float, float]) -> float:
            return first[0] * second[1] - first[1] * second[0]

        def subtract(
            first: tuple[float, float], second: tuple[float, float]
        ) -> tuple[float, float]:
            return first[0] - second[0], first[1] - second[1]

        def intersects(
            first_start: tuple[float, float],
            first_end: tuple[float, float],
            second_start: tuple[float, float],
            second_end: tuple[float, float],
        ) -> bool:
            first = subtract(first_end, first_start)
            second = subtract(second_end, second_start)
            denominator = cross(first, second)
            offset = subtract(second_start, first_start)
            tolerance = 1.0e-10
            if abs(denominator) <= tolerance:
                if abs(cross(offset, first)) > tolerance:
                    return False
                axis = 0 if abs(first[0]) >= abs(first[1]) else 1
                if abs(first[axis]) <= tolerance:
                    delta = subtract(first_start, second_start)
                    return float(np.hypot(*delta)) <= tolerance
                interval = sorted(
                    (
                        (second_start[axis] - first_start[axis]) / first[axis],
                        (second_end[axis] - first_start[axis]) / first[axis],
                    )
                )
                return max(0.0, interval[0]) <= min(1.0, interval[1]) + tolerance
            first_parameter = cross(offset, second) / denominator
            second_parameter = cross(offset, first) / denominator
            return (
                -tolerance <= first_parameter <= 1.0 + tolerance
                and -tolerance <= second_parameter <= 1.0 + tolerance
            )

        if intersects(points[0], points[1], points[2], points[3]) or intersects(
            points[1], points[2], points[3], points[0]
        ):
            result.append(f"face {face_id} loop 0 self-intersects")
        return result

    def _validate_planar_straight_face_geometry(self, face_id: int) -> List[str]:
        """Validate a straight-trimmed plane without sampling or per-face SVD.

        This exact common path keeps full geometric validation linear in the
        trim size for large plate grids.  Curved trims and non-planar supports
        continue through the general sampled/residual-qualified path above.
        """

        face = self.faces[face_id]
        surface = face.surface
        assert isinstance(surface, Plane)
        loops = (face.loop,) + tuple(face.holes)
        points_3d = [
            np.asarray(
                [
                    self.vertices[self.oriented_start_vertex(item)].position
                    for item in loop
                ],
                dtype=float,
            )
            for loop in loops
        ]
        if any(len(points) < 3 for points in points_3d):
            return []

        combined = np.vstack(points_3d)
        extent = float(np.linalg.norm(np.ptp(combined, axis=0)))
        normal = np.cross(surface.u_vector, surface.v_vector)
        normal_length = float(np.linalg.norm(normal))
        if not np.isfinite(normal_length) or normal_length <= 0.0:
            return [f"face {face_id} has invalid surface geometry"]
        normal = normal / normal_length
        residuals = np.abs((combined - surface.origin) @ normal)
        if float(np.max(residuals, initial=0.0)) > self.tolerance.effective_surface_residual(extent):
            return [
                f"face {face_id} boundary is inconsistent with its explicit surface"
            ]

        first_axis = surface.u_vector / float(np.linalg.norm(surface.u_vector))
        second_axis = np.cross(normal, first_axis)
        polygons = [
            np.column_stack(
                (
                    (points - surface.origin) @ first_axis,
                    (points - surface.origin) @ second_axis,
                )
            )
            for points in points_3d
        ]

        result: List[str] = []
        for index, polygon in enumerate(polygons):
            following = np.roll(polygon, -1, axis=0)
            area = 0.5 * abs(
                float(
                    np.sum(
                        polygon[:, 0] * following[:, 1]
                        - following[:, 0] * polygon[:, 1]
                    )
                )
            )
            local_extent = float(np.linalg.norm(np.ptp(polygon, axis=0)))
            if area <= self.tolerance.effective_area(local_extent):
                result.append(f"face {face_id} loop {index} has zero area")
            if self._polygon_self_intersects(polygon):
                result.append(f"face {face_id} loop {index} self-intersects")

        outer = polygons[0]
        for index, hole in enumerate(polygons[1:], start=1):
            if not all(
                self._point_in_polygon(point, outer, include_boundary=False)
                for point in hole
            ):
                result.append(
                    f"face {face_id} hole {index} is not strictly inside the outer loop"
                )
            if self._polygons_intersect(outer, hole):
                result.append(f"face {face_id} hole {index} intersects the outer loop")
        for first in range(1, len(polygons)):
            for second in range(first + 1, len(polygons)):
                if (
                    self._polygons_intersect(polygons[first], polygons[second])
                    or self._point_in_polygon(
                        polygons[first][0], polygons[second], include_boundary=True
                    )
                    or self._point_in_polygon(
                        polygons[second][0], polygons[first], include_boundary=True
                    )
                ):
                    result.append(
                        f"face {face_id} holes {first} and {second} overlap"
                    )
        return result

    def edge_tangent(self, edge_id: int, t: float) -> np.ndarray:
        """Unit tangent along the edge's own direction at parameter ``t``."""

        edge = self._require_edge(edge_id)
        if isinstance(edge.curve, Arc):
            return arc_tangent(self._arc_frame(edge), t)
        if isinstance(edge.curve, Spline):
            return spline_tangent(self._spline_points(edge), t)
        return straight_tangent(
            self.vertices[edge.start].position, self.vertices[edge.end].position
        )

    def closest_edge_point(
        self, edge_id: int, point: Sequence[float]
    ) -> Tuple[np.ndarray, float, float]:
        """Closest point, edge parameter and distance on a bounded curve."""

        edge = self._require_edge(edge_id)
        target = np.asarray(point, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise GeometryError("point must be a finite 3-vector")
        if isinstance(edge.curve, Straight):
            start = self.vertex_position(edge.start)
            vector = self.vertex_position(edge.end) - start
            denominator = float(vector @ vector)
            if denominator <= 0.0:
                raise GeometryError(f"edge {edge_id} has zero length")
            parameter = float(np.clip((target-start) @ vector / denominator, 0.0, 1.0))
            made = start + parameter*vector
            return made, parameter, float(np.linalg.norm(made-target))
        parameters = np.linspace(0.0, 1.0, 65)
        samples = self.sample_edge(edge_id, parameters)
        index = int(np.argmin(np.linalg.norm(samples-target, axis=1)))
        lower = float(parameters[max(0, index-1)])
        upper = float(parameters[min(len(parameters)-1, index+1)])
        ratio = 0.5*(np.sqrt(5.0)-1.0)
        first = upper-ratio*(upper-lower)
        second = lower+ratio*(upper-lower)

        def objective(parameter: float) -> float:
            offset = self.sample_edge(edge_id, np.asarray([parameter]))[0]-target
            return float(offset @ offset)

        first_value, second_value = objective(first), objective(second)
        for _ in range(48):
            if first_value <= second_value:
                upper, second, second_value = second, first, first_value
                first = upper-ratio*(upper-lower)
                first_value = objective(first)
            else:
                lower, first, first_value = first, second, second_value
                second = lower+ratio*(upper-lower)
                second_value = objective(second)
        parameter = 0.5*(lower+upper)
        made = self.sample_edge(edge_id, np.asarray([parameter]))[0]
        return made, parameter, float(np.linalg.norm(made-target))

    def arc_frame(self, edge_id: int) -> ArcFrame:
        """The resolved circle of an arc edge: centre, radius, axes and sweep.

        Public because a mesh backend that rebuilds the model in another kernel
        needs the circle, not just samples along it.  Raises for a straight edge
        rather than returning a degenerate frame.
        """

        edge = self._require_edge(edge_id)
        if not isinstance(edge.curve, Arc):
            raise GeometryError(f"edge {edge_id} is not an arc")
        return self._arc_frame(edge)

    def _arc_frame(self, edge: Edge) -> ArcFrame:
        """Resolve and cache an arc's circle, invalidated when points move."""

        assert isinstance(edge.curve, Arc)
        stamp = self._geometry_stamp(edge)
        cached = self._arc_cache.get(edge.id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        frame = arc_frame(
            self.vertices[edge.start].position,
            self.vertices[edge.curve.via_vertex].position,
            self.vertices[edge.end].position,
        )
        self._arc_cache[edge.id] = (stamp, frame)
        return frame

    def _geometry_stamp(self, edge: Edge) -> int:
        assert isinstance(edge.curve, Arc)
        return hash(
            (
                self.vertices[edge.start].position.tobytes(),
                self.vertices[edge.curve.via_vertex].position.tobytes(),
                self.vertices[edge.end].position.tobytes(),
            )
        )

    @_transactional
    def move_point(self, vertex_id: int, x: float, y: float, z: float = 0.0) -> None:
        """Move a point; every curve referencing it follows."""

        vertex = self._require_vertex(vertex_id)
        position = np.asarray((x, y, z), dtype=float)
        if not np.all(np.isfinite(position)):
            raise GeometryError("point coordinates must be finite")
        if np.array_equal(vertex.position, position):
            return
        affected_edges = tuple(self.edges_using_vertex(vertex_id))
        affected_faces = {
            face_id
            for edge_id in affected_edges
            for face_id in self.faces_using_edge(edge_id)
        }
        assert self._transaction_journal is not None
        # A face AABB is derived through its boundary edges.  Capture every
        # dependent bound while the original vertex is still live; capturing
        # it later from an unchanged immutable Face record would already see
        # the new point and lose the old changed region.
        for face_id in sorted(affected_faces):
            face_key = ("face", face_id)
            self._transaction_journal.bounds_before.setdefault(
                face_key, self._entity_bounds(face_key)
            )
            self._transaction_journal.spatial_updates.add(face_key)
        self._put_entity("vertex", replace(vertex, position=position))
        if affected_edges:
            for edge_id in affected_edges:
                edge_key = ("edge", edge_id)
                if edge_key not in self._transaction_journal.bounds_before:
                    # Reconstruct the previous bound from the journalled point
                    # without copying unrelated geometry.
                    current = self._vertices[vertex_id]
                    self._vertices[vertex_id] = vertex
                    try:
                        self._transaction_journal.bounds_before[edge_key] = (
                            self._entity_bounds(edge_key)
                        )
                    finally:
                        self._vertices[vertex_id] = current
                self._transaction_journal.spatial_updates.add(edge_key)
        for edge_id in affected_edges:
            edge = self._edges[edge_id]
            if isinstance(edge.curve, Arc):
                try:
                    self._arc_frame(edge)
                except (ValueError, GeometryError) as exc:
                    raise GeometryError(
                        f"moving point {vertex_id} creates invalid arc geometry: {exc}"
                    ) from exc

        # A point-wise edit is not, in general, a rigid transform of an
        # analytical surface.  Boundary-backed Coons faces remain exact.
        # Non-mapped faces keep a still-valid explicit surface, or receive a
        # deterministic fitted plane when their edited boundary remains
        # planar; otherwise the transaction fails closed.
        for face_id in sorted(affected_faces):
            face = self.faces[face_id]
            if len(face.corners) == 4:
                self._put_entity(
                    "face", replace(face, surface=CoonsSurface())
                )
                continue
            if face.surface is not None and self._face_matches_surface(
                face_id, face.surface
            ):
                continue
            fitted = self._fit_face_plane(face_id)
            if fitted is None:
                raise GeometryError(
                    f"moving point {vertex_id} would leave face {face_id} "
                    "without a valid evaluable surface"
                )
            self._put_entity("face", replace(face, surface=fitted))

    def _face_matches_surface(self, face_id: int, surface: object) -> bool:
        if isinstance(surface, CoonsSurface) and not surface.has_boundaries:
            return True
        face = self.faces[face_id]
        try:
            for loop in (face.loop,) + face.holes:
                for point in self._validation_loop_points(loop):
                    uv = surface.local_uv(point)  # type: ignore[union-attr]
                    projected = np.asarray(surface.evaluate(*uv), dtype=float)  # type: ignore[union-attr]
                    extent = float(
                        np.linalg.norm(
                            np.ptp(self._validation_loop_points(loop), axis=0)
                        )
                    )
                    if float(np.linalg.norm(projected - point)) > self.tolerance.effective_surface_residual(extent):
                        return False
        except (AttributeError, ValueError, GeometryError, np.linalg.LinAlgError):
            return False
        return True

    def _fit_face_plane(self, face_id: int) -> Plane | None:
        face = self.faces[face_id]
        points = np.vstack(
            [
                self._validation_loop_points(loop)
                for loop in (face.loop,) + face.holes
            ]
        )
        centre = points.mean(axis=0)
        _values, singular, vectors = np.linalg.svd(points - centre)
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        if len(singular) >= 3 and float(singular[-1]) > self.tolerance.effective_surface_residual(extent):
            return None
        coordinates = np.column_stack(
            ((points - centre) @ vectors[0], (points - centre) @ vectors[1])
        )
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        spans = maximum - minimum
        if np.any(spans <= self.tolerance.effective_length(extent)):
            return None
        origin = centre + minimum[0] * vectors[0] + minimum[1] * vectors[1]
        return Plane(origin, spans[0] * vectors[0], spans[1] * vectors[1])

    def _spline_points(self, edge: Edge) -> np.ndarray:
        assert isinstance(edge.curve, Spline)
        ids = (edge.start,) + edge.curve.control_vertices + (edge.end,)
        return np.asarray([self.vertices[item].position for item in ids])

    @staticmethod
    def _split_bezier(points: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        levels = [np.asarray(points, dtype=float)]
        while len(levels[-1]) > 1:
            previous = levels[-1]
            levels.append((1.0 - t) * previous[:-1] + t * previous[1:])
        left = np.asarray([level[0] for level in levels])
        right = np.asarray([level[-1] for level in reversed(levels)])
        return left, right

    # ------------------------------------------------------------------
    # oriented traversal helpers
    # ------------------------------------------------------------------
    def oriented_start_vertex(self, oriented: OrientedEdge) -> int:
        edge = self._require_edge(oriented.edge)
        return edge.start if oriented.forward else edge.end

    def oriented_end_vertex(self, oriented: OrientedEdge) -> int:
        edge = self._require_edge(oriented.edge)
        return edge.end if oriented.forward else edge.start

    def oriented_start_tangent(self, oriented: OrientedEdge) -> np.ndarray:
        if oriented.forward:
            return self.edge_tangent(oriented.edge, 0.0)
        return -self.edge_tangent(oriented.edge, 1.0)

    def oriented_end_tangent(self, oriented: OrientedEdge) -> np.ndarray:
        if oriented.forward:
            return self.edge_tangent(oriented.edge, 1.0)
        return -self.edge_tangent(oriented.edge, 0.0)

    def face_corner_vertices(self, face_id: int) -> Tuple[int, ...]:
        """The four corner points of a face, in loop order."""

        face = self._require_face(face_id)
        return tuple(  # type: ignore[return-value]
            self.oriented_start_vertex(face.loop[index]) for index in face.corners
        )

    def side_length(self, side: Sequence[OrientedEdge]) -> float:
        return float(sum(self.edge_length(item.edge) for item in side))

    def face_point(self, face_id: int, u: float, v: float) -> np.ndarray:
        """Evaluate a face at local coordinates without a mesher dependency."""

        face = self._require_face(face_id)
        if face.surface is not None and (
            not isinstance(face.surface, CoonsSurface) or face.surface.has_boundaries
        ):
            return np.asarray(face.surface.evaluate(float(u), float(v)), dtype=float)
        if len(face.corners) != 4:
            raise GeometryError(
                f"face {face_id} has no explicit surface or four-side Coons mapping"
            )
        sides = face.sides()
        point_a = self._chain_point(sides[0], u)
        point_b = self._chain_point(sides[1], v)
        point_c = self._chain_point(sides[2], 1.0 - u)
        point_d = self._chain_point(sides[3], 1.0 - v)
        corner_00 = self._chain_point(sides[0], 0.0)
        corner_10 = self._chain_point(sides[0], 1.0)
        corner_11 = self._chain_point(sides[2], 0.0)
        corner_01 = self._chain_point(sides[2], 1.0)
        return (
            (1.0 - v) * point_a
            + v * point_c
            + (1.0 - u) * point_d
            + u * point_b
            - (
                (1.0 - u) * (1.0 - v) * corner_00
                + u * (1.0 - v) * corner_10
                + u * v * corner_11
                + (1.0 - u) * v * corner_01
            )
        )

    def face_local_uv(
        self, face_id: int, point: Sequence[float]
    ) -> Tuple[float, float]:
        """Return bounded closest local coordinates on a face."""

        face = self._require_face(face_id)
        if face.surface is not None and (
            not isinstance(face.surface, CoonsSurface) or face.surface.has_boundaries
        ):
            local = face.surface.local_uv(point)
            return (
                float(np.clip(local[0], 0.0, 1.0)),
                float(np.clip(local[1], 0.0, 1.0)),
            )

        model = self

        class _TopologySurface:
            def evaluate(self, u: float, v: float) -> np.ndarray:
                return model.face_point(face_id, u, v)

            def local_uv(self, candidate: object) -> Tuple[float, float]:
                return closest_uv(self, candidate)

        return closest_uv(_TopologySurface(), point)

    def project_to_face(
        self, face_id: int, point: Sequence[float]
    ) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """Project a point to a face and return point, UV and distance."""

        uv = self.face_local_uv(face_id, point)
        projected = self.face_point(face_id, *uv)
        if not self.face_contains_uv(face_id, uv):
            projected = self._closest_face_boundary_point(face_id, point)
            uv = self.face_local_uv(face_id, projected)
        distance = float(np.linalg.norm(projected - np.asarray(point, dtype=float)))
        return projected, uv, distance

    def face_contains_uv(self, face_id: int, uv: Sequence[float]) -> bool:
        """Whether local coordinates lie in the outer trim and outside holes."""

        face = self._require_face(face_id)
        candidate = np.asarray(uv, dtype=float)
        if candidate.shape != (2,):
            raise GeometryError("local coordinates must contain u and v")

        polygons = self.face_trim_loops_uv(face_id)
        if not self._point_in_polygon(candidate, polygons[0]):
            return False
        return not any(
            self._point_in_polygon(candidate, hole, include_boundary=False)
            for hole in polygons[1:]
        )

    def face_trim_loops_uv(
        self, face_id: int, *, curve_samples: int = 17
    ) -> Tuple[np.ndarray, ...]:
        """Return outer and hole trim loops in the face's local UV plane.

        This is the authoritative public bridge for trim-aware tessellation,
        hit testing, and planar export. Curves are sampled deterministically;
        straight segments contribute only their start vertex.
        """

        face = self._require_face(face_id)
        count = int(curve_samples)
        if count < 3:
            raise GeometryError("trim curve sampling needs at least three points")

        def polygon(loop: Sequence[OrientedEdge]) -> np.ndarray:
            points: List[np.ndarray] = []
            for item in loop:
                edge = self.edges[item.edge]
                samples = self.sample_edge(
                    item.edge,
                    np.linspace(
                        0.0,
                        1.0,
                        2 if isinstance(edge.curve, Straight) else count,
                    ),
                )
                if not item.forward:
                    samples = samples[::-1]
                points.extend(samples[:-1])
            return np.asarray(
                [self.face_local_uv(face_id, point) for point in points],
                dtype=float,
            )

        return tuple(polygon(loop) for loop in (face.loop,) + tuple(face.holes))

    @staticmethod
    def _point_in_polygon(
        point: np.ndarray, polygon: np.ndarray, *, include_boundary: bool = True
    ) -> bool:
        if len(polygon) < 3:
            return False
        x, y = float(point[0]), float(point[1])
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = float(previous[0]), float(previous[1])
            x2, y2 = float(current[0]), float(current[1])
            segment = np.asarray((x2 - x1, y2 - y1))
            offset = np.asarray((x - x1, y - y1))
            cross = abs(float(segment[0] * offset[1] - segment[1] * offset[0]))
            if cross <= 1.0e-10 and min(x1, x2)-1e-10 <= x <= max(x1, x2)+1e-10 and min(y1, y2)-1e-10 <= y <= max(y1, y2)+1e-10:
                return include_boundary
            if (y1 > y) != (y2 > y):
                crossing = x1 + (y-y1)*(x2-x1)/(y2-y1)
                if x < crossing:
                    inside = not inside
            previous = current
        return inside

    def _closest_face_boundary_point(
        self, face_id: int, point: Sequence[float]
    ) -> np.ndarray:
        target = np.asarray(point, dtype=float)
        face = self._require_face(face_id)
        best_distance = float("inf")
        best_point: np.ndarray | None = None
        for loop in (face.loop,) + face.holes:
            for item in loop:
                candidate, _parameter, distance = self.closest_edge_point(
                    item.edge, target
                )
                if distance < best_distance:
                    best_distance = distance
                    best_point = candidate
        if best_point is None:
            raise GeometryError(f"face {face_id} has no boundary")
        return best_point

    def face_normal(self, face_id: int, u: float, v: float) -> np.ndarray:
        """Deterministic unit normal, including topology-backed Coons faces."""

        step = 1.0e-6
        u0, v0 = float(u), float(v)
        point = self.face_point(face_id, u0, v0)
        du = self.face_point(face_id, min(1.0, u0 + step), v0) - point
        dv = self.face_point(face_id, u0, min(1.0, v0 + step)) - point
        if float(np.linalg.norm(du)) <= 0.0:
            du = point - self.face_point(face_id, max(0.0, u0 - step), v0)
        if float(np.linalg.norm(dv)) <= 0.0:
            dv = point - self.face_point(face_id, u0, max(0.0, v0 - step))
        normal = np.cross(du, dv)
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise GeometryError(f"face {face_id} has a degenerate normal")
        return normal / length

    def closest_face(
        self, point: Sequence[float], face_ids: Iterable[int] | None = None
    ) -> Tuple[int, np.ndarray, Tuple[float, float], float]:
        """Find the closest face with deterministic ID tie-breaking."""

        candidates = sorted(self.faces if face_ids is None else set(face_ids))
        if not candidates:
            raise GeometryError("closest_face needs at least one face")
        ranked = []
        for face_id in candidates:
            projected, uv, distance = self.project_to_face(face_id, point)
            ranked.append((distance, face_id, projected, uv))
        distance, face_id, projected, uv = min(ranked, key=lambda item: (item[0], item[1]))
        return face_id, projected, uv, distance

    def _chain_point(self, chain: Sequence[OrientedEdge], fraction: float) -> np.ndarray:
        if not chain:
            raise GeometryError("cannot evaluate an empty boundary chain")
        lengths = np.asarray([self.edge_length(item.edge) for item in chain])
        total = float(lengths.sum())
        if total <= 0.0:
            raise GeometryError("cannot evaluate a zero-length boundary chain")
        target = float(np.clip(fraction, 0.0, 1.0)) * total
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(chain) - 1)
        local = (target - cumulative[index]) / lengths[index]
        item = chain[index]
        parameter = local if item.forward else 1.0 - local
        return self.sample_edge(item.edge, np.asarray([parameter]))[0]

    # ------------------------------------------------------------------
    # loop ordering and corner detection
    # ------------------------------------------------------------------
    def _order_loop(self, edge_ids: Sequence[int]) -> Tuple[OrientedEdge, ...]:
        remaining = list(dict.fromkeys(int(e) for e in edge_ids))
        if len(remaining) < 3:
            raise GeometryError(
                "a face needs at least three edges forming a closed loop"
            )
        for edge_id in remaining:
            self._require_edge(edge_id)

        first = remaining.pop(0)
        loop = [OrientedEdge(first, True)]
        start_vertex = self.edges[first].start
        current = self.edges[first].end

        while remaining:
            for index, edge_id in enumerate(remaining):
                edge = self.edges[edge_id]
                if edge.start == current:
                    loop.append(OrientedEdge(edge_id, True))
                    current = edge.end
                elif edge.end == current:
                    loop.append(OrientedEdge(edge_id, False))
                    current = edge.start
                else:
                    continue
                remaining.pop(index)
                break
            else:
                raise GeometryError(
                    "edges do not form a single closed loop: "
                    f"no edge continues from vertex {current}"
                )

        if current != start_vertex:
            raise GeometryError(
                "edges do not form a closed loop: the chain ends at vertex "
                f"{current} but starts at vertex {start_vertex}"
            )
        return tuple(loop)

    def _detect_corners(
        self, loop: Tuple[OrientedEdge, ...]
    ) -> Tuple[int, int, int, int]:
        """Pick the four sharpest boundary turns as the mapped-face corners."""

        count = len(loop)
        if count < 4:
            raise GeometryError(
                f"a mapped face needs at least four edges, got {count}; "
                "split the boundary so it forms four sides"
            )
        if count == 4:
            return (0, 1, 2, 3)

        deviations = []
        for index in range(count):
            incoming = self.oriented_end_tangent(loop[index - 1])
            outgoing = self.oriented_start_tangent(loop[index])
            cosine = float(np.clip(incoming @ outgoing, -1.0, 1.0))
            deviations.append(float(np.arccos(cosine)))

        sharpest = sorted(
            sorted(range(count), key=lambda i: (-deviations[i], i))[:4]
        )
        return self._validate_corners(tuple(sharpest), count)

    @staticmethod
    def _validate_corners(
        corners: Tuple[int, ...], loop_length: int
    ) -> Tuple[int, int, int, int]:
        if len(corners) != 4:
            raise GeometryError("a mapped face needs exactly four corners")
        if len(set(corners)) != 4:
            raise GeometryError("face corners must be four distinct loop positions")
        if any(not 0 <= c < loop_length for c in corners):
            raise GeometryError("face corner index outside the boundary loop")
        if list(corners) != sorted(corners):
            raise GeometryError("face corners must be given in loop order")
        return tuple(corners)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def _require_vertex(self, vertex_id: int) -> Vertex:
        try:
            return self.vertices[vertex_id]
        except KeyError:
            raise GeometryError(f"no vertex {vertex_id}") from None

    def _require_edge(self, edge_id: int) -> Edge:
        try:
            return self.edges[edge_id]
        except KeyError:
            raise GeometryError(f"no edge {edge_id}") from None

    def _require_face(self, face_id: int) -> Face:
        try:
            return self.faces[face_id]
        except KeyError:
            raise GeometryError(f"no face {face_id}") from None

    def entity_ref(self, kind: str, entity_id: int) -> EntityRef:
        """Build a reference after checking the entity exists."""

        if kind == "vertex":
            self._require_vertex(entity_id)
        elif kind == "edge":
            self._require_edge(entity_id)
        elif kind == "face":
            self._require_face(entity_id)
        else:
            raise GeometryError(f"unknown entity kind {kind!r}")
        return EntityRef(kind, entity_id)  # type: ignore[arg-type]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"GeometryModel(vertices={len(self.vertices)}, "
            f"edges={len(self.edges)}, faces={len(self.faces)})"
        )
