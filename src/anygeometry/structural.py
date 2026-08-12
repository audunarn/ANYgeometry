"""Immutable structural ownership and topology value objects.

These records describe neutral structural geometry, not finite elements.  They
refer to geometry and to each other by compact positive integer IDs; public
cross-model APIs add model identity with :class:`anygeometry.identity.EntityHandle`.
The records are intentionally independent of ``GeometryModel`` so a model
transaction can journal them as replace-on-write values.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Integral, Real
from typing import Final, TypeAlias, Union

from .errors import GeometryError, GeometryTopologyError
from .identity import EntityKey, STRUCTURAL_ENTITY_KINDS, validate_local_id

__all__ = [
    "Attachment",
    "AttachmentKind",
    "AttachmentEvidence",
    "AttachmentTargetKind",
    "BoundaryPolicy",
    "Coedge",
    "ConnectivityPolicy",
    "ConnectionIntent",
    "FaceUse",
    "FrozenMetadata",
    "Junction",
    "JunctionKind",
    "JunctionMemberUse",
    "Member",
    "MemberEdgeUse",
    "NonManifoldPolicy",
    "Orientation",
    "ParameterRange",
    "Part",
    "Sheet",
    "SheetTopologyPolicy",
    "freeze_metadata",
    "raise_for_structural_topology",
    "replace_member_edge_use",
    "structural_entity_keys",
    "validate_structural_topology",
]


MetadataPrimitive: TypeAlias = None | bool | int | float | str
FrozenMetadataValue: TypeAlias = Union[
    MetadataPrimitive, tuple["FrozenMetadataValue", ...], "FrozenMetadata"
]


def _freeze_metadata_value(value: object, path: str) -> FrozenMetadataValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        made = float(value)
        if not isfinite(made):
            raise GeometryError(f"{path} must be finite")
        return made
    if isinstance(value, Mapping):
        return FrozenMetadata(value, _path=path)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata_value(item, f"{path}[{position}]")
            for position, item in enumerate(value)
        )
    raise GeometryError(
        f"{path} value of type {type(value).__name__} is not JSON metadata"
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenMetadata(Mapping[str, FrozenMetadataValue]):
    """A deterministic, deeply immutable JSON-compatible mapping."""

    _items: tuple[tuple[str, FrozenMetadataValue], ...]

    def __init__(
        self,
        values: Mapping[str, object] | FrozenMetadata | None = None,
        *,
        _path: str = "metadata",
    ) -> None:
        if values is None:
            items: tuple[tuple[str, FrozenMetadataValue], ...] = ()
        elif isinstance(values, FrozenMetadata):
            items = values._items
        elif isinstance(values, Mapping):
            made: list[tuple[str, FrozenMetadataValue]] = []
            for key, value in values.items():
                if not isinstance(key, str):
                    raise GeometryError(f"{_path} keys must be strings")
                if not key:
                    raise GeometryError(f"{_path} keys cannot be empty")
                made.append(
                    (key, _freeze_metadata_value(value, f"{_path}.{key}"))
                )
            made.sort(key=lambda item: item[0])
            items = tuple(made)
        else:
            raise GeometryError("metadata must be a mapping")
        object.__setattr__(self, "_items", items)

    def __getitem__(self, key: str) -> FrozenMetadataValue:
        # Metadata collections are deliberately small.  Linear lookup keeps the
        # value object compact, hashable, and free of a second mutable cache.
        for current, value in self._items:
            if current == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, object]:
        """Return a mutable JSON-ready copy for serialization."""

        def thaw(value: FrozenMetadataValue) -> object:
            if isinstance(value, FrozenMetadata):
                return value.to_dict()
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return value

        return {key: thaw(value) for key, value in self._items}


_EMPTY_METADATA: Final = FrozenMetadata()


def freeze_metadata(
    values: Mapping[str, object] | FrozenMetadata | None = None,
) -> FrozenMetadata:
    """Normalize metadata into a deterministic immutable value."""

    return values if isinstance(values, FrozenMetadata) else FrozenMetadata(values)


def _enum_value(enum_type: type[StrEnum], value: object, name: str) -> StrEnum:
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except (TypeError, ValueError) as error:
        raise GeometryError(f"invalid {name} {value!r}") from error


def _orientation(value: object) -> Orientation:
    if isinstance(value, bool):
        raise GeometryError("orientation must be FORWARD or REVERSED")
    try:
        return Orientation(value)
    except (TypeError, ValueError) as error:
        raise GeometryError("orientation must be FORWARD or REVERSED") from error


def _name(value: object, kind: str) -> str:
    if not isinstance(value, str):
        raise GeometryError(f"{kind} name must be a string")
    if "\x00" in value:
        raise GeometryError(f"{kind} name cannot contain a NUL character")
    return value


def _ids(
    values: Collection[int] | Sequence[int],
    name: str,
    *,
    sort: bool,
    require_nonempty: bool = False,
) -> tuple[int, ...]:
    if not sort and isinstance(values, (set, frozenset, Mapping)):
        raise GeometryError(f"ordered {name}s cannot be supplied as an unordered collection")
    try:
        made = tuple(validate_local_id(value, name=name) for value in values)
    except TypeError as error:
        raise GeometryError(f"{name}s must be an iterable of IDs") from error
    if require_nonempty and not made:
        raise GeometryError(f"at least one {name} is required")
    if len(set(made)) != len(made):
        raise GeometryError(f"duplicate {name}")
    return tuple(sorted(made)) if sort else made


def _finite_tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GeometryError("parameter tolerance must be a non-negative finite number")
    made = float(value)
    if not isfinite(made) or made < 0.0:
        raise GeometryError("parameter tolerance must be a non-negative finite number")
    return made


def _optional_id(value: object, name: str) -> int | None:
    if value is None:
        return None
    return validate_local_id(value, name=name)


def _non_negative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GeometryError(f"{name} must be a non-negative finite number")
    made = float(value)
    if not isfinite(made) or made < 0.0:
        raise GeometryError(f"{name} must be a non-negative finite number")
    return made


class Orientation(IntEnum):
    """Traversal orientation relative to an underlying geometry definition."""

    REVERSED = -1
    FORWARD = 1


class BoundaryPolicy(StrEnum):
    """Whether a sheet may contain free boundary edges."""

    ALLOW = "allow"
    REQUIRE_CLOSED = "require_closed"


class NonManifoldPolicy(StrEnum):
    """How radial edge incidence greater than two is handled."""

    REJECT = "reject"
    ALLOW_DECLARED = "allow_declared"


class ConnectivityPolicy(StrEnum):
    """Whether all face uses of a sheet must form one connected component."""

    REQUIRE_CONNECTED = "require_connected"
    ALLOW_DISCONNECTED = "allow_disconnected"


class ConnectionIntent(StrEnum):
    """Explicit physical intent for a qualified structural relationship."""

    CONNECT = "connect"
    KEEP_DISCONNECTED = "keep_disconnected"
    CONTACT_ONLY = "contact_only"
    REJECT = "reject"
    REUSE_EXISTING = "reuse_existing"
    IMPRINT = "imprint"


class AttachmentEvidence(StrEnum):
    """Qualification strength carried by a persistent attachment witness."""

    EXACT = "exact"
    VERIFIED_APPROXIMATE = "verified_approximate"
    UNVERIFIED = "unverified"


class AttachmentTargetKind(StrEnum):
    """Geometry definition qualified by an attachment."""

    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"
    MEMBER = "member"
    SHEET = "sheet"


class AttachmentKind(StrEnum):
    """Declared member-to-sheet or member-to-boundary relationship."""

    MEMBER_ON_FACE = "member_on_face"
    MEMBER_ON_BOUNDARY = "member_on_boundary"
    MEMBER_THROUGH_FACE = "member_through_face"
    ENDPOINT = "endpoint"
    VERTEX_ON_EDGE = "vertex_on_edge"
    VERTEX_ON_FACE = "vertex_on_face"
    MEMBER_ENDPOINT_ON_MEMBER = "member_endpoint_on_member"
    MEMBER_ENDPOINT_ON_SHEET = "member_endpoint_on_sheet"
    MEMBER_CROSS_SHEET = "member_cross_sheet"
    MEMBER_ON_SHEET = "member_on_sheet"
    MEMBER_ON_FACE_BOUNDARY = "member_on_face_boundary"
    MEMBER_ON_SHEET_INTERSECTION = "member_on_sheet_intersection"
    COINCIDENT_MEMBER_AXES = "coincident_member_axes"
    INTENTIONALLY_DISCONNECTED = "intentionally_disconnected"


class JunctionKind(StrEnum):
    """Intent carried by a multi-member and/or member-sheet junction."""

    ENDPOINT = "endpoint"
    CROSSING = "crossing"
    OVERLAP = "overlap"
    MULTI_WAY = "multi_way"


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """Closed normalized parent-parameter interval.

    Degenerate intervals intentionally represent points.  Axis edge uses must
    be non-degenerate, while attachments and junctions may use either form.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise GeometryError(f"parameter {name} must be finite")
            made = float(value)
            if not isfinite(made):
                raise GeometryError(f"parameter {name} must be finite")
            object.__setattr__(self, name, made)
        if self.start < 0.0 or self.end > 1.0 or self.start > self.end:
            raise GeometryError("parameter range must satisfy 0 <= start <= end <= 1")

    @classmethod
    def point(cls, parameter: float) -> ParameterRange:
        """Construct a degenerate range at one normalized parameter."""

        return cls(parameter, parameter)

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def is_point(self) -> bool:
        return self.start == self.end

    def contains(self, parameter: float, *, tolerance: float = 0.0) -> bool:
        made_tolerance = _finite_tolerance(tolerance)
        if isinstance(parameter, bool) or not isinstance(parameter, Real):
            return False
        made = float(parameter)
        return isfinite(made) and (
            self.start - made_tolerance <= made <= self.end + made_tolerance
        )


_FULL_PARAMETER_RANGE: Final = ParameterRange(0.0, 1.0)


def _parameter_range(value: object, name: str) -> ParameterRange:
    if isinstance(value, ParameterRange):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return ParameterRange(value[0], value[1])  # type: ignore[arg-type]
        except GeometryError as error:
            raise GeometryError(f"invalid {name}: {error}") from error
    raise GeometryError(f"{name} must be a ParameterRange or pair")


@dataclass(frozen=True, slots=True)
class SheetTopologyPolicy:
    """Committed topology rules for one structural sheet."""

    boundary: BoundaryPolicy | str = BoundaryPolicy.ALLOW
    non_manifold: NonManifoldPolicy | str = NonManifoldPolicy.REJECT
    connectivity: ConnectivityPolicy | str = ConnectivityPolicy.REQUIRE_CONNECTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boundary",
            _enum_value(BoundaryPolicy, self.boundary, "boundary policy"),
        )
        object.__setattr__(
            self,
            "non_manifold",
            _enum_value(
                NonManifoldPolicy, self.non_manifold, "non-manifold policy"
            ),
        )
        object.__setattr__(
            self,
            "connectivity",
            _enum_value(
                ConnectivityPolicy, self.connectivity, "connectivity policy"
            ),
        )


_DEFAULT_SHEET_POLICY: Final = SheetTopologyPolicy()


@dataclass(frozen=True, slots=True)
class Part:
    """Persistent ownership boundary for sheets and structural members."""

    id: int
    sheet_ids: tuple[int, ...] = ()
    member_ids: tuple[int, ...] = ()
    name: str = ""
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="part ID"))
        object.__setattr__(self, "sheet_ids", _ids(self.sheet_ids, "sheet ID", sort=True))
        object.__setattr__(
            self, "member_ids", _ids(self.member_ids, "member ID", sort=True)
        )
        object.__setattr__(self, "name", _name(self.name, "part"))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Sheet:
    """A part-owned collection of oriented face uses."""

    id: int
    part_id: int
    face_use_ids: tuple[int, ...]
    policy: SheetTopologyPolicy = _DEFAULT_SHEET_POLICY
    declared_non_manifold_edges: tuple[int, ...] = ()
    name: str = ""
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="sheet ID"))
        object.__setattr__(
            self, "part_id", validate_local_id(self.part_id, name="part ID")
        )
        object.__setattr__(
            self,
            "face_use_ids",
            _ids(self.face_use_ids, "face-use ID", sort=True, require_nonempty=True),
        )
        if not isinstance(self.policy, SheetTopologyPolicy):
            raise GeometryError("sheet policy must be a SheetTopologyPolicy")
        object.__setattr__(
            self,
            "declared_non_manifold_edges",
            _ids(
                self.declared_non_manifold_edges,
                "declared non-manifold edge ID",
                sort=True,
            ),
        )
        object.__setattr__(self, "name", _name(self.name, "sheet"))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class FaceUse:
    """One oriented use of a geometry face within one sheet."""

    id: int
    sheet_id: int
    face_id: int
    loops: tuple[tuple[int, ...], ...]
    orientation: Orientation = Orientation.FORWARD
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="face-use ID"))
        object.__setattr__(
            self, "sheet_id", validate_local_id(self.sheet_id, name="sheet ID")
        )
        object.__setattr__(
            self, "face_id", validate_local_id(self.face_id, name="face ID")
        )
        try:
            if isinstance(self.loops, (set, frozenset, Mapping)):
                raise GeometryError(
                    "face-use loops cannot be supplied as an unordered collection"
                )
            loops = tuple(
                _ids(loop, "coedge ID", sort=False, require_nonempty=True)
                for loop in self.loops
            )
        except TypeError as error:
            raise GeometryError("face-use loops must be an iterable of loops") from error
        if not loops:
            raise GeometryError("face use requires an outer coedge loop")
        if any(len(loop) < 3 for loop in loops):
            raise GeometryError("each face-use loop requires at least three coedges")
        flattened = tuple(item for loop in loops for item in loop)
        if len(set(flattened)) != len(flattened):
            raise GeometryError("a coedge can occur only once in a face use")
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "orientation", _orientation(self.orientation))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def coedge_ids(self) -> tuple[int, ...]:
        """All coedges in deterministic outer-loop then inner-loop order."""

        return tuple(item for loop in self.loops for item in loop)


@dataclass(frozen=True, slots=True)
class Coedge:
    """Persistent oriented use of one geometry edge in one face loop."""

    id: int
    face_use_id: int
    edge_id: int
    orientation: Orientation = Orientation.FORWARD
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="coedge ID"))
        object.__setattr__(
            self,
            "face_use_id",
            validate_local_id(self.face_use_id, name="face-use ID"),
        )
        object.__setattr__(
            self, "edge_id", validate_local_id(self.edge_id, name="edge ID")
        )
        object.__setattr__(self, "orientation", _orientation(self.orientation))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Member:
    """Persistent physical member with an ordered geometry-edge axis."""

    id: int
    part_id: int
    edge_use_ids: tuple[int, ...]
    name: str = ""
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA
    orientation_reference: EntityKey | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="member ID"))
        object.__setattr__(
            self, "part_id", validate_local_id(self.part_id, name="part ID")
        )
        object.__setattr__(
            self,
            "edge_use_ids",
            _ids(
                self.edge_use_ids,
                "member-edge-use ID",
                sort=False,
                require_nonempty=True,
            ),
        )
        object.__setattr__(self, "name", _name(self.name, "member"))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        if self.orientation_reference is not None:
            try:
                kind, identifier = self.orientation_reference
            except (TypeError, ValueError) as error:
                raise GeometryError(
                    "member orientation reference must be a (kind, ID) pair"
                ) from error
            if kind not in ("vertex", "edge", "face"):
                raise GeometryError(
                    "member orientation reference must name geometry"
                )
            object.__setattr__(
                self,
                "orientation_reference",
                (kind, validate_local_id(identifier, name="orientation reference ID")),
            )


@dataclass(frozen=True, slots=True)
class MemberEdgeUse:
    """One oriented edge interval in a persistent member axis."""

    id: int
    member_id: int
    edge_id: int
    parent_range: ParameterRange = _FULL_PARAMETER_RANGE
    orientation: Orientation = Orientation.FORWARD
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "id", validate_local_id(self.id, name="member-edge-use ID")
        )
        object.__setattr__(
            self, "member_id", validate_local_id(self.member_id, name="member ID")
        )
        object.__setattr__(
            self, "edge_id", validate_local_id(self.edge_id, name="edge ID")
        )
        made_range = _parameter_range(self.parent_range, "parent range")
        if made_range.is_point:
            raise GeometryError("member-edge-use parent range cannot be degenerate")
        object.__setattr__(self, "parent_range", made_range)
        object.__setattr__(self, "orientation", _orientation(self.orientation))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def parameter_range(self) -> ParameterRange:
        """Compatibility spelling for callers that do not distinguish parents."""

        return self.parent_range


@dataclass(frozen=True, slots=True)
class Attachment:
    """Qualified member incidence with one geometry face or edge."""

    id: int
    member_id: int | None
    kind: AttachmentKind | str
    target_kind: AttachmentTargetKind | str
    target_id: int
    member_range: ParameterRange
    target_parameters: tuple[ParameterRange, ...]
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA
    connection_intent: ConnectionIntent | str = ConnectionIntent.CONNECT
    evidence: AttachmentEvidence | str = AttachmentEvidence.UNVERIFIED
    max_residual: float = 0.0
    tolerance_used: float = 0.0
    part_id: int | None = None
    sheet_id: int | None = None
    provenance: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA
    lineage: tuple[EntityKey, ...] = ()
    source_kind: str | None = None
    source_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="attachment ID"))
        member_id = _optional_id(self.member_id, "member ID")
        source_kind = "member" if self.source_kind is None else str(self.source_kind)
        source_id = member_id if self.source_id is None else validate_local_id(
            self.source_id, name="attachment source ID"
        )
        if source_kind not in ("vertex", "edge", "face", "member", "sheet"):
            raise GeometryError("invalid attachment source kind")
        if source_id is None:
            raise GeometryError("attachment source requires an ID")
        if source_kind == "member":
            if member_id is None:
                member_id = source_id
            elif member_id != source_id:
                raise GeometryError("attachment member/source IDs disagree")
        elif member_id is not None:
            raise GeometryError("non-member attachment source cannot carry member_id")
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_id", source_id)
        kind = _enum_value(AttachmentKind, self.kind, "attachment kind")
        target_kind = _enum_value(
            AttachmentTargetKind, self.target_kind, "attachment target kind"
        )
        expected_source = {
            AttachmentKind.MEMBER_ON_FACE: "member",
            AttachmentKind.MEMBER_ON_BOUNDARY: "member",
            AttachmentKind.MEMBER_THROUGH_FACE: "member",
            AttachmentKind.ENDPOINT: "member",
            AttachmentKind.VERTEX_ON_EDGE: "vertex",
            AttachmentKind.VERTEX_ON_FACE: "vertex",
            AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER: "member",
            AttachmentKind.MEMBER_ENDPOINT_ON_SHEET: "member",
            AttachmentKind.MEMBER_CROSS_SHEET: "member",
            AttachmentKind.MEMBER_ON_SHEET: "member",
            AttachmentKind.MEMBER_ON_FACE_BOUNDARY: "member",
            AttachmentKind.MEMBER_ON_SHEET_INTERSECTION: "member",
            AttachmentKind.COINCIDENT_MEMBER_AXES: "member",
        }.get(kind)
        if expected_source is not None and source_kind != expected_source:
            raise GeometryError(
                f"{kind.value} requires a {expected_source} source"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(
            self,
            "connection_intent",
            _enum_value(
                ConnectionIntent,
                self.connection_intent,
                "attachment connection intent",
            ),
        )
        object.__setattr__(
            self,
            "evidence",
            _enum_value(AttachmentEvidence, self.evidence, "attachment evidence"),
        )
        object.__setattr__(
            self, "target_id", validate_local_id(self.target_id, name="target ID")
        )
        member_range = _parameter_range(self.member_range, "member range")
        try:
            if isinstance(self.target_parameters, (set, frozenset, Mapping)):
                raise GeometryError(
                    "target parameters cannot be supplied as an unordered collection"
                )
            target_parameters = tuple(
                _parameter_range(value, "target parameter range")
                for value in self.target_parameters
            )
        except TypeError as error:
            raise GeometryError("target parameters must be an iterable of ranges") from error
        expected_parameters = {
            AttachmentTargetKind.FACE: 2,
            AttachmentTargetKind.SHEET: 2,
            AttachmentTargetKind.VERTEX: 0,
        }.get(target_kind, 1)
        if len(target_parameters) != expected_parameters:
            raise GeometryError(
                f"{target_kind.value} attachment requires "
                f"{expected_parameters} target parameter range(s)"
            )
        expected_target = {
            AttachmentKind.MEMBER_ON_FACE: AttachmentTargetKind.FACE,
            AttachmentKind.MEMBER_ON_BOUNDARY: AttachmentTargetKind.EDGE,
            AttachmentKind.MEMBER_THROUGH_FACE: AttachmentTargetKind.FACE,
            AttachmentKind.VERTEX_ON_EDGE: AttachmentTargetKind.EDGE,
            AttachmentKind.VERTEX_ON_FACE: AttachmentTargetKind.FACE,
            AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER: AttachmentTargetKind.MEMBER,
            AttachmentKind.MEMBER_ENDPOINT_ON_SHEET: AttachmentTargetKind.SHEET,
            AttachmentKind.MEMBER_CROSS_SHEET: AttachmentTargetKind.SHEET,
            AttachmentKind.MEMBER_ON_SHEET: AttachmentTargetKind.SHEET,
            AttachmentKind.MEMBER_ON_FACE_BOUNDARY: AttachmentTargetKind.EDGE,
            AttachmentKind.MEMBER_ON_SHEET_INTERSECTION: AttachmentTargetKind.EDGE,
            AttachmentKind.COINCIDENT_MEMBER_AXES: AttachmentTargetKind.MEMBER,
        }.get(kind)
        if expected_target is not None and target_kind is not expected_target:
            raise GeometryError(
                f"{kind.value} requires a {expected_target.value} target"
            )
        if kind is AttachmentKind.MEMBER_THROUGH_FACE:
            if not member_range.is_point or any(
                not value.is_point for value in target_parameters
            ):
                raise GeometryError("member-through-face attachment must be point-valued")
        if kind in (
            AttachmentKind.ENDPOINT,
            AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER,
            AttachmentKind.MEMBER_ENDPOINT_ON_SHEET,
        ):
            if not member_range.is_point or member_range.start not in (0.0, 1.0):
                raise GeometryError("endpoint attachment must use member parameter 0 or 1")
            if any(not value.is_point for value in target_parameters):
                raise GeometryError("endpoint attachment target must be point-valued")
        object.__setattr__(self, "member_range", member_range)
        object.__setattr__(self, "target_parameters", target_parameters)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "max_residual",
            _non_negative_finite(self.max_residual, "attachment maximum residual"),
        )
        object.__setattr__(
            self,
            "tolerance_used",
            _non_negative_finite(self.tolerance_used, "attachment tolerance"),
        )
        if self.evidence is not AttachmentEvidence.UNVERIFIED:
            if self.max_residual > self.tolerance_used:
                raise GeometryError(
                    "verified attachment residual exceeds its qualification tolerance"
                )
            if (
                self.evidence is AttachmentEvidence.VERIFIED_APPROXIMATE
                and self.tolerance_used == 0.0
            ):
                raise GeometryError(
                    "verified approximate attachment requires a positive tolerance"
                )
        object.__setattr__(self, "part_id", _optional_id(self.part_id, "part ID"))
        object.__setattr__(self, "sheet_id", _optional_id(self.sheet_id, "sheet ID"))
        object.__setattr__(self, "provenance", freeze_metadata(self.provenance))
        try:
            lineage = tuple(
                (
                    str(key[0]),
                    validate_local_id(key[1], name="lineage entity ID"),
                )
                for key in self.lineage
            )
        except (TypeError, IndexError) as error:
            raise GeometryError("attachment lineage must contain (kind, ID) pairs") from error
        if any(not kind for kind, _identifier in lineage):
            raise GeometryError("attachment lineage kinds must be non-empty")
        object.__setattr__(self, "lineage", lineage)

    @property
    def target_key(self) -> EntityKey:
        return (self.target_kind.value, self.target_id)

    @property
    def source_key(self) -> EntityKey:
        assert self.source_kind is not None and self.source_id is not None
        return (self.source_kind, self.source_id)


@dataclass(frozen=True, slots=True)
class JunctionMemberUse:
    """One member's participating parent interval at a junction."""

    member_id: int
    member_range: ParameterRange

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "member_id", validate_local_id(self.member_id, name="member ID")
        )
        object.__setattr__(
            self, "member_range", _parameter_range(self.member_range, "member range")
        )

    @property
    def sort_key(self) -> tuple[int, float, float]:
        return (self.member_id, self.member_range.start, self.member_range.end)


@dataclass(frozen=True, slots=True)
class Junction:
    """Explicit endpoint, crossing, overlap, or multi-way connection intent."""

    id: int
    kind: JunctionKind | str
    member_uses: tuple[JunctionMemberUse, ...]
    sheet_ids: tuple[int, ...] = ()
    attachment_ids: tuple[int, ...] = ()
    metadata: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA
    connection_intent: ConnectionIntent | str = ConnectionIntent.CONNECT
    provenance: FrozenMetadata | Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_local_id(self.id, name="junction ID"))
        kind = _enum_value(JunctionKind, self.kind, "junction kind")
        object.__setattr__(self, "kind", kind)
        intent = _enum_value(
            ConnectionIntent, self.connection_intent, "junction connection intent"
        )
        object.__setattr__(self, "connection_intent", intent)
        try:
            member_uses = tuple(self.member_uses)
        except TypeError as error:
            raise GeometryError("junction member uses must be iterable") from error
        if any(not isinstance(value, JunctionMemberUse) for value in member_uses):
            raise GeometryError("junction member uses must be JunctionMemberUse values")
        member_uses = tuple(sorted(member_uses, key=lambda value: value.sort_key))
        member_ids = tuple(value.member_id for value in member_uses)
        if len(set(member_ids)) != len(member_ids):
            raise GeometryError("a member can participate only once in a junction")
        if not member_uses:
            raise GeometryError("junction requires at least one member")
        sheet_ids = _ids(self.sheet_ids, "sheet ID", sort=True)
        attachment_ids = _ids(self.attachment_ids, "attachment ID", sort=True)
        participant_count = len(member_uses) + len(sheet_ids)
        if participant_count < 2:
            raise GeometryError("junction requires at least two participants")
        if sheet_ids and not attachment_ids:
            raise GeometryError("member-sheet junction requires an attachment")
        if kind is JunctionKind.ENDPOINT and any(
            not use.member_range.is_point
            or use.member_range.start not in (0.0, 1.0)
            for use in member_uses
        ):
            raise GeometryError("endpoint junction member parameters must be 0 or 1")
        if kind is JunctionKind.CROSSING and any(
            not use.member_range.is_point for use in member_uses
        ):
            raise GeometryError("crossing junction must be point-valued")
        if kind is JunctionKind.OVERLAP:
            if (
                any(use.member_range.is_point for use in member_uses)
                or (len(member_uses) < 2 and not sheet_ids)
            ):
                raise GeometryError(
                    "overlap junction requires non-degenerate ranges "
                    "with either two members or a participating sheet"
                )
        if kind is JunctionKind.MULTI_WAY and participant_count < 3:
            raise GeometryError("multi-way junction requires at least three participants")
        object.__setattr__(self, "member_uses", member_uses)
        object.__setattr__(self, "sheet_ids", sheet_ids)
        object.__setattr__(self, "attachment_ids", attachment_ids)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "provenance", freeze_metadata(self.provenance))

    @property
    def member_ids(self) -> tuple[int, ...]:
        return tuple(value.member_id for value in self.member_uses)


def structural_entity_keys(
    *,
    parts: Mapping[int, Part] | None = None,
    sheets: Mapping[int, Sheet] | None = None,
    face_uses: Mapping[int, FaceUse] | None = None,
    coedges: Mapping[int, Coedge] | None = None,
    members: Mapping[int, Member] | None = None,
    member_edge_uses: Mapping[int, MemberEdgeUse] | None = None,
    attachments: Mapping[int, Attachment] | None = None,
    junctions: Mapping[int, Junction] | None = None,
) -> tuple[EntityKey, ...]:
    """Return structural store keys in canonical kind/ID order."""

    stores: Final[tuple[tuple[str, Mapping[int, object]], ...]] = (
        ("part", {} if parts is None else parts),
        ("sheet", {} if sheets is None else sheets),
        ("face_use", {} if face_uses is None else face_uses),
        ("coedge", {} if coedges is None else coedges),
        ("member", {} if members is None else members),
        (
            "member_edge_use",
            {} if member_edge_uses is None else member_edge_uses,
        ),
        ("attachment", {} if attachments is None else attachments),
        ("junction", {} if junctions is None else junctions),
    )
    assert tuple(kind for kind, _store in stores) == STRUCTURAL_ENTITY_KINDS
    return tuple(
        (kind, identifier)
        for kind, store in stores
        for identifier in sorted(store)
    )


def _checked_store(
    name: str,
    store: Mapping[int, object],
    record_type: type,
    errors: list[str],
) -> dict[int, object]:
    checked: dict[int, object] = {}
    for key, record in sorted(store.items(), key=lambda item: repr(item[0])):
        try:
            identifier = validate_local_id(key, name=f"{name} store key")
        except GeometryError as error:
            errors.append(str(error))
            continue
        if not isinstance(record, record_type):
            errors.append(
                f"{name} store {identifier} has {type(record).__name__}, "
                f"expected {record_type.__name__}"
            )
            continue
        if record.id != identifier:
            errors.append(
                f"{name} store key {identifier} does not match record ID {record.id}"
            )
            continue
        checked[identifier] = record
    return checked


def _known_ids(
    values: Collection[int] | None, name: str, errors: list[str]
) -> set[int] | None:
    if values is None:
        return None
    made: set[int] = set()
    for value in values:
        try:
            made.add(validate_local_id(value, name=name))
        except GeometryError as error:
            errors.append(str(error))
    return made


def _oriented_vertices(
    edge_id: int,
    orientation: Orientation,
    edge_vertices: Mapping[int, tuple[int, int]],
) -> tuple[int, int] | None:
    vertices = edge_vertices.get(edge_id)
    if vertices is None:
        return None
    return vertices if orientation is Orientation.FORWARD else (vertices[1], vertices[0])


def validate_structural_topology(
    *,
    parts: Mapping[int, Part],
    sheets: Mapping[int, Sheet],
    face_uses: Mapping[int, FaceUse],
    coedges: Mapping[int, Coedge],
    members: Mapping[int, Member],
    member_edge_uses: Mapping[int, MemberEdgeUse],
    attachments: Mapping[int, Attachment],
    junctions: Mapping[int, Junction],
    edge_ids: Collection[int] | None = None,
    face_ids: Collection[int] | None = None,
    vertex_ids: Collection[int] | None = None,
    edge_vertices: Mapping[int, tuple[int, int]] | None = None,
    parameter_tolerance: float = 1.0e-12,
) -> tuple[str, ...]:
    """Validate all structural ownership and incidence deterministically.

    Geometry existence is checked when ``edge_ids``/``face_ids`` are supplied.
    Supplying ``edge_vertices`` additionally checks face-loop and member-axis
    continuity and implies the set of known edge IDs.
    """

    tolerance = _finite_tolerance(parameter_tolerance)
    errors: list[str] = []
    checked_parts = _checked_store("part", parts, Part, errors)
    checked_sheets = _checked_store("sheet", sheets, Sheet, errors)
    checked_face_uses = _checked_store("face-use", face_uses, FaceUse, errors)
    checked_coedges = _checked_store("coedge", coedges, Coedge, errors)
    checked_members = _checked_store("member", members, Member, errors)
    checked_member_uses = _checked_store(
        "member-edge-use", member_edge_uses, MemberEdgeUse, errors
    )
    checked_attachments = _checked_store(
        "attachment", attachments, Attachment, errors
    )
    checked_junctions = _checked_store("junction", junctions, Junction, errors)
    part_sheet_ids = {
        identifier: set(record.sheet_ids)
        for identifier, record in checked_parts.items()
        if isinstance(record, Part)
    }
    part_member_ids = {
        identifier: set(record.member_ids)
        for identifier, record in checked_parts.items()
        if isinstance(record, Part)
    }
    sheet_face_use_ids = {
        identifier: set(record.face_use_ids)
        for identifier, record in checked_sheets.items()
        if isinstance(record, Sheet)
    }

    known_edges = _known_ids(edge_ids, "edge ID", errors)
    known_faces = _known_ids(face_ids, "face ID", errors)
    known_vertices = _known_ids(vertex_ids, "vertex ID", errors)
    if edge_vertices is not None:
        normalized_vertices: dict[int, tuple[int, int]] = {}
        for raw_edge, raw_vertices in sorted(
            edge_vertices.items(), key=lambda item: repr(item[0])
        ):
            try:
                edge_id = validate_local_id(raw_edge, name="edge ID")
                if not isinstance(raw_vertices, (tuple, list)) or len(raw_vertices) != 2:
                    raise GeometryError(
                        f"edge {edge_id} endpoints must contain two vertex IDs"
                    )
                vertices = (
                    validate_local_id(raw_vertices[0], name="vertex ID"),
                    validate_local_id(raw_vertices[1], name="vertex ID"),
                )
                if vertices[0] == vertices[1]:
                    raise GeometryError(f"edge {edge_id} has identical endpoints")
                normalized_vertices[edge_id] = vertices
            except GeometryError as error:
                errors.append(str(error))
        edge_vertices = normalized_vertices
        endpoint_vertices = {
            vertex for pair in normalized_vertices.values() for vertex in pair
        }
        if known_vertices is None:
            known_vertices = endpoint_vertices
        elif not endpoint_vertices.issubset(known_vertices):
            errors.append("edge_vertices references vertices absent from vertex_ids")
        vertex_edge_ids = set(edge_vertices)
        if known_edges is None:
            known_edges = vertex_edge_ids
        elif known_edges != vertex_edge_ids:
            errors.append("edge_ids and edge_vertices describe different edge sets")

    # Part ownership is stored bidirectionally so local replacement can update
    # one compact parent record while validation catches stale reverse links.
    for part_id, raw_part in checked_parts.items():
        part = raw_part  # type narrowing for readers and static tools
        assert isinstance(part, Part)
        for sheet_id in part.sheet_ids:
            sheet = checked_sheets.get(sheet_id)
            if sheet is None:
                errors.append(f"part {part_id} references missing sheet {sheet_id}")
            elif isinstance(sheet, Sheet) and sheet.part_id != part_id:
                errors.append(
                    f"part {part_id} lists sheet {sheet_id} owned by part {sheet.part_id}"
                )
        for member_id in part.member_ids:
            member = checked_members.get(member_id)
            if member is None:
                errors.append(f"part {part_id} references missing member {member_id}")
            elif isinstance(member, Member) and member.part_id != part_id:
                errors.append(
                    f"part {part_id} lists member {member_id} owned by part {member.part_id}"
                )
    for sheet_id, raw_sheet in checked_sheets.items():
        sheet = raw_sheet
        assert isinstance(sheet, Sheet)
        part = checked_parts.get(sheet.part_id)
        if part is None:
            errors.append(f"sheet {sheet_id} references missing part {sheet.part_id}")
        elif sheet_id not in part_sheet_ids[sheet.part_id]:
            errors.append(f"sheet {sheet_id} is not listed by owning part {sheet.part_id}")
    for member_id, raw_member in checked_members.items():
        member = raw_member
        assert isinstance(member, Member)
        part = checked_parts.get(member.part_id)
        if part is None:
            errors.append(f"member {member_id} references missing part {member.part_id}")
        elif member_id not in part_member_ids[member.part_id]:
            errors.append(
                f"member {member_id} is not listed by owning part {member.part_id}"
            )
        if member.orientation_reference is not None:
            kind, identifier = member.orientation_reference
            if kind == "edge" and known_edges is not None and identifier not in known_edges:
                errors.append(
                    f"member {member_id} references missing orientation edge {identifier}"
                )
            elif kind == "face" and known_faces is not None and identifier not in known_faces:
                errors.append(
                    f"member {member_id} references missing orientation face {identifier}"
                )

    # Face-use/coedge ownership and optional geometry continuity.
    used_coedges: dict[int, int] = {}
    face_owner: dict[int, int] = {}
    for face_use_id, raw_face_use in checked_face_uses.items():
        face_use = raw_face_use
        assert isinstance(face_use, FaceUse)
        sheet = checked_sheets.get(face_use.sheet_id)
        if sheet is None:
            errors.append(
                f"face use {face_use_id} references missing sheet {face_use.sheet_id}"
            )
        elif face_use_id not in sheet_face_use_ids[face_use.sheet_id]:
            errors.append(
                f"face use {face_use_id} is not listed by owning sheet {face_use.sheet_id}"
            )
        if known_faces is not None and face_use.face_id not in known_faces:
            errors.append(
                f"face use {face_use_id} references missing face {face_use.face_id}"
            )
        previous_owner = face_owner.setdefault(face_use.face_id, face_use_id)
        if previous_owner != face_use_id:
            errors.append(
                f"face {face_use.face_id} has multiple structural uses "
                f"{previous_owner} and {face_use_id}"
            )
        for loop_number, loop in enumerate(face_use.loops):
            oriented: list[tuple[int, int] | None] = []
            loop_edges: set[int] = set()
            for coedge_id in loop:
                previous_use = used_coedges.setdefault(coedge_id, face_use_id)
                if previous_use != face_use_id:
                    errors.append(
                        f"coedge {coedge_id} occurs in face uses "
                        f"{previous_use} and {face_use_id}"
                    )
                coedge = checked_coedges.get(coedge_id)
                if coedge is None:
                    errors.append(
                        f"face use {face_use_id} references missing coedge {coedge_id}"
                    )
                    oriented.append(None)
                    continue
                assert isinstance(coedge, Coedge)
                if coedge.face_use_id != face_use_id:
                    errors.append(
                        f"face use {face_use_id} lists coedge {coedge_id} owned by "
                        f"face use {coedge.face_use_id}"
                    )
                if coedge.edge_id in loop_edges:
                    errors.append(
                        f"face use {face_use_id} loop {loop_number} uses edge "
                        f"{coedge.edge_id} more than once"
                    )
                loop_edges.add(coedge.edge_id)
                if known_edges is not None and coedge.edge_id not in known_edges:
                    errors.append(
                        f"coedge {coedge_id} references missing edge {coedge.edge_id}"
                    )
                oriented.append(
                    None
                    if edge_vertices is None
                    else _oriented_vertices(
                        coedge.edge_id, coedge.orientation, edge_vertices
                    )
                )
            if edge_vertices is not None:
                for position, (current, following) in enumerate(
                    zip(oriented, oriented[1:] + oriented[:1])
                ):
                    if current is not None and following is not None and current[1] != following[0]:
                        errors.append(
                            f"face use {face_use_id} loop {loop_number} is not "
                            f"continuous after coedge position {position}"
                        )
    for coedge_id, raw_coedge in checked_coedges.items():
        coedge = raw_coedge
        assert isinstance(coedge, Coedge)
        if coedge_id not in used_coedges:
            errors.append(f"coedge {coedge_id} is not used by a face loop")

    edge_to_sheets: dict[int, set[int]] = {}
    face_to_sheet: dict[int, int] = {}
    for sheet_id, raw_sheet in checked_sheets.items():
        sheet = raw_sheet
        assert isinstance(sheet, Sheet)
        radial: dict[int, list[tuple[int, int]]] = {}
        face_nodes: set[int] = set()
        adjacency: dict[int, set[int]] = {}
        for face_use_id in sheet.face_use_ids:
            face_use = checked_face_uses.get(face_use_id)
            if face_use is None:
                errors.append(
                    f"sheet {sheet_id} references missing face use {face_use_id}"
                )
                continue
            assert isinstance(face_use, FaceUse)
            if face_use.sheet_id != sheet_id:
                errors.append(
                    f"sheet {sheet_id} lists face use {face_use_id} owned by "
                    f"sheet {face_use.sheet_id}"
                )
                continue
            face_nodes.add(face_use_id)
            adjacency.setdefault(face_use_id, set())
            face_to_sheet[face_use.face_id] = sheet_id
            for coedge_id in face_use.coedge_ids:
                coedge = checked_coedges.get(coedge_id)
                if not isinstance(coedge, Coedge):
                    continue
                effective = int(face_use.orientation) * int(coedge.orientation)
                radial.setdefault(coedge.edge_id, []).append(
                    (face_use_id, effective)
                )
                edge_to_sheets.setdefault(coedge.edge_id, set()).add(sheet_id)
        for edge_id, uses in sorted(radial.items()):
            if len(uses) == 1 and sheet.policy.boundary is BoundaryPolicy.REQUIRE_CLOSED:
                errors.append(f"closed sheet {sheet_id} has boundary edge {edge_id}")
            if len(uses) == 2:
                first, second = uses
                adjacency[first[0]].add(second[0])
                adjacency[second[0]].add(first[0])
                if first[1] == second[1]:
                    errors.append(
                        f"sheet {sheet_id} has inconsistent orientation at edge {edge_id}"
                    )
            elif len(uses) > 2:
                for first, _first_direction in uses:
                    adjacency[first].update(
                        second for second, _direction in uses if second != first
                    )
                if (
                    sheet.policy.non_manifold is NonManifoldPolicy.REJECT
                    or edge_id not in sheet.declared_non_manifold_edges
                ):
                    errors.append(
                        f"sheet {sheet_id} has undeclared non-manifold edge {edge_id}"
                    )
        for edge_id in sheet.declared_non_manifold_edges:
            if len(radial.get(edge_id, ())) <= 2:
                errors.append(
                    f"sheet {sheet_id} declares edge {edge_id} non-manifold "
                    "without more than two uses"
                )
        if (
            sheet.policy.connectivity is ConnectivityPolicy.REQUIRE_CONNECTED
            and face_nodes
        ):
            pending = [min(face_nodes)]
            visited: set[int] = set()
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(sorted(adjacency[current] - visited, reverse=True))
            if visited != face_nodes:
                missing = ", ".join(str(value) for value in sorted(face_nodes - visited))
                errors.append(
                    f"sheet {sheet_id} is disconnected; unreachable face uses {missing}"
                )

    # Ordered member axes, subdivision ranges, and optional endpoint continuity.
    used_member_edge_uses: dict[int, int] = {}
    for member_id, raw_member in checked_members.items():
        member = raw_member
        assert isinstance(member, Member)
        previous_end = 0.0
        oriented_axis: list[tuple[int, int] | None] = []
        axis_edges: set[int] = set()
        for position, use_id in enumerate(member.edge_use_ids):
            previous_member = used_member_edge_uses.setdefault(use_id, member_id)
            if previous_member != member_id:
                errors.append(
                    f"member-edge use {use_id} occurs in members "
                    f"{previous_member} and {member_id}"
                )
            use = checked_member_uses.get(use_id)
            if use is None:
                errors.append(
                    f"member {member_id} references missing member-edge use {use_id}"
                )
                oriented_axis.append(None)
                continue
            assert isinstance(use, MemberEdgeUse)
            if use.member_id != member_id:
                errors.append(
                    f"member {member_id} lists member-edge use {use_id} owned by "
                    f"member {use.member_id}"
                )
            if abs(use.parent_range.start - previous_end) > tolerance:
                errors.append(
                    f"member {member_id} parent ranges are not contiguous at "
                    f"axis position {position}"
                )
            previous_end = use.parent_range.end
            if known_edges is not None and use.edge_id not in known_edges:
                errors.append(
                    f"member-edge use {use_id} references missing edge {use.edge_id}"
                )
            if use.edge_id in axis_edges:
                errors.append(
                    f"member {member_id} uses edge {use.edge_id} more than once"
                )
            axis_edges.add(use.edge_id)
            oriented_axis.append(
                None
                if edge_vertices is None
                else _oriented_vertices(use.edge_id, use.orientation, edge_vertices)
            )
        if abs(previous_end - 1.0) > tolerance:
            errors.append(f"member {member_id} parent ranges do not end at 1")
        if edge_vertices is not None:
            for position, (current, following) in enumerate(
                zip(oriented_axis, oriented_axis[1:])
            ):
                if current is not None and following is not None and current[1] != following[0]:
                    errors.append(
                        f"member {member_id} axis is not continuous after position {position}"
                    )
    for use_id, raw_use in checked_member_uses.items():
        use = raw_use
        assert isinstance(use, MemberEdgeUse)
        if use_id not in used_member_edge_uses:
            errors.append(f"member-edge use {use_id} is not used by a member axis")

    # Attachments and junction intent are checked independently of geometric
    # classification; strict audit later verifies their witnesses.
    for attachment_id, raw_attachment in checked_attachments.items():
        attachment = raw_attachment
        assert isinstance(attachment, Attachment)
        if attachment.source_kind == "member" and attachment.source_id not in checked_members:
            errors.append(
                f"attachment {attachment_id} references missing member "
                f"{attachment.source_id}"
            )
        elif attachment.source_kind == "vertex":
            if known_vertices is not None and attachment.source_id not in known_vertices:
                errors.append(
                    f"attachment {attachment_id} references missing source vertex "
                    f"{attachment.source_id}"
                )
        elif attachment.source_kind == "edge":
            if known_edges is not None and attachment.source_id not in known_edges:
                errors.append(
                    f"attachment {attachment_id} references missing source edge "
                    f"{attachment.source_id}"
                )
        elif attachment.source_kind == "face":
            if known_faces is not None and attachment.source_id not in known_faces:
                errors.append(
                    f"attachment {attachment_id} references missing source face "
                    f"{attachment.source_id}"
                )
        elif attachment.source_kind == "sheet":
            if attachment.source_id not in checked_sheets:
                errors.append(
                    f"attachment {attachment_id} references missing source sheet "
                    f"{attachment.source_id}"
                )
        if attachment.target_kind is AttachmentTargetKind.FACE:
            if known_faces is not None and attachment.target_id not in known_faces:
                errors.append(
                    f"attachment {attachment_id} references missing face "
                    f"{attachment.target_id}"
                )
        elif attachment.target_kind is AttachmentTargetKind.EDGE:
            if known_edges is not None and attachment.target_id not in known_edges:
                errors.append(
                    f"attachment {attachment_id} references missing edge "
                    f"{attachment.target_id}"
                )
        elif attachment.target_kind is AttachmentTargetKind.MEMBER:
            if attachment.target_id not in checked_members:
                errors.append(
                    f"attachment {attachment_id} references missing member "
                    f"{attachment.target_id}"
                )
        elif attachment.target_kind is AttachmentTargetKind.SHEET:
            if attachment.target_id not in checked_sheets:
                errors.append(
                    f"attachment {attachment_id} references missing sheet "
                    f"{attachment.target_id}"
                )
        elif known_vertices is not None and attachment.target_id not in known_vertices:
            errors.append(
                f"attachment {attachment_id} references missing vertex "
                f"{attachment.target_id}"
            )
        if attachment.part_id is not None:
            part = checked_parts.get(attachment.part_id)
            if part is None:
                errors.append(
                    f"attachment {attachment_id} references missing part "
                    f"{attachment.part_id}"
                )
            else:
                member = checked_members.get(attachment.member_id)
                if isinstance(member, Member) and member.part_id != attachment.part_id:
                    errors.append(
                        f"attachment {attachment_id} part context does not own member "
                        f"{attachment.member_id}"
                    )
        if attachment.sheet_id is not None and attachment.sheet_id not in checked_sheets:
            errors.append(
                f"attachment {attachment_id} references missing sheet context "
                f"{attachment.sheet_id}"
            )

    for junction_id, raw_junction in checked_junctions.items():
        junction = raw_junction
        assert isinstance(junction, Junction)
        member_ids = set(junction.member_ids)
        member_ranges = {
            use.member_id: use.member_range for use in junction.member_uses
        }
        for member_id in junction.member_ids:
            if member_id not in checked_members:
                errors.append(
                    f"junction {junction_id} references missing member {member_id}"
                )
        for sheet_id in junction.sheet_ids:
            if sheet_id not in checked_sheets:
                errors.append(
                    f"junction {junction_id} references missing sheet {sheet_id}"
                )
        target_sheets: set[int] = set()
        for attachment_id in junction.attachment_ids:
            attachment = checked_attachments.get(attachment_id)
            if attachment is None:
                errors.append(
                    f"junction {junction_id} references missing attachment "
                    f"{attachment_id}"
                )
                continue
            assert isinstance(attachment, Attachment)
            if attachment.member_id is None:
                errors.append(
                    f"junction {junction_id} attachment {attachment_id} has no member source"
                )
            elif attachment.member_id not in member_ids:
                errors.append(
                    f"junction {junction_id} attachment {attachment_id} belongs "
                    f"to non-participating member {attachment.member_id}"
                )
            else:
                junction_range = member_ranges[attachment.member_id]
                if not (
                    attachment.member_range.contains(
                        junction_range.start, tolerance=tolerance
                    )
                    and attachment.member_range.contains(
                        junction_range.end, tolerance=tolerance
                    )
                ):
                    errors.append(
                        f"junction {junction_id} attachment {attachment_id} "
                        "does not cover its participating member range"
                    )
            if attachment.target_kind is AttachmentTargetKind.FACE:
                sheet_id = face_to_sheet.get(attachment.target_id)
                if sheet_id is not None:
                    target_sheets.add(sheet_id)
            elif attachment.target_kind is AttachmentTargetKind.SHEET:
                target_sheets.add(attachment.target_id)
            elif attachment.target_kind is AttachmentTargetKind.EDGE:
                target_sheets.update(edge_to_sheets.get(attachment.target_id, ()))
        for sheet_id in junction.sheet_ids:
            if sheet_id not in target_sheets:
                errors.append(
                    f"junction {junction_id} has no attachment qualified on sheet "
                    f"{sheet_id}"
                )

    return tuple(sorted(set(errors)))


def raise_for_structural_topology(**kwargs: object) -> None:
    """Raise one topology exception when structural validation is not clean."""

    errors = validate_structural_topology(**kwargs)  # type: ignore[arg-type]
    if errors:
        raise GeometryTopologyError("invalid structural topology: " + "; ".join(errors))


def replace_member_edge_use(
    member: Member,
    original: MemberEdgeUse,
    replacements: Sequence[MemberEdgeUse],
    *,
    parameter_tolerance: float = 1.0e-12,
) -> Member:
    """Return ``member`` with one axis use replaced by a tiled child chain.

    The caller supplies children in member traversal order.  This makes the
    orientation behavior explicit: when a reversed geometry edge is split,
    the underlying child edges normally need to be supplied in reverse order.
    The persistent ``Member.id`` is retained.
    """

    if not isinstance(member, Member) or not isinstance(original, MemberEdgeUse):
        raise GeometryError("member and original use have invalid types")
    if original.member_id != member.id or original.id not in member.edge_use_ids:
        raise GeometryError("original member-edge use does not belong to member")
    if isinstance(replacements, (set, frozenset, Mapping)):
        raise GeometryError(
            "ordered member-edge-use replacements cannot be unordered"
        )
    made = tuple(replacements)
    if not made:
        raise GeometryError("member-edge-use replacement cannot be empty")
    if any(not isinstance(value, MemberEdgeUse) for value in made):
        raise GeometryError("replacement values must be MemberEdgeUse records")
    if any(value.member_id != member.id for value in made):
        raise GeometryError("replacement member-edge use belongs to another member")
    replacement_ids = tuple(value.id for value in made)
    if len(set(replacement_ids)) != len(replacement_ids):
        raise GeometryError("replacement member-edge-use IDs must be unique")
    if original.id in replacement_ids:
        raise GeometryError("a retired member-edge-use ID cannot be reused")
    tolerance = _finite_tolerance(parameter_tolerance)
    previous_end = original.parent_range.start
    for value in made:
        if abs(value.parent_range.start - previous_end) > tolerance:
            raise GeometryError("replacement parent ranges are not contiguous")
        previous_end = value.parent_range.end
    if abs(previous_end - original.parent_range.end) > tolerance:
        raise GeometryError("replacement parent ranges do not tile original range")
    position = member.edge_use_ids.index(original.id)
    edge_use_ids = (
        member.edge_use_ids[:position]
        + replacement_ids
        + member.edge_use_ids[position + 1 :]
    )
    return replace(member, edge_use_ids=edge_use_ids)
