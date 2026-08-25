"""Complete, deterministic qualification of a :class:`GeometryModel`.

The value types in :mod:`anygeometry.audit` deliberately do not know about a
geometry model.  This module is the corresponding model integration.  It
builds a fresh, side-effect-free broad phase, classifies every returned pair,
and reports unsupported narrow phases as blockers instead of treating them as
disjoint geometry.

The implementation intentionally starts with analytical straight-edge and
planar-face predicates.  Curved candidates that cannot be proved from shared
topology are reported as ``UNCLASSIFIED_CANDIDATE``.  That conservative rule
is important: adding a new curve predicate can turn a blocked report into a
certified one, but a missing predicate can never produce a false clean audit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

import numpy as np

from .audit import (
    AuditCode,
    AuditCollector,
    AuditContext,
    AuditEntity,
    AuditPolicy,
    AuditReport,
    AuditScope,
    AuditSeverity,
    AuditWitness,
    run_audit,
)
from .curves import Arc, Spline, Straight
from .errors import GeometryError
from .predicates import (
    DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
    IntersectionDimension,
    IntersectionKind,
    IntersectionQualificationPolicy,
    IntersectionResult,
    qualified_line_plane,
    qualified_plane_plane,
    qualified_segment_segment,
)
from .spatial import AABB, AABBTree, QueryDiagnostics, SpatialKey
from .structural import (
    AttachmentEvidence,
    AttachmentKind,
    AttachmentTargetKind,
    JunctionKind,
    NonManifoldPolicy,
    Orientation,
    validate_structural_topology,
)
from .surfaces import CoonsSurface, Cylinder, Plane, RuledSurface
from .tolerance import (
    DEFAULT_TOLERANCE_POLICY,
    TolerancePolicy,
    feature_extent,
)
from .transactions import ChangeSet

if TYPE_CHECKING:
    from .model import GeometryModel

__all__ = ["audit_changed_region", "strict_audit"]


_GEOMETRY_KINDS = frozenset(("vertex", "edge", "face"))
_PAIR_KINDS = frozenset(
    (
        frozenset(("vertex",)),
        frozenset(("vertex", "edge")),
        frozenset(("vertex", "face")),
        frozenset(("edge",)),
        frozenset(("edge", "face")),
        frozenset(("face",)),
    )
)


def _point(value: object) -> np.ndarray:
    made = np.asarray(value, dtype=float)
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        raise GeometryError("audit geometry contains a non-finite 3D point")
    return made


def _as_witness(label: str, point: Sequence[float]) -> AuditWitness:
    made = _point(point)
    return AuditWitness(label, tuple(float(item) for item in made))


def _component_witnesses(result: IntersectionResult) -> tuple[AuditWitness, ...]:
    return tuple(
        _as_witness(f"intersection/{index}", point)
        for index, point in enumerate(result.witnesses)
    )


def _surface_plane(face: object) -> Plane | None:
    surface = getattr(face, "surface", None)
    return surface if isinstance(surface, Plane) else None


def _structural_key_from_message(message: str) -> tuple[str, int] | None:
    match = re.search(
        r"\b(member-edge use|face use|attachment|junction|coedge|member|sheet|part) (\d+)\b",
        message,
    )
    if match is None:
        return None
    return match.group(1).replace("-", "_").replace(" ", "_"), int(match.group(2))


def _structural_code(message: str) -> AuditCode:
    lowered = message.lower()
    if "non-manifold" in lowered:
        return AuditCode.SHEET_NON_MANIFOLD
    if "orientation" in lowered:
        return AuditCode.SHEET_ORIENTATION
    if "junction" in lowered:
        return AuditCode.JUNCTION_INCONSISTENT
    if "attachment" in lowered:
        return AuditCode.ATTACHMENT_INCONSISTENT
    if "not used" in lowered or "missing" in lowered:
        return AuditCode.UNOWNED_STRUCTURAL_USE
    if "multiple structural uses" in lowered or "occurs in" in lowered:
        return AuditCode.MULTIPLY_OWNED_STRUCTURAL_USE
    return AuditCode.NONCONFORMAL_INTERFACE


def _topology_code(message: str) -> AuditCode:
    lowered = message.lower()
    if "replacement" in lowered or "lineage" in lowered:
        return AuditCode.UNRESOLVED_LINEAGE
    if "zero" in lowered or "negligible" in lowered or "sliver" in lowered:
        return AuditCode.SLIVER
    if "non-manifold" in lowered:
        return AuditCode.SHEET_NON_MANIFOLD
    return AuditCode.NONCONFORMAL_INTERFACE


@dataclass(slots=True)
class _PairClassification:
    classified: bool
    verified: bool = True


class _StrictAuditState:
    """Per-run caches and model-aware narrow-phase helpers."""

    def __init__(
        self,
        model: "GeometryModel",
        context: AuditContext,
        collector: AuditCollector,
        qualification: IntersectionQualificationPolicy | None = None,
    ) -> None:
        self.model = model
        self.context = context
        self.collector = collector
        candidate_policy = getattr(model, "tolerance", DEFAULT_TOLERANCE_POLICY)
        self.tolerance = (
            candidate_policy
            if isinstance(candidate_policy, TolerancePolicy)
            else DEFAULT_TOLERANCE_POLICY
        )
        self.qualification = (
            DEFAULT_INTERSECTION_QUALIFICATION_POLICY
            if qualification is None
            else qualification
        )
        if not isinstance(
            self.qualification, IntersectionQualificationPolicy
        ):
            raise TypeError(
                "qualification must be IntersectionQualificationPolicy"
            )
        self._edge_bounds: dict[int, AABB] = {}
        self._face_bounds: dict[int, AABB] = {}
        self._face_points: dict[int, np.ndarray] = {}
        self._face_support_trim_uv: dict[int, tuple[np.ndarray, ...]] = {}
        self._polygon_cache: dict[tuple[int, int], object] = {}

        # Ownership/incidence is resolved lazily from the model's maintained
        # reverse indices.  A changed-region audit therefore does not begin by
        # traversing every structural record in an otherwise unrelated model.
        self.edge_members: dict[int, set[int]] = {}
        self.vertex_members: dict[int, set[int]] = {}
        self.face_parts: dict[int, set[int]] = {}
        self.face_sheets: dict[int, set[int]] = {}
        self.edge_parts: dict[int, set[int]] = {}
        self.vertex_parts: dict[int, set[int]] = {}

    def members_for_edge(self, edge_id: int) -> set[int]:
        identifier = int(edge_id)
        if identifier not in self.edge_members:
            use_ids = getattr(self.model, "_edge_member_uses", {}).get(identifier, ())
            self.edge_members[identifier] = {
                self.model.member_edge_uses[use_id].member_id
                for use_id in use_ids
                if use_id in self.model.member_edge_uses
                and self.model.member_edge_uses[use_id].member_id in self.model.members
            }
        return self.edge_members[identifier]

    def members_for_vertex(self, vertex_id: int) -> set[int]:
        identifier = int(vertex_id)
        if identifier not in self.vertex_members:
            getter = getattr(self.model, "topological_edges_using_vertex", None)
            if callable(getter):
                edge_ids = getter(identifier)
            else:
                edge_ids = (
                    edge_id
                    for edge_id in self.model.edges_using_vertex(identifier)
                    if self.model.edges[edge_id].start == identifier
                    or self.model.edges[edge_id].end == identifier
                )
            self.vertex_members[identifier] = {
                member_id
                for edge_id in edge_ids
                for member_id in self.members_for_edge(edge_id)
            }
        return self.vertex_members[identifier]

    def sheets_for_face(self, face_id: int) -> set[int]:
        identifier = int(face_id)
        if identifier not in self.face_sheets:
            use_ids = getattr(self.model, "_face_structural_uses", {}).get(identifier, ())
            self.face_sheets[identifier] = {
                self.model.face_uses[use_id].sheet_id
                for use_id in use_ids
                if use_id in self.model.face_uses
                and self.model.face_uses[use_id].sheet_id in self.model.sheets
            }
        return self.face_sheets[identifier]

    def parts_for_face(self, face_id: int) -> set[int]:
        identifier = int(face_id)
        if identifier not in self.face_parts:
            self.face_parts[identifier] = {
                self.model.sheets[sheet_id].part_id
                for sheet_id in self.sheets_for_face(identifier)
            }
        return self.face_parts[identifier]

    def parts_for_edge(self, edge_id: int) -> set[int]:
        identifier = int(edge_id)
        if identifier not in self.edge_parts:
            parts = {
                self.model.members[member_id].part_id
                for member_id in self.members_for_edge(identifier)
            }
            parts.update(
                part_id
                for face_id in self.model.faces_using_edge(identifier)
                for part_id in self.parts_for_face(face_id)
            )
            self.edge_parts[identifier] = parts
        return self.edge_parts[identifier]

    def parts_for_vertex(self, vertex_id: int) -> set[int]:
        identifier = int(vertex_id)
        if identifier not in self.vertex_parts:
            self.vertex_parts[identifier] = {
                part_id
                for edge_id in self.model.edges_using_vertex(identifier)
                for part_id in self.parts_for_edge(edge_id)
            }
        return self.vertex_parts[identifier]

    def _bounds_length_tolerance(self, bounds: AABB) -> float:
        """Return tolerance scaled only by the bounded participating entity."""

        return self.tolerance.effective_length(float(np.linalg.norm(bounds.extents)))

    def _edge_pair_length_tolerance(self, first_id: int, second_id: int) -> float:
        return self.tolerance.effective_length(
            max(self.model.edge_length(first_id), self.model.edge_length(second_id))
        )

    def _face_pair_extent(self, first_id: int, second_id: int) -> float:
        return max(
            feature_extent(self.face_points(first_id)),
            feature_extent(self.face_points(second_id)),
        )

    def entity(self, key: SpatialKey) -> AuditEntity:
        return AuditEntity.from_key(self.context.model_id, key)

    def entities(self, *keys: SpatialKey) -> tuple[AuditEntity, ...]:
        return tuple(self.entity(key) for key in keys)

    def issue(
        self,
        code: AuditCode,
        severity: AuditSeverity,
        message: str,
        *,
        keys: Iterable[SpatialKey] = (),
        witnesses: Iterable[AuditWitness] = (),
        classification: object | None = None,
        details: Mapping[str, object] | None = None,
        **evidence: object,
    ) -> None:
        made_keys = tuple(keys)
        structural_context: set[SpatialKey] = set()
        for key in made_keys:
            structural_context.update(("part", item) for item in self.parts_for(key))
            if key[0] == "face":
                structural_context.update(
                    ("sheet", item) for item in self.sheets_for_face(key[1])
                )
            if key[0] == "edge":
                structural_context.update(
                    ("member", item) for item in self.members_for_edge(key[1])
                )
            if key[0] == "vertex":
                structural_context.update(
                    ("member", item) for item in self.members_for_vertex(key[1])
                )
            if key[0] == "sheet":
                sheet = self.model.sheets.get(key[1])
                if sheet is not None:
                    structural_context.add(("part", sheet.part_id))
            if key[0] == "attachment":
                attachment = self.model.attachments.get(key[1])
                if attachment is not None:
                    if attachment.member_id is not None:
                        structural_context.add(("member", attachment.member_id))
                        member = self.model.members.get(attachment.member_id)
                        if member is not None:
                            structural_context.add(("part", member.part_id))
                    if attachment.part_id is not None:
                        structural_context.add(("part", attachment.part_id))
                    if attachment.sheet_id is not None:
                        structural_context.add(("sheet", attachment.sheet_id))
            if key[0] == "junction":
                junction = self.model.junctions.get(key[1])
                if junction is not None:
                    structural_context.update(
                        ("member", member_id) for member_id in junction.member_ids
                    )
                    structural_context.update(
                        ("sheet", sheet_id) for sheet_id in junction.sheet_ids
                    )
        self.collector.issue(
            code,
            severity,
            message,
            entities=tuple(self.entity(key) for key in made_keys),
            context=tuple(self.entity(key) for key in sorted(structural_context)),
            witnesses=tuple(witnesses),
            classification=classification,
            details=details or {},
            **evidence,
        )
    def unclassified(
        self,
        first: SpatialKey,
        second: SpatialKey,
        reason: str,
        kind: IntersectionKind = IntersectionKind.UNCLASSIFIED,
    ) -> _PairClassification:
        code = {
            IntersectionKind.UNSUPPORTED: AuditCode.UNSUPPORTED_CANDIDATE,
            IntersectionKind.CAPABILITY_MISSING: AuditCode.CAPABILITY_MISSING,
        }.get(kind, AuditCode.UNCLASSIFIED_CANDIDATE)
        self.issue(
            code,
            AuditSeverity.BLOCKER,
            "a broad-phase candidate has no verified narrow-phase classification",
            keys=(first, second),
            classification=kind,
            classification_confidence=0.0,
            evidence_quality="unverified",
            recommended_action="add a verified qualifier or remove the unsupported relation",
            blocks_strict_handoff=True,
            details={"reason": reason},
        )
        return _PairClassification(False)

    def record_intersection_certificate(self, result) -> None:
        """Expose public narrow-phase work without duplicating its predicates."""

        certificate = getattr(result, "certificate", None)
        if certificate is None:
            return
        self.collector.record_intersection_work(
            boxes_examined=certificate.boxes_examined,
            subdivisions=certificate.subdivisions,
            trace_segments=certificate.trace_segments,
        )

    def parts_for(self, key: SpatialKey) -> frozenset[int]:
        kind, identifier = key
        if kind == "vertex":
            return frozenset(self.parts_for_vertex(identifier))
        if kind == "edge":
            return frozenset(self.parts_for_edge(identifier))
        if kind == "face":
            return frozenset(self.parts_for_face(identifier))
        if kind == "member" and identifier in self.model.members:
            return frozenset((self.model.members[identifier].part_id,))
        return frozenset()

    def separate_part_intent(self, first: SpatialKey, second: SpatialKey) -> bool:
        first_parts, second_parts = self.parts_for(first), self.parts_for(second)
        return bool(first_parts and second_parts and first_parts.isdisjoint(second_parts))

    def _member_parameter_tolerance(self, member_id: int) -> float:
        member = self.model.members[member_id]
        length = sum(
            self.model.edge_length(self.model.member_edge_uses[use_id].edge_id)
            for use_id in member.edge_use_ids
        )
        return self.tolerance.effective_parameter(length, length)

    def _member_ranges_for_edge_candidate(
        self,
        member_id: int,
        edge_id: int,
        kind: IntersectionKind,
        witnesses: Sequence[np.ndarray],
    ) -> tuple[tuple[float, float], ...]:
        """Map one edge-local relation to each matching member parent range."""

        if kind is IntersectionKind.COINCIDENT:
            local_ranges = ((0.0, 1.0),)
        else:
            made: list[float] = []
            for witness in witnesses:
                _closest, parameter, distance = self.model.closest_edge_point(
                    edge_id, witness
                )
                tolerance = self.tolerance.effective_surface_residual(
                    self.model.edge_length(edge_id)
                )
                if distance > tolerance:
                    return ()
                made.append(float(parameter))
            if not made:
                return ()
            if kind in (
                IntersectionKind.OVERLAP_CURVE,
                IntersectionKind.CONTAINED,
            ):
                local_ranges = ((min(made), max(made)),)
            else:
                # Point components remain distinct.  Collapsing two circle
                # intersections into one min/max interval makes two valid
                # point junctions look like one undeclared interval.
                local_ranges = tuple((value, value) for value in made)

        ranges: list[tuple[float, float]] = []
        member = self.model.members[member_id]
        for use_id in member.edge_use_ids:
            use = self.model.member_edge_uses[use_id]
            if use.edge_id != edge_id:
                continue
            for start, end in local_ranges:
                if use.orientation is Orientation.REVERSED:
                    start, end = 1.0 - start, 1.0 - end
                span = use.parent_range.end - use.parent_range.start
                first = use.parent_range.start + start * span
                second = use.parent_range.start + end * span
                ranges.append((min(first, second), max(first, second)))
        return tuple(ranges)

    def _member_ranges_at_vertex(
        self, member_id: int, vertex_id: int
    ) -> tuple[tuple[float, float], ...]:
        parameters: set[float] = set()
        member = self.model.members[member_id]
        for use_id in member.edge_use_ids:
            use = self.model.member_edge_uses[use_id]
            edge = self.model.edges[use.edge_id]
            local_values: list[float] = []
            if edge.start == vertex_id:
                local_values.append(0.0)
            if edge.end == vertex_id:
                local_values.append(1.0)
            for local in local_values:
                if use.orientation is Orientation.REVERSED:
                    local = 1.0 - local
                span = use.parent_range.end - use.parent_range.start
                parameters.add(use.parent_range.start + local * span)
        return tuple((value, value) for value in sorted(parameters))

    def declared_junction(
        self,
        first_member: int,
        second_member: int,
        kind: IntersectionKind,
        *,
        first_ranges: Sequence[tuple[float, float]] = (),
        second_ranges: Sequence[tuple[float, float]] = (),
    ) -> int | None:
        acceptable = {
            IntersectionKind.COINCIDENT: {JunctionKind.OVERLAP, JunctionKind.MULTI_WAY},
            IntersectionKind.OVERLAP_CURVE: {JunctionKind.OVERLAP, JunctionKind.MULTI_WAY},
            IntersectionKind.CONTAINED: {JunctionKind.OVERLAP, JunctionKind.MULTI_WAY},
            IntersectionKind.CROSS: {JunctionKind.CROSSING, JunctionKind.MULTI_WAY},
            IntersectionKind.TOUCH_POINT: {
                JunctionKind.ENDPOINT,
                JunctionKind.CROSSING,
                JunctionKind.OVERLAP,
                JunctionKind.MULTI_WAY,
            },
        }.get(kind, set())
        if not first_ranges or not second_ranges:
            return None
        first_tolerance = self._member_parameter_tolerance(first_member)
        second_tolerance = self._member_parameter_tolerance(second_member)

        def covers(declared: object, candidate: tuple[float, float], tolerance: float) -> bool:
            return (
                declared.start - tolerance <= candidate[0]
                and candidate[1] <= declared.end + tolerance
            )

        for identifier, junction in sorted(self.model.junctions.items()):
            if junction.kind not in acceptable:
                continue
            uses = {use.member_id: use for use in junction.member_uses}
            first_use = uses.get(first_member)
            second_use = uses.get(second_member)
            if first_use is None or second_use is None:
                continue
            if any(
                covers(first_use.member_range, candidate, first_tolerance)
                for candidate in first_ranges
            ) and any(
                covers(second_use.member_range, candidate, second_tolerance)
                for candidate in second_ranges
            ):
                return identifier
        return None

    def ownership_intent(
        self,
        first: SpatialKey,
        second: SpatialKey,
        kind: IntersectionKind,
        *,
        witnesses: Sequence[Sequence[float]] = (),
    ) -> tuple[str, int | None] | None:
        if self.separate_part_intent(first, second):
            return "separate_parts", None
        if {first[0], second[0]} in ({"vertex", "edge"}, {"vertex", "face"}):
            vertex_key = first if first[0] == "vertex" else second
            target_key = second if first[0] == "vertex" else first
            getter_name = (
                "attachments_for_face"
                if target_key[0] == "face"
                else "attachments_for_edge"
            )
            getter = getattr(self.model, getter_name, None)
            attachment_ids = (
                getter(target_key[1])
                if callable(getter)
                else getattr(self.model, "_target_attachments", {}).get(
                    target_key, ()
                )
            )
            for attachment_id in attachment_ids:
                attachment = self.model.attachments.get(attachment_id)
                if (
                    attachment is not None
                    and attachment.source_key == vertex_key
                    and attachment.target_key == target_key
                ):
                    return "declared_attachment", attachment_id
        points = tuple(_point(witness) for witness in witnesses)
        if first[0] == second[0] == "edge":
            for first_member in sorted(self.members_for_edge(first[1])):
                for second_member in sorted(self.members_for_edge(second[1])):
                    if first_member == second_member:
                        continue
                    junction = self.declared_junction(
                        first_member,
                        second_member,
                        kind,
                        first_ranges=self._member_ranges_for_edge_candidate(
                            first_member, first[1], kind, points
                        ),
                        second_ranges=self._member_ranges_for_edge_candidate(
                            second_member, second[1], kind, points
                        ),
                    )
                    if junction is not None:
                        return "declared_junction", junction
        if first[0] == second[0] == "vertex":
            for first_member in sorted(self.members_for_vertex(first[1])):
                for second_member in sorted(self.members_for_vertex(second[1])):
                    if first_member == second_member:
                        continue
                    junction = self.declared_junction(
                        first_member,
                        second_member,
                        IntersectionKind.TOUCH_POINT,
                        first_ranges=self._member_ranges_at_vertex(
                            first_member, first[1]
                        ),
                        second_ranges=self._member_ranges_at_vertex(
                            second_member, second[1]
                        ),
                    )
                    if junction is not None:
                        return "declared_junction", junction
        if {first[0], second[0]} == {"vertex", "edge"}:
            vertex_key = first if first[0] == "vertex" else second
            edge_key = second if first[0] == "vertex" else first
            for first_member in sorted(self.members_for_vertex(vertex_key[1])):
                for second_member in sorted(self.members_for_edge(edge_key[1])):
                    if first_member == second_member:
                        continue
                    junction = self.declared_junction(
                        first_member,
                        second_member,
                        IntersectionKind.TOUCH_POINT,
                        first_ranges=self._member_ranges_at_vertex(
                            first_member, vertex_key[1]
                        ),
                        second_ranges=self._member_ranges_for_edge_candidate(
                            second_member,
                            edge_key[1],
                            IntersectionKind.TOUCH_POINT,
                            (self.model.vertex_position(vertex_key[1]),),
                        ),
                    )
                    if junction is not None:
                        return "declared_junction", junction
        return None

    # --------------------------------------------------------------- bounds
    def edge_bounds(self, edge_id: int) -> AABB:
        cached = self._edge_bounds.get(edge_id)
        if cached is not None:
            return cached
        edge = self.model.edges[edge_id]
        if isinstance(edge.curve, Straight):
            bounds = AABB.from_points(
                (
                    self.model.vertex_position(edge.start),
                    self.model.vertex_position(edge.end),
                )
            )
        elif isinstance(edge.curve, Arc):
            frame = self.model.arc_frame(edge_id)
            # Coordinate extrema of a circular arc occur at an end point or
            # where one coordinate's sinusoid has zero derivative.  Using the
            # complete supporting circle here would be conservative but turns
            # a segmented ring into a quadratic all-pairs candidate set.
            angles = [0.0, float(frame.sweep)]

            def on_sweep(angle: float) -> float | None:
                two_pi = 2.0 * float(np.pi)
                if frame.sweep > 0.0:
                    made = angle % two_pi
                    return (
                        made
                        if made <= frame.sweep + self.tolerance.angular
                        else None
                    )
                made = -((-angle) % two_pi)
                return (
                    made
                    if made >= frame.sweep - self.tolerance.angular
                    else None
                )

            for axis in range(3):
                base = float(np.arctan2(frame.e2[axis], frame.e1[axis]))
                for candidate in (base, base + float(np.pi)):
                    included = on_sweep(candidate)
                    if included is not None:
                        angles.append(included)
            parameters = np.asarray(angles, dtype=float) / float(frame.sweep)
            bounds = AABB.from_points(
                (
                    self.model.vertex_position(edge.start),
                    self.model.vertex_position(edge.end),
                    *self.model.sample_edge(edge_id, parameters),
                )
            )
        elif isinstance(edge.curve, Spline):
            vertex_ids = (
                edge.start,
                *edge.curve.control_vertices,
                edge.end,
            )
            # A Bezier curve lies in the convex hull of its control polygon.
            bounds = AABB.from_points(
                self.model.vertex_position(identifier) for identifier in vertex_ids
            )
        else:  # pragma: no cover - future curve family, deliberately fail closed
            raise GeometryError(f"unsupported curve type {type(edge.curve).__qualname__}")
        self._edge_bounds[edge_id] = bounds
        return bounds

    def face_points(self, face_id: int) -> np.ndarray:
        cached = self._face_points.get(face_id)
        if cached is not None:
            return cached
        face = self.model.faces[face_id]
        made: list[np.ndarray] = []
        for loop in (face.loop,) + tuple(face.holes):
            for oriented in loop:
                edge = self.model.edges[oriented.edge]
                count = 2 if isinstance(edge.curve, Straight) else 33
                samples = self.model.sample_edge(
                    oriented.edge, np.linspace(0.0, 1.0, count)
                )
                if not oriented.forward:
                    samples = samples[::-1]
                made.extend(samples[:-1])
        if not made:
            raise GeometryError(f"face {face_id} has no boundary points")
        result = np.asarray(made, dtype=float)
        if result.ndim != 2 or result.shape[1:] != (3,) or not np.all(np.isfinite(result)):
            raise GeometryError(f"face {face_id} has invalid boundary samples")
        self._face_points[face_id] = result
        return result

    def face_support_trim_loops_uv(self, face_id: int) -> tuple[np.ndarray, ...]:
        """Return trim loops in the authoritative support's coordinates.

        ``GeometryModel.face_trim_loops_uv`` intentionally follows an optional
        display/meshing parameterization when one is present.  Qualification
        must instead invert the persistent support surface directly, so an
        unrelated parameterization can never move a physical relationship.
        """

        cached = self._face_support_trim_uv.get(face_id)
        if cached is not None:
            return cached
        face = self.model.faces[face_id]
        if face.support_surface is None:
            raise GeometryError(
                f"face {face_id} has no authoritative support surface"
            )
        loops: list[np.ndarray] = []
        for loop in (face.loop,) + tuple(face.holes):
            points: list[tuple[float, float]] = []
            for oriented in loop:
                edge = self.model.edges[oriented.edge]
                count = 2 if isinstance(edge.curve, Straight) else 33
                samples = self.model.sample_edge(
                    oriented.edge, np.linspace(0.0, 1.0, count)
                )
                if not oriented.forward:
                    samples = samples[::-1]
                points.extend(
                    self.model.face_support_local_uv(face_id, point)
                    for point in samples[:-1]
                )
            polygon = np.asarray(points, dtype=float)
            if (
                polygon.ndim != 2
                or polygon.shape[1:] != (2,)
                or not np.all(np.isfinite(polygon))
            ):
                raise GeometryError(
                    f"face {face_id} has an invalid support-space trim"
                )
            loops.append(polygon)
        result = tuple(loops)
        self._face_support_trim_uv[face_id] = result
        return result

    def face_support_contains_uv(
        self, face_id: int, uv: Sequence[float]
    ) -> bool:
        """Whether a support-space coordinate lies in the physical trim."""

        candidate = np.asarray(uv, dtype=float)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            raise GeometryError("support coordinates must contain finite u and v")
        polygons = self.face_support_trim_loops_uv(face_id)
        if not polygons or not self.model._point_in_polygon(  # noqa: SLF001
            candidate, polygons[0]
        ):
            return False
        return not any(
            self.model._point_in_polygon(  # noqa: SLF001
                candidate, hole, include_boundary=False
            )
            for hole in polygons[1:]
        )

    def project_to_face_support(
        self, face_id: int, point: Sequence[float]
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        """Project onto a face's authoritative support and physical trim."""

        target = _point(point)
        uv = self.model.face_support_local_uv(face_id, target)
        projected = self.model.face_support_point(face_id, *uv)
        if not self.face_support_contains_uv(face_id, uv):
            face = self.model.faces[face_id]
            candidates = tuple(
                self.model.closest_edge_point(oriented.edge, target)
                for loop in (face.loop,) + tuple(face.holes)
                for oriented in loop
            )
            if not candidates:
                raise GeometryError(f"face {face_id} has no trim boundary")
            boundary, _parameter, _distance = min(
                candidates, key=lambda item: item[2]
            )
            uv = self.model.face_support_local_uv(face_id, boundary)
            projected = self.model.face_support_point(face_id, *uv)
        distance = float(np.linalg.norm(projected - target))
        return projected, (float(uv[0]), float(uv[1])), distance

    def face_bounds(self, face_id: int) -> AABB:
        cached = self._face_bounds.get(face_id)
        if cached is not None:
            return cached
        face = self.model.faces[face_id]
        boxes = [
            self.edge_bounds(oriented.edge)
            for loop in (face.loop,) + tuple(face.holes)
            for oriented in loop
        ]
        if not boxes:
            raise GeometryError(f"face {face_id} has no bounded trim")
        bounds = AABB.union_all(boxes)
        surface = face.surface
        if isinstance(surface, RuledSurface):
            # A ruled patch is a convex blend of its two piecewise-linear
            # boundary curves, so their joint AABB contains every interior
            # point.  Include topology bounds as well so an inconsistent
            # document is still conservatively broad-phased before topology
            # diagnostics block certification.
            bounds = bounds.union(
                AABB.from_points(
                    np.vstack((surface.first_boundary, surface.second_boundary))
                )
            )
        elif isinstance(surface, CoonsSurface):
            # Coons interiors need not lie in the AABB of their boundary: the
            # two blended boundary terms can add before the bilinear corner
            # term is subtracted.  Bound A + B - C coordinate-wise, where A
            # blends bottom/top, B blends left/right, and C is the bilinear
            # corner patch.  This is deliberately loose but analytical and
            # conservative, preventing an interior contact from being missed.
            if surface.has_boundaries:
                assert (
                    surface.bottom is not None
                    and surface.right is not None
                    and surface.top is not None
                    and surface.left is not None
                )
                bottom_top = AABB.from_points(
                    np.vstack((surface.bottom, surface.top))
                )
                left_right = AABB.from_points(
                    np.vstack((surface.left, surface.right))
                )
                corners = AABB.from_points(
                    (
                        surface.bottom[0],
                        surface.bottom[-1],
                        surface.top[0],
                        surface.top[-1],
                    )
                )
            elif len(face.corners) == 4:
                sides = face.sides()

                def side_bounds(indices: Sequence[int]) -> AABB:
                    return AABB.union_all(
                        self.edge_bounds(oriented.edge)
                        for index in indices
                        for oriented in sides[index]
                    )

                bottom_top = side_bounds((0, 2))
                left_right = side_bounds((1, 3))
                corners = AABB.from_points(
                    self.model.vertex_position(identifier)
                    for identifier in self.model.face_corner_vertices(face_id)
                )
            else:
                raise GeometryError(
                    f"face {face_id} has an unbounded topology-backed Coons surface"
                )
            coons_bounds = AABB(
                tuple(
                    bottom_top.minimum[axis]
                    + left_right.minimum[axis]
                    - corners.maximum[axis]
                    for axis in range(3)
                ),
                tuple(
                    bottom_top.maximum[axis]
                    + left_right.maximum[axis]
                    - corners.minimum[axis]
                    for axis in range(3)
                ),
            )
            bounds = bounds.union(coons_bounds)
        self._face_bounds[face_id] = bounds
        return bounds

    def bounds(self, key: SpatialKey) -> AABB:
        kind, identifier = key
        if kind == "vertex":
            return AABB.around_point(self.model.vertex_position(identifier))
        if kind == "edge":
            return self.edge_bounds(identifier)
        if kind == "face":
            return self.face_bounds(identifier)
        raise GeometryError(f"unsupported spatial kind {kind!r}")

    # ---------------------------------------------------------- topology pass
    def audit_topology(self) -> None:
        for message in self.model.validate_topology():
            match = re.search(r"\b(vertex|edge|face) (\d+)\b", message)
            keys: tuple[SpatialKey, ...] = ()
            if match is not None:
                keys = ((match.group(1), int(match.group(2))),)
            self.issue(
                _topology_code(message),
                AuditSeverity.ERROR,
                message,
                keys=keys,
                classification="invalid_topology",
            )

        structural_errors = validate_structural_topology(
            parts=self.model.parts,
            sheets=self.model.sheets,
            face_uses=self.model.face_uses,
            coedges=self.model.coedges,
            members=self.model.members,
            member_edge_uses=self.model.member_edge_uses,
            attachments=self.model.attachments,
            junctions=self.model.junctions,
            edge_ids=tuple(self.model.edges),
            face_ids=tuple(self.model.faces),
            vertex_ids=tuple(self.model.vertices),
            edge_vertices={
                edge.id: (edge.start, edge.end)
                for edge in self.model.edges.values()
            },
        )
        for message in structural_errors:
            key = _structural_key_from_message(message)
            self.issue(
                _structural_code(message),
                AuditSeverity.ERROR,
                message,
                keys=() if key is None else (key,),
                classification="invalid_structural_topology",
            )

        self.audit_non_manifold_edges()
        self.audit_orphan_control_geometry()
        self.audit_existing_spatial_index()

    def audit_orphan_control_geometry(
        self, vertex_ids: Iterable[int] | None = None
    ) -> None:
        """Report explicitly designated control points without an owner.

        Control-geometry storage is model-owned and intentionally kept out of
        this audit module.  The small callback-style public protocol avoids an
        import cycle while allowing older schema models (which expose no such
        designation) to remain compatible.
        """

        getter = getattr(self.model, "orphan_control_vertices", None)
        if not callable(getter):
            return
        selected = None if vertex_ids is None else frozenset(int(item) for item in vertex_ids)
        construction = getattr(self.model, "construction_vertices", {})
        for vertex_id in sorted(int(item) for item in getter()):
            if selected is not None and vertex_id not in selected:
                continue
            # Part-owned construction/reference points are legitimate
            # structural geometry even when no curve consumes them.  Only an
            # unowned designation is an orphan curve-control finding.
            if construction.get(vertex_id) is not None:
                continue
            self.issue(
                AuditCode.ORPHAN_CONTROL_GEOMETRY,
                AuditSeverity.ERROR,
                "a curve-control vertex has no owning curve dependency",
                keys=(("vertex", vertex_id),),
                classification="orphan_control_vertex",
                recommended_action="remove the orphan control point or assign it to a curve",
                blocks_strict_handoff=True,
            )

    def audit_existing_spatial_index(self) -> None:
        index = getattr(self.model, "_spatial_index", None)
        if index is None:
            return
        try:
            index.validate()
            expected = tuple(
                sorted(
                    key
                    for key in self.model.entity_keys()
                    if key[0] in _GEOMETRY_KINDS
                )
            )
            if index.keys != expected:
                raise AssertionError("index keys do not match active geometry")
            for key in expected:
                indexed = index.bounds(key)
                required = self.bounds(key)
                if not indexed.expanded(
                    self._bounds_length_tolerance(required)
                ).contains(required):
                    raise AssertionError(
                        f"index bounds are not conservative for {key!r}"
                    )
        except Exception as error:
            self.issue(
                AuditCode.SPATIAL_INDEX_INCONSISTENT,
                AuditSeverity.BLOCKER,
                "the maintained spatial index is inconsistent with active geometry",
                classification="index_inconsistent",
                details={"exception_type": type(error).__qualname__},
            )

    def audit_non_manifold_edges(
        self, edge_ids: Iterable[int] | None = None
    ) -> None:
        uses: dict[int, set[int]] = defaultdict(set)
        if edge_ids is None:
            for face_id, face in self.model.faces.items():
                for loop in (face.loop,) + tuple(face.holes):
                    for oriented in loop:
                        uses[oriented.edge].add(face_id)
        else:
            for edge_id in sorted(set(int(item) for item in edge_ids)):
                if edge_id in self.model.edges:
                    uses[edge_id].update(self.model.faces_using_edge(edge_id))
        for edge_id, face_ids in sorted(uses.items()):
            if len(face_ids) <= 2:
                continue
            owning_sheets = {
                sheet_id
                for face_id in face_ids
                for sheet_id in self.sheets_for_face(face_id)
            }
            declared = bool(owning_sheets) and all(
                edge_id in self.model.sheets[sheet_id].declared_non_manifold_edges
                and self.model.sheets[sheet_id].policy.non_manifold
                is NonManifoldPolicy.ALLOW_DECLARED
                for sheet_id in owning_sheets
            )
            keys: tuple[SpatialKey, ...] = (
                ("edge", edge_id),
                *(("face", face_id) for face_id in sorted(face_ids)),
            )
            if declared:
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "a radial edge use greater than two is explicitly declared",
                    keys=keys,
                    classification="declared_non_manifold",
                    details={"radial_use_count": len(face_ids)},
                )
            else:
                self.issue(
                    AuditCode.SHEET_NON_MANIFOLD,
                    AuditSeverity.ERROR,
                    "an edge has more than two undeclared face uses",
                    keys=keys,
                    classification="non_manifold",
                    details={"radial_use_count": len(face_ids)},
                )

    # ------------------------------------------------------------ broad phase
    def pair_filter(self, first: SpatialKey, second: SpatialKey) -> bool:
        kinds = frozenset((first[0], second[0]))
        if kinds not in _PAIR_KINDS:
            return False
        return True

    def audit_spatial_pairs(self) -> None:
        items: list[tuple[SpatialKey, AABB]] = []
        failed: list[tuple[SpatialKey, str]] = []
        for key in sorted(
            key for key in self.model.entity_keys() if key[0] in _GEOMETRY_KINDS
        ):
            try:
                bounds = self.bounds(key)
                items.append(
                    (key, bounds.expanded(self._bounds_length_tolerance(bounds)))
                )
            except Exception as error:
                failed.append((key, type(error).__qualname__))
        for key, error_type in failed:
            self.issue(
                AuditCode.SPATIAL_INDEX_INCONSISTENT,
                AuditSeverity.BLOCKER,
                "an active entity could not be conservatively bounded",
                keys=(key,),
                classification="unbounded_entity",
                details={"exception_type": error_type},
            )
        tree = AABBTree(items)
        tree.validate()
        result = tree.overlap_pairs(pair_filter=self.pair_filter)
        self.collector.record_broad_phase(result.diagnostics)
        for first, second in result.pairs:
            self.collector.record_narrow_phase()
            try:
                classification = self.classify_pair(first, second)
            except Exception as error:
                classification = self.unclassified(
                    first, second, f"{type(error).__qualname__}:narrow_phase_failed"
                )
            self.collector.record_classification(
                classified=classification.classified,
                verified=classification.verified,
            )

    def changed_geometry_closure(self, change_set: ChangeSet) -> set[SpatialKey]:
        """Resolve a committed ChangeSet to active geometry without global scans."""

        result: set[SpatialKey] = set()

        def add_vertex(vertex_id: int, *, dependants: bool = True) -> None:
            if vertex_id not in self.model.vertices:
                return
            result.add(("vertex", vertex_id))
            if dependants:
                for edge_id in self.model.edges_using_vertex(vertex_id):
                    add_edge(edge_id)

        def add_edge(edge_id: int, *, neighbouring_faces: bool = True) -> None:
            edge = self.model.edges.get(edge_id)
            if edge is None:
                return
            result.add(("edge", edge_id))
            add_vertex(edge.start, dependants=False)
            add_vertex(edge.end, dependants=False)
            if isinstance(edge.curve, Arc):
                add_vertex(edge.curve.via_vertex, dependants=False)
            elif isinstance(edge.curve, Spline):
                for vertex_id in edge.curve.control_vertices:
                    add_vertex(vertex_id, dependants=False)
            if neighbouring_faces:
                for face_id in self.model.faces_using_edge(edge_id):
                    result.add(("face", face_id))

        def add_face(face_id: int) -> None:
            face = self.model.faces.get(face_id)
            if face is None:
                return
            result.add(("face", face_id))
            for loop in (face.loop,) + tuple(face.holes):
                for oriented in loop:
                    add_edge(oriented.edge, neighbouring_faces=False)

        def add_member(member_id: int) -> None:
            member = self.model.members.get(member_id)
            if member is None:
                return
            for use_id in member.edge_use_ids:
                use = self.model.member_edge_uses.get(use_id)
                if use is not None:
                    add_edge(use.edge_id)

        def add_sheet(sheet_id: int) -> None:
            sheet = self.model.sheets.get(sheet_id)
            if sheet is None:
                return
            for use_id in sheet.face_use_ids:
                use = self.model.face_uses.get(use_id)
                if use is not None:
                    add_face(use.face_id)

        active_geometry = {
            *change_set.changed,
            *change_set.invalidated_caches,
            *change_set.spatial_updates,
        }
        for kind, identifier in sorted(active_geometry):
            if kind == "vertex":
                add_vertex(identifier)
            elif kind == "edge":
                add_edge(identifier)
            elif kind == "face":
                add_face(identifier)

        for kind, identifier in sorted(change_set.member_changes):
            if kind == "member":
                add_member(identifier)
            elif kind == "member_edge_use":
                use = self.model.member_edge_uses.get(identifier)
                if use is not None:
                    add_edge(use.edge_id)
        for _kind, identifier in sorted(change_set.attachment_changes):
            attachment = self.model.attachments.get(identifier)
            if attachment is None:
                continue
            if attachment.member_id is not None:
                add_member(attachment.member_id)
            if attachment.target_kind is AttachmentTargetKind.FACE:
                add_face(attachment.target_id)
            else:
                add_edge(attachment.target_id)
        for kind, identifier in sorted(change_set.ownership_changes):
            if kind == "part":
                part = self.model.parts.get(identifier)
                if part is not None:
                    for sheet_id in part.sheet_ids:
                        add_sheet(sheet_id)
                    for member_id in part.member_ids:
                        add_member(member_id)
            elif kind == "sheet":
                add_sheet(identifier)
            elif kind == "face_use":
                use = self.model.face_uses.get(identifier)
                if use is not None:
                    add_face(use.face_id)
            elif kind == "coedge":
                coedge = self.model.coedges.get(identifier)
                if coedge is not None:
                    add_edge(coedge.edge_id)
            elif kind == "junction":
                junction = self.model.junctions.get(identifier)
                if junction is not None:
                    for member_id in junction.member_ids:
                        add_member(member_id)
                    for sheet_id in junction.sheet_ids:
                        add_sheet(sheet_id)
        return result

    def audit_changed_spatial_pairs(self, change_set: ChangeSet) -> set[SpatialKey]:
        """Qualify current pairs incident to changed bounds through the index."""

        tree = self.model._spatial()  # noqa: SLF001 - maintained kernel index
        seeds = self.changed_geometry_closure(change_set)
        regions: list[AABB] = []
        for change in change_set.affected_aabbs:
            for raw in (change.before, change.after):
                if raw is None:
                    continue
                bounds = AABB(tuple(raw[:3]), tuple(raw[3:]))
                regions.append(bounds.expanded(self._bounds_length_tolerance(bounds)))
        totals = QueryDiagnostics()
        if regions:
            nearby = tree.query_regions(regions, kinds=_GEOMETRY_KINDS)
            seeds.update(nearby.keys)
            totals = totals.combined(nearby.diagnostics)

        active = tuple(sorted(key for key in seeds if key in tree))
        pairs: set[tuple[SpatialKey, SpatialKey]] = set()
        for seed in active:
            bounds = self.bounds(seed)
            nearby = tree.query(
                bounds.expanded(self._bounds_length_tolerance(bounds)),
                kinds=_GEOMETRY_KINDS,
            )
            totals = totals.combined(nearby.diagnostics)
            for other in nearby.keys:
                if other == seed:
                    continue
                pair = (seed, other) if seed < other else (other, seed)
                if self.pair_filter(*pair):
                    pairs.add(pair)

        ordered = tuple(sorted(pairs))
        self.collector.record_broad_phase(
            QueryDiagnostics(
                region_count=totals.region_count,
                node_visits=totals.node_visits,
                branch_visits=totals.branch_visits,
                leaf_tests=totals.leaf_tests,
                raw_candidate_hits=totals.raw_candidate_hits,
                candidate_count=len(ordered),
            )
        )
        for first, second in ordered:
            self.collector.record_narrow_phase()
            try:
                classification = self.classify_pair(first, second)
            except Exception as error:
                classification = self.unclassified(
                    first,
                    second,
                    f"{type(error).__qualname__}:narrow_phase_failed",
                )
            self.collector.record_classification(
                classified=classification.classified,
                verified=classification.verified,
            )
        return set(active)

    def classify_pair(
        self, first: SpatialKey, second: SpatialKey
    ) -> _PairClassification:
        kinds = (first[0], second[0])
        if kinds == ("vertex", "vertex"):
            return self.classify_vertex_vertex(first, second)
        if set(kinds) == {"vertex", "edge"}:
            vertex = first if first[0] == "vertex" else second
            edge = second if first[0] == "vertex" else first
            return self.classify_vertex_edge(vertex, edge)
        if set(kinds) == {"vertex", "face"}:
            vertex = first if first[0] == "vertex" else second
            face = second if first[0] == "vertex" else first
            return self.classify_vertex_face(vertex, face)
        if kinds == ("edge", "edge"):
            return self.classify_edge_edge(first, second)
        if set(kinds) == {"edge", "face"}:
            edge = first if first[0] == "edge" else second
            face = second if first[0] == "edge" else first
            return self.classify_member_face(edge, face)
        if kinds == ("face", "face"):
            return self.classify_face_face(first, second)
        return self.unclassified(first, second, "unsupported_pair_kind")

    # ----------------------------------------------------------- vertex pairs
    def _is_control_only_vertex(self, vertex_id: int) -> bool:
        role_getter = getattr(self.model, "vertex_role", None)
        if callable(role_getter):
            role = role_getter(vertex_id)
            value = getattr(role, "value", role)
            if str(value) in {"curve_control", "construction"}:
                return True
            if str(value) in {"topological", "mixed"}:
                return False
        users = self.model.edges_using_vertex(vertex_id)
        if not users:
            return False
        return not any(
            self.model.edges[edge_id].start == vertex_id
            or self.model.edges[edge_id].end == vertex_id
            for edge_id in users
        )

    def classify_vertex_vertex(
        self, first: SpatialKey, second: SpatialKey
    ) -> _PairClassification:
        if self._is_control_only_vertex(first[1]) or self._is_control_only_vertex(
            second[1]
        ):
            return _PairClassification(True)
        first_point = self.model.vertex_position(first[1])
        second_point = self.model.vertex_position(second[1])
        extent = feature_extent((first_point, second_point))
        tolerance = self.tolerance.effective_length(extent)
        distance = float(np.linalg.norm(first_point - second_point))
        if distance <= tolerance:
            intent = self.ownership_intent(
                first, second, IntersectionKind.TOUCH_POINT
            )
            if intent is not None:
                reason, junction_id = intent
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "coincident vertices are separated by ownership or a declared junction",
                    keys=(first, second),
                    witnesses=(_as_witness("coincident", 0.5 * (first_point + second_point)),),
                    classification=IntersectionKind.TOUCH_POINT,
                    measured_gap=distance,
                    tolerance_used=tolerance,
                    details={"intent": reason, "junction_id": junction_id},
                )
            else:
                self.issue(
                    AuditCode.VERTEX_COINCIDENCE,
                    AuditSeverity.ERROR,
                    "distinct vertex identities occupy the same geometric point",
                    keys=(first, second),
                    witnesses=(_as_witness("coincident", 0.5 * (first_point + second_point)),),
                    classification=IntersectionKind.TOUCH_POINT,
                    measured_gap=distance,
                    tolerance_used=tolerance,
                    recommended_action="reuse one vertex or declare separate-part intent",
                    details={"distance": distance},
                )
        return _PairClassification(True)

    def classify_vertex_edge(
        self, vertex_key: SpatialKey, edge_key: SpatialKey
    ) -> _PairClassification:
        vertex_id, edge_id = vertex_key[1], edge_key[1]
        if self._is_control_only_vertex(vertex_id):
            return _PairClassification(True)
        edge = self.model.edges[edge_id]
        dependent_vertices = {edge.start, edge.end}
        if isinstance(edge.curve, Arc):
            dependent_vertices.add(edge.curve.via_vertex)
        elif isinstance(edge.curve, Spline):
            dependent_vertices.update(edge.curve.control_vertices)
        if vertex_id in dependent_vertices:
            return _PairClassification(True)
        point = self.model.vertex_position(vertex_id)
        closest, parameter, distance = self.model.closest_edge_point(edge_id, point)
        extent = max(self.model.edge_length(edge_id), distance)
        tolerance = self.tolerance.effective_length(extent)
        if distance > tolerance:
            return _PairClassification(True)
        parameter_tolerance = self.tolerance.effective_parameter(
            self.model.edge_length(edge_id), extent
        )
        if parameter_tolerance < parameter < 1.0 - parameter_tolerance:
            intent = self.ownership_intent(
                vertex_key, edge_key, IntersectionKind.TOUCH_POINT
            )
            if intent is not None:
                reason, junction_id = intent
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "a vertex-on-edge incidence is intentional",
                    keys=(vertex_key, edge_key),
                    witnesses=(_as_witness("t_junction", closest),),
                    classification=IntersectionKind.TOUCH_POINT,
                    details={"intent": reason, "junction_id": junction_id},
                )
            else:
                self.issue(
                    AuditCode.VERTEX_EDGE_T_JUNCTION,
                    AuditSeverity.ERROR,
                    "a vertex lies in the interior of an edge without shared topology",
                    keys=(vertex_key, edge_key),
                    witnesses=(_as_witness("t_junction", closest),),
                    classification=IntersectionKind.TOUCH_POINT,
                    measured_gap=distance,
                    tolerance_used=tolerance,
                    recommended_action="split the edge at this vertex or declare connection intent",
                    details={"edge_parameter": parameter, "distance": distance},
                )
        return _PairClassification(True)

    def classify_vertex_face(
        self, vertex_key: SpatialKey, face_key: SpatialKey
    ) -> _PairClassification:
        vertex_id, face_id = vertex_key[1], face_key[1]
        if self._is_control_only_vertex(vertex_id):
            return _PairClassification(True)
        face = self.model.faces[face_id]
        boundary_vertices = {
            self.model.oriented_start_vertex(oriented)
            for loop in (face.loop,) + tuple(face.holes)
            for oriented in loop
        }
        if vertex_id in boundary_vertices:
            return _PairClassification(True)
        point = self.model.vertex_position(vertex_id)
        projected, uv, distance = self.project_to_face_support(face_id, point)
        extent = feature_extent(self.face_points(face_id))
        tolerance = self.tolerance.effective_surface_residual(extent)
        if distance > tolerance or not self.face_support_contains_uv(face_id, uv):
            return _PairClassification(True)
        boundary_distance = min(
            (
                self.model.closest_edge_point(oriented.edge, point)[2]
                for loop in (face.loop,) + tuple(face.holes)
                for oriented in loop
            ),
            default=float("inf"),
        )
        location = "boundary" if boundary_distance <= tolerance else "interior"
        intent = self.ownership_intent(
            vertex_key, face_key, IntersectionKind.TOUCH_POINT
        )
        if intent is not None:
            reason, junction_id = intent
            self.issue(
                AuditCode.INTENTIONAL_COINCIDENCE,
                AuditSeverity.INFO,
                "a vertex-on-face incidence is separated by ownership intent",
                keys=(vertex_key, face_key),
                witnesses=(_as_witness("vertex_face", projected),),
                classification=f"vertex_on_face_{location}",
                measured_gap=distance,
                tolerance_used=tolerance,
                details={"intent": reason, "junction_id": junction_id},
            )
        else:
            self.issue(
                AuditCode.VERTEX_FACE_INTERIOR,
                AuditSeverity.ERROR,
                "a vertex lies on a face without shared topology",
                keys=(vertex_key, face_key),
                witnesses=(_as_witness("vertex_face", projected),),
                classification=f"vertex_on_face_{location}",
                measured_gap=distance,
                tolerance_used=tolerance,
                recommended_action="imprint the vertex into the face or declare separate-part intent",
                details={"location": location, "uv": tuple(float(item) for item in uv)},
            )
        return _PairClassification(True)

    # ------------------------------------------------------------- edge pairs
    def _edge_definition_matches(
        self, first_id: int, second_id: int
    ) -> tuple[bool, bool]:
        first, second = self.model.edges[first_id], self.model.edges[second_id]
        if type(first.curve) is not type(second.curve):
            return False, False

        def defining_points(edge: object) -> np.ndarray:
            curve = edge.curve
            identifiers = [edge.start]
            if isinstance(curve, Arc):
                identifiers.append(curve.via_vertex)
            elif isinstance(curve, Spline):
                identifiers.extend(curve.control_vertices)
            identifiers.append(edge.end)
            return np.asarray(
                [self.model.vertex_position(identifier) for identifier in identifiers]
            )

        first_points, second_points = defining_points(first), defining_points(second)
        if first_points.shape != second_points.shape:
            return False, False
        extent = max(feature_extent(first_points), feature_extent(second_points))
        tolerance = self.tolerance.effective_length(extent)
        forward = float(np.max(np.linalg.norm(first_points - second_points, axis=1))) <= tolerance
        reversed_match = (
            float(np.max(np.linalg.norm(first_points - second_points[::-1], axis=1)))
            <= tolerance
        )
        return forward, reversed_match

    def _shared_endpoint_touch_is_bounded(
        self, first_id: int, second_id: int
    ) -> bool:
        first, second = self.model.edges[first_id], self.model.edges[second_id]
        shared = {first.start, first.end} & {second.start, second.end}
        if not shared:
            return False
        intersection = self.edge_bounds(first_id).intersection(self.edge_bounds(second_id))
        if intersection is None:
            return False
        return (
            float(np.linalg.norm(intersection.extents))
            <= self._edge_pair_length_tolerance(first_id, second_id)
        )

    def _cocircular_arc_relation(
        self, first_id: int, second_id: int
    ) -> tuple[IntersectionKind, tuple[np.ndarray, ...]] | None:
        """Return the exact set relation for two arcs on one support circle."""

        first_frame = self.model.arc_frame(first_id)
        second_frame = self.model.arc_frame(second_id)
        local_extent = max(first_frame.radius, second_frame.radius)
        tolerance = self.tolerance.effective_length(local_extent)
        if (
            float(np.linalg.norm(first_frame.center - second_frame.center))
            > tolerance
            or abs(first_frame.radius - second_frame.radius) > tolerance
            or abs(abs(float(first_frame.normal @ second_frame.normal)) - 1.0)
            > self.tolerance.angular
        ):
            return None

        two_pi = 2.0 * float(np.pi)

        def angle(point: np.ndarray) -> float:
            radial = point - first_frame.center
            return float(
                np.arctan2(radial @ first_frame.e2, radial @ first_frame.e1)
            ) % two_pi

        def segments(start: float, delta: float) -> tuple[tuple[float, float], ...]:
            made = start % two_pi
            if delta > 0.0:
                end = made + delta
                if end <= two_pi:
                    return ((made, end),)
                return ((made, two_pi), (0.0, end - two_pi))
            end = made + delta
            if end >= 0.0:
                return ((end, made),)
            return ((0.0, made), (end + two_pi, two_pi))

        first_start = self.model.vertex_position(
            self.model.edges[first_id].start
        )
        second_start = self.model.vertex_position(
            self.model.edges[second_id].start
        )
        first_segments = segments(0.0, float(first_frame.sweep))
        normal_sign = 1.0 if float(first_frame.normal @ second_frame.normal) >= 0.0 else -1.0
        second_sweep = normal_sign * float(second_frame.sweep)
        second_segments = segments(angle(second_start), second_sweep)
        angle_tolerance = max(
            self.tolerance.angular,
            tolerance / max(first_frame.radius, np.finfo(float).tiny),
        )
        positive: list[tuple[float, float]] = []
        point_angles: list[float] = []
        for first_interval in first_segments:
            for second_interval in second_segments:
                lower = max(first_interval[0], second_interval[0])
                upper = min(first_interval[1], second_interval[1])
                if upper - lower > angle_tolerance:
                    positive.append((lower, upper))
                elif upper >= lower - angle_tolerance:
                    point_angles.append(0.5 * (lower + upper))

        def world(value: float) -> np.ndarray:
            return first_frame.center + first_frame.radius * (
                np.cos(value) * first_frame.e1
                + np.sin(value) * first_frame.e2
            )

        if positive:
            overlap = sum(upper - lower for lower, upper in positive)
            first_length = abs(float(first_frame.sweep))
            second_length = abs(second_sweep)
            kind = (
                IntersectionKind.COINCIDENT
                if abs(overlap - first_length) <= angle_tolerance
                and abs(overlap - second_length) <= angle_tolerance
                else IntersectionKind.OVERLAP_CURVE
            )
            witnesses = tuple(
                world(value)
                for lower, upper in positive
                for value in (lower, upper)
            )
            return kind, witnesses
        if not point_angles:
            return IntersectionKind.DISJOINT, ()
        witnesses: list[np.ndarray] = []
        for value in sorted(point_angles):
            point = world(value)
            if not any(
                float(np.linalg.norm(point - existing)) <= tolerance
                for existing in witnesses
            ):
                witnesses.append(point)
        return IntersectionKind.TOUCH_POINT, tuple(witnesses)

    def _arc_contains_point(
        self, edge_id: int, point: np.ndarray, tolerance: float
    ) -> bool:
        frame = self.model.arc_frame(edge_id)
        offset = point - frame.center
        if (
            abs(float(offset @ frame.normal)) > tolerance
            or abs(float(np.linalg.norm(offset)) - frame.radius) > tolerance
        ):
            return False
        angle = float(np.arctan2(offset @ frame.e2, offset @ frame.e1))
        angular_tolerance = max(
            self.tolerance.angular,
            tolerance / max(frame.radius, np.finfo(float).tiny),
        )
        if frame.sweep > 0.0:
            made = angle % (2.0 * float(np.pi))
            return made <= frame.sweep + angular_tolerance
        made = -((-angle) % (2.0 * float(np.pi)))
        return made >= frame.sweep - angular_tolerance

    def _arc_arc_relation(
        self, first_id: int, second_id: int
    ) -> tuple[IntersectionKind, tuple[np.ndarray, ...]]:
        cocircular = self._cocircular_arc_relation(first_id, second_id)
        if cocircular is not None:
            return cocircular
        first = self.model.arc_frame(first_id)
        second = self.model.arc_frame(second_id)
        extent = max(first.radius, second.radius)
        tolerance = self.tolerance.effective_surface_residual(extent)
        normal_cross = np.cross(first.normal, second.normal)
        sine = float(np.linalg.norm(normal_cross))
        candidates: list[np.ndarray] = []
        if sine <= self.tolerance.angular:
            plane_distance = abs(
                float((second.center - first.center) @ first.normal)
            )
            if plane_distance > tolerance:
                return IntersectionKind.DISJOINT, ()
            delta = second.center - first.center
            delta -= float(delta @ first.normal) * first.normal
            distance = float(np.linalg.norm(delta))
            if (
                distance > first.radius + second.radius + tolerance
                or distance
                < abs(first.radius - second.radius) - tolerance
                or distance <= tolerance
            ):
                return IntersectionKind.DISJOINT, ()
            direction = delta / distance
            station = (
                first.radius * first.radius
                - second.radius * second.radius
                + distance * distance
            ) / (2.0 * distance)
            height_squared = first.radius * first.radius - station * station
            if height_squared < -(tolerance * max(1.0, extent)):
                return IntersectionKind.DISJOINT, ()
            base = first.center + station * direction
            perpendicular = np.cross(first.normal, direction)
            height = float(np.sqrt(max(0.0, height_squared)))
            candidates.append(base + height * perpendicular)
            if height > tolerance:
                candidates.append(base - height * perpendicular)
        else:
            line_direction = normal_cross / sine
            delta = second.center - first.center
            matrix = np.vstack((first.normal, second.normal, line_direction))
            right_hand_side = np.asarray(
                (
                    0.0,
                    float(second.normal @ delta),
                    0.5 * float(line_direction @ delta),
                )
            )
            try:
                local_anchor = np.linalg.solve(matrix, right_hand_side)
            except np.linalg.LinAlgError:
                return IntersectionKind.UNCLASSIFIED, ()
            anchor = first.center + local_anchor
            offset = anchor - first.center
            station = -float(offset @ line_direction)
            closest = anchor + station * line_direction
            height_squared = first.radius * first.radius - float(
                (closest - first.center) @ (closest - first.center)
            )
            if height_squared < -(tolerance * max(1.0, extent)):
                return IntersectionKind.DISJOINT, ()
            height = float(np.sqrt(max(0.0, height_squared)))
            candidates.append(closest + height * line_direction)
            if height > tolerance:
                candidates.append(closest - height * line_direction)

        accepted: list[np.ndarray] = []
        for point in candidates:
            if self._arc_contains_point(
                first_id, point, tolerance
            ) and self._arc_contains_point(second_id, point, tolerance):
                if not any(
                    float(np.linalg.norm(point - existing)) <= tolerance
                    for existing in accepted
                ):
                    accepted.append(point)
        if not accepted:
            return IntersectionKind.DISJOINT, ()
        return (
            IntersectionKind.TOUCH_POINT
            if len(accepted) == 1
            else IntersectionKind.CROSS,
            tuple(accepted),
        )

    def classify_edge_edge(
        self, first_key: SpatialKey, second_key: SpatialKey
    ) -> _PairClassification:
        first, second = self.model.edges[first_key[1]], self.model.edges[second_key[1]]
        pair_length_tolerance = self._edge_pair_length_tolerance(
            first.id, second.id
        )
        if not isinstance(first.curve, Straight) or not isinstance(second.curve, Straight):
            from .intersections import query_intersection

            result = query_intersection(
                self.model,
                self.model.handle("edge", first.id),
                self.model.handle("edge", second.id),
                qualification=self.qualification,
            )
            self.record_intersection_certificate(result)
            if result.kind in (
                IntersectionKind.UNCLASSIFIED,
                IntersectionKind.UNSUPPORTED,
                IntersectionKind.CAPABILITY_MISSING,
            ):
                return self.unclassified(
                    first_key,
                    second_key,
                    result.diagnostics[0]
                    if result.diagnostics
                    else "curved_edge_pair_unclassified",
                )
            if result.certificate is None or not result.certificate.complete:
                return self.unclassified(
                    first_key,
                    second_key,
                    "curved_edge_pair_certificate_incomplete",
                )
            if result.kind is IntersectionKind.DISJOINT:
                return _PairClassification(True)
            shared_vertices = {first.start, first.end} & {
                second.start,
                second.end,
            }
            components = result.components or ()
            if result.kind is IntersectionKind.COINCIDENT:
                forward, reversed_match = self._edge_definition_matches(
                    first.id, second.id
                )
                code = (
                    AuditCode.EDGE_REVERSED_DUPLICATE
                    if reversed_match and not forward
                    else AuditCode.EDGE_DUPLICATE
                )
                message = "distinct curved edge identities describe the same curve"
            elif result.kind is IntersectionKind.OVERLAP_CURVE:
                code = AuditCode.EDGE_COLLINEAR_OVERLAP
                message = "curved edges overlap over a positive length"
            else:
                code = AuditCode.EDGE_CROSSING
                message = "curved edges touch or cross without shared topology"
            for component in components or (None,):
                raw_points = (
                    tuple(np.asarray(point, dtype=float) for point in component.witnesses)
                    if component is not None
                    else ()
                )
                witnesses = tuple(
                    _as_witness(f"intersection/{index}", point)
                    for index, point in enumerate(raw_points)
                )
                ordinary_shared_touch = (
                    result.kind is IntersectionKind.TOUCH_POINT
                    and bool(shared_vertices)
                    and raw_points
                    and all(
                        any(
                            float(
                                np.linalg.norm(
                                    point
                                    - self.model.vertex_position(vertex_id)
                                )
                            )
                            <= pair_length_tolerance
                            for vertex_id in shared_vertices
                        )
                        for point in raw_points
                    )
                )
                if ordinary_shared_touch:
                    self.audit_member_relation(
                        first_key,
                        second_key,
                        result.kind,
                        witnesses,
                    )
                    continue
                intent = self.ownership_intent(
                    first_key,
                    second_key,
                    result.kind,
                    witnesses=raw_points,
                )
                if intent is None:
                    self.issue(
                        code,
                        AuditSeverity.ERROR,
                        message,
                        keys=(first_key, second_key),
                        witnesses=witnesses,
                        classification=result.kind,
                        measured_gap=result.certificate.max_residual
                        if result.certificate is not None
                        else None,
                        tolerance_used=result.tolerance_used,
                    )
                else:
                    reason, junction_id = intent
                    self.issue(
                        AuditCode.INTENTIONAL_COINCIDENCE,
                        AuditSeverity.INFO,
                        "a curved-edge relationship is separated by ownership or junction intent",
                        keys=(first_key, second_key),
                        witnesses=witnesses,
                        classification=result.kind,
                        details={"intent": reason, "junction_id": junction_id},
                    )
                self.audit_member_relation(
                    first_key, second_key, result.kind, witnesses
                )
            return _PairClassification(True)
            if isinstance(first.curve, Arc) and isinstance(second.curve, Arc):
                relation = self._arc_arc_relation(first.id, second.id)
                if relation[0] is IntersectionKind.UNCLASSIFIED:
                    return self.unclassified(
                        first_key,
                        second_key,
                        "ill_conditioned_arc_arc_intersection",
                    )
                if relation is not None:
                    kind, points = relation
                    if kind is IntersectionKind.DISJOINT:
                        return _PairClassification(True)
                    witnesses = tuple(
                        _as_witness(f"intersection/{index}", point)
                        for index, point in enumerate(points)
                    )
                    shared_vertices = {first.start, first.end} & {
                        second.start,
                        second.end,
                    }
                    if kind in (
                        IntersectionKind.TOUCH_POINT,
                        IntersectionKind.CROSS,
                    ) and shared_vertices:
                        if all(
                            any(
                                float(
                                    np.linalg.norm(
                                        point
                                        - self.model.vertex_position(vertex_id)
                                    )
                                )
                                <= pair_length_tolerance
                                for vertex_id in shared_vertices
                            )
                            for point in points
                        ):
                            self.audit_member_relation(
                                first_key,
                                second_key,
                                IntersectionKind.TOUCH_POINT,
                                witnesses,
                            )
                            return _PairClassification(True)
                    if kind is IntersectionKind.COINCIDENT:
                        tolerance = self.tolerance.effective_length(
                            max(
                                self.model.edge_length(first.id),
                                self.model.edge_length(second.id),
                            )
                        )
                        reversed_duplicate = (
                            float(
                                np.linalg.norm(
                                    self.model.vertex_position(first.start)
                                    - self.model.vertex_position(second.end)
                                )
                            )
                            <= tolerance
                            and float(
                                np.linalg.norm(
                                    self.model.vertex_position(first.end)
                                    - self.model.vertex_position(second.start)
                                )
                            )
                            <= tolerance
                        )
                        code = (
                            AuditCode.EDGE_REVERSED_DUPLICATE
                            if reversed_duplicate
                            else AuditCode.EDGE_DUPLICATE
                        )
                        message = (
                            "distinct edge identities describe the same circular arc"
                        )
                    elif kind is IntersectionKind.OVERLAP_CURVE:
                        code = AuditCode.EDGE_COLLINEAR_OVERLAP
                        message = "circular arcs overlap over a positive length"
                    else:
                        code = AuditCode.EDGE_CROSSING
                        message = "circular arcs touch without shared topology"
                    components = (
                        tuple(
                            (
                                (point,),
                                (_as_witness(f"intersection/{index}", point),),
                            )
                            for index, point in enumerate(points)
                        )
                        if kind in (
                            IntersectionKind.CROSS,
                            IntersectionKind.TOUCH_POINT,
                        )
                        else ((points, witnesses),)
                    )
                    for component_points, component_witnesses in components:
                        if shared_vertices and all(
                            any(
                                float(
                                    np.linalg.norm(
                                        point
                                        - self.model.vertex_position(vertex_id)
                                    )
                                )
                                <= pair_length_tolerance
                                for vertex_id in shared_vertices
                            )
                            for point in component_points
                        ):
                            self.audit_member_relation(
                                first_key,
                                second_key,
                                IntersectionKind.TOUCH_POINT,
                                component_witnesses,
                            )
                            continue
                        intent = self.ownership_intent(
                            first_key,
                            second_key,
                            kind,
                            witnesses=component_points,
                        )
                        if intent is None:
                            self.issue(
                                code,
                                AuditSeverity.ERROR,
                                message,
                                keys=(first_key, second_key),
                                witnesses=component_witnesses,
                                classification=kind,
                            )
                        else:
                            reason, junction_id = intent
                            self.issue(
                                AuditCode.INTENTIONAL_COINCIDENCE,
                                AuditSeverity.INFO,
                                "a circular-edge relationship is separated by "
                                "ownership or junction intent",
                                keys=(first_key, second_key),
                                witnesses=component_witnesses,
                                classification=kind,
                                details={
                                    "intent": reason,
                                    "junction_id": junction_id,
                                },
                            )
                        self.audit_member_relation(
                            first_key, second_key, kind, component_witnesses
                        )
                    return _PairClassification(True)
            forward, reversed_match = self._edge_definition_matches(
                first.id, second.id
            )
            if forward or reversed_match:
                result_kind = IntersectionKind.COINCIDENT
                code = (
                    AuditCode.EDGE_REVERSED_DUPLICATE
                    if reversed_match and not forward
                    else AuditCode.EDGE_DUPLICATE
                )
                intent = self.ownership_intent(first_key, second_key, result_kind)
                if intent is not None:
                    reason, junction_id = intent
                    self.issue(
                        AuditCode.INTENTIONAL_COINCIDENCE,
                        AuditSeverity.INFO,
                        "coincident curved edges are separated by ownership or junction intent",
                        keys=(first_key, second_key),
                        classification=result_kind,
                        details={"intent": reason, "junction_id": junction_id},
                    )
                else:
                    self.issue(
                        code,
                        AuditSeverity.ERROR,
                        "distinct curved edge identities describe the same curve",
                        keys=(first_key, second_key),
                        classification=result_kind,
                    )
                self.audit_member_relation(first_key, second_key, result_kind, ())
                return _PairClassification(True)
            if self._shared_endpoint_touch_is_bounded(first.id, second.id):
                return _PairClassification(True)
            return self.unclassified(
                first_key, second_key, "curved_edge_pair_requires_a_qualified_predicate"
            )

        first_start = self.model.vertex_position(first.start)
        first_end = self.model.vertex_position(first.end)
        second_start = self.model.vertex_position(second.start)
        second_end = self.model.vertex_position(second.end)
        result = qualified_segment_segment(
            first_start,
            first_end,
            second_start,
            second_end,
            policy=self.tolerance,
        )
        if result.kind in (
            IntersectionKind.UNCLASSIFIED,
            IntersectionKind.UNSUPPORTED,
            IntersectionKind.CAPABILITY_MISSING,
        ):
            reason = result.diagnostics[0] if result.diagnostics else "segment_unclassified"
            return self.unclassified(first_key, second_key, reason)
        if result.kind is IntersectionKind.DISJOINT:
            return _PairClassification(True)

        shared_vertices = {first.start, first.end} & {second.start, second.end}
        witnesses = _component_witnesses(result)
        if result.kind is IntersectionKind.TOUCH_POINT and shared_vertices:
            # A shared identity at the verified point is ordinary topology.
            witness = np.asarray(result.witnesses[0])
            if any(
                float(np.linalg.norm(witness - self.model.vertex_position(identifier)))
                <= pair_length_tolerance
                for identifier in shared_vertices
            ):
                self.audit_member_relation(first_key, second_key, result.kind, witnesses)
                return _PairClassification(True)

        intent = self.ownership_intent(
            first_key, second_key, result.kind, witnesses=result.witnesses
        )
        overlap_length = (
            float(
                np.linalg.norm(
                    np.asarray(result.witnesses[-1])
                    - np.asarray(result.witnesses[0])
                )
            )
            if result.kind
            in (
                IntersectionKind.COINCIDENT,
                IntersectionKind.OVERLAP_CURVE,
                IntersectionKind.CONTAINED,
            )
            and len(result.witnesses) >= 2
            else None
        )
        if result.kind is IntersectionKind.COINCIDENT:
            tolerance = self.tolerance.effective_length(
                max(self.model.edge_length(first.id), self.model.edge_length(second.id))
            )
            reversed_duplicate = (
                float(np.linalg.norm(first_start - second_end)) <= tolerance
                and float(np.linalg.norm(first_end - second_start)) <= tolerance
            )
            code = (
                AuditCode.EDGE_REVERSED_DUPLICATE
                if reversed_duplicate
                else AuditCode.EDGE_DUPLICATE
            )
            if intent is None:
                self.issue(
                    code,
                    AuditSeverity.ERROR,
                    "distinct edge identities describe the same bounded segment",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=result.kind,
                    overlap_length=overlap_length,
                    tolerance_used=result.tolerance_used,
                )
            else:
                reason, junction_id = intent
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "coincident edges are separated by ownership or junction intent",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=result.kind,
                    overlap_length=overlap_length,
                    tolerance_used=result.tolerance_used,
                    details={"intent": reason, "junction_id": junction_id},
                )
        elif result.kind in (
            IntersectionKind.OVERLAP_CURVE,
            IntersectionKind.CONTAINED,
        ):
            if intent is None:
                self.issue(
                    AuditCode.EDGE_COLLINEAR_OVERLAP,
                    AuditSeverity.ERROR,
                    "collinear edges overlap over a positive length",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=result.kind,
                    overlap_length=overlap_length,
                    tolerance_used=result.tolerance_used,
                    recommended_action="split/imprint or declare overlap intent",
                )
            else:
                reason, junction_id = intent
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "an edge overlap is separated by ownership or junction intent",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=result.kind,
                    overlap_length=overlap_length,
                    tolerance_used=result.tolerance_used,
                    details={"intent": reason, "junction_id": junction_id},
                )
        elif result.kind in (IntersectionKind.CROSS, IntersectionKind.TOUCH_POINT):
            if intent is None:
                self.issue(
                    AuditCode.EDGE_CROSSING,
                    AuditSeverity.ERROR,
                    "edges intersect without shared topology",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=result.kind,
                    tolerance_used=result.tolerance_used,
                    recommended_action="split/imprint the edges or declare connection intent",
                )
            else:
                reason, junction_id = intent
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "an edge intersection is separated by ownership or junction intent",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=result.kind,
                    tolerance_used=result.tolerance_used,
                    details={"intent": reason, "junction_id": junction_id},
                )

        self.audit_member_relation(first_key, second_key, result.kind, witnesses)
        return _PairClassification(True)

    def audit_member_relation(
        self,
        first_edge: SpatialKey,
        second_edge: SpatialKey,
        kind: IntersectionKind,
        witnesses: Iterable[AuditWitness],
    ) -> None:
        if kind not in (
            IntersectionKind.COINCIDENT,
            IntersectionKind.OVERLAP_CURVE,
            IntersectionKind.CONTAINED,
            IntersectionKind.CROSS,
            IntersectionKind.TOUCH_POINT,
        ):
            return
        made_witnesses = tuple(witnesses)
        points = tuple(np.asarray(witness.point) for witness in made_witnesses)
        for first_member in sorted(self.members_for_edge(first_edge[1])):
            for second_member in sorted(self.members_for_edge(second_edge[1])):
                if first_member == second_member:
                    continue
                first_geometry = self.model.edges[first_edge[1]]
                second_geometry = self.model.edges[second_edge[1]]
                if (
                    kind is IntersectionKind.TOUCH_POINT
                    and {first_geometry.start, first_geometry.end}
                    & {second_geometry.start, second_geometry.end}
                ):
                    # A shared topology vertex is an explicit structural
                    # connection, unlike two merely coincident coordinates.
                    continue
                declared = self.declared_junction(
                    first_member,
                    second_member,
                    kind,
                    first_ranges=self._member_ranges_for_edge_candidate(
                        first_member, first_edge[1], kind, points
                    ),
                    second_ranges=self._member_ranges_for_edge_candidate(
                        second_member, second_edge[1], kind, points
                    ),
                )
                first_key, second_key = ("member", first_member), ("member", second_member)
                separate = self.separate_part_intent(first_key, second_key)
                if declared is not None or separate:
                    continue
                code = (
                    AuditCode.MEMBER_MEMBER_OVERLAP
                    if kind in (
                        IntersectionKind.COINCIDENT,
                        IntersectionKind.OVERLAP_CURVE,
                        IntersectionKind.CONTAINED,
                    )
                    else AuditCode.MEMBER_MEMBER_CROSSING
                )
                self.issue(
                    code,
                    AuditSeverity.ERROR,
                    "structural members geometrically meet without a compatible junction",
                    keys=(first_key, second_key, first_edge, second_edge),
                    witnesses=made_witnesses,
                    classification=kind,
                    details={"junction_required": True},
                )

    # -------------------------------------------------------- member / faces
    def _face_boundary_edges(self, face_id: int) -> frozenset[int]:
        face = self.model.faces[face_id]
        return frozenset(
            oriented.edge
            for loop in (face.loop,) + tuple(face.holes)
            for oriented in loop
        )

    def _member_face_attachment(
        self, member_id: int, face_id: int, edge_id: int
    ) -> tuple[int, object] | None:
        boundary = self._face_boundary_edges(face_id)
        for identifier, attachment in sorted(self.model.attachments.items()):
            if attachment.member_id is None or attachment.member_id != member_id:
                continue
            if (
                attachment.target_kind is AttachmentTargetKind.FACE
                and attachment.target_id == face_id
            ) or (
                attachment.target_kind is AttachmentTargetKind.EDGE
                and attachment.target_id in boundary
            ):
                return identifier, attachment.kind
        return None

    def _attachment_compatible(
        self, relationship: str, attachment_kind: object
    ) -> bool:
        if relationship == "embedded":
            return attachment_kind is AttachmentKind.MEMBER_ON_FACE
        if relationship == "crossing":
            return attachment_kind is AttachmentKind.MEMBER_THROUGH_FACE
        if relationship == "boundary":
            return attachment_kind in (
                AttachmentKind.MEMBER_ON_BOUNDARY,
                AttachmentKind.ENDPOINT,
            )
        return False

    def _same_cylinder_support(
        self, first: Cylinder, second: Cylinder
    ) -> bool:
        direction = float(first.axis @ second.axis)
        if abs(abs(direction) - 1.0) > self.tolerance.angular:
            return False
        extent = max(first.radius, second.radius, abs(first.height), abs(second.height))
        tolerance = self.tolerance.effective_surface_residual(extent)
        delta = second.origin - first.origin
        radial = delta - float(delta @ first.axis) * first.axis
        return (
            float(np.linalg.norm(radial)) <= tolerance
            and abs(first.radius - second.radius) <= tolerance
        )

    @staticmethod
    def _overlap_interval(
        first: float,
        second: float,
        *,
        tolerance: float,
    ) -> tuple[str, tuple[float, float] | None]:
        lower, upper = sorted((float(first), float(second)))
        clipped_lower, clipped_upper = max(0.0, lower), min(1.0, upper)
        if clipped_upper - clipped_lower > tolerance:
            return "overlap", (clipped_lower, clipped_upper)
        if clipped_upper >= clipped_lower - tolerance:
            value = min(max(0.5 * (clipped_lower + clipped_upper), 0.0), 1.0)
            return "touch", (value, value)
        return "disjoint", None

    def _cylinder_edge_relation(
        self, edge_id: int, surface: Cylinder
    ) -> tuple[str, tuple[np.ndarray, ...]] | None:
        """Qualify a bounded line/arc against one cylindrical patch."""

        edge = self.model.edges[edge_id]
        extent = max(surface.radius, abs(surface.height), self.model.edge_length(edge_id))
        tolerance = self.tolerance.effective_surface_residual(extent)
        parameter_tolerance = max(
            self.tolerance.parameter,
            tolerance / max(extent, np.finfo(float).tiny),
        )

        def point_on_patch(u: float, v: float) -> np.ndarray:
            return np.asarray(surface.evaluate(float(u), float(v)), dtype=float)

        if isinstance(edge.curve, Arc):
            frame = self.model.arc_frame(edge_id)
            center_offset = frame.center - surface.origin
            center_radial = center_offset - float(center_offset @ surface.axis) * surface.axis
            if (
                abs(abs(float(frame.normal @ surface.axis)) - 1.0)
                > self.tolerance.angular
                or float(np.linalg.norm(center_radial)) > tolerance
                or abs(frame.radius - surface.radius) > tolerance
            ):
                return None
            start = self.model.vertex_position(edge.start)
            radial = start - frame.center
            start_angle = float(
                np.arctan2(
                    radial @ surface.circumferential_direction,
                    radial @ surface.radial_direction,
                )
            )
            angular_delta = float(frame.sweep) * (
                1.0 if float(frame.normal @ surface.axis) >= 0.0 else -1.0
            )
            raw = start_angle - float(surface.start_angle)
            candidates: list[tuple[str, tuple[float, float] | None]] = []
            for turn in range(-2, 3):
                u_start = (raw + turn * 2.0 * float(np.pi)) / float(
                    surface.sweep_angle
                )
                u_end = u_start + angular_delta / float(surface.sweep_angle)
                candidates.append(
                    self._overlap_interval(
                        u_start, u_end, tolerance=parameter_tolerance
                    )
                )
            relation, u_interval = max(
                candidates,
                key=lambda item: (
                    2 if item[0] == "overlap" else 1 if item[0] == "touch" else 0,
                    0.0
                    if item[1] is None
                    else abs(item[1][1] - item[1][0]),
                ),
            )
            axial = float((frame.center - surface.origin) @ surface.axis)
            v = axial / float(surface.height)
            if (
                relation == "disjoint"
                or v < -parameter_tolerance
                or v > 1.0 + parameter_tolerance
                or u_interval is None
            ):
                return "disjoint", ()
            v = min(max(v, 0.0), 1.0)
            witnesses = tuple(
                point_on_patch(u, v)
                for u in dict.fromkeys(u_interval)
            )
            if relation == "touch":
                return "touch", witnesses
            boundary = v <= parameter_tolerance or v >= 1.0 - parameter_tolerance
            return ("boundary" if boundary else "embedded"), witnesses

        if not isinstance(edge.curve, Straight):
            return None
        start = self.model.vertex_position(edge.start)
        end = self.model.vertex_position(edge.end)
        direction = end - start
        length = float(np.linalg.norm(direction))
        unit = direction / length
        start_offset = start - surface.origin
        end_offset = end - surface.origin
        start_axial = float(start_offset @ surface.axis)
        end_axial = float(end_offset @ surface.axis)
        start_radial = start_offset - start_axial * surface.axis
        end_radial = end_offset - end_axial * surface.axis
        generator = (
            float(np.linalg.norm(np.cross(unit, surface.axis)))
            <= self.tolerance.angular
            and abs(float(np.linalg.norm(start_radial)) - surface.radius)
            <= tolerance
            and abs(float(np.linalg.norm(end_radial)) - surface.radius)
            <= tolerance
            and float(np.linalg.norm(start_radial - end_radial)) <= tolerance
        )
        if generator:
            angle = float(
                np.arctan2(
                    start_radial @ surface.circumferential_direction,
                    start_radial @ surface.radial_direction,
                )
            )
            u_values = tuple(
                (angle - float(surface.start_angle) + turn * 2.0 * float(np.pi))
                / float(surface.sweep_angle)
                for turn in range(-2, 3)
            )
            matching_u = next(
                (
                    value
                    for value in u_values
                    if -parameter_tolerance <= value <= 1.0 + parameter_tolerance
                ),
                None,
            )
            if matching_u is None:
                return "disjoint", ()
            u = min(max(matching_u, 0.0), 1.0)
            v_start, v_end = (
                start_axial / float(surface.height),
                end_axial / float(surface.height),
            )
            relation, v_interval = self._overlap_interval(
                v_start, v_end, tolerance=parameter_tolerance
            )
            if relation == "disjoint" or v_interval is None:
                return "disjoint", ()
            witnesses = tuple(
                point_on_patch(u, v)
                for v in dict.fromkeys(v_interval)
            )
            if relation == "touch":
                return "touch", witnesses
            boundary = u <= parameter_tolerance or u >= 1.0 - parameter_tolerance
            return ("boundary" if boundary else "embedded"), witnesses

        # Exact finite line/cylinder roots for a loose transverse segment.
        radial_direction = direction - float(direction @ surface.axis) * surface.axis
        a = float(radial_direction @ radial_direction)
        b = 2.0 * float(start_radial @ radial_direction)
        c = float(start_radial @ start_radial) - surface.radius * surface.radius
        if a <= tolerance * tolerance:
            return "disjoint", ()
        discriminant = b * b - 4.0 * a * c
        if discriminant < -(tolerance * max(1.0, length)) ** 2:
            return "disjoint", ()
        discriminant = max(0.0, discriminant)
        roots = tuple(
            (-b + sign * float(np.sqrt(discriminant))) / (2.0 * a)
            for sign in (-1.0, 1.0)
        )
        witnesses: list[np.ndarray] = []
        for parameter in roots:
            if -parameter_tolerance <= parameter <= 1.0 + parameter_tolerance:
                point = start + min(max(parameter, 0.0), 1.0) * direction
                u, v = surface.local_uv(point)
                if (
                    -parameter_tolerance <= u <= 1.0 + parameter_tolerance
                    and -parameter_tolerance <= v <= 1.0 + parameter_tolerance
                ):
                    witnesses.append(point)
        if not witnesses:
            return "disjoint", ()
        unique: list[np.ndarray] = []
        for point in witnesses:
            if not any(float(np.linalg.norm(point - made)) <= tolerance for made in unique):
                unique.append(point)
        return "crossing", tuple(unique)

    def _face_polygon(self, face_id: int, plane: Plane):
        cache_key = (face_id, id(plane))
        cached = self._polygon_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            from shapely.geometry import Polygon
        except ImportError as error:  # pragma: no cover - optional dependency
            raise GeometryError("planar strict audit requires shapely") from error
        face = self.model.faces[face_id]

        def loop_points(loop: object) -> list[tuple[float, float]]:
            result: list[tuple[float, float]] = []
            for oriented in loop:
                edge = self.model.edges[oriented.edge]
                if not isinstance(edge.curve, Straight):
                    raise GeometryError("curved planar trims are not yet certified")
                vertex_id = (
                    edge.start
                    if oriented.forward
                    else edge.end
                )
                uv = plane.local_uv(self.model.vertex_position(vertex_id))
                result.append((float(uv[0]), float(uv[1])))
            return result

        polygon = Polygon(
            loop_points(face.loop),
            [loop_points(loop) for loop in face.holes],
        )
        if not polygon.is_valid:
            raise GeometryError(f"face {face_id} has an invalid planar polygon")
        self._polygon_cache[cache_key] = polygon
        return polygon

    def _plane_uv_length_tolerance(self, plane: Plane, extent: float) -> float:
        singular = np.linalg.svd(
            np.column_stack((plane.u_vector, plane.v_vector)),
            compute_uv=False,
        )
        minimum_scale = float(np.min(singular))
        if minimum_scale <= 0.0:
            raise GeometryError("plane parameterization is singular")
        return self.tolerance.effective_length(extent) / minimum_scale

    def classify_member_face(
        self, edge_key: SpatialKey, face_key: SpatialKey
    ) -> _PairClassification:
        edge_id, face_id = edge_key[1], face_key[1]
        edge = self.model.edges[edge_id]
        members = sorted(self.members_for_edge(edge_id))
        edge_length = self.model.edge_length(edge_id)
        face_extent = feature_extent(self.face_points(face_id))
        local_extent = max(edge_length, face_extent)
        local_length_tolerance = self.tolerance.effective_length(local_extent)
        boundary = self._face_boundary_edges(face_id)
        if edge_id in boundary:
            # Reusing the exact edge definition is explicit conformal topology.
            # It needs no additional attachment merely to prove coincidence.
            for member_id in members:
                attachment = self._member_face_attachment(member_id, face_id, edge_id)
                if attachment is not None and not self._attachment_compatible(
                    "boundary", attachment[1]
                ):
                    self.issue(
                        AuditCode.ATTACHMENT_INCONSISTENT,
                        AuditSeverity.ERROR,
                        "a member-face attachment kind does not match shared boundary topology",
                        keys=(
                            ("attachment", attachment[0]),
                            ("member", member_id),
                            edge_key,
                            face_key,
                        ),
                        classification="boundary",
                    )
            return _PairClassification(True)

        face = self.model.faces[face_id]
        plane = _surface_plane(face)
        relationship: str | None = None
        witnesses: tuple[AuditWitness, ...] = ()
        if plane is None or not isinstance(edge.curve, Straight):
            from .intersections import query_intersection

            result = query_intersection(
                self.model,
                self.model.handle("edge", edge_id),
                self.model.handle("face", face_id),
                qualification=self.qualification,
            )
            self.record_intersection_certificate(result)
            if result.kind in (
                IntersectionKind.UNCLASSIFIED,
                IntersectionKind.UNSUPPORTED,
                IntersectionKind.CAPABILITY_MISSING,
            ):
                return self.unclassified(
                    edge_key,
                    face_key,
                    result.diagnostics[0]
                    if result.diagnostics
                    else "edge_face_pair_unclassified",
                    result.kind,
                )
            if result.certificate is None or not result.certificate.complete:
                return self.unclassified(
                    edge_key,
                    face_key,
                    "edge_face_pair_certificate_incomplete",
                )
            if result.kind is IntersectionKind.DISJOINT:
                return _PairClassification(True)
            witnesses = _component_witnesses(result)
            if result.dimension is IntersectionDimension.POINT:
                relationship = "touch" if result.kind is IntersectionKind.TOUCH_POINT else "crossing"
            elif result.dimension in (
                IntersectionDimension.CURVE,
                IntersectionDimension.REGION,
            ):
                relationship = (
                    "boundary"
                    if any(edge_id == item for item in boundary)
                    else "embedded"
                )
            else:
                return self.unclassified(
                    edge_key,
                    face_key,
                    "qualified edge-face result has no material dimension",
                )
        else:
            start = self.model.vertex_position(edge.start)
            end = self.model.vertex_position(edge.end)
            result = qualified_line_plane(
                start,
                end - start,
                plane,
                policy=self.tolerance,
                characteristic_length=max(
                    self.model.edge_length(edge_id),
                    feature_extent(self.face_points(face_id)),
                ),
            )
            if result.kind in (
                IntersectionKind.UNCLASSIFIED,
                IntersectionKind.UNSUPPORTED,
                IntersectionKind.CAPABILITY_MISSING,
            ):
                return self.unclassified(
                    edge_key,
                    face_key,
                    result.diagnostics[0]
                    if result.diagnostics
                    else "line_plane_unclassified",
                )
            polygon = self._face_polygon(face_id, plane)
            parameter_tolerance = self.tolerance.effective_parameter(
                edge_length, local_extent
            )
            if result.kind is IntersectionKind.DISJOINT:
                return _PairClassification(True)
            if result.kind is IntersectionKind.CROSS:
                component = result.components[0]
                physical = (
                    component.first_parameter[0]
                    if component.first_parameter
                    else float("inf")
                )
                normalized = physical / edge_length
                if not (
                    -parameter_tolerance
                    <= normalized
                    <= 1.0 + parameter_tolerance
                ):
                    return _PairClassification(True)
                world = np.asarray(component.witnesses[0])
                uv = plane.local_uv(world)
                try:
                    from shapely.geometry import Point
                except ImportError as error:  # pragma: no cover
                    raise GeometryError("planar strict audit requires shapely") from error
                if polygon.buffer(
                    self._plane_uv_length_tolerance(plane, local_extent)
                ).covers(Point(*uv)):
                    relationship = "crossing"
                    witnesses = (_as_witness("edge_face_crossing", world),)
            elif result.kind in (
                IntersectionKind.OVERLAP_CURVE,
                IntersectionKind.CONTAINED,
            ):
                try:
                    from shapely.geometry import LineString
                except ImportError as error:  # pragma: no cover
                    raise GeometryError("planar strict audit requires shapely") from error
                start_uv, end_uv = plane.local_uv(start), plane.local_uv(end)
                segment = LineString((start_uv, end_uv))
                material = segment.intersection(polygon)
                boundary_material = segment.intersection(polygon.boundary)
                parameter_scale = float(
                    np.linalg.norm(
                        plane.evaluate(*end_uv) - plane.evaluate(*start_uv)
                    )
                ) / max(float(segment.length), np.finfo(float).tiny)
                if (
                    float(boundary_material.length) * parameter_scale
                    > local_length_tolerance
                ):
                    relationship = "boundary"
                elif (
                    float(material.length) * parameter_scale
                    > local_length_tolerance
                ):
                    relationship = "embedded"
                if relationship is not None:
                    witnesses = (
                        _as_witness("edge_face_start", start),
                        _as_witness("edge_face_end", end),
                    )
        if relationship is None:
            return _PairClassification(True)

        shared_vertices = {edge.start, edge.end} & {
            self.model.oriented_start_vertex(oriented)
            for loop in (face.loop,) + tuple(face.holes)
            for oriented in loop
        }
        if relationship == "touch" and shared_vertices:
            if all(
                any(
                    float(
                        np.linalg.norm(
                            np.asarray(witness.point)
                            - self.model.vertex_position(vertex_id)
                        )
                    )
                    <= local_length_tolerance
                    for vertex_id in shared_vertices
                )
                for witness in witnesses
            ):
                return _PairClassification(True)

        if not members:
            self.issue(
                AuditCode.NONCONFORMAL_INTERFACE,
                AuditSeverity.ERROR,
                "a loose edge meets face material without shared topology",
                keys=(edge_key, face_key),
                witnesses=witnesses,
                classification=relationship,
            )
            return _PairClassification(True)

        for member_id in members:
            member_key = ("member", member_id)
            attachment = self._member_face_attachment(member_id, face_id, edge_id)
            compatible = (
                attachment is not None
                and self._attachment_compatible(relationship, attachment[1])
            )
            if compatible or self.separate_part_intent(member_key, face_key):
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "a member-face relationship is qualified by attachment or ownership",
                    keys=(member_key, edge_key, face_key),
                    witnesses=witnesses,
                    classification=relationship,
                    details={
                        "attachment_id": None if attachment is None else attachment[0],
                        "intent": "separate_parts" if attachment is None else "attachment",
                    },
                )
            else:
                if attachment is not None:
                    self.issue(
                        AuditCode.ATTACHMENT_INCONSISTENT,
                        AuditSeverity.ERROR,
                        "a member-face attachment kind does not match the qualified geometry",
                        keys=(
                            ("attachment", attachment[0]),
                            member_key,
                            edge_key,
                            face_key,
                        ),
                        witnesses=witnesses,
                        classification=relationship,
                    )
                code = (
                    AuditCode.MEMBER_FACE_EMBEDDED
                    if relationship == "embedded"
                    else AuditCode.MEMBER_FACE_BOUNDARY_COINCIDENT
                    if relationship == "boundary"
                    else AuditCode.MEMBER_FACE_CROSSING
                )
                self.issue(
                    code,
                    AuditSeverity.ERROR,
                    "a member-face relationship has no qualified attachment",
                    keys=(member_key, edge_key, face_key),
                    witnesses=witnesses,
                    classification=relationship,
                )
        return _PairClassification(True)

    # ------------------------------------------------------------- face pairs
    def _face_intervals_on_line(
        self,
        face_id: int,
        plane: Plane,
        polygon: object,
        witness: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[tuple[float, float], ...]:
        try:
            from shapely.geometry import LineString
        except ImportError as error:  # pragma: no cover
            raise GeometryError("planar strict audit requires shapely") from error
        points = self.face_points(face_id)
        local_tolerance = self.tolerance.effective_length(feature_extent(points))
        span = max(
            1.0,
            float(np.max(np.linalg.norm(points - witness, axis=1))) + local_tolerance,
        )
        world_start, world_end = witness - 2.0 * span * direction, witness + 2.0 * span * direction
        line = LineString((plane.local_uv(world_start), plane.local_uv(world_end)))
        clipped = polygon.intersection(line)
        components = []
        if clipped.is_empty:
            return ()
        if clipped.geom_type == "LineString":
            components = [clipped]
        elif clipped.geom_type == "MultiLineString":
            components = list(clipped.geoms)
        elif clipped.geom_type == "GeometryCollection":
            components = [item for item in clipped.geoms if item.geom_type == "LineString"]
        intervals: list[tuple[float, float]] = []
        for component in components:
            coordinates = list(component.coords)
            if len(coordinates) < 2:
                continue
            endpoints = (
                plane.evaluate(*coordinates[0]),
                plane.evaluate(*coordinates[-1]),
            )
            parameters = sorted(float((point - witness) @ direction) for point in endpoints)
            intervals.append((parameters[0], parameters[1]))
        return tuple(sorted(intervals))

    @staticmethod
    def _periodic_segments(
        start: float, delta: float
    ) -> tuple[tuple[float, float], ...]:
        two_pi = 2.0 * float(np.pi)
        made = float(start) % two_pi
        if delta >= 0.0:
            end = made + float(delta)
            if end <= two_pi:
                return ((made, end),)
            return ((made, two_pi), (0.0, end - two_pi))
        end = made + float(delta)
        if end >= 0.0:
            return ((end, made),)
        return ((0.0, made), (end + two_pi, two_pi))

    def _cylinder_ranges(
        self, reference: Cylinder, candidate: Cylinder
    ) -> tuple[tuple[tuple[float, float], ...], tuple[float, float]]:
        start_point = candidate.evaluate(0.0, 0.0)
        offset = start_point - reference.origin
        axial = float(offset @ reference.axis)
        radial = offset - axial * reference.axis
        angle = float(
            np.arctan2(
                radial @ reference.circumferential_direction,
                radial @ reference.radial_direction,
            )
        )
        orientation = 1.0 if float(reference.axis @ candidate.axis) >= 0.0 else -1.0
        angular = self._periodic_segments(
            angle, orientation * float(candidate.sweep_angle)
        )
        end_axial = float(
            (candidate.evaluate(0.0, 1.0) - reference.origin) @ reference.axis
        )
        return angular, tuple(sorted((axial, end_axial)))

    def _classify_same_cylinder_faces(
        self,
        first_key: SpatialKey,
        second_key: SpatialKey,
        first: Cylinder,
        second: Cylinder,
        shared_edges: frozenset[int],
        shared_vertices: frozenset[int],
    ) -> _PairClassification:
        first_angular, first_axial = self._cylinder_ranges(first, first)
        second_angular, second_axial = self._cylinder_ranges(first, second)
        local_extent = max(
            first.radius,
            second.radius,
            abs(first.height),
            abs(second.height),
        )
        length_tolerance = self.tolerance.effective_length(local_extent)
        area_tolerance = self.tolerance.effective_area(local_extent)
        angular_tolerance = max(
            self.tolerance.angular,
            length_tolerance / max(first.radius, np.finfo(float).tiny),
        )
        angular_overlaps: list[tuple[float, float]] = []
        angular_touches: list[float] = []
        for first_interval in first_angular:
            for second_interval in second_angular:
                lower = max(first_interval[0], second_interval[0])
                upper = min(first_interval[1], second_interval[1])
                if upper - lower > angular_tolerance:
                    angular_overlaps.append((lower, upper))
                elif upper >= lower - angular_tolerance:
                    angular_touches.append(0.5 * (lower + upper))
        axial_lower = max(first_axial[0], second_axial[0])
        axial_upper = min(first_axial[1], second_axial[1])
        axial_overlap = axial_upper - axial_lower
        angular_overlap = sum(
            upper - lower for lower, upper in angular_overlaps
        )
        angular_touch = bool(angular_touches)
        axial_touch = axial_upper >= axial_lower - length_tolerance
        if (
            (angular_overlap <= angular_tolerance and not angular_touch)
            or (axial_overlap <= length_tolerance and not axial_touch)
        ):
            return _PairClassification(True)

        positive_region = (
            angular_overlap > angular_tolerance
            and axial_overlap > length_tolerance
        )
        if not positive_region:
            if shared_edges or shared_vertices:
                return _PairClassification(True)
            self.issue(
                AuditCode.NONCONFORMAL_INTERFACE,
                AuditSeverity.ERROR,
                "cylindrical face trims meet without shared boundary topology",
                keys=(first_key, second_key),
                classification="boundary_contact",
            )
            return _PairClassification(True)

        area = first.radius * angular_overlap * axial_overlap
        first_angular_length = sum(
            upper - lower for lower, upper in first_angular
        )
        second_angular_length = sum(
            upper - lower for lower, upper in second_angular
        )
        first_area = (
            first.radius
            * first_angular_length
            * (first_axial[1] - first_axial[0])
        )
        second_area = (
            first.radius
            * second_angular_length
            * (second_axial[1] - second_axial[0])
        )
        if (
            abs(area - first_area) <= area_tolerance
            and abs(area - second_area) <= area_tolerance
        ):
            code, relationship = AuditCode.FACE_COINCIDENT, "coincident"
        elif abs(area - min(first_area, second_area)) <= area_tolerance:
            code, relationship = AuditCode.FACE_CONTAINMENT, "containment"
        else:
            code, relationship = AuditCode.FACE_COPLANAR_OVERLAP, "overlap_region"
        if self.separate_part_intent(first_key, second_key):
            self.issue(
                AuditCode.INTENTIONAL_COINCIDENCE,
                AuditSeverity.INFO,
                "cylindrical face overlap is retained across separate parts",
                keys=(first_key, second_key),
                classification=relationship,
                details={"area": area, "intent": "separate_parts"},
            )
        else:
            self.issue(
                code,
                AuditSeverity.ERROR,
                "faces have a positive-area overlap on one cylinder support",
                keys=(first_key, second_key),
                classification=relationship,
                details={"area": area},
            )
        return _PairClassification(True)

    def classify_face_face(
        self, first_key: SpatialKey, second_key: SpatialKey
    ) -> _PairClassification:
        first_face, second_face = self.model.faces[first_key[1]], self.model.faces[second_key[1]]
        first_edges = self._face_boundary_edges(first_face.id)
        second_edges = self._face_boundary_edges(second_face.id)
        shared_edges = first_edges & second_edges
        first_vertices = frozenset(
            self.model.oriented_start_vertex(oriented)
            for loop in (first_face.loop,) + tuple(first_face.holes)
            for oriented in loop
        )
        second_vertices = frozenset(
            self.model.oriented_start_vertex(oriented)
            for loop in (second_face.loop,) + tuple(second_face.holes)
            for oriented in loop
        )
        shared_vertices = first_vertices & second_vertices
        if not (
            isinstance(first_face.surface, Plane)
            and isinstance(second_face.surface, Plane)
        ):
            from .intersections import query_intersection

            qualified = query_intersection(
                self.model,
                self.model.handle("face", first_face.id),
                self.model.handle("face", second_face.id),
                qualification=self.qualification,
            )
            self.record_intersection_certificate(qualified)
            if qualified.kind in (
                IntersectionKind.UNCLASSIFIED,
                IntersectionKind.UNSUPPORTED,
                IntersectionKind.CAPABILITY_MISSING,
            ):
                return self.unclassified(
                    first_key,
                    second_key,
                    qualified.diagnostics[0]
                    if qualified.diagnostics
                    else "curved_face_pair_unclassified",
                    qualified.kind,
                )
            if qualified.certificate is None or not qualified.certificate.complete:
                return self.unclassified(
                    first_key,
                    second_key,
                    "curved_face_pair_certificate_incomplete",
                )
            if qualified.kind is IntersectionKind.DISJOINT:
                return _PairClassification(True)
            witnesses = _component_witnesses(qualified)
            if (
                qualified.dimension is IntersectionDimension.CURVE
                and shared_edges
                and qualified.kind
                in (IntersectionKind.OVERLAP_CURVE, IntersectionKind.COINCIDENT)
            ):
                return _PairClassification(True)
            if (
                qualified.dimension is IntersectionDimension.POINT
                and shared_vertices
                and qualified.witnesses
            ):
                tolerance = self.tolerance.effective_length(
                    self._face_pair_extent(first_face.id, second_face.id)
                )
                if all(
                    any(
                        float(
                            np.linalg.norm(
                                np.asarray(point, dtype=float)
                                - self.model.vertex_position(vertex_id)
                            )
                        )
                        <= tolerance
                        for vertex_id in shared_vertices
                    )
                    for point in qualified.witnesses
                ):
                    return _PairClassification(True)
            if qualified.dimension is IntersectionDimension.REGION:
                code = (
                    AuditCode.FACE_COINCIDENT
                    if qualified.kind is IntersectionKind.COINCIDENT
                    else AuditCode.FACE_CONTAINMENT
                    if qualified.kind is IntersectionKind.CONTAINED
                    else AuditCode.FACE_COPLANAR_OVERLAP
                )
                message = "curved faces have a qualified coincident region"
            else:
                code = AuditCode.FACE_FACE_CROSSING
                message = "curved faces intersect without conformal shared topology"
            if self.separate_part_intent(first_key, second_key):
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "a curved-face relationship is retained across separate parts",
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=qualified.kind,
                    details={"intent": "separate_parts"},
                )
            else:
                self.issue(
                    code,
                    AuditSeverity.ERROR,
                    message,
                    keys=(first_key, second_key),
                    witnesses=witnesses,
                    classification=qualified.kind,
                    measured_gap=(
                        qualified.certificate.max_residual
                        if qualified.certificate is not None
                        else None
                    ),
                    tolerance_used=qualified.tolerance_used,
                )
            return _PairClassification(True)
        if (
            isinstance(first_face.surface, Cylinder)
            and isinstance(second_face.surface, Cylinder)
            and self._same_cylinder_support(
                first_face.surface, second_face.surface
            )
        ):
            return self._classify_same_cylinder_faces(
                first_key,
                second_key,
                first_face.surface,
                second_face.surface,
                frozenset(shared_edges),
                frozenset(shared_vertices),
            )
        first_plane, second_plane = _surface_plane(first_face), _surface_plane(second_face)
        if first_plane is None or second_plane is None:
            if shared_edges:
                return self.unclassified(
                    first_key,
                    second_key,
                    "shared_curved_face_topology_cannot_exclude_an_additional_intersection",
                )
            return self.unclassified(
                first_key, second_key, "non_planar_face_pair_requires_a_surface_predicate"
            )
        result = qualified_plane_plane(
            first_plane,
            second_plane,
            policy=self.tolerance,
            characteristic_length=max(
                feature_extent(self.face_points(first_face.id)),
                feature_extent(self.face_points(second_face.id)),
            ),
        )
        if result.kind in (
            IntersectionKind.UNCLASSIFIED,
            IntersectionKind.UNSUPPORTED,
            IntersectionKind.CAPABILITY_MISSING,
        ):
            return self.unclassified(
                first_key,
                second_key,
                result.diagnostics[0] if result.diagnostics else "plane_pair_unclassified",
            )
        if result.kind is IntersectionKind.DISJOINT:
            return _PairClassification(True)
        if result.kind is IntersectionKind.COINCIDENT:
            pair_extent = self._face_pair_extent(
                first_face.id, second_face.id
            )
            pair_area_tolerance = self.tolerance.effective_area(pair_extent)
            # A zero-width overlap of projected AABBs proves that two planar
            # trims can share only boundary, avoiding polygon work for the
            # common structured-grid adjacency case.
            first_uv = np.asarray(
                [first_plane.local_uv(point) for point in self.face_points(first_face.id)]
            )
            second_uv = np.asarray(
                [first_plane.local_uv(point) for point in self.face_points(second_face.id)]
            )
            overlap_extents = np.minimum(first_uv.max(axis=0), second_uv.max(axis=0)) - np.maximum(
                first_uv.min(axis=0), second_uv.min(axis=0)
            )
            if float(np.min(overlap_extents)) <= self._plane_uv_length_tolerance(
                first_plane, pair_extent
            ):
                return _PairClassification(True)
            first_polygon = self._face_polygon(first_face.id, first_plane)
            # Project both trims into the same coordinate system for coplanar area.
            second_in_first = self._face_polygon(second_face.id, first_plane)
            intersection = first_polygon.intersection(second_in_first)
            area_scale = float(
                np.linalg.norm(np.cross(first_plane.u_vector, first_plane.v_vector))
            )
            area = float(intersection.area) * area_scale
            if area <= pair_area_tolerance:
                return _PairClassification(True)
            first_area = float(first_polygon.area) * area_scale
            second_area = float(second_in_first.area) * area_scale
            symmetric = (
                float(first_polygon.symmetric_difference(second_in_first).area)
                * area_scale
            )
            if symmetric <= pair_area_tolerance:
                code = AuditCode.FACE_COINCIDENT
                relation = "coincident"
            elif abs(area - min(first_area, second_area)) <= pair_area_tolerance:
                code = AuditCode.FACE_CONTAINMENT
                relation = "containment"
            else:
                code = AuditCode.FACE_COPLANAR_OVERLAP
                relation = "overlap_region"
            if self.separate_part_intent(first_key, second_key):
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "coplanar face overlap is retained across separate parts",
                    keys=(first_key, second_key),
                    classification=relation,
                    overlap_area=area,
                    tolerance_used=pair_area_tolerance,
                    details={"area": area, "intent": "separate_parts"},
                )
            else:
                representative = intersection.representative_point()
                world = first_plane.evaluate(representative.x, representative.y)
                self.issue(
                    code,
                    AuditSeverity.ERROR,
                    "faces have a positive-area coplanar overlap",
                    keys=(first_key, second_key),
                    witnesses=(_as_witness("overlap", world),),
                    classification=relation,
                    overlap_area=area,
                    tolerance_used=pair_area_tolerance,
                    recommended_action="imprint/fragment the overlap or declare separate-part intent",
                    details={"area": area},
                )
            return _PairClassification(True)

        if result.kind is not IntersectionKind.CROSS:
            return self.unclassified(first_key, second_key, "unexpected_plane_relation")
        component = result.components[0]
        if component.direction is None:
            return self.unclassified(first_key, second_key, "plane_crossing_has_no_direction")
        witness = np.asarray(component.witnesses[0], dtype=float)
        direction = np.asarray(component.direction, dtype=float)
        first_polygon = self._face_polygon(first_face.id, first_plane)
        second_polygon = self._face_polygon(second_face.id, second_plane)
        first_intervals = self._face_intervals_on_line(
            first_face.id, first_plane, first_polygon, witness, direction
        )
        second_intervals = self._face_intervals_on_line(
            second_face.id, second_plane, second_polygon, witness, direction
        )
        pair_length_tolerance = self.tolerance.effective_length(
            self._face_pair_extent(first_face.id, second_face.id)
        )
        overlaps: list[tuple[float, float]] = []
        for first_interval in first_intervals:
            for second_interval in second_intervals:
                lower = max(first_interval[0], second_interval[0])
                upper = min(first_interval[1], second_interval[1])
                if upper - lower > pair_length_tolerance:
                    overlaps.append((lower, upper))
        if not overlaps:
            return _PairClassification(True)
        if shared_edges:
            allowed: list[tuple[float, float]] = []
            for edge_id in sorted(shared_edges):
                edge = self.model.edges[edge_id]
                if not isinstance(edge.curve, Straight):
                    return self.unclassified(
                        first_key,
                        second_key,
                        "a non-straight shared edge cannot be qualified on intersecting planes",
                    )
                parameters = sorted(
                    float((self.model.vertex_position(vertex_id) - witness) @ direction)
                    for vertex_id in (edge.start, edge.end)
                )
                allowed.append((parameters[0], parameters[1]))

            def covered(interval: tuple[float, float]) -> bool:
                pending = interval[0]
                for lower, upper in sorted(allowed):
                    if upper < pending - pair_length_tolerance:
                        continue
                    if lower > pending + pair_length_tolerance:
                        break
                    pending = max(pending, upper)
                    if pending >= interval[1] - pair_length_tolerance:
                        return True
                return False

            overlaps = [interval for interval in overlaps if not covered(interval)]
            if not overlaps:
                return _PairClassification(True)
        witnesses = tuple(
            _as_witness(label, witness + parameter * direction)
            for index, interval in enumerate(overlaps)
            for label, parameter in (
                (f"crossing/{index}/start", interval[0]),
                (f"crossing/{index}/end", interval[1]),
            )
        )
        if self.separate_part_intent(first_key, second_key):
            self.issue(
                AuditCode.INTENTIONAL_COINCIDENCE,
                AuditSeverity.INFO,
                "crossing faces are retained across separate parts",
                keys=(first_key, second_key),
                witnesses=witnesses,
                classification=IntersectionKind.CROSS,
                details={"intent": "separate_parts", "component_count": len(overlaps)},
            )
        else:
            self.issue(
                AuditCode.FACE_FACE_CROSSING,
                AuditSeverity.ERROR,
                "faces cross over one or more un-imprinted material intervals",
                keys=(first_key, second_key),
                witnesses=witnesses,
                classification=IntersectionKind.CROSS,
                overlap_length=sum(upper - lower for lower, upper in overlaps),
                tolerance_used=pair_length_tolerance,
                recommended_action="imprint the shared intersection curves",
                details={"component_count": len(overlaps)},
            )
        return _PairClassification(True)

    # ------------------------------------------------------ structural pairs
    def audit_same_edge_members(
        self, edge_ids: Iterable[int] | None = None
    ) -> None:
        selected = None if edge_ids is None else frozenset(int(item) for item in edge_ids)
        candidates = [
            (edge_id, first, second)
            for edge_id in sorted(self.model.edges if selected is None else selected)
            if edge_id in self.model.edges
            for member_ids in (self.members_for_edge(edge_id),)
            for index, first in enumerate(sorted(member_ids))
            for second in sorted(member_ids)[index + 1 :]
        ]
        self.collector.record_candidates(len(candidates))
        for edge_id, first, second in candidates:
            self.collector.record_narrow_phase()
            declared = self.declared_junction(
                first,
                second,
                IntersectionKind.COINCIDENT,
                first_ranges=self._member_ranges_for_edge_candidate(
                    first, edge_id, IntersectionKind.COINCIDENT, ()
                ),
                second_ranges=self._member_ranges_for_edge_candidate(
                    second, edge_id, IntersectionKind.COINCIDENT, ()
                ),
            )
            separate = self.separate_part_intent(("member", first), ("member", second))
            if declared is not None or separate:
                self.issue(
                    AuditCode.INTENTIONAL_COINCIDENCE,
                    AuditSeverity.INFO,
                    "members intentionally share one geometry-edge definition",
                    keys=(("member", first), ("member", second), ("edge", edge_id)),
                    classification=IntersectionKind.COINCIDENT,
                    details={
                        "intent": "separate_parts" if separate else "declared_junction",
                        "junction_id": declared,
                    },
                )
            else:
                self.issue(
                    AuditCode.MEMBER_MEMBER_OVERLAP,
                    AuditSeverity.ERROR,
                    "members share an axis edge without an overlap junction",
                    keys=(("member", first), ("member", second), ("edge", edge_id)),
                    classification=IntersectionKind.COINCIDENT,
                    details={"junction_required": True},
                )
            self.collector.record_classification(classified=True)

    def member_point(self, member_id: int, parameter: float) -> np.ndarray:
        member = self.model.members[member_id]
        made = float(parameter)
        uses = [self.model.member_edge_uses[identifier] for identifier in member.edge_use_ids]
        selected = next(
            (
                use
                for use in uses
                if use.parent_range.start - self.tolerance.parameter
                <= made
                <= use.parent_range.end + self.tolerance.parameter
            ),
            None,
        )
        if selected is None:
            raise GeometryError("junction parameter is outside a member axis")
        span = selected.parent_range.end - selected.parent_range.start
        local = (made - selected.parent_range.start) / span
        if selected.orientation is Orientation.REVERSED:
            local = 1.0 - local
        return self.model.sample_edge(selected.edge_id, np.asarray((local,)))[0]

    def _member_breakpoints(
        self, member_id: int, lower: float, upper: float
    ) -> tuple[float, ...]:
        member = self.model.members[member_id]
        values = {float(lower), float(upper)}
        for use_id in member.edge_use_ids:
            use = self.model.member_edge_uses[use_id]
            for value in (use.parent_range.start, use.parent_range.end):
                if lower < value < upper:
                    values.add(float(value))
        ordered = sorted(values)
        values.update(
            0.5 * (first + second)
            for first, second in zip(ordered, ordered[1:])
        )
        return tuple(sorted(values))

    def audit_attachment_geometry(
        self, attachment_ids: Iterable[int] | None = None
    ) -> None:
        """Qualify declared relationships even when their AABBs do not meet.

        Spatial member/face classification cannot discover a false attachment
        whose member is far from its target.  Attachments are therefore their
        own audit candidates.  Piecewise-straight members against planes or
        straight target edges are checked exactly at every member-use break;
        point-valued declarations are exact for every supported surface.
        Other curve/surface combinations remain explicitly unclassified.
        """

        selected = (
            None
            if attachment_ids is None
            else frozenset(int(item) for item in attachment_ids)
        )
        attachments = tuple(
            (identifier, attachment)
            for identifier, attachment in sorted(self.model.attachments.items())
            if selected is None or identifier in selected
        )
        self.collector.record_candidates(len(attachments))
        for attachment_id, attachment in attachments:
            self.collector.record_narrow_phase()
            attachment_key = ("attachment", attachment_id)
            if attachment.evidence is AttachmentEvidence.UNVERIFIED:
                self.issue(
                    AuditCode.UNVERIFIED_CLASSIFICATION,
                    AuditSeverity.BLOCKER,
                    "attachment evidence is explicitly unverified",
                    keys=(attachment_key, attachment.source_key, attachment.target_key),
                    classification="unverified_attachment",
                    classification_confidence=0.0,
                    evidence_quality="unverified",
                    tolerance_used=attachment.tolerance_used,
                    recommended_action=(
                        "requalify the attachment through a verified intersection plan"
                    ),
                    blocks_strict_handoff=True,
                    details={
                        "declared_max_residual": attachment.max_residual,
                        "declared_tolerance": attachment.tolerance_used,
                    },
                )
            try:
                if attachment.source_kind == "vertex":
                    source_point = self.model.vertex_position(attachment.source_id)
                    if attachment.target_kind is AttachmentTargetKind.FACE:
                        u_range, v_range = attachment.target_parameters
                        uv = (u_range.start, v_range.start)
                        target_point = self.model.face_support_point(
                            attachment.target_id, *uv
                        )
                        trim_valid = self.face_support_contains_uv(
                            attachment.target_id, uv
                        )
                        extent = feature_extent(
                            self.face_points(attachment.target_id)
                        )
                    elif attachment.target_kind is AttachmentTargetKind.EDGE:
                        parameter = attachment.target_parameters[0].start
                        target_point = self.model.sample_edge(
                            attachment.target_id, np.asarray((parameter,))
                        )[0]
                        trim_valid = True
                        extent = self.model.edge_length(attachment.target_id)
                    elif attachment.target_kind is AttachmentTargetKind.VERTEX:
                        target_point = self.model.vertex_position(attachment.target_id)
                        trim_valid = True
                        extent = feature_extent((source_point, target_point))
                    elif attachment.target_kind is AttachmentTargetKind.MEMBER:
                        parameter = attachment.target_parameters[0].start
                        target_point = self.member_point(
                            attachment.target_id, parameter
                        )
                        trim_valid = True
                        extent = feature_extent((source_point, target_point))
                    else:
                        self.issue(
                            AuditCode.UNSUPPORTED_CANDIDATE,
                            AuditSeverity.BLOCKER,
                            "vertex-to-sheet attachment has no unambiguous face parameterization",
                            keys=(
                                attachment_key,
                                attachment.source_key,
                                attachment.target_key,
                            ),
                            classification=IntersectionKind.UNSUPPORTED,
                            classification_confidence=0.0,
                            evidence_quality="unverified",
                            recommended_action="record the qualified target face for this attachment",
                            blocks_strict_handoff=True,
                        )
                        self.collector.record_classification(classified=False)
                        continue
                    tolerance = self.tolerance.effective_surface_residual(extent)
                    distance = float(np.linalg.norm(source_point - target_point))
                    if distance > tolerance or not trim_valid:
                        self.issue(
                            AuditCode.ATTACHMENT_INCONSISTENT,
                            AuditSeverity.ERROR,
                            "declared vertex attachment does not match target geometry",
                            keys=(
                                attachment_key,
                                attachment.source_key,
                                attachment.target_key,
                            ),
                            witnesses=(
                                _as_witness("attachment/source", source_point),
                                _as_witness("attachment/target", target_point),
                            ),
                            classification="inconsistent",
                            measured_gap=distance,
                            tolerance_used=tolerance,
                            recommended_action="update or remove the stale attachment",
                            details={"target_trim_contains_point": trim_valid},
                        )
                    self.collector.record_classification(classified=True)
                    continue

                if attachment.member_id is None:
                    raise GeometryError("attachment source is not a member or vertex")
                member = self.model.members[attachment.member_id]
                uses = tuple(
                    self.model.member_edge_uses[use_id]
                    for use_id in member.edge_use_ids
                )
                point_valued = attachment.member_range.is_point
                exactly_supported = point_valued or all(
                    isinstance(self.model.edges[use.edge_id].curve, Straight)
                    for use in uses
                    if not (
                        use.parent_range.end < attachment.member_range.start
                        or use.parent_range.start > attachment.member_range.end
                    )
                )
                if attachment.target_kind is AttachmentTargetKind.FACE:
                    face = self.model.faces[attachment.target_id]
                    exactly_supported = exactly_supported and (
                        point_valued or isinstance(face.surface, Plane)
                    )
                elif attachment.target_kind is AttachmentTargetKind.EDGE:
                    target_edge = self.model.edges[attachment.target_id]
                    exactly_supported = exactly_supported and (
                        point_valued or isinstance(target_edge.curve, Straight)
                    )
                elif attachment.target_kind is AttachmentTargetKind.MEMBER:
                    target_member = self.model.members[attachment.target_id]
                    target_uses = tuple(
                        self.model.member_edge_uses[use_id]
                        for use_id in target_member.edge_use_ids
                    )
                    exactly_supported = exactly_supported and (
                        point_valued
                        or all(
                            isinstance(self.model.edges[use.edge_id].curve, Straight)
                            for use in target_uses
                        )
                    )
                elif attachment.target_kind is AttachmentTargetKind.VERTEX:
                    exactly_supported = point_valued
                else:
                    self.issue(
                        AuditCode.UNSUPPORTED_CANDIDATE,
                        AuditSeverity.BLOCKER,
                        "sheet attachment has no unambiguous target face parameterization",
                        keys=(
                            attachment_key,
                            attachment.source_key,
                            attachment.target_key,
                        ),
                        classification=IntersectionKind.UNSUPPORTED,
                        classification_confidence=0.0,
                        evidence_quality="unverified",
                        recommended_action="record the qualified target face fragments",
                        blocks_strict_handoff=True,
                    )
                    self.collector.record_classification(classified=False)
                    continue
                if not exactly_supported:
                    self.issue(
                        AuditCode.UNCLASSIFIED_CANDIDATE,
                        AuditSeverity.BLOCKER,
                        "an attachment uses a curve/surface combination without an exact qualifier",
                        keys=(
                            attachment_key,
                            ("member", attachment.member_id),
                            attachment.target_key,
                        ),
                        classification=IntersectionKind.UNCLASSIFIED,
                        details={"reason": "unsupported_attachment_geometry"},
                    )
                    self.collector.record_classification(classified=False)
                    continue

                lower = attachment.member_range.start
                upper = attachment.member_range.end
                parameters = (
                    (lower,)
                    if point_valued
                    else self._member_breakpoints(
                        attachment.member_id, lower, upper
                    )
                )
                member_points: list[np.ndarray] = []
                target_points: list[np.ndarray] = []
                trim_valid = True
                span = upper - lower
                for parameter in parameters:
                    fraction = 0.0 if point_valued else (parameter - lower) / span
                    member_points.append(
                        self.member_point(attachment.member_id, parameter)
                    )
                    if attachment.target_kind is AttachmentTargetKind.FACE:
                        u_range, v_range = attachment.target_parameters
                        u = u_range.start + fraction * (
                            u_range.end - u_range.start
                        )
                        v = v_range.start + fraction * (
                            v_range.end - v_range.start
                        )
                        trim_valid = trim_valid and self.face_support_contains_uv(
                            attachment.target_id, (u, v)
                        )
                        target_points.append(
                            self.model.face_support_point(
                                attachment.target_id, u, v
                            )
                        )
                    elif attachment.target_kind is AttachmentTargetKind.EDGE:
                        edge_range = attachment.target_parameters[0]
                        edge_parameter = edge_range.start + fraction * (
                            edge_range.end - edge_range.start
                        )
                        target_points.append(
                            self.model.sample_edge(
                                attachment.target_id,
                                np.asarray((edge_parameter,)),
                            )[0]
                        )
                    elif attachment.target_kind is AttachmentTargetKind.MEMBER:
                        target_range = attachment.target_parameters[0]
                        target_parameter = target_range.start + fraction * (
                            target_range.end - target_range.start
                        )
                        target_points.append(
                            self.member_point(attachment.target_id, target_parameter)
                        )
                    else:
                        target_points.append(
                            self.model.vertex_position(attachment.target_id)
                        )

                if (
                    attachment.target_kind is AttachmentTargetKind.FACE
                    and not point_valued
                ):
                    assert isinstance(face.surface, Plane)
                    try:
                        from shapely.geometry import LineString
                    except ImportError as error:  # pragma: no cover
                        raise GeometryError(
                            "planar strict audit requires shapely"
                        ) from error
                    u_range, v_range = attachment.target_parameters
                    target_path = LineString(
                        (
                            (u_range.start, v_range.start),
                            (u_range.end, v_range.end),
                        )
                    )
                    polygon = self._face_polygon(attachment.target_id, face.surface)
                    attachment_extent = feature_extent(
                        self.face_points(attachment.target_id)
                    )
                    trim_valid = trim_valid and polygon.buffer(
                        self._plane_uv_length_tolerance(
                            face.surface, attachment_extent
                        )
                    ).covers(target_path)

                local_extent = max(
                    feature_extent(np.vstack(member_points)),
                    feature_extent(np.vstack(target_points)),
                )
                tolerance = self.tolerance.effective_surface_residual(local_extent)
                distances = tuple(
                    float(np.linalg.norm(member_point - target_point))
                    for member_point, target_point in zip(
                        member_points, target_points
                    )
                )
                maximum = max(distances, default=float("inf"))
                if maximum > tolerance or not trim_valid:
                    self.issue(
                        AuditCode.ATTACHMENT_INCONSISTENT,
                        AuditSeverity.ERROR,
                        "declared attachment parameters do not match member and target geometry",
                        keys=(
                            attachment_key,
                            ("member", attachment.member_id),
                            attachment.target_key,
                        ),
                        witnesses=(
                            _as_witness(
                                "attachment/member",
                                member_points[int(np.argmax(distances))],
                            ),
                            _as_witness(
                                "attachment/target",
                                target_points[int(np.argmax(distances))],
                            ),
                        ),
                        classification="inconsistent",
                        measured_gap=maximum,
                        tolerance_used=tolerance,
                        recommended_action="update or remove the stale attachment",
                        details={
                            "maximum_distance": maximum,
                            "target_trim_contains_path": trim_valid,
                        },
                    )
                self.collector.record_classification(classified=True)
            except Exception as error:
                self.issue(
                    AuditCode.UNCLASSIFIED_CANDIDATE,
                    AuditSeverity.BLOCKER,
                    "a declared attachment could not be geometrically qualified",
                    keys=(attachment_key,),
                    classification=IntersectionKind.UNCLASSIFIED,
                    details={"reason": type(error).__qualname__},
                )
                self.collector.record_classification(classified=False)

    def audit_junction_geometry(
        self, junction_ids: Iterable[int] | None = None
    ) -> None:
        selected = (
            None if junction_ids is None else frozenset(int(item) for item in junction_ids)
        )
        junctions = tuple(
            (identifier, junction)
            for identifier, junction in sorted(self.model.junctions.items())
            if selected is None or identifier in selected
        )
        self.collector.record_candidates(len(junctions))
        for junction_id, junction in junctions:
            self.collector.record_narrow_phase()
            try:
                positive_ranges = tuple(
                    use for use in junction.member_uses if not use.member_range.is_point
                )
                # A straight member is affine between MemberEdgeUse
                # breakpoints.  Comparing every endpoint in the union of both
                # parameterizations proves an overlap exactly; start/mid/end
                # sampling can miss a detour that returns to the shared axis.
                unsupported = False
                fractions = {0.0} if not positive_ranges else {0.0, 1.0}
                for junction_use in positive_ranges:
                    parameter_range = junction_use.member_range
                    span = parameter_range.end - parameter_range.start
                    member = self.model.members[junction_use.member_id]
                    for edge_use_id in member.edge_use_ids:
                        edge_use = self.model.member_edge_uses[edge_use_id]
                        lower = max(
                            parameter_range.start, edge_use.parent_range.start
                        )
                        upper = min(
                            parameter_range.end, edge_use.parent_range.end
                        )
                        if upper < lower - self.tolerance.parameter:
                            continue
                        if not isinstance(
                            self.model.edges[edge_use.edge_id].curve, Straight
                        ):
                            unsupported = True
                            break
                        for value in (lower, upper):
                            fraction = min(
                                max(
                                    (value - parameter_range.start) / span,
                                    0.0,
                                ),
                                1.0,
                            )
                            fractions.add(float(fraction))
                            # The same set must partition a candidate evaluated
                            # in reverse orientation as well.
                            fractions.add(float(1.0 - fraction))
                    if unsupported:
                        break
                if unsupported:
                    self.issue(
                        AuditCode.UNCLASSIFIED_CANDIDATE,
                        AuditSeverity.BLOCKER,
                        "a positive-length junction uses a curved member without an exact overlap qualifier",
                        keys=(
                            ("junction", junction_id),
                            *(
                                ("member", use.member_id)
                                for use in junction.member_uses
                            ),
                        ),
                        classification=IntersectionKind.UNCLASSIFIED,
                        details={"reason": "unsupported_curved_junction_overlap"},
                    )
                    self.collector.record_classification(classified=False)
                    continue

                normalized = tuple(sorted(fractions))
                samples: list[list[np.ndarray]] = []
                for use in junction.member_uses:
                    parameter_range = use.member_range
                    span = parameter_range.end - parameter_range.start
                    parameters = (
                        (parameter_range.start,)
                        if parameter_range.is_point
                        else tuple(
                            parameter_range.start + fraction * span
                            for fraction in normalized
                        )
                    )
                    samples.append(
                        [
                            self.member_point(use.member_id, parameter)
                            for parameter in parameters
                        ]
                    )
                inconsistent = False
                maximum = 0.0
                if len(samples) >= 2:
                    reference = samples[0]
                    for candidate in samples[1:]:
                        if len(candidate) != len(reference):
                            inconsistent = True
                            continue
                        # The tolerance scale is the participating members'
                        # own path extent, never the distance between already
                        # inconsistent candidates.  Otherwise two long-range
                        # but locally short paths can make their own mismatch
                        # tolerance arbitrarily large.
                        reference_extent = feature_extent(np.vstack(reference))
                        candidate_extent = feature_extent(np.vstack(candidate))
                        junction_tolerance = self.tolerance.effective_length(
                            max(reference_extent, candidate_extent)
                        )
                        direct = max(
                            float(np.linalg.norm(left - right))
                            for left, right in zip(reference, candidate)
                        )
                        reverse = max(
                            float(np.linalg.norm(left - right))
                            for left, right in zip(reference, reversed(candidate))
                        )
                        distance = min(direct, reverse)
                        maximum = max(maximum, distance)
                        inconsistent = inconsistent or distance > junction_tolerance
                if inconsistent:
                    self.issue(
                        AuditCode.JUNCTION_INCONSISTENT,
                        AuditSeverity.ERROR,
                        "declared junction parameters do not share a geometric witness",
                        keys=(
                            ("junction", junction_id),
                            *(("member", use.member_id) for use in junction.member_uses),
                        ),
                        classification="inconsistent",
                        details={"maximum_sample_distance": maximum},
                    )
                self.collector.record_classification(classified=True)
            except Exception as error:
                self.issue(
                    AuditCode.UNCLASSIFIED_CANDIDATE,
                    AuditSeverity.BLOCKER,
                    "a declared junction could not be geometrically qualified",
                    keys=(("junction", junction_id),),
                    classification=IntersectionKind.UNCLASSIFIED,
                    details={"reason": type(error).__qualname__},
                )
                self.collector.record_classification(classified=False)

    def run(self) -> None:
        self.audit_topology()
        self.audit_spatial_pairs()
        self.audit_same_edge_members()
        self.audit_attachment_geometry()
        self.audit_junction_geometry()

    def run_changed(self, change_set: ChangeSet) -> None:
        if change_set.revision_after != self.model.revision:
            raise GeometryError(
                "changed-region audit requires the ChangeSet for the current model revision"
            )
        if change_set.revision_before > change_set.revision_after:
            raise GeometryError("ChangeSet revisions are not monotonic")

        self.audit_changed_structural(change_set)
        active = self.audit_changed_spatial_pairs(change_set)
        edge_ids = {identifier for kind, identifier in active if kind == "edge"}
        face_ids = {identifier for kind, identifier in active if kind == "face"}
        vertex_ids = {identifier for kind, identifier in active if kind == "vertex"}
        member_ids = {
            member_id
            for edge_id in edge_ids
            for member_id in self.members_for_edge(edge_id)
        }
        for kind, identifier in change_set.member_changes:
            if kind == "member" and identifier in self.model.members:
                member_ids.add(identifier)
            elif kind == "member_edge_use":
                use = self.model.member_edge_uses.get(identifier)
                if use is not None:
                    member_ids.add(use.member_id)
        sheet_ids = {
            sheet_id
            for face_id in face_ids
            for sheet_id in self.sheets_for_face(face_id)
        }
        for kind, identifier in change_set.ownership_changes:
            if kind == "sheet" and identifier in self.model.sheets:
                sheet_ids.add(identifier)
            elif kind == "face_use":
                use = self.model.face_uses.get(identifier)
                if use is not None:
                    sheet_ids.add(use.sheet_id)
            elif kind == "coedge":
                coedge = self.model.coedges.get(identifier)
                if coedge is not None:
                    use = self.model.face_uses.get(coedge.face_use_id)
                    if use is not None:
                        sheet_ids.add(use.sheet_id)

        self.audit_orphan_control_geometry(vertex_ids)
        self.audit_non_manifold_edges(edge_ids)
        self.audit_same_edge_members(edge_ids)

        attachment_ids = {
            identifier
            for kind, identifier in change_set.attachment_changes
            if kind == "attachment" and identifier in self.model.attachments
        }
        source_getter = getattr(self.model, "attachments_for_source", None)
        target_index = getattr(self.model, "_target_attachments", {})
        for key in sorted(active):
            attachment_ids.update(int(item) for item in target_index.get(key, ()))
            if callable(source_getter):
                attachment_ids.update(
                    int(item) for item in source_getter(key[0], key[1])
                )
        for member_id in sorted(member_ids):
            getter = getattr(self.model, "attachments_for_member", None)
            if callable(getter):
                attachment_ids.update(int(item) for item in getter(member_id))
        for face_id in sorted(face_ids):
            getter = getattr(self.model, "attachments_for_face", None)
            if callable(getter):
                attachment_ids.update(int(item) for item in getter(face_id))
        for sheet_id in sorted(sheet_ids):
            getter = getattr(self.model, "attachments_for_sheet", None)
            if callable(getter):
                attachment_ids.update(int(item) for item in getter(sheet_id))
        self.audit_attachment_geometry(attachment_ids)

        junction_ids = {
            identifier
            for kind, identifier in (
                *change_set.ownership_changes,
                *change_set.attachment_changes,
            )
            if kind == "junction" and identifier in self.model.junctions
        }
        junction_getter = getattr(self.model, "junctions_for_member", None)
        for member_id in sorted(member_ids):
            if callable(junction_getter):
                junction_ids.update(int(item) for item in junction_getter(member_id))
            else:
                junction_ids.update(
                    int(item)
                    for item in getattr(self.model, "_member_junctions", {}).get(
                        member_id, ()
                    )
                )
        structural_keys: set[SpatialKey] = {
            *(("member", identifier) for identifier in member_ids),
            *(("sheet", identifier) for identifier in sheet_ids),
            *(("attachment", identifier) for identifier in attachment_ids),
            *(("junction", identifier) for identifier in junction_ids),
        }
        for member_id in member_ids:
            member = self.model.members.get(member_id)
            if member is not None:
                structural_keys.add(("part", member.part_id))
                structural_keys.update(
                    ("member_edge_use", use_id) for use_id in member.edge_use_ids
                )
        for sheet_id in sheet_ids:
            sheet = self.model.sheets.get(sheet_id)
            if sheet is not None:
                structural_keys.add(("part", sheet.part_id))
                structural_keys.update(
                    ("face_use", use_id) for use_id in sheet.face_use_ids
                )
                for use_id in sheet.face_use_ids:
                    use = self.model.face_uses.get(use_id)
                    if use is not None:
                        structural_keys.update(
                            ("coedge", coedge_id)
                            for loop in use.loops
                            for coedge_id in loop
                        )
        self.collector.record_affected_structural_keys(len(structural_keys))
        self.audit_junction_geometry(junction_ids)

        # A removed structural record has no old parent closure in ChangeSet.
        # If no geometry bound changed with it, the current API cannot localize
        # the relation.  Report that limitation explicitly instead of claiming
        # a false-clean changed-region qualification.
        semantic_keys = {
            *change_set.member_changes,
            *change_set.attachment_changes,
            *change_set.ownership_changes,
        }
        unresolved_removed = tuple(
            sorted(
                key
                for key in semantic_keys
                if key[0]
                in {
                    "part",
                    "sheet",
                    "face_use",
                    "coedge",
                    "member",
                    "member_edge_use",
                    "attachment",
                    "junction",
                }
                and not self.model._contains_entity(*key)  # noqa: SLF001
            )
        )
        if unresolved_removed:
            self.issue(
                AuditCode.CAPABILITY_MISSING,
                AuditSeverity.BLOCKER,
                "removed structural relationships lack old geometric dependency bounds",
                keys=unresolved_removed,
                classification=IntersectionKind.CAPABILITY_MISSING,
                classification_confidence=0.0,
                evidence_quality="unverified",
                recommended_action=(
                    "run a full strict audit or supply structural dependency bounds in ChangeSet"
                ),
                blocks_strict_handoff=True,
                details={"removed_keys": unresolved_removed},
            )

    def audit_changed_structural(self, change_set: ChangeSet) -> None:
        """Validate active structural records in the semantic delta closure."""

        errors: list[tuple[AuditCode, str, tuple[SpatialKey, ...]]] = []

        def error(
            code: AuditCode, message: str, *keys: SpatialKey
        ) -> None:
            errors.append((code, message, tuple(keys)))

        member_ids: set[int] = set()
        for kind, identifier in change_set.member_changes:
            if kind == "member" and identifier in self.model.members:
                member_ids.add(identifier)
            elif kind == "member_edge_use":
                use = self.model.member_edge_uses.get(identifier)
                if use is not None:
                    member_ids.add(use.member_id)
        for member_id in sorted(member_ids):
            member = self.model.members[member_id]
            part = self.model.parts.get(member.part_id)
            if part is None or member_id not in part.member_ids:
                error(
                    AuditCode.UNOWNED_STRUCTURAL_USE,
                    f"member {member_id} is not owned by part {member.part_id}",
                    ("member", member_id),
                    ("part", member.part_id),
                )
            previous_end: int | None = None
            previous_parameter: float | None = None
            for use_id in member.edge_use_ids:
                use = self.model.member_edge_uses.get(use_id)
                if use is None or use.member_id != member_id:
                    error(
                        AuditCode.UNOWNED_STRUCTURAL_USE,
                        f"member {member_id} has an invalid axis use {use_id}",
                        ("member", member_id),
                        ("member_edge_use", use_id),
                    )
                    continue
                edge = self.model.edges.get(use.edge_id)
                if edge is None:
                    error(
                        AuditCode.UNOWNED_STRUCTURAL_USE,
                        f"member-edge use {use_id} references missing edge {use.edge_id}",
                        ("member_edge_use", use_id),
                        ("edge", use.edge_id),
                    )
                    continue
                start, end = (
                    (edge.start, edge.end)
                    if use.orientation is Orientation.FORWARD
                    else (edge.end, edge.start)
                )
                if previous_end is not None and previous_end != start:
                    error(
                        AuditCode.NONCONFORMAL_INTERFACE,
                        f"member {member_id} axis is discontinuous at use {use_id}",
                        ("member", member_id),
                        ("member_edge_use", use_id),
                    )
                if (
                    previous_parameter is not None
                    and abs(use.parent_range.start - previous_parameter)
                    > self.tolerance.parameter
                ):
                    error(
                        AuditCode.NONCONFORMAL_INTERFACE,
                        f"member {member_id} parent ranges are discontinuous at use {use_id}",
                        ("member", member_id),
                        ("member_edge_use", use_id),
                    )
                previous_end = end
                previous_parameter = use.parent_range.end

        sheet_ids: set[int] = set()
        for kind, identifier in change_set.ownership_changes:
            if kind == "sheet" and identifier in self.model.sheets:
                sheet_ids.add(identifier)
            elif kind == "face_use":
                use = self.model.face_uses.get(identifier)
                if use is not None:
                    sheet_ids.add(use.sheet_id)
            elif kind == "coedge":
                coedge = self.model.coedges.get(identifier)
                if coedge is not None:
                    use = self.model.face_uses.get(coedge.face_use_id)
                    if use is not None:
                        sheet_ids.add(use.sheet_id)
        for sheet_id in sorted(sheet_ids):
            sheet = self.model.sheets[sheet_id]
            part = self.model.parts.get(sheet.part_id)
            if part is None or sheet_id not in part.sheet_ids:
                error(
                    AuditCode.UNOWNED_STRUCTURAL_USE,
                    f"sheet {sheet_id} is not owned by part {sheet.part_id}",
                    ("sheet", sheet_id),
                    ("part", sheet.part_id),
                )
            for use_id in sheet.face_use_ids:
                use = self.model.face_uses.get(use_id)
                if (
                    use is None
                    or use.sheet_id != sheet_id
                    or use.face_id not in self.model.faces
                ):
                    error(
                        AuditCode.UNOWNED_STRUCTURAL_USE,
                        f"sheet {sheet_id} has an invalid face use {use_id}",
                        ("sheet", sheet_id),
                        ("face_use", use_id),
                    )
                    continue
                for coedge_id in use.coedge_ids:
                    coedge = self.model.coedges.get(coedge_id)
                    if (
                        coedge is None
                        or coedge.face_use_id != use_id
                        or coedge.edge_id not in self.model.edges
                    ):
                        error(
                            AuditCode.UNOWNED_STRUCTURAL_USE,
                            f"face use {use_id} has an invalid coedge {coedge_id}",
                            ("face_use", use_id),
                            ("coedge", coedge_id),
                        )

        attachment_ids = {
            identifier
            for kind, identifier in change_set.attachment_changes
            if kind == "attachment" and identifier in self.model.attachments
        }
        for attachment_id in sorted(attachment_ids):
            attachment = self.model.attachments[attachment_id]
            if not self.model._contains_entity(*attachment.source_key):  # noqa: SLF001
                error(
                    AuditCode.ATTACHMENT_INCONSISTENT,
                    f"attachment {attachment_id} references a missing source",
                    ("attachment", attachment_id),
                    attachment.source_key,
                )
            if not self.model._contains_entity(*attachment.target_key):  # noqa: SLF001
                error(
                    AuditCode.ATTACHMENT_INCONSISTENT,
                    f"attachment {attachment_id} references a missing target",
                    ("attachment", attachment_id),
                    attachment.target_key,
                )

        junction_ids = {
            identifier
            for kind, identifier in (
                *change_set.ownership_changes,
                *change_set.attachment_changes,
            )
            if kind == "junction" and identifier in self.model.junctions
        }
        for junction_id in sorted(junction_ids):
            junction = self.model.junctions[junction_id]
            for use in junction.member_uses:
                if use.member_id not in self.model.members:
                    error(
                        AuditCode.JUNCTION_INCONSISTENT,
                        f"junction {junction_id} references missing member {use.member_id}",
                        ("junction", junction_id),
                        ("member", use.member_id),
                    )
            for attachment_id in junction.attachment_ids:
                if attachment_id not in self.model.attachments:
                    error(
                        AuditCode.JUNCTION_INCONSISTENT,
                        f"junction {junction_id} references missing attachment {attachment_id}",
                        ("junction", junction_id),
                        ("attachment", attachment_id),
                    )

        for code, message, keys in sorted(
            errors, key=lambda item: (item[0].value, item[2], item[1])
        ):
            self.issue(
                code,
                AuditSeverity.ERROR,
                message,
                keys=keys,
                classification="invalid_structural_topology",
                recommended_action="repair the changed structural ownership or incidence",
                blocks_strict_handoff=True,
            )


def _strict_check(
    model: "GeometryModel",
    context: AuditContext,
    collector: AuditCollector,
) -> None:
    _StrictAuditState(model, context, collector).run()


def _changed_check(change_set: ChangeSet):
    def check(
        model: "GeometryModel",
        context: AuditContext,
        collector: AuditCollector,
    ) -> None:
        _StrictAuditState(model, context, collector).run_changed(change_set)

    check.audit_name = "changed_region"  # type: ignore[attr-defined]
    return check


def _strict_check_with_qualification(
    qualification: IntersectionQualificationPolicy,
):
    def check(
        model: "GeometryModel",
        context: AuditContext,
        collector: AuditCollector,
    ) -> None:
        _StrictAuditState(
            model, context, collector, qualification=qualification
        ).run()

    check.audit_name = "strict_geometry"  # type: ignore[attr-defined]
    return check


def _changed_check_with_qualification(
    change_set: ChangeSet,
    qualification: IntersectionQualificationPolicy,
):
    def check(
        model: "GeometryModel",
        context: AuditContext,
        collector: AuditCollector,
    ) -> None:
        _StrictAuditState(
            model, context, collector, qualification=qualification
        ).run_changed(change_set)

    check.audit_name = "changed_region"  # type: ignore[attr-defined]
    return check


def strict_audit(
    model: "GeometryModel",
    *,
    policy: AuditPolicy | None = None,
    qualification: IntersectionQualificationPolicy | None = None,
) -> AuditReport:
    """Return a full-model strict qualification report.

    The model is never edited.  A fresh AABB tree avoids trusting a stale
    model cache while the maintained index, when already materialized, is
    independently checked for key/invariant consistency.  Any checker error
    is caught by :func:`run_audit` and becomes a stable fail-closed blocker.
    """

    qualified_policy = (
        DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        if qualification is None
        else qualification
    )
    if not isinstance(qualified_policy, IntersectionQualificationPolicy):
        raise TypeError("qualification must be IntersectionQualificationPolicy")
    return run_audit(
        model,
        (_strict_check_with_qualification(qualified_policy),),
        model_id_getter=lambda source: source.model_id,
        revision_getter=lambda source: source.revision,
        scope=AuditScope.FULL_MODEL,
        policy=policy,
    )


def audit_changed_region(
    model: "GeometryModel",
    change_set: ChangeSet,
    policy: AuditPolicy | None = None,
    *,
    qualification: IntersectionQualificationPolicy | None = None,
) -> AuditReport:
    """Qualify only the committed region touched by ``change_set``.

    The maintained spatial index supplies current neighbours while both old
    and new affected AABBs localize moves/removals.  The result is always
    labelled :attr:`AuditScope.CHANGED_REGION` and can never certify the full
    model, even when it is locally clean.
    """

    if not isinstance(change_set, ChangeSet):
        raise TypeError("change_set must be a ChangeSet")
    changed = {
        *change_set.changed,
        *change_set.member_changes,
        *change_set.attachment_changes,
        *change_set.ownership_changes,
    }
    entities = tuple(
        AuditEntity.from_key(model.model_id, key) for key in sorted(changed)
    )
    qualified_policy = (
        DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        if qualification is None
        else qualification
    )
    if not isinstance(qualified_policy, IntersectionQualificationPolicy):
        raise TypeError("qualification must be IntersectionQualificationPolicy")
    return run_audit(
        model,
        (_changed_check_with_qualification(change_set, qualified_policy),),
        model_id_getter=lambda source: source.model_id,
        revision_getter=lambda source: source.revision,
        scope=AuditScope.CHANGED_REGION,
        policy=policy,
        changed_entities=entities,
        index_updates=len(change_set.spatial_updates),
    )
