"""Fixed reference snapshots of existing model geometry inside a sketch.

A reference is an immutable snapshot of the points (and display curve) in which
existing geometry meets the sketch plane. The snapshot is persisted with the
sketch and becomes hard solver anchors in a bound, solve-time definition. A
reference never follows its source and never resolves the model while solving
or regenerating.

The optional :class:`ReferenceLink` is separate metadata: the identity of the
source and an exact, materialized-ID-independent signature of it at capture
time. When the source moves, disappears or becomes ambiguous the link is
dropped (:func:`surviving_reference_links`); the snapshot stays where it was.
Links are not feature inputs and declare no dependency.

Supported sources: vertices lying in the plane; edges crossing the plane (every
owner root, including curved edges accepted by ``arrangement_geometry``); edges
lying in the plane (start/end points plus a display curve); flat, straight-trimmed
faces (exact straight section segments via the owner's plane/plane and
face-clipping queries, every qualified interval; a coplanar face contributes its
straight boundary loops). Curved or non-flat faces and curved trims are refused
explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .entities import EntityRef
from .errors import GeometryError
from .tolerance import feature_extent

if TYPE_CHECKING:  # pragma: no cover
    from .features import FeatureInputRef
    from .model import GeometryModel
    from .sketch import SketchDefinition, SketchPlane

__all__ = [
    "REFERENCE_PREFIX", "ReferenceLink", "SketchReference", "SketchReferenceError",
    "bind_references", "capture_reference_link", "capture_sketch_reference", "is_reference_key",
    "reference_links_from_parameters", "reference_links_to_parameters",
    "reference_source_signature", "resolve_reference_source", "surviving_reference_links",
]

REFERENCE_PREFIX = "ref:"
_KINDS = ("vertex", "crossing", "inplane", "section")
_DISPLAY_SAMPLES = 48


class SketchReferenceError(GeometryError):
    """A reference cannot be captured; ``reason`` is machine-readable."""

    def __init__(self, reason: str, message: str, reference_id: str | None = None):
        super().__init__(message)
        self.reason = reason  # missing | ambiguous | no_intersection | unsupported | invalid
        self.reference_id = reference_id


def is_reference_key(key: str) -> bool:
    return str(key).startswith(REFERENCE_PREFIX)


def _pair(value: Sequence[float], label: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise GeometryError(f"{label} needs two finite plane coordinates")
    return float(values[0]), float(values[1])


@dataclass(frozen=True)
class SketchReference:
    """Persisted fixed snapshot: plane coordinates of every reference point."""

    id: str
    kind: str
    count: int = 1
    points: tuple = ()
    curve: tuple = ()

    def __post_init__(self) -> None:
        text = str(self.id).strip()
        if not text or ":" in text or "/" in text:
            raise GeometryError("a sketch reference needs a plain identifier")
        if self.kind not in _KINDS:
            raise GeometryError(f"unknown sketch reference kind {self.kind!r}")
        points = tuple(_pair(item, f"reference {text!r} point") for item in self.points)
        if self.kind == "section":
            # Straight segments of a planar face section: endpoint pairs (2k, 2k+1).
            count = len(points)
            if count < 2 or count % 2:
                raise GeometryError(f"section reference {text!r} needs whole segments (point pairs)")
        else:
            count = 1 if self.kind == "vertex" else (2 if self.kind == "inplane" else int(self.count))
        if count < 1:
            raise GeometryError("a crossing reference needs at least one point")
        if len(points) != count:
            raise GeometryError(f"reference {text!r} needs a snapshot of {count} point(s)")
        curve = tuple(_pair(item, f"reference {text!r} curve point") for item in self.curve)
        if curve and self.kind != "inplane":
            raise GeometryError("only an in-plane edge reference has a display curve")
        object.__setattr__(self, "id", text)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "curve", curve)

    @property
    def keys(self) -> tuple[str, ...]:
        base = f"{REFERENCE_PREFIX}{self.id}"
        if self.kind == "vertex":
            return (base,)
        if self.kind == "inplane":
            return (f"{base}:start", f"{base}:end")
        return tuple(f"{base}:{index}" for index in range(self.count))

    @property
    def source_kind(self) -> str:
        return {"vertex": "vertex", "section": "face"}.get(self.kind, "edge")

    @property
    def segments(self) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
        """Straight display/constraint segments of a face-section reference."""

        if self.kind != "section":
            return ()
        return tuple((self.points[i], self.points[i + 1]) for i in range(0, len(self.points), 2))

    def anchors(self) -> dict[str, tuple[float, float]]:
        return dict(zip(self.keys, self.points))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id, "kind": self.kind, "count": self.count,
            "points": [list(item) for item in self.points],
        }
        if self.curve:
            result["curve"] = [list(item) for item in self.curve]
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | "SketchReference") -> "SketchReference":
        if isinstance(data, SketchReference):
            return data
        return cls(
            str(data.get("id", "")), str(data.get("kind", "")), int(data.get("count", 1)),
            tuple(tuple(item) for item in data.get("points", ())),  # type: ignore[arg-type]
            tuple(tuple(item) for item in data.get("curve", ())),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ReferenceLink:
    """Optional source identity and capture signature of one reference."""

    reference_id: str
    source: "FeatureInputRef"
    signature: str

    def to_dict(self) -> dict[str, object]:
        from .features import FeatureOutputRef

        if isinstance(self.source, FeatureOutputRef):
            source: dict[str, object] = {
                "feature_id": self.source.feature_id, "output_key": self.source.output_key,
                "kind": self.source.kind}
        else:
            source = {"kind": self.source.kind, "id": self.source.id}
        return {"reference": self.reference_id, "source": source, "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReferenceLink":
        from .features import FeatureOutputRef

        raw = data.get("source")
        if not isinstance(raw, Mapping):
            raise GeometryError("a reference link needs a source")
        if "feature_id" in raw:
            source: "FeatureInputRef" = FeatureOutputRef(
                int(raw["feature_id"]), str(raw["output_key"]), str(raw["kind"]))  # type: ignore[arg-type]
        else:
            source = EntityRef(str(raw["kind"]), int(raw["id"]))  # type: ignore[arg-type]
        return cls(str(data.get("reference", "")), source, str(data.get("signature", "")))


def reference_links_to_parameters(links: Sequence[ReferenceLink]) -> list[dict[str, object]]:
    return [item.to_dict() for item in links]


def reference_links_from_parameters(parameters: Mapping[str, object]) -> tuple[ReferenceLink, ...]:
    """Read the optional ``reference_links`` metadata; absent means no links."""

    return tuple(ReferenceLink.from_dict(item) for item in parameters.get("reference_links", ()))  # type: ignore[union-attr]


# ---------------------------------------------------------------- capture


def _local(plane: "SketchPlane", point: np.ndarray) -> tuple[float, float]:
    value = plane.local(point)
    return float(value[0]), float(value[1])


def _vertex_position(geometry: "GeometryModel", source: EntityRef, reference_id: str) -> np.ndarray:
    try:
        return np.asarray(geometry.vertex_position(source.id), dtype=float)
    except (KeyError, IndexError, GeometryError):
        raise SketchReferenceError(
            "missing", f"reference {reference_id!r}: source vertex {source.id} no longer exists",
            reference_id) from None


def _edge_curve(geometry: "GeometryModel", source: EntityRef, reference_id: str):
    from .arrangement_geometry import freeze_edge

    if source.id not in geometry.edges:
        raise SketchReferenceError(
            "missing", f"reference {reference_id!r}: source edge {source.id} no longer exists",
            reference_id)
    try:
        return freeze_edge(geometry, source.id)
    except GeometryError as error:
        raise SketchReferenceError(
            "unsupported", f"reference {reference_id!r}: edge {source.id} has an unsupported curve ({error})",
            reference_id) from None


def _roots(curve, plane: "SketchPlane", reference_id: str):
    """Plane roots in parameter order, ``None`` when the curve lies in the plane.

    Only bit-identical parameters are merged; distinct roots are never merged by
    coordinate proximity.
    """

    from .arrangement_geometry import plane_roots

    normal = np.asarray(plane.normal, dtype=float)
    try:
        roots = plane_roots(curve, normal, float(normal @ np.asarray(plane.origin, dtype=float)))
    except GeometryError as error:
        raise SketchReferenceError(
            "unsupported", f"reference {reference_id!r}: plane intersection is unsupported ({error})",
            reference_id) from None
    if roots is None:
        return None
    return tuple(
        (root, np.asarray(curve.evaluate(root), dtype=float))
        for root in sorted({float(item) for item in roots}))


def capture_sketch_reference(
    geometry: "GeometryModel", plane: "SketchPlane", source: EntityRef, reference_id: str,
) -> SketchReference:
    """Snapshot ``source`` against ``plane`` or refuse explicitly.

    ``source`` must be an active entity (see :func:`resolve_reference_source`).
    """

    if source.kind == "vertex":
        point = _vertex_position(geometry, source, reference_id)
        offset = point - np.asarray(plane.origin, dtype=float)
        # A single vertex has zero local extent: the adopted coincidence policy
        # alone decides, so the plane origin choice cannot change membership.
        if abs(float(np.asarray(plane.normal, dtype=float) @ offset)) > geometry.tolerance.effective_coincidence(0.0):
            raise SketchReferenceError(
                "no_intersection", f"vertex {source.id} does not lie in the sketch plane", reference_id)
        return SketchReference(reference_id, "vertex", 1, (_local(plane, point),))
    if source.kind == "edge":
        curve = _edge_curve(geometry, source, reference_id)
        roots = _roots(curve, plane, reference_id)
        if roots is None:
            ends = tuple(_local(plane, np.asarray(curve.evaluate(t), dtype=float)) for t in (0.0, 1.0))
            display = tuple(
                _local(plane, np.asarray(curve.evaluate(float(t)), dtype=float))
                for t in np.linspace(0.0, 1.0, _DISPLAY_SAMPLES))
            return SketchReference(reference_id, "inplane", 2, ends, display)
        if not roots:
            raise SketchReferenceError(
                "no_intersection", f"edge {source.id} does not meet the sketch plane", reference_id)
        return SketchReference(
            reference_id, "crossing", len(roots), tuple(_local(plane, point) for _t, point in roots))
    if source.kind == "face":
        return _capture_face_section(geometry, plane, source, reference_id)
    raise SketchReferenceError(
        "unsupported", f"a {source.kind} cannot be a sketch reference", reference_id)


def _capture_face_section(
    geometry: "GeometryModel", plane: "SketchPlane", source: EntityRef, reference_id: str,
) -> SketchReference:
    """Straight segments where a flat, straight-trimmed face meets the sketch plane.

    Uses the owner's ``plane_plane`` and ``clip_line_to_face`` (exact region
    clipping, holes and disconnected intervals preserved) and, for a coplanar
    face, its straight boundary loops. Curved supports or trims are refused.
    """

    from .intersections import clip_line_to_face, plane_plane
    from .predicates import IntersectionKind
    from .sketch import face_sketch_plane
    from .surfaces import Plane
    from .curves import Straight

    face = geometry.faces.get(source.id)
    if face is None:
        raise SketchReferenceError(
            "missing", f"reference {reference_id!r}: source face {source.id} no longer exists", reference_id)
    try:
        support = face_sketch_plane(geometry, source.id)
    except (GeometryError, KeyError, IndexError, ValueError) as error:
        raise SketchReferenceError(
            "unsupported",
            f"reference {reference_id!r}: face {source.id} is curved or not flat ({error}); "
            "only flat plates can be sections", reference_id) from None
    if any(not isinstance(geometry.edges[item.edge].curve, Straight)
           for loop in (face.loop,) + tuple(face.holes) for item in loop):
        raise SketchReferenceError(
            "unsupported",
            f"reference {reference_id!r}: face {source.id} has curved trim edges; "
            "curved trims cannot be sections", reference_id)
    sketch_surface = Plane(plane.origin, plane.x_axis, plane.y_axis)
    face_surface = Plane(support.origin, support.x_axis, support.y_axis)
    line = plane_plane(sketch_surface, face_surface)
    if line is None:
        loops = [
            [np.asarray(geometry.vertex_position(geometry.oriented_start_vertex(item)), dtype=float)
             for item in loop]
            for loop in (face.loop,) + tuple(face.holes)]
        normal = np.asarray(plane.normal, dtype=float)
        origin = np.asarray(plane.origin, dtype=float)
        distances = [float(normal @ (start - origin)) for starts in loops for start in starts]
        extent = feature_extent([start for starts in loops for start in starts])
        tolerance = geometry.tolerance.effective_coincidence(extent)
        if any(abs(item) > tolerance for item in distances):
            # plane_plane may decline a small non-zero angle: only a face whose
            # every boundary vertex lies in the plane may be projected.
            if min(distances) > tolerance or max(distances) < -tolerance:
                raise SketchReferenceError(
                    "no_intersection", f"face {source.id} is parallel to and off the sketch plane",
                    reference_id)
            raise SketchReferenceError(
                "unsupported",
                f"reference {reference_id!r}: face {source.id} is nearly parallel to the sketch plane "
                "but not coplanar within tolerance; its section is not supported", reference_id)
        points = []
        for starts in loops:
            for index, start in enumerate(starts):
                points.extend((_local(plane, start), _local(plane, starts[(index + 1) % len(starts)])))
        return SketchReference(reference_id, "section", len(points), tuple(points))
    point, direction = line
    result = clip_line_to_face(geometry, source.id, point, direction)
    components = tuple(getattr(result, "components", ()))
    if result.kind is IntersectionKind.DISJOINT:
        raise SketchReferenceError(
            "no_intersection", f"face {source.id} does not meet the sketch plane", reference_id)
    if result.kind is IntersectionKind.OVERLAP_CURVE and components:
        points = [_local(plane, np.asarray(world, dtype=float))
                  for component in components
                  for world in (component.witnesses[0], component.witnesses[-1])
                  if len(component.witnesses) >= 2]
        if points:
            return SketchReference(reference_id, "section", len(points), tuple(points))
    if result.kind in (IntersectionKind.TOUCH_POINT, IntersectionKind.CROSS) and components:
        points = [_local(plane, np.asarray(world, dtype=float))
                  for component in components for world in component.witnesses[:1]]
        return SketchReference(reference_id, "crossing", len(points), tuple(points))
    raise SketchReferenceError(
        "unsupported",
        f"face {source.id} section is not supported ({result.kind.value}: "
        f"{', '.join(result.diagnostics) or 'no diagnostics'})", reference_id)


def bind_references(definition: "SketchDefinition") -> "SketchDefinition":
    """Return a solve-time definition with stored snapshots as hard anchors.

    No model access: the result carries no ``references`` and is never persisted.
    """

    if not definition.references:
        return definition
    from .sketch import SketchConstraint

    anchors: dict[str, tuple[float, float]] = {}
    for item in definition.references:
        anchors.update(item.anchors())
    fixed = tuple(
        SketchConstraint("fixed", key, position=position, id=f"reference-anchor:{key}")
        for key, position in sorted(anchors.items()))
    return replace(
        definition, points={**definition.points, **anchors},
        constraints=definition.constraints + fixed, references=(),
        bound_reference_keys=frozenset(anchors))


# ---------------------------------------------------------------- source links


def resolve_reference_source(geometry: "GeometryModel", source: "FeatureInputRef") -> EntityRef | None:
    """Return the one active entity ``source`` resolves to by explicit lineage, else ``None``."""

    try:
        resolved = geometry.features.resolve(source, geometry)
    except (KeyError, GeometryError):
        return None
    if len(resolved) != 1 or resolved[0].kind not in ("vertex", "edge", "face"):
        return None
    return resolved[0]


def reference_source_signature(geometry: "GeometryModel", source: "FeatureInputRef") -> str | None:
    """Exact closure signature of a source, independent of materialized IDs."""

    from .features import _exact_entity_payload

    active = resolve_reference_source(geometry, source)
    if active is None:
        return None
    try:
        payload = _exact_entity_payload(active, geometry, {})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, GeometryError, TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def capture_reference_link(
    geometry: "GeometryModel", source: "FeatureInputRef", reference_id: str,
) -> ReferenceLink:
    """Record source identity and signature now; refuses a missing/ambiguous source."""

    signature = reference_source_signature(geometry, source)
    if signature is None:
        raise SketchReferenceError(
            "ambiguous", "the reference source is missing or does not resolve to one vertex/edge",
            reference_id)
    return ReferenceLink(reference_id, source, signature)


def surviving_reference_links(
    geometry: "GeometryModel", links: Sequence[ReferenceLink],
) -> tuple[ReferenceLink, ...]:
    """Links whose source still resolves uniquely to the captured signature.

    Everything else is detached by omission; snapshots are not touched.
    """

    return tuple(
        item for item in links if reference_source_signature(geometry, item.source) == item.signature)
