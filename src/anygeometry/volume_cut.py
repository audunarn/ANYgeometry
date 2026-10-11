"""Cut a box volume out of a shell model for submodelling (slices S2a and S2b).

Implements ``docs/SUBMODEL_VOLUME_CUT_CONTRACT.md`` for shells on planes and
cylinders without attachments or junctions in or across the volume.  Members on
face edges (seam members) are partitioned at the box (section 10):

* :func:`plan_volume_cut` is read-only and bound to the model's id, revision and
  checksum; it selects the faces the box reaches (by conservative extents) and
  refuses unsupported input before anything mutates.
* :func:`apply_volume_cut` works on a detached clone: the six box faces are
  imprinted into the selected faces through the batch intersection workflow
  (``ConnectionIntent.IMPRINT``), every descendant face is classified
  ``INSIDE``/``OUTSIDE`` by a point verified strictly inside its material, and
  the inside faces are extracted as a bounded closure.  The result carries the
  artificial cut edges grouped by box side, the original boundary edges, and the
  lineage from every extracted entity back to the source model; anything without
  a source is listed, never dropped.

Extents are conservative: exact for straight edges and circular arcs, the
control hull for Bezier splines, and the curve's own outward enclosure for the
exact curve families (transformed into the box frame by its corners).  The
extent of a planar or cylindrical face is that of its boundary: on a plane the
face lies in the convex hull of its boundary, and on a cylinder any linear
function of position is linear in the axial coordinate on each generator, so its
extremes over the face lie on the boundary.

The intersection code is used through its public API only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import UUID

import numpy as np

from .batch_intersections import IntersectionBatchPolicy, apply_intersections, plan_intersections
from .closure import ModelClosure, extract_model_closure
from .curves import Arc, Spline, Straight
from .entities import EntityRef
from .errors import GeometryError
from .serialization import to_dict
from .structural import ConnectionIntent
from .surfaces import Cylinder, Plane

__all__ = [
    "BoxVolume",
    "INSIDE",
    "ON",
    "OUTSIDE",
    "VolumeCutError",
    "VolumeCutPlan",
    "VolumeCutResult",
    "apply_volume_cut",
    "plan_volume_cut",
]

INSIDE, OUTSIDE, ON = "inside", "outside", "on"
SIDES = ("x0", "x1", "y0", "y1", "z0", "z1")
_SCAN_SAMPLES = (129, 1025)


class VolumeCutError(GeometryError):
    """A typed, fail-closed volume-cut refusal."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ----------------------------------------------------------------------
# volume
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BoxVolume:
    """A closed box ``lower <= q <= upper`` in its own frame.

    ``transform`` (4x4, proper rigid) places the box frame in the model frame:
    ``x = R q + t``.  Without it the box is axis-aligned in the model frame.
    """

    lower: Tuple[float, float, float]
    upper: Tuple[float, float, float]
    transform: Optional[Tuple[Tuple[float, ...], ...]] = None

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float).reshape(3)
        upper = np.asarray(self.upper, dtype=float).reshape(3)
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(lower < upper)):
            raise VolumeCutError("VOLUME_INVALID", "the box needs finite corners with lower < upper on every axis")
        object.__setattr__(self, "lower", tuple(lower.tolist()))
        object.__setattr__(self, "upper", tuple(upper.tolist()))
        if self.transform is not None:
            matrix = np.asarray(self.transform, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise VolumeCutError("VOLUME_INVALID", "the box transform must be a finite 4x4 matrix")
            rotation = matrix[:3, :3]
            if (np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-12 or abs(np.linalg.det(rotation) - 1.0) > 1e-12
                    or np.max(np.abs(matrix[3] - (0.0, 0.0, 0.0, 1.0))) > 0.0):
                raise VolumeCutError("VOLUME_INVALID", "the box transform must be a proper rigid transform")
            object.__setattr__(self, "transform", tuple(tuple(float(v) for v in row) for row in matrix))

    @property
    def rotation(self) -> np.ndarray:
        return np.eye(3) if self.transform is None else np.asarray(self.transform)[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return np.zeros(3) if self.transform is None else np.asarray(self.transform)[:3, 3]

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(np.subtract(self.upper, self.lower)))

    def local(self, points: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(points) - self.translation) @ self.rotation

    def world(self, local: np.ndarray) -> np.ndarray:
        return np.atleast_2d(local) @ self.rotation.T + self.translation

    def classify(self, point: np.ndarray, tolerance: float) -> str:
        q = self.local(np.asarray(point, dtype=float))[0]
        lower, upper = np.asarray(self.lower), np.asarray(self.upper)
        if np.any(q < lower - tolerance) or np.any(q > upper + tolerance):
            return OUTSIDE
        if np.all(q > lower + tolerance) and np.all(q < upper - tolerance):
            return INSIDE
        return ON

    def sides_of(self, extent: Tuple[np.ndarray, np.ndarray], tolerance: float) -> Tuple[str, ...]:
        """The box sides containing an entity with local extent ``(lo, hi)``."""

        lo, hi = extent
        found = []
        for axis in range(3):
            for side, bound in enumerate((self.lower[axis], self.upper[axis])):
                if abs(lo[axis] - bound) <= tolerance and abs(hi[axis] - bound) <= tolerance:
                    found.append(SIDES[2 * axis + side])
        return tuple(found)

    def contains_extent(self, extent: Tuple[np.ndarray, np.ndarray], tolerance: float) -> bool:
        lo, hi = extent
        return bool(np.all(lo >= np.asarray(self.lower) - tolerance) and np.all(hi <= np.asarray(self.upper) + tolerance))

    def reaches(self, extent: Tuple[np.ndarray, np.ndarray], tolerance: float) -> bool:
        lo, hi = extent
        return bool(np.all(hi >= np.asarray(self.lower) - tolerance) and np.all(lo <= np.asarray(self.upper) + tolerance))

    def to_dict(self) -> dict:
        return {"kind": "box", "lower": list(self.lower), "upper": list(self.upper),
                "transform": None if self.transform is None else [list(row) for row in self.transform]}


# ----------------------------------------------------------------------
# conservative extents in the box frame
# ----------------------------------------------------------------------
def _arc_range(a: float, b: float, sweep: float, start: float = 0.0) -> Tuple[float, float]:
    """Min and max of ``a cos(phi) + b sin(phi)`` for ``phi`` from ``start`` over ``sweep``."""

    end = start + sweep
    lo_phi, hi_phi = (start, end) if sweep >= 0.0 else (end, start)
    amplitude, phase = math.hypot(a, b), math.atan2(b, a)
    values = [a * math.cos(start) + b * math.sin(start), a * math.cos(end) + b * math.sin(end)]

    def reached(angle: float) -> bool:
        k = math.ceil((lo_phi - angle) / (2.0 * math.pi))
        return angle + 2.0 * math.pi * k <= hi_phi

    if reached(phase):
        values.append(amplitude)
    if reached(phase + math.pi):
        values.append(-amplitude)
    return min(values), max(values)


def _edge_extent(model, edge_id: int, volume: BoxVolume) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative extent of an edge in the box frame (see :func:`_edge_bounds`)."""

    lo, hi, _tight = _edge_bounds(model, edge_id, volume)
    return lo, hi


def _edge_bounds(model, edge_id: int, volume: BoxVolume) -> Tuple[np.ndarray, np.ndarray, bool]:
    """``(lo, hi, tight)`` in the box frame.

    Tight (exact per box axis) for straight edges, circular arcs and elliptic arcs;
    sound but possibly loose for Bezier splines (control hull) and the other exact
    curve families (their world-axis enclosure transformed by its corners).  Loose
    bounds are sound for containment and box-side membership but may refuse a valid
    cut, so callers that need a decision refuse with ``VOLUME_CURVE_UNSUPPORTED``.
    """

    from .exact_curves import EllipticArc

    edge = model.edges[edge_id]
    curve = edge.curve
    if isinstance(curve, Straight):
        local = volume.local(np.array([model.vertex_position(edge.start), model.vertex_position(edge.end)]))
        return local.min(axis=0), local.max(axis=0), True
    if isinstance(curve, EllipticArc):
        rotation, translation = volume.rotation, volume.translation
        centre = (np.asarray(curve.center) - translation) @ rotation
        u, v = np.asarray(curve.u_vector) @ rotation, np.asarray(curve.v_vector) @ rotation
        lo, hi = np.empty(3), np.empty(3)
        for axis in range(3):
            low, high = _arc_range(float(u[axis]), float(v[axis]), float(curve.sweep_angle), float(curve.start_angle))
            lo[axis], hi[axis] = centre[axis] + low, centre[axis] + high
        return lo, hi, True
    if isinstance(curve, Arc):
        frame = model.arc_frame(edge_id)
        rotation, translation = volume.rotation, volume.translation
        centre = (np.asarray(frame.center) - translation) @ rotation
        lo, hi = np.empty(3), np.empty(3)
        for axis in range(3):
            direction = rotation[:, axis]
            low, high = _arc_range(float(frame.radius * (frame.e1 @ direction)),
                                   float(frame.radius * (frame.e2 @ direction)), float(frame.sweep))
            lo[axis], hi[axis] = centre[axis] + low, centre[axis] + high
        return lo, hi, True
    if isinstance(curve, Spline):
        points = [model.vertex_position(v) for v in (edge.start, *curve.control_vertices, edge.end)]
        local = volume.local(np.asarray(points, dtype=float))
        return local.min(axis=0), local.max(axis=0), False
    bounds = getattr(curve, "bounds", None)
    if callable(bounds):
        box = np.asarray(bounds(), dtype=float).reshape(-1)
        if box.size != 6 or not np.all(np.isfinite(box)):
            raise VolumeCutError("VOLUME_CURVE_UNSUPPORTED", f"edge {edge_id} has no usable enclosure")
        low, high = box[:3], box[3:]
        corners = np.array([[(low, high)[i][0], (low, high)[j][1], (low, high)[k][2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        local = volume.local(corners)
        return local.min(axis=0), local.max(axis=0), False
    raise VolumeCutError("VOLUME_CURVE_UNSUPPORTED", f"edge {edge_id} is a {type(curve).__name__} without an enclosure")


def _face_edges(model, face_id: int) -> List[int]:
    face = model.faces[face_id]
    return [use.edge for loop in (face.loop,) + tuple(face.holes) for use in loop]


def _face_extent(model, face_id: int, volume: BoxVolume) -> Tuple[np.ndarray, np.ndarray]:
    extents = [_edge_extent(model, e, volume) for e in _face_edges(model, face_id)]
    return np.min([e[0] for e in extents], axis=0), np.max([e[1] for e in extents], axis=0)


def _contained(model, face_id: int, volume: BoxVolume, tolerance: float, what: str) -> None:
    """Refuse unless the face's extent lies in the box; loose bounds refuse as unsupported."""

    for edge_id in _face_edges(model, face_id):
        lo, hi, tight = _edge_bounds(model, edge_id, volume)
        if not volume.contains_extent((lo, hi), tolerance):
            if not tight:
                raise VolumeCutError("VOLUME_CURVE_UNSUPPORTED",
                                     f"edge {edge_id} of {what} {face_id} has no tight enclosure to prove it inside")
            raise VolumeCutError("VOLUME_LEAK", f"{what} {face_id} reaches outside the box")


# ----------------------------------------------------------------------
# interior points, verified against the face's material domain
# ----------------------------------------------------------------------
def _material(model, face_id: int):
    from .material_arrangement import MaterialDomain

    domain = MaterialDomain.from_model(model, face_id)
    size = float(np.linalg.norm(np.ptp(_samples(model, face_id, _SCAN_SAMPLES[0]), axis=0)))
    return domain, model.tolerance.effective_length(size)


def _strictly_inside(domain, point: np.ndarray, tolerance: float) -> bool:
    uv = np.asarray(domain.support.local_uv(point), dtype=float)
    if np.linalg.norm(np.asarray(domain.support.evaluate(*uv)) - point) > tolerance:
        return False
    return (domain.contains_loop(domain.boundaries[0], uv, tolerance, boundary=False)
            and not any(domain.contains_loop(loop, uv, tolerance, boundary=True) for loop in domain.boundaries[1:]))


def _samples(model, face_id: int, count: int) -> np.ndarray:
    points = []
    for edge_id in _face_edges(model, face_id):
        points.extend(model.sample_edge(edge_id, np.linspace(0.0, 1.0, 2 if isinstance(
            model.edges[edge_id].curve, Straight) else count)))
    return np.asarray(points, dtype=float)


def _scan_candidates(model, face_id: int, count: int) -> List[np.ndarray]:
    """Candidate interior points: scanline gap midpoints of the sampled boundary, widest first."""

    face = model.faces[face_id]
    surface = getattr(face, "surface", None)
    loops = []
    for loop in (face.loop,) + tuple(face.holes):
        points = []
        for oriented in loop:
            n = 2 if isinstance(model.edges[oriented.edge].curve, Straight) else count
            samples = model.sample_edge(oriented.edge, np.linspace(0.0, 1.0, n))
            points.extend((samples if oriented.forward else samples[::-1])[:-1])
        loops.append(np.asarray(points, dtype=float))
    every = np.vstack(loops)
    if isinstance(surface, Cylinder):
        axis = np.asarray(surface.axis)
        first = np.asarray(surface.radial_direction)
        second = np.cross(axis, first)
        offset0 = every[0] - surface.origin
        reference = float(np.arctan2(offset0 @ second, offset0 @ first))

        def chart(x):
            offset = np.atleast_2d(x) - surface.origin
            z = offset @ axis
            radial = offset - np.outer(z, axis)
            angle = np.arctan2(radial @ second, radial @ first)
            angle = reference + np.angle(np.exp(1j * (angle - reference)))
            return np.column_stack([angle * surface.radius, z])

        def back(u, v):
            angle = u / surface.radius
            return surface.origin + surface.radius * (np.cos(angle) * first + np.sin(angle) * second) + v * axis
    else:
        origin = every.mean(axis=0)
        _u, _s, vt = np.linalg.svd(every - origin, full_matrices=False)
        basis = vt[:2]

        def chart(x):
            return (np.atleast_2d(x) - origin) @ basis.T

        def back(u, v):
            return origin + u * basis[0] + v * basis[1]
    planar_loops = [chart(loop) for loop in loops]
    values = np.unique(np.concatenate([loop[:, 1] for loop in planar_loops]))
    found = []
    for k in range(len(values) - 1):
        level = 0.5 * (values[k] + values[k + 1])
        crossings = []
        for loop in planar_loops:
            a, b = loop, np.roll(loop, -1, axis=0)
            straddle = (a[:, 1] - level) * (b[:, 1] - level) < 0.0
            t = (level - a[straddle, 1]) / (b[straddle, 1] - a[straddle, 1])
            crossings.extend((a[straddle, 0] + t * (b[straddle, 0] - a[straddle, 0])).tolist())
        crossings.sort()
        for start, end in zip(crossings[0::2], crossings[1::2]):
            found.append((min(end - start, values[k + 1] - values[k]), 0.5 * (start + end), level))
    found.sort(key=lambda item: -item[0])
    return [np.asarray(back(u, v), dtype=float) for _w, u, v in found[:16]]


def _interior_point(model, face_id: int) -> np.ndarray:
    """A point verified strictly inside the face's material (``VOLUME_FACE_DEGENERATE`` otherwise)."""

    domain, tolerance = _material(model, face_id)
    for count in _SCAN_SAMPLES:
        for point in _scan_candidates(model, face_id, count):
            if _strictly_inside(domain, point, tolerance):
                return point
    raise VolumeCutError("VOLUME_FACE_DEGENERATE", f"no point could be verified inside face {face_id}")


# ----------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeCutPlan:
    model_id: UUID
    revision: int
    source_checksum: str
    volume: BoxVolume
    tolerance: float
    operand_faces: Tuple[int, ...]  # faces whose conservative extent reaches the box
    dropped_members: Tuple[int, ...] = ()  # droppable members that reach the box (not carried)
    partitioned_members: Tuple[int, ...] = ()  # seam members that reach the box (carried, cut at the box)


def _refuse_coincident(model, face_id: int, extent, volume: BoxVolume, tolerance: float) -> None:
    """A face lying in a box plane with material over a positive area of that box side refuses.

    Material is found by verified points; an overlap too small to be found here
    still refuses later, when the imprint itself rejects it.
    """

    lo, hi = extent
    lower, upper = np.asarray(volume.lower), np.asarray(volume.upper)
    for axis in range(3):
        for bound in (lower[axis], upper[axis]):
            if abs(lo[axis] - bound) > tolerance or abs(hi[axis] - bound) > tolerance:
                continue
            others = [a for a in range(3) if a != axis]
            ranges = [(max(lo[a], lower[a]), min(hi[a], upper[a])) for a in others]
            if any(b - a <= tolerance for a, b in ranges):
                continue
            domain, face_tolerance = _material(model, face_id)
            grid = np.linspace(0.0, 1.0, 17)[1:-1]
            for s in grid:
                for t in grid:
                    q = np.empty(3)
                    q[axis] = bound
                    q[others[0]] = ranges[0][0] + s * (ranges[0][1] - ranges[0][0])
                    q[others[1]] = ranges[1][0] + t * (ranges[1][1] - ranges[1][0])
                    if _strictly_inside(domain, volume.world(q)[0], face_tolerance):
                        raise VolumeCutError(
                            "VOLUME_COINCIDENT",
                            f"face {face_id} has material in a box plane (a zero-thickness slab)",
                        )


def _planar_coons(model, face_id: int, surface) -> bool:
    """A topology-backed Coons patch with straight coplanar boundaries is exactly planar.

    This is the support ANYgeometry's own intersection planning accepts for such a
    face (``_qualified_face_plane``), so the imprint treats it as the plane it is.
    """

    from .intersections import _qualified_face_plane
    from .surfaces import CoonsSurface

    if not isinstance(surface, CoonsSurface):
        return False
    try:
        _qualified_face_plane(model, face_id)
    except GeometryError:
        return False
    return True


def plan_volume_cut(model, volume: BoxVolume, *, droppable_members: Iterable[int] = ()) -> VolumeCutPlan:
    """Read-only plan: the faces the box reaches, and every refusal that needs no mutation.

    ``droppable_members`` names members the caller declares carry no structure of
    their own (for example members with no beam section assigned).  Such a member
    that reaches the box is not carried into the extract and is reported in
    ``dropped_members``; it still refuses if it has attachments or junctions.

    Any other member that reaches the box is partitioned at the box (slice S2b) when
    it is a seam member there: every axis edge that reaches the box is an edge of a
    face the box reaches, so the imprint splits it with that face and the inside
    part is extracted with the inside faces.  Such members are reported in
    ``partitioned_members``.  A free-standing axis edge near the box, and an
    attachment or junction on a partitioned member, refuse.
    """

    if not isinstance(volume, BoxVolume):
        raise VolumeCutError("VOLUME_INVALID", "the volume must be a BoxVolume")
    tolerance = model.tolerance.effective_length(volume.diagonal)
    operands = []
    for face_id, face in sorted(model.faces.items()):
        extent = _face_extent(model, face_id, volume)
        if not volume.reaches(extent, tolerance):
            continue
        surface = getattr(face, "surface", None)
        if surface is not None and not isinstance(surface, (Plane, Cylinder)):
            if not _planar_coons(model, face_id, surface):
                raise VolumeCutError("VOLUME_SURFACE_UNSUPPORTED",
                                     f"face {face_id} near the box is a {type(surface).__name__}; planes, cylinders "
                                     "and planar four-sided patches are supported")
        if not isinstance(surface, Cylinder):
            _refuse_coincident(model, face_id, extent, volume, tolerance)
        operands.append(face_id)
    operand_set = set(operands)
    operand_edges = {e for f in operands for e in _face_edges(model, f)}
    droppable = {int(m) for m in droppable_members}
    unknown = sorted(droppable - set(model.members))
    if unknown:
        raise VolumeCutError("VOLUME_INVALID", f"droppable members {unknown} do not exist")
    dropped: List[int] = []
    partitioned: List[int] = []
    for member_id, member in sorted(model.members.items()):
        reaching = [model.member_edge_uses[u].edge_id for u in member.edge_use_ids
                    if volume.reaches(_edge_extent(model, model.member_edge_uses[u].edge_id, volume), tolerance)]
        if not reaching:
            continue
        if member_id in droppable:
            dropped.append(member_id)
            continue
        free = sorted(set(reaching) - operand_edges)
        if free:
            raise VolumeCutError("VOLUME_MEMBERS_UNSUPPORTED",
                                 f"member {member_id} reaches the box along free-standing axis edges {free}; only "
                                 "members on face edges (seam members) are partitioned")
        partitioned.append(member_id)
    for attachment_id, attachment in sorted(model.attachments.items()):
        if attachment.member_id in dropped or attachment.member_id in partitioned:
            raise VolumeCutError("VOLUME_RELATIONS_UNSUPPORTED",
                                 f"attachment {attachment_id} belongs to member {attachment.member_id}, "
                                 "which reaches the box")
        target = (str(getattr(attachment.target_kind, "value", attachment.target_kind)), attachment.target_id)
        if (target[0] == "face" and target[1] in operand_set) or (target[0] == "edge" and target[1] in operand_edges):
            raise VolumeCutError("VOLUME_RELATIONS_UNSUPPORTED",
                                 f"attachment {attachment_id} targets {target[0]} {target[1]} near the box (slice S2b)")
    operand_sheets = {model.face_uses[use].sheet_id for use in model.face_uses
                      if model.face_uses[use].face_id in operand_set} if hasattr(model, "face_uses") else set()
    for junction_id, junction in sorted(model.junctions.items()):
        if {use.member_id for use in junction.member_uses} & (set(dropped) | set(partitioned)):
            raise VolumeCutError("VOLUME_RELATIONS_UNSUPPORTED",
                                 f"junction {junction_id} uses a member that reaches the box")
        if set(junction.sheet_ids) & operand_sheets:
            raise VolumeCutError("VOLUME_RELATIONS_UNSUPPORTED",
                                 f"junction {junction_id} involves a sheet near the box (slice S2b)")
    if not operands:
        raise VolumeCutError("VOLUME_EMPTY", "no face reaches the box")
    return VolumeCutPlan(model.model_id, model.revision, to_dict(model)["checksum"]["value"], volume, tolerance,
                         tuple(operands), tuple(dropped), tuple(partitioned))


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeCutResult:
    closure: ModelClosure  # the extracted inside model and its maps to the working clone
    working_model: object  # the clone after the imprint (keeps the temporary cutting faces)
    cutting_faces: Tuple[int, ...]  # temporary box faces (and their descendants) in the working model
    classification: Mapping[int, str]  # working-model face -> INSIDE / OUTSIDE
    cut_edges: Mapping[str, Tuple[int, ...]]  # box side -> extracted artificial cut edges on it
    original_boundary_edges: Tuple[int, ...]  # extracted free edges that were free in the source
    lineage: Mapping[Tuple[str, int], Tuple[int, ...]]  # extracted (kind, id) -> source ids of that kind
    created: Tuple[Tuple[str, int], ...]  # made by the cut: on the box, arc via vertices, split points of source edges
    unresolved: Tuple[Tuple[str, int], ...]  # extracted entities with no source and not made by the cut
    volume: BoxVolume = field(default=None)
    dropped_members: Tuple[int, ...] = ()  # members the caller declared droppable that reached the box
    # extracted member -> (source member, the source member's parameter range it covers)
    member_ranges: Mapping[int, Tuple[int, Tuple[float, float]]] = field(default_factory=dict)

    @property
    def source_faces(self) -> Dict[int, Tuple[int, ...]]:
        return {i: s for (k, i), s in self.lineage.items() if k == "face"}

    @property
    def source_edges(self) -> Dict[int, Tuple[int, ...]]:
        return {i: s for (k, i), s in self.lineage.items() if k == "edge"}

    @property
    def source_vertices(self) -> Dict[int, Tuple[int, ...]]:
        return {i: s for (k, i), s in self.lineage.items() if k == "vertex"}

    @property
    def source_members(self) -> Dict[int, Tuple[int, ...]]:
        return {i: s for (k, i), s in self.lineage.items() if k == "member"}


def _add_box_faces(model, volume: BoxVolume) -> Dict[str, int]:
    lower, upper = np.asarray(volume.lower), np.asarray(volume.upper)

    def position(index):
        return volume.world(np.array([(lower, upper)[index[d]][d] for d in range(3)]))[0]

    corner_ids = {(i, j, k): model.add_point(*position((i, j, k))) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    edges: Dict[Tuple, int] = {}

    def edge(a, b):
        key = tuple(sorted((a, b)))
        if key not in edges:
            edges[key] = model.add_line(corner_ids[a], corner_ids[b])
        return edges[key]

    faces = {}
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for side in (0, 1):
            quad = []
            for s, t in ((0, 0), (1, 0), (1, 1), (0, 1)):
                index = [0, 0, 0]
                index[axis], index[others[0]], index[others[1]] = side, s, t
                quad.append(tuple(index))
            loop = [edge(quad[n], quad[(n + 1) % 4]) for n in range(4)]
            origin = position(quad[0])
            surface = Plane(origin, position(quad[1]) - origin, position(quad[3]) - origin)
            faces[SIDES[2 * axis + side]] = model.add_face(loop, surface=surface)
    return faces


def _rebuild_structure(work, extracted, work_of: Mapping, inside: Set[int]) -> Dict[Tuple[str, int], Tuple[str, int]]:
    """Recreate parts and sheets in the extracted model from their inside face uses.

    Returns ``extracted (kind, id) -> working (kind, id)`` for the new parts, sheets,
    face uses and coedges, each one to one with its working counterpart: a sheet with
    its working sheet, a part with its working part, a face use with the working use
    of the same face in that sheet, a coedge with that use's coedge on the same edge.
    A sheet that declares non-manifold edges with inside material is refused here.
    """

    clone_of: Dict[Tuple[str, int], Tuple[str, int]] = {}
    work_edge_of = {target[1]: key[1] for key, target in work_of.items() if key[0] == "edge"}
    parts: Dict[int, int] = {}
    for sheet_id, sheet in sorted(work.sheets.items()):
        uses = [(u, work.face_uses[u]) for u in sheet.face_use_ids if work.face_uses[u].face_id in inside]
        if not uses:
            continue
        if sheet.declared_non_manifold_edges:
            raise VolumeCutError("VOLUME_RELATIONS_UNSUPPORTED",
                                 f"sheet {sheet_id} declares non-manifold edges and has material inside the box")
        if sheet.part_id not in parts:
            part = work.parts[sheet.part_id]
            parts[sheet.part_id] = extracted.add_part(name=part.name, metadata=dict(part.metadata))
            clone_of[("part", parts[sheet.part_id])] = ("part", sheet.part_id)
        new_sheet = extracted.add_sheet([work_of[("face", record.face_id)][1] for _u, record in uses],
                                        part_id=parts[sheet.part_id], name=sheet.name, policy=sheet.policy,
                                        orientations=[record.orientation for _u, record in uses])
        clone_of[("sheet", new_sheet)] = ("sheet", sheet_id)
        by_face = {work_of[("face", record.face_id)][1]: (u, record) for u, record in uses}
        for new_use in extracted.sheets[new_sheet].face_use_ids:
            work_use, record = by_face[extracted.face_uses[new_use].face_id]
            clone_of[("face_use", new_use)] = ("face_use", work_use)
            work_coedges = {work.coedges[c].edge_id: c for c in record.coedge_ids}
            for new_coedge in extracted.face_uses[new_use].coedge_ids:
                work_edge = work_edge_of[extracted.coedges[new_coedge].edge_id]
                if work_edge not in work_coedges:
                    raise VolumeCutError("VOLUME_LINEAGE", f"face use {work_use} has no coedge on edge {work_edge}")
                clone_of[("coedge", new_coedge)] = ("coedge", work_coedges[work_edge])
    return clone_of


def _partition_members(work, extracted, work_of: Mapping, clone_of: Dict, members: Sequence[int],
                       volume: BoxVolume, tolerance: float) -> Dict[int, Tuple[int, Tuple[float, float]]]:
    """Carry the inside part of each partitioned working member into the extract.

    The imprint split the seam member's axis edges at the box together with their
    faces, so the member's edge uses on extracted edges are its inside part.  They
    must be one contiguous run of the member's axis; a use left out must lie
    outside the box (by its midpoint), and an orientation reference must have been
    extracted.  Each new member and member edge use maps one to one onto its
    working counterpart (added to ``clone_of``).  Returns ``new member ->
    (working member, covered parameter range)``.
    """

    from .entities import OrientedEdge
    from .structural import Orientation

    parts = {work_id: new_id for (kind, new_id), (_k, work_id) in clone_of.items() if kind == "part"}
    ranges: Dict[int, Tuple[int, Tuple[float, float]]] = {}
    for member_id in members:
        if member_id not in work.members:
            raise VolumeCutError("VOLUME_LINEAGE", f"member {member_id} is missing from the working model")
        member = work.members[member_id]
        uses = [work.member_edge_uses[u] for u in member.edge_use_ids]
        kept = [i for i, use in enumerate(uses) if ("edge", use.edge_id) in work_of]
        for i, use in enumerate(uses):
            if i in kept:
                continue
            middle = work.sample_edge(use.edge_id, np.array([0.5]))[0]
            if volume.classify(middle, tolerance) != OUTSIDE:
                raise VolumeCutError("VOLUME_MEMBERS_UNSUPPORTED",
                                     f"member {member_id} runs on edge {use.edge_id} inside the box, which no "
                                     "inside face carries")
        if not kept:
            continue  # the member only touches the box
        if kept != list(range(kept[0], kept[-1] + 1)):
            raise VolumeCutError("VOLUME_MEMBERS_UNSUPPORTED", f"member {member_id} enters the box more than once")
        reference = member.orientation_reference
        if reference is not None:
            if tuple(reference) not in work_of:
                raise VolumeCutError("VOLUME_RELATIONS_UNSUPPORTED",
                                     f"member {member_id} is oriented by {tuple(reference)} outside the box")
            reference = work_of[tuple(reference)]
        if member.part_id not in parts:
            part = work.parts[member.part_id]
            parts[member.part_id] = extracted.add_part(name=part.name, metadata=dict(part.metadata))
            clone_of[("part", parts[member.part_id])] = ("part", member.part_id)
        chain = [uses[i] for i in kept]
        new_member = extracted.add_member(
            [OrientedEdge(work_of[("edge", use.edge_id)][1], use.orientation is Orientation.FORWARD) for use in chain],
            part_id=parts[member.part_id], name=member.name, metadata=dict(member.metadata),
            orientation_reference=reference)
        clone_of[("member", new_member)] = ("member", member_id)
        for new_use, use in zip(extracted.members[new_member].edge_use_ids, chain):
            clone_of[("member_edge_use", new_use)] = ("member_edge_use", use.id)
        ranges[new_member] = (member_id, (min(u.parent_range.start for u in chain),
                                          max(u.parent_range.end for u in chain)))
    return ranges


def apply_volume_cut(model, plan: VolumeCutPlan) -> VolumeCutResult:
    """Cut ``plan.volume`` out of a detached clone of ``model`` and extract the inside."""

    if (model.model_id != plan.model_id or model.revision != plan.revision
            or to_dict(model)["checksum"]["value"] != plan.source_checksum):
        raise VolumeCutError("VOLUME_STALE_PLAN", "the model changed since the cut was planned")
    volume, tolerance = plan.volume, plan.tolerance
    work = model.clone(include_features=False)
    cutting = _add_box_faces(work, volume)
    operands = [EntityRef("face", f) for f in plan.operand_faces] + [EntityRef("face", f) for f in cutting.values()]
    policy = IntersectionBatchPolicy(intent=ConnectionIntent.IMPRINT, member_connections=False,
                                     member_face_connections=False)
    try:
        imprint = plan_intersections(work, operands, policy=policy)
        apply_intersections(work, imprint, policy=policy)
    except GeometryError as error:
        raise VolumeCutError("VOLUME_IMPRINT_FAILED", f"the box could not be imprinted: {error}") from error

    cutting_ids: Set[int] = set()
    for face_id in cutting.values():
        cutting_ids |= {ref.id for ref in work.resolve_ref(EntityRef("face", face_id))}
    # Only the plan's faces were imprinted; every other face's conservative extent
    # misses the box, so it is outside without further work.
    descendants = sorted({ref.id for f in plan.operand_faces for ref in work.resolve_ref(EntityRef("face", f))
                          if ref.kind == "face"} - cutting_ids)

    # The imprint must trace only the box: every new edge on a source face lies on a box side.
    source_edge_descendants = {ref.id for e in model.edges for ref in work.resolve_ref(EntityRef("edge", e))
                               if ref.kind == "edge"}
    for face_id in descendants:
        for edge_id in _face_edges(work, face_id):
            if edge_id in source_edge_descendants:
                continue
            lo, hi, tight = _edge_bounds(work, edge_id, volume)
            if not volume.sides_of((lo, hi), tolerance):
                if not tight:
                    raise VolumeCutError("VOLUME_CURVE_UNSUPPORTED",
                                         f"new edge {edge_id} has no tight enclosure to prove it on the box")
                raise VolumeCutError(
                    "VOLUME_SOURCE_INTERSECTION",
                    f"the imprint traced edge {edge_id} off the box on face {face_id}: source faces intersect one "
                    "another without shared topology; prepare their intersections first",
                )

    classification: Dict[int, str] = {}
    for face_id in descendants:
        state = volume.classify(_interior_point(work, face_id), tolerance)
        if state == ON:
            raise VolumeCutError("VOLUME_COINCIDENT", f"face {face_id} lies on the box boundary (a zero-thickness slab)")
        classification[face_id] = state
    inside = [f for f in descendants if classification[f] == INSIDE]
    if not inside:
        raise VolumeCutError("VOLUME_EMPTY", "no face lies inside the box")
    for face_id in inside:
        _contained(work, face_id, volume, tolerance, "inside face")

    # Geometry only: the structural closure follows coedges across the shared cut edges
    # and keeps sheets complete, so it would pull the outside parts of split faces back
    # in.  Parts and sheets are rebuilt in the extract from their inside face uses (the
    # contract's remap onto inside descendants), and the closure is completed with them.
    geometric = extract_model_closure(work, [EntityRef("face", f) for f in inside], include_structural_closure=False)
    extracted = geometric.working_model
    work_of = {handle.key: target.key for handle, target in geometric.source_to_work.items()}
    outside = {("face", f) for f in descendants if classification[f] == OUTSIDE} | {("face", f) for f in cutting_ids}
    leaked = sorted(key for key in work_of if key in outside)
    if leaked:
        raise VolumeCutError("VOLUME_LEAK", f"the bounded extraction pulled in {leaked}")
    for face_id in extracted.faces:
        _contained(extracted, face_id, volume, tolerance, "extracted face")
    clone_of = _rebuild_structure(work, extracted, work_of, set(inside))
    member_ranges = _partition_members(work, extracted, work_of, clone_of, plan.partitioned_members, volume,
                                       tolerance)
    work_to_source = dict(geometric.work_to_source)
    for (kind, new_id), (work_kind, work_id) in clone_of.items():
        work_to_source[extracted.handle(kind, new_id)] = work.handle(work_kind, work_id)
    closure = ModelClosure(
        working_model=extracted,
        source_to_work=MappingProxyType({source: target for target, source in work_to_source.items()}),
        work_to_source=MappingProxyType(work_to_source),
        source_model_id=geometric.source_model_id, source_revision=geometric.source_revision,
        source_handles=geometric.source_handles,
    )
    work_key = {target.key: source.key for target, source in work_to_source.items()}  # extracted -> working

    # Lineage: extracted -> working (closure) -> source (replacement history for
    # geometry; ids for parts and sheets; same sheet and ancestor face for face uses;
    # same source use and ancestor edge for coedges).
    ancestors: Dict[Tuple[str, int], Set[int]] = {}
    for kind, store in (("face", model.faces), ("edge", model.edges), ("vertex", model.vertices),
                        ("sheet", model.sheets), ("part", model.parts), ("member", model.members)):
        for source_id in store:
            for ref in work.resolve_ref(EntityRef(kind, source_id)):
                if ref.kind == kind:
                    ancestors.setdefault((kind, ref.id), set()).add(int(source_id))

    def use_sources(work_use: int) -> Set[int]:
        record = work.face_uses[work_use]
        faces = ancestors.get(("face", record.face_id), set())
        sheets = ancestors.get(("sheet", record.sheet_id), set())
        return {u for u, source in model.face_uses.items() if source.sheet_id in sheets and source.face_id in faces}

    lineage: Dict[Tuple[str, int], Set[int]] = {}
    for target, (kind, work_id) in work_key.items():
        if kind == "face_use":
            sources = use_sources(work_id)
        elif kind == "member_edge_use":
            record = work.member_edge_uses[work_id]
            members = ancestors.get(("member", record.member_id), set())
            edges = ancestors.get(("edge", record.edge_id), set())
            sources = {u for u, source in model.member_edge_uses.items()
                       if source.member_id in members and source.edge_id in edges}
        elif kind == "coedge":
            record = work.coedges[work_id]
            uses, edges = use_sources(record.face_use_id), ancestors.get(("edge", record.edge_id), set())
            sources = {c for c, source in model.coedges.items() if source.face_use_id in uses and source.edge_id in edges}
        else:
            sources = ancestors.get((kind, work_id), set())
        if sources:
            lineage[target] = sources
    lineage_final = {key: tuple(sorted(value)) for key, value in sorted(lineage.items())}
    missing = sorted(set(extracted.faces) - {i for k, i in lineage_final if k == "face"})
    if missing:
        raise VolumeCutError("VOLUME_LINEAGE", f"extracted faces {missing} have no source face")
    via_vertices = {edge.curve.via_vertex for edge in extracted.edges.values() if isinstance(edge.curve, Arc)}
    ends: Dict[int, List[Tuple[int, ...]]] = {}
    for edge_id, edge in extracted.edges.items():
        for vertex in (edge.start, edge.end):
            ends.setdefault(vertex, []).append(lineage_final.get(("edge", edge_id), ()))
    split_points = {v for v, sources in ends.items()
                    if len(sources) == 2 and sources[0] and sources[0] == sources[1] and len(sources[0]) == 1}
    created, unresolved = [], []
    for target in sorted(work_key):
        if target in lineage_final:
            continue
        made = False
        if target[0] == "edge":
            made = bool(volume.sides_of(_edge_extent(extracted, target[1], volume), tolerance))
        elif target[0] == "coedge":
            made = ("edge", extracted.coedges[target[1]].edge_id) not in lineage_final  # a use of a cut edge
        elif target[0] == "vertex":
            q = volume.local(np.asarray(extracted.vertex_position(target[1])))[0]
            made = any(abs(q[a] - b) <= tolerance for a in range(3) for b in (volume.lower[a], volume.upper[a]))
            # The control (via) vertex of an arc the imprint split or traced, and a point
            # where the imprint split a source edge, are the cut's own.
            made = made or target[1] in via_vertices or target[1] in split_points
        (created if made else unresolved).append(target)

    # Artificial cuts: material was removed beside the edge.  An extracted edge is a
    # cut when it is free here and either new (made on the box) or its source edge
    # had more face uses in the source than it has here.
    def uses_of(geometry) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for face_id in geometry.faces:
            for edge_id in _face_edges(geometry, face_id):
                counts[edge_id] = counts.get(edge_id, 0) + 1
        return counts

    extracted_uses, source_uses = uses_of(extracted), uses_of(model)
    cut_edges: Dict[str, List[int]] = {}
    original_boundary = []
    for edge_id, count in sorted(extracted_uses.items()):
        sources = lineage_final.get(("edge", edge_id), ())
        removed = (not sources) or any(source_uses.get(s, 0) > count for s in sources)
        if removed:
            sides = volume.sides_of(_edge_extent(extracted, edge_id, volume), tolerance)
            if not sides:
                raise VolumeCutError("VOLUME_LINEAGE", f"edge {edge_id} lost material beside it away from the box")
            for side in sides:
                cut_edges.setdefault(side, []).append(edge_id)
        elif count == 1:
            original_boundary.append(edge_id)
    return VolumeCutResult(
        closure=closure, working_model=work, cutting_faces=tuple(sorted(cutting_ids)),
        classification=classification, cut_edges={k: tuple(v) for k, v in cut_edges.items()},
        original_boundary_edges=tuple(original_boundary), lineage=lineage_final, created=tuple(created),
        unresolved=tuple(unresolved), volume=volume, dropped_members=plan.dropped_members,
        member_ranges=MappingProxyType({new: (lineage_final[("member", new)][0], span)
                                        for new, (_work_member, span) in sorted(member_ranges.items())}),
    )
