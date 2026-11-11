"""Derive a solid detail box from a rectangular footprint on a planar shell face.

``derive_shell_detail_box`` reads one source model without mutating it and
returns a *new, detached* ``GeometryModel`` that owns exactly one closed
convex normal prism (made by the canonical solid-prism construction) plus the
persistent region id.  See ``docs/solid_volume_contract.md`` ("Shell detail
box") for the convention, in particular the meaning of ``reference_offset``.

Validation is fail-closed and uses the owner's material-domain predicates
(``MaterialDomain`` through ``volume_cut._material``/``_strictly_inside``):
the whole closed rectangle must lie strictly inside the material of one planar
face, so neither a hole nor a concave rim may touch or enter it.  Only exact
straight trim edges (outer rim and holes) are accepted.  A curved edge is
refused with ``DETAIL_CURVED_BOUNDARY``: sampling cannot rule out a tangent or
missed incursion between samples, and no certified owner predicate for curved
enclosure is available.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from typing import Tuple
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from .curves import Straight
from .errors import GeometryError
from .identity import EntityHandle
from .serialization import to_dict
from .solid_regions import add_prism
from .volume_cut import _face_edges, _material, _planar_coons, _strictly_inside

__all__ = ["ShellDetailError", "derive_shell_detail_box"]

_SCHEMA = 1


class ShellDetailError(GeometryError):
    """A typed, fail-closed refusal of a shell detail request."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ShellDetailError("DETAIL_INPUT", f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ShellDetailError("DETAIL_INPUT", f"{name} must be finite")
    return number


def _array(value, shape, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ShellDetailError("DETAIL_INPUT", f"{name} must be numeric") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ShellDetailError("DETAIL_INPUT", f"{name} must be a finite array of shape {shape}")
    return array


def _point_segment_distance(point, a, b) -> float:
    edge = b - a
    length2 = float(edge @ edge)
    t = 0.0 if length2 == 0.0 else min(1.0, max(0.0, float((point - a) @ edge) / length2))
    return float(np.linalg.norm(point - (a + t * edge)))


def _orient(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _coplanar_segment_distance(p, q, r, s, frame) -> float:
    """Distance between two coplanar segments; 0 when they properly cross."""

    origin, axis_u, axis_v = frame
    flat = [np.array([(x - origin) @ axis_u, (x - origin) @ axis_v]) for x in (p, q, r, s)]
    d1, d2 = _orient(flat[0], flat[1], flat[2]), _orient(flat[0], flat[1], flat[3])
    d3, d4 = _orient(flat[2], flat[3], flat[0]), _orient(flat[2], flat[3], flat[1])
    if d1 * d2 < 0.0 and d3 * d4 < 0.0:
        return 0.0
    return min(_point_segment_distance(p, r, s), _point_segment_distance(q, r, s),
               _point_segment_distance(r, p, q), _point_segment_distance(s, p, q))


def _boundary_polylines(model, face_id: int):
    face = model.faces[face_id]
    for loop in (face.loop,) + tuple(face.holes):
        for use in loop:
            curve = model.edges[use.edge].curve
            if type(curve) is not Straight:
                raise ShellDetailError("DETAIL_CURVED_BOUNDARY",
                                       f"edge {use.edge} of face {face_id} is a {type(curve).__name__}; enclosure by a curved "
                                       "trim boundary has no certified predicate, so only exact straight edges are accepted")
            yield np.asarray(model.sample_edge(use.edge, np.array([0.0, 1.0])), dtype=float)


def _validate_rectangle(points: np.ndarray, tolerance: float, angular: float):
    """Return ``(origin, unit_a, unit_b, length_a, length_b)`` of a finite rectangle."""

    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= tolerance):
        raise ShellDetailError("DETAIL_FOOTPRINT", "footprint has a degenerate or repeated corner")
    if np.linalg.norm(points[0] + points[2] - points[1] - points[3]) > tolerance:
        raise ShellDetailError("DETAIL_FOOTPRINT", "footprint is not a parallelogram (opposite edges differ)")
    for i in range(4):
        cosine = float(edges[i] @ edges[(i + 1) % 4]) / float(lengths[i] * lengths[(i + 1) % 4])
        if abs(cosine) > angular:
            raise ShellDetailError("DETAIL_FOOTPRINT", "footprint corners are not right angles")
    if abs(lengths[0] - lengths[2]) > tolerance or abs(lengths[1] - lengths[3]) > tolerance:
        raise ShellDetailError("DETAIL_FOOTPRINT", "footprint opposite edges differ in length")
    return (points[0], edges[0] / lengths[0], (points[3] - points[0]) / lengths[3],
            float(lengths[0]), float(lengths[3]))


def _check_material(model, face_id: int, corners: np.ndarray, frame) -> None:
    polylines = list(_boundary_polylines(model, face_id))  # refuses any non-straight trim edge first
    domain, tolerance = _material(model, face_id)
    origin, axis_u, axis_v, length_u, length_v = frame
    centre = corners.mean(axis=0)
    for name, point in [("corner", c) for c in corners] + [("centre", centre)]:
        if not _strictly_inside(domain, np.asarray(point, dtype=float), tolerance):
            raise ShellDetailError("DETAIL_OUTSIDE_MATERIAL",
                                   f"footprint {name} {np.round(point, 9).tolist()} is not strictly inside "
                                   f"the material of face {face_id} (outside, on a rim, or in a hole)")
    sides = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    plane_frame = (origin, axis_u, axis_v)
    for polyline in polylines:
        for a, b in zip(polyline[:-1], polyline[1:]):
            for r, s in sides:
                if _coplanar_segment_distance(a, b, r, s, plane_frame) <= tolerance:
                    raise ShellDetailError("DETAIL_BOUNDARY",
                                           f"a boundary edge of face {face_id} touches or crosses the footprint")
            # No contact with the rectangle outline: the segment is wholly inside or outside.
            s_coord, t_coord = float((a - origin) @ axis_u), float((a - origin) @ axis_v)
            if 0.0 <= s_coord <= length_u and 0.0 <= t_coord <= length_v:
                raise ShellDetailError("DETAIL_BOUNDARY",
                                       f"a hole or rim of face {face_id} lies inside the footprint")


def _signature(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def derive_shell_detail_box(model, face_id, footprint, *, director, thickness, reference_offset=0.0) -> Tuple[object, str]:
    """Return ``(detached GeometryModel, region_id)`` for a detail box on a shell face.

    ``footprint``: four ordered xyz points on the shell reference plane, strictly
    inside the material of the single planar source face ``face_id``.  Either
    winding and any rotation of the rectangle is accepted; ``side_i`` of the box
    is always the face standing on the input edge ``footprint[i] -> footprint[i+1]``.

    ``director``: physical thickness direction; it must point the way of the
    authoritative face normal (``model.face_normal``).  ``thickness``: positive,
    metres.  ``reference_offset``: signed height of the box midsurface above the
    supplied shell plane along the director; ``start`` lies at
    ``reference_offset - thickness/2`` and ``end`` at ``reference_offset + thickness/2``.

    The source model is only read (same object, revision and content afterwards).
    The result is a new model with its own id; the region's provenance records
    ``shell_detail`` (source model id, revision, content checksum, face id, exact
    footprint, director, thickness, offset and a SHA-256 ``footprint_signature``).
    """

    from .model import GeometryModel

    if not isinstance(model, GeometryModel):
        raise ShellDetailError("DETAIL_INPUT", "model must be a GeometryModel")
    if model.units != "m":
        raise ShellDetailError("DETAIL_UNITS", f"shell detail boxes require SI metres, not {model.units!r}; "
                               "convert the source model first")
    if isinstance(face_id, EntityHandle):
        if face_id.kind != "face" or face_id.model_id != model.model_id:
            raise ShellDetailError("DETAIL_INPUT", "face handle does not belong to this model")
        face_id = face_id.id
    if isinstance(face_id, bool) or not isinstance(face_id, (int, np.integer)):
        raise ShellDetailError("DETAIL_INPUT", "face_id must be an integer face ID")
    face_id = int(face_id)
    face = model.faces.get(face_id)
    if face is None:
        raise ShellDetailError("DETAIL_FACE", f"unknown face {face_id}")
    from .surfaces import Plane
    support = face.support_surface
    # A topology-backed Coons patch with straight coplanar boundaries is exactly planar; the owner's
    # volume_cut predicate decides (it refuses nonplanar, crossed or otherwise unqualified patches).
    planar_coons = not isinstance(support, Plane) and _planar_coons(model, face_id, support)
    if not isinstance(support, Plane) and not planar_coons:
        raise ShellDetailError("DETAIL_FACE", f"face {face_id} is a {type(support).__name__}; only planar faces are supported")

    thickness = _scalar(thickness, "thickness")
    offset = _scalar(reference_offset, "reference_offset")
    if thickness <= 0.0:
        raise ShellDetailError("DETAIL_INPUT", "thickness must be positive")
    points = _array(footprint, (4, 3), "footprint")
    wanted = _array(director, (3,), "director")
    if float(np.linalg.norm(wanted)) == 0.0:
        raise ShellDetailError("DETAIL_DIRECTOR", "director must be nonzero")
    wanted = wanted / float(np.linalg.norm(wanted))

    _domain, tolerance = _material(model, face_id)
    angular = max(float(model.tolerance.angular), 256 * float(np.finfo(float).eps))

    # Authoritative normal of the source face at the footprint.
    try:
        projected, uv, _distance = model.project_to_face(face_id, points.mean(axis=0))
        normal = np.asarray(model.face_normal(face_id, *uv), dtype=float)
    except GeometryError as error:
        raise ShellDetailError("DETAIL_FACE", f"face {face_id} has no usable normal: {error}") from error
    normal = normal / float(np.linalg.norm(normal))
    if float(np.linalg.norm(np.cross(wanted, normal))) > angular or float(wanted @ normal) <= 0.0:
        raise ShellDetailError("DETAIL_DIRECTOR",
                               f"director {wanted.tolist()} is not the authoritative normal {normal.tolist()} of face {face_id}")

    # A Coons patch has no ``origin``: its reference plane passes through the source projection of the footprint.
    plane_origin = np.asarray(projected if planar_coons else support.origin, dtype=float)
    heights = (points - plane_origin) @ normal
    if float(np.max(np.abs(heights))) > tolerance:
        raise ShellDetailError("DETAIL_FOOTPRINT", "footprint is not on the shell reference plane of the face")
    flat = points - np.outer(heights, normal)  # exact projection onto the reference plane

    frame = _validate_rectangle(flat, tolerance, angular)
    _check_material(model, face_id, flat, frame)

    # Winding normal exactly as the canonical prism derives it, so the extrusion is accepted as normal.
    q = flat - flat[0]
    q = q / float(np.max(np.linalg.norm(q, axis=1)))
    area = sum((np.cross(a, b) for a, b in zip(q, np.roll(q, -1, axis=0))), np.zeros(3))
    winding = area / float(np.linalg.norm(area))
    sense = 1.0 if float(winding @ normal) > 0.0 else -1.0
    start_height, end_height = offset - thickness / 2.0, offset + thickness / 2.0
    if not start_height < end_height:
        raise ShellDetailError("DETAIL_INPUT", "thickness is not representable at this offset")

    source_checksum = to_dict(model)["checksum"]["value"]
    authored = {
        "schema_version": _SCHEMA,
        "source_model_id": str(model.model_id),
        "source_revision": int(model.revision),
        "source_checksum": source_checksum,
        "face_id": face_id,
        "footprint": points.tolist(),
        "director": np.asarray(director, dtype=float).tolist(),
        "thickness": thickness,
        "reference_offset": offset,
        "units": "m",
    }
    signature = _signature(authored)
    region_id = str(uuid5(NAMESPACE_URL, "anygeometry:shell_detail:" + signature))

    detail = GeometryModel(tolerance=model.tolerance)
    if (np.any(model.local_origin != 0.0) or model.coordinate_transform is not None or len(model.crs_metadata)):
        detail.set_document_settings(local_origin=model.local_origin,
                                     coordinate_transform=model.coordinate_transform,
                                     crs_metadata=model.crs_metadata.to_dict())
    with detail.transaction():
        made = add_prism(detail, flat, sense * thickness * winding, offset=sense * start_height,
                         name=f"shell-detail-{signature[:16]}", region_id=region_id,
                         provenance={"shell_detail": {**authored, "footprint_signature": signature,
                                                       "start_height": start_height, "end_height": end_height,
                                                       "face_normal": normal.tolist()}})

    _verify_roles(detail, made, flat, normal, start_height, end_height)
    return detail, made


def _verify_roles(detail, region_id: str, flat: np.ndarray, normal: np.ndarray, start: float, end: float) -> None:
    """Independent post-check of the face roles on the published snapshot."""

    snapshot = detail.solid_region_snapshot(region_id)
    xyz = np.asarray(snapshot.coordinates)
    scale = max(float(np.ptp(xyz, axis=0).max()), np.finfo(float).tiny)
    slack = 1e-9 * scale
    expected = np.vstack((flat + start * normal, flat + end * normal))
    if float(np.max(np.abs(xyz - expected))) > slack:
        raise ShellDetailError("DETAIL_VERIFY", "box coordinates do not match the requested heights")
    for name in ("start", "end", *(f"side_{i}" for i in range(4))):
        if name not in snapshot.face_groups or len(snapshot.face_groups[name]) != 1:
            raise ShellDetailError("DETAIL_VERIFY", f"face group {name} is missing")
    refs = list(snapshot.face_refs)
    for i in range(4):
        loop = set(snapshot.face_loops[refs.index(snapshot.face_groups[f"side_{i}"][0])])
        if loop != {i, (i + 1) % 4, i + 4, (i + 1) % 4 + 4}:
            raise ShellDetailError("DETAIL_VERIFY", f"side_{i} does not stand on footprint edge {i}")
