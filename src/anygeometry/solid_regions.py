"""Native closed convex normal prisms; no meshing or backend dependencies."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

import numpy as np

from .curves import Straight
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .identity import EntityHandle, canonical_model_id
from .structural import FrozenMetadata
from .surfaces import Plane


def _readonly(value):
    source = np.asarray(value, dtype=np.float64)
    # Immutable bytes backing also prevents consumers re-enabling writeability.
    return np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)


def _prism(profile, vector, offset=0.0):
    try:
        p = np.array(profile, dtype=float, copy=True)
        v = np.asarray(vector, dtype=float)
        offset = float(offset)
    except (ValueError, TypeError, OverflowError) as exc:
        raise GeometryError("solid prism requires finite numeric profile, vector and offset") from exc
    if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 3 or not np.isfinite(p).all():
        raise GeometryError("solid profile must contain at least three finite 3D points")
    if v.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(offset):
        raise GeometryError("solid vector and offset must be finite")
    q = p - p[0]
    extent = float(np.max(np.linalg.norm(q, axis=1)))
    # Relative floating point guards, not modelling repair or tolerance relaxation.
    eps = 256 * np.finfo(float).eps
    if extent == 0 or not np.isfinite(extent):
        raise GeometryError("degenerate solid profile")
    q = q / extent
    for i in range(len(q)):
        if np.any(np.linalg.norm(q[i+1:] - q[i], axis=1) <= eps):
            raise GeometryError("solid profile repeats a vertex")
    area_vector = sum((np.cross(a, b) for a, b in zip(q, np.roll(q, -1, axis=0))), np.zeros(3))
    twice_area = float(np.linalg.norm(area_vector))
    if twice_area <= eps:
        raise GeometryError("solid profile has zero area or inconsistent winding")
    normal = area_vector / twice_area
    if np.max(np.abs(q @ normal)) > eps:
        raise GeometryError("solid profile must be planar")
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        edge = b-a
        if np.linalg.norm(edge) <= eps:
            raise GeometryError("solid profile repeats a vertex")
        side = np.cross(edge, q-a) @ normal
        # Every other vertex must be strictly on the interior side: excludes
        # concavity, self crossings, duplicates and redundant collinear corners.
        index = np.flatnonzero(np.linalg.norm(q-a, axis=1) <= eps)
        allowed = np.linalg.norm(q-b, axis=1) <= eps
        allowed[index] = True
        if np.any(side[~allowed] <= eps * np.linalg.norm(edge)):
            raise GeometryError("solid profile must be strictly convex and simple")
    length = float(np.linalg.norm(v))
    height = float(v @ normal)
    if length == 0 or not np.isfinite(length) or abs(height) <= eps*length:
        raise GeometryError("solid extrusion must be nonzero")
    if np.linalg.norm(np.cross(v / length, normal)) > eps:
        raise GeometryError("solid extrusion must be normal to the profile")
    p += offset * normal
    xyz = np.vstack((p, p+v))
    if not np.isfinite(xyz).all() or np.any(np.all(xyz[:len(p)] == xyz[len(p):], axis=1)):
        raise GeometryError("solid extrusion is not representable at these coordinates")
    n = len(p)
    loops = [tuple(reversed(range(n))), tuple(range(n, 2*n))]
    loops += [(i, (i+1) % n, (i+1) % n+n, i+n) for i in range(n)]
    if height < 0:
        loops = [tuple(reversed(loop)) for loop in loops]
    volume = twice_area * extent**2 * abs(height) / 2
    if not np.isfinite(volume) or volume <= 0:
        raise GeometryError("solid volume must be finite and positive")
    return xyz, tuple(loops), volume


@dataclass(frozen=True, slots=True)
class ClosedSolidRegion:
    """Owner-bound persistent region, with explicit oriented existing faces."""
    model_id: str
    region_id: str
    vertex_ids: tuple[int, ...]
    face_refs: tuple[EntityHandle, ...]
    face_orientations: tuple[bool, ...]
    face_groups: Mapping[str, tuple[EntityHandle, ...]]
    name: str | None = None
    provenance: FrozenMetadata = FrozenMetadata()
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "model_id", str(canonical_model_id(self.model_id)))
        object.__setattr__(self, "region_id", str(canonical_model_id(self.region_id)))
        object.__setattr__(self, "vertex_ids", tuple(self.vertex_ids))
        object.__setattr__(self, "face_refs", tuple(self.face_refs))
        object.__setattr__(self, "face_orientations", tuple(self.face_orientations))
        object.__setattr__(self, "face_groups", MappingProxyType({k: tuple(v) for k, v in self.face_groups.items()}))
        object.__setattr__(self, "provenance", FrozenMetadata(self.provenance))
        if type(self.schema_version) is not int or self.schema_version != 1 or self.name is not None and not isinstance(self.name, str):
            raise GeometryError("invalid solid region schema or name")
        if len(self.vertex_ids) < 6 or len(self.vertex_ids) % 2 or len(set(self.vertex_ids)) != len(self.vertex_ids):
            raise GeometryError("solid region requires distinct prism vertices")
        if any(type(i) is not int or i <= 0 for i in self.vertex_ids):
            raise GeometryError("solid vertex IDs must be positive integers")
        if len(self.face_refs) != len(self.vertex_ids)//2+2 or len(self.face_orientations) != len(self.face_refs):
            raise GeometryError("solid face roles do not match its prism topology")
        if len(set(self.face_refs)) != len(self.face_refs) or any(type(v) is not bool for v in self.face_orientations):
            raise GeometryError("solid face roles must be distinct and oriented")
        if any(not isinstance(h, EntityHandle) or h.kind != 'face' or str(h.model_id) != self.model_id for h in self.face_refs):
            raise GeometryError("solid faces must belong to the region owner")
        defaults = dict(zip(("start", "end", *(f"side_{i}" for i in range(len(self.vertex_ids)//2))), self.face_refs))
        for key, refs in self.face_groups.items():
            if not isinstance(key, str) or not key or not refs or len(set(refs)) != len(refs) or any(h not in self.face_refs for h in refs):
                raise GeometryError("solid groups must select distinct whole source faces")
        if any(self.face_groups.get(key) != (ref,) for key, ref in defaults.items()):
            raise GeometryError("solid default face roles cannot be changed")

    def rebound(self, model_id):
        owner = str(model_id)
        refs = tuple(EntityHandle(owner, 'face', h.id) for h in self.face_refs)
        by_id = {h.id: h for h in refs}
        return replace(self, model_id=owner, face_refs=refs,
                       face_groups={k: tuple(by_id[h.id] for h in v) for k, v in self.face_groups.items()})


@dataclass(frozen=True, slots=True)
class SolidRegionSnapshot:
    schema_version: int
    model_id: str
    revision: int
    region_id: str
    units: str
    geometry_class: str
    coordinates: np.ndarray
    face_loops: tuple[tuple[int, ...], ...]
    face_refs: tuple[EntityHandle, ...]
    face_groups: Mapping[str, tuple[EntityHandle, ...]]
    volume: float
    bounds: tuple[tuple[float, ...], tuple[float, ...]]
    source_signature: str
    provenance: FrozenMetadata


def region_snapshot(model, region_id):
    try:
        region = model.solid_regions[region_id]
    except (KeyError, TypeError) as exc:
        raise GeometryError(f"unknown solid region {region_id!r}") from exc
    if region.model_id != str(model.model_id) or model.units != 'm':
        raise GeometryError("solid region owner/units mismatch")
    try:
        xyz = np.array([model.vertices[i].position for i in region.vertex_ids])
    except KeyError as exc:
        raise GeometryError("solid region references a missing vertex") from exc
    n = len(xyz)//2
    expected, expected_loops, volume = _prism(xyz[:n], xyz[n]-xyz[0])
    scale = max(float(np.ptp(xyz, axis=0).max()), np.finfo(float).tiny)
    if np.max(np.abs(xyz-expected)) > 256*np.finfo(float).eps*scale:
        raise GeometryError("solid end profile no longer matches its normal extrusion")
    rows = {v:i for i, v in enumerate(region.vertex_ids)}
    loops = []
    edges = {}
    source_edges = {}
    for handle, forward in zip(region.face_refs, region.face_orientations):
        face = model.faces.get(handle.id)
        if face is None or face.holes or not isinstance(face.surface, Plane) or face.parameterization is not None:
            raise GeometryError("solid requires existing native planar faces without holes")
        if any(not isinstance(model.edges.get(e.edge).curve if e.edge in model.edges else None, Straight) for e in face.loop):
            raise GeometryError("solid boundary requires straight existing edges")
        try:
            loop = tuple(rows[model.oriented_start_vertex(e)] for e in face.loop)
        except (KeyError, ValueError) as exc:
            raise GeometryError("solid face references incompatible topology") from exc
        if not forward:
            loop = tuple(reversed(loop))
        # Cyclic start is immaterial; orientation and role are explicit.
        wanted = expected_loops[len(loops)]
        if len(loop) != len(wanted) or not any(loop == wanted[i:]+wanted[:i] for i in range(len(wanted))):
            raise GeometryError("solid face role/orientation conflicts with prism topology")
        for use in face.loop:
            source_edges.setdefault(use.edge, []).append(use.forward if forward else not use.forward)
        points = xyz[list(loop)]
        support = face.surface
        support_normal = np.cross(support.u_vector, support.v_vector)
        support_normal /= np.linalg.norm(support_normal)
        if np.max(np.abs((points-support.origin) @ support_normal)) > 256*np.finfo(float).eps*scale:
            raise GeometryError("solid face support is stale")
        loops.append(loop)
        for a,b in zip(loop, loop[1:]+loop[:1]):
            edges[(a,b)] = edges.get((a,b), 0)+1
    if any(len(uses) != 2 or uses[0] == uses[1] for uses in source_edges.values()):
        raise GeometryError("solid faces do not share a closed oriented source-edge topology")
    if any(count != 1 or edges.get((b,a)) != 1 for (a,b), count in edges.items()):
        raise GeometryError("solid boundary is not a closed oriented manifold")
    provenance = FrozenMetadata({**region.provenance.to_dict(), "name": region.name})
    payload = {"schema_version":1, "model_id":str(model.model_id), "region_id":region.region_id,
               "units":"m", "geometry_class":"convex_normal_prism", "coordinates":xyz.tolist(),
               "vertex_ids":region.vertex_ids, "face_loops":loops, "face_refs":[h.id for h in region.face_refs],
               "face_orientations":region.face_orientations,
               "source_faces":[{"edges":[[e.edge,e.forward] for e in model.faces[h.id].loop],
                                "support":{"origin":model.faces[h.id].surface.origin.tolist(),
                                           "u_vector":model.faces[h.id].surface.u_vector.tolist(),
                                           "v_vector":model.faces[h.id].surface.v_vector.tolist()}}
                               for h in region.face_refs],
               "face_groups":{k:[h.id for h in v] for k,v in region.face_groups.items()},
               "provenance":provenance.to_dict()}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return SolidRegionSnapshot(1, str(model.model_id), model.revision, region.region_id, 'm',
                               'convex_normal_prism', _readonly(xyz), tuple(loops), region.face_refs,
                               region.face_groups, volume, tuple(tuple(float(x) for x in a) for a in (xyz.min(axis=0), xyz.max(axis=0))),
                               signature, provenance)


def validate_regions(model):
    errors = []
    owned = set()
    for key, region in model.solid_regions.items():
        try:
            if key != region.region_id:
                raise GeometryError("solid region key/identity mismatch")
            if owned.intersection(region.face_refs):
                raise GeometryError("solid regions cannot share boundary faces")
            owned.update(region.face_refs)
            region_snapshot(model, key)
        except GeometryError as exc:
            errors.append(f"solid region {key}: {exc}")
    return tuple(errors)


def add_prism(model, profile, vector, *, offset=0, name=None, region_id=None, provenance=None):
    xyz, loops, _ = _prism(profile, vector, offset)
    if model.units != 'm':
        raise GeometryError("solid regions require SI metres")
    ids = tuple(model.add_point(*p) for p in xyz)
    edges = {}
    refs = []
    for loop in loops:
        oriented = []
        for a,b in zip(loop, loop[1:]+loop[:1]):
            key = tuple(sorted((a,b)))
            if key not in edges:
                edges[key] = model.add_line(ids[key[0]], ids[key[1]])
            oriented.append(OrientedEdge(edges[key], a < b))
        points = xyz[list(loop)]
        support = Plane(points[0], points[1]-points[0], points[-1]-points[0])
        face_id = model.add_face_from_loop(oriented, surface=support)
        refs.append(EntityHandle(model.model_id, 'face', face_id))
    refs = tuple(refs)
    names = ('start', 'end', *(f'side_{i}' for i in range(len(xyz)//2)))
    record = ClosedSolidRegion(str(model.model_id), str(uuid4()) if region_id is None else region_id,
                               ids, refs, (True,)*len(refs), dict(zip(names, ((h,) for h in refs))), name,
                               FrozenMetadata({**({} if provenance is None else dict(provenance)),
                                               "construction":{"profile":np.asarray(profile,dtype=float).tolist(),
                                                               "vector":np.asarray(vector,dtype=float).tolist(),
                                                               "offset":float(offset)}}))
    if record.region_id in model.solid_regions:
        raise GeometryError("solid region identity already exists")
    model._set_solid_region(record.region_id, record)
    return record.region_id


def regenerate_prism(model, region_id, profile, vector, *, offset=0):
    xyz, loops, _ = _prism(profile, vector, offset)
    region = model.solid_regions.get(region_id)
    if region is None:
        raise GeometryError("unknown solid region")
    old = region_snapshot(model, region_id)
    if len(xyz) != len(region.vertex_ids) or loops != old.face_loops:
        raise GeometryError("solid regeneration topology/orientation conflict; explicit rebind required")
    for vertex_id, point in zip(region.vertex_ids, xyz):
        model._put_entity("vertex", replace(model.vertices[vertex_id], position=point))
    construction = {"profile":np.asarray(profile,dtype=float).tolist(),
                    "vector":np.asarray(vector,dtype=float).tolist(), "offset":float(offset)}
    updated = replace(region, provenance=FrozenMetadata({**region.provenance.to_dict(), "construction":construction}))
    if updated != region:
        model._set_solid_region(region_id, updated)
    for handle, loop in zip(region.face_refs, loops):
        points = xyz[list(loop)]
        support = Plane(points[0], points[1]-points[0], points[-1]-points[0])
        model._put_entity("face", replace(model.faces[handle.id], surface=support))
    return region_id


def region_document(region):
    return {"schema_version":1, "model_id":region.model_id, "region_id":region.region_id,
            "vertex_ids":list(region.vertex_ids), "face_refs":[h.id for h in region.face_refs],
            "face_orientations":list(region.face_orientations),
            "face_groups":{k:[h.id for h in v] for k,v in region.face_groups.items()},
            "name":region.name, "provenance":region.provenance.to_dict()}


def decode_regions(model, records):
    if not isinstance(records, list):
        raise GeometryError("solid_regions must be an array")
    required = {'schema_version','model_id','region_id','vertex_ids','face_refs','face_orientations','face_groups','name','provenance'}
    for item in records:
        if not isinstance(item, dict) or set(item) != required:
            raise GeometryError("solid region record has unexpected fields")
        if not isinstance(item['face_groups'], dict) or not isinstance(item['provenance'], dict):
            raise GeometryError("solid region groups/provenance must be objects")
        refs = tuple(EntityHandle(item['model_id'], 'face', i) for i in item['face_refs'])
        region = ClosedSolidRegion(item['model_id'], item['region_id'], tuple(item['vertex_ids']), refs,
                                   tuple(item['face_orientations']),
                                   {k:tuple(EntityHandle(item['model_id'], 'face', i) for i in v) for k,v in item['face_groups'].items()},
                                   item['name'], FrozenMetadata(item['provenance']), item['schema_version'])
        if region.region_id in model.solid_regions:
            raise GeometryError("duplicate solid region identity")
        model._solid_regions[region.region_id] = region
