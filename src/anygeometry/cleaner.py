"""Conservative, non-mutating geometry review and explicitly selected repairs.

This is a quality review, not a strict intersection/validity audit. Coordinates,
lengths and areas use the owner's SI values. No finding implies design intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from weakref import WeakKeyDictionary, ref

import numpy as np

from .curves import Straight
from .entities import EntityRef
from .errors import GeometryError
from .model import GeometryModel
from .surfaces import CoonsSurface, Plane


@dataclass(frozen=True)
class CleanerOptions:
    short_edge: float = 1e-3
    small_face: float = 1e-6
    near_vertex: float = 1e-5
    sliver_quality: float = 0.01
    max_vertex_pairs: int = 10000

    def __post_init__(self):
        for name in ('short_edge', 'small_face', 'near_vertex', 'sliver_quality'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.sliver_quality > 1:
            raise ValueError('sliver_quality must be at most 1')
        if type(self.max_vertex_pairs) is not int or self.max_vertex_pairs < 1:
            raise ValueError('max_vertex_pairs must be a positive integer')


@dataclass(frozen=True)
class CleanerFinding:
    id: str
    kind: str
    refs: tuple[EntityRef, ...]
    message: str
    value: float | None = None
    repairable: bool = False


@dataclass(frozen=True, eq=False)
class CleanerReport:
    model_id: str
    revision: int
    options: CleanerOptions
    findings: tuple[CleanerFinding, ...]
    coverage: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class RepairChange:
    ref: EntityRef
    operation: str
    before: tuple[float, float, float] | None
    after: tuple[float, float, float] | None


@dataclass(frozen=True, eq=False)
class RepairPlan:
    model_id: str
    revision: int
    changes: tuple[RepairChange, ...]
    dependencies: tuple[EntityRef, ...] = ()

    @property
    def affected_refs(self):
        return tuple(sorted({change.ref for change in self.changes} | set(self.dependencies),
                            key=lambda item: (item.kind, item.id)))

    @property
    def is_noop(self):
        return not self.changes


# Issued objects are capabilities for one live model state, not serialization.
# Copies, reconstructed dataclasses and modified fields cannot authorize edits.
_reviews = WeakKeyDictionary()
_plans = WeakKeyDictionary()


def _require_coordinate_contract(model):
    if model.units != 'm':
        raise GeometryError('Geometry cleaner currently requires metre (m) document coordinates; convert units explicitly before review.')
    transform = model.coordinate_transform
    if transform is not None and not np.array_equal(transform, np.eye(4)):
        raise GeometryError('Geometry cleaner currently requires an absent or identity coordinate transform; transformed coordinates are unsupported.')


def _position(model, vertex):
    return tuple(float(x) for x in model.vertex_position(vertex))


def _protected_refs(model):
    protected = {entity for entity, tags in model.tags.items() if tags}
    for group in model.groups.values():
        protected.update(group)
    protected.update(EntityRef('vertex', attachment.target_id)
                     for attachment in model.attachments.values()
                     if attachment.target_kind == 'vertex')
    return protected


def _unused_vertex(model, vertex, protected=None):
    if protected is None:
        protected = _protected_refs(model)
    return (not model.edges_using_vertex(vertex)
            and vertex not in model.construction_vertices
            and EntityRef('vertex', vertex) not in protected)

def _polygon_data(model, face):
    """Exact polygon area/perimeter for supported planar straight boundaries."""
    loops = (face.loop,) + face.holes
    planar_support = (face.surface is None or isinstance(face.surface, Plane)
                      or (isinstance(face.surface, CoonsSurface) and not face.surface.has_boundaries))
    if not planar_support:
        return None
    if any(not isinstance(model.edges[item.edge].curve, Straight) for loop in loops for item in loop):
        return None
    polygons = [np.array([model.vertex_position(model.edges[item.edge].start if item.forward
                     else model.edges[item.edge].end) for item in loop]) for loop in loops]
    outer = polygons[0]
    centered = outer - outer[0]
    normal = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 2 * model.tolerance.area:
        return None
    normal /= norm
    if any(np.max(np.abs((points - outer[0]) @ normal)) > model.tolerance.length for points in polygons):
        return None
    areas = [abs(float(np.sum(np.cross(points - outer[0], np.roll(points, -1, axis=0) - outer[0]), axis=0) @ normal)) / 2
             for points in polygons]
    area = areas[0] - sum(areas[1:])
    perimeter = sum(float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))) for points in polygons)
    if not math.isfinite(area) or not math.isfinite(perimeter) or area <= 0 or perimeter <= 0:
        return None
    return area, perimeter, polygons


def _canonical_loop(points):
    items = tuple(tuple(float(x) for x in p) for p in points)
    # Canonical cyclic orientation without an O(n^2) rotation list. Unique
    # polygon vertices make the lexicographic minimum anchor unambiguous.
    forward = items.index(min(items))
    reversed_items = tuple(reversed(items))
    backward = reversed_items.index(min(reversed_items))
    return min(items[forward:] + items[:forward], reversed_items[backward:] + reversed_items[:backward])


def review_geometry(model: GeometryModel, options: CleanerOptions | None = None) -> CleanerReport:
    _require_coordinate_contract(model)
    options = CleanerOptions() if options is None else options
    if not isinstance(options, CleanerOptions):
        raise TypeError('options must be CleanerOptions')
    findings = []
    complete = True
    coverage = ['Straight-edge lengths; exact planar polygon area and compactness including holes.',
                'Exact-coordinate duplicates; nearby vertices; endpoint-connected regions; isolated vertices.',
                'No intersection, overlap, near-face, near-region or intended-connectivity audit.']

    def add(kind, entities, message, value=None, repairable=False):
        entities = tuple(sorted(entities, key=lambda r: (r.kind, r.id)))
        identifier = kind + ':' + ','.join(f'{r.kind}{r.id}' for r in entities)
        findings.append(CleanerFinding(identifier, kind, entities, message, value, repairable))

    positions = {v: _position(model, v) for v in sorted(model.vertices)}
    exact = {}
    protected = _protected_refs(model)
    for vertex, position in positions.items():
        exact.setdefault(position, []).append(vertex)
        if _unused_vertex(model, vertex, protected):
            add('unused_vertex', (EntityRef('vertex', vertex),),
                'Isolated unannotated point; remove only if it is not intended construction geometry.', repairable=True)
    for vertices in exact.values():
        if len(vertices) > 1:
            add('duplicate_vertices', (EntityRef('vertex', v) for v in vertices),
                'Distinct point identities have exactly equal coordinates. Review intent; no automatic welding.', 0.)

    # A cell hash avoids all-pairs work on spatially separated points. Duplicate
    # coordinates are grouped above, avoiding quadratic duplicate expansion.
    buckets = {}
    pairs = 0
    truncated = False
    for position, vertices in sorted(exact.items()):
        quotients = [value / options.near_vertex for value in position]
        if not all(math.isfinite(value) for value in quotients):
            complete = False
            coverage.append('Nearby-point check skipped coordinates whose cell index overflows.')
            continue
        cell = tuple(math.floor(value) for value in quotients)
        for offset in product((-1, 0, 1), repeat=3):
            for previous, previous_vertices in buckets.get(tuple(a + b for a, b in zip(cell, offset)), ()):
                distance = math.dist(position, previous)
                if distance <= options.near_vertex:
                    if pairs >= options.max_vertex_pairs:
                        truncated = True
                        break
                    add('near_vertices', tuple(EntityRef('vertex', v) for v in (*previous_vertices, *vertices)),
                        'Nearby distinct coordinates. Review dimensions and intent before moving; never silently stitch.', distance)
                    pairs += 1
            if truncated:
                break
        if truncated:
            break
        buckets.setdefault(cell, []).append((position, vertices))
    if truncated:
        complete = False
        coverage.append(f'Nearby-point findings truncated at {options.max_vertex_pairs} coordinate pairs.')

    duplicates = {}
    for edge_id, edge in sorted(model.edges.items()):
        entity = EntityRef('edge', edge_id)
        if isinstance(edge.curve, Straight):
            length = math.dist(positions[edge.start], positions[edge.end])
            if length <= options.short_edge:
                add('short_edge', (entity,), 'Short straight edge. Review structural intent; do not delete blindly.', length)
            duplicates.setdefault(tuple(sorted((positions[edge.start], positions[edge.end]))), []).append(entity)
        else:
            complete = False
            add('unsupported_edge', (entity,), 'Curved edge length and geometric duplication are outside this review coverage.')
        if not model.faces_using_edge(edge_id) and not model.members_using_edge(edge_id):
            add('unowned_edge', (entity,), 'Edge has no face/member owner. It may be intentional standalone geometry; no removal proposed.')
    for entities in duplicates.values():
        if len(entities) > 1:
            add('duplicate_edges', entities, 'Coincident straight edges with distinct identity. Attributes and connectivity require manual review.')

    duplicate_faces = {}
    for face_id, face in sorted(model.faces.items()):
        entity = EntityRef('face', face_id)
        data = _polygon_data(model, face)
        if data is None:
            complete = False
            add('unsupported_face', (entity,), 'Curved, nonplanar or degenerate face: area/sliver/duplicate checks not performed.')
            continue
        area, perimeter, polygons = data
        quality = min(1., 4 * math.pi * area / perimeter ** 2)
        if area <= options.small_face:
            add('small_face', (entity,), 'Small planar face. Review local model intent; no automatic deletion.', area)
        if quality <= options.sliver_quality:
            add('sliver_face', (entity,), 'Low polygon compactness (4*pi*area/perimeter^2). Review thin/sliver intent.', quality)
        key = (_canonical_loop(polygons[0]), tuple(sorted(_canonical_loop(p) for p in polygons[1:])))
        duplicate_faces.setdefault(key, []).append(entity)
    for entities in duplicate_faces.values():
        if len(entities) > 1:
            add('duplicate_faces', entities, 'Coincident planar polygon boundaries. Review owners/attributes; no deletion proposed.')

    # Connectivity means shared topology IDs, not proximity or geometric crossing.
    parent = {}
    def root(vertex):
        parent.setdefault(vertex, vertex)
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex
    def join(first, second):
        a, b = root(first), root(second)
        parent[max(a, b)] = min(a, b)
    for edge in model.edges.values():
        join(edge.start, edge.end)
    for face in model.faces.values():
        vertices = [model.edges[item.edge].start for loop in (face.loop,) + face.holes for item in loop]
        for vertex in vertices[1:]:
            join(vertices[0], vertex)
    components = {}
    for vertex in parent:
        components.setdefault(root(vertex), []).append(EntityRef('vertex', vertex))
    for edge in model.edges.values():
        components[root(edge.start)].append(edge.ref)
    for face in model.faces.values():
        components[root(model.edges[face.loop[0].edge].start)].append(face.ref)
    if len(components) > 1:
        for entities in components.values():
            add('disconnected_region', entities, 'Topologically separate region. Separation may be intentional; no connection inferred.')

    report = CleanerReport(str(model.model_id), model.revision, options,
                           tuple(sorted(findings, key=lambda item: item.id)), tuple(coverage), complete)
    _reviews[report] = (ref(model), report.model_id, report.revision, report.options, report.findings)
    return report


def _validate_report(model, report):
    _require_coordinate_contract(model)
    binding = _reviews.get(report)
    if (binding is None or binding[0]() is not model or binding[1:] !=
            (str(model.model_id), model.revision, report.options, report.findings)
            or report.model_id != str(model.model_id) or report.revision != model.revision):
        raise GeometryError('Review is stale, forged or belongs to another model; run review again.')


def validate_repair_plan(model: GeometryModel, plan: RepairPlan) -> None:
    _require_coordinate_contract(model)
    binding = _plans.get(plan)
    if (binding is None or binding[0]() is not model or binding[1:] !=
            (str(model.model_id), model.revision, plan.changes, plan.dependencies)
            or plan.model_id != str(model.model_id) or plan.revision != model.revision):
        raise GeometryError('Repair preview is stale, forged or belongs to another model; preview again.')


def _execute(model, changes):
    with model.transaction():
        for change in changes:
            if change.operation == 'remove_unused_vertex':
                if not _unused_vertex(model, change.ref.id):
                    raise GeometryError('Point is no longer unused and unannotated.')
                model.remove_vertex(change.ref.id)
            elif change.operation == 'snap_vertex':
                model.move_point(change.ref.id, *change.after)
            else:
                raise GeometryError('Unsupported cleaner operation.')


def propose_repair(model: GeometryModel, report: CleanerReport, *, finding_ids=(),
                   snap_vertices=(), grid_size=None, max_displacement=None) -> RepairPlan:
    _validate_report(model, report)
    if len(model.features) or model.features.baseline is not None:
        raise GeometryError('Feature-backed geometry requires edits to feature intent; direct cleanup is refused.')
    chosen = set(finding_ids)
    known = {item.id: item for item in report.findings}
    if chosen - known.keys():
        raise GeometryError('Selected finding is not part of this review.')
    changes = []
    removed = set()
    for identifier in sorted(chosen):
        finding = known[identifier]
        if finding.kind != 'unused_vertex' or not finding.repairable:
            raise GeometryError(f'{finding.kind} requires manual geometry/feature editing.')
        vertex = finding.refs[0].id
        if not _unused_vertex(model, vertex):
            raise GeometryError('Point is not unused and unannotated.')
        changes.append(RepairChange(finding.refs[0], 'remove_unused_vertex', _position(model, vertex), None))
        removed.add(vertex)
    snapped = tuple(sorted(set(snap_vertices)))
    if snapped:
        for name, value in (('grid_size', grid_size), ('max_displacement', max_displacement)):
            if value is None or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite for grid snapping')
        for vertex in snapped:
            if type(vertex) is not int or vertex not in model.vertices:
                raise GeometryError('Grid snapping requires existing vertex IDs.')
            if vertex in removed:
                raise GeometryError('Cannot remove and snap the same point.')
            edges = model.edges_using_vertex(vertex)
            if any(not isinstance(model.edges[edge].curve, Straight) for edge in edges):
                raise GeometryError('Snapping curve-defining vertices is unsupported.')
            if any(_polygon_data(model, model.faces[face]) is None for edge in edges for face in model.faces_using_edge(edge)):
                raise GeometryError('Snapping vertices on unsupported faces is refused.')
            before = _position(model, vertex)
            scaled = [value / grid_size for value in before]
            if not all(math.isfinite(value) for value in scaled):
                raise GeometryError('Grid coordinate overflow.')
            after = tuple(round(value) * grid_size for value in scaled)
            if not all(math.isfinite(value) for value in after):
                raise GeometryError('Grid coordinate overflow.')
            if math.dist(before, after) > max_displacement:
                raise GeometryError('Grid displacement exceeds the explicitly configured maximum.')
            if before != after:
                changes.append(RepairChange(EntityRef('vertex', vertex), 'snap_vertex', before, after))
    # Exact collisions are refused rather than introducing unwelded coincident
    # vertices. Proximity remains diagnostic; kernel invariants validate edits.
    moved = {change.ref.id: change.after for change in changes if change.operation == 'snap_vertex'}
    coordinates = {}
    for vertex in sorted(model.vertices):
        if vertex in removed:
            continue
        position = moved.get(vertex, _position(model, vertex))
        if position in coordinates and (vertex in moved or coordinates[position] in moved):
            raise GeometryError('Grid snapping would create coincident point identities; welding is unsupported.')
        coordinates[position] = vertex
    dependencies = set()
    for vertex in moved:
        for edge_id in model.edges_using_vertex(vertex):
            dependencies.add(EntityRef('edge', edge_id))
            dependencies.update(EntityRef('face', face_id) for face_id in model.faces_using_edge(edge_id))
    working = model.clone()
    _execute(working, changes)
    if any(_polygon_data(working, working.faces[item.id]) is None
           for item in dependencies if item.kind == 'face'):
        raise GeometryError('Grid snapping would leave the supported planar-face domain.')
    plan = RepairPlan(str(model.model_id), model.revision, tuple(changes),
                      tuple(sorted(dependencies, key=lambda item: (item.kind, item.id))))
    _plans[plan] = (ref(model), plan.model_id, plan.revision, plan.changes, plan.dependencies)
    return plan


def apply_repair(model: GeometryModel, plan: RepairPlan) -> tuple[RepairChange, ...]:
    validate_repair_plan(model, plan)
    # Recheck the detached candidate so a failure never publishes partial edits.
    working = model.clone()
    _execute(working, plan.changes)
    _execute(model, plan.changes)
    return plan.changes



