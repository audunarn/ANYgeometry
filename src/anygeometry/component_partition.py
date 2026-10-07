"""Conservative geometry-owner independence certificates for worker extraction.

Disjoint grown support boxes certify separation; overlap never certifies contact.
Selections must be dependency complete. Refusals carry no parallel certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from uuid import UUID

import numpy as np

from .curves import Arc, Straight, Spline
from .exact_curves import EXACT_CURVES
from .definition_binding import definition_checksum
from .errors import GeometryError
from .identity import EntityHandle
from .serialization import to_dict, _serialized_model_state, _qualified_model_state
from .surfaces import Plane, Cylinder, Cone


def _certified_face_bound(model, face_id, edges):
    """Enclose the complete material, including legal support extrapolation.

    Linear functionals on a bounded material domain attain extrema on its
    boundary. This bounds planar coordinates and axial coordinates exactly by
    conservative curve boxes. Angular extent uses a *full* radial revolution.
    Normalized nonanalytic patches have no complete trim-containment proof in
    this API, so their otherwise useful normalized patch bounds cannot certify
    independent extrapolated material and are explicitly refused.
    """
    support = model.faces[face_id].support_surface
    if type(support) not in (Plane, Cylinder, Cone):
        raise GeometryError("face support family lacks certified full trim-domain conservative bounds")
    boundary = model.bounds(tuple(('edge', e) for e in sorted(edges)))
    if boundary is None:
        raise GeometryError("face has no conservative trim bound")
    lower, upper = np.asarray(boundary[:3]), np.asarray(boundary[3:])
    if type(support) is Plane:
        return boundary
    axis = support.axis
    offset_low, offset_high = lower-support.origin, upper-support.origin
    axial_low = float(np.minimum(axis*offset_low, axis*offset_high).sum())
    axial_high = float(np.maximum(axis*offset_low, axis*offset_high).sum())
    # Outward accounting for subtract/product/sum rounding, rather than an
    # altered scientific tolerance or an approximate angular sampling oracle.
    error = 64*np.finfo(float).eps*max(1.0, float(np.abs(offset_low).sum()), float(np.abs(offset_high).sum()))
    axial_low -= error
    axial_high += error
    endpoints = support.origin + np.asarray((axial_low, axial_high))[:, None]*axis
    radius = support.radius if type(support) is Cylinder else max(abs(support.radius_start +
        (support.radius_end-support.radius_start)*value/support.height) for value in (axial_low, axial_high))
    extent = radius*np.hypot(support.radial_direction, support.circumferential_direction)
    margin = 64*np.finfo(float).eps*np.maximum(1.0, np.max(np.abs(endpoints), axis=0)+extent)
    return (*np.nextafter(endpoints.min(axis=0)-extent-margin, -np.inf),
            *np.nextafter(endpoints.max(axis=0)+extent+margin, np.inf))


@dataclass(frozen=True, slots=True)
class IndependentComponent:
    """Selected root handles, suitable for ``extract_model_closure``."""
    handles: tuple[EntityHandle, ...]

    @property
    def face_ids(self):
        return tuple(h.id for h in self.handles if h.kind == "face")

    @property
    def member_ids(self):
        return tuple(h.id for h in self.handles if h.kind == "member")

    @property
    def edge_ids(self):
        return tuple(h.id for h in self.handles if h.kind == "edge")


@dataclass(frozen=True, slots=True)
class ComponentMergeReason:
    first: EntityHandle
    second: EntityHandle
    reason: str
    relation: EntityHandle | None = None


@dataclass(frozen=True, slots=True)
class ComponentRefusal:
    reason: str
    entities: tuple[EntityHandle, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentPartition:
    model_id: UUID
    revision: int
    source_checksum: str
    selection: tuple[EntityHandle, ...]
    separation: float
    components: tuple[IndependentComponent, ...]
    merge_reasons: tuple[ComponentMergeReason, ...]
    refusals: tuple[ComponentRefusal, ...]

    @property
    def certified(self):
        """Whether the selected roots admit this conservative partition."""
        return not self.refusals


def _ids(model, kind, values):
    table = getattr(model, {"face": "faces", "member": "members", "edge": "edges"}[kind])
    if values is None:
        return set(table)
    result = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise GeometryError(f"component selection requires positive {kind} IDs")
        model.handle(kind, int(value))
        result.add(int(value))
    return result


def plan_independent_components(model, *, face_ids=None, member_ids=None,
                                edge_ids=(), separation=0.0, expected_revision=None,
                                cancellation_check=None):
    """Certify independent selected components without modifying the model.

    ``separation`` grows *each* support box in every direction. Overlap, shared
    topology, complete sheets and every declared relationship merge roots.
    Partial sheets, omitted dependent owners, unknown supports and eccentric
    definitions without a geometry-owned offset contract refuse the partition.
    Refusal returns an empty ``components`` tuple: callers must use their serial
    route, never treat it as an empty workload. No contact/noncontact claim is
    made for roots merged by boxes.
    """
    if isinstance(separation, bool):
        raise GeometryError("component separation must be finite and nonnegative")
    try:
        separation = float(separation)
    except (TypeError, ValueError) as error:
        raise GeometryError("component separation must be finite and nonnegative") from error
    if not isfinite(separation) or separation < 0:
        raise GeometryError("component separation must be finite and nonnegative")
    revision = model.revision
    if expected_revision is not None and expected_revision != revision:
        raise GeometryError("component partition revision is stale")
    def check():
        if cancellation_check is not None and cancellation_check("component partition"):
            raise GeometryError("component partition cancelled")
    checksum = to_dict(model)["checksum"]["value"]
    check()
    selected = {*(('face', i) for i in _ids(model, 'face', face_ids)),
                *(('member', i) for i in _ids(model, 'member', member_ids)),
                *(('edge', i) for i in _ids(model, 'edge', edge_ids))}
    # Include unselected face/member owners in the dependency graph. Otherwise
    # one omitted owner can connect apparently independent selected roots.
    units = tuple(sorted({*(('face', i) for i in model.faces),
                          *(('member', i) for i in model.members), *selected}))
    index = {key: i for i, key in enumerate(units)}
    parents = list(range(len(units)))
    reasons = []
    refusals = []
    def handle(key):
        return model.handle(*key)
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def join(keys, reason, relation=None):
        keys = sorted(set(keys))
        if not keys:
            return
        first = keys[0]
        for other in keys[1:]:
            a, b = find(index[first]), find(index[other])
            if a != b:
                parents[max(a, b)] = min(a, b)
                reasons.append(ComponentMergeReason(handle(first), handle(other), reason, relation))
    owners = {}
    vertices = {}
    root_edges = {}
    root_geometry = {}
    for key in units:
        kind, identifier = key
        if kind == "face":
            face = model.faces[identifier]
            edges = {item.edge for loop in (face.loop, *face.holes) for item in loop}
        elif kind == "member":
            edges = {model.member_edge_uses[i].edge_id for i in model.members[identifier].edge_use_ids}
        else:
            edges = {identifier}
        root_edges[key] = edges
        root_geometry[key] = ({key} if kind == "face" else set()) | {('edge', e) for e in edges}
        for edge_id in sorted(edges):
            owners.setdefault(('edge', edge_id), set()).add(key)
            edge = model.edges[edge_id]
            controls = ((edge.curve.via_vertex,) if isinstance(edge.curve, Arc) else
                        edge.curve.control_vertices if isinstance(edge.curve, Spline) else ())
            for vertex in (edge.start, edge.end, *controls):
                vertices.setdefault(vertex, set()).add(key)
    for edge_key, keys in sorted(owners.items()):
        join(keys, "shared edge", handle(edge_key))
    for vertex, keys in sorted(vertices.items()):
        owners[('vertex', vertex)] = keys
        join(keys, "shared vertex/control dependency", handle(('vertex', vertex)))
    for key in units:
        owners.setdefault(key, set()).add(key)
    for sheet_id, sheet in sorted(model.sheets.items()):
        keys = {('face', model.face_uses[i].face_id) for i in sheet.face_use_ids}
        owners[('sheet', sheet_id)] = keys
        join(keys, "sheet membership", handle(('sheet', sheet_id)))
        if keys & selected and not keys <= selected:
            refusals.append(ComponentRefusal("partial sheet selection", (handle(('sheet', sheet_id)),)))
    for identifier, use in model.face_uses.items():
        owners[('face_use', identifier)] = {('face', use.face_id)}
    for identifier, coedge in model.coedges.items():
        owners[('coedge', identifier)] = owners[('face_use', coedge.face_use_id)]
    for identifier, use in model.member_edge_uses.items():
        owners[('member_edge_use', identifier)] = {('member', use.member_id)}
    for identifier, part in model.parts.items():
        owners[('part', identifier)] = {('member', i) for i in part.member_ids}
        for sheet_id in part.sheet_ids:
            owners[('part', identifier)].update(owners[('sheet', sheet_id)])
    def referenced(key):
        return owners.get(key, set())

    expanded_pairs = set()
    def claim(keys, dependency):
        """Register every consumer of a copied dependency, even an orphan one."""
        kind, identifier = dependency
        bucket = owners.setdefault(dependency, set())
        before = len(bucket)
        bucket.update(keys)
        changed = len(bucket) != before
        if kind not in ("vertex", "edge", "face"):
            return changed
        # Owner buckets predate this closure walk, so already-expanded
        # (consumer, dependency) pairs are tracked separately. Only fresh
        # pairs recurse through the face/edge/vertex closure; a repeated
        # pair would only repeat idempotent set insertions.
        fresh = []
        for key in keys:
            before = len(root_geometry[key])
            root_geometry[key].add(dependency)
            changed |= len(root_geometry[key]) != before
            pair = (key, dependency)
            if pair not in expanded_pairs:
                expanded_pairs.add(pair)
                fresh.append(key)
        if not fresh:
            return changed
        if kind == "face":
            face = model.faces[identifier]
            for loop in (face.loop, *face.holes):
                for item in loop:
                    changed |= claim(fresh, ('edge', item.edge))
        elif kind == "edge":
            edge = model.edges[identifier]
            controls = ((edge.curve.via_vertex,) if isinstance(edge.curve, Arc) else
                        edge.curve.control_vertices if isinstance(edge.curve, Spline) else ())
            for vertex in (edge.start, edge.end, *controls):
                changed |= claim(fresh, ('vertex', vertex))
        return changed

    for member_id, member in sorted(model.members.items()):
        if member.orientation_reference is not None:
            claim({('member', member_id)}, member.orientation_reference)
            join({('member', member_id), *referenced(member.orientation_reference)},
                 "member orientation dependency", handle(('member', member_id)))
    attachment_keys = {}
    for identifier, attachment in sorted(model.attachments.items()):
        keys = {*referenced(attachment.source_key), *referenced(attachment.target_key)}
        if attachment.member_id is not None:
            keys.update(referenced(('member', attachment.member_id)))
        if attachment.sheet_id is not None:
            keys.update(referenced(('sheet', attachment.sheet_id)))
        for parent in attachment.lineage:
            if model._contains_entity(*parent):
                keys.update(referenced(parent))
        attachment_keys[identifier] = keys
        owners[('attachment', identifier)] = keys
        join(keys, "declared attachment (all connection intents)", handle(('attachment', identifier)))
    for identifier, junction in sorted(model.junctions.items()):
        keys = {('member', i) for i in junction.member_ids}
        for sheet_id in junction.sheet_ids:
            keys.update(referenced(('sheet', sheet_id)))
        for attachment_id in junction.attachment_ids:
            keys.update(attachment_keys[attachment_id])
        owners[('junction', identifier)] = keys
        join(keys, "declared junction", handle(('junction', identifier)))
    # Active lineage may name later structural relations or a complete Part.
    # Resolve their transitive owners before claiming extraction independence.
    while True:
        changed = False
        for identifier, attachment in sorted(model.attachments.items()):
            keys = attachment_keys[identifier]
            before = len(keys)
            dependencies = [attachment.source_key, attachment.target_key]
            if attachment.sheet_id is not None:
                dependencies.append(('sheet', attachment.sheet_id))
            dependencies.extend(parent for parent in attachment.lineage if model._contains_entity(*parent))
            for dependency in dependencies:
                keys.update(referenced(dependency))
            changed |= len(keys) != before
            # Freeze the consumer set while registering it; dependency buckets
            # may be this very set when lineage references another attachment.
            consumers = set(keys)
            for dependency in dependencies:
                changed |= claim(consumers, dependency)
            join(keys, "active attachment lineage", handle(('attachment', identifier)))
        for identifier, junction in sorted(model.junctions.items()):
            keys = owners[('junction', identifier)]
            before = len(keys)
            for attachment_id in junction.attachment_ids:
                keys.update(attachment_keys[attachment_id])
            changed |= len(keys) != before
            join(keys, "declared junction", handle(('junction', identifier)))
        if not changed:
            break
    for dependency, keys in sorted(owners.items()):
        if dependency[0] in ("vertex", "edge", "face"):
            join(keys, "shared extraction dependency", handle(dependency))
    # Metadata has no authoritative eccentric-offset schema in this version.
    # Refuse such declarations rather than guessing axis/radius conventions.
    for kind, table in (("member", model.members), ("member_edge_use", model.member_edge_uses),
                        ("attachment", model.attachments)):
        for identifier, entity in sorted(table.items()):
            metadata = entity.metadata
            if selected and any("eccentr" in str(k).lower() or "offset" in str(k).lower() for k in metadata):
                # An unselected owner's unknown physical reach can interact
                # with selected material despite disjoint root-axis boxes.
                refusals.append(ComponentRefusal("eccentric/offset metadata has no certified extent contract",
                                                 (handle((kind, identifier)),)))
    # Certified bounds are pure functions of the unmutated model within this
    # one invocation, so each unique dependency is certified once and reused
    # for every consumer unit. A cached failure re-raises per consumer so
    # refusal evidence still names each affected unit.
    dependency_bounds = {}
    def certified_dependency_bound(dependency_kind, dependency_id):
        cache_key = (dependency_kind, dependency_id)
        if cache_key in dependency_bounds:
            cached = dependency_bounds[cache_key]
            if isinstance(cached, BaseException):
                raise cached
            return cached
        try:
            if dependency_kind == "face":
                face = model.faces[dependency_id]
                edges = {item.edge for loop in (face.loop, *face.holes) for item in loop}
                bound = _certified_face_bound(model, dependency_id, edges)
            else:
                if dependency_kind == "edge" and type(model.edges[dependency_id].curve) not in (Straight, Arc, Spline, *EXACT_CURVES):
                    raise GeometryError("edge curve family has no certified conservative bound")
                bound = model.bounds(((dependency_kind, dependency_id),))
        except (GeometryError, TypeError, ValueError) as error:
            dependency_bounds[cache_key] = error
            raise
        dependency_bounds[cache_key] = bound
        return bound
    boxes = {}
    for key in units:
        check()
        kind, identifier = key
        try:
            dependencies = root_geometry[key]
            enclosures = []
            for dependency_kind, dependency_id in sorted(dependencies):
                enclosures.append(certified_dependency_bound(dependency_kind, dependency_id))
            if any(box is None for box in enclosures) or not enclosures:
                raise GeometryError("missing conservative dependency bound")
            values = np.asarray(enclosures, dtype=float)
            box = (*values[:, :3].min(axis=0), *values[:, 3:].max(axis=0))
            if box is None or len(box) != 6 or not np.all(np.isfinite(box)) or any(box[i] > box[i+3] for i in range(3)):
                raise GeometryError("missing or invalid conservative bound")
            boxes[key] = tuple(float(x) for x in box)
        except (GeometryError, TypeError, ValueError) as error:
            # An unbounded unselected support can interact with anything selected.
            if selected:
                refusals.append(ComponentRefusal(str(error), (handle(key),)))
    # Include the model's existing length tolerance. Distant global extents may
    # overmerge, which is safe; no geometric tolerance is weakened here.
    scene = np.asarray(tuple(boxes.values()), dtype=float)
    length = 0.0 if not len(scene) else float(np.linalg.norm(scene[:, 3:].max(axis=0) - scene[:, :3].min(axis=0)))
    padding = separation + model.tolerance.effective_length(length)
    active = []
    for key in sorted(boxes, key=lambda item: (boxes[item][0], item)):
        box = boxes[key]
        active = [other for other in active if boxes[other][3] + 2*padding >= box[0]]
        for other in active:
            second = boxes[other]
            if all(box[d] <= second[d+3] + 2*padding and second[d] <= box[d+3] + 2*padding
                   for d in (1, 2)):
                join((key, other), "separation-grown conservative bounds overlap")
        active.append(key)
    groups = {}
    for key in units:
        groups.setdefault(find(index[key]), set()).add(key)
    components = []
    for keys in groups.values():
        chosen = keys & selected
        if not chosen:
            continue
        omitted = keys - selected
        if omitted:
            refusals.append(ComponentRefusal("selection omits interacting/dependent owners",
                                             tuple(handle(key) for key in sorted(omitted))))
        components.append(IndependentComponent(tuple(handle(key) for key in sorted(chosen))))
    check()
    if revision != model.revision or checksum != _qualified_model_state(model)["checksum"]["value"]:
        raise GeometryError("geometry changed during component partition")
    return ComponentPartition(model.model_id, revision, checksum,
        tuple(handle(key) for key in sorted(selected)), separation,
        () if refusals else tuple(sorted(components, key=lambda item: item.handles)),
        tuple(reasons), tuple(refusals))


def validate_component_partition_binding(model, partition, *, expected_revision=None,
                                         cancellation_check=None):
    """Reject stale, raw-mutated or tampered certificates before worker dispatch."""
    if not isinstance(partition, ComponentPartition):
        raise TypeError("component binding requires a ComponentPartition")
    if model.model_id != partition.model_id:
        raise GeometryError("component partition belongs to another model")
    if model.revision != partition.revision or (expected_revision is not None and expected_revision != model.revision):
        raise GeometryError("component partition is stale")
    if _serialized_model_state(model)["checksum"]["value"] != partition.source_checksum:
        raise GeometryError("component partition source binding changed")
    expected = plan_independent_components(model,
        face_ids=(h.id for h in partition.selection if h.kind == "face"),
        member_ids=(h.id for h in partition.selection if h.kind == "member"),
        edge_ids=(h.id for h in partition.selection if h.kind == "edge"),
        separation=partition.separation, cancellation_check=cancellation_check)
    if definition_checksum(expected) != definition_checksum(partition):
        raise GeometryError("component partition evidence binding changed")
