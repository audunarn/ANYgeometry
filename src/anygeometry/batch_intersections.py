"""Atomic intersection preparation over original operand material domains.

Planning completes every narrow phase and arrangement before mutation begins.
Application journals all edge splits, canonicalization and face replacements in
one transaction. A resource refusal never returns partially accepted topology.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
from uuid import UUID

import numpy as np

from .analytic_supports import plane_cylinder_support, cylinder_cylinder_support
from .arrangement_geometry import (LinePath, BezierPath, freeze_edge,
                                   point_parameters, curve_junctions)
from .entities import EntityRef, OrientedEdge
from .curves import Straight
from .errors import GeometryError
from .exact_curves import EllipticArc, CylinderIntersectionCurve
from .identity import EntityHandle
from .material_arrangement import (ArrangementPath, ArrangementPoint, ArrangementCell, MaterialDomain,
                                   MaterialArrangement, arrange_material, _clip, _clip_intervals)
from .member_arrangements import (MemberAxisArrangement, MemberPointContact,
                                  plan_member_arrangements)
from .predicates import IntersectionDimension, IntersectionKind, qualified_plane_plane
from .serialization import to_dict
from .structural import ConnectionIntent, Orientation
from .surfaces import Plane, Cylinder
from .transactions import ChangeSet
from .definition_binding import definition_checksum


@dataclass(frozen=True, slots=True)
class IntersectionBatchPolicy:
    intent: ConnectionIntent = ConnectionIntent.CONNECT
    max_predicates: int | None = None
    max_candidate_pairs: int | None = None
    cancellation_check: object = None
    face_connections: bool = True
    member_connections: bool = True
    member_face_connections: bool = True

    def __post_init__(self):
        object.__setattr__(self, "intent", ConnectionIntent(self.intent))
        if self.intent not in (ConnectionIntent.CONNECT,ConnectionIntent.IMPRINT,
                               ConnectionIntent.REUSE_EXISTING):
            raise GeometryError("batch preparation requires connect, imprint or reuse_existing intent")
        if self.max_predicates is not None and (type(self.max_predicates) is not int or self.max_predicates < 1):
            raise GeometryError("intersection predicate budget must be a positive integer")
        if self.max_candidate_pairs is not None and (
                type(self.max_candidate_pairs) is not int or self.max_candidate_pairs < 1):
            raise GeometryError("intersection candidate pair budget must be a positive integer")
        if self.cancellation_check is not None and not callable(self.cancellation_check):
            raise GeometryError("intersection cancellation check must be callable")
        for name in ("face_connections", "member_connections", "member_face_connections"):
            if type(getattr(self,name)) is not bool:
                raise GeometryError(f"intersection {name} policy must be boolean")


@dataclass(frozen=True, slots=True)
class FacePointContact:
    face_ids: tuple[int, ...]
    position: tuple[float, float, float]
    world_tolerance: float


@dataclass(frozen=True, slots=True)
class IntersectionPlan:
    model_id: UUID
    revision: int
    source_checksum: str
    operands: tuple[EntityHandle, ...]
    policy: IntersectionBatchPolicy
    arrangements: tuple[MaterialArrangement, ...]
    axes: tuple[MemberAxisArrangement, ...] = ()
    contacts: tuple[MemberPointContact, ...] = ()
    content_checksum: str = ''
    face_contacts: tuple[FacePointContact, ...] = ()


@dataclass(frozen=True, slots=True)
class IntersectionApplication:
    plan: IntersectionPlan
    change_set: ChangeSet
    joint_edges: tuple[EntityHandle, ...]
    reused: bool = False


def _policy(value):
    if isinstance(value, IntersectionBatchPolicy):
        return value
    try:
        return IntersectionBatchPolicy(ConnectionIntent(value))
    except (TypeError, ValueError) as exc:
        raise GeometryError("intersection batch needs an explicit connection policy") from exc


def _plan_content(plan):
    return definition_checksum((plan.operands,plan.policy.intent,
        plan.policy.face_connections,plan.policy.member_connections,plan.policy.member_face_connections,
        plan.arrangements,plan.axes,plan.contacts,plan.face_contacts))


def _validate_policy(plan, policy):
    if any(getattr(policy,name) != getattr(plan.policy,name) for name in (
        'intent','face_connections','member_connections','member_face_connections')):
        raise GeometryError("apply policy does not match intersection planning")
    if plan.content_checksum != _plan_content(plan):
        raise GeometryError("intersection plan analytic content binding changed")


def _normalise_cells(arrangement):
    # Kernel loops have at least three edges. Analytic lenses and circles can
    # have two mathematical edges; split one chart without adding a seam.
    split = set()
    for cell in arrangement.cells:
        for loop in (cell.outer, *cell.holes):
            if len(loop) == 2:
                split.add(loop[0][0])
    if not split:
        return arrangement
    paths, mapping = [], {}
    for index, path in enumerate(arrangement.paths):
        indices = []
        for a, b in (((0., .5), (.5, 1.)) if index in split else ((0., 1.),)):
            indices.append(len(paths))
            paths.append(replace(path, curve=path.curve.subcurve(a, b)))
        mapping[index] = tuple(indices)
    def loop(old):
        result = []
        for index, forward in old:
            result.extend((child, forward) for child in (mapping[index] if forward else mapping[index][::-1]))
        return tuple(result)
    return replace(arrangement, paths=tuple(paths), cells=tuple(
        ArrangementCell(loop(cell.outer), tuple(loop(hole) for hole in cell.holes))
        for cell in arrangement.cells))


def _synchronize_boundary_events(arrangements, check):
    """Propagate decomposition ports across all exact shared boundaries.

    A construction seam is local to one domain, but its endpoint splits the
    physical boundary used by every incident domain. Synchronize those events
    after every original-operand arrangement, before any face is rebuilt.
    """
    from .material_arrangement import _parameters
    by_edge, by_face = {}, {}
    for arrangement in arrangements:
        events=[]
        for path in arrangement.paths:
            endpoints=(path.curve.evaluate(0.),path.curve.evaluate(1.))
            events.extend(endpoints)
            if path.source_edge is not None:
                by_edge.setdefault(path.source_edge,[]).extend(endpoints)
        by_face[arrangement.face_id]=events
    result=[]
    for arrangement in arrangements:
        paths,mapping=[],{}
        for index,path in enumerate(arrangement.paths):
            values={0.,1.}
            events=list(by_edge.get(path.source_edge,()))
            if len(path.owners)>1:
                events.extend(point for owner in path.owners for point in by_face.get(owner,()))
            lo,hi=path.curve.bounds()
            for point in events:
                if np.any(point<lo-arrangement.world_tolerance) or np.any(point>hi+arrangement.world_tolerance):
                    continue
                check()
                values.update(point_parameters(path.curve,point,tolerance=arrangement.world_tolerance))
            parameters=_parameters(values,arrangement.world_tolerance,path.curve)
            children=[]
            for start,end in zip(parameters,parameters[1:]):
                children.append(len(paths))
                paths.append(replace(path,curve=path.curve.subcurve(start,end)))
            mapping[index]=tuple(children)
        def loop(old):
            return tuple((child,forward) for index,forward in old
                         for child in (mapping[index] if forward else mapping[index][::-1]))
        result.append(replace(arrangement,paths=tuple(paths),cells=tuple(
            ArrangementCell(loop(cell.outer),tuple(loop(hole) for hole in cell.holes))
            for cell in arrangement.cells)))
    return tuple(result)


def _domain_bounds(domain):
    boxes = [path.curve.bounds() for loop in domain.boundaries for path in loop]
    return np.min([box[0] for box in boxes], axis=0), np.max([box[1] for box in boxes], axis=0)


def _coplanar_traces(first, second, tolerance, check, model):
    if isinstance(first.support, Plane) and isinstance(second.support, Plane) and all(
        isinstance(path.curve, LinePath) for domain in (first, second)
        for loop in domain.boundaries for path in loop):
        # The established polygon predicate is exact for straight trim
        # segments. Reuse its robust point/interval/area distinction; no
        # sampled curved boundary enters this backend.
        from .intersections import _query_planar_faces
        result = _query_planar_faces(model, model.handle("face", first.face_id),
                                    model.handle("face", second.face_id), first.support, second.support)
        if not result.classified:
            raise GeometryError("; ".join(result.diagnostics))
        if result.dimension is IntersectionDimension.REGION:
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
        if result.dimension is not IntersectionDimension.CURVE:
            return ()
        return tuple(LinePath(component.witnesses[0], component.witnesses[-1])
                     for component in result.components if len(component.witnesses) >= 2)
    # Partition original material by the other domain's complete boundary.
    # Classify cells in both domains; positive-area ownership stays explicit.
    traces = tuple(ArrangementPath(path.curve, owners=(first.face_id, second.face_id))
                   for loop in second.boundaries for path in loop)
    arranged = arrange_material(first, traces, tolerance=tolerance, cancellation_check=lambda: (check() or False))
    for cell in arranged.cells:
        index, forward = cell.outer[0]
        path = arranged.paths[index]
        curve = path.curve if forward else path.curve.subcurve(1., 0.)
        uv = first.uv(curve, .5)
        tangent = first.tangent(curve, .5)
        point = first.support.evaluate(*first.left_probe(uv, tangent, 2*tolerance))
        probe = LinePath(tuple(point), tuple(point))
        if second.contains(probe, .5, tolerance, boundary=False):
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
    # Only common boundary intervals remain. Retain each exact interval in
    # both material domains; point-only contact is not an implicit weld.
    result = []
    for loop in first.boundaries:
        for path in loop:
            for part in _clip(second, path.curve, tolerance, check):
                if any(point_parameters(boundary.curve, part.evaluate(.5), tolerance=tolerance)
                       for boundary_loop in second.boundaries for boundary in boundary_loop):
                    result.append(part)
    return tuple(result)


def _rectangular_cylinder(domain, tolerance):
    """Prove a native rectangle from analytic boundary definitions."""
    if not isinstance(domain.support, Cylinder) or len(domain.boundaries) != 1:
        return False
    support = domain.support
    covered = {side: [] for side in range(4)}
    for path in domain.boundaries[0]:
        curve = path.curve
        start, end = domain.uv(curve, 0.), domain.uv(curve, 1.)
        if isinstance(curve, LinePath):
            direction = np.asarray(curve.end)-curve.start
            if np.linalg.norm(np.cross(direction, support.axis)) > tolerance:
                return False
            coordinate, varying = 0, 1
        elif isinstance(curve, EllipticArc):
            normal = np.cross(curve.u_vector, curve.v_vector)
            normal /= np.linalg.norm(normal)
            if np.linalg.norm(np.cross(normal, support.axis)) > tolerance:
                return False
            center = np.asarray(curve.center)-support.origin
            radial = center-float(center @ support.axis)*support.axis
            if np.linalg.norm(radial) > tolerance or max(abs(np.linalg.norm(vector)-support.radius)
                for vector in (curve.u_vector, curve.v_vector)) > tolerance:
                return False
            coordinate, varying = 1, 0
        else:
            return False
        if max(abs(start[coordinate]), abs(end[coordinate])) <= tolerance:
            side = 2*coordinate
        elif max(abs(start[coordinate]-1), abs(end[coordinate]-1)) <= tolerance:
            side = 2*coordinate+1
        else:
            return False
        covered[side].append(tuple(sorted((start[varying], end[varying]))))
    for intervals in covered.values():
        right = 0.
        for start, end in sorted(intervals):
            if start > right+tolerance:
                return False
            right = max(right, end)
        if abs(right-1) > tolerance:
            return False
    return True


def _coincident_cylinder_rectangles(first, second, model, tolerance):
    """Exact angular/axial interval intersection, including periodic seams."""
    a, b = first.support, second.support
    orientation = 1 if float(a.axis @ b.axis) > 0 else -1
    phase = math.atan2(float(b.radial_direction @ a.circumferential_direction),
                       float(b.radial_direction @ a.radial_direction))
    alo, ahi = sorted((a.start_angle, a.start_angle+a.sweep_angle))
    blo, bhi = sorted((phase+orientation*b.start_angle,
                      phase+orientation*(b.start_angle+b.sweep_angle)))
    second_z = float((b.origin-a.origin) @ a.axis)
    zlo = max(min(0., a.height), min(second_z, second_z+orientation*b.height))
    zhi = min(max(0., a.height), max(second_z, second_z+orientation*b.height))
    if zhi < zlo-tolerance:
        return ()
    angle_tolerance = tolerance/a.radius
    paths = []
    from .intersections import _sheet_ids_for_face
    same_sheet = bool(set(_sheet_ids_for_face(model, first.face_id)) & set(_sheet_ids_for_face(model, second.face_id)))
    for turn in range(math.ceil((alo-bhi-angle_tolerance)/math.tau),
                      math.floor((ahi-blo+angle_tolerance)/math.tau)+1):
        lower, upper = max(alo, blo+turn*math.tau), min(ahi, bhi+turn*math.tau)
        if upper < lower-angle_tolerance:
            continue
        if upper-lower > angle_tolerance and zhi-zlo > tolerance:
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
        if same_sheet:
            # Existing chart boundaries of one source Sheet are decomposition
            # seams. They are already connected by their source topology.
            continue
        if upper-lower <= angle_tolerance and zhi-zlo > tolerance:
            angle = .5*(lower+upper)
            base = a.origin+a.radius*(math.cos(angle)*a.radial_direction+math.sin(angle)*a.circumferential_direction)
            paths.append(LinePath(tuple(base+zlo*a.axis), tuple(base+zhi*a.axis)))
        elif upper-lower > angle_tolerance:
            center = a.origin+.5*(zlo+zhi)*a.axis
            paths.append(EllipticArc(center, a.radius*a.radial_direction, a.radius*a.circumferential_direction,
                                     lower, upper-lower))
    return tuple(paths)


def _pair_paths(first, second, model, tolerance, check, *, point_contacts=None):
    a, b = first.support, second.support
    if isinstance(a, Plane) and isinstance(b, Plane):
        result = qualified_plane_plane(a, b, policy=model.tolerance)
        if not result.classified:
            raise GeometryError("; ".join(result.diagnostics))
        if result.kind is IntersectionKind.DISJOINT:
            return ()
        if result.kind is IntersectionKind.COINCIDENT:
            return _coplanar_traces(first, second, tolerance, check, model)
        component = result.components[0]
        point, direction = np.asarray(component.witnesses[0]), np.asarray(component.direction)
        boxes = (_domain_bounds(first), _domain_bounds(second))
        extent = max(np.linalg.norm(lo-point)+np.linalg.norm(hi-point) for lo, hi in boxes)+1.
        paths = (LinePath(tuple(point-extent*direction), tuple(point+extent*direction)),)
    elif isinstance(a, Plane) and isinstance(b, Cylinder):
        result = plane_cylinder_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    elif isinstance(a, Cylinder) and isinstance(b, Plane):
        result = plane_cylinder_support(b, a, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    else:
        result = cylinder_cylinder_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        if result.coincident:
            if _rectangular_cylinder(first, tolerance) and _rectangular_cylinder(second, tolerance):
                return _coincident_cylinder_rectangles(first, second, model, tolerance)
            return _coplanar_traces(first, second, tolerance, check, model)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    candidates=list(getattr(result,'points',()))
    def clip(domain,curve):
        if point_contacts is None:
            return _clip(domain,curve,tolerance,check)
        return tuple(curve.subcurve(a,b) for a,b in _clip_intervals(
            domain,curve,tolerance,check,isolated_contacts=candidates))
    curves=tuple(part for path in paths for clipped in clip(first,path)
                 for part in clip(second,clipped))
    if point_contacts is not None:
        for point in candidates:
            check()
            station=LinePath(tuple(point),tuple(point))
            if (first.contains(station,0.,tolerance) and second.contains(station,0.,tolerance)
                    and not any(point_parameters(curve,point,tolerance=tolerance) for curve in curves)
                    and not any(np.linalg.norm(np.asarray(point)-old)<=tolerance for old in point_contacts)):
                point_contacts.append(tuple(point))
    return curves


def plan_intersections(model, operands, *, policy):
    """Plan intersections between the complete original operand set, read-only."""
    from .intersections import _normalize_operand
    policy = _policy(policy)
    if policy.intent not in (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT, ConnectionIntent.REUSE_EXISTING):
        raise GeometryError("intersection preparation requires CONNECT, IMPRINT or REUSE_EXISTING intent")
    revision, checksum = model.revision, to_dict(model)["checksum"]["value"]
    handles = tuple(sorted(set(_normalize_operand(model, item) for item in operands)))
    faces, members = set(), set()
    for handle in handles:
        if handle.kind == "face":
            faces.add(handle.id)
        elif handle.kind == "sheet":
            faces.update(model.face_uses[use_id].face_id for use_id in model.sheets[handle.id].face_use_ids)
        elif handle.kind == "member":
            members.add(handle.id)
        else:
            raise GeometryError("intersection batch operands must be faces, sheets or members")
    from .boundary_contacts import bilinear_boundary, plane_boundary_contacts
    domains, boundary_only = {}, {}
    for face_id in sorted(faces):
        try:
            domains[face_id] = MaterialDomain.from_model(model, face_id)
        except GeometryError:
            boundary_only[face_id] = bilinear_boundary(model, face_id)
    traces = {face_id: [] for face_id in faces}
    points = {face_id: [] for face_id in faces}
    face_contacts=[]
    examined = 0
    candidate_pairs = 0
    def check_pair():
        nonlocal candidate_pairs
        check()
        candidate_pairs += 1
        if policy.max_candidate_pairs is not None and candidate_pairs > policy.max_candidate_pairs:
            raise GeometryError(
                f"intersection planning exceeds maximum_candidate_pairs={policy.max_candidate_pairs}")
    def check():
        nonlocal examined
        examined += 1
        if policy.cancellation_check is not None and policy.cancellation_check():
            raise GeometryError("intersection planning cancelled")
        if policy.max_predicates is not None and examined > policy.max_predicates:
            raise GeometryError("intersection planning predicate budget exhausted")
    lengths = {face_id: float(np.linalg.norm(hi-lo)) for face_id, domain in domains.items()
               for lo, hi in (_domain_bounds(domain),)}
    if policy.face_connections:
        for face_id, corners in boundary_only.items():
            lo1, hi1 = np.min(corners, axis=0), np.max(corners, axis=0)
            for other_id, domain in domains.items():
                check()
                lo2, hi2 = _domain_bounds(domain)
                tolerance = model.tolerance.effective_length(max(
                    float(np.linalg.norm(hi1-lo1)), lengths[other_id]))
                if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
                    continue
                check_pair()
                traces[other_id].extend(plane_boundary_contacts(
                    model, face_id, corners, domain, tolerance, check))
        for first, second in combinations(sorted(boundary_only), 2):
            a, b = np.asarray(boundary_only[first]), np.asarray(boundary_only[second])
            tolerance = model.tolerance.effective_length(max(
                float(np.linalg.norm(np.ptp(a, axis=0))), float(np.linalg.norm(np.ptp(b, axis=0)))))
            if np.all(a.max(axis=0) >= b.min(axis=0)-tolerance) and np.all(b.max(axis=0) >= a.min(axis=0)-tolerance):
                check_pair()
                raise GeometryError("intersecting nonplanar Coons supports require a qualified surface predicate")
    if members and policy.member_face_connections and boundary_only:
        # These compatibility surfaces have no general axis/support predicate.
        # Never silently omit a requested beam/shell relation.
        raise GeometryError("member intersections with nonplanar Coons supports are unsupported")
    for first_id, second_id in combinations(sorted(domains), 2):
        if not policy.face_connections:
            break
        check()
        first, second = domains[first_id], domains[second_id]
        tolerance = model.tolerance.effective_length(max(lengths[first_id], lengths[second_id]))
        lo1, hi1 = _domain_bounds(first); lo2, hi2 = _domain_bounds(second)
        if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
            continue
        check_pair()
        try:
            pair_points=[]
            pair_paths = _pair_paths(first, second, model, tolerance, check,point_contacts=pair_points)
        except GeometryError as exc:
            raise GeometryError(f"intersection batch face:{first_id}/face:{second_id}: {exc}") from exc
        for curve in pair_paths:
            path = ArrangementPath(curve, owners=(first_id, second_id))
            traces[first_id].append(path); traces[second_id].append(path)
        for point in pair_points:
            face_contacts.append(FacePointContact((first_id,second_id),point,tolerance))
            for face_id in (first_id,second_id):
                points[face_id].append(ArrangementPoint(point))
    axes, contacts = plan_member_arrangements(model, members, domains, traces, points, check,
        include_axis_axis=policy.member_connections, include_axis_face=policy.member_face_connections,
        check_pair=check_pair)
    arrangements = []
    for face_id, domain in domains.items():
        if not traces[face_id] and not points[face_id]:
            continue
        try:
            arrangement = arrange_material(domain, traces[face_id],
                tolerance=model.tolerance.effective_length(lengths[face_id]),
                area_tolerance=model.tolerance.effective_area(lengths[face_id]),
                points=points[face_id],
                cancellation_check=lambda: (check() or False))
        except GeometryError as exc:
            raise GeometryError(f"intersection material arrangement face:{face_id}: {exc}") from exc
        arrangements.append(_normalise_cells(arrangement))
    if model.revision != revision or to_dict(model)["checksum"]["value"] != checksum:
        raise GeometryError("geometry changed during intersection planning")
    arrangements=_synchronize_boundary_events(arrangements,check)
    plan=IntersectionPlan(model.model_id, revision, checksum, handles, policy,
                          arrangements, axes, contacts,face_contacts=tuple(face_contacts))
    return replace(plan,content_checksum=_plan_content(plan))


def _coincident(first, second, tolerance):
    if isinstance(first, LinePath) and isinstance(second, LinePath):
        return True  # Caller already established both matching endpoints.
    if isinstance(first, EllipticArc) and isinstance(second, EllipticArc):
        inverse = np.linalg.pinv(np.column_stack((first.u_vector, first.v_vector)))
        normal = np.cross(first.u_vector, first.v_vector)
        normal /= np.linalg.norm(normal)
        center = np.asarray(second.center)-first.center
        u, v = inverse @ second.u_vector, inverse @ second.v_vector
        return (abs(float(normal @ center)) <= tolerance and np.linalg.norm(inverse @ center) <= tolerance
                and abs(float(u @ u)-1) <= tolerance and abs(float(v @ v)-1) <= tolerance
                and abs(float(u @ v)) <= tolerance
                and bool(point_parameters(first, second.evaluate(.5), tolerance=tolerance)))
    if isinstance(first, BezierPath) and isinstance(second, BezierPath):
        a, b = np.asarray(first.controls), np.asarray(second.controls)
        return a.shape == b.shape and min(np.max(np.linalg.norm(a-b, axis=1)), np.max(np.linalg.norm(a-b[::-1], axis=1))) <= tolerance
    if isinstance(first, CylinderIntersectionCurve) and isinstance(second, CylinderIntersectionCurve):
        from .cylinder_curve_events import cylinder_roots
        return (all(cylinder_roots(first,support,transform=second.transform,tolerance=tolerance) is None
                    for support in (second.first,second.second))
                and bool(point_parameters(first,second.evaluate(.5),tolerance=tolerance)))
    if isinstance(first,CylinderIntersectionCurve) and isinstance(second,EllipticArc):
        from .arrangement_geometry import ellipse_on_branch_supports
        return (ellipse_on_branch_supports(second,first,tolerance)
                and bool(point_parameters(first,second.evaluate(.5),tolerance=tolerance)))
    if isinstance(first,EllipticArc) and isinstance(second,CylinderIntersectionCurve):
        return _coincident(second,first,tolerance)
    return False


def _canonicalize_member_edges(model, curves, axes, joint_ids, check):
    """Share qualified axis/boundary definitions without losing owner records."""
    selected = {member for axis in axes for member in axis.member_ids}
    candidates = sorted({use.edge_id for use in model.member_edge_uses.values()
                         if use.member_id in selected})
    for axis_id in candidates:
        check()
        if axis_id not in model.edges:
            continue
        axis = model.edges[axis_id]
        tolerance = model.tolerance.effective_length(model.edge_length(axis_id))
        duplicates = [identifier for identifier in model.edges_using_vertex(axis.start)
                      if identifier != axis_id and
                      {model.edges[identifier].start, model.edges[identifier].end} == {axis.start, axis.end}
                      and _coincident(curves[axis_id], curves[identifier], tolerance)]
        if not duplicates:
            continue
        canonical_id = min(axis_id, *duplicates)
        canonical = model.edges[canonical_id]
        for duplicate_id in sorted({axis_id, *duplicates}-{canonical_id}):
            if duplicate_id not in model.edges:
                continue
            duplicate = model.edges[duplicate_id]
            direct = duplicate.start == canonical.start
            for use_id in tuple(model._edge_member_uses.get(duplicate_id, ())):
                use = model.member_edge_uses[use_id]
                model._put_structural("member_edge_use", replace(use, edge_id=canonical_id,
                    orientation=use.orientation if direct else (
                        Orientation.REVERSED if use.orientation is Orientation.FORWARD else Orientation.FORWARD)))
            for face_id in tuple(model.faces_using_edge(duplicate_id)):
                face = model.faces[face_id]
                def loop(values):
                    return tuple(OrientedEdge(canonical_id, item.forward == direct)
                                 if item.edge == duplicate_id else item for item in values)
                model._put_entity("face", replace(face, loop=loop(face.loop),
                    holes=tuple(loop(hole) for hole in face.holes)))
            from .structural import ParameterRange
            for attachment_id in tuple(model._target_attachments.get(("edge", duplicate_id), ())):
                attachment = model.attachments[attachment_id]
                parameters = attachment.target_parameters if direct else tuple(
                    ParameterRange(1-value.end, 1-value.start) for value in attachment.target_parameters)
                model._put_structural("attachment", replace(attachment,
                    target_id=canonical_id, target_parameters=parameters))
            for member_id in tuple(model._orientation_members.get(("edge", duplicate_id), ())):
                member = model.members[member_id]
                model._put_structural("member", replace(member, orientation_reference=("edge", canonical_id)))
            for attachment in tuple(model.attachments.values()):
                if attachment.source_kind == "edge" and attachment.source_id == duplicate_id:
                    model._put_structural("attachment", replace(attachment, source_id=canonical_id))
            model._delete_entity("edge", duplicate_id)
            model.record_replacement(EntityRef("edge", duplicate_id), (EntityRef("edge", canonical_id),))
            joint_ids.discard(duplicate_id)
        joint_ids.add(canonical_id)


def _child_support(model, face, outer, holes, tolerance):
    """Rebase a proven rectangular support; general trims stay native charts.

    A mapped child must not evaluate the untrimmed parent's whole unit square.
    This changes the parameter extent, never the underlying physical surface.
    """
    support=face.surface
    if holes or not isinstance(support,(Plane,Cylinder)):
        return outer,support,face.parameterization,None
    points=[model.vertex_position(model.oriented_start_vertex(use)) for use in outer]
    if isinstance(support,Cylinder):
        domain=MaterialDomain(0,support,(tuple(ArrangementPath(freeze_edge(model,use.edge)
                    if use.forward else freeze_edge(model,use.edge).subcurve(1.,0.)) for use in outer),))
        rows=[domain.uv(path.curve,0.) for path in domain.boundaries[0]]
        lower,upper=np.min(rows,axis=0),np.max(rows,axis=0)
        native_tolerance=tolerance/min(support.radius*abs(support.sweep_angle),support.height)
        rectangular=all((abs(row[0]-lower[0])<=native_tolerance or abs(row[0]-upper[0])<=native_tolerance)
                        or (abs(row[1]-lower[1])<=native_tolerance or abs(row[1]-upper[1])<=native_tolerance)
                        for row in rows)
        for path in domain.boundaries[0]:
            first,last=domain.uv(path.curve,0.),domain.uv(path.curve,1.)
            if isinstance(path.curve,LinePath):
                rectangular &= abs(first[0]-last[0]) <= native_tolerance
            elif isinstance(path.curve,EllipticArc):
                rectangular &= (abs(first[1]-last[1])<=native_tolerance
                    and abs(float(np.asarray(path.curve.u_vector) @ support.axis))<=tolerance
                    and abs(float(np.asarray(path.curve.v_vector) @ support.axis))<=tolerance)
            else:
                rectangular=False
        if rectangular and np.all(upper-lower>native_tolerance):
            index=min(range(len(rows)),key=lambda i:(rows[i][1],rows[i][0]))
            # Numerical equality on the lower axial level is qualified in
            # physical tolerance before selecting the lowest angular corner.
            index=min((i for i,row in enumerate(rows) if abs(row[1]-lower[1])<=native_tolerance),
                      key=lambda i:rows[i][0])
            outer=outer[index:]+outer[:index]
            support=Cylinder(support.origin+lower[1]*support.height*support.axis,
                support.axis,support.radial_direction,support.radius,(upper[1]-lower[1])*support.height,
                support.start_angle+lower[0]*support.sweep_angle,(upper[0]-lower[0])*support.sweep_angle)
            return outer,support,None,model._detect_corners(outer)
        return outer,support,None,None
    corners=model._detect_corners(outer) if len(outer)>=4 else ()
    if len(corners)==4 and all(isinstance(model.edges[use.edge].curve,Straight) for use in outer):
        p0,p1,p2,p3=(points[index] for index in corners)
        if np.linalg.norm(p0+p2-p1-p3)<=tolerance:
            return outer,Plane(p0,p1-p0,p3-p0),None,corners
    return outer,support,None,None


def _apply_intersections_in_place(model, plan, *, policy):
    """Atomically apply a model/revision-bound material arrangement."""
    from .intersections import _merge_vertex
    from .attachment_remapping import capture_face_attachments, remap_face_attachments
    outer_journal=model._transaction_journal
    def keys():
        return {(kind,identifier) for kind in model._next_id for identifier in model._entity_store(kind)} | {
            (kind,identifier) for kind in model._next_structural_id for identifier in model._structural_store(kind)}
    before_keys=keys() if outer_journal is not None else set()
    if not isinstance(plan, IntersectionPlan):
        raise TypeError("apply_intersections needs an IntersectionPlan")
    policy = _policy(policy)
    _validate_policy(plan,policy)
    if model.model_id != plan.model_id:
        raise GeometryError("intersection plan belongs to another model")
    if model.revision != plan.revision:
        raise GeometryError("intersection plan is stale")
    if to_dict(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("intersection plan source binding changed")
    tolerances_by_face = {item.face_id: item.world_tolerance for item in plan.arrangements}
    tolerance = model.tolerance.length
    joint_ids = set()
    examined = 0
    def check():
        nonlocal examined
        examined += 1
        if policy.cancellation_check is not None and policy.cancellation_check():
            raise GeometryError("intersection application cancelled")
        if policy.max_predicates is not None and examined > policy.max_predicates:
            raise GeometryError("intersection application budget exhausted")
    with model.transaction():
        check()
        if policy.intent is ConnectionIntent.CONNECT:
            # Declare physical owners before fragmentation. Creating one
            # owner for each descendant would turn decomposition seams into
            # artificial multi-Sheet joints in downstream mesh preparation.
            for arrangement in plan.arrangements:
                if not any(use.face_id == arrangement.face_id for use in model.face_uses.values()):
                    model.add_sheet((arrangement.face_id,),
                                    name=f"intersection owner for face {arrangement.face_id}")
        # Split original trim edges globally before rebuilding any face. This
        # also updates untouched face loops and MemberEdgeUse parent intervals.
        fractions = {}
        originals = {}
        for axis in plan.axes:
            originals[axis.edge_id] = axis.curve
            fractions[axis.edge_id] = set(axis.split_parameters)
        for arrangement in plan.arrangements:
            for path in arrangement.paths:
                if path.source_edge is None:
                    continue
                edge_id = path.source_edge
                original = originals.setdefault(edge_id, freeze_edge(model, edge_id))
                for parameter in (0., 1.):
                    fractions.setdefault(edge_id, set()).update(point_parameters(original, path.curve.evaluate(parameter), tolerance=tolerance))
        for edge_id, values in sorted(fractions.items()):
            tolerance = min((tolerances_by_face[face_id] for face_id in model.faces_using_edge(edge_id)
                             if face_id in tolerances_by_face),
                            default=next((axis.world_tolerance for axis in plan.axes
                                          if axis.edge_id == edge_id), model.tolerance.length))
            current, previous = edge_id, 0.
            original = originals[edge_id]
            unique = []
            for fraction in sorted(values):
                point = original.evaluate(fraction)
                if min(np.linalg.norm(point-original.evaluate(0.)),
                       np.linalg.norm(point-original.evaluate(1.))) <= tolerance:
                    continue
                if unique and np.linalg.norm(point-original.evaluate(unique[-1])) <= tolerance:
                    continue
                unique.append(fraction)
            for fraction in unique:
                check()
                _vertex, (_left, right) = model.split_edge(current, (fraction-previous)/(1-previous))
                current, previous = right, fraction

        edge_curves = {edge_id: freeze_edge(model, edge_id) for edge_id in model.edges}
        used_vertices = {vertex for edge in model.edges.values() for vertex in (edge.start, edge.end)}
        vertex_sources = {identifier: {
            *[("face", face_id) for edge_id in model.edges_using_vertex(identifier)
              for face_id in model.faces_using_edge(edge_id)],
            *[("member", member_id) for edge_id in model.edges_using_vertex(identifier)
              for member_id in model.members_using_edge(edge_id)]}
            for identifier in used_vertices}
        canonical = []
        def vertex(point, sources):
            candidates = [identifier for identifier in sorted(used_vertices) if identifier in model.vertices
                          and vertex_sources.get(identifier, set()) & sources
                          and np.linalg.norm(model.vertex_position(identifier)-point) <= tolerance]
            if candidates:
                chosen = candidates[0]
                vertex_sources[chosen].update(sources)
                for duplicate in candidates[1:]:
                    vertex_sources[chosen].update(vertex_sources.get(duplicate, ()))
                    _merge_vertex(model, duplicate, chosen)
                    used_vertices.discard(duplicate)
                return chosen
            for identifier in canonical:
                if identifier in model.vertices and vertex_sources[identifier] & sources and np.linalg.norm(model.vertex_position(identifier)-point) <= tolerance:
                    vertex_sources[identifier].update(sources)
                    return identifier
            identifier = model.add_point(*point)
            canonical.append(identifier); used_vertices.add(identifier)
            vertex_sources[identifier] = set(sources)
            return identifier

        def edge(path, face_id):
            check()
            curve = path.curve
            sources = {("face", owner) for owner in (face_id, *path.owners)}
            sources.update(("member", member) for member in path.member_ids)
            start, end = vertex(curve.evaluate(0.), sources), vertex(curve.evaluate(1.), sources)
            candidates = []
            for edge_id in model.edges_using_vertex(start):
                record = model.edges[edge_id]
                if {record.start, record.end} == {start, end} and _coincident(curve, edge_curves[edge_id], tolerance):
                    candidates.append(edge_id)
            if candidates:
                edge_id = min(candidates)
            elif isinstance(curve, LinePath):
                edge_id = model.add_line(start, end)
            elif isinstance(curve, BezierPath):
                controls = tuple(model.add_point(*point) for point in curve.controls[1:-1])
                model.mark_construction_vertices(controls)
                edge_id = model.add_spline(start, controls, end)
            elif isinstance(curve, EllipticArc) and (
                abs(np.linalg.norm(curve.u_vector)-np.linalg.norm(curve.v_vector)) <= tolerance
                and abs(float(np.asarray(curve.u_vector) @ curve.v_vector)) <= tolerance):
                via = model.add_point(*curve.evaluate(.5)); model.mark_construction_vertices((via,))
                edge_id = model.add_arc(start, via, end)
            else:
                edge_id = model.add_curve(start, end, curve)
            edge_curves.setdefault(edge_id, freeze_edge(model, edge_id))
            if path.owners:
                joint_ids.add(edge_id)
            if path.decomposition:
                model.tag(EntityRef("edge", edge_id), "intersection_decomposition_seam")
            return OrientedEdge(edge_id, model.edges[edge_id].start == start)

        for arrangement in plan.arrangements:
            check()
            tolerance = arrangement.world_tolerance
            face = model.faces[arrangement.face_id]
            mapping = {index: edge(path, arrangement.face_id) for index, path in enumerate(arrangement.paths)}
            def loop(values):
                return tuple(OrientedEdge(mapping[index].edge, mapping[index].forward == forward) for index, forward in values)
            loops = [(loop(cell.outer), tuple(loop(hole) for hole in cell.holes)) for cell in arrangement.cells]
            if len(loops) == 1 and {item.edge for item in loops[0][0]} == {item.edge for item in face.loop} and (
                    {tuple(sorted(item.edge for item in hole)) for hole in loops[0][1]} ==
                    {tuple(sorted(item.edge for item in hole)) for hole in face.holes}):
                continue
            if len(loops) == 1:
                # A one-cell, area-conserving arrangement only updates trim
                # ports and canonical boundary edges. Keep its face identity
                # and support parameter frame, including existing attachments.
                outer,holes=loops[0]
                corners=model._detect_corners(outer)
                model._put_entity('face',replace(face,loop=outer,holes=holes,
                    corners=corners if len(corners)==4 else face.corners))
                continue
            snapshots = capture_face_attachments(model, face.id,check)
            child_definition = (replace(face,surface=arrangement.support,parameterization=None)
                                if not isinstance(face.surface,(Plane,Cylinder)) else face)
            model._delete_entity("face", face.id)
            children = []
            for outer, holes in loops:
                outer,support,parameterization,corners=_child_support(model,child_definition,outer,holes,tolerance)
                if corners is not None and len(corners)!=4:
                    corners=None
                child = model.add_face_from_loop(outer, corners, surface=support)
                model._put_entity("face", replace(model.faces[child], holes=holes,
                    metadata=face.metadata, parameterization=parameterization))
                children.append(child)
            remap_face_attachments(model, face, children, snapshots,check)
            model.record_replacement(face.ref, tuple(EntityRef("face", child) for child in children))
        member_contacts=[]
        for contact in plan.contacts:
            check()
            tolerance = contact.world_tolerance
            sources = {("member", member) for member, _parameter in contact.member_parameters}
            if contact.face_id is not None:
                sources.add(("face", contact.face_id))
            member_contacts.append((contact,vertex(np.asarray(contact.position), sources)))
        _canonicalize_member_edges(model, edge_curves, plan.axes, joint_ids, check)
        for contact in plan.face_contacts:
            check()
            tolerance=contact.world_tolerance
            station=vertex(np.asarray(contact.position),{('face',face) for face in contact.face_ids})
            if policy.intent is ConnectionIntent.CONNECT:
                from .joint_edges import query_joint_edge
                incident={face for edge_id in model.edges_using_vertex(station)
                          for face in model.faces_using_edge(edge_id)}
                owners={sheet for face in incident for use_id in model._face_structural_uses.get(face,())
                        for sheet in (model.face_uses[use_id].sheet_id,)}
                selected_owners={sheet for parent in contact.face_ids
                    for ref in model.resolve_ref(EntityRef('face',parent)) if ref.id in incident
                    for use_id in model._face_structural_uses.get(ref.id,())
                    for sheet in (model.face_uses[use_id].sheet_id,)}
                connected={owner:{owner} for owner in owners}
                for edge_id in model.edges_using_vertex(station):
                    joint=query_joint_edge(model,edge_id)
                    if joint.declared:
                        edge_owners={owner.id for owner in joint.sheets}
                        for owner in edge_owners:
                            connected.setdefault(owner,{owner}).update(edge_owners)
                reached=set();pending=list(sorted(selected_owners)[:1])
                while pending:
                    owner=pending.pop()
                    if owner in reached:continue
                    reached.add(owner);pending.extend(connected.get(owner,{owner})-reached)
                if selected_owners and selected_owners<=reached:
                    # Existing declared incident joints also connect their
                    # common station. Descendant point pairs must not add
                    # redundant relations on every idempotent batch replay.
                    continue
                from .structural import AttachmentKind,AttachmentTargetKind,AttachmentEvidence,ParameterRange
                for parent in contact.face_ids:
                    for face in sorted(ref.id for ref in model.resolve_ref(EntityRef('face',parent))
                                       if ref.id in incident):
                        uv=model.face_local_uv(face,model.vertex_position(station))
                        residual=float(np.linalg.norm(model.faces[face].surface.evaluate(*uv)
                                                      -model.vertex_position(station)))
                        if residual>tolerance:
                            raise GeometryError('canonical face contact exceeds its qualified tolerance')
                        model.ensure_attachment(None,AttachmentKind.VERTEX_ON_FACE,
                            AttachmentTargetKind.FACE,face,ParameterRange(0.,0.),
                            tuple(ParameterRange(float(value),float(value)) for value in uv),
                            source_kind='vertex',source_id=station,connection_intent=ConnectionIntent.CONNECT,
                            evidence=AttachmentEvidence.EXACT,max_residual=residual,tolerance_used=tolerance,
                            provenance={'contract':'ANYGEOMETRY_ANALYTIC_FACE_POINT_JOINT_V1'})
        if policy.intent is ConnectionIntent.CONNECT:
            from .joint_edges import declare_joint_edge
            from .member_joints import declare_member_contacts, declare_member_boundaries
            declare_member_contacts(model,member_contacts,check,
                                    sheet_target_ids={operand.id for operand in plan.operands if operand.kind=='sheet'})
            declare_member_boundaries(model,(member for axis in plan.axes for member in axis.member_ids),check)
            for edge_id in sorted(joint_ids):
                declare_joint_edge(model,edge_id,
                    model.tolerance.effective_length(model.edge_length(edge_id)))
        check()
    change_set=model.last_change_set
    reused=model.revision == plan.revision
    if outer_journal is not None:
        after_keys=keys()
        added=tuple(sorted(after_keys-before_keys)); removed=tuple(sorted(before_keys-after_keys))
        updated=tuple(sorted((set(outer_journal.changed)|set(outer_journal.structural_before)) &
                             before_keys & after_keys))
        change_set=ChangeSet(plan.revision,model.revision,added=added,removed=removed,modified=updated)
        reused=not (added or removed or updated)
    return IntersectionApplication(plan, change_set,
        tuple(model.handle("edge", identifier) for identifier in sorted(joint_ids)), reused)


def apply_intersections(model, plan, *, policy):
    """Apply on a detached candidate, then publish one complete owner change.

    Failed or cancelled candidates never consume live allocator IDs. This is
    stronger than the normal edit transaction's monotonic allocation rule.
    """
    if not isinstance(plan, IntersectionPlan):
        raise TypeError("apply_intersections needs an IntersectionPlan")
    effective = _policy(policy)
    _validate_policy(plan,effective)
    if model.model_id != plan.model_id:
        raise GeometryError("intersection plan belongs to another model")
    if model.revision != plan.revision:
        receipt = getattr(model, "_intersection_application_receipt", None)
        if (receipt is not None and receipt[0] is plan and receipt[1] == model.revision
                and receipt[2] == to_dict(model)["checksum"]["value"]):
            if effective.cancellation_check is not None and effective.cancellation_check():
                raise GeometryError("intersection application cancelled")
            return IntersectionApplication(plan, ChangeSet(model.revision, model.revision),
                tuple(model.handle("edge", identifier) for identifier in receipt[3]), True)
        raise GeometryError("intersection plan is stale")
    if to_dict(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("intersection plan source binding changed")
    if model._transaction_journal is not None:
        # Join the caller's atomic transaction. Snapshot adoption is forbidden
        # while a journal is active; ordinary owner writes participate in that
        # journal and its complete rollback instead.
        if effective.intent is ConnectionIntent.REUSE_EXISTING:
            candidate=model.clone(preserve_identity=True)
            outcome=_apply_intersections_in_place(candidate,plan,policy=effective)
            if candidate.revision != plan.revision:
                raise GeometryError("REUSE_EXISTING requires compatible existing topology")
            return IntersectionApplication(plan,ChangeSet(model.revision,model.revision),
                                           outcome.joint_edges,True)
        model._transaction_journal.exact_allocator_rollback=True
        return _apply_intersections_in_place(model,plan,policy=effective)
    candidate = model.clone(preserve_identity=True)
    outcome = _apply_intersections_in_place(candidate, plan, policy=policy)
    if effective.intent is ConnectionIntent.REUSE_EXISTING and candidate.revision != plan.revision:
        raise GeometryError("REUSE_EXISTING requires compatible existing topology")
    if effective.cancellation_check is not None and effective.cancellation_check():
        raise GeometryError("intersection application cancelled before commit")
    if model.revision != plan.revision or to_dict(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("geometry changed before intersection commit")
    before = model.revision
    model.restore_topology(candidate.topology_snapshot())
    result = IntersectionApplication(plan,
        model.last_change_set if model.revision != before else ChangeSet(before, before),
        tuple(model.handle("edge", edge.id) for edge in outcome.joint_edges), model.revision == before)
    if model.revision != before:
        # One receipt recognizes a repeat of the exact applied plan. It is
        # bound to the complete committed checksum and revision; unrelated
        # edits, undo/load, another plan or a forged plan never reuse it.
        model._intersection_application_receipt = (plan, model.revision,
            to_dict(model)["checksum"]["value"], tuple(edge.id for edge in result.joint_edges))
    return result
