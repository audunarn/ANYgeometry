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

from ._cross3 import _cross3
from .analytic_roots import root_isolation_memo
from .analytic_supports import plane_cylinder_support, cylinder_cylinder_support
from .arrangement_geometry import (LinePath, BezierPath, freeze_edge,
                                   point_parameters, curve_junctions)
from .entities import EntityRef, OrientedEdge
from .curves import Straight
from .errors import GeometryError
from .exact_curves import EllipticArc, CylinderIntersectionCurve
from .extruded_supports import extruded_support
from .branch_curves import BezierQuadricCurve
from .quadric_curves import QuadricIntersectionCurve
from .identity import EntityHandle
from .material_arrangement import (ArrangementPath, ArrangementPoint, ArrangementCell, MaterialDomain,
                                   MaterialArrangement, arrange_material, _clip, _clip_intervals)
from .member_arrangements import (MemberAxisArrangement, MemberPointContact,
                                  plan_member_arrangements)
from .predicates import IntersectionDimension, IntersectionKind, qualified_plane_plane
from .serialization import to_dict, _serialized_model_state, _qualified_model_state
from .structural import ConnectionIntent, Orientation
from .surfaces import Cone, Cylinder, ExtrudedSurface, Plane
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


@dataclass(frozen=True, slots=True, weakref_slot=True)
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
            if events:
                # One vectorized box test; the same elementwise comparisons
                # as testing each point separately, in the same order.
                points=np.asarray(events)
                outside=(np.any(points<lo-arrangement.world_tolerance,axis=1)
                         | np.any(points>hi+arrangement.world_tolerance,axis=1))
                events=[events[index] for index in np.flatnonzero(~outside)]
            for point in events:
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


def _synchronize_member_events(arrangements, axes, check):
    """Split member axes at ports introduced while decomposing incident faces."""
    from .material_arrangement import _parameters
    by_member = {}
    values = [set(axis.split_parameters) for axis in axes]
    for index, axis in enumerate(axes):
        for member in axis.member_ids:
            by_member.setdefault(member, []).append(index)
    for arrangement in arrangements:
        for path in arrangement.paths:
            indices = sorted({index for member in path.member_ids
                              for index in by_member.get(member, ())})
            if not indices:
                continue
            for parameter in (0., 1.):
                point = path.curve.evaluate(parameter)
                for index in indices:
                    check()
                    axis = axes[index]
                    values[index].update(point_parameters(
                        axis.curve, point, tolerance=axis.world_tolerance))
    return tuple(replace(axis, split_parameters=_parameters(
        values[index], axis.world_tolerance, axis.curve))
        for index, axis in enumerate(axes))


def _domain_bounds(domain):
    boxes = [path.curve.bounds() for loop in domain.boundaries for path in loop]
    return np.min([box[0] for box in boxes], axis=0), np.max([box[1] for box in boxes], axis=0)


def _coplanar_traces(first, second, tolerance, check, model):
    from .intersections import _sheet_ids_for_face
    same_sheet = bool(set(_sheet_ids_for_face(model, first.face_id)) &
                      set(_sheet_ids_for_face(model, second.face_id)))
    shared_seams = {path.source_edge: path.curve for loop in first.boundaries for path in loop
                    if path.decomposition and path.source_edge is not None} if same_sheet else {}
    shared_seams = {path.source_edge: shared_seams[path.source_edge]
                    for loop in second.boundaries for path in loop
                    if path.decomposition and path.source_edge in shared_seams}
    def physical(curve):
        # Source edge identity proves shared topology, not just common Sheet
        # ownership. For affine paths, endpoint containment certifies the
        # complete interval; other curve families remain conservatively kept.
        return not (isinstance(curve, LinePath) and any(
            isinstance(seam, LinePath) and
            all(point_parameters(seam, point, tolerance=tolerance)
                for point in (curve.start, curve.end))
            for seam in shared_seams.values()))
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
                     for component in result.components if len(component.witnesses) >= 2
                     and physical(LinePath(component.witnesses[0], component.witnesses[-1])))
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
    return tuple(curve for curve in result if physical(curve))


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
            if np.linalg.norm(_cross3(direction, support.axis)) > tolerance:
                return False
            coordinate, varying = 0, 1
        elif isinstance(curve, EllipticArc):
            normal = _cross3(curve.u_vector, curve.v_vector)
            normal /= np.linalg.norm(normal)
            if np.linalg.norm(_cross3(normal, support.axis)) > tolerance:
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
    elif isinstance(a, ExtrudedSurface) or isinstance(b, ExtrudedSurface):
        from .extruded_pair_supports import extruded_pair_support
        result = extruded_pair_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        if result.coincident:
            return _coplanar_traces(first, second, tolerance, check, model)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    elif isinstance(a, Cone) or isinstance(b, Cone):
        from .quadric_supports import cone_support
        result = cone_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        if result.coincident:
            return _coplanar_traces(first, second, tolerance, check, model)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
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
    from .preparation_epochs import _epoch_plan_required,_issue_native_plan
    from .native_support_snapshots import capture_native_supports
    native=capture_native_supports(model) if _epoch_plan_required(model) else None
    if native is not None and any(kind not in ('plane','cylinder') for _,kind,_ in native):
        raise GeometryError('epoch planning native support binding is unavailable')
    with root_isolation_memo():
        plan=_plan_intersections(model, operands, policy=policy)
    if native is not None:
        if (capture_native_supports(model)!=native or model.model_id!=plan.model_id or
                model.revision!=plan.revision or to_dict(model)['checksum']['value']!=plan.source_checksum):
            raise GeometryError('geometry or native support changed during epoch planning')
        _issue_native_plan(plan,native)
    return plan


def _plan_intersections(model, operands, *, policy):
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
    domain_bounds = {face_id: _domain_bounds(domain) for face_id, domain in domains.items()}
    lengths = {face_id: float(np.linalg.norm(hi-lo))
               for face_id, (lo, hi) in domain_bounds.items()}
    if policy.face_connections:
        for face_id, corners in boundary_only.items():
            lo1, hi1 = np.min(corners, axis=0), np.max(corners, axis=0)
            for other_id, domain in domains.items():
                check()
                lo2, hi2 = domain_bounds[other_id]
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
        lo1, hi1 = domain_bounds[first_id]; lo2, hi2 = domain_bounds[second_id]
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
        check_pair=check_pair, domain_bounds=domain_bounds)
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
    if model.revision != revision or _qualified_model_state(model)["checksum"]["value"] != checksum:
        raise GeometryError("geometry changed during intersection planning")
    arrangements=_synchronize_boundary_events(arrangements,check)
    axes=_synchronize_member_events(arrangements,axes,check)
    plan=IntersectionPlan(model.model_id, revision, checksum, handles, policy,
                          arrangements, axes, contacts,face_contacts=tuple(face_contacts))
    return replace(plan,content_checksum=_plan_content(plan))


def _coincident(first, second, tolerance):
    if isinstance(first, LinePath) and isinstance(second, LinePath):
        return True  # Caller already established both matching endpoints.
    if isinstance(first, EllipticArc) and isinstance(second, EllipticArc):
        inverse = np.linalg.pinv(np.column_stack((first.u_vector, first.v_vector)))
        normal = _cross3(first.u_vector, first.v_vector)
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
    if isinstance(first, QuadricIntersectionCurve) or isinstance(second, QuadricIntersectionCurve):
        from .quadric_events import branch_supports, curve_quadric_roots
        branch, other = (first, second) if isinstance(first, QuadricIntersectionCurve) else (second, first)
        return (all(curve_quadric_roots(other, support, tolerance=tolerance) is None
                    for support in branch_supports(branch))
                and bool(point_parameters(branch, other.evaluate(.5), tolerance=tolerance)))
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
    if isinstance(first, BezierQuadricCurve) or isinstance(second, BezierQuadricCurve):
        from .branch_events import world_quadric
        from .quadric_events import curve_quadric_roots
        branch, other = (first, second) if isinstance(first, BezierQuadricCurve) else (second, first)
        # the other curve lies on the branch's quadric and on its wall (every sample is a point of the branch)
        return (curve_quadric_roots(other, world_quadric(branch), tolerance=tolerance) is None
                and all(point_parameters(branch, other.evaluate(float(t)), tolerance=tolerance)
                        for t in np.linspace(0., 1., 9)))
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


def _extruded_child_support(model, support, outer, tolerance):
    """Rebase a rectangular child of an extruded surface onto its own parameter ranges (no new geometry)."""
    domain=MaterialDomain(0,support,(tuple(ArrangementPath(freeze_edge(model,use.edge)
                if use.forward else freeze_edge(model,use.edge).subcurve(1.,0.)) for use in outer),))
    rows=[domain.uv(path.curve,0.) for path in domain.boundaries[0]]
    lower,upper=np.min(rows,axis=0),np.max(rows,axis=0)
    (u0,u1),(v0,v1)=support.u_range,support.v_range
    t=u0+np.linspace(0.,1.,33)*(u1-u0)
    speed=min(float(np.max(np.linalg.norm(support.directrix.derivative(t),axis=1)))*abs(u1-u0),
              float(np.linalg.norm(support.vector))*abs(v1-v0))
    native_tolerance=tolerance/speed
    rectangular=all((abs(row[0]-lower[0])<=native_tolerance or abs(row[0]-upper[0])<=native_tolerance)
                    or (abs(row[1]-lower[1])<=native_tolerance or abs(row[1]-upper[1])<=native_tolerance)
                    for row in rows)
    for path in domain.boundaries[0]:
        first,middle,last=domain.uv(path.curve,0.),domain.uv(path.curve,.5),domain.uv(path.curve,1.)
        if isinstance(path.curve,LinePath):
            rectangular &= abs(first[0]-last[0]) <= native_tolerance and abs(first[0]-middle[0]) <= native_tolerance
        else:
            rectangular &= abs(first[1]-last[1]) <= native_tolerance and abs(first[1]-middle[1]) <= native_tolerance
    if not rectangular or not np.all(upper-lower>native_tolerance):
        return outer,support,None,None
    index=min((i for i,row in enumerate(rows) if abs(row[1]-lower[1])<=native_tolerance),key=lambda i:rows[i][0])
    outer=outer[index:]+outer[:index]
    child=support.subpatch((lower[0],upper[0]),(lower[1],upper[1]))
    return outer,child,None,model._detect_corners(outer) if len(outer) >= 4 else None


def _store_extruded_support(model, face_id, support, check):
    """Store the exact support of a recovered extrusion whose loop is no longer four edges.

    A contact that splits a boundary edge leaves the face as one cell with a longer loop, which topology
    recognition cannot recover. The stored support is the same surface in the chart of its own directrix,
    so attachments are remapped by position exactly as for the children of a split.
    """
    from .attachment_remapping import capture_face_attachments, remap_face_attachments
    face = model.faces[face_id]
    if (isinstance(support, ExtrudedSurface) and not isinstance(face.surface, ExtrudedSurface)
            and face.parameterization is None and extruded_support(model, face_id) is None):
        snapshots = capture_face_attachments(model, face_id, check)
        model._put_entity('face', replace(face, surface=support, corners=()))
        remap_face_attachments(model, face, (face_id,), snapshots, check)


def _child_support(model, face, outer, holes, tolerance):
    """Rebase a proven rectangular support; general trims stay native charts.

    A mapped child must not evaluate the untrimmed parent's whole unit square.
    This changes the parameter extent, never the underlying physical surface.
    """
    support=face.surface
    if holes or not isinstance(support,(Plane,Cylinder,Cone,ExtrudedSurface)):
        return outer,support,face.parameterization,None
    if isinstance(support,ExtrudedSurface):
        return _extruded_child_support(model,support,outer,tolerance)
    points=[model.vertex_position(model.oriented_start_vertex(use)) for use in outer]
    if isinstance(support,(Cylinder,Cone)):
        domain=MaterialDomain(0,support,(tuple(ArrangementPath(freeze_edge(model,use.edge)
                    if use.forward else freeze_edge(model,use.edge).subcurve(1.,0.)) for use in outer),))
        rows=[domain.uv(path.curve,0.) for path in domain.boundaries[0]]
        lower,upper=np.min(rows,axis=0),np.max(rows,axis=0)
        widest=max(support.radius_start,support.radius_end) if isinstance(support,Cone) else support.radius
        native_tolerance=tolerance/min(widest*abs(support.sweep_angle),support.height)
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
            origin=support.origin+lower[1]*support.height*support.axis
            height=(upper[1]-lower[1])*support.height
            start,sweep=support.start_angle+lower[0]*support.sweep_angle,(upper[0]-lower[0])*support.sweep_angle
            if isinstance(support,Cone):
                def radius(v):
                    return (1-v)*support.radius_start+v*support.radius_end
                support=Cone(origin,support.axis,support.radial_direction,radius(lower[1]),radius(upper[1]),
                             height,start,sweep)
            else:
                support=Cylinder(origin,support.axis,support.radial_direction,support.radius,height,start,sweep)
            return outer,support,None,model._detect_corners(outer) if len(outer) >= 4 else None
        return outer,support,None,None
    corners=model._detect_corners(outer) if len(outer)>=4 else ()
    if len(corners)==4 and all(isinstance(model.edges[use.edge].curve,Straight) for use in outer):
        p0,p1,p2,p3=(points[index] for index in corners)
        if np.linalg.norm(p0+p2-p1-p3)<=tolerance:
            return outer,Plane(p0,p1-p0,p3-p0),None,corners
    return outer,support,None,None


def _apply_intersections_in_place(model, plan, *, policy, _edge_preimage_draft=None):
    """Atomically apply a model/revision-bound material arrangement."""
    from .intersections import _merge_vertex
    from .attachment_remapping import capture_face_attachments, remap_face_attachments
    from .edge_attachment_remapping import split_edge_attachments, _member_parameter
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
    if _qualified_model_state(model)["checksum"]["value"] != plan.source_checksum:
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
        contact_ranges = {
            (member, parameter): next(model.member_edge_uses[use_id].parent_range
                for use_id in model.members[member].edge_use_ids
                if model.member_edge_uses[use_id].parent_range.contains(parameter,
                    tolerance=model.tolerance.parameter))
            for contact in plan.contacts for member, parameter in contact.member_parameters}
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
                # Earlier children can use different regularized charts; invert
                # the original station on the current exact descendant.
                current_curve = freeze_edge(model, current)
                stations = point_parameters(current_curve, original.evaluate(fraction), tolerance=tolerance)
                if len(stations) != 1:
                    raise GeometryError("split station has no unique descendant parameter")
                parent_definition = None
                if _edge_preimage_draft is not None:
                    from .edge_subcurve_preimages import _edge_ancestry_definition
                    parent_definition = _edge_ancestry_definition(model, current)
                _vertex, (left, right) = split_edge_attachments(model, current, stations[0], check)
                if _edge_preimage_draft is not None:
                    from .edge_subcurve_preimages import (_drop_edge_subcurve_records,
                        _record_edge_subcurve_split, _SubcurveEnclosureUnavailable)
                    try:
                        _record_edge_subcurve_split(_edge_preimage_draft, current, stations[0],
                            (left, right), tolerance, model=model,
                            parent_definition=parent_definition)
                    except _SubcurveEnclosureUnavailable as exc:
                        if _edge_preimage_draft.enclosure_failure is not exc:
                            raise  # Never reinterpret a callback exception.
                        # Optional approximation proof is absent; geometry keeps
                        # its established acceptance. Requested proof queries
                        # must explicitly refuse these unqualified children, and
                        # the dead parent's captured occurrences are dropped with
                        # its primary record instead of lingering unsealed.
                        _drop_edge_subcurve_records(_edge_preimage_draft,
                            (current, left, right))
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
        # Uniform grids over the used vertices (one per tolerance in force) so
        # a point is compared only with vertices in the 27 neighbouring cells.
        # Cells are twice the tolerance, so every vertex within tolerance of a
        # point lies in one of them; the exact predicate below is unchanged.
        grids = {}
        def cell(size, position):
            return tuple(math.floor(float(value)/size) for value in position)
        def grid_for(size):
            grid = grids.get(size)
            if grid is None:
                grid = {}
                for identifier in used_vertices:
                    if identifier in model.vertices:
                        grid.setdefault(cell(size, model.vertex_position(identifier)), []).append(identifier)
                grids[size] = grid
            return grid
        def nearby(point):
            size = 2*tolerance
            if not (math.isfinite(size) and size > 0. and np.all(np.isfinite(point))):
                return sorted(used_vertices)
            try:
                grid = grid_for(size)
                x, y, z = cell(size, point)
            except (ValueError, OverflowError):
                return sorted(used_vertices)
            found = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        found.extend(grid.get((x+dx, y+dy, z+dz), ()))
            return sorted(identifier for identifier in found if identifier in used_vertices)
        def vertex(point, sources):
            candidates = [identifier for identifier in nearby(point) if identifier in model.vertices
                          and vertex_sources.get(identifier, set()) & sources
                          and np.linalg.norm(model.vertex_position(identifier)-point) <= tolerance]
            if candidates:
                chosen = candidates[0]
                vertex_sources[chosen].update(sources)
                for duplicate in candidates[1:]:
                    vertex_sources[chosen].update(vertex_sources.get(duplicate, ()))
                    prior_definitions = {}
                    if _edge_preimage_draft is not None:
                        from .edge_subcurve_preimages import _edge_ancestry_definition
                        prior_definitions = {edge: _edge_ancestry_definition(model, edge)
                            for edge in model.edges_using_vertex(duplicate)}
                    _merge_vertex(model, duplicate, chosen)
                    if _edge_preimage_draft is not None:
                        from .edge_subcurve_preimages import _rebind_edge_subcurve_incidence
                        _rebind_edge_subcurve_incidence(_edge_preimage_draft,
                            prior_definitions, model=model)
                    used_vertices.discard(duplicate)
                return chosen
            for identifier in canonical:
                if identifier in model.vertices and vertex_sources[identifier] & sources and np.linalg.norm(model.vertex_position(identifier)-point) <= tolerance:
                    vertex_sources[identifier].update(sources)
                    return identifier
            identifier = model.add_point(*point)
            canonical.append(identifier); used_vertices.add(identifier)
            vertex_sources[identifier] = set(sources)
            for size, grid in grids.items():
                grid.setdefault(cell(size, model.vertex_position(identifier)), []).append(identifier)
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
                if _edge_preimage_draft is not None and len(candidates) > 1:
                    # Canonical shared-edge reuse: capture every participating
                    # authenticated occurrence before the duplicates orphan.
                    from .edge_subcurve_preimages import _record_edge_subcurve_unification
                    _record_edge_subcurve_unification(_edge_preimage_draft, edge_id,
                        tuple(candidate for candidate in candidates if candidate != edge_id),
                        model=model)
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
            if arrangement.orientation < 0:
                # The arrangement walks counterclockwise in the support chart; a face stored clockwise
                # (a pointed cone's facets are) keeps its sense so its neighbours still agree with it.
                def reverse(values):
                    return tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(values))
                loops = [(reverse(outer), tuple(reverse(hole) for hole in holes)) for outer, holes in loops]
            if len(loops) == 1 and {item.edge for item in loops[0][0]} == {item.edge for item in face.loop} and (
                    {tuple(sorted(item.edge for item in hole)) for hole in loops[0][1]} ==
                    {tuple(sorted(item.edge for item in hole)) for hole in face.holes}):
                _store_extruded_support(model,face.id,arrangement.support,check)
                continue
            if len(loops) == 1:
                # A one-cell, area-conserving arrangement only updates trim
                # ports and canonical boundary edges. Keep its face identity
                # and support parameter frame, including existing attachments.
                outer,holes=loops[0]
                corners=model._detect_corners(outer)
                model._put_entity('face',replace(face,loop=outer,holes=holes,
                    corners=corners if len(corners)==4 else face.corners))
                _store_extruded_support(model,face.id,arrangement.support,check)
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
        # A later physical trace can coincide with a previously artificial
        # seam. Reconcile after all faces and member edges are canonicalized:
        # another face's old boundary must not restore the artificial marker.
        for edge_id in sorted(joint_ids):
            check()
            reference = EntityRef("edge", edge_id)
            if "intersection_decomposition_seam" in model.tags_for(reference):
                model.untag(reference, "intersection_decomposition_seam")
        # Contacts in the immutable plan use the original axis coordinates.
        # Resolve their retained world stations after every split/canonicalization.
        member_contacts = [(replace(contact, member_parameters=tuple(
            (member, _member_parameter(model, member, np.asarray(contact.position),
                contact_ranges[member, parameter], contact.world_tolerance))
            for member, parameter in contact.member_parameters)), station)
            for contact, station in member_contacts]
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
        if policy.intent in (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT):
            from .member_joints import declare_member_contacts, declare_member_boundaries
            declare_member_contacts(model,member_contacts,check,
                                    sheet_target_ids={operand.id for operand in plan.operands if operand.kind=='sheet'},
                                    intent=policy.intent)
            declare_member_boundaries(model,(member for axis in plan.axes for member in axis.member_ids),check,
                                      intent=policy.intent)
        if policy.intent is ConnectionIntent.CONNECT:
            from .joint_edges import declare_joint_edge
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
    with root_isolation_memo():
        return _apply_intersections(model, plan, policy=policy)


def _apply_intersections(model, plan, *, policy):
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
            from .preparation_epochs import _seals
            face_receipt=getattr(model,'_prepared_face_preimages_receipt',None)
            epoch_bound=model in _seals or (face_receipt is not None and
                getattr(face_receipt[0],'attachment_source_ids',None) is not None)
            epoch_digest=None
            if epoch_bound:
                from .prepared_face_preimages import _current_receipt,_binding_checksum
                from .native_support_snapshots import capture_native_supports
                binding=_current_receipt(model)
                if (binding.attachment_source_ids is None or binding.current_native_supports is None or
                        any(kind not in ('plane','cylinder') for _,kind,_ in binding.current_native_supports) or
                        capture_native_supports(model)!=binding.current_native_supports):
                    raise GeometryError('cached epoch native support binding is unavailable or changed')
                epoch_digest=_binding_checksum(binding)
            if effective.cancellation_check is not None and effective.cancellation_check():
                raise GeometryError("intersection application cancelled")
            # A callback can perform unrelated authoring; refuse the cached
            # result without rolling back that authoring.
            if (getattr(model,'_intersection_application_receipt',None) is not receipt or
                    model.revision!=receipt[1] or _qualified_model_state(model)['checksum']['value']!=receipt[2]):
                raise GeometryError('geometry changed during intersection reuse')
            if epoch_bound:
                binding=_current_receipt(model)
                if (_binding_checksum(binding)!=epoch_digest or
                        capture_native_supports(model)!=binding.current_native_supports):
                    raise GeometryError('native support or epoch source binding changed during cached application')
            return IntersectionApplication(plan, ChangeSet(model.revision, model.revision),
                tuple(model.handle("edge", identifier) for identifier in receipt[3]), True)
        raise GeometryError("intersection plan is stale")
    if to_dict(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("intersection plan source binding changed")
    from .preparation_epochs import _epoch_plan_required,_native_plan_supports
    if _epoch_plan_required(model):
        from .native_support_snapshots import capture_native_supports
        if capture_native_supports(model)!=_native_plan_supports(plan):
            raise GeometryError('epoch intersection plan native support binding changed')
    if model._transaction_journal is not None:
        from .preparation_epochs import _has_epoch_permit
        if _has_epoch_permit(model):
            raise GeometryError('first preparation of a new epoch requires the committed owner path')
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
    from .prepared_face_preimages import (_capture_application_preimages,
        _compose_application_preimages, _publish_application_preimages)
    from .edge_subcurve_preimages import (_capture_edge_subcurve_preimages,
        _finalize_edge_subcurve_preimages, _publish_edge_subcurve_preimages)

    def provenance_check(*_stage):
        if effective.cancellation_check is not None and effective.cancellation_check():
            raise GeometryError("intersection edge provenance cancelled")

    authored_preimages = _capture_application_preimages(model, allow_seed=effective.face_connections)
    from .native_support_snapshots import capture_native_supports
    epoch_native_source=capture_native_supports(model) if authored_preimages is not None and authored_preimages.attachment_source_ids is not None else None
    edge_draft = _capture_edge_subcurve_preimages(model, allow_seed=effective.face_connections,
        cancellation_check=provenance_check)
    candidate = model.clone(preserve_identity=True)
    from .preparation_epochs import (_begin_attachment_draft, _finish_attachment_draft,
                                    _discard_attachment_draft, _consume_epoch)
    _begin_attachment_draft(candidate,authored_preimages)
    try:
        outcome = _apply_intersections_in_place(candidate, plan, policy=policy,
            _edge_preimage_draft=edge_draft)
        finished_sources=_finish_attachment_draft(candidate)
        attachment_sources,derived_sources=finished_sources if finished_sources is not None else (None,None)
    finally:
        _discard_attachment_draft(candidate)
    authored_preimages = _compose_application_preimages(candidate, authored_preimages,
        outcome.change_set.replacements,attachment_source_ids=attachment_sources,epoch_derived_attachment_ids=derived_sources)
    edge_preimages = _finalize_edge_subcurve_preimages(candidate, edge_draft,
        cancellation_check=provenance_check)
    epoch_native_candidate=capture_native_supports(candidate) if epoch_native_source is not None else None
    sealed_candidate_checksum = (edge_preimages.binding.source_checksum if edge_preimages is not None
        else authored_preimages.source_checksum if authored_preimages is not None
        else to_dict(candidate)["checksum"]["value"])
    if effective.intent is ConnectionIntent.REUSE_EXISTING and candidate.revision != plan.revision:
        raise GeometryError("REUSE_EXISTING requires compatible existing topology")
    if effective.cancellation_check is not None and effective.cancellation_check():
        raise GeometryError("intersection application cancelled before commit")
    if model.revision != plan.revision or _qualified_model_state(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("geometry changed before intersection commit")
    # The last policy callback must not alter the detached result after its
    # topology/provenance proofs. No callback follows this preflight before the
    # committed snapshot is adopted.
    if _qualified_model_state(candidate)["checksum"]["value"] != sealed_candidate_checksum:
        raise GeometryError("intersection candidate changed before commit")
    if epoch_native_source is not None and (capture_native_supports(model)!=epoch_native_source or
            capture_native_supports(candidate)!=epoch_native_candidate):
        raise GeometryError('native geometry changed before epoch commit')
    before = model.revision
    original_faces = set(model.faces)
    covered_faces = {operand.id for operand in plan.operands if operand.kind == "face"}
    for operand in plan.operands:
        if operand.kind == "sheet":
            covered_faces.update(model.face_uses[use].face_id
                                 for use in model.sheets[operand.id].face_use_ids)
    complete_material = effective.face_connections and original_faces <= covered_faces
    staged_epoch=authored_preimages is not None and authored_preimages.attachment_source_ids is not None
    if staged_epoch:
        _publish_application_preimages(candidate,authored_preimages,
            tuple(sorted(candidate.faces)) if complete_material else (),sealed_candidate_checksum)
        import weakref
        if edge_preimages is not None and edge_preimages.owner() is not model:
            raise GeometryError('epoch edge provenance belongs to another owner')
        staged_edges=replace(edge_preimages,owner=weakref.ref(candidate)) if edge_preimages is not None else None
        _publish_edge_subcurve_preimages(candidate,staged_edges,sealed_candidate_checksum)
        candidate._intersection_preparation_receipt=(plan,candidate.revision,sealed_candidate_checksum,
            tuple(sorted(candidate.faces)) if complete_material else ())
    if staged_epoch:
        from copy import deepcopy
        saved_state=dict(model.__dict__)
        # Keep the original store/view and feature-owner objects. Snapshot only
        # mutable containers changed by adoption, and restore them in place.
        saved_containers=[(value,deepcopy(value)) for value in saved_state.values()
                          if type(value) in (dict,list,set)]
        from .preparation_epochs import _seals
        saved_seal=_seals.get(model)
    try:
        model.restore_topology(candidate.topology_snapshot())
        result = IntersectionApplication(plan,
            model.last_change_set if model.revision != before else ChangeSet(before, before),
            tuple(model.handle("edge", edge.id) for edge in outcome.joint_edges), model.revision == before)
        # Bind the exact committed result, including an unchanged/idempotent batch.
        # Only complete original-face classification certifies material ownership.
        checksum = to_dict(model)["checksum"]["value"]
        if staged_epoch:
            from .native_support_snapshots import capture_native_supports
            if capture_native_supports(model)!=capture_native_supports(candidate):
                raise GeometryError('native geometry changed during epoch publication')
        if model.revision != before:
            model._intersection_application_receipt = (plan, model.revision,
                checksum, tuple(edge.id for edge in result.joint_edges))
        model._intersection_preparation_receipt = (plan, model.revision, checksum,
            tuple(sorted(model.faces)) if complete_material else ())
        if staged_epoch:
            _publish_application_preimages(model,authored_preimages,
                model._intersection_preparation_receipt[3],checksum)
            _publish_edge_subcurve_preimages(model,edge_preimages,checksum)
        else:
            _publish_application_preimages(model, authored_preimages,
                                          model._intersection_preparation_receipt[3], checksum)
            _publish_edge_subcurve_preimages(model, edge_preimages, checksum)
        if authored_preimages is not None and edge_preimages is not None:
            _consume_epoch(model)
        return result
    except BaseException:
        if staged_epoch:
            for container,contents in saved_containers:
                if type(container) is list:container[:]=contents
                else:container.clear();container.update(contents)
            model.__dict__.clear();model.__dict__.update(saved_state)
            if saved_seal is None:_seals.pop(model,None)
            else:_seals[model]=saved_seal
        raise



def has_current_intersection_preparation(model, *, face_ids=None):
    """Whether the owner has a current, complete material-preparation receipt.

    This read-only query certifies a previous complete batch, not arbitrary
    existing topology. Receipts are local, revision/checksum bound and excluded
    from documents. Edits, clone/load/undo and partial batches require fresh
    classification. Invalid requested faces raise GeometryError.
    """
    from .intersections import _normalize_operand
    from .serialization import _serialized_model_state
    selected = set(model.faces) if face_ids is None else set()
    if face_ids is not None:
        for value in face_ids:
            operand = _normalize_operand(model, value)
            if operand.kind != "face":
                raise GeometryError("preparation binding requires face operands")
            selected.add(operand.id)
    receipt = getattr(model, "_intersection_preparation_receipt", None)
    if receipt is None:
        return False
    plan, revision, checksum, coverage = receipt
    eligible = bool(coverage and selected <= set(coverage)
                and model.revision == revision
                and model.model_id == plan.model_id
                and plan.content_checksum == _plan_content(plan))
    if not eligible:
        return False
    if _serialized_model_state(model)["checksum"]["value"] == checksum:
        return True
    # A changed raw state no longer has prior qualification. Preserve the
    # public invalid-topology error before reporting a valid stale receipt.
    # Exact unchanged content keeps the fingerprint-only read path above.
    to_dict(model)
    return False


def clone_prepared_geometry(model, *, new_preparation_epoch=False):
    """Detach an exact mesh-attempt copy, preserving a valid local owner proof.

    Ordinary unprepared copies retain normal clone behavior. Prepared copies
    preserve logical document identity and their complete bound receipt only
    after verifying the source and copy have identical committed checksums.
    The source stays unchanged; edits to either copy invalidate its binding.
    """
    if type(new_preparation_epoch) is not bool:
        raise GeometryError('new_preparation_epoch must be a plain boolean')
    if new_preparation_epoch:
        if model._transaction_journal is not None:
            raise GeometryError('a new preparation epoch requires committed source geometry')
        from .native_support_snapshots import capture_native_supports
        before=to_dict(model);native=capture_native_supports(model)
        made=model.clone(preserve_identity=False)
        copied=to_dict(made)
        source_payload={k:v for k,v in before.items() if k not in ('model_id','checksum')}
        clone_payload={k:v for k,v in copied.items() if k not in ('model_id','checksum')}
        if (made.model_id==model.model_id or source_payload!=clone_payload or
                capture_native_supports(made)!=native or capture_native_supports(model)!=native or to_dict(model)!=before):
            raise GeometryError('preparation epoch clone changed source or native geometry')
        from .preparation_epochs import _issue_epoch
        _issue_epoch(made)
        return made
    prepared = has_current_intersection_preparation(model)
    receipt = getattr(model, "_intersection_preparation_receipt", None)
    made = model.clone(preserve_identity=prepared)
    if prepared:
        checksum = receipt[2]
        if (to_dict(model)["checksum"]["value"] != checksum
                or to_dict(made)["checksum"]["value"] != checksum):
            raise GeometryError("prepared geometry changed during detached copying")
        made._intersection_preparation_receipt = receipt
        from .prepared_face_preimages import _copy_current_preimages
        _copy_current_preimages(model, made)
        from .edge_subcurve_preimages import _copy_current_edge_subcurve_preimages
        _copy_current_edge_subcurve_preimages(model, made)
    return made


def set_prepared_face_corners(model, updates):
    """Atomically change corner indices without losing a current material proof.

    Only explicit analytic supports without a separate parameterization qualify:
    their material is defined by the unchanged support and trim loops, not corner
    indices. A stale or partial preparation, topology-backed map, nested transaction
    or any other document change is rejected. This does not refresh application
    receipts or make old plans applicable after an edit.

    As with batch application, change hooks precede publication of the new
    receipt. The proof query is fail-closed during hooks and current on return.
    """
    if model._transaction_journal is not None:
        raise GeometryError("prepared corner editing requires no active transaction")
    if not has_current_intersection_preparation(model):
        raise GeometryError("prepared corner editing requires a current complete preparation")
    original = to_dict(model)
    receipt = model._intersection_preparation_receipt
    from .prepared_face_preimages import _current_receipt, _publish_application_preimages
    from .edge_subcurve_preimages import (_capture_edge_subcurve_preimages,
        _finalize_edge_subcurve_preimages, _publish_edge_subcurve_preimages)
    edge_draft = _capture_edge_subcurve_preimages(model, allow_seed=False)
    try:
        authored_preimages = _current_receipt(model)
    except GeometryError:
        authored_preimages = None
    try:
        requested = {key: tuple(value) for key, value in updates.items()}
    except (AttributeError, TypeError) as exc:
        raise GeometryError("prepared corner updates must map face IDs to corner indices") from exc
    for identifier in requested:
        if type(identifier) is not int or identifier not in model.faces:
            raise GeometryError("prepared corner update references an invalid face")
        face = model.faces[identifier]
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in requested[identifier]):
            raise GeometryError("prepared corner indices must be integers")
        if (type(face.surface) not in (Plane, Cylinder, Cone, ExtrudedSurface)
                or face.parameterization is not None):
            raise GeometryError("prepared corner editing requires an explicit corner-independent support")
    candidate = clone_prepared_geometry(model)
    with candidate.transaction():
        for identifier in sorted(requested):
            candidate.set_face_corners(identifier, requested[identifier])
    actual = to_dict(candidate)
    # Compare the entire certified document, not only the edited faces. Even
    # derived feature/ownership changes must fail this narrow operation closed.
    expected = dict(original)
    expected['faces'] = [dict(face, corners=list(candidate.faces[face['id']].corners))
                         if face['id'] in requested else face for face in original['faces']]
    expected['revision'], expected['checksum'] = actual['revision'], actual['checksum']
    if actual != expected:
        raise GeometryError("prepared corner update changed fields other than corner indices")
    if to_dict(model) != original:
        raise GeometryError("geometry changed during prepared corner editing")
    if actual == original:
        return
    edge_preimages = _finalize_edge_subcurve_preimages(candidate, edge_draft)
    model.restore_topology(candidate.topology_snapshot())
    plan, _revision, _checksum, coverage = receipt
    model._intersection_preparation_receipt = (
        plan, model.revision, to_dict(model)['checksum']['value'], coverage)
    _publish_application_preimages(model, authored_preimages, coverage,
                                  model._intersection_preparation_receipt[2])
    _publish_edge_subcurve_preimages(model, edge_preimages,
                                   model._intersection_preparation_receipt[2])
