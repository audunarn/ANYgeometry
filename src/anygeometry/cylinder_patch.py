"""Bounded standalone Cylinder proof (candidate; not yet adopted by consumers)."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from fractions import Fraction
import hashlib
import json
import math
import types
from typing import get_args, get_origin, get_type_hints
from uuid import UUID

import numpy as np

from .cylinder_charts import _I, _Proof, _Refusal, _q, _upper_abs, _hull
from .curves import Arc, Straight, ArcFrame, arc_frame, sample_arc, sample_straight, _COLLINEAR_RTOL
from .errors import GeometryError
from .identity import EntityHandle
from .model import GeometryModel
from .surfaces import Cylinder


class CylinderPatchStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    UNRESOLVED = "UNRESOLVED"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"


class CylinderPatchErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    WRONG_MODEL = "WRONG_MODEL"
    INACTIVE_ENTITY = "INACTIVE_ENTITY"
    STALE_REVISION = "STALE_REVISION"
    STALE_RESULT = "STALE_RESULT"
    INVALID_RESULT = "INVALID_RESULT"
    BUSY_MODEL = "BUSY_MODEL"
    UNQUALIFIED_RESULT = "UNQUALIFIED_RESULT"


class CylinderPatchError(GeometryError):
    def __init__(self, code, diagnostics):
        object.__setattr__(self, '_code', CylinderPatchErrorCode(code))
        object.__setattr__(self, '_diagnostics', tuple(diagnostics))
        super().__init__(f"{self.code.value}: {'; '.join(self.diagnostics)}")

    code = property(lambda self: self._code)
    diagnostics = property(lambda self: self._diagnostics)

    def __setattr__(self, name, value):
        if name in ('code', 'diagnostics', '_code', '_diagnostics'):
            raise AttributeError('immutable patch diagnostics')
        super().__setattr__(name, value)


def _error(code, text):
    return CylinderPatchError(code, (text,))


Interval = tuple[float, float]
Pair = tuple[float, float]
Vector = tuple[float, float, float]
Boxes = tuple[Interval, Interval]


class _EvidenceBudget:
    def __init__(self, proof=None):
        self.nodes = self.bytes = 0
        self.proof = proof

    def visit(self, value):
        self.nodes += 1
        if self.nodes > 200000:
            raise _error(CylinderPatchErrorCode.INVALID_RESULT,'evidence node budget')
        if self.proof is not None and self.nodes % 16 == 0:
            self.proof.cancel('patch evidence validation')
        if isinstance(value,str):
            if len(value)>3200000-self.bytes:
                raise _error(CylinderPatchErrorCode.INVALID_RESULT,'evidence byte budget')
            self.bytes += len(value.encode('utf-8'))
        else:
            self.bytes += 16
        if self.bytes>3200000:
            raise _error(CylinderPatchErrorCode.INVALID_RESULT,'evidence byte budget')


def _typed(value, annotation, budget, depth=0):
    budget.visit(value)
    if depth>16:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'evidence nesting budget')
    origin,args=get_origin(annotation),get_args(annotation)
    if origin is types.UnionType:
        candidates=[a for a in args if (a is type(None) and value is None) or
                    (get_origin(a) is tuple and type(value) is tuple) or
                    (isinstance(a,type) and type(value) is a)]
        if len(candidates)!=1: raise _error(CylinderPatchErrorCode.INVALID_RESULT,'field type')
        return _typed(value,candidates[0],budget,depth+1)
    if origin is tuple:
        if type(value) is not tuple or len(value)>4096:
            raise _error(CylinderPatchErrorCode.INVALID_RESULT,'bounded tuple required')
        if len(args)==2 and args[1] is Ellipsis:
            annotations=(args[0],)*len(value)
        else:
            if len(value)!=len(args): raise _error(CylinderPatchErrorCode.INVALID_RESULT,'tuple shape')
            annotations=args
        return [_typed(v,a,budget,depth+1) for v,a in zip(value,annotations)]
    if annotation is type(None) and value is None: return None
    if type(value) is not annotation:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'field type')
    if annotation is float:
        if not math.isfinite(value): raise _error(CylinderPatchErrorCode.INVALID_RESULT,'finite value required')
    if annotation is int and abs(value).bit_length()>8192:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'integer budget')
    if isinstance(value,UUID): return value.hex
    if isinstance(value,EntityHandle):
        return [value.model_id.hex,value.kind,value.id]
    if isinstance(value,StrEnum): return value.value
    if is_dataclass(value):
        hints=get_type_hints(type(value))
        return {f.name:_typed(getattr(value,f.name),hints[f.name],budget,depth+1) for f in fields(value)}
    return value


class _PatchRecord:
    __slots__=()
    def __post_init__(self):
        _typed(self,type(self),_EvidenceBudget())


@dataclass(frozen=True,slots=True)
class CylinderPatchCertificate(_PatchRecord):
    algorithm: str
    complete: bool
    tolerance_length: float
    tolerance_surface: float
    tolerance_angular: float
    max_residual: float
    max_enclosure_width: float
    work_counts: tuple[tuple[str,int],...]


@dataclass(frozen=True,slots=True)
class CylinderPatchOccurrence(_PatchRecord):
    coedge: EntityHandle
    edge: EntityHandle
    start_vertex: EntityHandle
    end_vertex: EntityHandle
    loop_index: int
    coedge_forward: bool
    face_use_orientation: int
    carrier: str
    source_range: Pair
    native_endpoint_uv: tuple[Pair,Pair]
    native_endpoint_uv_enclosures: tuple[Boxes,Boxes]
    lifted_angle_endpoints: Boxes
    axial_endpoints: Boxes
    branch_lifts: tuple[int,int]
    residual_bound: float
    external_coedges: tuple[EntityHandle,...]


@dataclass(frozen=True,slots=True)
class CylinderPatchLoop(_PatchRecord):
    index: int
    role: str
    coedges: tuple[EntityHandle,...]
    winding: int
    uv_bounds: tuple[Interval,Interval,Interval,Interval]
    physical_bounds: tuple[Interval,Interval,Interval,Interval]
    seed_uv: Pair
    seed_enclosure: Boxes


@dataclass(frozen=True,slots=True)
class CylinderPatchResult(_PatchRecord):
    contract_version: int
    model_id: UUID
    revision: int
    requested_face_uses: tuple[EntityHandle,...]
    policy: CylinderPatchPolicy
    status: CylinderPatchStatus
    part: EntityHandle | None
    sheet: EntityHandle | None
    face: EntityHandle | None
    face_use: EntityHandle
    face_use_orientation: int | None
    source_origin: Vector | None
    axis: Vector | None
    radial_direction: Vector | None
    radius: float | None
    height: float | None
    start_angle: float | None
    sweep_angle: float | None
    sweep_sign: int | None
    height_sign: int | None
    local_to_physical_scale: Pair | None
    inverse_scale: Pair | None
    loops: tuple[CylinderPatchLoop,...]
    occurrences: tuple[CylinderPatchOccurrence,...]
    certificate: CylinderPatchCertificate
    diagnostics: tuple[str,...]
    digest: str


@dataclass(frozen=True,slots=True)
class CylinderPatchOccurrenceRequest(_PatchRecord):
    coedge: EntityHandle
    parameter: float

    def __post_init__(self):
        _PatchRecord.__post_init__(self)
        if self.coedge.kind!='coedge' or not 0<=self.parameter<=1:
            raise _error(CylinderPatchErrorCode.INVALID_REQUEST,'native coedge parameter required')


@dataclass(frozen=True,slots=True)
class CylinderPatchSample(_PatchRecord):
    coedge: EntityHandle
    edge: EntityHandle
    parameter: float
    point: Vector
    local_uv: Pair
    physical_chart: Pair
    local_uv_enclosure: Boxes
    physical_enclosure: Boxes
    endpoint_vertex: EntityHandle | None
    identity_key: str
    residual_bound: float
    status: CylinderPatchStatus


@dataclass(frozen=True,slots=True)
class CylinderPatchEvaluation(_PatchRecord):
    model_id: UUID
    revision: int
    patch_digest: str
    samples: tuple[CylinderPatchSample,...]
    certificate: CylinderPatchCertificate


@dataclass(frozen=True, slots=True)
class CylinderPatchPolicy:
    max_occurrences: int = 32
    max_vertices: int = 96
    max_interval_operations: int = 200000
    max_pair_tests: int = 65536
    max_evaluations: int = 4096
    max_external_incidences: int = 256

    def __post_init__(self):
        for field, cap in zip(fields(self), (32, 96, 200000, 65536, 4096, 256)):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= cap:
                raise _error(CylinderPatchErrorCode.INVALID_REQUEST, 'invalid patch policy')


class _PatchProof(_Proof):
    def __init__(self, policy, callback):
        super().__init__(policy, callback)
        self.counts.update(lift_candidates=0, encoding_nodes=0, encoding_bytes=0)
        self.freshness_check = None

    def cancel(self, phase='patch qualification'):
        if self.freshness_check is not None:
            self.freshness_check()
        super().cancel(phase)
        if self.freshness_check is not None:
            self.freshness_check()

    def step(self, name, maximum, period=16):
        self.charge(name, maximum, period)


def _fresh(model, revision):
    if model._transaction_journal is not None:
        raise _error(CylinderPatchErrorCode.BUSY_MODEL, 'committed model required')
    if model.revision != revision:
        raise _error(CylinderPatchErrorCode.STALE_REVISION, 'model changed')


def _selection(model, face_uses, revision, callback):
    if not isinstance(model, GeometryModel):
        raise TypeError('GeometryModel required')
    if type(revision) is not int or revision < 0 or (callback is not None and not callable(callback)):
        raise _error(CylinderPatchErrorCode.INVALID_REQUEST, 'revision/callback invalid')
    if isinstance(face_uses, (str, bytes)):
        raise _error(CylinderPatchErrorCode.INVALID_REQUEST, 'one ordered FaceUse required')
    iterator = iter(face_uses)
    sentinel=object()
    first, extra = next(iterator, sentinel), next(iterator, sentinel)
    if extra is not sentinel or not isinstance(first, EntityHandle) or first.kind != 'face_use':
        raise _error(CylinderPatchErrorCode.INVALID_REQUEST, 'one ordered FaceUse required')
    if first.model_id != model.model_id:
        raise _error(CylinderPatchErrorCode.WRONG_MODEL, 'FaceUse belongs to another model')
    _fresh(model, revision)
    if first.id not in model.face_uses:
        raise _error(CylinderPatchErrorCode.INACTIVE_ENTITY, 'inactive FaceUse')
    return (first,)


class _PatchSource:
    """Bounded immutable source references, no live cache or evaluator calls."""
    def __init__(self, model, selected, proof):
        self.model_id = model.model_id
        self.use = model.face_uses[selected[0].id]
        self.face = model.faces[self.use.face_id]
        self.sheet = model.sheets[self.use.sheet_id]
        self.part = model.parts[self.sheet.part_id]
        self.surface = self.face.surface
        if self.use.id not in self.sheet.face_use_ids:
            raise _Refusal('invalid_patch_ownership')
        if not isinstance(self.surface, Cylinder) or (
                self.face.parameterization is not None and self.face.parameterization is not self.surface):
            raise _Refusal('support_or_parameterization_out_of_scope', missing=True)
        loops = (self.face.loop, *self.face.holes)
        if len(loops) > 2 or len(self.use.loops) != len(loops):
            raise _Refusal('patch_loop_family', missing=True)
        if sum(len(loop) for loop in loops) > proof.policy.max_occurrences:
            raise _Refusal('patch_occurrence_budget')
        self.edges, self.vertices, self.coedges, self.incidence = {}, {}, {}, {}
        copied = 0
        for loop, coedges in zip(loops, self.use.loops):
            if len(loop) < 4 or len(loop) != len(coedges):
                raise _Refusal('patch_loop_topology')
            for oriented, key in zip(loop, coedges):
                proof.cancel('patch source copy')
                coedge = model.coedges[key]
                if (key in self.coedges or oriented.edge in self.edges
                        or coedge.face_use_id != self.use.id or coedge.edge_id != oriented.edge
                        or int(coedge.orientation) != (1 if oriented.forward else -1)):
                    raise _Refusal('patch_occurrence_topology')
                edge = model.edges[oriented.edge]
                if not isinstance(edge.curve, (Straight, Arc)):
                    raise _Refusal('patch_curve_family', missing=True)
                self.coedges[key], self.edges[edge.id] = coedge, edge
                ids = (edge.start, edge.end, edge.curve.via_vertex) if isinstance(edge.curve, Arc) else (edge.start, edge.end)
                for key in ids:
                    if key not in self.vertices and len(self.vertices) >= proof.policy.max_vertices:
                        raise _Refusal('patch_vertex_budget')
                    self.vertices[key] = model.vertices[key]
                    copied += 1
                    if copied % 16 == 0:
                        proof.cancel('patch vertex copy')
            for a, b in zip(loop, loop[1:] + loop[:1]):
                ea, eb = self.edges[a.edge], self.edges[b.edge]
                if (ea.end if a.forward else ea.start) != (eb.start if b.forward else eb.end):
                    raise _Refusal('patch_open_loop')
        total = 0
        for key in sorted(self.edges):
            raw = model._edge_coedges.get(key, ())
            if len(raw) > proof.policy.max_external_incidences - total:
                raise _Refusal('patch_incidence_budget')
            total += len(raw)
            made = []
            for i, item in enumerate(raw):
                if i % 16 == 0:
                    proof.cancel('patch incidence copy')
                if model.coedges[item].edge_id != key:
                    raise _Refusal('patch_incidence_mismatch')
                made.append(item)
            self.incidence[key] = tuple(sorted(made))
        proof.counts.update(faces=1, coedges=len(self.coedges), edges=len(self.edges),
                            vertices=len(self.vertices), external_incidences=total)

    def handle(self, kind, identifier):
        return EntityHandle(self.model_id, kind, identifier)


class _PatchGeometry:
    """Patch-local branch/frame proof, not a quarter-sector atlas adapter."""
    def __init__(self, source, tolerance, proof):
        self.source, self.surface, self.p = source, source.surface, proof
        p, s = proof, self.surface
        self.anchor = tuple(_q(x) for x in s.origin)
        raw = [tuple(_q(x) for x in v.position) for v in source.vertices.values()]
        raw.append(self.anchor)
        for vector, length in ((s.axis, s.height), (s.radial_direction, s.radius), (s.circumferential_direction, s.radius)):
            raw.append(tuple(a + _q(length)*_q(b) for a,b in zip(self.anchor, vector)))
        extent = p.norm(tuple(p.sub(max(q[i] for q in raw), min(q[i] for q in raw)) for i in range(3)))
        self.extent = p.out(extent)[1]
        if self.extent <= 0 or not math.isfinite(self.extent):
            raise _Refusal('invalid_patch_extent')
        exponent = math.frexp(max(self.extent, s.radius, abs(s.height)))[1]
        if exponent > 1023:
            raise _Refusal('normalization_unrepresentable')
        self.scale = _q(math.ldexp(1., exponent))
        self.tau = _q(tolerance.effective_length(self.extent))/self.scale
        self.residual_limit = _q(tolerance.effective_surface_residual(self.extent))/self.scale
        self.angular = _q(tolerance.angular)
        self.parameter = _q(tolerance.parameter)
        self.radius, self.height = _q(s.radius)/self.scale, _q(s.height)/self.scale
        self.basis = tuple(self.vector(v) for v in (s.radial_direction, s.circumferential_direction, s.axis))
        r,c,a = self.basis
        determinant = p.dot(r,p.cross(c,a))
        if p.square(determinant).lo <= Fraction(1,2**40):
            raise _Refusal('ill_conditioned_patch_frame')
        self.inverse = tuple(p.vmul(v,p.div(1,determinant)) for v in (p.cross(c,a),p.cross(a,r),p.cross(r,c)))
        proof.counts['frame_tests'] += 1
        pi = p.pi_bound()
        width = abs(_q(s.sweep_angle))
        self.start, self.end = _q(s.start_angle), _q(s.start_angle)+_q(s.sweep_angle)
        if max(abs(self.start),abs(self.end)) > 8*pi.lo:
            raise _Refusal('patch_angle_range', missing=True)
        if width > Fraction(15,8)*pi.lo+self.angular:
            raise _Refusal('patch_span_out_of_scope', missing=True)
        gap = p.sub(p.mul(2,pi),width)
        if width <= self.angular or min(self.radius,abs(self.height)) <= self.tau or gap.lo <= self.angular or gap.lo*self.radius <= self.tau:
            raise _Refusal('patch_not_strictly_nonperiodic')
        self.low, self.high = min(self.start,self.end),max(self.start,self.end)
        self.max_residual = Fraction(0)
        self.max_width = Fraction(0)
        self.frames, self.frame_errors = {}, {}

    def vector(self, value):
        return tuple(self.p.i(_q(x)) for x in value)

    def point(self, value):
        return tuple(self.p.i((_q(x)-a)/self.scale) for x,a in zip(value,self.anchor))

    def coordinates(self, point):
        return tuple(self.p.dot(point,row) for row in self.inverse)

    def accept(self, bound):
        if bound.hi > self.residual_limit:
            raise _Refusal('patch_surface_residual')
        self.max_residual = max(self.max_residual,bound.hi*self.scale)

    def observe_uv_width(self, boxes):
        """Enclose the width of the actual outward-float evidence in length units."""
        p,s=self.p,self.surface
        scales=(_q(s.radius)*_q(s.sweep_angle),_q(s.height))
        physical=tuple(p.mul(p.i(*p.out(box)),scale) for box,scale in zip(boxes,scales))
        bound=p.norm(tuple(p.i(b.hi-b.lo) for b in physical))
        self.max_width=max(self.max_width,bound.hi)

    def coordinate_errors(self, residual):
        p = self.p
        radial = p.norm((*self.inverse[0],*self.inverse[1]))
        ratio = p.div(p.mul(radial,residual),self.radius)
        if ratio.lo < 0 or ratio.hi >= 1:
            raise _Refusal('patch_inverse_error_unqualified')
        angular = p.atan(p.div(ratio,p.sqrt(p.sub(1,p.square(ratio)))))
        return angular,p.mul(p.norm(self.inverse[2]),residual)

    def lift(self, x, y):
        """Split the principal cut, then certify one local lift component."""
        p = self.p
        if x.hi < 0 and y.lo < 0 < y.hi:
            principals = (p.atan2(p.i(y.lo,0),x),p.atan2(p.i(0,y.hi),x))
        else:
            principals = (p.atan2(y,x),)
        period = p.mul(2,p.pi_bound())
        if len(principals)==2:
            # The two principal images meet at the negative radial ray. Lift
            # the negative image once before enumerating the 17 local periods;
            # never average values on opposite sides of the principal cut.
            negative=p.add(principals[0],period)
            positive=principals[1]
            if negative.lo>positive.hi or positive.lo>negative.hi:
                raise _Refusal('patch_cut_images_not_connected')
            principals=(_hull((negative,positive)),)
        candidates = []
        for angle in principals:
            for k in range(-8,9):
                p.step('lift_candidates',p.policy.max_interval_operations)
                lifted = p.add(angle,p.mul(k,period))
                if lifted.hi < self.low-self.angular or lifted.lo > self.high+self.angular:
                    continue
                candidates.append((lifted,k))
        if not candidates:
            raise _Refusal('patch_no_admissible_lift')
        candidates.sort(key=lambda item:item[0].lo)
        combined = candidates[0][0]
        for item,k in candidates[1:]:
            # Both cut-side interval images must overlap on the same branch.
            if item.lo > combined.hi:
                raise _Refusal('patch_multiple_admissible_lifts')
            combined = _I(min(combined.lo,item.lo),max(combined.hi,item.hi))
        if combined.hi-combined.lo >= period.lo-(self.high-self.low):
            raise _Refusal('patch_lift_gap_unqualified')
        return combined,candidates[0][1]

    def vertex(self, key):
        p,s = self.p,self.surface
        x,y,z = self.coordinates(self.point(self.source.vertices[key].position))
        angle,lift = self.lift(x,y)
        uv = (p.div(p.sub(angle,s.start_angle),s.sweep_angle),p.div(z,self.height))
        if any(v.lo < -self.parameter or v.hi > 1+self.parameter for v in uv):
            raise _Refusal('patch_trim_outside_support')
        return angle,z,uv,lift

    def frame(self, edge):
        if edge.id in self.frames:
            return self.frames[edge.id]
        p = self.p
        start,via,end = (self.source.vertices[k].position for k in
                         (edge.start,edge.curve.via_vertex,edge.end))
        a,b,c = (self.point(v) for v in (start,via,end))
        ab,ac = p.vsub(b,a),p.vsub(c,a)
        cross = p.cross(ab,ac)
        cross2 = p.dot(cross,cross)
        ab2,ac2 = p.dot(ab,ab),p.dot(ac,ac)
        denominator = p.mul(ab2,ac2)
        if denominator.lo <= 0 or p.div(cross2,denominator).lo <= max(Fraction(1,2**40),_q(_COLLINEAR_RTOL)):
            raise _Refusal('patch_arc_conditioning')
        center = p.vadd(a,p.vmul(p.vadd(p.vmul(p.cross(ac,cross),ab2),
                                      p.vmul(p.cross(cross,ab),ac2)),p.div(1,p.mul(2,cross2))))
        radial = p.vsub(a,center)
        radius = p.norm(radial)
        e1 = p.vmul(radial,p.div(1,radius))
        normal = p.vmul(cross,p.div(1,p.sqrt(cross2)))
        e2 = p.cross(normal,e1)
        def positive_angle(point):
            value = p.vsub(point,center)
            angle = p.atan2(p.dot(value,e2),p.dot(value,e1))
            if angle.hi < 0: return p.add(angle,p.mul(2,p.pi_bound()))
            if angle.lo <= 0: raise _Refusal('patch_arc_source_angle')
            return angle
        via_angle,end_angle = positive_angle(b),positive_angle(c)
        if via_angle.hi < end_angle.lo: exact_sweep = end_angle
        elif via_angle.lo > end_angle.hi: exact_sweep = p.sub(end_angle,p.mul(2,p.pi_bound()))
        else: raise _Refusal('patch_arc_source_sweep')
        frame = arc_frame(start,via,end)
        if (not isinstance(frame,ArcFrame) or not math.isfinite(frame.radius)
                or not math.isfinite(frame.sweep) or frame.radius <= 0
                or any(np.shape(getattr(frame,n)) != (3,) or not np.all(np.isfinite(getattr(frame,n)))
                       for n in ('center','e1','e2','normal'))):
            raise _Refusal('patch_malformed_arc_frame')
        error = p.norm(p.vsub(self.point(frame.center),center))
        for supplied,certified in ((frame.e1,e1),(frame.e2,e2)):
            error = p.add(error,p.norm(p.vsub(p.vmul(self.vector(supplied),_q(frame.radius)/self.scale),
                                            p.vmul(certified,radius))))
        error = p.add(error,p.mul(radius,_upper_abs(p.sub(frame.sweep,exact_sweep))))
        self.accept(error)
        if abs(_q(frame.sweep)) > p.pi_bound().lo/2+self.angular:
            raise _Refusal('patch_nonminor_arc',missing=True)
        self.frames[edge.id],self.frame_errors[edge.id] = frame,error
        return frame

    def curve(self, edge):
        p,s = self.p,self.surface
        first,last = self.vertex(edge.start),self.vertex(edge.end)
        angle0,z0 = first[:2]
        r,c,axis = self.basis
        if isinstance(edge.curve,Straight):
            sine,cosine = p.sincos(angle0)
            radial = p.vmul(p.vadd(p.vmul(r,cosine),p.vmul(c,sine)),self.radius)
            errors = []
            for key,z in ((edge.start,z0),(edge.end,last[1])):
                expected = p.vadd(radial,p.vmul(axis,z))
                errors.append(p.norm(p.vsub(self.point(self.source.vertices[key].position),expected)))
            residual = _hull(errors)
            dz = p.sub(last[1],z0)
            if dz.lo <= 0 <= dz.hi or min(abs(dz.lo),abs(dz.hi)) <= self.tau:
                raise _Refusal('patch_nonaxial_or_degenerate_line',missing=True)
            slope = p.i(0)
            carrier = 'AXIAL'
            c0,c1 = (self.coordinates(self.point(self.source.vertices[k].position)) for k in (edge.start,edge.end))
            delta = p.vsub(c1,c0)
            radial_error = p.mul(p.norm((*self.inverse[0],*self.inverse[1])),residual)
            lower_radius = p.sub(self.radius,radial_error)
            if lower_radius.lo <= 0: raise _Refusal('patch_line_radial_gap')
            numerator = p.sub(p.mul(c0[0],delta[1]),p.mul(c0[1],delta[0]))
            minor = p.div(p.mul(self.radius,_upper_abs(numerator)),p.square(lower_radius))
            major = min(abs(delta[2].lo),abs(delta[2].hi))
        else:
            frame = self.frame(edge)
            center = self.point(frame.center)
            center_coords = self.coordinates(center)
            e1,e2 = self.vector(frame.e1),self.vector(frame.e2)
            e1coords = self.coordinates(e1)
            angle0,_ = self.lift(e1coords[0],e1coords[1])
            sign_box = p.dot(self.vector(frame.normal),axis)
            if sign_box.lo > 0: sign=1
            elif sign_box.hi < 0: sign=-1
            else: raise _Refusal('patch_arc_orientation')
            slope = p.i(sign*_q(frame.sweep))
            sine,cosine = p.sincos(angle0)
            expected_cos = p.vmul(p.vadd(p.vmul(r,cosine),p.vmul(c,sine)),self.radius)
            expected_sin = p.vmul(p.vadd(p.vmul(r,p.neg(sine)),p.vmul(c,cosine)),sign*self.radius)
            residual = self.frame_errors[edge.id]
            for delta in (p.vsub(center,p.vmul(axis,center_coords[2])),
                          p.vsub(p.vmul(e1,_q(frame.radius)/self.scale),expected_cos),
                          p.vsub(p.vmul(e2,_q(frame.radius)/self.scale),expected_sin)):
                residual = p.add(residual,p.norm(delta))
            aa = self.coordinates(p.vmul(e1,_q(frame.radius)/self.scale))
            bb = self.coordinates(p.vmul(e2,_q(frame.radius)/self.scale))
            def det(a,b): return p.sub(p.mul(a[0],b[1]),p.mul(a[1],b[0]))
            core = p.mul(sign,det(aa,bb))
            variation = _upper_abs(det(center_coords,aa))+_upper_abs(det(center_coords,bb))
            if core.lo <= variation or abs(slope.mid) <= self.angular:
                raise _Refusal('patch_arc_monotonicity')
            self.accept(residual)
            angle_error,_ = self.coordinate_errors(residual)
            predicted = p.add(angle0,slope)
            padding = angle_error.hi+self.angular
            if _upper_abs(p.sub(first[0],angle0)) > padding or _upper_abs(p.sub(last[0],predicted)) > padding:
                raise _Refusal('patch_arc_endpoint_branch')
            via = self.vertex(edge.curve.via_vertex)[0]
            direction = 1 if slope.lo > 0 else -1
            if p.mul(direction,p.sub(via,angle0)).lo <= 0 or p.mul(direction,p.sub(predicted,via)).lo <= 0:
                raise _Refusal('patch_arc_via_branch')
            z0,dz = center_coords[2],p.i(0)
            carrier = 'CIRCULAR'
            radial_upper = p.add(p.norm(center_coords[:2]),p.add(p.norm(aa[:2]),p.norm(bb[:2])))
            angular_lower = p.div(p.mul(p.sub(core,variation),abs(_q(frame.sweep))),p.square(radial_upper))
            major = p.mul(self.radius,angular_lower).lo
            minor = p.mul(_upper_abs(aa[2])+_upper_abs(bb[2]),abs(_q(frame.sweep)))
        self.accept(residual)
        # Whole-curve derivative cones at shared corners: adjacent orthogonal
        # sides diverge from the actual common Vertex, not merely from tubes.
        if major <= 0 or minor.hi >= major/16:
            raise _Refusal('patch_carrier_derivative_cone')
        angle_error,z_error = self.coordinate_errors(residual)
        error = (angle_error.hi/abs(_q(s.sweep_angle)),z_error.hi/abs(self.height))
        endpoint_boxes = []
        for theta,z in ((angle0,z0),(p.add(angle0,slope),p.add(z0,dz))):
            values=(p.div(p.sub(theta,s.start_angle),s.sweep_angle),p.div(z,self.height))
            endpoint_boxes.append(tuple(p.add(v,p.i(-e,e)) for v,e in zip(values,error)))
        for boxes in endpoint_boxes:
            self.observe_uv_width(boxes)
        return dict(edge=edge,carrier=carrier,angle0=angle0,z0=z0,slope=slope,dz=dz,
                    residual=residual,error=error,endpoints=tuple(endpoint_boxes),
                    source_endpoints=(first,last))


def _rectangle(source, geometry, proof, curves, loop_index):
    """Four monotone side cones plus disjoint nonadjacent carrier bands."""
    p = proof
    identifiers = source.use.loops[loop_index]
    rows = []
    vertices = []
    for key in identifiers:
        p.step('pair_tests',p.policy.max_pair_tests)
        use = source.coedges[key]
        data = curves[use.edge_id]
        forward = int(use.orientation)==1
        first,last = data['endpoints'] if forward else tuple(reversed(data['endpoints']))
        edge = data['edge']
        vertices.append(edge.start if forward else edge.end)
        axis = 0 if data['carrier']=='CIRCULAR' else 1
        delta = p.sub(last[axis],first[axis])
        if delta.lo > 0: sign=1
        elif delta.hi < 0: sign=-1
        else: raise _Refusal('patch_side_progression')
        band = _hull((first[1-axis],last[1-axis]))
        rows.append(dict(key=key,axis=axis,sign=sign,band=band,first=first,last=last))
    if len(set(vertices)) != len(vertices):
        raise _Refusal('patch_repeated_loop_vertex')
    starts = [i for i,row in enumerate(rows) if (row['axis'],row['sign']) !=
              (rows[i-1]['axis'],rows[i-1]['sign'])]
    if len(starts)!=4:
        raise _Refusal('patch_nonrectangular_loop',missing=True)
    rows=rows[starts[0]:]+rows[:starts[0]]
    groups=[]
    for row in rows:
        if not groups or (groups[-1][0]['axis'],groups[-1][0]['sign'])!=(row['axis'],row['sign']):
            groups.append([])
        groups[-1].append(row)
    if len(groups)!=4 or any(groups[i][0]['axis']==groups[(i+1)%4][0]['axis'] for i in range(4)):
        raise _Refusal('patch_rectangle_side_order')
    if any(groups[i][0]['sign'] != -groups[i+2][0]['sign'] for i in range(2)):
        raise _Refusal('patch_rectangle_side_orientation')
    bands=[]
    for group in groups:
        band=_hull(tuple(row['band'] for row in group))
        axis=group[0]['axis']
        error=max(data['error'][1-axis] for data in curves.values())
        if band.hi-band.lo > 4*(geometry.parameter+error):
            raise _Refusal('patch_side_not_one_carrier')
        bands.append((axis,band))
    xs=sorted((b for axis,b in bands if axis==1),key=lambda b:b.mid)
    ys=sorted((b for axis,b in bands if axis==0),key=lambda b:b.mid)
    bounds=(xs[0],xs[1],ys[0],ys[1])
    physical_scales=(abs(geometry.radius*_q(geometry.surface.sweep_angle)),abs(geometry.height))
    for low,high,scale in ((xs[0],xs[1],physical_scales[0]),(ys[0],ys[1],physical_scales[1])):
        p.step('pair_tests',p.policy.max_pair_tests)
        gap=high.lo-low.hi
        if gap <= geometry.parameter or gap*scale <= 2*geometry.tau:
            raise _Refusal('patch_opposite_sides_touch')
    # Sign of the first two orthogonal progress directions is winding sign.
    a,b=groups[0][0],groups[1][0]
    winding=a['sign']*b['sign']*(1 if a['axis']==0 else -1)
    return dict(index=loop_index,coedges=tuple(identifiers),bounds=bounds,winding=winding)


def _material(source, geometry, proof, curves):
    loops=tuple(_rectangle(source,geometry,proof,curves,i) for i in range(len(source.use.loops)))
    p=proof
    outer=loops[0]['bounds']
    for value in outer:
        if value.lo < -geometry.parameter or value.hi > 1+geometry.parameter:
            raise _Refusal('patch_material_outside_support')
    if len(loops)==2:
        inner=loops[1]['bounds']
        if loops[0]['winding'] != -loops[1]['winding']:
            raise _Refusal('patch_hole_winding')
        for i,scale in ((0,abs(geometry.radius*_q(geometry.surface.sweep_angle))),
                        (2,abs(geometry.height))):
            for gap in (inner[i].lo-outer[i].hi,outer[i+1].lo-inner[i+1].hi):
                p.step('pair_tests',p.policy.max_pair_tests)
                if gap<=geometry.parameter or gap*scale<=2*geometry.tau:
                    raise _Refusal('patch_hole_clearance')
        outer_seed=((outer[0].hi+inner[0].lo)/2,(outer[2].hi+outer[3].lo)/2)
        inner_seed=((inner[0].hi+inner[1].lo)/2,(inner[2].hi+inner[3].lo)/2)
        seeds=(outer_seed,inner_seed)
    else:
        seeds=(((outer[0].hi+outer[1].lo)/2,(outer[2].hi+outer[3].lo)/2),)
    for loop,seed in zip(loops,seeds):
        loop['seed']=seed
        # Material seed for outer, void seed for the hole; never a mesh node.
        for row in curves.values():
            axis=0 if row['carrier']=='CIRCULAR' else 1
            band=_hull(tuple(e[1-axis] for e in row['endpoints']))
            if band.lo<=seed[1-axis]<=band.hi:
                # Collinear extension is harmless only beyond segment range.
                extent=_hull(tuple(e[axis] for e in row['endpoints']))
                if extent.lo<=seed[axis]<=extent.hi:
                    raise _Refusal('patch_seed_on_carrier')
    return loops


def _encoded(value, proof=None, omit_digest=False):
    budget=_EvidenceBudget(proof)
    canonical=_typed(value,type(value),budget)
    if omit_digest: canonical.pop('digest')
    chunks=[]
    size=0
    for i,chunk in enumerate(json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(canonical)):
        if len(chunk)>3200000-size:
            raise _error(CylinderPatchErrorCode.INVALID_RESULT,'encoded evidence budget')
        block=chunk.encode('utf-8')
        size+=len(block)
        if size>3200000: raise _error(CylinderPatchErrorCode.INVALID_RESULT,'encoded evidence budget')
        chunks.append(block)
        if proof is not None and i%16==0: proof.cancel('patch evidence encoding')
    if proof is not None:
        proof.counts['encoding_nodes']+=budget.nodes
        proof.counts['encoding_bytes']+=size
        if proof.counts['encoding_nodes']>200000 or proof.counts['encoding_bytes']>3200000:
            raise _error(CylinderPatchErrorCode.INVALID_RESULT,'aggregate evidence budget')
    return b''.join(chunks)


def _certificate(model, proof, geometry, complete):
    extent=geometry.extent if geometry is not None else 0.
    def upper(value):
        rounded=float(value)
        return math.nextafter(rounded,math.inf) if _q(rounded)<value else rounded
    return CylinderPatchCertificate('cylinder-partial-rectangle-v1',complete,
        float(model.tolerance.effective_length(extent)),float(model.tolerance.effective_surface_residual(extent)),
        float(model.tolerance.angular),upper(geometry.max_residual) if geometry else 0.,
        upper(geometry.max_width) if geometry else 0.,tuple(sorted(proof.counts.items())))


def _seal(value, model, proof, geometry, *, omit_digest=False):
    """Include final encoding work in the immutable receipt, without recursion.

    Encoding work depends on the shape and decimal widths of the counters. Each
    bounded sizing pass is real, charged work. Predict one further pass, then
    accept only if its measured counters equal that prediction exactly. A digit
    boundary may require another pass; uncertainty never produces a receipt.
    """
    delta=None
    for _ in range(8):
        if delta is not None:
            predicted={key:count+delta[key] for key,count in proof.counts.items()}
            certificate=replace(_certificate(model,proof,geometry,value.certificate.complete),
                                work_counts=tuple(sorted(predicted.items())))
            value=replace(value,certificate=certificate)
        before=dict(proof.counts)
        encoded=_encoded(value,proof,omit_digest)
        proof.cancel('patch receipt complete')
        _fresh(model,value.revision)
        if delta is not None and dict(value.certificate.work_counts)==proof.counts:
            return value,encoded
        delta={key:count-before[key] for key,count in proof.counts.items()}
    raise _error(CylinderPatchErrorCode.INVALID_RESULT,'receipt accounting did not converge')


def _run(model, face_uses, revision, policy, callback):
    policy=CylinderPatchPolicy() if policy is None else policy
    if type(policy) is not CylinderPatchPolicy:
        raise _error(CylinderPatchErrorCode.INVALID_REQUEST,'CylinderPatchPolicy required')
    replace(policy)
    selected=_selection(model,face_uses,revision,callback)
    p=_PatchProof(policy,callback)
    p.freshness_check=lambda: _fresh(model,revision)
    source=geometry=None
    curves={};loops=occurrences=()
    status=CylinderPatchStatus.QUALIFIED;diagnostics=()
    try:
        p.cancel('patch query')
        source=_PatchSource(model,selected,p)
        geometry=_PatchGeometry(source,model.tolerance,p)
        for edge in source.edges.values():
            p.cancel('patch carrier')
            curves[edge.id]=geometry.curve(edge)
        rectangles=_material(source,geometry,p,curves)
        made=[]
        for rectangle in rectangles:
            physical=[]
            for i,box in enumerate(rectangle['bounds']):
                scale=_q(source.surface.radius)*_q(source.surface.sweep_angle) if i<2 else _q(source.surface.height)
                physical.append(p.out(p.mul(box,scale)))
            seed=tuple(float(x) for x in rectangle['seed'])
            geometry.observe_uv_width(tuple(p.i(x) for x in rectangle['seed']))
            # Bounds remain paired with u_min/u_max/v_min/v_max even for negative scales.
            made.append(CylinderPatchLoop(rectangle['index'],'OUTER' if rectangle['index']==0 else 'HOLE',
                tuple(source.handle('coedge',k) for k in rectangle['coedges']),rectangle['winding'],
                tuple(p.out(b) for b in rectangle['bounds']),tuple(physical),seed,
                tuple(p.out(p.i(x)) for x in rectangle['seed'])))
        loops=tuple(made);made=[]
        for index,keys in enumerate(source.use.loops):
            for key in keys:
                p.cancel('patch occurrence record')
                use=source.coedges[key];data=curves[use.edge_id];edge=data['edge']
                first,last=data['source_endpoints']
                made.append(CylinderPatchOccurrence(source.handle('coedge',key),source.handle('edge',edge.id),
                    source.handle('vertex',edge.start),source.handle('vertex',edge.end),index,
                    int(use.orientation)==1,int(source.use.orientation),data['carrier'],
                    (0.,1.) if int(use.orientation)==1 else (1.,0.),
                    tuple(tuple(float(b.mid) for b in endpoint) for endpoint in data['endpoints']),
                    tuple(tuple(p.out(b) for b in endpoint) for endpoint in data['endpoints']),
                    (p.out(first[0]),p.out(last[0])),(p.out(p.mul(first[1],geometry.scale)),p.out(p.mul(last[1],geometry.scale))),
                    (first[3],last[3]),p.out(p.mul(data['residual'],geometry.scale))[1],
                    tuple(source.handle('coedge',i) for i in source.incidence[edge.id] if i not in source.coedges)))
        occurrences=tuple(made)
    except _Refusal as error:
        if error is p.callback_error: raise
        status=CylinderPatchStatus.CAPABILITY_MISSING if error.missing else CylinderPatchStatus.UNRESOLVED
        diagnostics=(str(error),);loops=occurrences=()
    qualified=status is CylinderPatchStatus.QUALIFIED
    s=source.surface if qualified else None
    result=CylinderPatchResult(1,model.model_id,revision,selected,policy,status,
        source.handle('part',source.part.id) if source else None,
        source.handle('sheet',source.sheet.id) if source else None,
        source.handle('face',source.face.id) if source else None,selected[0],
        int(source.use.orientation) if qualified else None,
        tuple(float(x) for x in s.origin) if s else None,
        tuple(float(x) for x in s.axis) if s else None,
        tuple(float(x) for x in s.radial_direction) if s else None,
        float(s.radius) if s else None,float(s.height) if s else None,
        float(s.start_angle) if s else None,float(s.sweep_angle) if s else None,
        (1 if s.sweep_angle>0 else -1) if s else None,(1 if s.height>0 else -1) if s else None,
        (float(s.radius*s.sweep_angle),float(s.height)) if s else None,
        (float(1/(s.radius*s.sweep_angle)),float(1/s.height)) if s else None,
        loops,occurrences,_certificate(model,p,geometry,qualified),diagnostics,'')
    result,encoded=_seal(result,model,p,geometry,omit_digest=True)
    result=replace(result,digest=hashlib.sha256(encoded).hexdigest())
    return result,source,geometry,curves,p


def query_cylinder_patch(model, face_uses, *, expected_revision, policy=None, cancellation_check=None):
    """Qualify one existing nonperiodic FaceUse, never fabricate a sector family."""
    return _run(model,face_uses,expected_revision,policy,cancellation_check)[0]


def _bound(model,result,face_uses,revision,callback):
    if type(result) is not CylinderPatchResult:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'CylinderPatchResult required')
    selected=_selection(model,face_uses,revision,callback)
    if result.model_id!=model.model_id:
        raise _error(CylinderPatchErrorCode.WRONG_MODEL,'foreign patch result')
    if result.revision!=revision:
        raise _error(CylinderPatchErrorCode.STALE_RESULT,'stale patch result')
    if result.requested_face_uses!=selected:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'different caller FaceUse')
    if type(result.policy) is not CylinderPatchPolicy:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'invalid result policy')
    replace(result.policy)
    budget_proof=_PatchProof(result.policy,callback)
    budget_proof.freshness_check=lambda: _fresh(model,revision)
    if hashlib.sha256(_encoded(result,budget_proof,True)).hexdigest()!=result.digest:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'patch digest mismatch')
    fresh,source,geometry,curves,p=_run(model,selected,revision,result.policy,callback)
    if result.digest!=fresh.digest:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'patch differs from fresh qualification')
    # Binding validation is part of this call's aggregate evidence work, even
    # though the freshly reconstructed result retains its query-only receipt.
    for key in ('encoding_nodes','encoding_bytes','cancellation_checks'):
        p.counts[key]+=budget_proof.counts[key]
    if p.counts['encoding_nodes']>200000 or p.counts['encoding_bytes']>3200000:
        raise _error(CylinderPatchErrorCode.INVALID_RESULT,'aggregate binding evidence budget')
    p.cancel('patch binding complete')
    return fresh,source,geometry,curves,p


def validate_cylinder_patch_binding(model,result,face_uses,*,expected_revision,cancellation_check=None):
    """Digest integrity AND fresh geometry proof, not a revision-only cache."""
    _bound(model,result,face_uses,expected_revision,cancellation_check)


def evaluate_cylinder_patch_occurrences(model,result,requests,*,face_uses,expected_revision,cancellation_check=None):
    fresh,source,geometry,curves,p=_bound(model,result,face_uses,expected_revision,cancellation_check)
    if fresh.status is not CylinderPatchStatus.QUALIFIED:
        raise _error(CylinderPatchErrorCode.UNQUALIFIED_RESULT,'unqualified patch')
    made=[]
    for request in requests:
        if len(made)%16==0:
            p.cancel('patch request collection')
        if len(made)>=fresh.policy.max_evaluations:
            raise _error(CylinderPatchErrorCode.INVALID_REQUEST,'sample count budget')
        if type(request) is not CylinderPatchOccurrenceRequest:
            raise _error(CylinderPatchErrorCode.INVALID_REQUEST,'native occurrence request required')
        made.append(replace(request))
    samples=[]
    try:
        for request in made:
            p.step('evaluations',fresh.policy.max_evaluations)
            if request.coedge.model_id!=model.model_id or request.coedge.id not in source.coedges:
                raise _error(CylinderPatchErrorCode.INVALID_REQUEST,'occurrence not selected')
            coedge=source.coedges[request.coedge.id];data=curves[coedge.edge_id];edge=data['edge'];t=request.parameter
            endpoint=source.handle('vertex',edge.start if t==0 else edge.end) if t in (0.,1.) else None
            if endpoint: point=np.asarray(source.vertices[endpoint.id].position)
            elif isinstance(edge.curve,Arc): point=sample_arc(geometry.frame(edge),np.array([t]))[0]
            else: point=sample_straight(source.vertices[edge.start].position,source.vertices[edge.end].position,np.array([t]))[0]
            if np.shape(point)!=(3,) or not np.all(np.isfinite(point)): raise _Refusal('nonfinite_patch_sample')
            angle=p.add(data['angle0'],p.mul(t,data['slope']));z=p.add(data['z0'],p.mul(t,data['dz']))
            s=source.surface
            uv=(float(p.div(p.sub(angle,s.start_angle),s.sweep_angle).mid),float(p.div(z,geometry.height).mid))
            returned_angle=p.add(s.start_angle,p.mul(uv[0],s.sweep_angle));returned_z=p.mul(uv[1],geometry.height)
            sine,cosine=p.sincos(returned_angle);r,c,a=geometry.basis
            expected=p.vadd(p.vmul(p.vadd(p.vmul(r,cosine),p.vmul(c,sine)),geometry.radius),p.vmul(a,returned_z))
            residual=p.add(p.norm(p.vsub(geometry.point(point),expected)),data['residual'])
            geometry.accept(residual)
            angular,axial=geometry.coordinate_errors(residual)
            errors=(angular.hi/abs(_q(s.sweep_angle)),axial.hi/abs(geometry.height))
            boxes=tuple(p.add(u,p.i(-e,e)) for u,e in zip(uv,errors))
            geometry.observe_uv_width(boxes)
            physical=(float(p.mul(p.sub(returned_angle,s.start_angle),s.radius).mid),
                      float(p.mul(returned_z,geometry.scale).mid))
            scales=(_q(s.radius)*_q(s.sweep_angle),_q(s.height))
            physical_boxes=tuple(p.mul(b,k) for b,k in zip(boxes,scales))
            width=p.norm(tuple(p.i(b.hi-b.lo) for b in physical_boxes))
            geometry.max_width=max(geometry.max_width,width.hi)
            identity=[model.model_id.hex,expected_revision,fresh.digest,
                      ['vertex',endpoint.id] if endpoint else ['edge',edge.id,t.hex()]]
            samples.append(CylinderPatchSample(request.coedge,source.handle('edge',edge.id),t,
                tuple(float(x) for x in point),uv,physical,tuple(p.out(b) for b in boxes),tuple(p.out(b) for b in physical_boxes),
                endpoint,hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest(),
                p.out(p.mul(residual,geometry.scale))[1],CylinderPatchStatus.QUALIFIED))
    except _Refusal as error:
        if error is p.callback_error: raise
        raise _error(CylinderPatchErrorCode.UNQUALIFIED_RESULT,str(error)) from error
    p.cancel('patch samples complete');_fresh(model,expected_revision)
    evaluation=CylinderPatchEvaluation(model.model_id,expected_revision,fresh.digest,tuple(samples),_certificate(model,p,geometry,True))
    return _seal(evaluation,model,p,geometry)[0]
