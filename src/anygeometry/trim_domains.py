"""Bounded owner evidence for material-interior relations of planar trims.

The analytic path proves a deliberately restricted, topology-identical hole/fill
family. It is not a curved overlay or a sampled-area approximation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import StrEnum
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType
from uuid import UUID

import numpy as np

from .curves import Arc, Straight, _COLLINEAR_RTOL, arc_frame
from .errors import GeometryError
from .identity import EntityHandle
from .intersections import query_intersection
from .model import GeometryModel
from .predicates import IntersectionDimension, IntersectionKind
from .surfaces import Plane

__all__ = [
    "TrimBoundaryContact", "TrimDomainError", "TrimDomainErrorCode",
    "TrimDomainResult", "TrimInteriorRelation", "query_trim_domain_relation",
    "validate_trim_domain_binding",
]


class TrimInteriorRelation(StrEnum):
    DISJOINT_INTERIORS = "DISJOINT_INTERIORS"
    OVERLAPPING_INTERIORS = "OVERLAPPING_INTERIORS"
    UNRESOLVED = "UNRESOLVED"


class TrimBoundaryContact(StrEnum):
    NONE = "NONE"
    POINT = "POINT"
    CURVE = "CURVE"
    UNKNOWN = "UNKNOWN"


class TrimDomainErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    WRONG_MODEL = "WRONG_MODEL"
    INACTIVE_ENTITY = "INACTIVE_ENTITY"
    STALE_REVISION = "STALE_REVISION"
    STALE_RESULT = "STALE_RESULT"
    INVALID_RESULT = "INVALID_RESULT"
    BUSY_MODEL = "BUSY_MODEL"


class TrimDomainError(GeometryError):
    """Request/binding failure; operational exceptions are never wrapped here."""

    def __init__(self, code: TrimDomainErrorCode, diagnostics: tuple[str, ...]):
        object.__setattr__(self, "_code", TrimDomainErrorCode(code))
        object.__setattr__(self, "_diagnostics", tuple(diagnostics))
        super().__init__(f"{self.code.value}: {'; '.join(self.diagnostics)}")

    @property
    def code(self):
        return self._code

    @property
    def diagnostics(self):
        return self._diagnostics

    def __setattr__(self, name, value):
        if name in ("code", "diagnostics", "_code", "_diagnostics"):
            raise AttributeError("trim error evidence is immutable")
        super().__setattr__(name, value)


_COUNT_KEYS = (
    "copied_faces", "copied_edges", "copied_vertices", "ring_edges",
    "halfspace_tests", "winding_edges", "rational_operations", "owner_queries",
)
_ARC_CONDITIONING_MINIMUM = max(Fraction(1, 2**40), Fraction.from_float(_COLLINEAR_RTOL))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _error(code, diagnostic):
    return TrimDomainError(code, (diagnostic,))


def _canonical(value):
    if isinstance(value, EntityHandle):
        return {"model_id": value.model_id.hex, "kind": value.kind, "id": value.id}
    if isinstance(value, UUID):
        return value.hex
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def _digest(data):
    return hashlib.sha256(json.dumps(
        _canonical(data), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrimDomainResult:
    model_id: UUID
    revision: int
    first_parent: EntityHandle
    second_parent: EntityHandle
    relation: TrimInteriorRelation
    complementary: bool
    boundary: TrimBoundaryContact
    shared_boundary: tuple[EntityHandle, ...]
    algorithm: str
    tolerance: float
    max_residual: float
    diagnostics: tuple[str, ...]
    work_counts: Mapping[str, int]
    digest: str

    def __post_init__(self):
        def invalid():
            raise _error(TrimDomainErrorCode.INVALID_RESULT, "malformed trim evidence")

        if not isinstance(self.model_id, UUID) or not self.model_id.int:
            invalid()
        if not _integer(self.revision) or self.revision < 0:
            invalid()
        if any(not isinstance(p, EntityHandle) or p.kind != "face"
               or p.model_id != self.model_id
               for p in (self.first_parent, self.second_parent)):
            invalid()
        if self.first_parent == self.second_parent:
            invalid()
        if not isinstance(self.relation, TrimInteriorRelation) or not isinstance(self.boundary, TrimBoundaryContact):
            invalid()
        if type(self.complementary) is not bool:
            invalid()
        if type(self.shared_boundary) is not tuple or any(
            not isinstance(h, EntityHandle) or h.kind != "edge" or h.model_id != self.model_id
            for h in self.shared_boundary
        ):
            invalid()
        if tuple(sorted(set(self.shared_boundary))) != self.shared_boundary:
            invalid()
        if not isinstance(self.algorithm, str) or not self.algorithm:
            invalid()
        if type(self.diagnostics) is not tuple or any(not isinstance(d, str) for d in self.diagnostics):
            invalid()
        for name in ("tolerance", "max_residual"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating)):
                invalid()
            if not math.isfinite(value) or value < 0 or (name == "tolerance" and value == 0):
                invalid()
            object.__setattr__(self, name, float(value))
        if self.complementary and (
            self.relation is not TrimInteriorRelation.DISJOINT_INTERIORS
            or self.boundary is not TrimBoundaryContact.CURVE or not self.shared_boundary
        ):
            invalid()
        if self.relation is TrimInteriorRelation.UNRESOLVED and not self.diagnostics:
            invalid()
        if self.relation is not TrimInteriorRelation.UNRESOLVED and self.max_residual > self.tolerance:
            invalid()
        if not isinstance(self.work_counts, Mapping) or set(self.work_counts) != set(_COUNT_KEYS):
            invalid()
        if any(not _integer(v) or v < 0 for v in self.work_counts.values()):
            invalid()
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "work_counts", MappingProxyType({k: int(v) for k, v in self.work_counts.items()}))
        if not isinstance(self.digest, str) or len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            invalid()
        if self.digest != _digest(_result_data(self)):
            invalid()


def _result_data(result):
    return {field.name: getattr(result, field.name) for field in fields(TrimDomainResult) if field.name != "digest"}


class _Unresolved(Exception):
    """Only locally established inability; never used to catch API failures."""


def _request(model, first, second, revision, callback):
    if not isinstance(model, GeometryModel):
        raise TypeError("trim queries need a GeometryModel")
    if callback is not None and not callable(callback):
        raise TypeError("cancellation_check must be callable")
    if (not _integer(revision) or revision < 0
            or any(not isinstance(p, EntityHandle) or p.kind != "face" for p in (first, second))
            or first == second):
        raise _error(TrimDomainErrorCode.INVALID_REQUEST, "distinct face handles and integer revision required")
    if first.model_id != model.model_id or second.model_id != model.model_id:
        raise _error(TrimDomainErrorCode.WRONG_MODEL, "parents belong to another model")
    if revision != model.revision:
        raise _error(TrimDomainErrorCode.STALE_REVISION, "expected revision differs from live model")
    if first.id not in model.faces or second.id not in model.faces:
        raise _error(TrimDomainErrorCode.INACTIVE_ENTITY, "parent is not active")
    _unchanged(model, revision)


def _unchanged(model, revision):
    if model._transaction_journal is not None:  # owner-internal read, never modified
        raise _error(TrimDomainErrorCode.BUSY_MODEL, "trim evidence requires committed state")
    if model.revision != revision:
        raise _error(TrimDomainErrorCode.STALE_REVISION, "model changed during qualification")


class _Proof:
    def __init__(self, callback):
        self.callback = callback
        self.counts = dict.fromkeys(_COUNT_KEYS, 0)
        self.tolerance = None

    def cancel(self):
        if self.callback is not None:
            self.callback("trim domain qualification")

    def q(self, value):
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise _Unresolved("nonfinite_geometry_unqualified")
        value = Fraction(value)
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 8192:
            raise _Unresolved("qualification_budget_exhausted")
        return value

    def op(self, a, b, operation):
        a, b = self.q(a), self.q(b)
        # Bound unreduced intermediates before multiplication/addition.
        bits = max(abs(a.numerator).bit_length(), a.denominator.bit_length()) + max(
            abs(b.numerator).bit_length(), b.denominator.bit_length()) + 1
        if bits > 8192 or self.counts["rational_operations"] >= 200_000:
            raise _Unresolved("qualification_budget_exhausted")
        self.counts["rational_operations"] += 1
        if self.counts["rational_operations"] % 256 == 0:
            self.cancel()
        if operation == "+":
            return self.q(a + b)
        if operation == "-":
            return self.q(a - b)
        if operation == "*":
            return self.q(a * b)
        if b == 0:
            raise _Unresolved("invalid_trim_geometry")
        return self.q(a / b)

    def add(self, a, b): return self.op(a, b, "+")
    def sub(self, a, b): return self.op(a, b, "-")
    def mul(self, a, b): return self.op(a, b, "*")
    def div(self, a, b): return self.op(a, b, "/")
    def square(self, a): return self.mul(a, a)
    def vadd(self, a, b): return tuple(self.add(x, y) for x, y in zip(a, b))
    def vsub(self, a, b): return tuple(self.sub(x, y) for x, y in zip(a, b))
    def vmul(self, a, k): return tuple(self.mul(x, k) for x in a)

    def dot(self, a, b):
        return self.add(self.add(self.mul(a[0], b[0]), self.mul(a[1], b[1])), self.mul(a[2], b[2]))

    def cross(self, a, b):
        return tuple(self.sub(self.mul(a[i], b[j]), self.mul(a[j], b[i])) for i, j in ((1, 2), (2, 0), (0, 1)))

    def det(self, normal, a, b): return self.dot(normal, self.cross(a, b))


def _copy_pair(model, parents, proof):
    faces = [model.faces[p.id] for p in parents]
    if any(len(f.holes) > 1 for f in faces):
        raise _Unresolved("extra_holes_out_of_scope")
    for f in faces:
        if not isinstance(f.surface, Plane) or f.parameterization is not None:
            raise _Unresolved("support_or_parameterization_out_of_scope")
        for loop in (f.loop, *f.holes):
            if len(loop) > 64:
                raise _Unresolved("qualification_budget_exhausted")
            if len(loop) < 3 or len({e.edge for e in loop}) != len(loop):
                raise _Unresolved("invalid_trim_topology")
    edge_ids = sorted({e.edge for f in faces for loop in (f.loop, *f.holes) for e in loop})
    if len(edge_ids) > 256:
        raise _Unresolved("qualification_budget_exhausted")
    edges = []
    vertex_ids = set()
    for index, key in enumerate(edge_ids):
        if index % 16 == 0:
            proof.cancel()
        edge = model.edges[key]  # Unexpected store/API failures propagate.
        if not isinstance(edge.curve, (Arc, Straight)):
            raise _Unresolved("curve_family_out_of_scope")
        edges.append(edge)
        vertex_ids.update((edge.start, edge.end))
        if isinstance(edge.curve, Arc):
            vertex_ids.add(edge.curve.via_vertex)
    if len(vertex_ids) > 512:
        raise _Unresolved("qualification_budget_exhausted")
    detached = GeometryModel(tolerance=model.tolerance)
    for kind, records in (
        ("vertex", (model.vertices[k] for k in sorted(vertex_ids))),
        ("edge", edges), ("face", faces),
    ):
        for record in records:
            detached._set_entity_unjournalled(kind, record.id, deepcopy(record))
    proof.counts.update(copied_faces=2, copied_edges=len(edges), copied_vertices=len(vertex_ids))
    for f in faces:
        for loop in (f.loop, *f.holes):
            if any(detached.oriented_end_vertex(item) != detached.oriented_start_vertex(loop[(i + 1) % len(loop)])
                   for i, item in enumerate(loop)):
                raise _Unresolved("invalid_trim_topology")
    return detached


def _common_plane(model, parents, proof):
    face = model.faces[parents[0].id]
    origin = tuple(proof.q(float(x)) for x in model.vertex_position(model.oriented_start_vertex(face.loop[0])))

    def vector(value): return tuple(proof.q(float(x)) for x in value)

    normal = proof.cross(vector(face.surface.u_vector), vector(face.surface.v_vector))
    if not any(normal):
        raise _Unresolved("common_plane_unqualified")
    if next(x for x in normal if x) < 0:
        normal = proof.vmul(normal, -1)
    points = {k: proof.vsub(vector(v.position), origin) for k, v in model.vertices.items()}
    for point in points.values():
        if proof.dot(normal, point) != 0:
            raise _Unresolved("common_plane_unqualified")
    for parent in parents:
        plane = model.faces[parent.id].surface
        if (proof.dot(normal, proof.vsub(vector(plane.origin), origin)) != 0
                or proof.dot(normal, vector(plane.u_vector)) != 0
                or proof.dot(normal, vector(plane.v_vector)) != 0):
            raise _Unresolved("common_plane_unqualified")
    return normal, points


def _same_ring(first, second):
    a = tuple((e.edge, e.forward) for e in first)
    b = tuple((e.edge, e.forward) for e in second)
    if len(a) != len(b):
        return False
    for candidate in (b, tuple((key, not direction) for key, direction in reversed(b))):
        if a[0] in candidate:
            start = candidate.index(a[0])
            if a == candidate[start:] + candidate[:start]:
                return True
    return False


def _length_clearance(proof, h, h2, tau):
    return h > 0 and proof.square(h) > proof.mul(proof.square(tau), h2)


def _circle_clearance(proof, outer2, inner2, tau):
    x = proof.sub(proof.sub(outer2, inner2), proof.square(tau))
    return x > 0 and proof.square(x) > proof.mul(4, proof.mul(inner2, proof.square(tau)))


def _complement(model, parents, normal, points, proof):
    first, second = (model.faces[p.id] for p in parents)
    if len(first.holes) == 1 and _same_ring(first.holes[0], second.loop):
        outer_face, fill = first, second
    elif len(second.holes) == 1 and _same_ring(second.holes[0], first.loop):
        outer_face, fill = second, first
    else:
        return None
    if fill.holes:
        raise _Unresolved("extra_holes_out_of_scope")
    if any(not isinstance(model.edges[e.edge].curve, Straight) for e in outer_face.loop):
        raise _Unresolved("enclosing_curve_family_out_of_scope")
    curve_types = {type(model.edges[e.edge].curve) for e in fill.loop}
    if curve_types not in ({Straight}, {Arc}):
        raise _Unresolved("mixed_ring_out_of_scope")
    outer = [points[model.oriented_start_vertex(e)] for e in outer_face.loop]
    ring = [points[model.oriented_start_vertex(e)] for e in fill.loop]
    proof.counts["ring_edges"] = len(outer) + len(ring)
    extent_vector = tuple(proof.sub(max(p[i] for p in outer), min(p[i] for p in outer)) for i in range(3))
    extent2 = proof.dot(extent_vector, extent_vector)
    if extent2 <= 0:
        raise _Unresolved("invalid_trim_geometry")
    # Conversion failures here are explicit representability limits, not an API fallback.
    if extent2 > Fraction.from_float(float(np.finfo(float).max)):
        raise _Unresolved("extent_representation_unqualified")
    extent = math.sqrt(float(extent2))
    for _ in range(4):
        if proof.square(proof.q(extent)) >= extent2:
            break
        extent = math.nextafter(extent, math.inf)
    if not math.isfinite(extent) or extent <= 0 or proof.square(proof.q(extent)) < extent2:
        raise _Unresolved("extent_representation_unqualified")
    proof.tolerance = model.tolerance.effective_length(extent)
    tau = proof.q(proof.tolerance)
    anchor = tuple(proof.div(sum_coordinate, len(ring)) for sum_coordinate in (
        _sum(proof, (p[i] for p in ring)) for i in range(3)))
    turn = proof.det(normal, proof.vsub(outer[1], outer[0]), proof.vsub(outer[2], outer[1]))
    if turn == 0:
        raise _Unresolved("invalid_trim_geometry")
    sign = 1 if turn > 0 else -1
    halfspaces = []
    tau2 = proof.square(tau)
    for i, p in enumerate(outer):
        d = proof.vsub(outer[(i + 1) % len(outer)], p)
        if proof.dot(d, d) <= tau2:
            raise _Unresolved("invalid_trim_geometry")
        inward = proof.vmul(proof.cross(normal, d), sign)
        h2 = proof.dot(inward, inward)
        for j, q in enumerate(outer):
            if j not in (i, (i + 1) % len(outer)):
                proof.counts["halfspace_tests"] += 1
                if not _length_clearance(proof, proof.dot(proof.vsub(q, p), inward), h2, tau):
                    raise _Unresolved("outer_convexity_unqualified")
        halfspaces.append((p, inward, h2))
    ring_sign = None
    x_axis = proof.vsub(outer[1], outer[0])
    y_axis = proof.cross(normal, x_axis)
    winding = 0
    for index, use in enumerate(fill.loop):
        if index % 16 == 0:
            proof.cancel()
        s, e = ring[index], ring[(index + 1) % len(ring)]
        chord = proof.vsub(e, s)
        if proof.dot(chord, chord) <= tau2:
            raise _Unresolved("invalid_trim_geometry")
        sa, ea = proof.vsub(s, anchor), proof.vsub(e, anchor)
        det = proof.det(normal, sa, ea)
        if det == 0:
            raise _Unresolved("polar_progression_unqualified")
        current_sign = 1 if det > 0 else -1
        if ring_sign is not None and ring_sign != current_sign:
            raise _Unresolved("polar_progression_unqualified")
        ring_sign = current_sign
        chord_normal = proof.cross(normal, chord)
        h = proof.dot(sa, chord_normal)
        if not _length_clearance(proof, abs(h), proof.dot(chord_normal, chord_normal), tau):
            raise _Unresolved("polar_progression_unqualified")
        sy, ey = proof.dot(sa, y_axis), proof.dot(ea, y_axis)
        if sy <= 0 < ey and det > 0:
            winding += 1
        elif ey <= 0 < sy and det < 0:
            winding -= 1
        proof.counts["winding_edges"] += 1
        if curve_types == {Straight}:
            for p, inward, h2 in halfspaces:
                proof.counts["halfspace_tests"] += 1
                if not _length_clearance(proof, proof.dot(proof.vsub(s, p), inward), h2, tau):
                    raise _Unresolved("ring_containment_unqualified")
            continue
        via = points[model.edges[use.edge].curve.via_vertex]
        a, b = proof.vsub(via, s), chord
        a2, b2 = proof.dot(a, a), proof.dot(b, b)
        k = proof.cross(a, b)
        k2 = proof.dot(k, k)
        if min(a2, b2, proof.dot(proof.vsub(e, via), proof.vsub(e, via))) <= tau2 or k2 == 0:
            raise _Unresolved("invalid_trim_geometry")
        if proof.div(k2, proof.mul(a2, b2)) <= _ARC_CONDITIONING_MINIMUM:
            raise _Unresolved("arc_conditioning_unqualified")
        # The exact proof must not certify a record the numerical owner cannot
        # evaluate. This pure call uses detached records; operational exceptions
        # (including injected evaluator failures) deliberately propagate unchanged.
        frame = arc_frame(
            model.vertex_position(model.oriented_start_vertex(use)),
            model.vertex_position(model.edges[use.edge].curve.via_vertex),
            model.vertex_position(model.oriented_end_vertex(use)),
        )
        if (not math.isfinite(frame.radius) or frame.radius <= 0
                or not math.isfinite(frame.sweep) or frame.sweep == 0
                or any(np.asarray(v).shape != (3,) or not np.all(np.isfinite(v))
                       for v in (frame.center, frame.e1, frame.e2, frame.normal))):
            raise _Unresolved("owner_arc_evaluation_unqualified")
        center = proof.vadd(s, proof.vmul(proof.vadd(
            proof.vmul(proof.cross(b, k), a2), proof.vmul(proof.cross(k, a), b2),
        ), proof.div(1, proof.mul(2, k2))))
        sc, vc, ec = (proof.vsub(p, center) for p in (s, via, e))
        radius2 = proof.dot(sc, sc)
        for lhs, rhs in ((sc, ec), (sc, vc), (vc, ec)):
            if proof.mul(ring_sign, proof.det(normal, lhs, rhs)) <= 0:
                raise _Unresolved("minor_arc_progression_unqualified")
        ac = proof.vsub(anchor, center)
        if not _circle_clearance(proof, radius2, proof.dot(ac, ac), tau):
            raise _Unresolved("star_anchor_unqualified")
        for p, inward, h2 in halfspaces:
            proof.counts["halfspace_tests"] += 1
            h = proof.dot(proof.vsub(center, p), inward)
            if h <= 0 or not _circle_clearance(proof, proof.div(proof.square(h), h2), radius2, tau):
                raise _Unresolved("full_circle_containment_unqualified")
    if winding != ring_sign:
        raise _Unresolved("one_turn_winding_unqualified")
    return tuple(sorted(use.edge for use in fill.loop))


def _sum(proof, values):
    total = Fraction(0)
    for value in values:
        total = proof.add(total, value)
    return total


def query_trim_domain_relation(
    model: GeometryModel, first: EntityHandle, second: EntityHandle, *,
    expected_revision: int,
    cancellation_check: Callable[[str], None] | None = None,
) -> TrimDomainResult:
    """Read-only, revision-bound evidence; see ``docs/TRIM_DOMAIN_CONTRACT.md``."""
    _request(model, first, second, expected_revision, cancellation_check)
    proof = _Proof(cancellation_check)
    proof.cancel()
    relation = TrimInteriorRelation.UNRESOLVED
    boundary = TrimBoundaryContact.UNKNOWN
    complementary = False
    shared = ()
    algorithm = "trim_domain_preflight_v1"
    residual = 0.0
    diagnostics = ()
    try:
        detached = _copy_pair(model, (first, second), proof)
        _unchanged(model, expected_revision)
        normal, points = _common_plane(detached, (first, second), proof)
        all_straight = all(isinstance(e.curve, Straight) for e in detached.edges.values())
        shared_ids = _complement(detached, (first, second), normal, points, proof)
        complementary_tolerance = proof.tolerance
        owner = None
        if all_straight:
            proof.cancel()
            proof.counts["owner_queries"] += 1
            owner = query_intersection(detached, *(detached.handle("face", p.id) for p in (first, second)))
            if owner.kind in (IntersectionKind.UNCLASSIFIED, IntersectionKind.UNSUPPORTED,
                              IntersectionKind.CAPABILITY_MISSING):
                raise _Unresolved("owner_" + owner.kind.value + ":" + ";".join(owner.diagnostics))
            if (not owner.classified or owner.certificate is None
                    or not owner.certificate.complete
                    or any(c.certificate is None or not c.certificate.complete for c in owner.components)):
                raise _Unresolved("owner_incomplete_evidence")
            proof.tolerance = owner.tolerance_used or model.tolerance.length
            residual = owner.max_residual
        if shared_ids is not None:
            if owner is not None and owner.dimension is IntersectionDimension.REGION:
                raise _Unresolved("owner_evidence_conflict")
            relation = TrimInteriorRelation.DISJOINT_INTERIORS
            boundary = TrimBoundaryContact.CURVE
            complementary = True
            shared = tuple(EntityHandle(model.model_id, "edge", key) for key in shared_ids)
            algorithm = "exact_topology_star_ring_v1"
            proof.tolerance = complementary_tolerance
            residual = 0.0
        elif owner is not None:
            algorithm = "owner_straight_planar_relation_v1"
            if owner.kind is IntersectionKind.DISJOINT:
                relation = TrimInteriorRelation.DISJOINT_INTERIORS
                boundary = TrimBoundaryContact.NONE
            elif owner.dimension is IntersectionDimension.REGION:
                relation = TrimInteriorRelation.OVERLAPPING_INTERIORS
            else:
                raise _Unresolved("owner_lower_dimension_does_not_prove_zero_area")
            diagnostics = owner.diagnostics
        else:
            raise _Unresolved("curved_trim_family_unresolved")
    except _Unresolved as unresolved:
        diagnostics = (str(unresolved),)
    proof.cancel()
    _unchanged(model, expected_revision)
    data = dict(
        model_id=model.model_id, revision=int(expected_revision), first_parent=first,
        second_parent=second, relation=relation, complementary=complementary,
        boundary=boundary, shared_boundary=shared, algorithm=algorithm,
        tolerance=float(proof.tolerance or model.tolerance.length), max_residual=float(residual),
        diagnostics=tuple(diagnostics), work_counts=proof.counts,
    )
    return TrimDomainResult(**data, digest=_digest(data))


def validate_trim_domain_binding(
    model: GeometryModel, result: TrimDomainResult,
    first: EntityHandle, second: EntityHandle, *, expected_revision: int,
    cancellation_check: Callable[[str], None] | None = None,
) -> None:
    """Validate freshness AND requalify evidence; a self-rehashed forgery is not proof."""
    _request(model, first, second, expected_revision, cancellation_check)
    if not isinstance(result, TrimDomainResult):
        raise _error(TrimDomainErrorCode.INVALID_RESULT, "expected TrimDomainResult")
    if result.model_id != model.model_id:
        raise _error(TrimDomainErrorCode.WRONG_MODEL, "result belongs to another model")
    if result.revision != model.revision:
        raise _error(TrimDomainErrorCode.STALE_RESULT, "result is from another revision")
    if result.first_parent != first or result.second_parent != second:
        raise _error(TrimDomainErrorCode.INVALID_RESULT, "ordered result parents differ")
    data = _result_data(result)
    # Construction revalidates every field, including evidence changed via bypasses.
    TrimDomainResult(**data, digest=result.digest)
    fresh = query_trim_domain_relation(
        model, first, second, expected_revision=expected_revision,
        cancellation_check=cancellation_check,
    )
    if _canonical(data) != _canonical(_result_data(fresh)) or result.digest != fresh.digest:
        raise _error(TrimDomainErrorCode.INVALID_RESULT, "evidence differs from fresh qualification")
