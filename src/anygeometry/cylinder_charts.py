"""Bounded, revision-bound occurrence charts for existing Cylinder sectors.

This module never authors geometry or merges physical sector identities. The
proof arithmetic is local and outward rounded; sampled polygons are not evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextvars import ContextVar
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

from .curves import Arc, ArcFrame, Straight, _COLLINEAR_RTOL, arc_frame, sample_arc, sample_straight
from .errors import GeometryError
from .identity import EntityHandle
from .model import GeometryModel
from .surfaces import Cylinder


class CylinderAtlasStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    UNRESOLVED = "UNRESOLVED"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"


class CylinderAtlasErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    WRONG_MODEL = "WRONG_MODEL"
    INACTIVE_ENTITY = "INACTIVE_ENTITY"
    STALE_REVISION = "STALE_REVISION"
    STALE_RESULT = "STALE_RESULT"
    INVALID_RESULT = "INVALID_RESULT"
    BUSY_MODEL = "BUSY_MODEL"
    UNQUALIFIED_RESULT = "UNQUALIFIED_RESULT"


class CylinderAtlasError(GeometryError):
    def __init__(self, code: CylinderAtlasErrorCode, diagnostics: tuple[str, ...]):
        object.__setattr__(self, "_code", CylinderAtlasErrorCode(code))
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
            raise AttributeError("atlas error evidence is immutable")
        super().__setattr__(name, value)


def _error(code, text):
    return CylinderAtlasError(code, (text,))


def _invalid(text="malformed atlas record"):
    return _error(CylinderAtlasErrorCode.INVALID_RESULT, text)


_binding_budget = ContextVar("cylinder_atlas_binding_budget", default=None)


class _BindingBudget:
    """Aggregate validation/encoding cap, independent of geometric work counts."""

    def __init__(self, policy, callback):
        self.limit = policy.max_interval_operations
        self.nodes = self.bytes = 0
        self.callback, self.callback_error = callback, None

    def step(self, byte_count=0):
        if self.nodes >= self.limit or byte_count > self.limit*16-self.bytes:
            raise _invalid("binding evidence aggregate budget exhausted")
        self.nodes += 1
        self.bytes += byte_count
        if self.callback is not None and self.nodes % 16 == 0:
            try:
                self.callback("cylinder atlas bounded evidence")
            except BaseException as error:
                self.callback_error = error
                raise


def _binding_step(value=None):
    budget = _binding_budget.get()
    if budget is not None:
        budget.step(len(value)*4 if isinstance(value, str) else 0)


def _typed(value, annotation):
    """Copy only the finite, immutable contract vocabulary; no mutable payloads."""
    _binding_step(value)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is types.UnionType:
        for alternative in args:
            try:
                return _typed(value, alternative)
            except CylinderAtlasError as error:
                budget = _binding_budget.get()
                if budget is not None and error is budget.callback_error:
                    raise
                pass
        raise _invalid()
    if origin is tuple:
        if type(value) is not tuple or len(value) > 16384:
            raise _invalid()
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_typed(item, args[0]) for item in value)
        if len(value) != len(args):
            raise _invalid()
        return tuple(_typed(item, kind) for item, kind in zip(value, args))
    if annotation is float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise _invalid()
        number = float(value)
        if not math.isfinite(number):
            raise _invalid()
        return 0.0 if number == 0 else number
    if annotation is int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise _invalid()
        if int(value).bit_length() > 8192:
            raise _invalid("oversized integer evidence")
        return int(value)
    if annotation is str:
        if type(value) is not str or len(value) > 2048:
            raise _invalid()
        return value
    if annotation is type(None):
        if value is not None:
            raise _invalid()
        return None
    if not isinstance(value, annotation):
        raise _invalid()
    if isinstance(value, _Record):
        return replace(value)  # Revalidate nested bypass-tampered records.
    return value


class _Record:
    __slots__ = ()

    def __post_init__(self):
        annotations = get_type_hints(type(self))
        for field in fields(self):
            object.__setattr__(self, field.name, _typed(getattr(self, field.name), annotations[field.name]))
        self._validate()

    def _validate(self):
        pass


Interval = tuple[float, float]
UVBox = tuple[Interval, Interval]
Vector = tuple[float, float, float]


def _intervals(*items):
    if any(lo > hi for lo, hi in items):
        raise _invalid("reversed enclosure")


def _handles(kind, *values):
    if any(handle.kind != kind for handle in values):
        raise _invalid("wrong handle kind")


@dataclass(frozen=True, slots=True)
class CylinderAtlasPolicy(_Record):
    max_face_uses: int = 256
    max_occurrences: int = 4096
    max_vertices: int = 8192
    max_interval_operations: int = 200000
    max_pair_tests: int = 65536
    max_evaluations: int = 4096

    def _validate(self):
        caps = (256, 4096, 8192, 200000, 65536, 4096)
        if any(not 1 <= getattr(self, field.name) <= cap for field, cap in zip(fields(self), caps)):
            raise _invalid("policy exceeds supported work limits")


_COUNTS = (
    "faces", "coedges", "edges", "vertices", "external_incidences", "frame_tests",
    "interval_operations", "series_terms", "pair_tests", "winding_steps",
    "evaluations", "cancellation_checks",
)


@dataclass(frozen=True, slots=True)
class CylinderAtlasCertificate(_Record):
    algorithm: str
    complete: bool
    tolerance_length: float
    tolerance_surface: float
    tolerance_angular: float
    max_residual: float
    max_enclosure_width: float
    work_counts: tuple[tuple[str, int], ...]

    def _validate(self):
        if self.algorithm != "cylinder-sector-orthogonal-v1":
            raise _invalid()
        if min(self.tolerance_length, self.tolerance_surface, self.tolerance_angular) <= 0:
            raise _invalid()
        if min(self.max_residual, self.max_enclosure_width) < 0:
            raise _invalid()
        if tuple(name for name, _ in self.work_counts) != _COUNTS or any(n < 0 for _, n in self.work_counts):
            raise _invalid()
        if self.complete and self.max_residual > self.tolerance_surface:
            raise _invalid("residual exceeds certificate tolerance")


@dataclass(frozen=True, slots=True)
class CylinderSectorChart(_Record):
    face: EntityHandle
    face_use: EntityHandle
    sheet: EntityHandle
    face_use_orientation: int
    loops: tuple[tuple[str, ...], ...]
    axis_sign: int
    reference_angle_offset: Interval
    reference_axial_offset: Interval
    start: float
    sweep: float
    height: float
    period_shift: int

    def _validate(self):
        _handles("face", self.face)
        _handles("face_use", self.face_use)
        _handles("sheet", self.sheet)
        _intervals(self.reference_angle_offset, self.reference_axial_offset)
        if self.face_use_orientation not in (-1, 1) or self.axis_sign not in (-1, 1):
            raise _invalid()
        if not self.sweep or not self.height or not self.loops:
            raise _invalid()


@dataclass(frozen=True, slots=True)
class CylinderOccurrence(_Record):
    id: str
    face_use: EntityHandle
    coedge: EntityHandle
    edge: EntityHandle
    start_vertex: EntityHandle
    end_vertex: EntityHandle
    loop_index: int
    loop_position: int
    traversal: int
    source_range: tuple[float, float]
    role: str
    carrier: str
    local_uv_endpoints: tuple[UVBox, UVBox]
    reference_endpoints: tuple[UVBox, UVBox]
    carrier_residual_bound: float
    parameter_enclosure: float

    def _validate(self):
        for kind, handle in (("face_use", self.face_use), ("coedge", self.coedge), ("edge", self.edge)):
            _handles(kind, handle)
        _handles("vertex", self.start_vertex, self.end_vertex)
        if self.traversal not in (-1, 1) or self.source_range != ((0., 1.) if self.traversal == 1 else (1., 0.)):
            raise _invalid()
        if self.role not in ("OUTER", "HOLE") or self.carrier not in ("AXIAL", "CIRCULAR"):
            raise _invalid()
        if min(self.loop_index, self.loop_position, self.carrier_residual_bound, self.parameter_enclosure) < 0:
            raise _invalid()
        for point in (*self.local_uv_endpoints, *self.reference_endpoints):
            _intervals(*point)


@dataclass(frozen=True, slots=True)
class CylinderInterface(_Record):
    id: str
    edge: EntityHandle
    occurrences: tuple[str, str]
    traversal_same: bool
    lift_delta: int
    is_reference_seam: bool
    external_coedges: tuple[EntityHandle, ...]

    def _validate(self):
        _handles("edge", self.edge)
        _handles("coedge", *self.external_coedges)
        if self.occurrences[0] == self.occurrences[1] or abs(self.lift_delta) > 1:
            raise _invalid()
        if self.is_reference_seam != (self.lift_delta != 0):
            raise _invalid()


@dataclass(frozen=True, slots=True)
class CylinderBoundaryCycle(_Record):
    occurrences: tuple[tuple[str, int], ...]
    role: str
    winding: int
    source_vertices: tuple[EntityHandle, ...]

    def _validate(self):
        _handles("vertex", *self.source_vertices)
        if self.role not in ("LOWER", "UPPER", "HOLE") or self.winding not in (-1, 0, 1):
            raise _invalid()
        if (self.role == "HOLE") != (self.winding == 0):
            raise _invalid()
        if not self.occurrences or len(self.occurrences) != len(self.source_vertices):
            raise _invalid()
        if any(direction not in (-1, 1) for _, direction in self.occurrences):
            raise _invalid()


@dataclass(frozen=True, slots=True)
class CylinderAtlasResult(_Record):
    contract_version: int
    model_id: UUID
    revision: int
    requested_face_uses: tuple[EntityHandle, ...]
    reference_face_use: EntityHandle
    policy: CylinderAtlasPolicy
    status: CylinderAtlasStatus
    sectors: tuple[CylinderSectorChart, ...]
    occurrences: tuple[CylinderOccurrence, ...]
    interfaces: tuple[CylinderInterface, ...]
    boundary_cycles: tuple[CylinderBoundaryCycle, ...]
    certificate: CylinderAtlasCertificate
    diagnostics: tuple[str, ...]
    digest: str

    def _validate(self):
        _handles("face_use", self.reference_face_use, *self.requested_face_uses)
        if self.contract_version != 1 or self.revision < 0 or not self.model_id.int:
            raise _invalid()
        if self.reference_face_use not in self.requested_face_uses or len(set(self.requested_face_uses)) != len(self.requested_face_uses):
            raise _invalid()
        if any(h.model_id != self.model_id for h in self.requested_face_uses):
            raise _invalid()
        qualified = self.status is CylinderAtlasStatus.QUALIFIED
        if qualified != self.certificate.complete:
            raise _invalid()
        if qualified and (not self.sectors or not self.occurrences or not self.boundary_cycles):
            raise _invalid()
        if not qualified and (self.sectors or self.occurrences or self.interfaces or self.boundary_cycles or not self.diagnostics):
            raise _invalid()
        if len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            raise _invalid()


@dataclass(frozen=True, slots=True)
class CylinderOccurrenceRequest(_Record):
    occurrence_id: str
    parameter: float

    def _validate(self):
        if not self.occurrence_id or not 0 <= self.parameter <= 1:
            raise _invalid("invalid source parameter request")


@dataclass(frozen=True, slots=True)
class CylinderOccurrenceSample(_Record):
    occurrence_id: str
    parameter: float
    point: Vector
    local_uv: tuple[float, float]
    lifted_reference: tuple[float, float]
    local_uv_enclosure: UVBox
    reference_enclosure: UVBox
    edge: EntityHandle
    endpoint_vertex: EntityHandle | None
    equivalence_key: str
    residual_bound: float
    status: CylinderAtlasStatus

    def _validate(self):
        _handles("edge", self.edge)
        if self.endpoint_vertex is not None:
            _handles("vertex", self.endpoint_vertex)
        _intervals(*self.local_uv_enclosure, *self.reference_enclosure)
        if self.status is not CylinderAtlasStatus.QUALIFIED or not 0 <= self.parameter <= 1 or self.residual_bound < 0:
            raise _invalid()


@dataclass(frozen=True, slots=True)
class CylinderOccurrenceEvaluation(_Record):
    model_id: UUID
    revision: int
    atlas_digest: str
    samples: tuple[CylinderOccurrenceSample, ...]
    certificate: CylinderAtlasCertificate

    def _validate(self):
        if self.revision < 0 or not self.certificate.complete or len(self.atlas_digest) != 64:
            raise _invalid()


def _canonical(value):
    _binding_step(value)
    if isinstance(value, EntityHandle):
        return {"model_id": value.model_id.hex, "kind": value.kind, "id": value.id}
    if isinstance(value, UUID):
        return value.hex
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {f.name: _canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    return value


def _digest(data):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(_canonical(data)):
        _binding_step(chunk)
        digest.update(chunk.encode())
    return digest.hexdigest()


class _Refusal(Exception):
    def __init__(self, reason, *, missing=False):
        super().__init__(reason)
        self.missing = missing


_GRID = 1 << 80


@dataclass(frozen=True, slots=True)
class _I:
    lo: Fraction
    hi: Fraction

    @property
    def mid(self):
        return (self.lo + self.hi) / 2


def _q(value):
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise _Refusal("nonfinite_geometry")
        value = float(value)
    made = Fraction(value)
    if max(abs(made.numerator).bit_length(), made.denominator.bit_length()) > 8192:
        raise _Refusal("rational_bit_budget")
    return made


class _Proof:
    """Private fixed-precision outward arithmetic with a counted work budget."""

    def __init__(self, policy, callback):
        self.policy, self.callback = policy, callback
        self.counts = dict.fromkeys(_COUNTS, 0)
        self.callback_error = None
        self.pi = None

    def cancel(self, phase="cylinder atlas qualification"):
        self.counts["cancellation_checks"] += 1
        if self.callback is not None:
            try:
                self.callback(phase)
            except BaseException as error:
                self.callback_error = error
                raise

    def charge(self, key="interval_operations", limit=None, period=64):
        maximum = self.policy.max_interval_operations if limit is None else limit
        if self.counts[key] >= maximum:
            raise _Refusal("qualification_budget_exhausted:" + key)
        self.counts[key] += 1
        if self.counts[key] % period == 0:
            self.cancel()

    def i(self, value, hi=None):
        if isinstance(value, _I) and hi is None:
            return value
        a, b = _q(value), _q(value if hi is None else hi)
        if a > b:
            raise _Refusal("invalid_interval")
        return _I(a, b)

    def rounded(self, lo, hi):
        lo, hi = _q(lo), _q(hi)
        if lo > hi:
            raise _Refusal("invalid_interval")
        if any(abs(x.numerator).bit_length() + 80 > 8192 for x in (lo, hi)):
            raise _Refusal("rational_bit_budget")
        return _I(Fraction((lo * _GRID).__floor__(), _GRID), Fraction((hi * _GRID).__ceil__(), _GRID))

    def add(self, a, b):
        self.charge()
        a, b = self.i(a), self.i(b)
        for x, y in ((a.lo, b.lo), (a.hi, b.hi)):
            if max(abs(x.numerator).bit_length() + y.denominator.bit_length(),
                   abs(y.numerator).bit_length() + x.denominator.bit_length(),
                   x.denominator.bit_length() + y.denominator.bit_length()) + 1 > 8192:
                raise _Refusal("rational_bit_budget")
        return self.rounded(a.lo + b.lo, a.hi + b.hi)

    def neg(self, a):
        a = self.i(a)
        return _I(-a.hi, -a.lo)

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        self.charge()
        a, b = self.i(a), self.i(b)
        for x, y in ((a.lo, b.lo), (a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi)):
            if max(abs(x.numerator).bit_length(), x.denominator.bit_length()) + max(abs(y.numerator).bit_length(), y.denominator.bit_length()) + 1 > 8192:
                raise _Refusal("rational_bit_budget")
        values = (a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
        return self.rounded(min(values), max(values))

    def div(self, a, b):
        b = self.i(b)
        if b.lo <= 0 <= b.hi:
            raise _Refusal("uncertain_zero_divisor")
        return self.mul(a, self.rounded(1/b.hi, 1/b.lo))

    def square(self, a):
        a = self.i(a)
        result = self.mul(a, a)
        return _I(Fraction(0), result.hi) if a.lo <= 0 <= a.hi else result

    def sqrt(self, a):
        self.charge()
        a = self.i(a)
        if a.lo < 0:
            raise _Refusal("negative_sqrt_enclosure")
        def floor_root(x):
            if abs(x.numerator).bit_length() + 160 > 8192:
                raise _Refusal("rational_bit_budget")
            return math.isqrt((x.numerator * _GRID * _GRID) // x.denominator)
        low, high = floor_root(a.lo), floor_root(a.hi)
        if Fraction(high * high, _GRID * _GRID) < a.hi:
            high += 1
        return _I(Fraction(low, _GRID), Fraction(high, _GRID))

    def dot(self, a, b):
        result = self.i(0)
        for x, y in zip(a, b):
            result = self.add(result, self.mul(x, y))
        return result

    def norm(self, vector):
        result = self.i(0)
        for v in vector:
            result = self.add(result, self.square(v))
        return self.sqrt(result)

    def cross(self, a, b):
        return tuple(self.sub(self.mul(a[i], b[j]), self.mul(a[j], b[i])) for i, j in ((1, 2), (2, 0), (0, 1)))

    def vsub(self, a, b):
        return tuple(self.sub(x, y) for x, y in zip(a, b))

    def vadd(self, a, b):
        return tuple(self.add(x, y) for x, y in zip(a, b))

    def vmul(self, a, k):
        return tuple(self.mul(x, k) for x in a)

    def remainder_power(self, value, exponent, denominator):
        """Preflight exact Taylor remainder work before allocating its power."""
        value, divisor = _q(value), _q(denominator)
        if (max(abs(value.numerator).bit_length(), value.denominator.bit_length()) * exponent
                + max(abs(divisor.numerator).bit_length(), divisor.denominator.bit_length()) + 1 > 8192):
            raise _Refusal("rational_bit_budget")
        return value ** exponent / divisor

    def atan_series(self, x):
        x = self.i(x)
        if max(abs(x.lo), abs(x.hi)) > Fraction(1, 2):
            raise _Refusal("atan_reduction_unqualified")
        power, square, result = x, self.square(x), self.i(0)
        bound = max(abs(x.lo), abs(x.hi))
        for n in range(128):
            self.charge("series_terms", self.policy.max_interval_operations)
            term = self.div(power, 2*n+1)
            result = self.add(result, term if n % 2 == 0 else self.neg(term))
            power = self.mul(power, square)
            remainder = self.remainder_power(bound, 2*n+3, 2*n+3)
            if remainder <= Fraction(1, _GRID * 16):
                return self.add(result, self.i(-remainder, remainder))
        raise _Refusal("series_budget_exhausted")

    def pi_bound(self):
        if self.pi is None:
            self.pi = self.sub(self.mul(16, self.atan_series(Fraction(1, 5))), self.mul(4, self.atan_series(Fraction(1, 239))))
        return self.pi

    def atan(self, x):
        x = self.i(x)
        if x.hi < 0:
            return self.neg(self.atan(self.neg(x)))
        if x.lo < 0:
            low = self.atan(self.i(x.lo))
            high = self.atan(self.i(x.hi))
            return _I(low.lo, high.hi)
        if x.lo > 1:
            return self.sub(self.div(self.pi_bound(), 2), self.atan(self.div(1, x)))
        if x.hi > 1:
            low, high = self.atan(self.i(x.lo)), self.atan(self.i(x.hi))
            return _I(low.lo, high.hi)
        reduced = self.div(x, self.add(1, self.sqrt(self.add(1, self.square(x)))))
        return self.mul(2, self.atan_series(reduced))

    def atan2(self, y, x):
        y, x = self.i(y), self.i(x)
        if x.lo > 0:
            return self.atan(self.div(y, x))
        if x.hi < 0 and y.lo >= 0:
            return self.add(self.pi_bound(), self.atan(self.div(y, x)))
        if x.hi < 0 and y.hi <= 0:
            return self.sub(self.atan(self.div(y, x)), self.pi_bound())
        if y.lo > 0:
            return self.sub(self.div(self.pi_bound(), 2), self.atan(self.div(x, y)))
        if y.hi < 0:
            return self.sub(self.neg(self.div(self.pi_bound(), 2)), self.atan(self.div(x, y)))
        raise _Refusal("angular_branch_unqualified")

    def sincos(self, x):
        x = self.i(x)
        pi = self.pi_bound()
        if max(abs(x.lo), abs(x.hi)) > 8*pi.hi:
            raise _Refusal("angle_range_out_of_scope", missing=True)
        half = self.div(pi, 2)
        quotient = self.div(x, half)
        low = math.floor(quotient.lo + Fraction(1, 2))
        high = math.floor(quotient.hi + Fraction(1, 2))
        if high - low > 1:
            raise _Refusal("trig_branch_width_unqualified")
        first = self._sincos_quadrant(x, low, half, pi)
        if low == high:
            return first
        second = self._sincos_quadrant(x, high, half, pi)
        return tuple(_I(min(a.lo, b.lo), max(a.hi, b.hi)) for a, b in zip(first, second))

    def _sincos_quadrant(self, x, quadrant, half, pi):
        reduced = self.sub(x, self.mul(quadrant, half))
        # Taylor remainder works on the entire reduced interval even if an
        # uncertain quadrant boundary extends infinitesimally past pi/4.
        if max(abs(reduced.lo), abs(reduced.hi)) > pi.hi / 2:
            raise _Refusal("trig_reduction_unqualified")
        power, factorial, sine, cosine = self.i(1), 1, self.i(0), self.i(0)
        magnitude = max(abs(reduced.lo), abs(reduced.hi))
        for n in range(128):
            self.charge("series_terms", self.policy.max_interval_operations)
            term = self.div(power, factorial)
            if n % 2:
                sine = self.add(sine, term if n % 4 == 1 else self.neg(term))
            else:
                cosine = self.add(cosine, term if n % 4 == 0 else self.neg(term))
            power = self.mul(power, reduced)
            factorial *= n + 1
            if n > 3:
                remainder = self.remainder_power(magnitude, n+1, factorial)
                if remainder <= Fraction(1, _GRID * 16):
                    padding = self.i(-remainder, remainder)
                    sine, cosine = self.add(sine, padding), self.add(cosine, padding)
                    return ((sine, cosine), (cosine, self.neg(sine)),
                            (self.neg(sine), self.neg(cosine)), (self.neg(cosine), sine))[quadrant % 4]
        raise _Refusal("series_budget_exhausted")

    def out(self, value):
        value = self.i(value)
        lo, hi = float(value.lo), float(value.hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise _Refusal("nonfinite_output")
        if Fraction(lo) > value.lo:
            lo = math.nextafter(lo, -math.inf)
        if Fraction(hi) < value.hi:
            hi = math.nextafter(hi, math.inf)
        return (0.0 if lo == 0 else lo, 0.0 if hi == 0 else hi)


def _unchanged(model, revision):
    if model._transaction_journal is not None:
        raise _error(CylinderAtlasErrorCode.BUSY_MODEL, "atlas requires committed state")
    if model.revision != revision:
        raise _error(CylinderAtlasErrorCode.STALE_REVISION, "model changed during qualification")


def _bounded(values, maximum, message):
    if isinstance(values, (str, bytes, dict)):
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, message)
    try:
        iterator = iter(values)
    except TypeError as error:
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, message) from error
    result = []
    for item in iterator:
        if len(result) >= maximum:
            raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, message)
        result.append(item)
    return tuple(result)


def _request(model, face_uses, reference, revision, callback, policy):
    if not isinstance(model, GeometryModel):
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "expected GeometryModel")
    if callback is not None and not callable(callback):
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "cancellation_check must be callable")
    if isinstance(revision, (bool, np.bool_)) or not isinstance(revision, (int, np.integer)) or revision < 0:
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "integer revision required")
    if not isinstance(policy, CylinderAtlasPolicy):
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "expected CylinderAtlasPolicy")
    replace(policy)
    requested = _bounded(face_uses, policy.max_face_uses, "bounded FaceUse handles required")
    if (not requested or any(not isinstance(h, EntityHandle) or h.kind != "face_use" for h in (*requested, reference))
            or reference not in requested or len(set(requested)) != len(requested)):
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "distinct FaceUses and selected reference required")
    if any(h.model_id != model.model_id for h in requested):
        raise _error(CylinderAtlasErrorCode.WRONG_MODEL, "FaceUse belongs to another model")
    _unchanged(model, revision)
    if any(h.id not in model.face_uses for h in requested):
        raise _error(CylinderAtlasErrorCode.INACTIVE_ENTITY, "FaceUse is not active")
    return requested


class _Context:
    """Only selected immutable owner records; no live evaluator/cache access."""

    def __init__(self, model, selected, proof):
        self.model_id = model.model_id
        self.uses, self.faces, self.coedges, self.edges, self.vertices = {}, {}, {}, {}, {}
        self.incidence = {}
        face_ids = set()
        copied = 0
        for handle in sorted(selected):
            proof.cancel("cylinder atlas source face")
            use = model.face_uses[handle.id]
            face = model.faces[use.face_id]
            if face.id in face_ids:
                raise _Refusal("repeated_physical_face", missing=True)
            face_ids.add(face.id)
            if not isinstance(face.surface, Cylinder) or (face.parameterization is not None and face.parameterization is not face.surface):
                raise _Refusal("support_or_parameterization_out_of_scope", missing=True)
            if use.sheet_id not in model.sheets or use.id not in model.sheets[use.sheet_id].face_use_ids:
                raise _Refusal("invalid_source_ownership")
            source_loops = (face.loop, *face.holes)
            if len(use.loops) != len(source_loops):
                raise _Refusal("invalid_source_loops")
            self.uses[use.id], self.faces[face.id] = use, face
            proof.counts["faces"] += 1
            visited = set()
            for coedge_ids, loop in zip(use.loops, source_loops):
                if len(loop) < 3 or len(loop) != len(coedge_ids):
                    raise _Refusal("invalid_source_loop")
                for coedge_id, oriented in zip(coedge_ids, loop):
                    if len(self.coedges) >= proof.policy.max_occurrences:
                        raise _Refusal("qualification_budget_exhausted:coedges")
                    coedge = model.coedges[coedge_id]
                    if coedge.face_use_id != use.id or coedge.edge_id != oriented.edge or int(coedge.orientation) != (1 if oriented.forward else -1):
                        raise _Refusal("invalid_source_occurrence")
                    edge = model.edges[oriented.edge]
                    if edge.id in visited or coedge_id in self.coedges:
                        raise _Refusal("repeated_source_occurrence")
                    visited.add(edge.id)
                    if not isinstance(edge.curve, (Arc, Straight)):
                        raise _Refusal("curve_family_out_of_scope", missing=True)
                    self.coedges[coedge_id], self.edges[edge.id] = coedge, edge
                    vertex_ids = (edge.start, edge.end, edge.curve.via_vertex) if isinstance(edge.curve, Arc) else (edge.start, edge.end)
                    for vertex_id in vertex_ids:
                        if vertex_id not in self.vertices and len(self.vertices) >= proof.policy.max_vertices:
                            raise _Refusal("qualification_budget_exhausted:vertices")
                        self.vertices[vertex_id] = model.vertices[vertex_id]
                        copied += 1
                        if copied % 16 == 0:
                            proof.cancel("cylinder atlas source copy")
                for a, b in zip(loop, loop[1:] + loop[:1]):
                    first, second = self.edges[a.edge], self.edges[b.edge]
                    if (first.end if a.forward else first.start) != (second.start if b.forward else second.end):
                        raise _Refusal("invalid_source_connectivity")
        incidence_count = 0
        for edge_id in sorted(self.edges):
            count = len(model._edge_coedges.get(edge_id, ()))
            if count > proof.policy.max_occurrences-incidence_count:
                raise _Refusal("qualification_budget_exhausted:external_incidences")
            incidence_count += count  # Reserve before public sorting/copying.
            proof.cancel("cylinder atlas incidence copy")
            copied_ids = []
            for key in model.coedges_using_edge(edge_id):
                copied += 1
                if copied % 16 == 0:
                    proof.cancel("cylinder atlas incidence copy")
                copied_ids.append(key)
                proof.counts["external_incidences"] += int(key not in self.coedges)
            self.incidence[edge_id] = tuple(copied_ids)
        proof.counts.update(coedges=len(self.coedges), edges=len(self.edges), vertices=len(self.vertices))

    def handle(self, kind, identifier):
        return EntityHandle(self.model_id, kind, identifier)


def _positive_sign(value, reason):
    if value.lo > 0:
        return 1
    if value.hi < 0:
        return -1
    raise _Refusal(reason)


def _upper_abs(value):
    return max(abs(value.lo), abs(value.hi))


def _hull(values):
    return _I(min(x.lo for x in values), max(x.hi for x in values))


def _interval_overlap(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


class _GeometryProof:
    def __init__(self, context, reference, tolerance, proof):
        self.context, self.p, self.tolerance = context, proof, tolerance
        self.reference = context.faces[context.uses[reference.id].face_id].surface
        self.anchor = tuple(_q(x) for x in self.reference.origin)
        points = [tuple(_q(x) for x in v.position) for v in context.vertices.values()]
        for face in context.faces.values():
            s = face.surface
            origin = tuple(_q(x) for x in s.origin)
            points.append(origin)
            for vector, length in ((s.axis, s.height), (s.radial_direction, s.radius), (s.circumferential_direction, s.radius)):
                points.append(tuple(a + _q(length)*_q(b) for a, b in zip(origin, vector)))
        diagonal2 = sum((max(v[k] for v in points)-min(v[k] for v in points))**2 for k in range(3))
        extent = proof.sqrt(diagonal2)
        self.extent = proof.out(extent)[1]
        if not math.isfinite(self.extent) or self.extent <= 0:
            raise _Refusal("degenerate_participating_extent")
        largest = max(self.extent, *(s.surface.radius for s in context.faces.values()), *(abs(s.surface.height) for s in context.faces.values()))
        exponent = math.frexp(largest)[1]
        if exponent > 1023:
            raise _Refusal("normalization_scale_unrepresentable")
        self.scale = _q(math.ldexp(1., exponent))
        self.tau = _q(tolerance.effective_length(self.extent)) / self.scale
        self.residual_tolerance = _q(tolerance.effective_surface_residual(self.extent)) / self.scale
        self.angular_tolerance = _q(tolerance.angular)
        self.parameter_tolerance = _q(tolerance.parameter)
        self.reference_basis = self.basis(self.reference)
        self.maps = {}
        self.vertex_coordinates = {}
        self.arc_frames = {}
        self.arc_frame_errors = {}
        self.max_residual = Fraction(0)
        self.max_width = Fraction(0)

    def vector(self, value):
        return tuple(self.p.i(_q(x)) for x in value)

    def point(self, value):
        return tuple(self.p.i((_q(x)-a)/self.scale) for x, a in zip(value, self.anchor))

    def basis(self, surface):
        p = self.p
        radial, axial = self.vector(surface.radial_direction), self.vector(surface.axis)
        circum = self.vector(surface.circumferential_direction)
        determinant = p.dot(radial, p.cross(circum, axial))
        gramdet = p.square(determinant)
        if gramdet.lo <= Fraction(1, 2**40):
            raise _Refusal("ill_conditioned_surface_frame")
        return radial, circum, axial, determinant

    def coordinates(self, vector, basis):
        r, c, a, determinant = basis
        p = self.p
        return tuple(p.div(p.dot(vector, cross), determinant) for cross in (p.cross(c, a), p.cross(a, r), p.cross(r, c)))

    def periodic_near(self, angle, target):
        p = self.p
        period = p.mul(2, p.pi_bound())
        turns = p.div(p.sub(target, angle), period)
        low = math.floor(turns.lo + Fraction(1, 2))
        high = math.floor(turns.hi + Fraction(1, 2))
        if low != high:
            raise _Refusal("angular_lift_ambiguous")
        return p.add(angle, p.mul(low, period)), low

    def lifted_vertex_angle(self, x, y, target):
        """Lift a connected Cartesian interval across the principal cut.

        Scalar atan2 deliberately cannot represent both principal images at
        once. Only the strict negative-radial-ray case is split here; boxes
        containing the origin retain the arithmetic owner's refusal.
        """
        p = self.p
        if not (x.hi < 0 and y.lo < 0 < y.hi):
            return self.periodic_near(p.atan2(y, x), target)[0]
        negative = p.atan2(p.i(y.lo, 0), x)
        positive = p.atan2(p.i(0, y.hi), x)
        first = self.periodic_near(negative, target)[0]
        second = self.periodic_near(positive, target)[0]
        if first.lo > second.hi or second.lo > first.hi:
            raise _Refusal("angular_lift_disconnected")
        joined = _hull((first, second))
        if joined.hi - joined.lo >= p.pi_bound().lo:
            raise _Refusal("angular_lift_ambiguous")
        return joined

    def frame_map(self, surface):
        p = self.p
        p.counts["frame_tests"] += 1
        if abs(_q(surface.start_angle)) > 8*p.pi_bound().hi or abs(_q(surface.sweep_angle)) > p.pi_bound().hi/2+self.angular_tolerance:
            raise _Refusal("sector_angle_out_of_scope", missing=True)
        if abs(_q(surface.sweep_angle)) <= self.angular_tolerance or abs(_q(surface.height))/self.scale <= self.tau or _q(surface.radius)/self.scale <= self.tau:
            raise _Refusal("degenerate_sector")
        basis = self.basis(surface)
        r, c, a, _ = basis
        rr, rc, ra, _ = self.reference_basis
        sign = _positive_sign(p.dot(a, ra), "axis_orientation_unqualified")
        rx, ry, _ = self.coordinates(r, self.reference_basis)
        phase = p.atan2(ry, rx)
        origin = self.point(surface.origin)
        _, _, z = self.coordinates(origin, self.reference_basis)
        sine, cosine = p.sincos(phase)
        radius = _q(self.reference.radius)/self.scale
        expected_cos = p.vmul(p.vadd(p.vmul(rr, cosine), p.vmul(rc, sine)), radius)
        expected_sin = p.vmul(p.vadd(p.vmul(rr, p.neg(sine)), p.vmul(rc, cosine)), sign*radius)
        local_radius = _q(surface.radius)/self.scale
        discrepancies = (
            p.vsub(origin, p.vmul(ra, z)),
            p.vsub(p.vmul(r, local_radius), expected_cos),
            p.vsub(p.vmul(c, local_radius), expected_sin),
            p.vsub(p.vmul(a, _q(surface.height)/self.scale), p.vmul(ra, sign*_q(surface.height)/self.scale)),
        )
        bound = p.i(0)
        for vector in discrepancies:
            bound = p.add(bound, p.norm(vector))
        self.accept_residual(bound)
        return {"basis": basis, "origin": origin, "phase": phase, "z": z, "sign": sign,
                "surface": surface, "frame_error": bound}

    def coordinate_errors(self, residual, basis, radius):
        """World-ball error through the certified inverse frame, then atan.

        The actual point may be off the nominal radius: asin(delta/R), not
        a same-circle chord approximation, bounds its angular displacement.
        """
        p = self.p
        r, c, a, determinant = basis
        rows = tuple(p.vmul(cross, p.div(1, determinant))
                     for cross in (p.cross(c, a), p.cross(a, r), p.cross(r, c)))
        radial_metric = p.norm((*rows[0], *rows[1]))
        axial_metric = p.norm(rows[2])
        ratio = p.div(p.mul(radial_metric, residual), radius)
        if ratio.lo < 0 or ratio.hi >= 1:
            raise _Refusal("inverse_metric_angular_branch_unqualified")
        angular = p.atan(p.div(ratio, p.sqrt(p.sub(1, p.square(ratio)))))
        return angular, p.mul(axial_metric, residual)

    def accept_residual(self, bound):
        if bound.hi > self.residual_tolerance:
            raise _Refusal("surface_residual_unqualified")
        self.max_residual = max(self.max_residual, bound.hi*self.scale)

    def local_vertex(self, vertex_id, mapping):
        p = self.p
        surface = mapping["surface"]
        point = self.point(self.context.vertices[vertex_id].position)
        x, y, z = self.coordinates(p.vsub(point, mapping["origin"]), mapping["basis"])
        angle = self.lifted_vertex_angle(x, y, _q(surface.start_angle)+_q(surface.sweep_angle)/2)
        u = p.div(p.sub(angle, surface.start_angle), surface.sweep_angle)
        v = p.div(z, _q(surface.height)/self.scale)
        if min(u.lo, v.lo) < -self.parameter_tolerance or max(u.hi, v.hi) > 1+self.parameter_tolerance:
            raise _Refusal("trim_outside_sector")
        return angle, z, (u, v)

    def reference_vertex(self, vertex_id, target):
        p = self.p
        if vertex_id not in self.vertex_coordinates:
            point = self.point(self.context.vertices[vertex_id].position)
            x, y, z = self.coordinates(point, self.reference_basis)
            self.vertex_coordinates[vertex_id] = (x, y, z)
        x, y, z = self.vertex_coordinates[vertex_id]
        angle = self.lifted_vertex_angle(x, y, target)
        return angle, z

    def frame(self, edge):
        if edge.id not in self.arc_frames:
            p = self.p
            start, via, end = (self.context.vertices[k].position for k in (edge.start, edge.curve.via_vertex, edge.end))
            ab = tuple((_q(b)-_q(a))/self.scale for a, b in zip(start, via))
            ac = tuple((_q(b)-_q(a))/self.scale for a, b in zip(start, end))
            cross = p.cross(ab, ac)
            numerator = p.dot(cross, cross)
            denominator = p.mul(p.dot(ab, ab), p.dot(ac, ac))
            if denominator.lo <= 0 or p.div(numerator, denominator).lo <= max(Fraction(1, 2**40), _q(_COLLINEAR_RTOL)):
                raise _Refusal("ill_conditioned_arc")
            frame = arc_frame(start, via, end)  # Operational exceptions propagate unchanged.
            if (not isinstance(frame, ArcFrame)
                    or any(np.shape(getattr(frame, name)) != (3,) for name in ("center", "e1", "e2", "normal"))
                    or not all(np.all(np.isfinite(getattr(frame, name))) for name in ("center", "e1", "e2", "normal"))
                    or not math.isfinite(frame.radius) or not math.isfinite(frame.sweep) or frame.radius <= 0):
                raise _Refusal("malformed_owner_arc_frame")
            # Enclose the unique circle implied by the actual three source
            # vertices, then bound the public floating frame against it over
            # the entire native parameter interval. This is proof, not a
            # replacement evaluator or a three-sample approximation.
            a = self.point(start)
            ab2, ac2 = p.dot(ab, ab), p.dot(ac, ac)
            normal2 = p.dot(cross, cross)
            offset = p.vmul(p.vadd(p.vmul(p.cross(ac, cross), ab2),
                                   p.vmul(p.cross(cross, ab), ac2)), p.div(1, p.mul(2, normal2)))
            exact_center = p.vadd(a, offset)
            radial = p.vsub(a, exact_center)
            exact_radius = p.norm(radial)
            e1 = p.vmul(radial, p.div(1, exact_radius))
            unit_normal = p.vmul(cross, p.div(1, p.sqrt(normal2)))
            e2 = p.cross(unit_normal, e1)
            def wrapped(vertex):
                vector = p.vsub(self.point(vertex), exact_center)
                angle = p.atan2(p.dot(vector, e2), p.dot(vector, e1))
                if angle.hi < 0:
                    return p.add(angle, p.mul(2, p.pi_bound()))
                if angle.lo <= 0:
                    raise _Refusal("source_arc_angle_unqualified")
                return angle
            via_angle, end_angle = wrapped(via), wrapped(end)
            if via_angle.hi < end_angle.lo:
                exact_sweep = end_angle
            elif via_angle.lo > end_angle.hi:
                exact_sweep = p.sub(end_angle, p.mul(2, p.pi_bound()))
            else:
                raise _Refusal("source_arc_sweep_unqualified")
            frame_error = p.norm(p.vsub(self.point(frame.center), exact_center))
            for supplied, certified in ((frame.e1, e1), (frame.e2, e2)):
                frame_error = p.add(frame_error, p.norm(p.vsub(
                    p.vmul(self.vector(supplied), _q(frame.radius)/self.scale),
                    p.vmul(certified, exact_radius))))
            frame_error = p.add(frame_error, p.mul(exact_radius, _upper_abs(p.sub(frame.sweep, exact_sweep))))
            self.accept_residual(frame_error)
            self.arc_frames[edge.id] = frame
            self.arc_frame_errors[edge.id] = frame_error
        return self.arc_frames[edge.id]

    def curve(self, edge, mapping):
        p = self.p
        surface = mapping["surface"]
        start_angle, start_z, start_uv = self.local_vertex(edge.start, mapping)
        end_angle, end_z, end_uv = self.local_vertex(edge.end, mapping)
        r, c, axis, _ = mapping["basis"]
        radius = _q(surface.radius)/self.scale
        if isinstance(edge.curve, Straight):
            constant_angle = start_angle
            sine, cosine = p.sincos(constant_angle)
            radial = p.vmul(p.vadd(p.vmul(r, cosine), p.vmul(c, sine)), radius)
            errors = []
            for vertex_id, z in ((edge.start, start_z), (edge.end, end_z)):
                expected = p.vadd(mapping["origin"], p.vadd(radial, p.vmul(axis, z)))
                errors.append(p.norm(p.vsub(self.point(self.context.vertices[vertex_id].position), expected)))
            residual = _hull(errors)
            self.accept_residual(residual)
            dz = p.sub(end_z, start_z)
            if min(abs(dz.lo), abs(dz.hi)) <= self.tau or dz.lo <= 0 <= dz.hi:
                raise _Refusal("non_axial_or_degenerate_straight", missing=True)
            slope = p.i(0)
            carrier, constant = "AXIAL", constant_angle
            carrier_angle0, carrier_z0 = constant_angle, start_z
        else:
            frame = self.frame(edge)
            center = self.point(frame.center)
            _, _, center_z = self.coordinates(p.vsub(center, mapping["origin"]), mapping["basis"])
            e1, e2 = self.vector(frame.e1), self.vector(frame.e2)
            e1x, e1y, _ = self.coordinates(e1, mapping["basis"])
            phase = p.atan2(e1y, e1x)
            phase, _ = self.periodic_near(phase, start_angle)
            sign = _positive_sign(p.dot(self.vector(frame.normal), axis), "arc_orientation_unqualified")
            slope = p.i(sign*_q(frame.sweep))
            sine, cosine = p.sincos(phase)
            expected_cos = p.vmul(p.vadd(p.vmul(r, cosine), p.vmul(c, sine)), radius)
            expected_sin = p.vmul(p.vadd(p.vmul(r, p.neg(sine)), p.vmul(c, cosine)), sign*radius)
            residual = p.i(0)
            for delta in (
                p.vsub(center, p.vadd(mapping["origin"], p.vmul(axis, center_z))),
                p.vsub(p.vmul(e1, _q(frame.radius)/self.scale), expected_cos),
                p.vsub(p.vmul(e2, _q(frame.radius)/self.scale), expected_sin),
            ):
                residual = p.add(residual, p.norm(delta))
            residual = p.add(residual, self.arc_frame_errors[edge.id])
            self.accept_residual(residual)
            # Projected angular momentum must keep one sign for the WHOLE Arc.
            cc = self.coordinates(p.vsub(center, mapping["origin"]), mapping["basis"])
            aa = self.coordinates(p.vmul(e1, _q(frame.radius)/self.scale), mapping["basis"])
            bb = self.coordinates(p.vmul(e2, _q(frame.radius)/self.scale), mapping["basis"])
            def det(a, b):
                return p.sub(p.mul(a[0], b[1]), p.mul(a[1], b[0]))
            core = p.mul(sign, det(aa, bb))
            variation = _upper_abs(det(cc, aa)) + _upper_abs(det(cc, bb))
            if core.lo <= variation or abs(slope.mid) <= self.angular_tolerance:
                raise _Refusal("arc_angular_monotonicity_unqualified")
            angle_error, _ = self.coordinate_errors(residual, mapping["basis"], radius)
            padding = angle_error.hi + self.angular_tolerance
            predicted_end = p.add(phase, slope)
            if _upper_abs(p.sub(start_angle, phase)) > padding or _upper_abs(p.sub(end_angle, predicted_end)) > padding:
                raise _Refusal("owner_arc_endpoint_branch_unqualified")
            via_angle, _, _ = self.local_vertex(edge.curve.via_vertex, mapping)
            sign_slope = _positive_sign(slope, "arc_sweep_unqualified")
            first = p.mul(sign_slope, p.sub(via_angle, phase))
            last = p.mul(sign_slope, p.sub(predicted_end, via_angle))
            if first.lo <= 0 or last.lo <= 0:
                raise _Refusal("owner_arc_via_branch_unqualified")
            carrier, constant = "CIRCULAR", center_z
            carrier_angle0, carrier_z0 = phase, center_z
        # All endpoints remain source-labelled; parameter uncertainty is explicit.
        angular, axial_error = self.coordinate_errors(residual, mapping["basis"], radius)
        parameter_error = max(angular.hi/abs(_q(surface.sweep_angle)), axial_error.hi/(abs(_q(surface.height))/self.scale))
        return {
            "edge": edge, "carrier": carrier, "constant": constant,
            "start_angle": start_angle, "start_z": start_z,
            "end_angle": end_angle, "end_z": end_z,
            "uv": (start_uv, end_uv), "slope": slope,
            "residual": residual, "parameter_error": parameter_error,
            "carrier_angle0": carrier_angle0, "carrier_z0": carrier_z0,
        }


def _pair_charge(proof):
    proof.charge("pair_tests", proof.policy.max_pair_tests, 32)


def _bounds(points):
    return tuple(_hull([point[axis] for point in points]) for axis in range(2))


def _separated(first, second, clearance):
    return any(a.hi+clearance < b.lo or b.hi+clearance < a.lo for a, b in zip(first, second))


def _area(proof, points):
    # Translation-local shoelace avoids making cancellation an origin test.
    anchor = points[0]
    local = [tuple(proof.sub(x, y) for x, y in zip(point, anchor)) for point in points]
    total = proof.i(0)
    for a, b in zip(local, local[1:]+local[:1]):
        total = proof.add(total, proof.sub(proof.mul(a[0], b[1]), proof.mul(a[1], b[0])))
    return proof.div(total, 2)


def _inside(proof, point, polygon):
    """Interval ray crossing for a point proven off the polygon boundary."""
    crossings = 0
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        _pair_charge(proof)
        if _interval_overlap(a[1], b[1]):
            # A horizontal carrier cannot cross this ray if the query is off
            # its level. Uncertain non-boundary event ordering is not guessed.
            if _interval_overlap(point[1], _hull((a[1], b[1]))):
                if _separated((point[0], point[1]), _bounds((a, b)), 0):
                    continue
                raise _Refusal("ray_event_order_unqualified")
            continue
        lower, upper = (a, b) if a[1].hi < b[1].lo else (b, a)
        if not lower[1].hi < upper[1].lo:
            raise _Refusal("ray_event_order_unqualified")
        if point[1].hi < lower[1].lo or point[1].lo > upper[1].hi:
            continue
        if not (lower[1].hi < point[1].lo and point[1].hi < upper[1].lo):
            raise _Refusal("ray_vertex_event_unqualified")
        level = _hull((a[0], b[0]))
        if point[0].hi < level.lo:
            crossings += 1
        elif point[0].lo <= level.hi:
            raise _Refusal("ray_boundary_event_unqualified")
    return bool(crossings % 2)


def _material_seed(proof, polygon, clearance):
    """A certified local cell midpoint, not a sampled point used as geometry."""
    xs, ys = set(), set()
    for point in polygon:
        _pair_charge(proof)
        xs.add(point[0].mid)
        ys.add(point[1].mid)
    xs, ys = sorted(xs), sorted(ys)
    for low_x, high_x in zip(xs, xs[1:]):
        for low_y, high_y in zip(ys, ys[1:]):
            _pair_charge(proof)
            if min(high_x-low_x, high_y-low_y) <= 4*clearance:
                continue
            point = (proof.i((low_x+high_x)/2), proof.i((low_y+high_y)/2))
            clear = True
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                _pair_charge(proof)
                if not _separated(point, _bounds((a, b)), clearance):
                    clear = False
                    break
            if clear:
                if _inside(proof, point, polygon):
                    return point
    raise _Refusal("material_cell_unqualified")


def _qualify(model, requested, reference, policy, proof):
    context = _Context(model, requested, proof)
    if len(requested) < 4:
        raise _Refusal("sector_ring_size_out_of_scope", missing=True)
    geometry = _GeometryProof(context, reference, model.tolerance, proof)
    proof.geometry = geometry
    p = proof
    period = p.mul(2, p.pi_bound())
    cut = _q(geometry.reference.start_angle)
    radius = _q(geometry.reference.radius)/geometry.scale
    sectors, occurrences, details, polygon_data = [], [], {}, {}
    by_edge = defaultdict(list)
    selected_orientation = None
    for use_id in sorted(context.uses):
        p.cancel("cylinder atlas face qualification")
        use = context.uses[use_id]
        face = context.faces[use.face_id]
        surface = face.surface
        mapping = geometry.frame_map(surface)
        middle = p.add(mapping["phase"], p.mul(mapping["sign"], _q(surface.start_angle)+_q(surface.sweep_angle)/2))
        _, shift = geometry.periodic_near(middle, p.add(cut, p.pi_bound()))
        mapping["shift"] = shift
        mapping["middle"] = p.add(middle, p.mul(shift, period))
        endpoints = tuple(p.add(p.add(mapping["phase"], p.mul(mapping["sign"], _q(surface.start_angle)+n*_q(surface.sweep_angle))), p.mul(shift, period)) for n in (0, 1))
        domain = _hull(endpoints)
        if domain.lo < cut-geometry.angular_tolerance or domain.hi > cut+period.hi+geometry.angular_tolerance:
            raise _Refusal("reference_cut_requires_source_partition", missing=True)
        geometry.maps[use_id] = mapping
        loop_ids, loop_records = [], []
        for loop_index, coedge_ids in enumerate(use.loops):
            ids, loop_vertices, points, loop_edges = [], [], [], []
            for loop_position, coedge_id in enumerate(coedge_ids):
                coedge = context.coedges[coedge_id]
                edge = context.edges[coedge.edge_id]
                curve = geometry.curve(edge, mapping)
                reference_points = tuple(geometry.reference_vertex(v, mapping["middle"]) for v in (edge.start, edge.end))
                physical_points = tuple((p.mul(radius, angle), z) for angle, z in reference_points)
                for point in physical_points:
                    width = p.norm(tuple(p.i(v.hi-v.lo) for v in point))
                    geometry.max_width = max(geometry.max_width, width.hi*geometry.scale, 2*curve["residual"].hi*geometry.scale)
                identifier = f"coedge/{coedge_id}/whole"
                occurrence = CylinderOccurrence(
                    id=identifier, face_use=context.handle("face_use", use_id),
                    coedge=context.handle("coedge", coedge_id), edge=context.handle("edge", edge.id),
                    start_vertex=context.handle("vertex", edge.start), end_vertex=context.handle("vertex", edge.end),
                    loop_index=loop_index, loop_position=loop_position, traversal=int(coedge.orientation),
                    source_range=(0., 1.) if int(coedge.orientation) == 1 else (1., 0.),
                    role="OUTER" if loop_index == 0 else "HOLE", carrier=curve["carrier"],
                    local_uv_endpoints=tuple(tuple(p.out(v) for v in point) for point in curve["uv"]),
                    reference_endpoints=tuple((p.out(a), p.out(p.mul(z, geometry.scale))) for a, z in reference_points),
                    carrier_residual_bound=p.out(p.mul(curve["residual"], geometry.scale))[1],
                    parameter_enclosure=p.out(p.i(curve["parameter_error"]))[1],
                )
                details[identifier] = {"occurrence": occurrence, "curve": curve, "points": physical_points,
                                       "reference": reference_points, "mapping": mapping, "use": use}
                occurrences.append(occurrence)
                by_edge[edge.id].append(identifier)
                ids.append(identifier)
                forward = int(coedge.orientation) == 1
                loop_vertices.append(edge.start if forward else edge.end)
                points.append(physical_points[0 if forward else 1])
                loop_edges.append((physical_points, curve["residual"].hi))
            if len(set(loop_vertices)) != len(loop_vertices):
                raise _Refusal("non_simple_source_loop")
            area = _area(p, points)
            area_sign = _positive_sign(area, "loop_area_unqualified")
            if min(abs(area.lo), abs(area.hi)) <= geometry.tau**2:
                raise _Refusal("degenerate_material_loop")
            effective_sign = area_sign*int(use.orientation)
            if loop_index == 0:
                if selected_orientation is None:
                    selected_orientation = effective_sign
                elif selected_orientation != effective_sign:
                    raise _Refusal("inconsistent_selected_material_orientation")
            elif effective_sign == selected_orientation:
                raise _Refusal("hole_orientation_out_of_scope", missing=True)
            for i, (first_points, first_error) in enumerate(loop_edges):
                for j in range(i+1, len(loop_edges)):
                    if j == i+1 or (i == 0 and j == len(loop_edges)-1):
                        continue
                    _pair_charge(p)
                    second_points, second_error = loop_edges[j]
                    if not _separated(_bounds(first_points), _bounds(second_points), geometry.tau+first_error+second_error):
                        raise _Refusal("loop_contact_or_clearance_unqualified")
            loop_ids.append(tuple(ids))
            loop_records.append({"ids": ids, "vertices": loop_vertices, "points": points, "edges": loop_edges})
        outer = loop_records[0]
        for hole_index, hole in enumerate(loop_records[1:]):
            for other in loop_records[:hole_index+1]:
                for first_points, first_error in hole["edges"]:
                    for second_points, second_error in other["edges"]:
                        _pair_charge(p)
                        if not _separated(_bounds(first_points), _bounds(second_points), geometry.tau+first_error+second_error):
                            raise _Refusal("hole_boundary_clearance_unqualified")
            seed = _material_seed(p, hole["points"], geometry.tau)
            if not _inside(p, seed, outer["points"]):
                raise _Refusal("hole_outside_material")
            if any(_inside(p, seed, other["points"]) for other in loop_records[1:hole_index+1]):
                raise _Refusal("nested_hole_out_of_scope", missing=True)
        polygon_data[use_id] = loop_records
        sectors.append(CylinderSectorChart(
            face=context.handle("face", face.id), face_use=context.handle("face_use", use_id),
            sheet=context.handle("sheet", use.sheet_id), face_use_orientation=int(use.orientation),
            loops=tuple(loop_ids), axis_sign=mapping["sign"],
            reference_angle_offset=p.out(mapping["phase"]),
            reference_axial_offset=p.out(p.mul(mapping["z"], geometry.scale)),
            start=surface.start_angle, sweep=surface.sweep_angle, height=surface.height, period_shift=shift,
        ))

    interfaces, paired, adjacency = [], set(), defaultdict(set)
    for edge_id, identifiers in sorted(by_edge.items()):
        if len(identifiers) == 1:
            continue
        if len(identifiers) != 2:
            raise _Refusal("ambiguous_selected_radial_incidence", missing=True)
        first, second = (details[key] for key in sorted(identifiers))
        a, b = first["occurrence"], second["occurrence"]
        if a.traversal*int(first["use"].orientation) == b.traversal*int(second["use"].orientation):
            raise _Refusal("interface_material_sides_unqualified")
        deltas = []
        for (angle1, z1), (angle2, z2) in zip(first["reference"], second["reference"]):
            turns = p.div(p.sub(angle2, angle1), period)
            low, high = math.ceil(turns.lo), math.floor(turns.hi)
            if low != high or abs(low) > 1 or _upper_abs(p.sub(z2, z1)) > geometry.tau:
                raise _Refusal("interface_correspondence_unqualified")
            deltas.append(low)
        if deltas[0] != deltas[1]:
            raise _Refusal("interface_lift_inconsistent")
        delta = deltas[0]
        if delta:
            if a.carrier != "AXIAL":
                raise _Refusal("non_axial_reference_seam")
            low_angle = min(first["reference"][0][0].mid, second["reference"][0][0].mid)
            if abs(low_angle-cut) > geometry.angular_tolerance:
                raise _Refusal("undeclared_periodic_cut")
        external = tuple(context.handle("coedge", key) for key in context.incidence[edge_id] if key not in context.coedges)
        interfaces.append(CylinderInterface(
            id=f"edge/{edge_id}/interface", edge=context.handle("edge", edge_id),
            occurrences=(a.id, b.id), traversal_same=a.traversal == b.traversal,
            lift_delta=delta, is_reference_seam=bool(delta), external_coedges=external,
        ))
        paired.update(identifiers)
        adjacency[a.face_use.id].add(b.face_use.id)
        adjacency[b.face_use.id].add(a.face_use.id)
    visited, pending = set(), [reference.id]
    while pending:
        p.charge("winding_steps", policy.max_pair_tests, 32)
        key = pending.pop()
        if key not in visited:
            visited.add(key)
            pending.extend(sorted(adjacency[key]-visited, reverse=True))
    if len(visited) != len(context.uses):
        raise _Refusal("disconnected_or_unpartitioned_sectors", missing=True)
    if not any(item.is_reference_seam for item in interfaces):
        raise _Refusal("missing_source_periodic_interface", missing=True)

    # Whole selected polygons: boundary crossing and contained-sector checks.
    use_ids = sorted(context.uses)
    for index, first_id in enumerate(use_ids):
        first_loops = polygon_data[first_id]
        for second_id in use_ids[index+1:]:
            second_loops = polygon_data[second_id]
            first_box, second_box = _bounds(first_loops[0]["points"]), _bounds(second_loops[0]["points"])
            if _separated(first_box, second_box, geometry.tau):
                continue
            for first_loop in first_loops:
                for second_loop in second_loops:
                    for a_id in first_loop["ids"]:
                        for b_id in second_loop["ids"]:
                            a, b = details[a_id], details[b_id]
                            _pair_charge(p)
                            a_record, b_record = a["occurrence"], b["occurrence"]
                            if a_record.edge == b_record.edge:
                                continue
                            shared = {a_record.start_vertex, a_record.end_vertex} & {b_record.start_vertex, b_record.end_vertex}
                            if shared:
                                # Only a unique incident endpoint is exempt. The
                                # other endpoint must remain separated; coincident
                                # carriers sharing both endpoints require real reuse.
                                if len(shared) != 1:
                                    raise _Refusal("source_partition_required", missing=True)
                                shared_vertex = next(iter(shared))
                                a_other = a["points"][1 if a_record.start_vertex == shared_vertex else 0]
                                b_other = b["points"][1 if b_record.start_vertex == shared_vertex else 0]
                                if not _separated(a_other, b_other, geometry.tau):
                                    raise _Refusal("source_partition_required", missing=True)
                                a_root = a["points"][0 if a_record.start_vertex == shared_vertex else 1]
                                b_root = b["points"][0 if b_record.start_vertex == shared_vertex else 1]
                                da, db = p.vsub(a_other, a_root), p.vsub(b_other, b_root)
                                if a_record.carrier == b_record.carrier:
                                    if p.dot(da, db).hi >= 0:
                                        raise _Refusal("source_partition_required", missing=True)
                                else:
                                    determinant = p.sub(p.mul(da[0], db[1]), p.mul(da[1], db[0]))
                                    _positive_sign(determinant, "incident_corner_unqualified")
                                continue
                            clearance = geometry.tau+a["curve"]["residual"].hi+b["curve"]["residual"].hi
                            if not _separated(_bounds(a["points"]), _bounds(b["points"]), clearance):
                                raise _Refusal("sector_boundary_contact_unqualified")
            # Genuine positive-area AABB intersection requires an interior test.
            overlap = [min(a.hi, b.hi)-max(a.lo, b.lo) for a, b in zip(first_box, second_box)]
            if min(overlap) > 2*geometry.tau:
                for own, other in ((first_loops, second_loops), (second_loops, first_loops)):
                    seed = _material_seed(p, own[0]["points"], geometry.tau)
                    if any(_inside(p, seed, h["points"]) for h in own[1:]):
                        raise _Refusal("material_seed_in_hole_unqualified")
                    if _inside(p, seed, other[0]["points"]) and not any(_inside(p, seed, h["points"]) for h in other[1:]):
                        raise _Refusal("overlapping_sector_material")

    outgoing, exposed = defaultdict(list), {}
    for occurrence in occurrences:
        if occurrence.id in paired:
            continue
        direction = occurrence.traversal*int(details[occurrence.id]["use"].orientation)
        start = occurrence.start_vertex if direction == 1 else occurrence.end_vertex
        end = occurrence.end_vertex if direction == 1 else occurrence.start_vertex
        outgoing[start.id].append(occurrence.id)
        exposed[occurrence.id] = (start, end, direction)
    # Compare exposed boundaries also across +/- one periodic image. This
    # rejects a zero-width slit made from equal coordinates/different identities.
    exposed_ids = sorted(exposed)
    for i, first_id in enumerate(exposed_ids):
        a = details[first_id]
        ar = a["occurrence"]
        for second_id in exposed_ids[i+1:]:
            b = details[second_id]
            br = b["occurrence"]
            common = {ar.start_vertex, ar.end_vertex} & {br.start_vertex, br.end_vertex}
            if common:
                if len(common) != 1:
                    raise _Refusal("source_partition_required", missing=True)
                vertex = next(iter(common))
                ai = 0 if ar.start_vertex == vertex else 1
                bi = 0 if br.start_vertex == vertex else 1
                da = p.vsub(a["points"][1-ai], a["points"][ai])
                db = p.vsub(b["points"][1-bi], b["points"][bi])
                if ar.carrier == br.carrier:
                    if p.dot(da, db).hi >= 0:
                        raise _Refusal("exposed_incident_overlap_unqualified")
                else:
                    _positive_sign(p.sub(p.mul(da[0], db[1]), p.mul(da[1], db[0])), "exposed_corner_unqualified")
                continue
            for image in (-1, 0, 1):
                _pair_charge(p)
                shift_x = p.mul(image, p.mul(radius, period))
                shifted = tuple((p.add(point[0], shift_x), point[1]) for point in b["points"])
                clearance = geometry.tau+a["curve"]["residual"].hi+b["curve"]["residual"].hi
                if not _separated(_bounds(a["points"]), _bounds(shifted), clearance):
                    raise _Refusal("periodic_boundary_clearance_unqualified")
    if any(len(items) != 1 for items in outgoing.values()):
        raise _Refusal("boundary_junction_ambiguous")
    cycles, unused = [], set(exposed)
    while unused:
        current = min(unused)
        initial = exposed[current][0]
        members, vertices, total_angle, axial = [], [], p.i(0), []
        while True:
            p.charge("winding_steps", policy.max_pair_tests, 32)
            if current not in unused:
                raise _Refusal("boundary_cycle_not_simple")
            unused.remove(current)
            start, end, direction = exposed[current]
            data = details[current]
            angles = tuple(v[0] for v in data["reference"])
            total_angle = p.add(total_angle, p.mul(direction, p.sub(angles[1], angles[0])))
            axial.extend(v[1] for v in data["reference"])
            members.append((current, direction))
            vertices.append(start)
            if end == initial:
                break
            if len(outgoing[end.id]) != 1:
                raise _Refusal("open_boundary_cycle")
            current = outgoing[end.id][0]
        winding_interval = p.div(total_angle, period)
        low = math.ceil(winding_interval.lo)
        high = math.floor(winding_interval.hi)
        if low != high or abs(low) > 1:
            raise _Refusal("boundary_winding_unqualified")
        cycles.append((members, vertices, low, _hull(axial)))
    rings = sorted((c for c in cycles if c[2]), key=lambda c: c[3].mid)
    if len(rings) != 2 or {c[2] for c in rings} != {-1, 1} or rings[0][3].hi+geometry.tau >= rings[1][3].lo:
        raise _Refusal("end_ring_winding_unqualified")
    boundary_cycles = []
    for cycle in sorted(cycles, key=lambda c: (c[2] == 0, c[3].mid, c[0][0][0])):
        members, vertices, winding, _ = cycle
        role = "HOLE" if winding == 0 else "LOWER" if cycle is rings[0] else "UPPER"
        boundary_cycles.append(CylinderBoundaryCycle(tuple(members), role, winding, tuple(vertices)))
    return context, geometry, tuple(sectors), tuple(occurrences), tuple(interfaces), tuple(boundary_cycles), details


def _certificate(model, proof, complete):
    geometry = getattr(proof, "geometry", None)
    extent = geometry.extent if geometry is not None else 0.
    return CylinderAtlasCertificate(
        algorithm="cylinder-sector-orthogonal-v1", complete=complete,
        tolerance_length=model.tolerance.effective_length(extent),
        tolerance_surface=model.tolerance.effective_surface_residual(extent),
        tolerance_angular=model.tolerance.angular,
        max_residual=proof.out(proof.i(geometry.max_residual))[1] if geometry is not None else 0.,
        max_enclosure_width=proof.out(proof.i(geometry.max_width))[1] if geometry is not None else 0.,
        work_counts=tuple(proof.counts.items()),
    )


def _run_query(model, face_uses, reference_face_use, expected_revision, policy, cancellation_check):
    policy = CylinderAtlasPolicy() if policy is None else policy
    requested = _request(model, face_uses, reference_face_use, expected_revision, cancellation_check, policy)
    proof = _Proof(policy, cancellation_check)
    proof.cancel()
    sectors = occurrences = interfaces = cycles = ()
    details = context = geometry = None
    status, diagnostics = CylinderAtlasStatus.QUALIFIED, ()
    try:
        context, geometry, sectors, occurrences, interfaces, cycles, details = _qualify(
            model, requested, reference_face_use, policy, proof,
        )
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        status = CylinderAtlasStatus.CAPABILITY_MISSING if error.missing else CylinderAtlasStatus.UNRESOLVED
        diagnostics = (str(error),)
        sectors = occurrences = interfaces = cycles = ()
    proof.cancel("cylinder atlas result")
    _unchanged(model, expected_revision)
    data = dict(
        contract_version=1, model_id=model.model_id, revision=int(expected_revision),
        requested_face_uses=requested, reference_face_use=reference_face_use,
        policy=policy, status=status, sectors=sectors, occurrences=occurrences,
        interfaces=interfaces, boundary_cycles=cycles,
        certificate=_certificate(model, proof, status is CylinderAtlasStatus.QUALIFIED),
        diagnostics=diagnostics,
    )
    return CylinderAtlasResult(**data, digest=_digest(data)), context, geometry, details, proof


def query_cylinder_atlas(
    model: GeometryModel, face_uses, *, reference_face_use: EntityHandle,
    expected_revision: int, policy: CylinderAtlasPolicy | None = None,
    cancellation_check: Callable[[str], None] | None = None,
) -> CylinderAtlasResult:
    """Qualify existing sector incidence and material branches without mutation.

    No virtual cut or missing source partition is created. A non-success result
    contains diagnostics, never a partially consumable set of sector charts.
    """
    return _run_query(model, face_uses, reference_face_use, expected_revision, policy, cancellation_check)[0]


def _validate_bound(model, atlas, face_uses, reference, revision, callback):
    if not isinstance(atlas, CylinderAtlasResult):
        raise _invalid("expected CylinderAtlasResult")
    if not isinstance(atlas.policy, CylinderAtlasPolicy):
        raise _invalid("invalid atlas policy")
    requested = _request(model, face_uses, reference, revision, callback, atlas.policy)
    if atlas.model_id != model.model_id:
        raise _error(CylinderAtlasErrorCode.WRONG_MODEL, "atlas belongs to another model")
    if atlas.revision != revision:
        raise _error(CylinderAtlasErrorCode.STALE_RESULT, "atlas revision differs")
    if atlas.requested_face_uses != requested or atlas.reference_face_use != reference:
        raise _invalid("caller-selected/reference FaceUses differ")
    # Length gates precede recursive validation of caller-provided evidence.
    if (type(atlas.sectors) is not tuple or len(atlas.sectors) > atlas.policy.max_face_uses
            or any(type(value) is not tuple or len(value) > atlas.policy.max_occurrences
                   for value in (atlas.occurrences, atlas.interfaces, atlas.boundary_cycles))):
        raise _invalid("oversized atlas evidence")
    if callback is not None:
        callback("cylinder atlas binding validation")
    budget = _BindingBudget(atlas.policy, callback)
    token = _binding_budget.set(budget)
    try:
        rebuilt = replace(atlas)
        data = {field.name: getattr(rebuilt, field.name) for field in fields(rebuilt) if field.name != "digest"}
        if rebuilt.digest != _digest(data):
            raise _invalid("atlas digest differs from content")
        canonical = _canonical(rebuilt)
    finally:
        _binding_budget.reset(token)
    fresh, context, geometry, details, proof = _run_query(model, requested, reference, revision, atlas.policy, callback)
    token = _binding_budget.set(budget)
    try:
        if canonical != _canonical(fresh):
            raise _invalid("atlas differs from fresh owner qualification")
    finally:
        _binding_budget.reset(token)
    return fresh, context, geometry, details, proof


def validate_cylinder_atlas_binding(
    model: GeometryModel, atlas: CylinderAtlasResult, face_uses, *,
    reference_face_use: EntityHandle, expected_revision: int,
    cancellation_check: Callable[[str], None] | None = None,
) -> None:
    """Bind to caller expectations AND requalify; a digest alone is not proof."""
    _validate_bound(model, atlas, face_uses, reference_face_use, expected_revision, cancellation_check)


def _sample(model, atlas, context, geometry, details, proof, request):
    if request.occurrence_id not in details:
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "unknown occurrence")
    data = details[request.occurrence_id]
    occurrence, curve, mapping = data["occurrence"], data["curve"], data["mapping"]
    edge, surface, p = curve["edge"], mapping["surface"], proof
    t = _q(request.parameter)
    endpoint = context.handle("vertex", edge.start if t == 0 else edge.end) if t in (0, 1) else None
    if endpoint is not None:
        point = np.asarray(context.vertices[endpoint.id].position, dtype=float)
    elif isinstance(edge.curve, Arc):
        point = sample_arc(geometry.frame(edge), np.asarray([float(t)]))[0]
    else:
        point = sample_straight(context.vertices[edge.start].position, context.vertices[edge.end].position, np.asarray([float(t)]))[0]
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise _Refusal("nonfinite_owner_evaluation")
    angle = p.add(curve["carrier_angle0"], p.mul(t, curve["slope"]))
    z = curve["carrier_z0"]
    if curve["carrier"] == "AXIAL":
        z = p.add(z, p.mul(t, p.sub(curve["end_z"], curve["start_z"])))
    u = p.div(p.sub(angle, surface.start_angle), surface.sweep_angle)
    v = p.div(z, _q(surface.height)/geometry.scale)
    local_uv = float(u.mid), float(v.mid)
    # Check the actual returned floating UV, not just an ideal interval midpoint.
    returned_angle = p.add(surface.start_angle, p.mul(local_uv[0], surface.sweep_angle))
    returned_z = p.mul(local_uv[1], _q(surface.height)/geometry.scale)
    sine, cosine = p.sincos(returned_angle)
    r, c, axis, _ = mapping["basis"]
    radial = p.vmul(p.vadd(p.vmul(r, cosine), p.vmul(c, sine)), _q(surface.radius)/geometry.scale)
    expected = p.vadd(mapping["origin"], p.vadd(radial, p.vmul(axis, returned_z)))
    residual = p.norm(p.vsub(geometry.point(point), expected))
    geometry.accept_residual(residual)
    angular_error, axial_error = geometry.coordinate_errors(residual, mapping["basis"], _q(surface.radius)/geometry.scale)
    padding = max(curve["parameter_error"], axial_error.hi/(abs(_q(surface.height))/geometry.scale),
                  angular_error.hi/abs(_q(surface.sweep_angle)))
    local_boxes = tuple(p.add(value, p.i(-padding, padding)) for value in local_uv)
    reference_angle = p.add(p.add(mapping["phase"], p.mul(mapping["sign"], returned_angle)), p.mul(mapping["shift"], p.mul(2, p.pi_bound())))
    turns = p.div(p.sub(reference_angle, geometry.reference.start_angle), p.mul(2, p.pi_bound()))
    axial = p.mul(p.add(mapping["z"], p.mul(mapping["sign"], returned_z)), geometry.scale)
    composed_residual = p.add(residual, p.mul(mapping["frame_error"], max(Fraction(1), abs(_q(local_uv[1])))))
    geometry.accept_residual(composed_residual)
    reference_angular_error, reference_axial_error = geometry.coordinate_errors(
        composed_residual, geometry.reference_basis, _q(geometry.reference.radius)/geometry.scale)
    turn_error = reference_angular_error.hi/p.mul(2, p.pi_bound()).lo
    axial_error = reference_axial_error.hi*geometry.scale
    ref_boxes = (p.add(turns, p.i(-turn_error, turn_error)), p.add(axial, p.i(-axial_error, axial_error)))
    physical_width = p.norm((p.mul(ref_boxes[0].hi-ref_boxes[0].lo,
                                  p.mul(p.mul(2, p.pi_bound()), geometry.reference.radius)),
                             p.i(ref_boxes[1].hi-ref_boxes[1].lo)))
    geometry.max_width = max(geometry.max_width, physical_width.hi)
    key_data = (atlas.model_id, atlas.revision, atlas.digest, endpoint) if endpoint is not None else (
        atlas.model_id, atlas.revision, atlas.digest, occurrence.edge, request.parameter.hex(),
    )
    return CylinderOccurrenceSample(
        occurrence_id=request.occurrence_id, parameter=request.parameter,
        point=tuple(float(x) for x in point), local_uv=local_uv,
        lifted_reference=(float(turns.mid), float(axial.mid)),
        local_uv_enclosure=tuple(p.out(x) for x in local_boxes),
        reference_enclosure=tuple(p.out(x) for x in ref_boxes), edge=occurrence.edge,
        endpoint_vertex=endpoint, equivalence_key=_digest(key_data),
        residual_bound=p.out(p.mul(residual, geometry.scale))[1],
        status=CylinderAtlasStatus.QUALIFIED,
    )


def evaluate_cylinder_occurrences(
    model: GeometryModel, atlas: CylinderAtlasResult, requests, *, face_uses,
    reference_face_use: EntityHandle, expected_revision: int,
    cancellation_check: Callable[[str], None] | None = None,
) -> CylinderOccurrenceEvaluation:
    """Forward native Edge parameters through a freshly verified owner atlas."""
    fresh, context, geometry, details, proof = _validate_bound(
        model, atlas, face_uses, reference_face_use, expected_revision, cancellation_check,
    )
    if fresh.status is not CylinderAtlasStatus.QUALIFIED:
        raise _error(CylinderAtlasErrorCode.UNQUALIFIED_RESULT, "atlas is not qualified")
    made = _bounded(requests, fresh.policy.max_evaluations, "bounded occurrence requests required")
    if any(not isinstance(request, CylinderOccurrenceRequest) for request in made):
        raise _error(CylinderAtlasErrorCode.INVALID_REQUEST, "expected CylinderOccurrenceRequest")
    samples = []
    try:
        for request in made:
            request = replace(request)
            proof.charge("evaluations", fresh.policy.max_evaluations, 16)
            samples.append(_sample(model, fresh, context, geometry, details, proof, request))
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        raise _error(CylinderAtlasErrorCode.UNQUALIFIED_RESULT, str(error)) from error
    proof.cancel("cylinder atlas evaluated result")
    _unchanged(model, expected_revision)
    return CylinderOccurrenceEvaluation(model.model_id, int(expected_revision), fresh.digest, tuple(samples), _certificate(model, proof, True))


__all__ = [
    "CylinderAtlasStatus", "CylinderAtlasErrorCode", "CylinderAtlasError",
    "CylinderAtlasPolicy", "CylinderAtlasCertificate", "CylinderSectorChart",
    "CylinderOccurrence", "CylinderInterface", "CylinderBoundaryCycle",
    "CylinderAtlasResult", "CylinderOccurrenceRequest", "CylinderOccurrenceSample",
    "CylinderOccurrenceEvaluation", "query_cylinder_atlas",
    "validate_cylinder_atlas_binding", "evaluate_cylinder_occurrences",
]
