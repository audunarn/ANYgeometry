"""Exact nonrational angular coordinate and authenticated Cylinder ruling lift.

A rational footpoint such as (3, 4) on a radius-5 cylinder has the exact world
angle ``atan2(4, 3)``, which is not a rational number of radians and cannot be
replaced by a rounded float or a Fraction chart coordinate.  This module
represents that angle exactly as a canonical rational ray plus an integer
winding, meaning ``atan2(y, x)`` in ``[0, 2*pi) + 2*pi*winding``, and certifies
support-window membership with the existing counted outward-arithmetic
enclosures of :mod:`anygeometry.cylinder_charts`.  Angle values are only ever
published as enclosures, never as exact rational angles.

The producer is deliberately narrow.  For one active Cylinder face and one of
its actual boundary edge occurrences it proves, entirely with exact rational
arithmetic on the live surface's RAW stored coefficients (no decode/rebuild
normalization round trip):

* the frame ``[radius*radial, radius*circumferential, height*axis]`` has a
  nonzero exact Fraction determinant, and the sweep is a strict subturn
  (``abs(sweep) < float(math.tau)``, a conservative boundary below the true
  ``2*pi``), so the chart is injective on the support window.  No constant
  Jacobian, area or orthonormality claim is made or needed.
* a whole straight boundary edge (``Straight`` truth via ``freeze_edge``)
  solves exactly to CONSTANT radial coefficients ``(qx, qy)`` with
  ``qx*qx + qy*qy == 1`` and distinct ``v`` endpoints in ``[0, 1]``.  Because
  the solve is linear, the edge's complete analytic line then lies on the
  captured carrier; off-carrier rounded rulings, curved edges, singular or
  full-turn carriers, explicit parameterizations, inactive entities and
  wrong-owner occurrences refuse without projection or snapping.
* the exact angle of ``(qx, qy)`` lies in the face support window
  ``[start, start + sweep]`` (either orientation) for exactly one integer
  winding, certified by interval enclosure comparison.  Unresolved boundary
  equality, ambiguous period lifts and exhausted budgets refuse; negative
  sweeps and far periodic gauges are certified, never clamped.

Every proof input -- the occurrence, the raw frame, the frozen LinePath
endpoints and the edge vertices -- is captured detached from the live model
BEFORE any cancellation callback runs, and the proof compiles only that
immutable entry evidence, so a callback that mutates and later restores the
model can never publish temporary geometry.  After every callback the guards
recheck the revision and a raw frame pin that includes the stored
circumferential coefficient (which the document checksum does not serialize
even though ``evaluate`` uses it), and after the FINAL callback the full
document checksum is rechecked as well.  The immutable receipt binds model
identity, revision, that checksum, the actual face/edge occurrence, the raw
captured frame, the exact ray/winding angle and the exact ``v`` endpoints.
Binding validation pins the entry receipt digest before any callback,
rederives the whole proof live, and accepts only a fresh receipt matching the
pinned digest -- never a caller object a callback could have repaired.

This module establishes no whole-face partition, source-current equivalence,
reference, joint, node or meshing acceptance, and its queries never author or
mutate geometry.  Existing public material-partition refusals are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from fractions import Fraction
from hashlib import sha256
import json
import math
from numbers import Integral
from uuid import UUID

from .arrangement_geometry import LinePath, freeze_edge
from .cylinder_charts import CylinderAtlasPolicy, _Proof, _Refusal, _q
from .errors import GeometryError
from .model import GeometryModel
from .serialization import _serialized_model_state
from .surfaces import Cylinder

__all__ = [
    "CylinderAngularLiftError",
    "CylinderAngularLiftErrorCode",
    "CylinderRulingLift",
    "ExactRay",
    "RAY_HALF_TURN",
    "RAY_QUARTER_TURN",
    "RAY_THREE_QUARTER_TURN",
    "RAY_ZERO",
    "compare_exact_rays",
    "cylinder_ruling_lift_digest",
    "exact_ray",
    "query_cylinder_straight_ruling_lift",
    "validate_cylinder_straight_ruling_lift_binding",
]


class CylinderAngularLiftErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    WRONG_MODEL = "WRONG_MODEL"
    INACTIVE_ENTITY = "INACTIVE_ENTITY"
    STALE_REVISION = "STALE_REVISION"
    CHANGED_MODEL = "CHANGED_MODEL"
    BUSY_MODEL = "BUSY_MODEL"
    UNQUALIFIED = "UNQUALIFIED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CANCELLED = "CANCELLED"
    INVALID_RESULT = "INVALID_RESULT"


class CylinderAngularLiftError(GeometryError):
    """Typed refusal with a machine-readable code; never a partial receipt."""

    def __init__(self, code, diagnostics):
        object.__setattr__(self, "_code", CylinderAngularLiftErrorCode(code))
        object.__setattr__(self, "_diagnostics", tuple(diagnostics))
        super().__init__(f"{self.code.value}: {'; '.join(self.diagnostics)}")

    code = property(lambda self: self._code)
    diagnostics = property(lambda self: self._diagnostics)

    def __setattr__(self, name, value):
        if name in ("code", "diagnostics", "_code", "_diagnostics"):
            raise AttributeError("immutable lift diagnostics")
        super().__setattr__(name, value)


def _error(code, text):
    return CylinderAngularLiftError(code, (text,))


def _rational(value, name):
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     f"{name} must be an exact integer or Fraction")
    return Fraction(int(value)) if isinstance(value, Integral) else Fraction(value)


@dataclass(frozen=True, slots=True)
class ExactRay:
    """Canonical exact rational ray: the direction of ``atan2(y, x)``.

    The pair is the primitive integer representative of the direction, so
    ``(x, y)`` and ``(k*x, k*y)`` for positive rational ``k`` share one value.
    The represented canonical angle is ``atan2(y, x)`` lifted into
    ``[0, 2*pi)``; it is generally irrational and is never stored as a number.
    """

    x: Fraction
    y: Fraction


def exact_ray(x, y):
    """Canonical primitive ray of an exact rational direction."""
    x = _rational(x, "ray x")
    y = _rational(y, "ray y")
    if x == 0 and y == 0:
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "a ray needs a nonzero direction")
    first = x.numerator * y.denominator
    second = y.numerator * x.denominator
    divisor = math.gcd(abs(first), abs(second))
    return ExactRay(Fraction(first, divisor), Fraction(second, divisor))


RAY_ZERO = exact_ray(1, 0)
RAY_QUARTER_TURN = exact_ray(0, 1)
RAY_HALF_TURN = exact_ray(-1, 0)
RAY_THREE_QUARTER_TURN = exact_ray(0, -1)


def _require_ray(ray):
    if type(ray) is not ExactRay:
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST, "ExactRay required")
    x, y = ray.x, ray.y
    if (type(x) is not Fraction or type(y) is not Fraction or x == 0 == y
            or x.denominator != 1 or y.denominator != 1
            or math.gcd(abs(int(x)), abs(int(y))) != 1):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "ray is not in canonical primitive form")


def _ray_half(ray):
    """0 for the half turn ``[0, pi)``, 1 for ``[pi, 2*pi)``."""
    return 0 if (ray.y > 0 or (ray.y == 0 and ray.x > 0)) else 1


def compare_exact_rays(first, second):
    """Exact order of the canonical angles in ``[0, 2*pi)``; -1, 0 or 1.

    Ordering uses only the rational determinant and half-plane membership of
    the two rays, so it is exact for irrational angles.
    """
    _require_ray(first)
    _require_ray(second)
    if first == second:
        return 0
    first_half, second_half = _ray_half(first), _ray_half(second)
    if first_half != second_half:
        return -1 if first_half < second_half else 1
    determinant = first.x * second.y - first.y * second.x
    if determinant > 0:
        return -1
    if determinant < 0:
        return 1
    raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                 "distinct rays share one direction")


@dataclass(frozen=True, slots=True)
class CylinderRulingLift:
    """Immutable authenticated lift of one whole straight Cylinder ruling.

    ``angle_enclosure`` is an outward enclosure of the canonical angle
    ``atan2(ray.y, ray.x)`` in ``[0, 2*pi)``; the certified lift into the
    support window is that angle plus ``2*pi*winding``.  All frame fields are
    the RAW stored live-surface coefficients captured without reconstruction.
    """

    model_id: UUID
    revision: int
    source_checksum: str
    face_id: int
    edge_id: int
    loop_index: int
    loop_position: int
    forward: bool
    start_vertex: int
    end_vertex: int
    origin: tuple
    axis: tuple
    radial_direction: tuple
    circumferential_direction: tuple
    radius: Fraction
    height: Fraction
    start_angle: Fraction
    sweep_angle: Fraction
    frame_determinant: Fraction
    ray: ExactRay
    winding: int
    angle_enclosure: tuple
    support_interval: tuple
    v_start: Fraction
    v_end: Fraction
    policy_limits: tuple
    work_counts: tuple


def _canonical(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Fraction):
        return ["fraction", str(value.numerator), str(value.denominator)]
    if isinstance(value, UUID):
        return value.hex
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        return {"type": type(value).__qualname__,
                "fields": {field.name: _canonical(getattr(value, field.name))
                           for field in fields(value)}}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                 "ruling lift content is not bindable")


def cylinder_ruling_lift_digest(receipt):
    """Content digest of a ruling-lift receipt for binding comparison."""
    if type(receipt) is not CylinderRulingLift:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "CylinderRulingLift required")
    payload = json.dumps(_canonical(receipt), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _document_checksum(model):
    return _serialized_model_state(model)["checksum"]["value"]


def _raw_pin(surface):
    """Detached raw coefficient pin, including the stored circumferential.

    The document checksum does not serialize ``surface._circumferential``
    even though ``evaluate`` uses it, so this pin is the only faithful
    identity of the raw frame the proof captured.
    """
    return (
        tuple(_q(float(value)) for value in surface.origin),
        tuple(_q(float(value)) for value in surface.axis),
        tuple(_q(float(value)) for value in surface.radial_direction),
        tuple(_q(float(value)) for value in surface.circumferential_direction),
        _q(float(surface.radius)),
        _q(float(surface.height)),
        _q(float(surface.start_angle)),
        _q(float(surface.sweep_angle)),
    )


def _recheck_pins(model, entry, revision):
    """Typed refusal when a callback visibly mutated the model or raw inputs.

    Runs after every cancellation callback invocation, including the FINAL
    one.  A mutation still visible at any guard refuses; the raw frame pin
    catches private late mutations the document checksum cannot see.  A
    temporary mutation a callback fully restores before the next guard
    never enters the proof, because the compiler uses only the immutable
    entry evidence captured before any callback.
    """
    try:
        current_face = model.faces.get(entry["face_id"])
        changed = (model.revision != revision
                  or model._transaction_journal is not None
                  or model._notifying_hooks
                  or current_face is None
                  or type(current_face.surface) is not Cylinder
                  or _raw_pin(current_face.surface) != entry["frame"]["pin"])
    except (GeometryError, _Refusal) as error:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     str(error)) from error
    if changed:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "the model changed during the query")


def _guard(callback, model, entry, revision):
    if callback is None:
        return None

    def checked(phase):
        if callback(phase):
            raise _error(CylinderAngularLiftErrorCode.CANCELLED, phase)
        _recheck_pins(model, entry, revision)

    return checked


def _request(model, face_id, edge_id, expected_revision, cancellation_check, policy):
    if not isinstance(model, GeometryModel):
        raise TypeError("GeometryModel required")
    for name, value in (("face ID", face_id), ("edge ID", edge_id)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                         f"positive integer {name} required")
    if expected_revision is not None and (isinstance(expected_revision, bool)
                                          or not isinstance(expected_revision, Integral)
                                          or expected_revision < 0):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "expected revision must be a non-negative integer")
    if cancellation_check is not None and not callable(cancellation_check):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "cancellation check must be callable")
    if policy is not None and type(policy) is not CylinderAtlasPolicy:
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "CylinderAtlasPolicy required")
    if model._transaction_journal is not None or model._notifying_hooks:
        raise _error(CylinderAngularLiftErrorCode.BUSY_MODEL,
                     "committed model required")
    if expected_revision is not None and int(expected_revision) != model.revision:
        raise _error(CylinderAngularLiftErrorCode.STALE_REVISION,
                     "model revision differs from the expected revision")
    return int(face_id), int(edge_id)


def _determinant(columns):
    """Exact determinant of the 3x3 matrix with the given column vectors."""
    first, second, third = columns
    return (first[0] * (second[1] * third[2] - second[2] * third[1])
            - second[0] * (first[1] * third[2] - first[2] * third[1])
            + third[0] * (first[1] * second[2] - first[2] * second[1]))


def _capture_frame(surface):
    """RAW stored live coefficients; no decode, reconstruction or normalization.

    ``circumferential_direction`` is the stored computed coefficient the public
    ``evaluate`` path uses, so the captured frame is faithful to the authored
    carrier exactly as the live surface stores it.  ``pin`` is the detached
    identity of that raw frame used by the callback guards.
    """
    pin = _raw_pin(surface)
    origin, axis, radial, circum, radius, height, start, sweep = pin
    frame = {
        "origin": origin,
        "axis": axis,
        "radial": radial,
        "circum": circum,
        "radius": radius,
        "height": height,
        "start": start,
        "sweep": sweep,
        "pin": pin,
    }
    if frame["radius"] <= 0 or frame["height"] == 0 or frame["sweep"] == 0:
        raise _Refusal("singular_cylinder_carrier")
    if abs(frame["sweep"]) >= Fraction(math.tau):
        # Conservative float-tau boundary: strictly below the true 2*pi, so
        # every accepted window is a strict subturn and the chart is injective.
        raise _Refusal("full_turn_sweep_refused", missing=True)
    radial, circum, axis = frame["radial"], frame["circum"], frame["axis"]
    radius, height = frame["radius"], frame["height"]
    frame["columns"] = (
        tuple(radius * radial[i] for i in range(3)),
        tuple(radius * circum[i] for i in range(3)),
        tuple(height * axis[i] for i in range(3)),
    )
    frame["determinant"] = _determinant(frame["columns"])
    if frame["determinant"] == 0:
        raise _Refusal("singular_chart_frame")
    return frame


def _frame_coordinates(frame, point):
    """Exact Cramer solve of the full-rank frame for one rational point."""
    origin, columns = frame["origin"], frame["columns"]
    determinant = frame["determinant"]
    offset = tuple(value - base for value, base in zip(point, origin))
    first, second, third = columns
    qx = _determinant((offset, second, third)) / determinant
    qy = _determinant((first, offset, third)) / determinant
    v = _determinant((first, second, offset)) / determinant
    return (qx, qy), v


def _certify_support(frame, ray, proof):
    """Certify exactly one winding puts the exact angle in the support window.

    The canonical angle enclosure comes from the existing counted outward
    arithmetic ``atan2``/``pi_bound``; membership is decided only when the
    enclosure is strictly inside or strictly outside the rational window.
    Candidate windings are enclosed by dividing ``[low - theta.hi,
    high - theta.lo]`` by the positive ``2*pi`` enclosure with the existing
    counted ``div``; ceil/floor of that outward enclosure covers every
    candidate for negative offsets too.  Each tested period is charged
    counted arithmetic, so the explicit policy budget alone controls the
    work; unresolved membership and budget exhaustion refuse typed, never a
    false outside claim.
    """
    start, sweep = frame["start"], frame["sweep"]
    low, high = (start, start + sweep) if sweep > 0 else (start + sweep, start)
    base = proof.atan2(ray.y, ray.x)
    theta = proof.add(base, proof.mul(2, proof.pi_bound())) if ray.y < 0 else base
    two_pi = proof.mul(2, proof.pi_bound())
    reach = proof.div(proof.i(low - theta.hi, high - theta.lo), two_pi)
    first = math.ceil(reach.lo)
    last = math.floor(reach.hi)
    winding = None
    for candidate in range(first, last + 1):
        shifted = proof.add(theta, proof.mul(candidate, two_pi))
        if shifted.lo > high or shifted.hi < low:
            continue
        if shifted.lo >= low and shifted.hi <= high:
            if winding is not None:
                raise _Refusal("ambiguous_period_lift")
            winding = candidate
            continue
        raise _Refusal("unresolved_support_membership")
    if winding is None:
        raise _Refusal("ruling_angle_outside_support")
    return winding, (theta.lo, theta.hi), (low, high)


def _occurrence(face, edge_id):
    found = None
    for index, loop in enumerate((face.loop, *face.holes)):
        for position, use in enumerate(loop):
            if use.edge != edge_id:
                continue
            if found is not None:
                raise _Refusal("ambiguous_boundary_occurrence")
            found = (index, position, use.forward)
    if found is None:
        raise _Refusal("edge_not_on_face_boundary")
    return found


def _capture_entry(model, face_id, edge_id):
    """Detach every proof input from the live model BEFORE any callbacks.

    Occurrence, raw frame, frozen LinePath endpoints and edge vertices are
    captured as immutable exact data here, so no later callback mutation -
    restored or not - can enter the proof as temporary geometry.
    """
    if face_id not in model.faces:
        raise _Refusal("inactive_face")
    if edge_id not in model.edges:
        raise _Refusal("inactive_edge")
    face = model.faces[face_id]
    loop_index, loop_position, forward = _occurrence(face, edge_id)
    surface = face.surface
    if type(surface) is not Cylinder:
        raise _Refusal("cylinder_support_required", missing=True)
    if face.parameterization is not None:
        raise _Refusal("explicit_parameterization_refused", missing=True)
    frame = _capture_frame(surface)
    try:
        path = freeze_edge(model, edge_id)
    except GeometryError as error:
        raise _Refusal("curved_boundary_edge") from error
    if type(path) is not LinePath:
        raise _Refusal("curved_boundary_edge", missing=True)
    edge = model.edges[edge_id]
    return {
        "model_id": model.model_id,
        "face_id": face_id,
        "edge_id": edge_id,
        "loop_index": loop_index,
        "loop_position": loop_position,
        "forward": forward,
        "surface": surface,
        "frame": frame,
        "start_point": tuple(_q(float(value)) for value in path.start),
        "end_point": tuple(_q(float(value)) for value in path.end),
        "start_vertex": edge.start,
        "end_vertex": edge.end,
    }


def _lift(entry, revision, source, proof):
    """Compile the proof from immutable entry evidence only."""
    proof.cancel("cylinder ruling lift: request")
    frame = entry["frame"]
    proof.cancel("cylinder ruling lift: exact solve")
    first_radial, v_start = _frame_coordinates(frame, entry["start_point"])
    last_radial, v_end = _frame_coordinates(frame, entry["end_point"])
    if first_radial != last_radial:
        raise _Refusal("radial_coefficients_not_constant")
    qx, qy = first_radial
    if qx * qx + qy * qy != 1:
        raise _Refusal("ruling_off_carrier")
    if v_start == v_end:
        raise _Refusal("degenerate_ruling")
    if not (0 <= v_start <= 1 and 0 <= v_end <= 1):
        raise _Refusal("ruling_outside_axial_extent")
    proof.cancel("cylinder ruling lift: angle certification")
    ray = exact_ray(qx, qy)
    winding, enclosure, support = _certify_support(frame, ray, proof)
    return CylinderRulingLift(
        entry["model_id"], revision, source, entry["face_id"], entry["edge_id"],
        entry["loop_index"], entry["loop_position"], entry["forward"],
        entry["start_vertex"], entry["end_vertex"],
        frame["origin"], frame["axis"], frame["radial"], frame["circum"],
        frame["radius"], frame["height"], frame["start"], frame["sweep"],
        frame["determinant"], ray, winding, enclosure, support,
        v_start, v_end,
        tuple(int(getattr(proof.policy, field.name)) for field in fields(proof.policy)),
        tuple(proof.counts.items()),
    )


def query_cylinder_straight_ruling_lift(model, face_id, edge_id, *,
                                        expected_revision=None,
                                        cancellation_check=None, policy=None):
    """Prove and return the exact lift of one whole straight boundary ruling.

    Every proof input is captured detached from the live model before any
    cancellation callback runs, and the proof compiles only that immutable
    entry evidence, so callback mutations can never publish temporary
    geometry.  Refuses typed for every unqualified input; a refusal never
    returns a partial receipt and never mutates the model.
    """
    face_id, edge_id = _request(model, face_id, edge_id, expected_revision,
                                cancellation_check, policy)
    if policy is None:
        policy = CylinderAtlasPolicy()
    revision = model.revision
    source = _document_checksum(model)
    try:
        entry = _capture_entry(model, face_id, edge_id)
    except _Refusal as error:
        raise _error(CylinderAngularLiftErrorCode.UNQUALIFIED,
                     str(error)) from error
    proof = _Proof(policy, _guard(cancellation_check, model, entry, revision))
    try:
        receipt = _lift(entry, revision, source, proof)
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        if str(error).startswith("qualification_budget_exhausted"):
            raise _error(CylinderAngularLiftErrorCode.BUDGET_EXHAUSTED,
                         str(error)) from error
        raise _error(CylinderAngularLiftErrorCode.UNQUALIFIED,
                     str(error)) from error
    # FINAL callback, then freshness: the guard already rechecked the revision
    # and the raw frame pin (including the unserialized circumferential
    # coefficient) after the callback; the document checksum covers the
    # remaining content, so a callback that changed the model without bumping
    # the revision still refuses here.
    proof.cancel("cylinder ruling lift: final")
    try:
        changed = (model.revision != revision
                   or model._transaction_journal is not None
                   or model._notifying_hooks
                   or _document_checksum(model) != source)
    except GeometryError as error:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     str(error)) from error
    if changed:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "the model changed during the query")
    return receipt


def _check_receipt(receipt):
    if type(receipt) is not CylinderRulingLift:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "CylinderRulingLift required")
    if not isinstance(receipt.model_id, UUID):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt model identity is invalid")
    if type(receipt.revision) is not int or receipt.revision < 0:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt revision is invalid")
    if (type(receipt.source_checksum) is not str or len(receipt.source_checksum) != 64
            or any(character not in "0123456789abcdef" for character in receipt.source_checksum)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt source checksum is invalid")
    for name in ("face_id", "edge_id", "start_vertex", "end_vertex"):
        value = getattr(receipt, name)
        if type(value) is not int or value <= 0:
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         f"receipt {name} is invalid")
    for name in ("loop_index", "loop_position"):
        value = getattr(receipt, name)
        if type(value) is not int or value < 0:
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         f"receipt {name} is invalid")
    if type(receipt.forward) is not bool or type(receipt.winding) is not int:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt occurrence orientation is invalid")
    for name in ("origin", "axis", "radial_direction", "circumferential_direction"):
        value = getattr(receipt, name)
        if (not isinstance(value, tuple) or len(value) != 3
                or any(type(item) is not Fraction for item in value)):
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         f"receipt {name} is not an exact rational 3-vector")
    for name in ("radius", "height", "start_angle", "sweep_angle",
                 "frame_determinant", "v_start", "v_end"):
        if type(getattr(receipt, name)) is not Fraction:
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         f"receipt {name} is not an exact rational")
    if receipt.radius <= 0 or receipt.height == 0 or receipt.sweep_angle == 0:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt carrier is singular")
    if abs(receipt.sweep_angle) >= Fraction(math.tau):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt sweep is not a strict subturn")
    if receipt.frame_determinant == 0:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt frame is singular")
    _require_ray(receipt.ray)
    for name in ("angle_enclosure", "support_interval"):
        value = getattr(receipt, name)
        if (not isinstance(value, tuple) or len(value) != 2
                or any(type(item) is not Fraction for item in value)
                or value[0] > value[1]):
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         f"receipt {name} is not a valid enclosure")
    if not (0 <= receipt.v_start <= 1 and 0 <= receipt.v_end <= 1
            and receipt.v_start != receipt.v_end):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt v endpoints are invalid")
    if (not isinstance(receipt.policy_limits, tuple) or len(receipt.policy_limits) != 6
            or any(type(value) is not int for value in receipt.policy_limits)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt policy limits are invalid")
    if (not isinstance(receipt.work_counts, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2
                   or type(item[0]) is not str or type(item[1]) is not int or item[1] < 0
                   for item in receipt.work_counts)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt work counts are invalid")


def validate_cylinder_straight_ruling_lift_binding(model, receipt, *,
                                                    cancellation_check=None):
    """Authenticate a receipt by live rederivation, not revision-only caching.

    The entry receipt digest and the query parameters are pinned BEFORE any
    callback runs, the whole proof runs again under the caller's
    cancellation callback (including the FINAL-callback freshness check),
    and the fresh receipt must match the pinned entry digest -- never a
    caller object a callback could have repaired in flight.  The original
    receipt must also be unchanged when the rederivation returns.
    """
    if not isinstance(model, GeometryModel):
        raise TypeError("GeometryModel required")
    _check_receipt(receipt)
    if model._transaction_journal is not None or model._notifying_hooks:
        raise _error(CylinderAngularLiftErrorCode.BUSY_MODEL,
                     "committed model required")
    if receipt.model_id != model.model_id:
        raise _error(CylinderAngularLiftErrorCode.WRONG_MODEL,
                     "ruling lift belongs to another model")
    if receipt.revision != model.revision:
        raise _error(CylinderAngularLiftErrorCode.STALE_REVISION,
                     "ruling lift is stale")
    try:
        current = _document_checksum(model)
    except GeometryError as error:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     str(error)) from error
    if current != receipt.source_checksum:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "model content changed since the ruling lift")
    entry_digest = cylinder_ruling_lift_digest(receipt)
    face_id, edge_id = receipt.face_id, receipt.edge_id
    revision = receipt.revision
    policy = CylinderAtlasPolicy(*receipt.policy_limits)
    fresh = query_cylinder_straight_ruling_lift(
        model, face_id, edge_id, expected_revision=revision,
        cancellation_check=cancellation_check, policy=policy)
    if cylinder_ruling_lift_digest(receipt) != entry_digest:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "the receipt changed during binding validation")
    if cylinder_ruling_lift_digest(fresh) != entry_digest:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "ruling lift differs from the live derivation")
