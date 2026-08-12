"""Typed, non-mutating analytical intersection predicates.

These functions deliberately operate on value inputs rather than a
``GeometryModel``.  They are suitable as narrow-phase building blocks for the
strict audit and as compatibility targets for the older intersection API.
Unsupported, degenerate, or numerically ill-conditioned inputs produce an
``UNCLASSIFIED`` result so callers cannot accidentally treat uncertainty as
disjoint geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from .errors import GeometryError
from .tolerance import DEFAULT_TOLERANCE_POLICY, TolerancePolicy, feature_extent

__all__ = [
    "IntersectionComponent",
    "IntersectionKind",
    "IntersectionQuality",
    "IntersectionResult",
    "ParameterRange",
    "qualified_line_line",
    "qualified_line_cylinder",
    "qualified_line_plane",
    "qualified_plane_plane",
    "qualified_segment_segment",
]


Point3 = tuple[float, float, float]
ParameterValue = tuple[float, ...]


class IntersectionKind(str, Enum):
    """Closed classification algebra shared by queries and strict audit."""

    DISJOINT = "disjoint"
    TOUCH_POINT = "touch_point"
    TANGENT = "tangent"
    CROSS = "cross"
    OVERLAP_CURVE = "overlap_curve"
    OVERLAP_REGION = "overlap_region"
    COINCIDENT = "coincident"
    UNCLASSIFIED = "unclassified"


class IntersectionQuality(str, Enum):
    """Qualification level attached to every returned component."""

    EXACT = "exact"
    VERIFIED_APPROXIMATE = "verified_approximate"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """Oriented scalar parameter range aligned with component witnesses.

    ``start`` may be greater than ``end``.  Keeping the parent orientation is
    important for a reversed coincident segment: the second parent's range is
    then ``1 -> 0`` while the first parent's remains ``0 -> 1``.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        start, end = float(self.start), float(self.end)
        if not np.isfinite(start) or not np.isfinite(end):
            raise GeometryError("intersection parameter ranges must be finite")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def lower(self) -> float:
        return min(self.start, self.end)

    @property
    def upper(self) -> float:
        return max(self.start, self.end)


def _parameter(value: Sequence[float] | None, name: str) -> ParameterValue | None:
    if value is None:
        return None
    try:
        made = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"{name} must contain finite values") from error
    if not made or not all(np.isfinite(item) for item in made):
        raise GeometryError(f"{name} must contain finite values")
    return made


def _point_tuple(value: Sequence[float], name: str) -> Point3:
    try:
        made = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"{name} must be a finite 3-vector") from error
    if len(made) != 3 or not all(np.isfinite(item) for item in made):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return made  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class IntersectionComponent:
    """One connected component of an intersection.

    Point components use ``first_parameter`` and ``second_parameter``.  Curve
    overlap components use oriented parameter ranges.  A plane parameter is a
    two-value ``(u, v)`` tuple; a line/segment parameter is a one-value tuple.
    Infinite curve components also carry a unit ``direction``.
    """

    witnesses: tuple[Point3, ...]
    quality: IntersectionQuality
    first_parameter: ParameterValue | None = None
    second_parameter: ParameterValue | None = None
    first_parameter_range: ParameterRange | None = None
    second_parameter_range: ParameterRange | None = None
    direction: Point3 | None = None
    max_residual: float = 0.0

    def __post_init__(self) -> None:
        witnesses = tuple(
            _point_tuple(item, "intersection witness") for item in self.witnesses
        )
        if not witnesses:
            raise GeometryError("an intersection component needs at least one witness")
        try:
            quality = IntersectionQuality(self.quality)
        except (TypeError, ValueError) as error:
            raise GeometryError("invalid intersection quality") from error
        residual = float(self.max_residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise GeometryError("intersection residual must be non-negative and finite")
        direction = (
            None
            if self.direction is None
            else _point_tuple(self.direction, "intersection direction")
        )
        object.__setattr__(self, "witnesses", witnesses)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(
            self,
            "first_parameter",
            _parameter(self.first_parameter, "first parameter"),
        )
        object.__setattr__(
            self,
            "second_parameter",
            _parameter(self.second_parameter, "second parameter"),
        )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "max_residual", residual)


@dataclass(frozen=True, slots=True)
class IntersectionResult:
    """Complete deterministic result for one predicate invocation."""

    kind: IntersectionKind
    components: tuple[IntersectionComponent, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            kind = IntersectionKind(self.kind)
        except (TypeError, ValueError) as error:
            raise GeometryError("invalid intersection kind") from error
        components = tuple(self.components)
        diagnostics = tuple(str(item) for item in self.diagnostics)
        if kind in (IntersectionKind.DISJOINT, IntersectionKind.UNCLASSIFIED):
            if components:
                raise GeometryError(f"{kind.value} results cannot contain components")
        elif not components:
            raise GeometryError(f"{kind.value} results require a component")
        if kind is IntersectionKind.UNCLASSIFIED and not diagnostics:
            raise GeometryError("unclassified results require a diagnostic")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def classified(self) -> bool:
        return self.kind is not IntersectionKind.UNCLASSIFIED

    @property
    def quality(self) -> IntersectionQuality:
        """Worst component quality, or the natural quality of an empty result."""

        qualities = {item.quality for item in self.components}
        if IntersectionQuality.UNVERIFIED in qualities:
            return IntersectionQuality.UNVERIFIED
        if IntersectionQuality.VERIFIED_APPROXIMATE in qualities:
            return IntersectionQuality.VERIFIED_APPROXIMATE
        if qualities:
            return IntersectionQuality.EXACT
        return (
            IntersectionQuality.UNVERIFIED
            if self.kind is IntersectionKind.UNCLASSIFIED
            else IntersectionQuality.EXACT
        )

    @property
    def witnesses(self) -> tuple[Point3, ...]:
        return tuple(point for component in self.components for point in component.witnesses)


@dataclass(frozen=True, slots=True)
class _PlaneData:
    origin: np.ndarray
    normal: np.ndarray
    u_vector: np.ndarray
    v_vector: np.ndarray
    extent: float


def _unclassified(reason: str) -> IntersectionResult:
    return IntersectionResult(IntersectionKind.UNCLASSIFIED, diagnostics=(reason,))


def _disjoint(reason: str) -> IntersectionResult:
    return IntersectionResult(IntersectionKind.DISJOINT, diagnostics=(reason,))


def _vector(value: object) -> np.ndarray | None:
    try:
        made = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        return None
    return np.array(made, dtype=float, copy=True)


def _stable_norm(vector: np.ndarray) -> float:
    largest = float(np.max(np.abs(vector)))
    if largest == 0.0:
        return 0.0
    made = largest * float(np.linalg.norm(vector / largest))
    return made if np.isfinite(made) else float("inf")


def _unit(value: np.ndarray) -> tuple[np.ndarray, float] | None:
    length = _stable_norm(value)
    if length == 0.0 or not np.isfinite(length):
        return None
    made = value / length
    return (made, length) if np.all(np.isfinite(made)) else None


def _canonical_direction(value: np.ndarray) -> np.ndarray:
    """Give an unoriented line a stable sign independent of parent order."""

    made = value.copy()
    index = int(np.argmax(np.abs(made)))
    if made[index] < 0.0:
        made *= -1.0
    return made


def _policy(value: TolerancePolicy | None) -> TolerancePolicy | None:
    if value is None:
        return DEFAULT_TOLERANCE_POLICY
    return value if isinstance(value, TolerancePolicy) else None


def _extent(explicit: float | None, derived: float) -> float | None:
    if explicit is None:
        return derived
    try:
        made = float(explicit)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(made) or made < 0.0:
        return None
    return max(made, derived)


def _snap_unit_parameter(value: float, tolerance: float) -> float:
    if abs(value) <= tolerance:
        return 0.0
    if abs(value - 1.0) <= tolerance:
        return 1.0
    return min(max(float(value), 0.0), 1.0)


def _component(
    witnesses: Sequence[np.ndarray],
    *,
    first_parameter: Sequence[float] | None = None,
    second_parameter: Sequence[float] | None = None,
    first_range: tuple[float, float] | None = None,
    second_range: tuple[float, float] | None = None,
    direction: np.ndarray | None = None,
    residual: float = 0.0,
) -> IntersectionComponent:
    return IntersectionComponent(
        witnesses=tuple(_point_tuple(item, "intersection witness") for item in witnesses),
        quality=IntersectionQuality.VERIFIED_APPROXIMATE,
        first_parameter=None if first_parameter is None else tuple(first_parameter),
        second_parameter=None if second_parameter is None else tuple(second_parameter),
        first_parameter_range=(
            None if first_range is None else ParameterRange(*first_range)
        ),
        second_parameter_range=(
            None if second_range is None else ParameterRange(*second_range)
        ),
        direction=(
            None if direction is None else _point_tuple(direction, "intersection direction")
        ),
        max_residual=max(0.0, float(residual)),
    )


def _canonical_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.zeros(3, dtype=float)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    first_data = _unit(np.cross(normal, axis))
    assert first_data is not None
    first = _canonical_direction(first_data[0])
    second = np.cross(normal, first)
    return first, second


def qualified_line_cylinder(
    line_point: Sequence[float],
    line_direction: Sequence[float],
    cylinder: object,
    *,
    bounded: bool = True,
    policy: TolerancePolicy | None = None,
    local_extent: float | None = None,
) -> IntersectionResult:
    """Qualify an infinite line against a cylindrical support or patch.

    Line parameters are signed physical distance because the supplied
    direction is normalized internally.  Axial and angular patch trimming is
    applied when ``bounded`` is true.  An axial generatrix on the support is
    returned as an overlap curve instead of being collapsed to no roots.
    """

    point = _vector(line_point)
    direction_value = _vector(line_direction)
    tolerance_policy = _policy(policy)
    required = ("origin", "axis", "radial_direction", "radius", "height")
    if (
        point is None
        or direction_value is None
        or tolerance_policy is None
        or any(not hasattr(cylinder, name) for name in required)
    ):
        return _unclassified("invalid_line_or_cylinder")
    direction_data = _unit(direction_value)
    origin = _vector(getattr(cylinder, "origin"))
    axis_value = _vector(getattr(cylinder, "axis"))
    axis_data = None if axis_value is None else _unit(axis_value)
    try:
        radius = float(getattr(cylinder, "radius"))
        height = float(getattr(cylinder, "height"))
    except (TypeError, ValueError, OverflowError):
        return _unclassified("invalid_cylinder_dimensions")
    if (
        direction_data is None
        or origin is None
        or axis_data is None
        or not np.isfinite(radius)
        or radius <= 0.0
        or not np.isfinite(height)
        or height == 0.0
    ):
        return _unclassified("invalid_line_or_cylinder")
    direction = direction_data[0]
    axis = axis_data[0]
    derived_extent = max(radius, abs(height), _stable_norm(point - origin))
    extent = _extent(local_extent, derived_extent)
    if extent is None:
        return _unclassified("invalid_local_extent")
    length_tolerance = tolerance_policy.effective_length(extent)
    parameter_tolerance = tolerance_policy.effective_parameter(extent)

    offset = point - origin
    radial_direction = direction - float(direction @ axis) * axis
    radial_offset = offset - float(offset @ axis) * axis
    a = float(radial_direction @ radial_direction)
    b = 2.0 * float(radial_offset @ radial_direction)
    c = float(radial_offset @ radial_offset) - radius * radius
    coefficient_tolerance = max(
        np.finfo(float).eps * 64.0,
        tolerance_policy.angular * tolerance_policy.angular,
    )

    def patch_parameters(witness: np.ndarray) -> tuple[float, float] | None:
        try:
            u, v = getattr(cylinder, "local_uv")(witness)
        except (AttributeError, ValueError, GeometryError, np.linalg.LinAlgError):
            return None
        u, v = float(u), float(v)
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        if bounded and not (
            -parameter_tolerance <= u <= 1.0 + parameter_tolerance
            and -parameter_tolerance <= v <= 1.0 + parameter_tolerance
        ):
            return ()  # type: ignore[return-value]
        return (
            _snap_unit_parameter(u, parameter_tolerance) if bounded else u,
            _snap_unit_parameter(v, parameter_tolerance) if bounded else v,
        )

    if a <= coefficient_tolerance:
        radial_residual = abs(_stable_norm(radial_offset) - radius)
        if radial_residual > length_tolerance:
            return _disjoint("axis_parallel_line_off_cylinder")
        if not bounded:
            component = _component(
                (point,), direction=_canonical_direction(direction), residual=radial_residual
            )
            return IntersectionResult(IntersectionKind.OVERLAP_CURVE, (component,))
        axial_speed = float(direction @ axis)
        if abs(axial_speed) <= tolerance_policy.angular:
            return _unclassified("ill_conditioned_axial_generatrix")
        axial_at_point = float(offset @ axis)
        roots = sorted(((-axial_at_point) / axial_speed, (height - axial_at_point) / axial_speed))
        witnesses = tuple(point + root * direction for root in roots)
        parameters = tuple(patch_parameters(item) for item in witnesses)
        if any(item is None for item in parameters):
            return _unclassified("cylinder_parameterization_failed")
        if any(item == () for item in parameters):
            return _disjoint("generatrix_outside_angular_patch")
        component = _component(
            witnesses,
            first_range=(roots[0], roots[1]),
            direction=_canonical_direction(direction),
            residual=radial_residual,
        )
        return IntersectionResult(IntersectionKind.OVERLAP_CURVE, (component,))

    discriminant = b * b - 4.0 * a * c
    discriminant_tolerance = max(
        length_tolerance * max(abs(a), abs(b), abs(c), 1.0),
        64.0 * np.finfo(float).eps * (b * b + abs(4.0 * a * c)),
    )
    if discriminant < -discriminant_tolerance:
        return _disjoint("line_misses_cylinder")
    tangent = abs(discriminant) <= discriminant_tolerance
    if tangent:
        roots = (-b / (2.0 * a),)
    else:
        square_root = float(np.sqrt(max(discriminant, 0.0)))
        # Stable quadratic formula avoids losing the smaller root.
        q = -0.5 * (b + np.copysign(square_root, b))
        roots = tuple(sorted((q / a, c / q))) if q != 0.0 else (-b / (2.0 * a),)

    components = []
    for root in roots:
        witness = point + float(root) * direction
        parameters = patch_parameters(witness)
        if parameters is None:
            return _unclassified("cylinder_parameterization_failed")
        if parameters == ():
            continue
        radial = witness - origin - float((witness - origin) @ axis) * axis
        residual = abs(_stable_norm(radial) - radius)
        if residual > length_tolerance:
            return _unclassified("unverified_line_cylinder_root")
        components.append(
            _component(
                (witness,),
                first_parameter=(float(root),),
                second_parameter=parameters,
                residual=residual,
            )
        )
    if not components:
        return _disjoint("line_hits_support_outside_cylinder_patch")
    kind = IntersectionKind.TANGENT if tangent and len(components) == 1 else IntersectionKind.CROSS
    return IntersectionResult(kind, tuple(components))


def _plane_data(value: object, normal_value: object | None = None) -> _PlaneData | None:
    patch_vectors: tuple[np.ndarray, np.ndarray] | None = None
    if normal_value is None and hasattr(value, "origin") and hasattr(value, "normal"):
        origin = _vector(getattr(value, "origin"))
        normal = _vector(getattr(value, "normal"))
        if hasattr(value, "u_vector") and hasattr(value, "v_vector"):
            first = _vector(getattr(value, "u_vector"))
            second = _vector(getattr(value, "v_vector"))
            if first is not None and second is not None:
                patch_vectors = (first, second)
    else:
        origin = _vector(value)
        normal = _vector(normal_value)
    if origin is None or normal is None:
        return None
    normal_data = _unit(normal)
    if normal_data is None:
        return None
    normal = normal_data[0]
    extent = 0.0
    if patch_vectors is not None:
        first, second = patch_vectors
        cross = _unit(np.cross(first, second))
        if _unit(first) is None or _unit(second) is None or cross is None:
            return None
        # A Plane's own parameterization is authoritative for returned (u, v).
        u_vector, v_vector = first, second
        try:
            extent = feature_extent(
                np.vstack((np.zeros(3), first, second, first + second))
            )
        except GeometryError:
            return None
    else:
        u_vector, v_vector = _canonical_plane_basis(normal)
    return _PlaneData(origin, normal, u_vector, v_vector, extent)


def _plane_parameters(plane: _PlaneData, point: np.ndarray) -> ParameterValue:
    matrix = np.column_stack((plane.u_vector, plane.v_vector))
    values, *_ = np.linalg.lstsq(matrix, point - plane.origin, rcond=None)
    return float(values[0]), float(values[1])


def qualified_line_line(
    first_point: Sequence[float],
    first_direction: Sequence[float],
    second_point: Sequence[float],
    second_direction: Sequence[float],
    *,
    policy: TolerancePolicy | None = None,
    characteristic_length: float | None = None,
) -> IntersectionResult:
    """Classify two infinite 3D lines.

    Returned line parameters are signed physical distances because input
    directions are normalized internally.  Direction magnitude therefore
    cannot change the classification or reported witness.
    """

    tolerance_policy = _policy(policy)
    p, q = _vector(first_point), _vector(second_point)
    first, second = _vector(first_direction), _vector(second_direction)
    if tolerance_policy is None or p is None or q is None or first is None or second is None:
        return _unclassified("invalid_line_input")
    first_data, second_data = _unit(first), _unit(second)
    if first_data is None or second_data is None:
        return _unclassified("degenerate_line_direction")
    a, b = first_data[0], second_data[0]
    local_extent = _extent(characteristic_length, 0.0)
    if local_extent is None:
        return _unclassified("invalid_characteristic_length")
    length_tolerance = tolerance_policy.effective_length(local_extent)
    delta = q - p
    cross = np.cross(a, b)
    sine = _stable_norm(cross)

    if sine <= tolerance_policy.angular:
        distance = _stable_norm(np.cross(delta, a))
        if distance > length_tolerance:
            return _disjoint("parallel_lines")
        second_on_first_anchor = q + float((p - q) @ b) * b
        witness = p + 0.5 * (second_on_first_anchor - p)
        first_parameter = float((witness - p) @ a)
        second_parameter = float((witness - q) @ b)
        component = _component(
            (witness,),
            first_parameter=(first_parameter,),
            second_parameter=(second_parameter,),
            direction=_canonical_direction(a),
            residual=0.5 * distance,
        )
        return IntersectionResult(IntersectionKind.COINCIDENT, (component,))

    if sine <= float(np.sqrt(np.finfo(float).eps)):
        return _unclassified("ill_conditioned_line_line")
    cosine = float(a @ b)
    denominator = sine * sine
    first_delta = float(delta @ a)
    second_delta = float(delta @ b)
    first_parameter = (first_delta - cosine * second_delta) / denominator
    second_parameter = (cosine * first_delta - second_delta) / denominator
    first_witness = p + first_parameter * a
    second_witness = q + second_parameter * b
    distance = _stable_norm(first_witness - second_witness)
    if not np.isfinite(distance):
        return _unclassified("non_finite_line_line_residual")
    if distance > length_tolerance:
        return _disjoint("skew_lines")
    witness = first_witness + 0.5 * (second_witness - first_witness)
    component = _component(
        (witness,),
        first_parameter=(first_parameter,),
        second_parameter=(second_parameter,),
        residual=0.5 * distance,
    )
    return IntersectionResult(IntersectionKind.CROSS, (component,))


def qualified_line_plane(
    line_point: Sequence[float],
    line_direction: Sequence[float],
    plane_or_origin: object,
    plane_normal: Sequence[float] | None = None,
    *,
    policy: TolerancePolicy | None = None,
    characteristic_length: float | None = None,
) -> IntersectionResult:
    """Classify an infinite line and plane.

    ``plane_or_origin`` may be a ``Plane``-like value with ``origin`` and
    ``normal`` attributes, or a raw origin followed by ``plane_normal``.
    Plane-like parameter vectors are preserved in the returned ``(u, v)``;
    raw planes use a deterministic orthonormal local basis.
    """

    tolerance_policy = _policy(policy)
    point, direction = _vector(line_point), _vector(line_direction)
    plane = _plane_data(plane_or_origin, plane_normal)
    if tolerance_policy is None or point is None or direction is None or plane is None:
        return _unclassified("invalid_line_plane_input")
    direction_data = _unit(direction)
    if direction_data is None:
        return _unclassified("degenerate_line_direction")
    unit_direction = direction_data[0]
    local_extent = _extent(characteristic_length, plane.extent)
    if local_extent is None:
        return _unclassified("invalid_characteristic_length")
    length_tolerance = tolerance_policy.effective_length(local_extent)
    denominator = float(plane.normal @ unit_direction)
    signed_distance = float(plane.normal @ (point - plane.origin))

    if abs(denominator) <= tolerance_policy.angular:
        if abs(signed_distance) > length_tolerance:
            return _disjoint("line_parallel_to_plane")
        projected = point - signed_distance * plane.normal
        witness = point + 0.5 * (projected - point)
        component = _component(
            (witness,),
            first_parameter=(float((witness - point) @ unit_direction),),
            second_parameter=_plane_parameters(plane, projected),
            direction=_canonical_direction(unit_direction),
            residual=0.5 * abs(signed_distance),
        )
        return IntersectionResult(IntersectionKind.OVERLAP_CURVE, (component,))

    if abs(denominator) <= float(np.sqrt(np.finfo(float).eps)):
        return _unclassified("ill_conditioned_line_plane")
    parameter = -signed_distance / denominator
    witness = point + parameter * unit_direction
    residual = abs(signed_distance + parameter * denominator)
    if not np.isfinite(residual) or residual > length_tolerance:
        return _unclassified("unverified_line_plane_residual")
    component = _component(
        (witness,),
        first_parameter=(parameter,),
        second_parameter=_plane_parameters(plane, witness),
        residual=residual,
    )
    return IntersectionResult(IntersectionKind.CROSS, (component,))


def qualified_plane_plane(
    first_plane_or_origin: object,
    first_normal_or_second_plane: object,
    second_origin: object | None = None,
    second_normal: object | None = None,
    *,
    policy: TolerancePolicy | None = None,
    characteristic_length: float | None = None,
) -> IntersectionResult:
    """Classify two infinite planes.

    Call this either as ``qualified_plane_plane(first_plane, second_plane)``
    or ``qualified_plane_plane(first_origin, first_normal, second_origin,
    second_normal)``.
    """

    tolerance_policy = _policy(policy)
    if second_origin is None and second_normal is None:
        first = _plane_data(first_plane_or_origin)
        second = _plane_data(first_normal_or_second_plane)
    else:
        first = _plane_data(first_plane_or_origin, first_normal_or_second_plane)
        second = _plane_data(second_origin, second_normal)
    if tolerance_policy is None or first is None or second is None:
        return _unclassified("invalid_plane_plane_input")
    local_extent = _extent(characteristic_length, max(first.extent, second.extent))
    if local_extent is None:
        return _unclassified("invalid_characteristic_length")
    length_tolerance = tolerance_policy.effective_length(local_extent)
    delta = second.origin - first.origin
    cross = np.cross(first.normal, second.normal)
    sine = _stable_norm(cross)

    if sine <= tolerance_policy.angular:
        signed = float(second.normal @ delta)
        distance = abs(signed)
        if distance > length_tolerance:
            return _disjoint("parallel_planes")
        second_projection = first.origin + signed * second.normal
        witness = first.origin + 0.5 * (second_projection - first.origin)
        component = _component(
            (witness,),
            first_parameter=_plane_parameters(first, first.origin),
            second_parameter=_plane_parameters(second, second_projection),
            residual=0.5 * distance,
        )
        return IntersectionResult(IntersectionKind.COINCIDENT, (component,))

    if sine <= float(np.sqrt(np.finfo(float).eps)):
        return _unclassified("ill_conditioned_plane_plane")
    direction = cross / sine
    halfway = 0.5 * delta
    matrix = np.vstack((first.normal, second.normal, direction))
    right_hand_side = np.asarray(
        (0.0, float(second.normal @ delta), float(direction @ halfway)),
        dtype=float,
    )
    try:
        local_anchor = np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError:
        return _unclassified("singular_plane_plane_system")
    if not np.all(np.isfinite(local_anchor)):
        return _unclassified("non_finite_plane_plane_solution")
    first_residual = abs(float(first.normal @ local_anchor))
    second_residual = abs(float(second.normal @ (local_anchor - delta)))
    residual = max(first_residual, second_residual)
    if residual > length_tolerance:
        return _unclassified("unverified_plane_plane_residual")
    witness = first.origin + local_anchor
    component = _component(
        (witness,),
        first_parameter=_plane_parameters(first, witness),
        second_parameter=_plane_parameters(second, witness),
        direction=_canonical_direction(direction),
        residual=residual,
    )
    return IntersectionResult(IntersectionKind.CROSS, (component,))


def _closest_endpoint_pair(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    first_vector, second_vector = first_end - first_start, second_end - second_start
    first_squared = float(first_vector @ first_vector)
    second_squared = float(second_vector @ second_vector)
    candidates: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
    for first_parameter, point in ((0.0, first_start), (1.0, first_end)):
        second_parameter = min(
            max(float((point - second_start) @ second_vector) / second_squared, 0.0),
            1.0,
        )
        other = second_start + second_parameter * second_vector
        candidates.append(
            (_stable_norm(point - other), first_parameter, second_parameter, point, other)
        )
    for second_parameter, point in ((0.0, second_start), (1.0, second_end)):
        first_parameter = min(
            max(float((point - first_start) @ first_vector) / first_squared, 0.0),
            1.0,
        )
        other = first_start + first_parameter * first_vector
        candidates.append(
            (_stable_norm(other - point), first_parameter, second_parameter, other, point)
        )
    distance, first_parameter, second_parameter, first_point, second_point = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return first_parameter, second_parameter, first_point, second_point, distance


def qualified_segment_segment(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
    *,
    policy: TolerancePolicy | None = None,
    characteristic_length: float | None = None,
) -> IntersectionResult:
    """Classify two bounded straight 3D segments.

    Segment parameters are normalized to ``[0, 1]``.  Collinear overlap
    witnesses are ordered by increasing first-parent parameter; the second
    range retains its own orientation.  Degenerate point-segments are left
    unclassified for a future explicit point/segment predicate.
    """

    tolerance_policy = _policy(policy)
    points = tuple(
        _vector(item) for item in (first_start, first_end, second_start, second_end)
    )
    if tolerance_policy is None or any(item is None for item in points):
        return _unclassified("invalid_segment_input")
    first_a, first_b, second_a, second_b = points
    assert first_a is not None and first_b is not None
    assert second_a is not None and second_b is not None
    try:
        derived_extent = feature_extent(np.vstack(points))
    except GeometryError:
        return _unclassified("invalid_segment_extent")
    local_extent = _extent(characteristic_length, derived_extent)
    if local_extent is None:
        return _unclassified("invalid_characteristic_length")
    length_tolerance = tolerance_policy.effective_length(local_extent)
    first_vector, second_vector = first_b - first_a, second_b - second_a
    first_data, second_data = _unit(first_vector), _unit(second_vector)
    if first_data is None or second_data is None:
        return _unclassified("degenerate_segment")
    first_direction, first_length = first_data
    second_direction, second_length = second_data
    if first_length <= length_tolerance or second_length <= length_tolerance:
        return _unclassified("segment_below_length_tolerance")
    first_parameter_tolerance = tolerance_policy.effective_parameter(
        first_length, local_extent
    )
    second_parameter_tolerance = tolerance_policy.effective_parameter(
        second_length, local_extent
    )
    delta = second_a - first_a
    cross = np.cross(first_direction, second_direction)
    sine = _stable_norm(cross)

    if sine <= tolerance_policy.angular:
        endpoint_distances = (
            _stable_norm(np.cross(second_a - first_a, first_direction)),
            _stable_norm(np.cross(second_b - first_a, first_direction)),
        )
        if max(endpoint_distances) > length_tolerance:
            return _disjoint("parallel_segments")
        first_squared = first_length * first_length
        second_on_first = (
            float((second_a - first_a) @ first_vector) / first_squared,
            float((second_b - first_a) @ first_vector) / first_squared,
        )
        lower = max(0.0, min(second_on_first))
        upper = min(1.0, max(second_on_first))
        overlap_length = max(0.0, upper - lower) * first_length
        if upper < lower or overlap_length <= length_tolerance:
            (
                first_parameter,
                second_parameter,
                first_point,
                second_point,
                distance,
            ) = _closest_endpoint_pair(first_a, first_b, second_a, second_b)
            if distance > length_tolerance:
                return _disjoint("collinear_segments_separated")
            first_parameter = _snap_unit_parameter(
                first_parameter, first_parameter_tolerance
            )
            second_parameter = _snap_unit_parameter(
                second_parameter, second_parameter_tolerance
            )
            witness = first_point + 0.5 * (second_point - first_point)
            component = _component(
                (witness,),
                first_parameter=(first_parameter,),
                second_parameter=(second_parameter,),
                residual=0.5 * distance,
            )
            return IntersectionResult(IntersectionKind.TOUCH_POINT, (component,))

        lower = _snap_unit_parameter(lower, first_parameter_tolerance)
        upper = _snap_unit_parameter(upper, first_parameter_tolerance)
        first_points = (
            first_a + lower * first_vector,
            first_a + upper * first_vector,
        )
        second_parameters = tuple(
            _snap_unit_parameter(
                float((point - second_a) @ second_vector)
                / (second_length * second_length),
                second_parameter_tolerance,
            )
            for point in first_points
        )
        second_points = tuple(
            second_a + parameter * second_vector for parameter in second_parameters
        )
        residuals = tuple(
            _stable_norm(first_point - second_point)
            for first_point, second_point in zip(first_points, second_points)
        )
        if max(residuals) > length_tolerance:
            return _unclassified("unverified_collinear_overlap")
        witnesses = tuple(
            first_point + 0.5 * (second_point - first_point)
            for first_point, second_point in zip(first_points, second_points)
        )
        component = _component(
            witnesses,
            first_range=(lower, upper),
            second_range=second_parameters,
            direction=_canonical_direction(first_direction),
            residual=0.5 * max(residuals),
        )
        first_full = lower <= first_parameter_tolerance and upper >= 1.0 - first_parameter_tolerance
        second_full = (
            min(second_parameters) <= second_parameter_tolerance
            and max(second_parameters) >= 1.0 - second_parameter_tolerance
        )
        kind = (
            IntersectionKind.COINCIDENT
            if first_full and second_full
            else IntersectionKind.OVERLAP_CURVE
        )
        return IntersectionResult(kind, (component,))

    if sine <= float(np.sqrt(np.finfo(float).eps)):
        return _unclassified("ill_conditioned_segment_segment")
    cosine = float(first_direction @ second_direction)
    denominator = sine * sine
    first_delta = float(delta @ first_direction)
    second_delta = float(delta @ second_direction)
    first_distance_parameter = (first_delta - cosine * second_delta) / denominator
    second_distance_parameter = (cosine * first_delta - second_delta) / denominator
    first_parameter = first_distance_parameter / first_length
    second_parameter = second_distance_parameter / second_length
    if (
        first_parameter < -first_parameter_tolerance
        or first_parameter > 1.0 + first_parameter_tolerance
        or second_parameter < -second_parameter_tolerance
        or second_parameter > 1.0 + second_parameter_tolerance
    ):
        return _disjoint("line_crossing_outside_segments")
    first_parameter = _snap_unit_parameter(
        first_parameter, first_parameter_tolerance
    )
    second_parameter = _snap_unit_parameter(
        second_parameter, second_parameter_tolerance
    )
    first_witness = first_a + first_parameter * first_vector
    second_witness = second_a + second_parameter * second_vector
    distance = _stable_norm(first_witness - second_witness)
    if not np.isfinite(distance):
        return _unclassified("non_finite_segment_residual")
    if distance > length_tolerance:
        return _disjoint("skew_segments")
    witness = first_witness + 0.5 * (second_witness - first_witness)
    component = _component(
        (witness,),
        first_parameter=(first_parameter,),
        second_parameter=(second_parameter,),
        residual=0.5 * distance,
    )
    endpoint = (
        first_parameter in (0.0, 1.0) or second_parameter in (0.0, 1.0)
    )
    kind = IntersectionKind.TOUCH_POINT if endpoint else IntersectionKind.CROSS
    return IntersectionResult(kind, (component,))
