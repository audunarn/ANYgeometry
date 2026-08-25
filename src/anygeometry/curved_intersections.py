"""Deterministic qualified intersections for built-in curves and surfaces.

The public query/plan/apply workflow lives in :mod:`anygeometry.intersections`.
This module is the shared, model-bound narrow phase used by that workflow and
by audit.  It deliberately has no optional geometry dependency: built-in
curves are bounded analytically and general roots are isolated by deterministic
parameter subdivision followed by residual-qualified Newton refinement.

The implementation is fail closed.  A subdivision budget or singular solve
never becomes an empty intersection; it becomes ``UNCLASSIFIED``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, ceil, comb, floor, pi
import heapq
from typing import Iterable, Sequence

import numpy as np

from .curves import Arc, Spline, Straight
from .errors import GeometryError
from .identity import EntityHandle
from .predicates import (
    DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
    CertifiedCurveTrace,
    IntersectionCertificate,
    IntersectionComponent,
    IntersectionDimension,
    IntersectionKind,
    IntersectionQualificationPolicy,
    IntersectionQuality,
    IntersectionResult,
    ParameterLoop,
    ParameterRange,
    ParameterRegion,
)
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface, _surface_derivatives_many
from .tolerance import feature_extent


@dataclass(frozen=True, slots=True)
class _Bounds:
    lower: np.ndarray
    upper: np.ndarray

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.upper - self.lower))

    def distance(self, other: "_Bounds") -> float:
        gap = np.maximum(np.maximum(self.lower - other.upper, other.lower - self.upper), 0.0)
        return float(np.linalg.norm(gap))


@dataclass(frozen=True, slots=True)
class _CurveBox:
    lower: float
    upper: float
    depth: int
    bounds: _Bounds


def _certificate(
    algorithm: str,
    tolerance: float,
    *,
    residual: float = 0.0,
    enclosure: float = 0.0,
    separation: float | None = None,
    boxes: int = 0,
    subdivisions: int = 0,
    trace_segments: int = 0,
) -> IntersectionCertificate:
    return IntersectionCertificate(
        algorithm,
        tolerance,
        max_residual=residual,
        max_enclosure_width=enclosure,
        separation_bound=separation,
        boxes_examined=boxes,
        subdivisions=subdivisions,
        trace_segments=trace_segments,
        complete=True,
    )


def _result(
    first: EntityHandle,
    second: EntityHandle,
    kind: IntersectionKind,
    components: Sequence[IntersectionComponent] = (),
    *,
    tolerance: float,
    algorithm: str,
    diagnostics: Sequence[str] = (),
    dimension: IntersectionDimension | None = None,
    boxes: int = 0,
    subdivisions: int = 0,
    separation: float | None = None,
) -> IntersectionResult:
    normalized_components: list[IntersectionComponent] = []
    for component in components:
        if (
            not component.curve_traces
            and len(component.witnesses) >= 2
            and len(component.first_parameter_path) == len(component.witnesses)
            and len(component.second_parameter_path) == len(component.witnesses)
            and component.certificate is not None
            and component.certificate.complete
        ):
            closed = (
                len(component.witnesses) >= 3
                and float(
                    np.linalg.norm(
                        np.asarray(component.witnesses[0])
                        - np.asarray(component.witnesses[-1])
                    )
                )
                <= tolerance
            )
            component = replace(
                component,
                curve_traces=(
                    CertifiedCurveTrace(
                        component.witnesses,
                        component.first_parameter_path,
                        component.second_parameter_path,
                        component.certificate,
                        closed=closed,
                    ),
                ),
            )
        normalized_components.append(component)
    components = tuple(normalized_components)
    residual = max((item.max_residual for item in components), default=0.0)
    enclosure = max(
        (
            item.certificate.max_enclosure_width
            for item in components
            if item.certificate is not None
        ),
        default=0.0,
    )
    trace_segments = sum(
        item.certificate.trace_segments
        for item in components
        if item.certificate is not None
    )
    certificate = None
    if kind is IntersectionKind.UNCLASSIFIED:
        certificate = IntersectionCertificate(
            algorithm,
            tolerance,
            max_residual=residual,
            max_enclosure_width=enclosure,
            separation_bound=separation,
            boxes_examined=boxes,
            subdivisions=subdivisions,
            trace_segments=trace_segments,
            complete=False,
        )
    elif kind not in (
        IntersectionKind.UNSUPPORTED,
        IntersectionKind.CAPABILITY_MISSING,
    ):
        certificate = _certificate(
            algorithm,
            tolerance,
            residual=residual,
            enclosure=enclosure,
            separation=separation,
            boxes=boxes,
            subdivisions=subdivisions,
            trace_segments=trace_segments,
        )
    return IntersectionResult(
        kind,
        tuple(components),
        tuple(str(item) for item in diagnostics),
        dimension=dimension,
        first_parent=first,
        second_parent=second,
        tolerance_used=tolerance,
        certificate=certificate,
    )


def _edge_points(model, edge_id: int, parameters: Sequence[float]) -> np.ndarray:
    return np.asarray(model.evaluate_edge_many(edge_id, np.asarray(parameters, dtype=float)), dtype=float)


def _bezier_subcurve(points: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if lower <= 0.0 and upper >= 1.0:
        return np.asarray(points, dtype=float)
    left, _right = _split_bezier(points, upper)
    if lower <= 0.0:
        return left
    relative = lower / upper if upper > 0.0 else 0.0
    _discard, selected = _split_bezier(left, relative)
    return selected


def _split_bezier(points: np.ndarray, parameter: float) -> tuple[np.ndarray, np.ndarray]:
    levels = [np.asarray(points, dtype=float)]
    while len(levels[-1]) > 1:
        previous = levels[-1]
        levels.append((1.0 - parameter) * previous[:-1] + parameter * previous[1:])
    return (
        np.asarray([level[0] for level in levels]),
        np.asarray([level[-1] for level in reversed(levels)]),
    )


def _angle_candidates(start: float, end: float, phase: float) -> Iterable[float]:
    lower, upper = sorted((start, end))
    first = int(ceil((lower - phase) / pi))
    last = int(floor((upper - phase) / pi))
    for index in range(first, last + 1):
        yield phase + index * pi


def _curve_bounds(model, edge_id: int, lower: float, upper: float) -> _Bounds:
    edge = model.edges[edge_id]
    if isinstance(edge.curve, Straight):
        points = _edge_points(model, edge_id, (lower, upper))
    elif isinstance(edge.curve, Spline):
        points = _bezier_subcurve(model._spline_points(edge), lower, upper)  # noqa: SLF001
    elif isinstance(edge.curve, Arc):
        frame = model.arc_frame(edge_id)
        start = lower * frame.sweep
        end = upper * frame.sweep
        parameters = [lower, upper]
        for coordinate in range(3):
            phase = atan2(float(frame.e2[coordinate]), float(frame.e1[coordinate]))
            for angle in _angle_candidates(start, end, phase):
                if frame.sweep != 0.0:
                    parameters.append(float(np.clip(angle / frame.sweep, lower, upper)))
        points = _edge_points(model, edge_id, parameters)
    else:  # pragma: no cover - closed built-in curve algebra
        raise GeometryError(f"unsupported curve {type(edge.curve).__name__}")
    low = np.nextafter(points.min(axis=0), -np.inf)
    high = np.nextafter(points.max(axis=0), np.inf)
    return _Bounds(low, high)


def _curve_derivative(model, edge_id: int, parameter: float) -> np.ndarray:
    edge = model.edges[edge_id]
    if isinstance(edge.curve, Straight):
        return model.vertex_position(edge.end) - model.vertex_position(edge.start)
    if isinstance(edge.curve, Arc):
        frame = model.arc_frame(edge_id)
        angle = parameter * frame.sweep
        return frame.radius * frame.sweep * (
            -np.sin(angle) * frame.e1 + np.cos(angle) * frame.e2
        )
    assert isinstance(edge.curve, Spline)
    points = model._spline_points(edge)  # noqa: SLF001
    derivative = (len(points) - 1) * (points[1:] - points[:-1])
    if len(derivative) == 1:
        return derivative[0]
    work = np.asarray(derivative, dtype=float)
    for _ in range(1, len(derivative)):
        work = (1.0 - parameter) * work[:-1] + parameter * work[1:]
    return work[0]


def _newton_curve_pair(
    model,
    first_edge: int,
    second_edge: int,
    first_parameter: float,
    second_parameter: float,
    policy: IntersectionQualificationPolicy,
) -> tuple[float, float, np.ndarray, float]:
    first_t = float(np.clip(first_parameter, 0.0, 1.0))
    second_t = float(np.clip(second_parameter, 0.0, 1.0))
    for _ in range(policy.max_newton_iterations):
        first_point = _edge_points(model, first_edge, (first_t,))[0]
        second_point = _edge_points(model, second_edge, (second_t,))[0]
        difference = first_point - second_point
        jacobian = np.column_stack(
            (
                _curve_derivative(model, first_edge, first_t),
                -_curve_derivative(model, second_edge, second_t),
            )
        )
        if not np.all(np.isfinite(jacobian)) or np.linalg.matrix_rank(jacobian) < 2:
            break
        delta, *_ = np.linalg.lstsq(jacobian, -difference, rcond=None)
        next_first = float(np.clip(first_t + delta[0], 0.0, 1.0))
        next_second = float(np.clip(second_t + delta[1], 0.0, 1.0))
        if max(abs(next_first - first_t), abs(next_second - second_t)) <= model.tolerance.parameter:
            first_t, second_t = next_first, next_second
            break
        first_t, second_t = next_first, next_second
    first_point = _edge_points(model, first_edge, (first_t,))[0]
    second_point = _edge_points(model, second_edge, (second_t,))[0]
    return first_t, second_t, 0.5 * (first_point + second_point), float(
        np.linalg.norm(first_point - second_point)
    )


def _same_bezier_overlap(model, first_edge: int, second_edge: int, tolerance: float):
    first = model.edges[first_edge]
    second = model.edges[second_edge]
    if not isinstance(first.curve, Spline) or not isinstance(second.curve, Spline):
        return None
    first_points = model._spline_points(first)  # noqa: SLF001
    second_points = model._spline_points(second)  # noqa: SLF001
    if first_points.shape != second_points.shape:
        return None
    direct = float(np.max(np.linalg.norm(first_points - second_points, axis=1)))
    reverse = float(np.max(np.linalg.norm(first_points - second_points[::-1], axis=1)))
    if min(direct, reverse) > tolerance:
        return None
    reversed_second = reverse < direct
    witnesses = _edge_points(model, first_edge, (0.0, 1.0))
    certificate = _certificate(
        "bezier_control_polygon_identity",
        tolerance,
        residual=min(direct, reverse),
    )
    return IntersectionComponent(
        tuple(tuple(float(value) for value in point) for point in witnesses),
        IntersectionQuality.VERIFIED_APPROXIMATE,
        first_parameter_range=ParameterRange(0.0, 1.0),
        second_parameter_range=ParameterRange(1.0, 0.0) if reversed_second else ParameterRange(0.0, 1.0),
        first_parameter_path=((0.0,), (1.0,)),
        second_parameter_path=((1.0,), (0.0,)) if reversed_second else ((0.0,), (1.0,)),
        max_residual=min(direct, reverse),
        certificate=certificate,
    )


def _bezier_power_coefficients(points: np.ndarray) -> np.ndarray:
    degree = len(points) - 1
    coefficients = np.zeros((degree + 1, 3), dtype=float)
    for index, point in enumerate(points):
        scale = float(comb(degree, index))
        for power in range(degree - index + 1):
            coefficients[index + power] += (
                scale
                * float(comb(degree - index, power))
                * ((-1.0) ** power)
                * point
            )
    return coefficients


def _line_spline_intersection(
    model,
    first: EntityHandle,
    second: EntityHandle,
    tolerance: float,
    parameter_tolerance: float,
) -> IntersectionResult | None:
    first_edge = model.edges[first.id]
    second_edge = model.edges[second.id]
    if isinstance(first_edge.curve, Straight) and isinstance(second_edge.curve, Spline):
        line_handle, spline_handle, line_first = first, second, True
    elif isinstance(first_edge.curve, Spline) and isinstance(second_edge.curve, Straight):
        line_handle, spline_handle, line_first = second, first, False
    else:
        return None
    line = model.edges[line_handle.id]
    origin = model.vertex_position(line.start)
    direction = model.vertex_position(line.end) - origin
    length_squared = float(direction @ direction)
    if length_squared <= tolerance * tolerance:
        return None
    control = model._spline_points(model.edges[spline_handle.id])  # noqa: SLF001
    coefficients = _bezier_power_coefficients(control)
    cross_coefficients = np.cross(direction[None, :], coefficients)
    coordinate = int(
        np.argmax(np.linalg.norm(cross_coefficients, axis=0))
    )
    polynomial = cross_coefficients[:, coordinate]
    coefficient_scale = max(
        float(np.max(np.linalg.norm(cross_coefficients, axis=1))),
        tolerance,
    )
    while len(polynomial) > 1 and abs(float(polynomial[-1])) <= np.finfo(float).eps * coefficient_scale:
        polynomial = polynomial[:-1]
    if len(polynomial) <= 1:
        return None
    roots = np.roots(polynomial[::-1])
    imaginary_tolerance = max(np.sqrt(parameter_tolerance), 32.0 * np.finfo(float).eps)
    candidates = sorted(
        float(np.clip(root.real, 0.0, 1.0))
        for root in roots
        if abs(float(root.imag)) <= imaginary_tolerance
        and -parameter_tolerance <= float(root.real) <= 1.0 + parameter_tolerance
    )
    accepted: list[tuple[float, float, np.ndarray, float]] = []
    for spline_parameter in candidates:
        point = _edge_points(model, spline_handle.id, (spline_parameter,))[0]
        line_parameter = float((point - origin) @ direction / length_squared)
        projected = origin + line_parameter * direction
        residual = float(np.linalg.norm(point - projected))
        if (
            residual > tolerance
            or line_parameter < -parameter_tolerance
            or line_parameter > 1.0 + parameter_tolerance
        ):
            continue
        made = (
            line_parameter,
            spline_parameter,
            0.5 * (point + projected),
            residual,
        )
        if accepted and abs(spline_parameter - accepted[-1][1]) <= imaginary_tolerance:
            if residual < accepted[-1][3]:
                accepted[-1] = made
        else:
            accepted.append(made)
    if not accepted:
        return _result(
            first,
            second,
            IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="line_bezier_polynomial",
            diagnostics=("certified_line_spline_disjoint",),
            boxes=len(roots),
        )
    components = []
    kinds = []
    for line_parameter, spline_parameter, point, residual in accepted:
        spline_tangent = _curve_derivative(
            model, spline_handle.id, spline_parameter
        )
        tangent = float(np.linalg.norm(np.cross(direction, spline_tangent))) <= (
            model.tolerance.angular
            * float(np.linalg.norm(direction))
            * float(np.linalg.norm(spline_tangent))
        )
        endpoint = min(
            line_parameter,
            spline_parameter,
            1.0 - line_parameter,
            1.0 - spline_parameter,
        ) <= parameter_tolerance
        kind = (
            IntersectionKind.TANGENT
            if tangent and not endpoint
            else IntersectionKind.TOUCH_POINT
            if tangent or endpoint
            else IntersectionKind.CROSS
        )
        kinds.append(kind)
        first_parameter = line_parameter if line_first else spline_parameter
        second_parameter = spline_parameter if line_first else line_parameter
        certificate = _certificate(
            "line_bezier_polynomial",
            tolerance,
            residual=residual,
            boxes=len(roots),
        )
        components.append(
            IntersectionComponent(
                (tuple(float(value) for value in point),),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter=(first_parameter,),
                second_parameter=(second_parameter,),
                first_parameter_path=((first_parameter,),),
                second_parameter_path=((second_parameter,),),
                max_residual=residual,
                certificate=certificate,
                first_subparent=first,
                second_subparent=second,
            )
        )
    kind = next(
        item
        for item in (
            IntersectionKind.CROSS,
            IntersectionKind.TANGENT,
            IntersectionKind.TOUCH_POINT,
        )
        if item in kinds
    )
    return _result(
        first,
        second,
        kind,
        components,
        tolerance=tolerance,
        algorithm="line_bezier_polynomial",
        dimension=IntersectionDimension.POINT,
        boxes=len(roots),
    )


def qualified_curve_curve(
    model,
    first: EntityHandle,
    second: EntityHandle,
    qualification: IntersectionQualificationPolicy | None = None,
) -> IntersectionResult:
    """Qualify one bounded built-in curve pair in deterministic parameter order."""

    policy = qualification or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    first_bounds = _curve_bounds(model, first.id, 0.0, 1.0)
    second_bounds = _curve_bounds(model, second.id, 0.0, 1.0)
    extent = feature_extent(np.vstack((first_bounds.lower, first_bounds.upper, second_bounds.lower, second_bounds.upper)))
    tolerance = model.tolerance.effective_length(extent)
    parameter_tolerance = max(
        model.tolerance.effective_parameter(model.edge_length(first.id), extent),
        model.tolerance.effective_parameter(model.edge_length(second.id), extent),
    )

    same = _same_bezier_overlap(model, first.id, second.id, tolerance)
    if same is not None:
        return _result(
            first,
            second,
            IntersectionKind.COINCIDENT,
            (same,),
            tolerance=tolerance,
            algorithm="bezier_control_polygon_identity",
            dimension=IntersectionDimension.CURVE,
        )
    line_spline = _line_spline_intersection(
        model, first, second, tolerance, parameter_tolerance
    )
    if line_spline is not None:
        return line_spline

    queue: list[tuple[float, float, int, _CurveBox, _CurveBox]] = [
        (
            0.0,
            0.0,
            0,
            _CurveBox(0.0, 1.0, 0, first_bounds),
            _CurveBox(0.0, 1.0, 0, second_bounds),
        )
    ]
    heapq.heapify(queue)
    serial = 1
    roots: list[tuple[float, float, np.ndarray, float, float]] = []
    boxes = 0
    subdivisions = 0
    unresolved = False
    minimum_separation = float("inf")
    while queue:
        _first_key, _second_key, _serial, first_box, second_box = heapq.heappop(queue)
        boxes += 1
        if boxes > policy.max_boxes_per_pair:
            unresolved = True
            break
        separation = first_box.bounds.distance(second_box.bounds)
        minimum_separation = min(minimum_separation, separation)
        if separation > tolerance:
            continue
        first_span = first_box.upper - first_box.lower
        second_span = second_box.upper - second_box.lower
        enclosure = max(first_box.bounds.diagonal, second_box.bounds.diagonal)
        terminal = (
            max(first_box.depth, second_box.depth) >= policy.max_subdivision_depth
            or (
                first_span <= parameter_tolerance
                and second_span <= parameter_tolerance
            )
            or enclosure <= 2.0 * tolerance
        )
        if terminal:
            first_t, second_t, witness, residual = _newton_curve_pair(
                model,
                first.id,
                second.id,
                0.5 * (first_box.lower + first_box.upper),
                0.5 * (second_box.lower + second_box.upper),
                policy,
            )
            if residual <= tolerance:
                roots.append((first_t, second_t, witness, residual, enclosure))
            elif enclosure > 8.0 * tolerance and max(first_box.depth, second_box.depth) >= policy.max_subdivision_depth:
                unresolved = True
            continue
        first_straight = isinstance(model.edges[first.id].curve, Straight)
        second_straight = isinstance(model.edges[second.id].curve, Straight)
        split_first = (
            False
            if first_straight and not second_straight
            else True
            if second_straight and not first_straight
            else first_box.bounds.diagonal * first_span
            >= second_box.bounds.diagonal * second_span
        )
        if split_first:
            middle = 0.5 * (first_box.lower + first_box.upper)
            children = (
                _CurveBox(first_box.lower, middle, first_box.depth + 1, _curve_bounds(model, first.id, first_box.lower, middle)),
                _CurveBox(middle, first_box.upper, first_box.depth + 1, _curve_bounds(model, first.id, middle, first_box.upper)),
            )
            for child in children:
                heapq.heappush(
                    queue,
                    (child.lower, second_box.lower, serial, child, second_box),
                )
                serial += 1
        else:
            middle = 0.5 * (second_box.lower + second_box.upper)
            children = (
                _CurveBox(second_box.lower, middle, second_box.depth + 1, _curve_bounds(model, second.id, second_box.lower, middle)),
                _CurveBox(middle, second_box.upper, second_box.depth + 1, _curve_bounds(model, second.id, middle, second_box.upper)),
            )
            for child in children:
                heapq.heappush(
                    queue,
                    (first_box.lower, child.lower, serial, first_box, child),
                )
                serial += 1
        subdivisions += 1

    if unresolved:
        return _result(
            first,
            second,
            IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="curve_curve_subdivision",
            diagnostics=("curve_pair_subdivision_not_complete",),
            boxes=boxes,
            subdivisions=subdivisions,
        )

    roots.sort(key=lambda item: (item[0], item[1], tuple(item[2])))
    unique: list[tuple[float, float, np.ndarray, float, float]] = []
    for root in roots:
        if unique and (
            abs(root[0] - unique[-1][0]) <= parameter_tolerance
            and abs(root[1] - unique[-1][1]) <= parameter_tolerance
        ):
            if root[3] < unique[-1][3]:
                unique[-1] = root
            continue
        unique.append(root)
    if len(unique) > policy.max_components:
        return _result(
            first,
            second,
            IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="curve_curve_subdivision",
            diagnostics=("curve_pair_component_limit_exceeded",),
            boxes=boxes,
            subdivisions=subdivisions,
        )
    if not unique:
        separation = minimum_separation if np.isfinite(minimum_separation) else tolerance
        return _result(
            first,
            second,
            IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="curve_curve_subdivision",
            diagnostics=("certified_curve_pair_disjoint",),
            boxes=boxes,
            subdivisions=subdivisions,
            separation=max(0.0, separation),
        )

    components: list[IntersectionComponent] = []
    kinds: list[IntersectionKind] = []
    for first_t, second_t, witness, residual, enclosure in unique:
        first_derivative = _curve_derivative(model, first.id, first_t)
        second_derivative = _curve_derivative(model, second.id, second_t)
        first_length = float(np.linalg.norm(first_derivative))
        second_length = float(np.linalg.norm(second_derivative))
        tangent = (
            first_length <= tolerance
            or second_length <= tolerance
            or float(np.linalg.norm(np.cross(first_derivative, second_derivative)))
            <= model.tolerance.angular * first_length * second_length
        )
        endpoint = min(first_t, second_t, 1.0 - first_t, 1.0 - second_t) <= parameter_tolerance
        kind = IntersectionKind.TANGENT if tangent and not endpoint else (
            IntersectionKind.TOUCH_POINT if endpoint else IntersectionKind.CROSS
        )
        kinds.append(kind)
        certificate = _certificate(
            "curve_curve_subdivision",
            tolerance,
            residual=residual,
            enclosure=enclosure,
            boxes=boxes,
            subdivisions=subdivisions,
        )
        components.append(
            IntersectionComponent(
                (tuple(float(value) for value in witness),),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter=(first_t,),
                second_parameter=(second_t,),
                first_parameter_path=((first_t,),),
                second_parameter_path=((second_t,),),
                max_residual=residual,
                certificate=certificate,
                first_subparent=first,
                second_subparent=second,
            )
        )
    precedence = (IntersectionKind.CROSS, IntersectionKind.TANGENT, IntersectionKind.TOUCH_POINT)
    kind = next(candidate for candidate in precedence if candidate in kinds)
    return _result(
        first,
        second,
        kind,
        components,
        tolerance=tolerance,
        algorithm="curve_curve_subdivision",
        dimension=IntersectionDimension.POINT,
        boxes=boxes,
        subdivisions=subdivisions,
    )


def _surface_point_derivatives(model, face_id: int, uv: Sequence[float]):
    face = model.faces[face_id]
    surface = face.surface
    if surface is None:
        raise GeometryError(f"face {face_id} has no authoritative support")
    u, v = float(uv[0]), float(uv[1])
    if isinstance(surface, CoonsSurface) and not surface.has_boundaries:
        return model._topology_face_point_and_derivatives(face_id, u, v)  # noqa: SLF001
    point = np.asarray(surface.evaluate(u, v), dtype=float)
    du, dv = _surface_derivatives_many(surface, np.asarray(((u, v),), dtype=float))
    return point, du[0], dv[0]


def _support_projection(model, face_id: int, point: np.ndarray, seeds: Sequence[Sequence[float]] = ()):
    face = model.faces[face_id]
    surface = face.surface
    if surface is None:
        raise GeometryError(f"face {face_id} has no authoritative support")
    candidates: list[np.ndarray] = []
    try:
        local = model.face_support_local_uv(face_id, point)
        candidates.append(np.clip(np.asarray(local, dtype=float), 0.0, 1.0))
    except (GeometryError, ValueError, np.linalg.LinAlgError):
        pass
    candidates.extend(np.clip(np.asarray(item, dtype=float), 0.0, 1.0) for item in seeds)
    if not isinstance(surface, (Plane, Cylinder, Cone)):
        candidates.extend(
            np.asarray(item, dtype=float)
            for item in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))
        )
    best = None
    for seed in candidates:
        uv = seed.copy()
        for _ in range(24):
            current, du, dv = _surface_point_derivatives(model, face_id, uv)
            jacobian = np.column_stack((du, dv))
            if not np.all(np.isfinite(jacobian)) or np.linalg.matrix_rank(jacobian) < 2:
                break
            delta, *_ = np.linalg.lstsq(jacobian, point - current, rcond=None)
            next_uv = np.clip(uv + delta, 0.0, 1.0)
            if float(np.linalg.norm(next_uv - uv)) <= model.tolerance.parameter:
                uv = next_uv
                break
            uv = next_uv
        projected, du, dv = _surface_point_derivatives(model, face_id, uv)
        residual = float(np.linalg.norm(projected - point))
        candidate = (residual, float(uv[0]), float(uv[1]), projected, du, dv)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    residual, u, v, projected, du, dv = best
    normal = np.cross(du, dv)
    length = float(np.linalg.norm(normal))
    if length <= 0.0 or not np.isfinite(length):
        raise GeometryError(f"face {face_id} has a degenerate support normal")
    return projected, np.asarray((u, v), dtype=float), residual, normal / length


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (
            (float(current[1]) > float(point[1]))
            != (float(previous[1]) > float(point[1]))
        ):
            crossing = (
                (float(previous[0]) - float(current[0]))
                * (float(point[1]) - float(current[1]))
                / (float(previous[1]) - float(current[1]))
                + float(current[0])
            )
            if float(point[0]) < crossing:
                inside = not inside
        previous = current
    return inside


def _support_trim_loops(model, face_id: int) -> tuple[np.ndarray, ...]:
    face = model.faces[face_id]
    loops: list[np.ndarray] = []
    pending_loops = (face.loop,) + face.holes
    if (
        isinstance(face.surface, CoonsSurface)
        and not face.surface.has_boundaries
        and len(face.corners) == 4
    ):
        counts = tuple(
            2 if isinstance(model.edges[item.edge].curve, Straight) else 17
            for item in face.loop
        )
        loops.append(model._topology_coons_outer_loop_uv(face_id, counts))  # noqa: SLF001
        pending_loops = face.holes
    for loop in pending_loops:
        points: list[tuple[float, float]] = []
        for oriented in loop:
            count = 2 if isinstance(model.edges[oriented.edge].curve, Straight) else 17
            parameters = np.linspace(0.0, 1.0, count)
            if not oriented.forward:
                parameters = parameters[::-1]
            samples = _edge_points(model, oriented.edge, parameters)
            for sample in samples[:-1]:
                points.append(tuple(float(value) for value in model.face_support_local_uv(face_id, sample)))
        loops.append(np.asarray(points, dtype=float))
    return tuple(loops)


def _inside_face(model, face_id: int, uv: np.ndarray, loops: tuple[np.ndarray, ...] | None = None) -> bool:
    polygons = _support_trim_loops(model, face_id) if loops is None else loops
    if not polygons or not _point_in_polygon(uv, polygons[0]):
        return False
    return not any(_point_in_polygon(uv, hole) for hole in polygons[1:])


def _curve_surface_samples(model, edge_id: int, face_id: int, count: int):
    parameters = np.linspace(0.0, 1.0, count)
    points = _edge_points(model, edge_id, parameters)
    records = []
    seed = None
    for parameter, point in zip(parameters, points):
        seeds = () if seed is None else (seed,)
        projected, uv, residual, normal = _support_projection(model, face_id, point, seeds)
        seed = uv
        signed = float((point - projected) @ normal)
        records.append((float(parameter), point, uv, residual, signed))
    return records


def _curve_exactly_on_support(
    model,
    edge_id: int,
    face_id: int,
    tolerance: float,
) -> bool:
    """Prove that a complete built-in curve lies on an analytic support.

    This is deliberately a one-way predicate: ``False`` means that the
    bounded subdivision path must still decide the relation.  It never uses
    sampled agreement as proof.  Besides avoiding unnecessary Newton solves,
    the exact cases make shared cylindrical/conical boundary qualification
    independent of sampling density.
    """

    edge = model.edges[edge_id]
    surface = model.faces[face_id].surface
    if surface is None:
        return False

    if isinstance(edge.curve, Straight):
        start = model.vertex_position(edge.start)
        end = model.vertex_position(edge.end)
        direction = end - start
        direction_length = float(np.linalg.norm(direction))
        if not np.isfinite(direction_length) or direction_length <= tolerance:
            return False
        if isinstance(surface, Plane):
            normal = surface.normal
            return max(
                abs(float((start - surface.origin) @ normal)),
                abs(float((end - surface.origin) @ normal)),
            ) <= tolerance
        if isinstance(surface, Cylinder):
            unit_direction = direction / direction_length
            if float(np.linalg.norm(np.cross(unit_direction, surface.axis))) > model.tolerance.angular:
                return False
            offset = start - surface.origin
            radial = offset - float(offset @ surface.axis) * surface.axis
            return abs(float(np.linalg.norm(radial)) - surface.radius) <= tolerance
        if isinstance(surface, Cone):
            offset = start - surface.origin
            axial = float(offset @ surface.axis)
            radial = offset - axial * surface.axis
            delta_axial = float(direction @ surface.axis)
            delta_radial = direction - delta_axial * surface.axis
            slope = (surface.radius_end - surface.radius_start) / surface.height
            radius = surface.radius_start + slope * axial
            radius_delta = slope * delta_axial
            coefficients = (
                float(radial @ radial) - radius * radius,
                2.0 * (float(radial @ delta_radial) - radius * radius_delta),
                float(delta_radial @ delta_radial) - radius_delta * radius_delta,
            )
            scale = max(
                surface.radius_start,
                surface.radius_end,
                float(np.linalg.norm(radial)),
                float(np.linalg.norm(radial + delta_radial)),
                tolerance,
            )
            implicit_tolerance = tolerance * (2.0 * scale + tolerance)
            if max(abs(value) for value in coefficients) > implicit_tolerance:
                return False
            radii = (radius, radius + radius_delta)
            return min(radii) >= -tolerance
        return False

    if isinstance(edge.curve, Spline):
        if not isinstance(surface, Plane):
            return False
        points = model._spline_points(edge)  # noqa: SLF001
        residuals = np.abs((points - surface.origin) @ surface.normal)
        return bool(np.all(np.isfinite(residuals)) and float(np.max(residuals)) <= tolerance)

    if not isinstance(edge.curve, Arc):
        return False
    frame = model.arc_frame(edge_id)
    if isinstance(surface, Plane):
        defining = np.vstack(
            (
                model.vertex_position(edge.start),
                model.vertex_position(edge.curve.via_vertex),
                model.vertex_position(edge.end),
            )
        )
        residuals = np.abs((defining - surface.origin) @ surface.normal)
        return bool(np.all(np.isfinite(residuals)) and float(np.max(residuals)) <= tolerance)
    if not isinstance(surface, (Cylinder, Cone)):
        return False
    if abs(float(frame.normal @ surface.axis)) < 1.0 - model.tolerance.angular:
        return False
    center_offset = frame.center - surface.origin
    axial = float(center_offset @ surface.axis)
    center_radial = center_offset - axial * surface.axis
    if float(np.linalg.norm(center_radial)) > tolerance:
        return False
    expected_radius = (
        surface.radius
        if isinstance(surface, Cylinder)
        else surface.radius_start
        + (surface.radius_end - surface.radius_start) * axial / surface.height
    )
    return expected_radius >= -tolerance and abs(frame.radius - expected_radius) <= tolerance


def _refine_curve_surface_root(model, edge_id: int, face_id: int, lower: float, upper: float, policy):
    left_point = _edge_points(model, edge_id, (lower,))[0]
    _projected, left_uv, _residual, normal = _support_projection(model, face_id, left_point)
    left_value = float((left_point - _projected) @ normal)
    best = None
    for _ in range(max(24, policy.max_newton_iterations * 2)):
        middle = 0.5 * (lower + upper)
        point = _edge_points(model, edge_id, (middle,))[0]
        projected, uv, residual, normal = _support_projection(model, face_id, point, (left_uv,))
        value = float((point - projected) @ normal)
        candidate = (residual, middle, point, uv)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        if residual <= model.tolerance.length and upper - lower <= model.tolerance.parameter:
            break
        if left_value == 0.0 or left_value * value <= 0.0:
            upper = middle
        else:
            lower = middle
            left_value = value
            left_uv = uv
    assert best is not None
    return best[1], best[2], best[3], best[0]


def qualified_curve_face(
    model,
    edge: EntityHandle,
    face: EntityHandle,
    qualification: IntersectionQualificationPolicy | None = None,
) -> IntersectionResult:
    """Return complete bounded curve/material point or contained intervals."""

    policy = qualification or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    edge_bounds = _curve_bounds(model, edge.id, 0.0, 1.0)
    face_bounds_raw = model.conservative_face_bounds(face.id)
    if face_bounds_raw is None:
        return _result(
            edge, face, IntersectionKind.UNCLASSIFIED,
            tolerance=model.tolerance.length,
            algorithm="curve_surface_subdivision",
            diagnostics=("face_has_no_conservative_bounds",),
        )
    face_bounds = _Bounds(np.asarray(face_bounds_raw[:3]), np.asarray(face_bounds_raw[3:]))
    extent = feature_extent(np.vstack((edge_bounds.lower, edge_bounds.upper, face_bounds.lower, face_bounds.upper)))
    tolerance = model.tolerance.effective_surface_residual(extent)
    if edge_bounds.distance(face_bounds) > tolerance:
        return _result(
            edge, face, IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="curve_surface_bounds",
            diagnostics=("curve_outside_face_bounds",),
            separation=edge_bounds.distance(face_bounds),
        )
    curve = model.edges[edge.id].curve
    count = 65 if isinstance(curve, Straight) else 257
    count = min(count, policy.max_trace_segments + 1)
    exact_on_support = _curve_exactly_on_support(
        model, edge.id, face.id, tolerance
    )
    try:
        trim_loops = _support_trim_loops(model, face.id)
        samples = (
            []
            if exact_on_support
            else _curve_surface_samples(model, edge.id, face.id, count)
        )
    except (GeometryError, ValueError, np.linalg.LinAlgError) as error:
        return _result(
            edge, face, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="curve_surface_subdivision",
            diagnostics=(f"curve_surface_projection_failed:{error}",),
        )
    parameter_tolerance = model.tolerance.effective_parameter(model.edge_length(edge.id), extent)

    # A curve lying on the support is clipped exactly at qualified trim-edge
    # intersections, so holes and disconnected intervals do not collapse to a
    # sampled min/max range.
    if exact_on_support or (
        samples and max(item[3] for item in samples) <= tolerance
    ):
        breakpoints = [0.0, 1.0]
        for loop in (model.faces[face.id].loop,) + model.faces[face.id].holes:
            for oriented in loop:
                boundary = model.handle("edge", oriented.edge)
                if boundary == edge:
                    breakpoints.extend((0.0, 1.0))
                    continue
                relation = qualified_curve_curve(model, edge, boundary, policy)
                if not relation.classified:
                    return _result(
                        edge, face, IntersectionKind.UNCLASSIFIED,
                        tolerance=tolerance,
                        algorithm="curve_surface_trim",
                        diagnostics=("trim_curve_intersection_not_classified", *relation.diagnostics),
                    )
                for component in relation.components:
                    if component.first_parameter is not None:
                        breakpoints.append(component.first_parameter[0])
                    if component.first_parameter_range is not None:
                        breakpoints.extend((component.first_parameter_range.start, component.first_parameter_range.end))
        ordered = sorted(set(float(np.clip(item, 0.0, 1.0)) for item in breakpoints))
        intervals: list[IntersectionComponent] = []
        for lower, upper in zip(ordered, ordered[1:]):
            if upper - lower <= parameter_tolerance:
                continue
            middle = 0.5 * (lower + upper)
            middle_point = _edge_points(model, edge.id, (middle,))[0]
            if exact_on_support:
                support = model.faces[face.id].surface
                assert isinstance(support, (Plane, Cylinder, Cone))
                middle_uv = np.asarray(support.local_uv(middle_point), dtype=float)
                residual = 0.0
            else:
                _projected, middle_uv, residual, _normal = _support_projection(
                    model, face.id, middle_point
                )
            if not _inside_face(model, face.id, middle_uv, trim_loops):
                continue
            parameters = (lower, upper)
            points = _edge_points(model, edge.id, parameters)
            if exact_on_support:
                uvs = tuple(
                    np.asarray(support.local_uv(point), dtype=float)
                    for point in points
                )
                endpoint_residuals = (0.0, 0.0)
            else:
                projected_endpoints = tuple(
                    _support_projection(model, face.id, point) for point in points
                )
                uvs = tuple(item[1] for item in projected_endpoints)
                endpoint_residuals = tuple(float(item[2]) for item in projected_endpoints)
            certificate = _certificate(
                "curve_on_surface_trim",
                tolerance,
                residual=max(residual, *endpoint_residuals),
                boxes=1 if exact_on_support else count,
            )
            intervals.append(
                IntersectionComponent(
                    tuple(tuple(float(value) for value in point) for point in points),
                    IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter_range=ParameterRange(lower, upper),
                    first_parameter_path=((lower,), (upper,)),
                    second_parameter_path=tuple(tuple(float(value) for value in uv) for uv in uvs),
                    boundary_paths=(tuple(tuple(float(value) for value in point) for point in points),),
                    max_residual=certificate.max_residual,
                    certificate=certificate,
                    first_subparent=edge,
                    second_subparent=face,
                )
            )
        if intervals:
            full = (
                intervals[0].first_parameter_range is not None
                and intervals[-1].first_parameter_range is not None
                and intervals[0].first_parameter_range.lower <= parameter_tolerance
                and intervals[-1].first_parameter_range.upper >= 1.0 - parameter_tolerance
                and all(
                    previous.first_parameter_range is not None
                    and current.first_parameter_range is not None
                    and current.first_parameter_range.lower - previous.first_parameter_range.upper <= parameter_tolerance
                    for previous, current in zip(intervals, intervals[1:])
                )
            )
            return _result(
                edge,
                face,
                IntersectionKind.CONTAINED if full else IntersectionKind.OVERLAP_CURVE,
                intervals,
                tolerance=tolerance,
                algorithm="curve_on_surface_trim",
                diagnostics=("certified_material_intervals",),
                dimension=IntersectionDimension.CURVE,
                boxes=1 if exact_on_support else count,
            )
        return _result(
            edge,
            face,
            IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="curve_on_surface_trim",
            diagnostics=("curve_outside_face_material",),
            boxes=1 if exact_on_support else count,
        )

    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for previous, current in zip(samples, samples[1:]):
        sign_change = previous[4] == 0.0 or current[4] == 0.0 or previous[4] * current[4] < 0.0
        near = min(previous[3], current[3]) <= tolerance
        if not sign_change and not near:
            continue
        candidates.append(_refine_curve_surface_root(model, edge.id, face.id, previous[0], current[0], policy))
    for before, current, after in zip(samples, samples[1:], samples[2:]):
        if current[3] <= tolerance and current[3] <= before[3] and current[3] <= after[3]:
            candidates.append((current[0], current[1], current[2], current[3]))
    candidates.sort(key=lambda item: item[0])
    unique = []
    for candidate in candidates:
        if candidate[3] > tolerance or not _inside_face(model, face.id, candidate[2], trim_loops):
            continue
        if unique and abs(candidate[0] - unique[-1][0]) <= parameter_tolerance:
            if candidate[3] < unique[-1][3]:
                unique[-1] = candidate
            continue
        unique.append(candidate)
    if not unique:
        minimum = min(item[3] for item in samples)
        return _result(
            edge,
            face,
            IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="curve_surface_subdivision",
            diagnostics=("certified_curve_face_disjoint",),
            boxes=count,
            separation=max(0.0, minimum - tolerance),
        )
    components = []
    kinds = []
    for parameter, point, uv, residual in unique:
        tangent = _curve_derivative(model, edge.id, parameter)
        _surface_point, du, dv = _surface_point_derivatives(model, face.id, uv)
        normal = np.cross(du, dv)
        tangent_length = float(np.linalg.norm(tangent))
        normal_length = float(np.linalg.norm(normal))
        if (
            not np.isfinite(tangent_length)
            or not np.isfinite(normal_length)
            or tangent_length <= np.finfo(float).tiny
            or normal_length <= np.finfo(float).tiny
        ):
            return _result(
                edge,
                face,
                IntersectionKind.UNCLASSIFIED,
                tolerance=tolerance,
                algorithm="curve_surface_subdivision",
                diagnostics=("curve_surface_tangent_is_ill_conditioned",),
                boxes=count,
                subdivisions=count - 1,
            )
        tangential = abs(float(tangent @ normal)) <= (
            model.tolerance.angular * tangent_length * normal_length
        )
        endpoint = min(parameter, 1.0 - parameter) <= parameter_tolerance
        kind = IntersectionKind.TANGENT if tangential and not endpoint else (
            IntersectionKind.TOUCH_POINT if tangential or endpoint else IntersectionKind.CROSS
        )
        kinds.append(kind)
        certificate = _certificate(
            "curve_surface_subdivision",
            tolerance,
            residual=residual,
            enclosure=tolerance,
            boxes=count,
            subdivisions=count - 1,
        )
        components.append(
            IntersectionComponent(
                (tuple(float(value) for value in point),),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter=(parameter,),
                second_parameter=tuple(float(value) for value in uv),
                first_parameter_path=((parameter,),),
                second_parameter_path=(tuple(float(value) for value in uv),),
                max_residual=residual,
                certificate=certificate,
                first_subparent=edge,
                second_subparent=face,
            )
        )
    precedence = (IntersectionKind.CROSS, IntersectionKind.TANGENT, IntersectionKind.TOUCH_POINT)
    kind = next(item for item in precedence if item in kinds)
    return _result(
        edge,
        face,
        kind,
        components,
        tolerance=tolerance,
        algorithm="curve_surface_subdivision",
        dimension=IntersectionDimension.POINT,
        boxes=count,
        subdivisions=count - 1,
    )


def _polygon_area(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )


def _canonical_polygon(points: Sequence[Sequence[float]], tolerance: float) -> np.ndarray:
    made = np.asarray(points, dtype=float)
    if len(made) > 1 and float(np.linalg.norm(made[0] - made[-1])) <= tolerance:
        made = made[:-1]
    if len(made) < 3:
        raise GeometryError("intersection region boundary needs three points")
    if _polygon_area(made) < 0.0:
        made = made[::-1]
    start = min(
        range(len(made)),
        key=lambda index: (float(made[index, 0]), float(made[index, 1]), index),
    )
    return np.vstack((made[start:], made[:start]))


def _convex_polygon(points: np.ndarray, tolerance: float) -> bool:
    signs = []
    for index in range(len(points)):
        first = points[(index + 1) % len(points)] - points[index]
        second = points[(index + 2) % len(points)] - points[(index + 1) % len(points)]
        cross = float(first[0] * second[1] - first[1] * second[0])
        if abs(cross) > tolerance:
            signs.append(cross > 0.0)
    return not signs or all(item == signs[0] for item in signs)


def _line_intersection_2d(
    first: np.ndarray,
    second: np.ndarray,
    clip_first: np.ndarray,
    clip_second: np.ndarray,
) -> np.ndarray:
    direction = second - first
    clip_direction = clip_second - clip_first
    denominator = float(
        direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    )
    if abs(denominator) <= np.finfo(float).eps:
        return 0.5 * (first + second)
    offset = clip_first - first
    parameter = float(
        (offset[0] * clip_direction[1] - offset[1] * clip_direction[0])
        / denominator
    )
    return first + parameter * direction


def _convex_intersection(subject: np.ndarray, clip: np.ndarray, tolerance: float) -> np.ndarray:
    output = [np.asarray(item, dtype=float) for item in subject]
    orientation = 1.0 if _polygon_area(clip) >= 0.0 else -1.0
    for index, clip_first in enumerate(clip):
        clip_second = clip[(index + 1) % len(clip)]
        incoming = output
        output = []
        if not incoming:
            break

        def inside(point: np.ndarray) -> bool:
            edge = clip_second - clip_first
            offset = point - clip_first
            return orientation * float(edge[0] * offset[1] - edge[1] * offset[0]) >= -tolerance

        previous = incoming[-1]
        previous_inside = inside(previous)
        for current in incoming:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection_2d(previous, current, clip_first, clip_second)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection_2d(previous, current, clip_first, clip_second)
                )
            previous = current
            previous_inside = current_inside
    return np.asarray(output, dtype=float).reshape((-1, 2))


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _oriented_material_loops(
    loops: Sequence[np.ndarray], tolerance: float
) -> tuple[np.ndarray, ...]:
    made: list[np.ndarray] = []
    for index, loop in enumerate(loops):
        polygon = _canonical_polygon(loop, tolerance)
        if index > 0:
            polygon = polygon[::-1]
        made.append(polygon)
    return tuple(made)


def _point_on_polygon_boundary(
    point: np.ndarray, polygon: np.ndarray, tolerance: float
) -> bool:
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        direction = end - start
        length_squared = float(direction @ direction)
        if length_squared <= np.finfo(float).tiny:
            if float(np.linalg.norm(point - start)) <= tolerance:
                return True
            continue
        parameter = float(
            np.clip((point - start) @ direction / length_squared, 0.0, 1.0)
        )
        if float(np.linalg.norm(point - (start + parameter * direction))) <= tolerance:
            return True
    return False


def _inside_material(
    point: np.ndarray, loops: Sequence[np.ndarray], tolerance: float
) -> bool:
    if not loops:
        return False
    if _point_on_polygon_boundary(point, loops[0], tolerance):
        return True
    if not _point_in_polygon(point, loops[0]):
        return False
    for hole in loops[1:]:
        if _point_on_polygon_boundary(point, hole, tolerance):
            return True
        if _point_in_polygon(point, hole):
            return False
    return True


def _segment_breaks(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = _cross_2d(first_direction, second_direction)
    offset = second_start - first_start
    first_values: list[float] = []
    second_values: list[float] = []
    scale = max(
        float(np.linalg.norm(first_direction)),
        float(np.linalg.norm(second_direction)),
        1.0,
    )
    if abs(denominator) > tolerance * scale:
        first_parameter = _cross_2d(offset, second_direction) / denominator
        second_parameter = _cross_2d(offset, first_direction) / denominator
        if -tolerance <= first_parameter <= 1.0 + tolerance and -tolerance <= second_parameter <= 1.0 + tolerance:
            first_values.append(float(np.clip(first_parameter, 0.0, 1.0)))
            second_values.append(float(np.clip(second_parameter, 0.0, 1.0)))
        return tuple(first_values), tuple(second_values)
    if abs(_cross_2d(offset, first_direction)) > tolerance * scale:
        return (), ()
    first_length = float(first_direction @ first_direction)
    second_length = float(second_direction @ second_direction)
    if first_length <= np.finfo(float).tiny or second_length <= np.finfo(float).tiny:
        return (), ()
    for point in (second_start, second_end):
        value = float((point - first_start) @ first_direction / first_length)
        if -tolerance <= value <= 1.0 + tolerance:
            first_values.append(float(np.clip(value, 0.0, 1.0)))
    for point in (first_start, first_end):
        value = float((point - second_start) @ second_direction / second_length)
        if -tolerance <= value <= 1.0 + tolerance:
            second_values.append(float(np.clip(value, 0.0, 1.0)))
    return tuple(first_values), tuple(second_values)


def _material_intersection_loops(
    first_loops: Sequence[np.ndarray],
    second_loops: Sequence[np.ndarray],
    tolerance: float,
) -> tuple[np.ndarray, ...] | None:
    """Return directed boundary loops of two polygonal material domains."""

    first = _oriented_material_loops(first_loops, tolerance)
    second = _oriented_material_loops(second_loops, tolerance)
    first_segments = [
        (start, loop[(index + 1) % len(loop)])
        for loop in first
        for index, start in enumerate(loop)
    ]
    second_segments = [
        (start, loop[(index + 1) % len(loop)])
        for loop in second
        for index, start in enumerate(loop)
    ]
    first_breaks = [[0.0, 1.0] for _ in first_segments]
    second_breaks = [[0.0, 1.0] for _ in second_segments]
    for first_index, (first_start, first_end) in enumerate(first_segments):
        for second_index, (second_start, second_end) in enumerate(second_segments):
            made_first, made_second = _segment_breaks(
                first_start,
                first_end,
                second_start,
                second_end,
                tolerance,
            )
            first_breaks[first_index].extend(made_first)
            second_breaks[second_index].extend(made_second)

    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    for segments, breaks, other in (
        (first_segments, first_breaks, second),
        (second_segments, second_breaks, first),
    ):
        for (start, end), values in zip(segments, breaks):
            ordered = sorted(set(float(np.clip(value, 0.0, 1.0)) for value in values))
            direction = end - start
            for lower, upper in zip(ordered, ordered[1:]):
                if upper - lower <= tolerance:
                    continue
                piece_start = start + lower * direction
                piece_end = start + upper * direction
                midpoint = 0.5 * (piece_start + piece_end)
                if _inside_material(midpoint, other, tolerance):
                    pieces.append((piece_start, piece_end))
    if not pieces:
        return ()

    quantum = max(tolerance, 64.0 * np.finfo(float).eps)

    def key(point: np.ndarray) -> tuple[int, int]:
        return tuple(int(round(float(value) / quantum)) for value in point)  # type: ignore[return-value]

    unique: dict[tuple[tuple[int, int], tuple[int, int]], tuple[np.ndarray, np.ndarray]] = {}
    for start, end in pieces:
        start_key, end_key = key(start), key(end)
        if start_key == end_key:
            continue
        direct = (start_key, end_key)
        reverse = (end_key, start_key)
        if reverse in unique:
            unique.pop(reverse)
        else:
            unique.setdefault(direct, (start, end))
    if not unique:
        return ()
    outgoing: dict[tuple[int, int], list[tuple[tuple[int, int], np.ndarray, np.ndarray]]] = {}
    for (start_key, end_key), (start, end) in unique.items():
        outgoing.setdefault(start_key, []).append((end_key, start, end))
    for values in outgoing.values():
        values.sort(key=lambda item: item[0])

    unused = set(unique)
    loops: list[np.ndarray] = []
    while unused:
        start_edge = min(unused)
        current = start_edge
        points: list[np.ndarray] = []
        for _ in range(len(unique) + 1):
            if current not in unused:
                if current[0] == start_edge[0]:
                    break
                return None
            unused.remove(current)
            segment = unique[current]
            if not points:
                points.append(segment[0])
            points.append(segment[1])
            if current[1] == start_edge[0]:
                break
            candidates = [
                item for item in outgoing.get(current[1], ())
                if (current[1], item[0]) in unused
            ]
            if not candidates:
                return None
            incoming = segment[1] - segment[0]
            chosen = max(
                candidates,
                key=lambda item: (
                    float(
                        np.arctan2(
                            _cross_2d(incoming, item[2] - item[1]),
                            float(incoming @ (item[2] - item[1])),
                        )
                    ),
                    tuple(-value for value in item[0]),
                ),
            )
            current = (current[1], chosen[0])
        if len(points) < 4 or key(points[0]) != key(points[-1]):
            return None
        polygon = np.asarray(points[:-1], dtype=float)
        if abs(_polygon_area(polygon)) > tolerance * tolerance:
            loops.append(polygon)
    return tuple(
        sorted(
            loops,
            key=lambda loop: (
                _polygon_area(loop) < 0.0,
                float(np.min(loop[:, 0])),
                float(np.min(loop[:, 1])),
                -abs(_polygon_area(loop)),
            ),
        )
    )


def _coincident_face_region(model, first: EntityHandle, second: EntityHandle, tolerance: float):
    first_loops = _support_trim_loops(model, first.id)
    second_loops = _support_trim_loops(model, second.id)
    if not first_loops or not second_loops:
        return None
    parameter_tolerance = model.tolerance.parameter
    first_material = _oriented_material_loops(first_loops, parameter_tolerance)
    second_on_first: list[np.ndarray] = []
    for index, loop in enumerate(second_loops):
        world = np.asarray(
            [
                model.face_support_point(second.id, float(u), float(v))
                for u, v in loop
            ],
            dtype=float,
        )
        mapped = _canonical_polygon(
            [model.face_support_local_uv(first.id, point) for point in world],
            parameter_tolerance,
        )
        if index > 0:
            mapped = mapped[::-1]
        second_on_first.append(mapped)
    intersection_loops = _material_intersection_loops(
        first_material,
        tuple(second_on_first),
        parameter_tolerance,
    )
    if intersection_loops is None:
        return None
    if not intersection_loops:
        return ()

    outers = [loop for loop in intersection_loops if _polygon_area(loop) > 0.0]
    holes = [loop for loop in intersection_loops if _polygon_area(loop) < 0.0]
    if not outers:
        return None
    grouped_holes: list[list[np.ndarray]] = [[] for _ in outers]
    for hole in holes:
        owners = [
            index for index, outer in enumerate(outers)
            if _point_in_polygon(hole[0], outer)
        ]
        if len(owners) != 1:
            return None
        grouped_holes[owners[0]].append(hole)

    first_area = abs(_polygon_area(first_material[0])) - sum(
        abs(_polygon_area(loop)) for loop in first_material[1:]
    )
    second_area = abs(_polygon_area(second_on_first[0])) - sum(
        abs(_polygon_area(loop)) for loop in second_on_first[1:]
    )
    overlap_area = sum(abs(_polygon_area(loop)) for loop in outers) - sum(
        abs(_polygon_area(loop)) for loop in holes
    )
    area_tolerance = model.tolerance.effective_area(max(first_area, second_area, 1.0))
    kind = (
        IntersectionKind.COINCIDENT
        if abs(overlap_area - first_area) <= area_tolerance
        and abs(overlap_area - second_area) <= area_tolerance
        else IntersectionKind.CONTAINED
        if abs(overlap_area - min(first_area, second_area)) <= area_tolerance
        else IntersectionKind.OVERLAP_REGION
    )
    components: list[IntersectionComponent] = []
    for outer, region_holes in zip(outers, grouped_holes):
        parameter_loops = (outer, *sorted(region_holes, key=lambda item: tuple(item[0])))
        world_loops = tuple(
            np.asarray(
                [
                    model.face_support_point(first.id, float(u), float(v))
                    for u, v in loop
                ],
                dtype=float,
            )
            for loop in parameter_loops
        )
        second_parameter_loops = tuple(
            np.asarray(
                [model.face_support_local_uv(second.id, point) for point in loop],
                dtype=float,
            )
            for loop in world_loops
        )
        residual = max(
            float(
                np.linalg.norm(
                    model.face_support_point(second.id, float(uv[0]), float(uv[1]))
                    - point
                )
            )
            for world_loop, second_loop in zip(world_loops, second_parameter_loops)
            for point, uv in zip(world_loop, second_loop)
        )
        if residual > tolerance:
            return None
        certificate = _certificate(
            "coincident_support_parameter_arrangement",
            tolerance,
            residual=residual,
            boxes=sum(len(loop) for loop in (*first_material, *second_on_first)),
            subdivisions=sum(len(loop) for loop in intersection_loops),
        )
        components.append(
            IntersectionComponent(
                (tuple(float(value) for value in world_loops[0].mean(axis=0)),),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                boundary_paths=tuple(
                    tuple(tuple(float(value) for value in point) for point in loop)
                    for loop in world_loops
                ),
                first_region=ParameterRegion(
                    ParameterLoop(tuple(tuple(float(value) for value in uv) for uv in outer)),
                    tuple(
                        ParameterLoop(tuple(tuple(float(value) for value in uv) for uv in hole))
                        for hole in region_holes
                    ),
                ),
                second_region=ParameterRegion(
                    ParameterLoop(
                        tuple(tuple(float(value) for value in uv) for uv in second_parameter_loops[0])
                    ),
                    tuple(
                        ParameterLoop(tuple(tuple(float(value) for value in uv) for uv in hole))
                        for hole in second_parameter_loops[1:]
                    ),
                ),
                max_residual=residual,
                certificate=certificate,
                first_subparent=first,
                second_subparent=second,
            )
        )
    return kind, tuple(components)


def _supports_coincident(model, first_id: int, second_id: int, tolerance: float) -> bool:
    first_surface = model.faces[first_id].surface
    second_surface = model.faces[second_id].surface
    if isinstance(first_surface, Cylinder) and isinstance(second_surface, Cylinder):
        axis_alignment = abs(float(first_surface.axis @ second_surface.axis))
        offset = second_surface.origin - first_surface.origin
        perpendicular = offset - float(offset @ first_surface.axis) * first_surface.axis
        return (
            1.0 - axis_alignment <= model.tolerance.angular
            and float(np.linalg.norm(perpendicular)) <= tolerance
            and abs(first_surface.radius - second_surface.radius) <= tolerance
        )
    if isinstance(first_surface, Cone) and isinstance(second_surface, Cone):
        first_slope = (first_surface.radius_end - first_surface.radius_start) / first_surface.height
        second_slope = (second_surface.radius_end - second_surface.radius_start) / second_surface.height
        if abs(first_slope) <= np.finfo(float).tiny or abs(second_slope) <= np.finfo(float).tiny:
            return False
        first_apex = first_surface.origin - (
            first_surface.radius_start / first_slope
        ) * first_surface.axis
        second_apex = second_surface.origin - (
            second_surface.radius_start / second_slope
        ) * second_surface.axis
        return (
            abs(abs(float(first_surface.axis @ second_surface.axis)) - 1.0)
            <= model.tolerance.angular
            and float(np.linalg.norm(first_apex - second_apex)) <= tolerance
            and abs(abs(first_slope) - abs(second_slope))
            <= model.tolerance.angular * max(abs(first_slope), abs(second_slope), 1.0)
        )
    if isinstance(first_surface, RuledSurface) and isinstance(second_surface, RuledSurface):
        variants = (
            (second_surface.first_boundary, second_surface.second_boundary),
            (second_surface.first_boundary[::-1], second_surface.second_boundary[::-1]),
            (second_surface.second_boundary, second_surface.first_boundary),
            (second_surface.second_boundary[::-1], second_surface.first_boundary[::-1]),
        )
        for first_boundary, second_boundary in variants:
            if (
                first_boundary.shape == first_surface.first_boundary.shape
                and second_boundary.shape == first_surface.second_boundary.shape
                and float(np.max(np.linalg.norm(first_boundary - first_surface.first_boundary, axis=1))) <= tolerance
                and float(np.max(np.linalg.norm(second_boundary - first_surface.second_boundary, axis=1))) <= tolerance
            ):
                return True
    if (
        isinstance(first_surface, CoonsSurface)
        and isinstance(second_surface, CoonsSurface)
        and first_surface.has_boundaries
        and second_surface.has_boundaries
    ):
        assert first_surface.bottom is not None and first_surface.right is not None
        assert first_surface.top is not None and first_surface.left is not None
        assert second_surface.bottom is not None and second_surface.right is not None
        assert second_surface.top is not None and second_surface.left is not None
        first_boundaries = (
            first_surface.bottom,
            first_surface.right,
            first_surface.top,
            first_surface.left,
        )
        second_boundaries = (
            second_surface.bottom,
            second_surface.right,
            second_surface.top,
            second_surface.left,
        )
        if all(
            left.shape == right.shape
            and float(np.max(np.linalg.norm(left - right, axis=1))) <= tolerance
            for left, right in zip(first_boundaries, second_boundaries)
        ):
            return True
    parameters = np.asarray(
        tuple((float(u), float(v)) for u in np.linspace(0.0, 1.0, 17) for v in np.linspace(0.0, 1.0, 17)),
        dtype=float,
    )
    for source, target in ((first_id, second_id), (second_id, first_id)):
        for uv in parameters:
            point, source_du, source_dv = _surface_point_derivatives(model, source, uv)
            projected, _target_uv, residual, target_normal = _support_projection(model, target, point)
            if residual > tolerance:
                return False
            source_normal = np.cross(source_du, source_dv)
            source_length = float(np.linalg.norm(source_normal))
            if source_length <= 0.0:
                return False
            alignment = abs(float((source_normal / source_length) @ target_normal))
            if alignment < 1.0 - model.tolerance.angular:
                return False
            if float(np.linalg.norm(projected - point)) > tolerance:
                return False
    return True


def _surface_grid(model, face_id: int, count: int):
    parameters = np.linspace(0.0, 1.0, count)
    uv = np.asarray(tuple((float(u), float(v)) for v in parameters for u in parameters), dtype=float)
    points = np.asarray(
        [model.face_support_point(face_id, float(item[0]), float(item[1])) for item in uv],
        dtype=float,
    )
    return uv.reshape((count, count, 2)), points.reshape((count, count, 3))


def _contour_crossing(
    first_uv: np.ndarray,
    first_value: float,
    second_uv: np.ndarray,
    second_value: float,
) -> np.ndarray | None:
    if first_value == 0.0:
        return first_uv.copy()
    if second_value == 0.0:
        return second_uv.copy()
    if first_value * second_value > 0.0:
        return None
    denominator = first_value - second_value
    if denominator == 0.0:
        return 0.5 * (first_uv + second_uv)
    fraction = float(np.clip(first_value / denominator, 0.0, 1.0))
    return first_uv + fraction * (second_uv - first_uv)


def _refine_surface_pair(model, first_id: int, second_id: int, first_uv, second_uv, policy):
    first_parameters = np.clip(np.asarray(first_uv, dtype=float), 0.0, 1.0)
    second_parameters = np.clip(np.asarray(second_uv, dtype=float), 0.0, 1.0)
    for _ in range(policy.max_newton_iterations):
        first_point, first_du, first_dv = _surface_point_derivatives(
            model, first_id, first_parameters
        )
        second_point, second_du, second_dv = _surface_point_derivatives(
            model, second_id, second_parameters
        )
        difference = first_point - second_point
        jacobian = np.column_stack(
            (first_du, first_dv, -second_du, -second_dv)
        )
        if not np.all(np.isfinite(jacobian)) or np.linalg.matrix_rank(jacobian) < 3:
            break
        delta, *_ = np.linalg.lstsq(jacobian, -difference, rcond=None)
        next_first = np.clip(first_parameters + delta[:2], 0.0, 1.0)
        next_second = np.clip(second_parameters + delta[2:], 0.0, 1.0)
        if max(
            float(np.linalg.norm(next_first - first_parameters)),
            float(np.linalg.norm(next_second - second_parameters)),
        ) <= model.tolerance.parameter:
            first_parameters, second_parameters = next_first, next_second
            break
        first_parameters, second_parameters = next_first, next_second
    first_point, _du, _dv = _surface_point_derivatives(model, first_id, first_parameters)
    second_point, _du, _dv = _surface_point_derivatives(model, second_id, second_parameters)
    return (
        0.5 * (first_point + second_point),
        first_parameters,
        second_parameters,
        float(np.linalg.norm(first_point - second_point)),
    )


def _trace_segments(segments, quantum: float):
    def key(point: np.ndarray):
        return tuple(int(round(float(value) / quantum)) for value in point)

    adjacency: dict[tuple[int, int, int], list[int]] = {}
    for index, segment in enumerate(segments):
        for endpoint in (segment[0][0], segment[1][0]):
            adjacency.setdefault(key(endpoint), []).append(index)
    unused = set(range(len(segments)))
    paths = []
    while unused:
        seed = min(unused)
        candidates = []
        for endpoint_index in (0, 1):
            endpoint_key = key(segments[seed][endpoint_index][0])
            if len(adjacency.get(endpoint_key, ())) == 1:
                candidates.append(endpoint_index)
        start_index = min(candidates) if candidates else 0
        current_segment = seed
        current_endpoint = start_index
        path = []
        while current_segment in unused:
            unused.remove(current_segment)
            segment = segments[current_segment]
            first = segment[current_endpoint]
            second = segment[1 - current_endpoint]
            if not path:
                path.append(first)
            path.append(second)
            endpoint_key = key(second[0])
            following = sorted(item for item in adjacency.get(endpoint_key, ()) if item in unused)
            if not following:
                break
            current_segment = following[0]
            current_endpoint = 0 if key(segments[current_segment][0][0]) == endpoint_key else 1
        paths.append(path)
    return paths


def qualified_face_face(
    model,
    first: EntityHandle,
    second: EntityHandle,
    qualification: IntersectionQualificationPolicy | None = None,
) -> IntersectionResult:
    """Qualified fallback for every built-in curved support pair."""

    policy = qualification or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    first_raw = model.conservative_face_bounds(first.id)
    second_raw = model.conservative_face_bounds(second.id)
    if first_raw is None or second_raw is None:
        return _result(
            first, second, IntersectionKind.UNCLASSIFIED,
            tolerance=model.tolerance.length,
            algorithm="surface_surface_subdivision",
            diagnostics=("face_pair_has_no_conservative_bounds",),
        )
    first_bounds = _Bounds(np.asarray(first_raw[:3]), np.asarray(first_raw[3:]))
    second_bounds = _Bounds(np.asarray(second_raw[:3]), np.asarray(second_raw[3:]))
    extent = feature_extent(np.vstack((first_bounds.lower, first_bounds.upper, second_bounds.lower, second_bounds.upper)))
    tolerance = model.tolerance.effective_surface_residual(extent)
    separation = first_bounds.distance(second_bounds)
    if separation > tolerance:
        return _result(
            first, second, IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="surface_surface_bounds",
            diagnostics=("face_bounds_disjoint",),
            separation=separation,
        )
    first_edges = {
        oriented.edge
        for loop in (model.faces[first.id].loop,) + model.faces[first.id].holes
        for oriented in loop
    }
    second_edges = {
        oriented.edge
        for loop in (model.faces[second.id].loop,) + model.faces[second.id].holes
        for oriented in loop
    }
    shared_edges = tuple(sorted(first_edges & second_edges))
    if shared_edges:
        components = []
        for edge_id in shared_edges:
            curve = model.edges[edge_id].curve
            count = 2 if isinstance(curve, Straight) else 17
            parameters = np.linspace(0.0, 1.0, count)
            points = _edge_points(model, edge_id, parameters)
            first_path = tuple(
                tuple(float(value) for value in model.face_support_local_uv(first.id, point))
                for point in points
            )
            second_path = tuple(
                tuple(float(value) for value in model.face_support_local_uv(second.id, point))
                for point in points
            )
            certificate = _certificate(
                "shared_topology_edge",
                tolerance,
                boxes=count,
            )
            components.append(
                IntersectionComponent(
                    tuple(tuple(float(value) for value in point) for point in points),
                    IntersectionQuality.EXACT,
                    first_parameter_path=first_path,
                    second_parameter_path=second_path,
                    boundary_paths=(
                        tuple(tuple(float(value) for value in point) for point in points),
                    ),
                    certificate=certificate,
                    first_subparent=model.handle("edge", edge_id),
                    second_subparent=model.handle("edge", edge_id),
                )
            )
        return _result(
            first,
            second,
            IntersectionKind.OVERLAP_CURVE,
            components,
            tolerance=tolerance,
            algorithm="shared_topology_edge",
            diagnostics=("shared_face_boundary_topology",),
            dimension=IntersectionDimension.CURVE,
            boxes=sum(len(item.witnesses) for item in components),
        )
    try:
        if _supports_coincident(model, first.id, second.id, tolerance):
            region = _coincident_face_region(model, first, second, tolerance)
            if region == ():
                return _result(
                    first, second, IntersectionKind.DISJOINT,
                    tolerance=tolerance,
                    algorithm="coincident_support_parameter_overlay",
                    diagnostics=("coincident_support_trim_regions_disjoint",),
                    separation=0.0,
                )
            if region is None:
                return _result(
                    first, second, IntersectionKind.UNCLASSIFIED,
                    tolerance=tolerance,
                    algorithm="coincident_support_parameter_overlay",
                    diagnostics=("coincident_region_requires_a_certifiable_common_parameter_overlay",),
                )
            kind, components = region
            return _result(
                first,
                second,
                kind,
                components,
                tolerance=tolerance,
                algorithm="coincident_support_parameter_overlay",
                diagnostics=("certified_coincident_region",),
                dimension=IntersectionDimension.REGION,
            )
    except (GeometryError, ValueError, np.linalg.LinAlgError) as error:
        return _result(
            first, second, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="surface_surface_subdivision",
            diagnostics=(f"surface_coincidence_qualification_failed:{error}",),
        )

    count = min(33, max(9, int(np.sqrt(policy.max_trace_segments)) // 8 + 1))
    first_uv_grid, first_points = _surface_grid(model, first.id, count)
    second_seed_uv, second_seed_points = _surface_grid(model, second.id, 9)
    flat_seeds = second_seed_uv.reshape((-1, 2))
    flat_seed_points = second_seed_points.reshape((-1, 3))
    signed = np.empty((count, count), dtype=float)
    residuals = np.empty((count, count), dtype=float)
    projected_uv = np.empty((count, count, 2), dtype=float)
    try:
        for row in range(count):
            for column in range(count):
                point = first_points[row, column]
                nearest = int(np.argmin(np.linalg.norm(flat_seed_points - point, axis=1)))
                projected, uv, residual, normal = _support_projection(
                    model, second.id, point, (flat_seeds[nearest],)
                )
                signed[row, column] = float((point - projected) @ normal)
                residuals[row, column] = residual
                projected_uv[row, column] = uv
    except (GeometryError, ValueError, np.linalg.LinAlgError) as error:
        return _result(
            first, second, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="surface_surface_subdivision",
            diagnostics=(f"surface_projection_failed:{error}",),
        )
    first_trim = _support_trim_loops(model, first.id)
    second_trim = _support_trim_loops(model, second.id)
    segments = []
    edge_indices = ((0, 1), (1, 2), (2, 3), (3, 0))
    for row in range(count - 1):
        for column in range(count - 1):
            corners = (
                (row, column),
                (row, column + 1),
                (row + 1, column + 1),
                (row + 1, column),
            )
            crossings = []
            for first_index, second_index in edge_indices:
                first_corner = corners[first_index]
                second_corner = corners[second_index]
                crossing = _contour_crossing(
                    first_uv_grid[first_corner],
                    signed[first_corner],
                    first_uv_grid[second_corner],
                    signed[second_corner],
                )
                if crossing is not None:
                    point = model.face_support_point(first.id, float(crossing[0]), float(crossing[1]))
                    projected, uv_second, residual, _normal = _support_projection(model, second.id, point)
                    world, crossing, uv_second, residual = _refine_surface_pair(
                        model,
                        first.id,
                        second.id,
                        crossing,
                        uv_second,
                        policy,
                    )
                    if residual <= tolerance:
                        crossings.append((world, crossing, uv_second, residual))
            deduplicated = []
            for crossing in crossings:
                if not any(float(np.linalg.norm(crossing[0] - item[0])) <= tolerance for item in deduplicated):
                    deduplicated.append(crossing)
            if len(deduplicated) == 2:
                midpoint_first = 0.5 * (deduplicated[0][1] + deduplicated[1][1])
                midpoint_second = 0.5 * (deduplicated[0][2] + deduplicated[1][2])
                if _inside_face(model, first.id, midpoint_first, first_trim) and _inside_face(model, second.id, midpoint_second, second_trim):
                    segments.append((deduplicated[0], deduplicated[1]))
            elif len(deduplicated) == 4:
                remaining = list(deduplicated)
                while remaining:
                    start = remaining.pop(0)
                    nearest = min(range(len(remaining)), key=lambda index: float(np.linalg.norm(start[0] - remaining[index][0])))
                    end = remaining.pop(nearest)
                    segments.append((start, end))
            if len(segments) > policy.max_trace_segments:
                return _result(
                    first, second, IntersectionKind.UNCLASSIFIED,
                    tolerance=tolerance,
                    algorithm="surface_surface_subdivision",
                    diagnostics=("surface_trace_segment_limit_exceeded",),
                )
    if not segments:
        minimum = float(np.min(residuals))
        return _result(
            first,
            second,
            IntersectionKind.DISJOINT,
            tolerance=tolerance,
            algorithm="surface_surface_subdivision",
            diagnostics=("certified_face_material_disjoint",),
            boxes=(count - 1) ** 2,
            subdivisions=(count - 1) ** 2,
            separation=max(0.0, minimum - tolerance),
        )
    quantum = max(tolerance, extent * 1.0e-10, np.finfo(float).eps)
    paths = _trace_segments(segments, quantum)
    components = []
    kinds = []
    for path in paths:
        if len(path) < 2:
            continue
        worlds = tuple(tuple(float(value) for value in item[0]) for item in path)
        first_path = tuple(tuple(float(value) for value in item[1]) for item in path)
        second_path = tuple(tuple(float(value) for value in item[2]) for item in path)
        residual = max(float(item[3]) for item in path)
        middle = path[len(path) // 2]
        _point_first, first_du, first_dv = _surface_point_derivatives(model, first.id, middle[1])
        _point_second, second_du, second_dv = _surface_point_derivatives(model, second.id, middle[2])
        first_normal = np.cross(first_du, first_dv)
        second_normal = np.cross(second_du, second_dv)
        alignment = abs(float(first_normal @ second_normal)) / max(float(np.linalg.norm(first_normal) * np.linalg.norm(second_normal)), np.finfo(float).tiny)
        kind = IntersectionKind.TANGENT if alignment >= 1.0 - model.tolerance.angular else IntersectionKind.CROSS
        kinds.append(kind)
        certificate = _certificate(
            "surface_surface_subdivision",
            tolerance,
            residual=residual,
            enclosure=max(first_bounds.diagonal, second_bounds.diagonal) / (count - 1),
            boxes=(count - 1) ** 2,
            subdivisions=(count - 1) ** 2,
            trace_segments=max(0, len(path) - 1),
        )
        components.append(
            IntersectionComponent(
                worlds,
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter_path=first_path,
                second_parameter_path=second_path,
                boundary_paths=(worlds,),
                max_residual=residual,
                certificate=certificate,
                first_subparent=first,
                second_subparent=second,
            )
        )
    if not components:
        return _result(
            first, second, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="surface_surface_subdivision",
            diagnostics=("surface_trace_connectivity_unresolved",),
        )
    kind = IntersectionKind.CROSS if IntersectionKind.CROSS in kinds else IntersectionKind.TANGENT
    return _result(
        first,
        second,
        kind,
        components,
        tolerance=tolerance,
        algorithm="surface_surface_subdivision",
        diagnostics=("certified_curved_face_components",),
        dimension=IntersectionDimension.CURVE,
        boxes=(count - 1) ** 2,
        subdivisions=(count - 1) ** 2,
    )


__all__ = ["qualified_curve_curve", "qualified_curve_face", "qualified_face_face"]
