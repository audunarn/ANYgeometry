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
from fractions import Fraction
from math import atan2, ceil, comb, floor, pi
import heapq
from typing import Iterable, Sequence

import numpy as np

from .curves import Arc, Spline, Straight, _wrapped_angle
from .errors import GeometryError
from .exact_curves import EXACT_CURVES, EllipticArc, CylinderIntersectionCurve
from .line_ranges import LineRange
from .nurbs import CurveOnSurface, NurbsCurve
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
    elif isinstance(edge.curve, EXACT_CURVES):
        low, high = edge.curve.bounds(lower, upper)
        return _Bounds(low, high)
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
    if isinstance(edge.curve, EXACT_CURVES):
        return edge.curve.derivative(float(parameter))
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


class _PlaneChart:
    """How the support uv of one face may be treated by this module.

    A ``Plane`` is an unbounded affine chart: its stored frame is free and
    need not cover the face (the ANYfem webs store frames whose unit square
    misses most of the face).  Clipping such uv to ``[0, 1]^2`` evaluates the
    support metres away from the queried point and certifies real crossings
    as disjoint.  ``unbounded`` is True only for an exact ``Plane`` support
    whose frame is not proven to cover the face; every other support, and a
    Plane whose frame provably covers the face, keeps the historical bounded
    treatment bit for bit.  ``box`` is the face's conservative support-uv
    box ``((u_min, v_min), (u_max, v_max))`` or ``None`` when no conservative
    box is available for its trim curves.
    """

    __slots__ = ("unbounded", "box")

    def __init__(self, unbounded: bool, box: np.ndarray | None = None) -> None:
        self.unbounded = unbounded
        self.box = box


_BOUNDED_CHART = _PlaneChart(False)


_COVER_PIECE_BUDGET = 4096
_COVER_MAX_DEPTH = 48
# Rounding pads of the covering test (S0g round 4, Codex R3 P1).
_UNIT_ROUNDOFF = 2.0 ** -53
_PAD_GAMMA = 16.0 * _UNIT_ROUNDOFF
_PAD_SAFETY = 2.0
_PAD_MEANINGFUL = 0.25


def _round_up(value: Fraction) -> float:
    """The smallest double not below a non-negative rational."""

    result = float(value)
    if Fraction(result) < value:
        result = float(np.nextafter(result, np.inf))
    return result


class _ExactPlaneUV:
    """The exact support uv of a ``Plane``: ``uv(p) = R p - c`` over the rationals.

    ``R = (A^T A)^-1 A^T`` with ``A = [u_vector, v_vector]`` and ``c = R origin``
    is the least-squares ``Plane.local_uv`` evaluated without rounding on the
    stored doubles.  ``rows`` / ``offset`` are ``R`` / ``c`` rounded to nearest
    once (relative error at most the unit roundoff per entry).
    """

    __slots__ = ("exact_rows", "exact_offset", "rows", "offset", "abs_rows", "abs_offset")

    def __init__(self, plane: Plane) -> None:
        u = tuple(Fraction(float(x)) for x in plane.u_vector)
        v = tuple(Fraction(float(x)) for x in plane.v_vector)
        o = tuple(Fraction(float(x)) for x in plane.origin)
        uu = sum(a * a for a in u)
        uv = sum(a * b for a, b in zip(u, v))
        vv = sum(b * b for b in v)
        det = uu * vv - uv * uv
        if det <= 0:
            raise GeometryError("plane frame is degenerate")
        first = tuple((vv * a - uv * b) / det for a, b in zip(u, v))
        second = tuple((uu * b - uv * a) / det for a, b in zip(u, v))
        self.exact_rows = (first, second)
        self.exact_offset = tuple(sum(r * x for r, x in zip(row, o)) for row in self.exact_rows)
        self.rows = np.asarray([[float(r) for r in row] for row in self.exact_rows], dtype=float)
        self.offset = np.asarray([float(c) for c in self.exact_offset], dtype=float)
        self.abs_rows = np.abs(self.rows)
        self.abs_offset = np.abs(self.offset)

    def exact_uv(self, point) -> tuple[Fraction, Fraction]:
        p = tuple(x if isinstance(x, Fraction) else Fraction(float(x)) for x in point)
        return tuple(
            sum(r * x for r, x in zip(row, p)) - c for row, c in zip(self.exact_rows, self.exact_offset)
        )

    def exactly_inside(self, points, lower_limit, upper_limit) -> bool:
        """Exact test that every point's uv lies in the (double) limits."""

        lows = tuple(Fraction(float(x)) for x in lower_limit)
        highs = tuple(Fraction(float(x)) for x in upper_limit)
        for point in points:
            for value, low, high in zip(self.exact_uv(point), lows, highs):
                if not low <= value <= high:
                    return False
        return True


_ARC_MAX_SCALE = 1.0e5


def _sqrt_bounds(value: Fraction) -> tuple[float, float]:
    """Doubles ``lo <= sqrt(value) <= hi`` for a non-negative rational."""

    root = float(np.sqrt(float(value)))
    lo = hi = root
    while lo > 0.0 and Fraction(lo) * Fraction(lo) > value:
        lo = float(np.nextafter(lo, 0.0))
    while Fraction(hi) * Fraction(hi) < value:
        hi = float(np.nextafter(hi, np.inf))
    return lo, hi


def _fraction_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _exact_circumcircle(points):
    """Exact circle through three stored points (the same formula as ``arc_frame``, in rationals).

    Returns ``(h, a - h, r^2, n, |n|^2)`` with ``n = ab x ac``, or ``None`` for
    collinear points.
    """

    a, b, c = (tuple(Fraction(float(x)) for x in point) for point in points)
    ab = tuple(q - p for p, q in zip(a, b))
    ac = tuple(q - p for p, q in zip(a, c))
    normal = _fraction_cross(ab, ac)
    normal_sq = sum(x * x for x in normal)
    if normal_sq == 0:
        return None
    ab_sq = sum(x * x for x in ab)
    ac_sq = sum(x * x for x in ac)
    first = _fraction_cross(ac, normal)
    second = _fraction_cross(normal, ab)
    h = tuple(p + (ab_sq * f + ac_sq * s) / (2 * normal_sq) for p, f, s in zip(a, first, second))
    radial = tuple(p - q for p, q in zip(a, h))
    return h, radial, sum(x * x for x in radial), normal, normal_sq


def _exact_arc_disc(points, exact):
    """Conservative uv box of the exact circle through three stored points, or ``None``.

    ``uv_k(X) = uv_k(h) + R_k (X - h)`` and ``|R_k (X - h)| <= |R_k|_2 r``.
    """

    if not all(np.all(np.isfinite(np.asarray(point, dtype=float))) for point in points):
        return None
    circle = _exact_circumcircle(points)
    if circle is None:
        return None
    h, _radial, radius_sq, _normal, _normal_sq = circle
    _r_lo, r_hi = _sqrt_bounds(radius_sq)
    centre = exact.exact_uv(h)
    lower, upper = np.zeros(2), np.zeros(2)
    for k in range(2):
        _n_lo, n_hi = _sqrt_bounds(sum(x * x for x in exact.exact_rows[k]))
        reach = Fraction(r_hi) * Fraction(n_hi)
        lower[k] = -_round_up(reach - centre[k])
        upper[k] = _round_up(centre[k] + reach)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    return lower, upper


def _arc_construction_bound(start, via, end, frame):
    """How far the rounded ``arc_frame`` can sit from the exact arc of its stored points.

    The arc's definition is the exact circle through the stored doubles
    ``a`` (start), ``b`` (via) and ``c`` (end), from ``a`` through ``b`` to
    ``c``; ``model.arc_frame`` is a rounded construction of it (centre ``c~``,
    radius ``rho``, axes ``e1, e2``, sweep ``s~``).  With the exact centre ``h``
    and exact axes ``r e1^ = a - h``, ``r e2^ = n^ x (a - h)`` (``n = ab x ac``),
    the exact arc point ``X(t) = h + r (cos t e1^ + sin t e2^)`` and the frame
    point ``F(t) = c~ + rho (cos t e1 + sin t e2)`` differ componentwise by at
    most ``dev = |h - c~| + |r e1^ - rho e1| + |r e2^ - rho e2|``; every term is
    measured in rationals from the stored points (``|n|`` enclosed by
    outward-rounded square roots) and rounded up, so this is an a-posteriori
    bound of the construction error rather than an a-priori conditioning
    estimate.  The exact sweep ``S`` differs from ``s~``: an angle of a point
    ``p`` on the circle (p = b, c) moves by at most ``(pi/2) sqrt2 eps / r``
    with ``eps = 3 |dev|_1 + 8 u rho + 8 u (|p|_1 + |c~|_1)`` (centre, axis and
    radius deviations plus the rounding of the frame's own dot products),
    plus ``128 u`` for arctan2 and the wrap.  Returns ``(dev, sweep_error)``,
    or ``None`` (decline: the caller does not prove coverage) when the arc is
    ill-conditioned: a non-finite intermediate, collinear points, radius
    above 1e5 m, ``sqrt2 eps > r / 2``, ``|dev|_1 > r / 10``, or when the via
    or end angle lies within its error of the wrap or of each other (the
    frame might then sweep the complementary arc).
    """

    points = [np.asarray(item, dtype=float) for item in (start, via, end)]
    center = np.asarray(frame.center, dtype=float)
    e1 = np.asarray(frame.e1, dtype=float)
    e2 = np.asarray(frame.e2, dtype=float)
    rho = float(frame.radius)
    values = np.concatenate(points + [center, e1, e2, np.asarray((rho, float(frame.sweep)))])
    if not np.all(np.isfinite(values)) or not 0.0 < rho <= _ARC_MAX_SCALE:
        return None
    circle = _exact_circumcircle(points)
    if circle is None:
        return None
    h, radial, radius_sq, normal, normal_sq = circle
    r_lo, r_hi = _sqrt_bounds(radius_sq)
    if not (0.0 < r_lo and r_hi <= _ARC_MAX_SCALE):
        return None
    n_lo, n_hi = _sqrt_bounds(normal_sq)
    if not n_lo > 0.0:
        return None
    axis = _fraction_cross(normal, radial)
    deviation = np.zeros(3)
    for j in range(3):
        rho_e1 = Fraction(rho) * Fraction(float(e1[j]))
        rho_e2 = Fraction(rho) * Fraction(float(e2[j]))
        ends = (axis[j] / Fraction(n_hi), axis[j] / Fraction(n_lo))
        deviation[j] = (
            _round_up(abs(h[j] - Fraction(float(center[j]))))
            + _round_up(abs(radial[j] - rho_e1))
            + _round_up(max(abs(end - rho_e2) for end in ends))
        )
    deviation = deviation * (1.0 + 4.0 * _UNIT_ROUNDOFF)
    total = float(deviation.sum())
    if not (np.isfinite(total) and total <= 0.1 * r_lo):
        return None

    def angle_error(point):
        eps = 3.0 * total + 8.0 * _UNIT_ROUNDOFF * (rho + float(np.abs(point).sum()) + float(np.abs(center).sum()))
        if not np.sqrt(2.0) * eps <= 0.5 * r_lo:
            return None
        return (np.pi / 2.0) * np.sqrt(2.0) * eps / r_lo * (1.0 + 1.0e-6) + 128.0 * _UNIT_ROUNDOFF

    via_error = angle_error(points[1])
    end_error = angle_error(points[2])
    if via_error is None or end_error is None:
        return None
    theta_via = _wrapped_angle(points[1] - center, e1, e2)
    theta_end = _wrapped_angle(points[2] - center, e1, e2)
    period = 2.0 * np.pi
    if not (
        via_error < theta_via < period - via_error
        and end_error < theta_end < period - end_error
        and abs(theta_via - theta_end) > via_error + end_error
    ):
        return None
    return deviation, end_error + 64.0 * _UNIT_ROUNDOFF


def _plane_face_uv_extent(model, face_id: int, plane: Plane, slack: np.ndarray):
    """Conservative support-uv box of a face on an exact ``Plane`` support,
    and whether the stored frame provably covers the face.

    ``Plane.local_uv`` is affine, so the uv image of the face lies in the
    convex hull of the uv images of its trim curves.  Straight edges map to
    their endpoints and arcs to their exact angular extrema (as in the model's
    spatial bounds).  Bezier splines are bounded by their control polygons and
    exact curves by the interval image of their conservative ``bounds``; both
    are subdivided adaptively (de Casteljau / parameter halving) until every
    piece is proven inside ``[-slack, 1 + slack]^2``, a curve point is found
    outside it, or the piece budget is spent.  Returns ``(box, covers)``:
    ``box`` is ``None`` for a curve type without conservative bounds;
    ``covers`` is True only when coverage is proven.

    Rounding (S0g round 4).  "Proven" refers to the exact uv of the stored
    doubles (``_ExactPlaneUV``), never to rounded values: every floating uv
    bound below carries an outward pad ``E_k`` (k = u, v) that bounds its
    rounding error, and coverage is claimed only when the padded bound lies
    inside the limits or an exact rational test of exact data proves it.
    With ``g = 16 u`` (u = 2^-53 the unit roundoff), ``|R~_k|`` the absolute
    row k of the rounded ``R``, ``|c~_k|`` its offset and ``P`` a point:

    * points evaluated as ``R~ p - c~``: ``E_k(p) = g (|R~_k| |p| + |c~_k|)``
      (the rounding of ``R`` and ``c`` to nearest, a three-term dot product
      and one subtraction need at most ``5 u (1 + u)``);
    * arc points ``q~ = c + r (cos t e1 + sin t e2)`` add the world error
      ``g (|c| + r (|e1| + |e2|))`` mapped by ``|R~_k|``, plus
      ``r a_k d^2 / 2`` for the extremum missed by an angle error ``d``.
      The arc is the exact circle through its stored start/via/end points,
      not the rounded ``model.arc_frame`` (S0g round 5, Codex R4 P2): the
      construction deviation ``dev`` and sweep error of
      ``_arc_construction_bound`` add ``2 |R~_k| dev`` and enter ``d``, and
      the exact end vertices join the candidates.  An ill-conditioned arc
      (no meaningful bound, radius above 1e5 m) or a frame vector above
      1e5 m never proves coverage; such an arc is boxed by its exact disc;
    * de Casteljau pieces at depth ``m`` of a degree-``n`` spline add the
      world error ``m n u max|P|`` (each midpoint is one rounding; averaging
      does not amplify earlier errors) mapped by ``|R~_k|``;
    * exact curves bounded in the frame (``curve.transformed(to_frame)``, the
      route Codex R3 P1 broke at 1e8 m): the transformed data differ from the
      exact uv of the original data by ``dM_k |P| + dt_k + g (|T_k| |P| +
      |t_k|)`` per datum, where ``dM_k = ||T_k - R_k||_1`` and ``dt_k =
      |t_k + c_k|`` are measured exactly between the float ``to_frame`` rows
      ``(T_k, t_k)`` and ``(R_k, -c_k)``; a curve point is a combination of
      its data with absolute coefficient sum at most ``L`` (1 for NURBS
      curves and curves on NURBS supports, convex hull with positive weights,
      and for line ranges; 1 + sqrt 2 < 3 for elliptic arcs).  Curves stored
      as a 4x4 ``transform`` of a reference chart instead use ``|C~_k - (T
      M)_k| |x_ref| + dM_k |w| + dt_k`` with the composed-matrix error
      measured exactly, ``|x_ref|`` from the reference chart's bounds and
      ``|w|`` from the curve's world bounds.  Other exact curves use the
      world-box route, padded by ``E_k`` of its centre plus the rounding of
      ``|R~| half``.

    All pads are multiplied by ``_PAD_SAFETY`` = 2 for the rounding of their
    own evaluation; the returned box is padded too.  A pad that is not
    finite or exceeds ``_PAD_MEANINGFUL`` (the bound no longer means
    anything at this coordinate scale) never proves coverage: the face takes
    the fail-closed unbounded path.  Exact data are tested exactly (straight
    edge vertices, a spline's own control polygon, a line range's exact end
    points, a NURBS curve's control points), so exact-fit frames keep their
    covering decision at every scale.
    """

    face = model.faces[face_id]
    try:
        exact = _ExactPlaneUV(plane)
    except (GeometryError, ZeroDivisionError, ValueError, OverflowError):
        return None, False
    rows, offset, abs_rows, abs_offset = exact.rows, exact.offset, exact.abs_rows, exact.abs_offset
    if not (np.all(np.isfinite(rows)) and np.all(np.isfinite(offset))):
        return None, False
    # World -> (u, v, n) affine map; the normal is orthogonal to the frame, so
    # its first two coordinates are the least-squares Plane.local_uv.
    to_frame = np.eye(4)
    to_frame[:3, :3] = np.linalg.inv(np.column_stack((plane.u_vector, plane.v_vector, plane.normal)))
    to_frame[:3, 3] = -to_frame[:3, :3] @ np.asarray(plane.origin, dtype=float)
    if not np.all(np.isfinite(to_frame)):
        return None, False
    # Exact distance of the float frame map's uv rows from (R, -c).
    frame_rows = tuple(tuple(Fraction(float(x)) for x in to_frame[k]) for k in range(2))
    frame_dm = np.asarray(
        [
            _round_up(sum(abs(t - r) for t, r in zip(frame_rows[k][:3], exact.exact_rows[k])))
            for k in range(2)
        ],
        dtype=float,
    )
    frame_dt = np.asarray(
        [_round_up(abs(frame_rows[k][3] + exact.exact_offset[k])) for k in range(2)], dtype=float
    )
    frame_abs = np.abs(to_frame[:2, :3]).sum(axis=1)
    frame_abs_t = np.abs(to_frame[:2, 3])
    lower_limit = -slack
    upper_limit = 1.0 + slack
    frame_scale = float(max(np.linalg.norm(plane.u_vector), np.linalg.norm(plane.v_vector)))
    state = {"lower": np.full(2, np.inf), "upper": np.full(2, -np.inf), "covers": True, "pieces": 0}

    def uv_of(point) -> np.ndarray:
        return rows @ np.asarray(point, dtype=float) - offset

    def point_pad(point) -> np.ndarray:
        return _PAD_GAMMA * (abs_rows @ np.abs(np.asarray(point, dtype=float)) + abs_offset)

    def padded(lower, upper, pad):
        pad = _PAD_SAFETY * np.asarray(pad, dtype=float)
        return lower - pad, upper + pad, pad

    def inside(lower, upper, pad) -> bool:
        if not (np.all(np.isfinite(pad)) and np.all(pad <= _PAD_MEANINGFUL)):
            return False
        return bool(np.all(lower >= lower_limit) and np.all(upper <= upper_limit))

    def outside(uv) -> bool:
        return bool(np.any(uv < lower_limit) or np.any(uv > upper_limit))

    def include(lower, upper) -> None:
        state["lower"] = np.minimum(state["lower"], lower)
        state["upper"] = np.maximum(state["upper"], upper)

    def include_points(points, world_pads=None, extra=None, exact_points=None) -> None:
        points = [np.asarray(point, dtype=float) for point in points]
        values = np.asarray([uv_of(point) for point in points], dtype=float)
        pads = np.asarray([point_pad(point) for point in points], dtype=float)
        if world_pads is not None:
            pads = pads + np.asarray([abs_rows @ item for item in world_pads], dtype=float)
        if extra is not None:
            pads = pads + extra
        pad = pads.max(axis=0)
        lower, upper, pad = padded(values.min(axis=0), values.max(axis=0), pad)
        include(lower, upper)
        if inside(lower, upper, pad):
            return
        if exact_points is not None and exact.exactly_inside(exact_points, lower_limit, upper_limit):
            return
        state["covers"] = False

    def refine(pieces, hull, on_curve, split, exact_proof=None) -> None:
        # pieces: list of (piece, depth); hull(piece, depth) -> padded uv box
        # (lower, upper, pad); on_curve(piece) -> uv of points on the curve;
        # split(piece) -> two pieces; exact_proof(piece, depth) -> True when
        # exact data prove the piece inside.
        while pieces:
            piece, depth = pieces.pop()
            lower, upper, pad = hull(piece, depth)
            if inside(lower, upper, pad) or (
                exact_proof is not None and exact_proof(piece, depth)
            ):
                include(lower, upper)
                continue
            state["pieces"] += 1
            if (
                not state["covers"]
                or depth >= _COVER_MAX_DEPTH
                or state["pieces"] > _COVER_PIECE_BUDGET
                or any(outside(uv) for uv in on_curve(piece))
            ):
                # Not proven inside: keep the conservative hull, stop refining.
                state["covers"] = False
                include(lower, upper)
                continue
            first, second = split(piece)
            pieces.extend(((second, depth + 1), (first, depth + 1)))

    def local_pad(curve, local_curve):
        """Pad of the frame-local bounds of an exact curve, or None (world route)."""

        if local_curve is None:
            return None
        transform = getattr(curve, "transform", None)
        local_transform = getattr(local_curve, "transform", None)
        if isinstance(curve, CurveOnSurface):
            if not np.array_equal(np.asarray(local_transform, dtype=float), np.eye(4)) or not np.array_equal(
                np.asarray(transform, dtype=float), np.eye(4)
            ):
                return None
            data, coefficient = np.asarray(curve.support.controls, dtype=float), 1.0
        elif isinstance(curve, NurbsCurve):
            data, coefficient = np.asarray(curve.controls, dtype=float), 1.0
        elif isinstance(curve, LineRange):
            data, coefficient = np.asarray((curve.a, curve.b), dtype=float), 1.0
        elif isinstance(curve, EllipticArc):
            data, coefficient = np.asarray((curve.center, curve.u_vector, curve.v_vector), dtype=float), 3.0
        elif transform is not None and local_transform is not None:
            original = np.asarray(transform, dtype=float)
            composed = np.asarray(local_transform, dtype=float)
            if (
                original.shape != (4, 4)
                or composed.shape != (4, 4)
                or not np.array_equal(original[3], (0.0, 0.0, 0.0, 1.0))
                or not np.all(np.isfinite(composed))
            ):
                return None
            error = np.zeros((2, 4))
            for k in range(2):
                for i in range(4):
                    product = sum(frame_rows[k][j] * Fraction(float(original[j, i])) for j in range(3))
                    if i == 3:
                        product += frame_rows[k][3]
                    error[k, i] = _round_up(abs(Fraction(float(composed[k, i])) - product))
            try:
                world_lower, world_upper = curve.bounds(0.0, 1.0)
                world = float(np.max(np.abs(np.concatenate((world_lower, world_upper)))))
                if np.any(error[:, :3]):
                    reference = getattr(curve, "_reference_copy", None)
                    if reference is None:
                        return None
                    ref_lower, ref_upper = reference().bounds(0.0, 1.0)
                    scale = float(np.max(np.abs(np.concatenate((ref_lower, ref_upper)))))
                else:
                    scale = 0.0
            except (GeometryError, ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
                return None
            return error[:, :3].sum(axis=1) * scale + error[:, 3] + frame_dm * world + frame_dt
        else:
            return None
        scale = float(np.max(np.abs(data))) if data.size else 0.0
        return coefficient * (
            frame_dm * scale + frame_dt + _PAD_GAMMA * (frame_abs * scale + frame_abs_t)
        )

    for loop in (face.loop,) + face.holes:
        for oriented in loop:
            edge = model.edges[oriented.edge]
            curve = edge.curve
            if isinstance(curve, Straight):
                ends = (model.vertex_position(edge.start), model.vertex_position(edge.end))
                include_points(ends, exact_points=ends)
            elif isinstance(curve, Arc):
                frame = model.arc_frame(oriented.edge)
                center = np.asarray(frame.center, dtype=float)
                e1 = np.asarray(frame.e1, dtype=float)
                e2 = np.asarray(frame.e2, dtype=float)
                radius = float(frame.radius)
                sweep = float(frame.sweep)
                spread = abs_rows @ (np.abs(e1) + np.abs(e2))
                amplitude = np.hypot(rows @ e1, rows @ e2)
                # Angle error of each computed extremum: the phase from R~
                # rows (dot products of relative error g), arctan2 and the
                # sweep-membership test.
                with np.errstate(divide="ignore", invalid="ignore"):
                    phase_error = np.where(
                        amplitude > 1.0e3 * _PAD_GAMMA * spread,
                        4.0 * _PAD_GAMMA * spread / amplitude + 4.0 * _UNIT_ROUNDOFF,
                        np.pi,
                    )
                ends = (
                    model.vertex_position(edge.start),
                    model.vertex_position(edge.curve.via_vertex),
                    model.vertex_position(edge.end),
                )
                try:
                    construction = _arc_construction_bound(*ends, frame)
                except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
                    construction = None
                if construction is None:
                    # Ill-conditioned arc (or no rigorous bound): never prove
                    # coverage; box the exact circle by its full disc.
                    try:
                        disc = _exact_arc_disc(ends, exact)
                    except (ValueError, OverflowError, ZeroDivisionError):
                        disc = None
                    if disc is None:
                        return None, False
                    include(*disc)
                    state["covers"] = False
                    continue
                deviation, sweep_error = construction
                # The exact arc's extremum may sit within the sweep error of
                # an end the frame excludes (flat there: quadratic term); the
                # frame and exact arcs differ by `deviation` at equal angles,
                # once for the curve and once for the exact end vertices.
                angle_error = phase_error + sweep_error + 8.0 * _UNIT_ROUNDOFF * (1.0 + abs(sweep))
                extremum = radius * (amplitude + _PAD_GAMMA * spread) * angle_error**2 / 2.0
                extremum = extremum + 2.0 * (abs_rows @ deviation) * (1.0 + _PAD_GAMMA)
                angles = [0.0, sweep]
                for row in rows:
                    phase = float(np.arctan2(float(row @ e2), float(row @ e1)))
                    for candidate in (phase, phase + np.pi):
                        if model._angle_on_arc_sweep(candidate, frame.sweep):  # noqa: SLF001
                            angles.append(candidate)
                world_pad = _PAD_GAMMA * (np.abs(center) + radius * (np.abs(e1) + np.abs(e2)))
                include_points(
                    [center + radius * (np.cos(angle) * e1 + np.sin(angle) * e2) for angle in angles]
                    + [ends[0], ends[2]],
                    world_pads=[world_pad] * len(angles) + [np.zeros(3), np.zeros(3)],
                    extra=extremum,
                )
                if frame_scale > _ARC_MAX_SCALE:
                    # Fail closed on frames above 1e5 m (owner rule, S0g round 5).
                    state["covers"] = False
            elif isinstance(curve, Spline):
                control = np.asarray(model._spline_points(edge), dtype=float)  # noqa: SLF001
                degree = len(control) - 1
                magnitude = np.max(np.abs(control), axis=0)

                def spline_hull(points, depth, degree=degree, magnitude=magnitude):
                    values = np.asarray([uv_of(point) for point in points], dtype=float)
                    pad = np.max([point_pad(point) for point in points], axis=0)
                    pad = pad + abs_rows @ (depth * degree * _UNIT_ROUNDOFF * 1.01 * magnitude)
                    return padded(values.min(axis=0), values.max(axis=0), pad)

                refine(
                    [(control, 0)],
                    spline_hull,
                    lambda points: (uv_of(points[0]), uv_of(points[-1])),
                    lambda points: model._split_bezier(points, 0.5),  # noqa: SLF001
                    lambda points, depth: depth == 0
                    and exact.exactly_inside(points, lower_limit, upper_limit),
                )
            elif isinstance(curve, EXACT_CURVES):
                # Bound the curve in the frame's own (u, v, n) coordinates when
                # it supports an affine transform, so the uv box keeps the
                # correlation of the coordinates (a world box mapped by |pinv|
                # does not); otherwise map its world box conservatively.
                try:
                    local_curve = curve.transformed(to_frame)
                except (GeometryError, ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
                    local_curve = None
                try:
                    curve_pad = local_pad(curve, local_curve)
                except (GeometryError, ValueError, TypeError, AttributeError, ZeroDivisionError, OverflowError):
                    curve_pad = None
                if curve_pad is not None and not np.all(np.isfinite(curve_pad)):
                    curve_pad = None

                def exact_hull(interval, depth, curve=curve, local_curve=local_curve, curve_pad=curve_pad):
                    if curve_pad is not None:
                        try:
                            box_lower, box_upper = local_curve.bounds(*interval)
                        except (GeometryError, ValueError, np.linalg.LinAlgError):
                            pass
                        else:
                            return padded(
                                np.asarray(box_lower, dtype=float)[:2],
                                np.asarray(box_upper, dtype=float)[:2],
                                curve_pad,
                            )
                    box_lower, box_upper = curve.bounds(*interval)
                    box_lower = np.asarray(box_lower, dtype=float)
                    box_upper = np.asarray(box_upper, dtype=float)
                    middle = 0.5 * (box_lower + box_upper)
                    # The box lies in middle +- half (each difference rounded
                    # once); |R| <= |R~| (1 + u) and the dot product adds g.
                    half = np.maximum(box_upper - middle, middle - box_lower) * (1.0 + 4.0 * _UNIT_ROUNDOFF)
                    radius = (abs_rows @ half) * (1.0 + _PAD_GAMMA)
                    center = uv_of(middle)
                    return padded(center - radius, center + radius, point_pad(middle))

                def exact_points(interval, edge_id=oriented.edge):
                    return tuple(uv_of(point) for point in _edge_points(model, edge_id, interval))

                def exact_proof(interval, depth, curve=curve):
                    if depth != 0:
                        return False
                    if isinstance(curve, LineRange):
                        ends = (curve.exact_point(0), curve.exact_point(1))
                        return exact.exactly_inside(ends, lower_limit, upper_limit)
                    if isinstance(curve, NurbsCurve) and len(curve.controls[0]) == 3:
                        # Positive weights (validated): the curve lies in the
                        # convex hull of its control points.
                        return exact.exactly_inside(curve.controls, lower_limit, upper_limit)
                    return False

                refine(
                    [((0.0, 1.0), 0)],
                    exact_hull,
                    exact_points,
                    lambda interval: (
                        (interval[0], 0.5 * (interval[0] + interval[1])),
                        (0.5 * (interval[0] + interval[1]), interval[1]),
                    ),
                    exact_proof,
                )
            else:
                return None, False
    lower, upper = state["lower"], state["upper"]
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None, False
    return np.vstack((lower, upper)), bool(state["covers"])


def _plane_chart(model, face_id: int) -> _PlaneChart:
    """Classify a face's support chart (see ``_PlaneChart``)."""

    face = model.faces[face_id]
    plane = face.surface
    if type(plane) is not Plane:
        return _BOUNDED_CHART
    from .nurbs_sweep_chart import has_sweep_chart
    if has_sweep_chart(face):
        return _BOUNDED_CHART
    # Covering is decided with a slack far below the length tolerance:
    # clipping inside it moves a point by at most 1e-3 * tolerance.length.
    slack = 1.0e-3 * model.tolerance.length / np.asarray(
        (np.linalg.norm(plane.u_vector), np.linalg.norm(plane.v_vector)), dtype=float
    )
    box, covers = _plane_face_uv_extent(model, face_id, plane, slack)
    if covers:
        return _BOUNDED_CHART
    return _PlaneChart(True, box)


def _support_projection(
    model,
    face_id: int,
    point: np.ndarray,
    seeds: Sequence[Sequence[float]] = (),
    chart: _PlaneChart | None = None,
):
    face = model.faces[face_id]
    surface = face.surface
    if surface is None:
        raise GeometryError(f"face {face_id} has no authoritative support")
    if chart is None:
        chart = _plane_chart(model, face_id)
    # An unbounded Plane chart is never clipped: the exact orthogonal
    # projection is the answer wherever the frame happens to lie.
    bound = (lambda uv: uv) if chart.unbounded else (lambda uv: np.clip(uv, 0.0, 1.0))
    candidates: list[np.ndarray] = []
    try:
        local = model.face_support_local_uv(face_id, point)
        candidates.append(bound(np.asarray(local, dtype=float)))
    except (GeometryError, ValueError, np.linalg.LinAlgError):
        pass
    candidates.extend(bound(np.asarray(item, dtype=float)) for item in seeds)
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
            next_uv = bound(uv + delta)
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


def _curve_surface_samples(model, edge_id: int, face_id: int, count: int, chart: _PlaneChart | None = None):
    parameters = np.linspace(0.0, 1.0, count)
    points = _edge_points(model, edge_id, parameters)
    records = []
    seed = None
    if chart is None:
        chart = _plane_chart(model, face_id)
    for parameter, point in zip(parameters, points):
        seeds = () if seed is None else (seed,)
        projected, uv, residual, normal = _support_projection(model, face_id, point, seeds, chart)
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

    if isinstance(edge.curve, EllipticArc):
        curve = edge.curve
        offset = np.asarray(curve.center)-surface.origin
        u, v = np.asarray(curve.u_vector), np.asarray(curve.v_vector)
        if isinstance(surface, Plane):
            return (abs(float(offset @ surface.normal))+abs(float(u @ surface.normal))
                    + abs(float(v @ surface.normal))) <= tolerance
        if isinstance(surface, Cylinder):
            project = np.eye(3)-np.outer(surface.axis, surface.axis)
            q0, qc, qs = project @ offset, project @ u, project @ v
            coefficients = (
                float(q0 @ q0+.5*(qc @ qc+qs @ qs)-surface.radius**2),
                float(2*q0 @ qc), float(2*q0 @ qs),
                float(.5*(qc @ qc-qs @ qs)), float(qc @ qs))
            return sum(abs(value) for value in coefficients) <= tolerance*(2*surface.radius+tolerance)
        return False
    if isinstance(edge.curve, CylinderIntersectionCurve):
        if not isinstance(surface, Cylinder) or not np.array_equal(edge.curve.transform, np.eye(4)):
            return False
        for support in (edge.curve.first, edge.curve.second):
            axis = np.asarray(support.axis)
            offset = np.asarray(support.origin)-surface.origin
            if (np.linalg.norm(np.cross(axis, surface.axis)) <= model.tolerance.angular
                    and np.linalg.norm(offset-float(offset @ surface.axis)*surface.axis) <= tolerance
                    and abs(support.radius-surface.radius) <= tolerance):
                return True
        return False
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


def _refine_curve_surface_root(
    model,
    edge_id: int,
    face_id: int,
    lower: float,
    upper: float,
    policy,
    chart: _PlaneChart | None = None,
):
    if chart is None:
        chart = _plane_chart(model, face_id)
    left_point = _edge_points(model, edge_id, (lower,))[0]
    _projected, left_uv, _residual, normal = _support_projection(model, face_id, left_point, (), chart)
    left_value = float((left_point - _projected) @ normal)
    best = None
    for _ in range(max(24, policy.max_newton_iterations * 2)):
        middle = 0.5 * (lower + upper)
        point = _edge_points(model, edge_id, (middle,))[0]
        projected, uv, residual, normal = _support_projection(model, face_id, point, (left_uv,), chart)
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
    from .nurbs import CurveOnSurface, NurbsCurve
    from .support_curves import PlaneNurbsCurve
    if isinstance(curve, (NurbsCurve, CurveOnSurface, PlaneNurbsCurve)):
        # The sampled root search below is not complete for arbitrary knot
        # vectors (a spike between samples is missed); NURBS curve x face
        # qualification arrives with the certified plane x NURBS route (A4).
        return _result(
            edge, face, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="curve_surface_subdivision",
            diagnostics=("nurbs_curve_face_not_certified",),
        )
    count = 65 if isinstance(curve, Straight) else 257
    count = min(count, policy.max_trace_segments + 1)
    exact_on_support = _curve_exactly_on_support(
        model, edge.id, face.id, tolerance
    )
    try:
        chart = _plane_chart(model, face.id)
        trim_loops = _support_trim_loops(model, face.id)
        samples = (
            []
            if exact_on_support
            else _curve_surface_samples(model, edge.id, face.id, count, chart)
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
                    model, face.id, middle_point, (), chart
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
                    _support_projection(model, face.id, point, (), chart) for point in points
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
        candidates.append(
            _refine_curve_surface_root(model, edge.id, face.id, previous[0], current[0], policy, chart)
        )
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
    charts = {first_id: _plane_chart(model, first_id), second_id: _plane_chart(model, second_id)}
    projection_charts = charts
    if any(chart.unbounded for chart in charts.values()):
        # An unbounded Plane is sampled over its whole face; a covering Plane
        # target is then projected unclipped too, because those samples may
        # leave the target's own face (pairs of two covering frames keep the
        # historical clipped check bit for bit).
        projection_charts = {
            face_id: (
                _PlaneChart(True, chart.box) if type(model.faces[face_id].surface) is Plane else chart
            )
            for face_id, chart in charts.items()
        }
    for source, target in ((first_id, second_id), (second_id, first_id)):
        u_values, v_values = _chart_axes(charts[source], 17)
        parameters = np.asarray(
            tuple((float(u), float(v)) for u in u_values for v in v_values),
            dtype=float,
        )
        for uv in parameters:
            point, source_du, source_dv = _surface_point_derivatives(model, source, uv)
            projected, _target_uv, residual, target_normal = _support_projection(
                model, target, point, (), projection_charts[target]
            )
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


def _chart_axes(chart: _PlaneChart, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample axes of a support chart: the unit square, or the face's uv box."""

    if not chart.unbounded:
        parameters = np.linspace(0.0, 1.0, count)
        return parameters, parameters
    if chart.box is None:
        # Fail closed: without a conservative box the face is not covered.
        raise GeometryError("plane support chart has no conservative face uv box")
    return (
        np.linspace(float(chart.box[0][0]), float(chart.box[1][0]), count),
        np.linspace(float(chart.box[0][1]), float(chart.box[1][1]), count),
    )


def _surface_grid(model, face_id: int, count: int, chart: _PlaneChart | None = None):
    if chart is None:
        chart = _plane_chart(model, face_id)
    u_values, v_values = _chart_axes(chart, count)
    uv = np.asarray(tuple((float(u), float(v)) for v in v_values for u in u_values), dtype=float)
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


def _refine_surface_pair(
    model,
    first_id: int,
    second_id: int,
    first_uv,
    second_uv,
    policy,
    first_chart: _PlaneChart | None = None,
    second_chart: _PlaneChart | None = None,
):
    if first_chart is None:
        first_chart = _plane_chart(model, first_id)
    if second_chart is None:
        second_chart = _plane_chart(model, second_id)

    # A bounded chart is refined in its own parameters, clipped to the unit
    # square, exactly as before.  An unbounded Plane is refined unclipped in
    # its face-box chart (uv = lower + w * (upper - lower)), so the
    # minimum-norm Newton steps do not depend on the stored frame's scale.
    def chart_map(chart):
        if not chart.unbounded:
            return None
        if chart.box is None:
            return np.zeros(2), np.ones(2)
        span = np.asarray(chart.box[1] - chart.box[0], dtype=float)
        span = np.where(span > 0.0, span, 1.0)
        return np.asarray(chart.box[0], dtype=float), span

    first_map = chart_map(first_chart)
    second_map = chart_map(second_chart)

    def to_work(mapping, uv):
        uv = np.asarray(uv, dtype=float)
        return np.clip(uv, 0.0, 1.0) if mapping is None else (uv - mapping[0]) / mapping[1]

    def to_uv(mapping, work):
        return work if mapping is None else mapping[0] + work * mapping[1]

    def bound(mapping, work):
        return np.clip(work, 0.0, 1.0) if mapping is None else work

    def derivatives(face_id, mapping, work):
        point, du, dv = _surface_point_derivatives(model, face_id, to_uv(mapping, work))
        if mapping is not None:
            du, dv = du * mapping[1][0], dv * mapping[1][1]
        return point, du, dv

    first_parameters = to_work(first_map, first_uv)
    second_parameters = to_work(second_map, second_uv)
    for _ in range(policy.max_newton_iterations):
        first_point, first_du, first_dv = derivatives(first_id, first_map, first_parameters)
        second_point, second_du, second_dv = derivatives(second_id, second_map, second_parameters)
        difference = first_point - second_point
        jacobian = np.column_stack(
            (first_du, first_dv, -second_du, -second_dv)
        )
        if not np.all(np.isfinite(jacobian)) or np.linalg.matrix_rank(jacobian) < 3:
            break
        delta, *_ = np.linalg.lstsq(jacobian, -difference, rcond=None)
        next_first = bound(first_map, first_parameters + delta[:2])
        next_second = bound(second_map, second_parameters + delta[2:])
        if max(
            float(np.linalg.norm(next_first - first_parameters)),
            float(np.linalg.norm(next_second - second_parameters)),
        ) <= model.tolerance.parameter:
            first_parameters, second_parameters = next_first, next_second
            break
        first_parameters, second_parameters = next_first, next_second
    first_parameters = to_uv(first_map, first_parameters)
    second_parameters = to_uv(second_map, second_parameters)
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
    from ._translational_pairs import separated_sweep_connectors, topology_boundary_uv
    sweep_connectors = separated_sweep_connectors(model, first.id, second.id)
    if sweep_connectors == ():
        return _result(
            first, second, IntersectionKind.DISJOINT, tolerance=tolerance,
            algorithm="exact_translational_nurbs_profiles",
            diagnostics=("rational_profile_hulls_disjoint",),
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
            def boundary_path(face_id):
                result = []
                for parameter, point in zip(parameters, points):
                    uv = (topology_boundary_uv(model, face_id, edge_id, parameter)
                          if sweep_connectors is not None and edge_id in sweep_connectors
                          else None)
                    if uv is None:
                        uv = model.face_support_local_uv(face_id, point)
                    result.append(tuple(float(value) for value in uv))
                return tuple(result)
            first_path = boundary_path(first.id)
            second_path = boundary_path(second.id)
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
    try:
        first_chart = _plane_chart(model, first.id)
        second_chart = _plane_chart(model, second.id)
    except (GeometryError, ValueError, np.linalg.LinAlgError) as error:
        return _result(
            first, second, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="surface_surface_subdivision",
            diagnostics=(f"plane_support_chart_failed:{error}",),
        )
    if any(chart.unbounded and chart.box is None for chart in (first_chart, second_chart)):
        return _result(
            first, second, IntersectionKind.UNCLASSIFIED,
            tolerance=tolerance,
            algorithm="surface_surface_subdivision",
            diagnostics=("plane_support_chart_has_no_conservative_face_uv_box",),
        )
    # An unbounded Plane is sampled over its face's uv box, not over the
    # unit square of an arbitrary stored frame.
    first_uv_grid, first_points = _surface_grid(model, first.id, count, first_chart)
    second_seed_uv, second_seed_points = _surface_grid(model, second.id, 9, second_chart)
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
                    model, second.id, point, (flat_seeds[nearest],), second_chart
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
                    projected, uv_second, residual, _normal = _support_projection(
                        model, second.id, point, (), second_chart
                    )
                    world, crossing, uv_second, residual = _refine_surface_pair(
                        model,
                        first.id,
                        second.id,
                        crossing,
                        uv_second,
                        policy,
                        first_chart,
                        second_chart,
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
