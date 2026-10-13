"""Exact stations for straight arrangement pieces (robust-predicate slice R1b, step 2a).

On a Plane support every straight piece of a material arrangement is held as a
pair of exact rational end positions (stations on its source line). The
functions here are exact replicas of the float rules of
``arrangement_geometry.point_parameters``/``curve_junctions`` (line x line) and
of ``material_arrangement._parameters``: the same candidates, the same
tolerance tests (compared through squares) and the same snapping windows, in
exact arithmetic on the stored doubles. Float ``LinePath`` pieces are emitted
with correctly rounded end positions.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .arrangement_geometry import LinePath
from .errors import GeometryError
from .exact_predicates import exact_dot, exact_vector

_EPS = Fraction(float(np.finfo(float).eps))

class Segment(tuple):
    """``(start, end)``: exact 3-tuples of Fractions, plus a float box for filtering.

    ``lo``/``hi`` are the rounded end coordinates' box widened by their own
    rounding (``4 eps |x|`` per coordinate covers the correct rounding of the
    exact values and the float comparisons), so a point whose rounded position
    lies outside the box widened by ``tol`` is certainly farther than ``tol``.
    """

    def __new__(cls, start, end):
        self = super().__new__(cls, (start, end))
        a, b = rounded(start), rounded(end)
        slack = [4 * _EPS_FLOAT * max(abs(x), abs(y)) + _TINY for x, y in zip(a, b)]
        self.lo = tuple(min(x, y) - e for x, y, e in zip(a, b, slack))
        self.hi = tuple(max(x, y) + e for x, y, e in zip(a, b, slack))
        return self

    def far_from(self, point, tolerance: float) -> bool:
        """``True`` only when ``point`` is certainly farther than ``tolerance`` from the segment."""
        for value, lo, hi in zip(point, self.lo, self.hi):
            x = float(value)
            margin = tolerance * (1 + 1e-12) + 4 * _EPS_FLOAT * abs(x) + _TINY
            if x < lo - margin or x > hi + margin:
                return True
        return False


_EPS_FLOAT = float(np.finfo(float).eps)
_TINY = 2.**-1070                                         # covers subnormal rounding


def segment_of(curve: LinePath) -> Segment:
    """The exact stations of a stored line's ends."""
    return Segment(exact_vector(curve.start), exact_vector(curve.end))


def at(segment: Segment, t) -> tuple:
    start, end = segment
    return tuple(a + t * (b - a) for a, b in zip(start, end))


def rounded(point) -> tuple:
    """Correctly rounded float position of an exact point (the canonical rounding)."""
    return tuple(float(value) for value in point)


def to_line(segment: Segment) -> LinePath:
    return LinePath(rounded(segment[0]), rounded(segment[1]))


def sub(segment: Segment, a, b) -> Segment:
    return Segment(at(segment, a), at(segment, b))


def _difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def distance2(a, b) -> Fraction:
    d = _difference(a, b)
    return exact_dot(d, d)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _check_line(segment: Segment, curve_floats=None):
    """``_point_parameters``' refusal: zero float squared length (or exactly degenerate)."""
    direction = _difference(segment[1], segment[0])
    if exact_dot(direction, direction) == 0:
        raise GeometryError("degenerate arrangement line")
    floats = np.asarray(rounded(segment[1])) - np.asarray(rounded(segment[0]))
    if float(floats @ floats) == 0:
        raise GeometryError("degenerate arrangement line")
    return direction


def point_parameters(segment: Segment, point, tolerance: float) -> tuple:
    """Exact replica of ``point_parameters(LinePath, point, tolerance=...)``."""
    if segment.far_from(point, tolerance):
        _check_line(segment)                           # the float route refuses first, whatever the point
        return ()
    tol2 = Fraction(tolerance) ** 2
    tolerance_exact = Fraction(tolerance)
    endpoints = tuple(t for t in (Fraction(0), Fraction(1)) if distance2(at(segment, t), point) <= tol2)
    direction = _check_line(segment)
    length2 = exact_dot(direction, direction)
    parameter = exact_dot(_difference(point, segment[0]), direction) / length2
    roots = ()
    if -tolerance_exact <= parameter <= 1 + tolerance_exact and distance2(at(segment, parameter), point) <= tol2:
        roots = (min(Fraction(1), max(Fraction(0), parameter)),)
    values = []
    for value in (*endpoints, *roots):
        snapped = next((t for t in endpoints if abs(value - t) <= 128 * _EPS), value)
        if snapped not in values:
            values.append(snapped)
    return tuple(values)


def _plane_roots(segment: Segment, normal, base, tolerance: float):
    """Exact replica of ``plane_roots(LinePath, unit(normal), unit(normal) . base)``.

    ``normal`` is exact and unnormalised; ``|n . x - d| <= tol`` for the unit
    normal is decided as ``(N . x - D)^2 <= tol^2 |N|^2``; the root is
    scale-invariant.
    """
    bound2 = Fraction(tolerance) ** 2 * exact_dot(normal, normal)
    start = exact_dot(normal, _difference(segment[0], base))
    delta = exact_dot(normal, _difference(segment[1], segment[0]))
    if start * start <= bound2 and (start + delta) ** 2 <= bound2:
        return None
    if delta == 0:
        return ()
    root = -start / delta
    tolerance_exact = Fraction(tolerance)
    return (min(Fraction(1), max(Fraction(0), root)),) if -tolerance_exact <= root <= 1 + tolerance_exact else ()


def _line_line(line: Segment, curve: Segment, tolerance: float) -> list:
    """Exact replica of ``line_curve_junctions(LinePath, LinePath)``."""
    direction = _difference(line[1], line[0])
    if exact_dot(direction, direction) <= Fraction(tolerance) ** 2:
        raise GeometryError("degenerate arrangement line")
    magnitudes = [abs(value) for value in direction]
    axis = magnitudes.index(min(magnitudes))          # np.argmin: first minimum
    coordinate = tuple(Fraction(int(k == axis)) for k in range(3))
    normal = _cross(direction, coordinate)
    other = _cross(direction, normal)
    first_roots = _plane_roots(curve, normal, line[0], tolerance)
    second_roots = _plane_roots(curve, other, line[0], tolerance)
    if first_roots is None and second_roots is None:
        roots = (Fraction(0), Fraction(1), *point_parameters(curve, line[0], tolerance),
                 *point_parameters(curve, line[1], tolerance))
    else:
        roots = (*(first_roots or ()), *(second_roots or ()))
    result = []
    for parameter in roots:
        point = at(curve, parameter)
        for line_parameter in point_parameters(line, point, tolerance):
            result.append((line_parameter, parameter))
    return result


def junctions(first: Segment, second: Segment, tolerance: float) -> tuple:
    """Exact replica of ``curve_junctions(LinePath, LinePath)`` (sorted, unique pairs)."""
    candidates = []
    for t in (Fraction(0), Fraction(1)):
        candidates.extend((t, s) for s in point_parameters(second, at(first, t), tolerance))
        candidates.extend((s, t) for s in point_parameters(first, at(second, t), tolerance))
    candidates.extend(_line_line(first, second, tolerance))
    return tuple(sorted(dict.fromkeys(candidates)))


def merge_parameters(values, tolerance: float, segment: Segment | None) -> list:
    """Exact replica of ``material_arrangement._parameters`` for a straight piece.

    Numerical duplicates within 32 eps merge; otherwise two parameters merge
    when the sub-piece between them is no longer than ``tolerance`` (main
    measured the diagonal of its rounded, outward-widened bounding box; the
    exact length is that diagonal without rounding).
    """
    values = sorted(Fraction(value) for value in values)
    tol2 = Fraction(tolerance) ** 2
    length2 = None if segment is None else distance2(segment[0], segment[1])
    result = []
    for value in values:
        duplicate = bool(result and value - result[-1] <= 32 * _EPS)
        if result and not duplicate and length2 is not None:
            duplicate = (value - result[-1]) ** 2 * length2 <= tol2
        if not result or not duplicate:
            result.append(value)
        elif value == 1:
            result[-1] = value
    return result


class ExactVertexLookup:
    """Lowest index ``i`` with ``|point - vertices[i]| <= tolerance`` on exact positions.

    A float grid (cell ``2 tol``, the neighbouring cells) only preselects
    candidates when the rounding of a position cannot move it out of them: the
    coordinates are finite, below ``1e150``, their rounding at most ``tol/4``,
    and the tolerance moderate; otherwise every vertex is scanned (from then
    on). The decision is always the exact squared distance.
    """

    def __init__(self, tolerance: float):
        self.tolerance, self.tol2 = tolerance, Fraction(tolerance) ** 2
        self.points: list[tuple] = []
        self.grid = {} if 1e-100 < tolerance < 1e100 else None

    def _key(self, point):
        values = rounded(point)
        if not all(math.isfinite(value) and abs(value) < 1e150 for value in values):
            return None
        # Rounding moves a coordinate by at most |x| eps; a true neighbour
        # (exact distance <= tol) stays within the +-1 cells while that is
        # <= tol / 4 (cell 2 tol).
        if max(abs(value) for value in values) * 2.**-52 > .25 * self.tolerance:
            return None
        return tuple(math.floor(value / (2. * self.tolerance)) for value in values)

    def find(self, point):
        key = self._key(point) if self.grid is not None else None
        if key is None:
            self.grid = None
            return next((index for index, candidate in enumerate(self.points)
                         if distance2(point, candidate) <= self.tol2), None)
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for index in self.grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if (found is None or index < found) and distance2(point, self.points[index]) <= self.tol2:
                            found = index
        return found

    def add(self, point):
        if self.grid is not None:
            key = self._key(point)
            if key is None:
                self.grid = None
            else:
                self.grid.setdefault(key, []).append(len(self.points))
        self.points.append(point)


__all__ = ["ExactVertexLookup", "at", "distance2", "junctions", "merge_parameters", "point_parameters",
           "rounded", "segment_of", "sub", "to_line"]
