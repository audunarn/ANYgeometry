"""Exact planar charts and line clipping (robust-predicate slice R1a).

Every decision here is made in exact rational arithmetic on stored doubles
(``exact_predicates`` T1/T4), so it is independent of floating-point rounding:

* :class:`ExactPlane` is the affine chart ``origin + u U + v V`` of a ``Plane``
  record; ``uv(X)`` gives the exact coordinates of the orthogonal projection
  of ``X`` (Cramer with ``N = U x V``), the rational counterpart of
  ``Plane.local_uv``'s least-squares solve.
* :class:`ExactLine` is ``anchor + s direction`` with exact anchor and
  direction; :func:`plane_plane_line` builds the intersection line of two
  plane records exactly (generated-curve definition, inventory A7).
* :func:`clip_line_to_region` clips a line in a chart against a straight-edged
  region (outer loop minus holes): exact crossing parameters, every gap
  classified by exact point-in-region, closed intervals plus isolated touch
  points, as ``polygon.intersection(line)`` would report them for the exact
  inputs.

Floats leave this module only as coordinates (``to_float``).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .errors import GeometryError
from .exact_predicates import Outcome, exact, exact_cross, exact_dot, exact_vector, point_in_polygon

__all__ = [
    "ExactLine",
    "ExactPlane",
    "clip_line_to_region",
    "plane_plane_line",
    "point_in_region",
    "region_is_simple",
    "to_float",
]


def to_float(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _scale(a, k):
    return tuple(x * k for x in a)


@dataclass(frozen=True)
class ExactPlane:
    """The exact affine chart of a ``Plane`` record (origin, u_vector, v_vector)."""

    origin: tuple
    u_vector: tuple
    v_vector: tuple

    @classmethod
    def from_plane(cls, plane) -> "ExactPlane":
        return cls(exact_vector(plane.origin), exact_vector(plane.u_vector), exact_vector(plane.v_vector))

    def __post_init__(self):
        normal = exact_cross(self.u_vector, self.v_vector)
        scale = exact_dot(normal, normal)
        if scale == 0:
            raise GeometryError("plane parameter vectors must be independent")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_dual_u", exact_cross(self.v_vector, normal))
        object.__setattr__(self, "_dual_v", exact_cross(normal, self.u_vector))

    def uv(self, point) -> tuple[Fraction, Fraction]:
        """Exact chart coordinates of the orthogonal projection of ``point``."""
        offset = _sub(exact_vector(point), self.origin)
        return exact_dot(offset, self._dual_u) / self._scale, exact_dot(offset, self._dual_v) / self._scale

    def uv_direction(self, vector) -> tuple[Fraction, Fraction]:
        vector = exact_vector(vector)
        return exact_dot(vector, self._dual_u) / self._scale, exact_dot(vector, self._dual_v) / self._scale

    def world(self, u, v) -> tuple:
        return _add(self.origin, _add(_scale(self.u_vector, exact(u)), _scale(self.v_vector, exact(v))))

    def offset(self) -> Fraction:
        """``N . origin`` (the plane is ``N . X = offset``)."""
        return exact_dot(self.normal, self.origin)


@dataclass(frozen=True)
class ExactLine:
    """``anchor + s * direction`` with exact anchor and non-zero exact direction."""

    anchor: tuple
    direction: tuple

    @classmethod
    def from_floats(cls, point, direction) -> "ExactLine":
        return cls(exact_vector(point), exact_vector(direction))

    def __post_init__(self):
        if all(value == 0 for value in self.direction):
            raise GeometryError("degenerate line direction")

    def at(self, s) -> tuple:
        return _add(self.anchor, _scale(self.direction, exact(s)))

    def squared_speed(self) -> Fraction:
        return exact_dot(self.direction, self.direction)

    def plane_parameter(self, plane: ExactPlane) -> Fraction | None:
        """Parameter of the crossing with ``plane``, or ``None`` when exactly parallel."""
        denominator = exact_dot(plane.normal, self.direction)
        if denominator == 0:
            return None
        return exact_dot(plane.normal, _sub(plane.origin, self.anchor)) / denominator


def plane_plane_line(first: ExactPlane, second: ExactPlane) -> ExactLine | None:
    """The exact intersection line of two planes, or ``None`` when exactly parallel.

    Direction ``N1 x N2``; anchor = the point of the line closest to the first
    plane's origin (exact Cramer solution of ``N1.x = N1.O1, N2.x = N2.O2,
    D.x = D.O1``).
    """
    direction = exact_cross(first.normal, second.normal)
    if all(value == 0 for value in direction):
        return None
    rows = (first.normal, second.normal, direction)
    right = (first.offset(), second.offset(), exact_dot(direction, first.origin))
    determinant = exact_dot(rows[0], exact_cross(rows[1], rows[2]))
    # Cramer: x = (r0 (n2 x d) + r1 (d x n1) + r2 (n1 x n2)) / det
    terms = (exact_cross(rows[1], rows[2]), exact_cross(rows[2], rows[0]), exact_cross(rows[0], rows[1]))
    anchor = tuple(sum(right[i] * terms[i][k] for i in range(3)) / determinant for k in range(3))
    return ExactLine(anchor, direction)


def _cross2(a, b) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]


def point_in_region(point, loops: Sequence[Sequence]) -> Outcome:
    """``ABOVE`` inside, ``EQUAL`` on a boundary, ``BELOW`` outside; ``loops[0]`` outer, the rest holes.

    Prerequisite (checked by callers with :func:`region_is_simple`): simple,
    non-degenerate loops whose holes lie inside the outer loop.
    """
    if not loops:
        raise GeometryError("a region needs at least its outer loop")
    outer = point_in_polygon(point, loops[0])
    if outer is not Outcome.ABOVE:
        return outer
    for hole in loops[1:]:
        inside_hole = point_in_polygon(point, hole)
        if inside_hole is Outcome.EQUAL:
            return Outcome.EQUAL
        if inside_hole is Outcome.ABOVE:
            return Outcome.BELOW
    return Outcome.ABOVE


def clip_line_to_region(anchor_uv, direction_uv, loops: Sequence[Sequence]):
    """Exact intersection of the chart line ``anchor + s direction`` with a closed region.

    ``loops`` are straight-edged chart polygons (exact rational vertices), the
    first the outer boundary, the rest holes (prerequisite: :func:`region_is_simple`).
    Returns ``(intervals, points)``: sorted closed parameter intervals
    ``(s0, s1)``, ``s0 < s1``, where the line is in the region (including runs
    along a boundary edge), split at every boundary point the line passes (as
    a polygon overlay nodes them), and sorted isolated parameters where it only
    touches the region.
    """
    a = tuple(exact(value) for value in anchor_uv)
    d = tuple(exact(value) for value in direction_uv)
    if d[0] == 0 and d[1] == 0:
        raise GeometryError("line direction vanishes in the chart")
    speed = d[0] * d[0] + d[1] * d[1]
    events: set[Fraction] = set()
    for loop in loops:
        vertices = [tuple(exact(value) for value in vertex) for vertex in loop]
        for p, q in zip(vertices, vertices[1:] + vertices[:1]):
            edge = (q[0] - p[0], q[1] - p[1])
            offset = (p[0] - a[0], p[1] - a[1])
            denominator = _cross2(d, edge)
            if denominator != 0:
                w = _cross2(offset, d) / denominator
                if 0 <= w <= 1:
                    events.add(_cross2(offset, edge) / denominator)
            elif _cross2(offset, d) == 0:                         # collinear edge: both ends are events
                events.add((offset[0] * d[0] + offset[1] * d[1]) / speed)
                events.add(((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / speed)
    ordered = sorted(events)

    def member(s) -> bool:
        return point_in_region((a[0] + s * d[0], a[1] + s * d[1]), loops) is not Outcome.BELOW

    # Every event is a boundary point of the line. As the polygon overlay does,
    # a run of included gaps is split at each event it contains (e.g. where it
    # passes a reflex vertex of the boundary), so each interval is one gap.
    intervals: list[tuple[Fraction, Fraction]] = [
        (left, right) for left, right in zip(ordered, ordered[1:]) if member((left + right) / 2)]
    covered = {end for interval in intervals for end in interval}
    points = [s for s in ordered if s not in covered and member(s)]
    return intervals, points


def _segments_cross(p, q, r, t) -> bool:
    """Whether closed segments pq and rt share a point (exact)."""
    def orientation(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (value > 0) - (value < 0)

    def on(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    o1, o2, o3, o4 = orientation(p, q, r), orientation(p, q, t), orientation(r, t, p), orientation(r, t, q)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on(p, q, r)) or (o2 == 0 and on(p, q, t))
            or (o3 == 0 and on(r, t, p)) or (o4 == 0 and on(r, t, q)))


def region_is_simple(loops: Sequence[Sequence]) -> bool:
    """Exact validity of a straight-edged region: every loop has at least three
    vertices, no two edges meet except consecutive edges of one loop at their
    shared vertex, and every loop has non-zero exact area."""
    edges = []
    for index, loop in enumerate(loops):
        vertices = [tuple(exact(value) for value in vertex) for vertex in loop]
        if len(vertices) < 3:
            return False
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(vertices, vertices[1:] + vertices[:1]))
        if area == 0:
            return False
        count = len(vertices)
        edges.extend((index, k, count, vertices[k], vertices[(k + 1) % count]) for k in range(count))
    for i, (loop_a, k_a, count_a, p, q) in enumerate(edges):
        for loop_b, k_b, count_b, r, t in edges[i + 1:]:
            if loop_a == loop_b and (k_b - k_a) % count_a in (1, count_a - 1):
                shared = q if (k_b - k_a) % count_a == 1 else p
                other_a = p if shared is q else q
                other_b = t if shared is r else r
                # adjacent edges may only share their common vertex (no overlap, no fold-back)
                cross = (other_a[0] - shared[0]) * (other_b[1] - shared[1]) - (other_a[1] - shared[1]) * (other_b[0] - shared[0])
                dot = (other_a[0] - shared[0]) * (other_b[0] - shared[0]) + (other_a[1] - shared[1]) * (other_b[1] - shared[1])
                if cross == 0 and dot > 0:
                    return False
                continue
            if _segments_cross(p, q, r, t):
                return False
    return True
