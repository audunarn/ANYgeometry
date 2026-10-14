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

    def __new__(cls, start, end, derived=False):
        self = super().__new__(cls, (start, end))
        # A derived piece (a split of a stored line) has exact stations only; main's
        # float-length refusal applies to stored lines (_check_line).
        self.derived = derived
        a, b = rounded(start), rounded(end)
        slack = [4 * _EPS_FLOAT * max(abs(x), abs(y)) + _TINY for x, y in zip(a, b)]
        self.lo = tuple(min(x, y) - e for x, y, e in zip(a, b, slack))
        self.hi = tuple(max(x, y) + e for x, y, e in zip(a, b, slack))
        return self

    def length2(self) -> Fraction:
        """Exact squared length (cached)."""
        value = self.__dict__.get("_length2")
        if value is None:
            value = self.__dict__["_length2"] = distance2(self[0], self[1])
        return value

    def far_from(self, point, tolerance: float) -> bool:
        """``True`` only when ``point_parameters`` certainly returns nothing for ``point``.

        Accepted points lie within ``tol`` of the segment or of its extension
        to parameters ``[-tol, 1 + tol]``, i.e. within ``tol + tol |d_k|`` of
        the box in each coordinate ``k`` (``d`` the segment's span).
        """
        for value, lo, hi in zip(point, self.lo, self.hi):
            x = float(value)
            margin = tolerance * (1 + 1e-12) * (1 + (hi - lo)) + 4 * _EPS_FLOAT * abs(x) + _TINY
            if x < lo - margin or x > hi + margin:
                return True
        return False

    def exact_box(self):
        return (tuple(min(a, b) for a, b in zip(*self)), tuple(max(a, b) for a, b in zip(*self)))


def boxes_separated(first: "Segment", second: "Segment", tolerance: float) -> bool:
    """Exact replica of ``arrange_material``'s pair skip: some coordinate gap of the boxes exceeds ``tol``.

    Main compares the outward-rounded float boxes of the rounded pieces; the
    exact rule compares the exact boxes. A float test on the widened boxes
    settles clear separations; the rest is decided exactly.
    """
    margin = tolerance * (1 + 1e-12)
    if any(h1 < l2 - margin or h2 < l1 - margin
           for l1, h1, l2, h2 in zip(first.lo, first.hi, second.lo, second.hi)):
        return True                                          # widened boxes already separated
    (l1, h1), (l2, h2) = first.exact_box(), second.exact_box()
    bound = Fraction(tolerance)
    return any(b < c - bound or d < a - bound for a, b, c, d in zip(l1, h1, l2, h2))


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
    return Segment(at(segment, a), at(segment, b), derived=True)


def _difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def distance2(a, b) -> Fraction:
    d = _difference(a, b)
    return exact_dot(d, d)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _check_line(segment: Segment, curve_floats=None):
    """``_point_parameters``' refusal: an exactly degenerate line, or (stored lines only) a
    zero float squared length as main computes it on the stored doubles. Derived pieces
    (exact splits, kept down to any length by deferred removal, S2) are refused only when
    exactly degenerate: their float rounding is not data."""
    direction = _difference(segment[1], segment[0])
    if exact_dot(direction, direction) == 0:
        raise GeometryError("degenerate arrangement line")
    if not getattr(segment, "derived", False):
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
    direction = _check_line(segment)
    length2 = exact_dot(direction, direction)
    endpoints = tuple(t for t in (Fraction(0), Fraction(1)) if distance2(at(segment, t), point) <= tol2)
    parameter = exact_dot(_difference(point, segment[0]), direction) / length2
    roots = ()
    if -tolerance_exact <= parameter <= 1 + tolerance_exact and distance2(at(segment, parameter), point) <= tol2:
        roots = (min(Fraction(1), max(Fraction(0), parameter)),)
    values = []
    for value in (*endpoints, *roots):
        # S1a (T-b/D8 design, slice E1): a root snaps to an accepted endpoint when the
        # sub-piece between them is no longer than ``tol`` (main: within 128 eps in
        # parameter space, i.e. 128 eps * length, independent of the tolerance).
        snapped = next((t for t in endpoints if (value - t) ** 2 * length2 <= tol2), value)
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


def _role_key(segment: Segment) -> tuple:
    """The canonical operand-role key of a derived piece (T7, owner Q-h): exact squared length,
    then the exact station tuples."""
    return segment.length2(), segment[0], segment[1]


def junctions(first: Segment, second: Segment, tolerance: float) -> tuple:
    """``curve_junctions(LinePath, LinePath)`` on exact stations, symmetric in its operands.

    Main's candidates: the four endpoint ``point_parameters`` tests and the
    line-curve junctions of ``line_curve_junctions``. The line role is assigned
    by the canonical data key (S1b): the segment with the larger exact squared
    length, ties by exact station tuples -- so ``junctions(a, b)`` is
    ``junctions(b, a)`` with each pair swapped. A segment no longer than
    ``tol`` never takes the line role; when both are, the line-curve step is
    skipped and the endpoint tests decide (S1c), so ``_line_line``'s
    short-line refusal is unreachable here.
    """
    candidates = []
    for t in (Fraction(0), Fraction(1)):
        candidates.extend((t, s) for s in point_parameters(second, at(first, t), tolerance))
        candidates.extend((s, t) for s in point_parameters(first, at(second, t), tolerance))
    tol2 = Fraction(tolerance) ** 2
    first_length, second_length = first.length2(), second.length2()
    if max(first_length, second_length) > tol2:
        if (first_length > second_length
                or (first_length == second_length and _role_key(first) >= _role_key(second))):
            candidates.extend(_line_line(first, second, tolerance))
        else:
            candidates.extend((a, b) for b, a in _line_line(second, first, tolerance))
    return tuple(sorted(dict.fromkeys(candidates)))


def merge_parameters(values, tolerance: float, segment: Segment | None) -> list:
    """The sorted, exactly distinct split parameters of a straight piece (S2, deferred removal).

    Main's ``_parameters`` merged parameters closer than 32 eps in parameter
    space (D8) or bounding a sub-piece no longer than ``tol``; on one path that
    merged stations which vertex identity kept distinct on another (lost
    crossings). Nothing is merged here: sub-pieces no longer than ``tol`` are
    removed once, after global vertex identity (S4 in ``arrange_material``).
    """
    return sorted(dict.fromkeys(Fraction(value) for value in values))


class ExactVertexLookup:
    """Lowest vertex index within ``tolerance`` of a query, as ``_VertexLookup``.

    Each vertex keeps its exact station (straight pieces) or ``None`` (curved
    pieces: their float evaluation is the position, as in main). Two exact
    stations are compared exactly (squared distance); any comparison involving
    a curved piece's point keeps main's float rule ``norm(a - b) <= tol`` on
    the float positions. A float grid (cell ``2 tol``) preselects candidates
    under ``_VertexLookup``'s conditions plus: the rounding of exact stations is
    at most ``tol / 4``; otherwise every vertex is scanned from then on.
    """

    def __init__(self, tolerance: float):
        self.tolerance, self.tol2 = tolerance, Fraction(tolerance) ** 2
        self.exact: list = []
        self.floats: list = []
        self.grid = {} if 1e-100 < tolerance < 1e100 else None

    def _key(self, values):
        # Main's eligibility (_VertexLookup._key): float64 3-arrays with
        # underflow ignored; anything else (e.g. float32 curve evaluations,
        # whose norms can underflow) switches to the scan.
        if (type(values) is not np.ndarray or values.shape != (3,) or values.dtype != np.float64
                or np.geterr()["under"] != "ignore"):
            return None
        values = [float(value) for value in values]
        if not all(math.isfinite(value) and abs(value) < 1e150 for value in values):
            return None
        if max(abs(value) for value in values) * 2.**-52 > .25 * self.tolerance:
            return None
        return tuple(math.floor(value / (2. * self.tolerance)) for value in values)

    def _near(self, index, exact, floats) -> bool:
        if exact is not None and self.exact[index] is not None:
            return distance2(exact, self.exact[index]) <= self.tol2
        return bool(np.linalg.norm(floats - self.floats[index]) <= self.tolerance)

    def find(self, exact, floats):
        key = self._key(floats) if self.grid is not None else None
        if key is None:
            self.grid = None
            return next((index for index in range(len(self.floats)) if self._near(index, exact, floats)), None)
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for index in self.grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if (found is None or index < found) and self._near(index, exact, floats):
                            found = index
        return found

    def add(self, exact, floats):
        if self.grid is not None:
            key = self._key(floats)
            if key is None:
                self.grid = None
            else:
                self.grid.setdefault(key, []).append(len(self.floats))
        self.exact.append(exact)
        self.floats.append(floats)


__all__ = ["ExactVertexLookup", "Segment", "at", "boxes_separated", "distance2", "junctions", "merge_parameters", "point_parameters",
           "rounded", "segment_of", "sub", "to_line"]
