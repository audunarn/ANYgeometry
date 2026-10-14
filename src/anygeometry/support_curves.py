"""Plane x NURBS section curves (generality slice B1-P2, architecture 2.1).

``PlaneNurbsCurve(plane, surface, path)`` is one path of the certified section
topology of ``plane_nurbs`` (slice P1). Its definition is the exact zero set;
the record holds the plane, the surface, the path index and a ``domain`` in the
path parameter (``subcurve``). The topology is recomputed exactly from that
definition and cached.

Parameterisation: the path's ``m`` arcs get equal path-parameter lengths;
arc ``k`` (a regular box with slicing coordinate ``s`` and dependent ``x``)
runs from its first root to its second, ``s`` linear in the arc parameter
between the roots' ``s``-coordinates (the arc's ends *are* the roots), and
``x(s)`` is the unique zero of ``G(s, .)`` in the box (``G`` is strictly
monotone in ``x`` there, P1). Arc ends coincide with the shared roots, so the
curve is continuous by definition.

* ``evaluate``/``derivative``: float evaluation (vectorised bisection on the
  box's Bernstein ``G``, implicit differentiation ``x' = -G_s / G_x``).
* ``chart_enclosure(t)``: an exact enclosure ``(s, [x_lo, x_hi])`` of the
  curve point by exact bisection of the rational ``G`` (the certificate a
  decision may use; architecture 4.3 uses ``delta`` from interval Newton, here
  the enclosure width is the ``delta``).
* ``bounds``: the support's sub-patch enclosure of every arc's chart
  rectangle (``s``-subrange of the parameter range x the box's ``x``-range):
  sound for the exact curve, widened for float evaluation (A3a).
* ``end_box(end)``: the world enclosure of an end root (``add_curve``'s end test).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math

import numpy as np

from .errors import GeometryError
from .nurbs import NurbsSurface, _float_down, _float_up
from .plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section

_SECTIONS: dict = {}
_EPS = float(np.finfo(float).eps)


def _plane_record(plane) -> tuple:
    return tuple(tuple(float(value) for value in vector) for vector in (plane.origin, plane.u_vector, plane.v_vector))


@dataclass(frozen=True)
class _Arc:
    box: object
    slicing: int                    # 0: s = u, x = v; 1: s = v, x = u (knot coordinates)
    s0: Fraction                    # exact slicing limits: a root's s on a constant-s edge, else the
    s1: Fraction                    # midpoint of its isolating interval (the arc's ends ARE the roots)
    first: object                   # Root records at the arc's ends
    second: object


@dataclass(frozen=True)
class PlaneNurbsCurve:
    plane_record: tuple             # (origin, u_vector, v_vector) as float tuples
    surface: NurbsSurface
    path: int
    domain: tuple = (0.0, 1.0)

    # ---------------------------------------------------------- construction and topology

    @classmethod
    def from_plane(cls, plane, surface: NurbsSurface, path: int = 0) -> "PlaneNurbsCurve":
        curve = cls(_plane_record(plane), surface, int(path))
        curve._arcs()                                      # certify now (refusals surface here)
        return curve

    @property
    def plane(self):
        from .surfaces import Plane
        return Plane(*self.plane_record)

    @property
    def dimension(self) -> int:
        return 3

    def _arcs(self) -> tuple:
        key = (self.plane_record, self.surface, self.path)
        cached = _SECTIONS.get(key)
        if cached is not None:
            return cached
        section = plane_nurbs_section(self.plane, self.surface)
        if section.coincident or not 0 <= self.path < len(section.paths):
            raise PlaneNurbsNotCertified("no such section path")
        roots = section.roots
        arcs = []
        for box_index, first, second in section.paths[self.path].arcs:
            box = section.boxes[box_index]
            slicing = 0 if box.kind == "v" else 1

            def s_of(root):
                if root.axis == slicing:
                    return root.value
                return (root.lo + root.hi) / 2

            arcs.append(_Arc(box, slicing, s_of(roots[first]), s_of(roots[second]), roots[first], roots[second]))
        result = tuple(arcs)
        if len(_SECTIONS) >= 64:
            _SECTIONS.clear()
        _SECTIONS[key] = result
        return result

    # ---------------------------------------------------------- parameter maps

    def _path_parameter(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        a, b = self.domain
        return np.clip((1.0 - t) * a + t * b, min(a, b), max(a, b))

    def _locate(self, path_t):
        count = len(self._arcs())
        scaled = np.asarray(path_t, dtype=float) * count
        index = np.clip(np.floor(scaled).astype(int), 0, count - 1)
        return index, np.clip(scaled - index, 0.0, 1.0), count

    def _chart_ranges(self):
        surface = self.surface
        return tuple(Fraction(value) for value in surface.u_range), tuple(Fraction(value) for value in surface.v_range)

    def _knot_to_chart(self, u, v):
        (r0, r1), (q0, q1) = (tuple(map(float, item)) for item in self._chart_ranges())
        return (np.asarray(u) - r0) / (r1 - r0), (np.asarray(v) - q0) / (q1 - q0)

    # ---------------------------------------------------------- evaluation

    # ---------------------------------------------------------- exact evaluation (owner option a)

    @staticmethod
    def _root_rect(root):
        if root.axis == 0:
            return (root.value, root.value), (root.lo, root.hi)
        return (root.lo, root.hi), (root.value, root.value)

    def _plane_slab(self):
        """``(m, d, mm, limit_squared, limit)``: plane ``m . X = d``; a defined point ``P`` has
        ``(m . P - d)**2 <= limit_squared`` (within ``2**-50`` of the surface's magnitude of the
        plane), and ``limit >= sqrt(limit_squared)`` (rational, for the distance bound)."""
        key = ("slab", self.plane_record, self.surface)
        cached = _SECTIONS.get(key)
        if cached is not None:
            return cached
        origin, u_vector, v_vector = ([Fraction(value) for value in vector] for vector in self.plane_record)
        m = (u_vector[1] * v_vector[2] - u_vector[2] * v_vector[1], u_vector[2] * v_vector[0] - u_vector[0] * v_vector[2],
             u_vector[0] * v_vector[1] - u_vector[1] * v_vector[0])
        mm = sum(value * value for value in m)
        d = sum(a * b for a, b in zip(m, origin))
        magnitude = max([1.0] + [abs(float(value)) for value in origin]
                        + [abs(float(value)) for row in self.surface.controls for point in row for value in point])
        delta = Fraction(magnitude) / 2 ** 50
        # rational upper bound of sqrt(mm): (isqrt(n d) + 1) / d for mm = n / d
        root_up = Fraction(math.isqrt(mm.numerator * mm.denominator) + 1, mm.denominator)
        result = (m, d, mm, delta * delta * mm, delta * root_up)
        if len(_SECTIONS) >= 64:
            _SECTIONS.clear()
        _SECTIONS[key] = result
        return result

    def _world(self, knot_point):
        """Exact world point of a knot point, and its signed plane value ``m . P - d``."""
        m, d, _mm, _limit, _l = self._plane_slab()
        cu, cv = self._exact_chart(knot_point)
        world = self.surface.exact_point(cu, cv, _surface_net(self.surface))
        return world, sum(a * b for a, b in zip(m, world)) - d

    @staticmethod
    def _root_rect(root):
        if root.axis == 0:
            return (root.value, root.value), (root.lo, root.hi)
        return (root.lo, root.hi), (root.value, root.value)

    _MAX_BISECTIONS = 1100

    def _refine(self, rect, line_point, lo, hi, side_lo, side_hi):
        """Bisect ``[lo, hi]`` on a knot line (``line_point(x)`` the knot point) until the midpoint's
        world point lies in the plane slab; ``side_lo``/``side_hi`` the end sides (``None`` unknown).

        The point is the midpoint (a deterministic definition); exact zeros end it.
        Refuses when the slab is not reached (the line's sign change is lost).
        """
        _m, _d, _mm, limit, _l = self._plane_slab()
        for _ in range(self._MAX_BISECTIONS):
            middle = (lo + hi) / 2
            point = line_point(middle)
            world, side = self._world(point)
            if side * side <= limit:
                return rect(lo, hi), point, world
            if side_lo is None or side_hi is None:
                side_lo, side_hi = self._world(line_point(lo))[1], self._world(line_point(hi))[1]
                if side_lo == 0:
                    return rect(lo, lo), line_point(lo), self._world(line_point(lo))[0]
                if side_hi == 0:
                    return rect(hi, hi), line_point(hi), self._world(line_point(hi))[0]
                if (side_lo > 0) == (side_hi > 0):
                    raise PlaneNurbsNotCertified("section point bracket without a sign change")
            if side == 0:
                return rect(middle, middle), point, world
            if (side > 0) == (side_lo > 0):
                lo, side_lo = middle, side
            else:
                hi, side_hi = middle, side
        raise PlaneNurbsNotCertified("section point not within the plane slab after bisection")

    def _root_point(self, root):
        """The defined point at an end root: its isolating interval bisected into the plane slab."""
        if root.lo == root.hi:
            rect = self._root_rect(root)
            point = ((rect[0][0] + rect[0][1]) / 2, (rect[1][0] + rect[1][1]) / 2)
            return rect, point, self._world(point)[0]
        if root.axis == 0:
            return self._refine(lambda lo, hi: ((root.value, root.value), (lo, hi)), lambda x: (root.value, x),
                                root.lo, root.hi, None, None)
        return self._refine(lambda lo, hi: ((lo, hi), (root.value, root.value)), lambda x: (x, root.value),
                            root.lo, root.hi, None, None)

    def _exact_point(self, arc, tau: Fraction):
        """Exact knot rectangle ``((u_lo, u_hi), (v_lo, v_hi))``, the rational knot point of the curve
        at ``tau`` and its exact world point.

        The definition of the curve's points: ``s = s0 + tau (s1 - s0)`` exactly;
        ``x`` the zero of ``G(s, .)`` in the box, bracketed by exact bisection to
        ``2**-60`` of the box and further until the bracket midpoint's world point
        lies within ``2**-50`` of the surface magnitude of the plane (review B1-P2
        exact round 2: steep surfaces); the midpoint is the point. At ``tau`` 0 or
        1, or where ``s`` lies outside the exact gap (only inside an end root's
        isolating interval, by construction of ``s0``/``s1``), the end root's
        interval bisected likewise; anything else refuses.
        """
        if tau == 0:
            return self._root_point(arc.first)
        if tau == 1:
            return self._root_point(arc.second)
        s = arc.s0 + tau * (arc.s1 - arc.s0)
        box = arc.box
        c = box.coefficients
        if arc.slicing == 0:
            a, b = box.u_range
            column = [_de_casteljau([row[j] for row in c], (s - a) / (b - a)) for j in range(len(c[0]))]
            x0, x1 = box.v_range
        else:
            a, b = box.v_range
            column = [_de_casteljau(list(row), (s - a) / (b - a)) for row in c]
            x0, x1 = box.u_range
        low_value, high_value = _de_casteljau(column, Fraction(0)), _de_casteljau(column, Fraction(1))
        if low_value != 0 and high_value != 0 and (low_value > 0) == (high_value > 0):
            root = arc.first if tau < Fraction(1, 2) else arc.second
            (u_lo, u_hi), (v_lo, v_hi) = self._root_rect(root)
            low, high = (u_lo, u_hi) if arc.slicing == 0 else (v_lo, v_hi)
            if not low <= s <= high:
                raise PlaneNurbsNotCertified("section point outside its arc's gap and its end root")
            return self._root_point(root)
        if low_value == 0:
            lo = hi = Fraction(0)
        elif high_value == 0:
            lo = hi = Fraction(1)
        else:
            lo, hi = Fraction(0), Fraction(1)
            for _ in range(60):
                middle = (lo + hi) / 2
                value = _de_casteljau(column, middle)
                if value == 0:
                    lo = hi = middle
                    break
                if (value > 0) == (low_value > 0):
                    lo = middle
                else:
                    hi = middle

        def knot(x):
            value = x0 + x * (x1 - x0)
            return (s, value) if arc.slicing == 0 else (value, s)

        def rect(lo_x, hi_x):
            x_rect = (x0 + lo_x * (x1 - x0), x0 + hi_x * (x1 - x0))
            return ((s, s), x_rect) if arc.slicing == 0 else (x_rect, (s, s))

        if lo == hi:
            point = knot(lo)
            return rect(lo, hi), point, self._world(point)[0]
        # the box polynomial and the plane value have the same sign (positive weights)
        return self._refine(rect, knot, lo, hi, None, None)

    def _exact_points(self, t):
        """Per parameter: (arc, exact knot point (u, v), exact world point) - the definition of the curve's points."""
        arcs = self._arcs()
        path_t = self._path_parameter(t).ravel()
        index, tau, _count = self._locate(path_t)
        result = []
        for i in range(len(path_t)):
            arc = arcs[int(index[i])]
            _rect, point, world = self._exact_point(arc, Fraction(float(tau[i])))
            result.append((arc, point, world))
        return result

    def _exact_chart(self, knot_point):
        (r0, r1), (q0, q1) = self._chart_ranges()
        u, v = knot_point
        return (u - r0) / (r1 - r0), (v - q0) / (q1 - q0)

    def _knot_points(self, t):
        """Knot coordinates (floats): the rounded exact points."""
        points = self._exact_points(t)
        return (np.array([float(point[0]) for _arc, point, _world in points]),
                np.array([float(point[1]) for _arc, point, _world in points]))

    def chart(self, parameters) -> np.ndarray:
        result = np.array([[float(value) for value in self._exact_chart(point)]
                           for _arc, point, _world in self._exact_points(np.atleast_1d(parameters))])
        result = np.clip(result, 0.0, 1.0)
        return result if np.ndim(parameters) else result[0]

    def evaluate(self, parameters) -> np.ndarray:
        """The rounded exact world points ``S(q)`` (exact surface evaluation at the exact chart point).

        Exact through the surface too: a float surface evaluation remaps the chart
        to knots in floats, which loses precision on shifted knot domains (review
        B1-P2 exact round 1).
        """
        points = [[float(value) for value in world] for _arc, _point, world in self._exact_points(np.atleast_1d(parameters))]
        result = np.asarray(points)
        return result if np.ndim(parameters) else result[0]

    def derivative(self, parameters) -> np.ndarray:
        """World derivative: ``S_u du + S_v dv`` (knot-coordinate partials, exact at the exact point:
        float surface derivatives remap to knots in floats, review B1-P2 exact round 2) with the
        slicing rate exact and ``dx/ds = -G_s / G_x`` exact (float partials cancelled to 0 on
        near-vertical arcs)."""
        t = np.atleast_1d(np.asarray(parameters, dtype=float))
        count = len(self._arcs())
        points = self._exact_points(t)
        net = _surface_net(self.surface)
        a, b = self.domain
        rate = Fraction(count) * (Fraction(b) - Fraction(a))
        result = np.empty((len(t), 3))
        for i, (arc, point, _world) in enumerate(points):
            ds = (arc.s1 - arc.s0) * rate
            g_s, g_x = _exact_partials(arc, point)
            if g_x == 0:
                raise PlaneNurbsNotCertified("section derivative undefined (G_x = 0 in a regular box)")
            dx = -g_s / g_x * ds
            du, dv = (ds, dx) if arc.slicing == 0 else (dx, ds)
            s_u, s_v = _exact_knot_partials(self.surface, net, *point, box=arc.box)
            result[i] = [float(x * du + y * dv) for x, y in zip(s_u, s_v)]
        return result if np.ndim(parameters) else result[0]

    def derivatives(self, parameter, order: int = 0) -> np.ndarray:
        """Point and derivatives up to ``order`` at a scalar parameter, shape ``(order + 1, 3)``.

        The second derivative is a central difference of the first (used only by
        the projection's closer-only Newton polish, which cannot be misled into a
        worse answer).
        """
        t = float(parameter)
        rows = [self.evaluate(t)]
        if order >= 1:
            rows.append(self.derivative(t))
        if order >= 2:
            h = 1e-6
            lo, hi = max(0.0, t - h), min(1.0, t + h)
            rows.append((self.derivative(hi) - self.derivative(lo)) / (hi - lo))
        return np.asarray(rows)

    @staticmethod
    def _partials(arc, u, v):
        """Float partial derivatives (G_s, G_x) of the box polynomial at knot point (u, v)."""
        c = _scaled_float_coefficients(arc.box)            # power-of-two scaled: -G_s / G_x is unchanged
        (u0, u1), (v0, v1) = (tuple(map(float, item)) for item in (arc.box.u_range, arc.box.v_range))
        a, b = (u - u0) / (u1 - u0), (v - v0) / (v1 - v0)
        p, q = c.shape[0] - 1, c.shape[1] - 1
        bu = np.array([math.comb(p, i) * a ** i * (1 - a) ** (p - i) for i in range(p + 1)])
        bv = np.array([math.comb(q, j) * b ** j * (1 - b) ** (q - j) for j in range(q + 1)])
        du = np.array([p * (math.comb(p - 1, i) * a ** i * (1 - a) ** (p - 1 - i) if p else 0) for i in range(p)]) \
            if p else np.zeros(0)
        dv = np.array([q * (math.comb(q - 1, j) * b ** j * (1 - b) ** (q - 1 - j) if q else 0) for j in range(q)]) \
            if q else np.zeros(0)
        g_u = (du @ (c[1:, :] - c[:-1, :]) @ bv) / (u1 - u0) if p else 0.0
        g_v = (bu @ (c[:, 1:] - c[:, :-1]) @ dv) / (v1 - v0) if q else 0.0
        return (g_u, g_v) if arc.slicing == 0 else (g_v, g_u)

    # ---------------------------------------------------------- certificates and bounds

    def chart_enclosure(self, parameter) -> tuple:
        """Exact knot rectangle ``((u_lo, u_hi), (v_lo, v_hi))`` containing the curve point (``_exact_point``)."""
        arcs = self._arcs()
        path_t = float(self._path_parameter(np.array([float(parameter)]))[0])
        index, tau, _count = self._locate(np.array([path_t]))
        rect, _point, _world = self._exact_point(arcs[int(index[0])], Fraction(float(tau[0])))
        return rect

    def _chart_box_enclosure(self, s_range, x_range, slicing):
        """World box of the exact surface over an exact knot rectangle, widened by the rounding of evaluation.

        Evaluated points are the roundings of exact points (``evaluate``), so the
        exact sub-patch hull over the knot rectangle (``knot_patch_hull``, no
        widening of the parameters) plus two ulps of each coordinate's magnitude
        contains them (review B1-P2 exact round 1: the float-oriented enclosure
        widened a 1e16-shifted knot domain by its knot ulps).
        """
        from .surface_capabilities import Enclosure
        u_range, v_range = (s_range, x_range) if slicing == 0 else (x_range, s_range)
        hull = self.surface.knot_patch_hull(u_range, v_range, _surface_net(self.surface))
        lower, upper = [], []
        for k in range(3):
            low = _float_down(min(point[k] for point in hull))
            high = _float_up(max(point[k] for point in hull))
            margin = 2 * math.ulp(max(abs(low), abs(high), 1e-300))
            lower.append(low - margin)
            upper.append(high + margin)
        return Enclosure(tuple(lower), tuple(upper))

    @staticmethod
    def _sub_box(box, slicing, s_low, s_high):
        """Exact Bernstein coefficients of the box polynomial over ``[s_low, s_high]`` (full x range), [s][x]."""
        c = [list(row) for row in box.coefficients]                 # [u][v]
        if slicing == 1:
            c = [list(column) for column in zip(*c)]              # [v][u]
        a, b = (box.u_range if slicing == 0 else box.v_range)
        t0, t1 = (s_low - a) / (b - a), (s_high - a) / (b - a)

        def segment(values, lo, hi):
            # the Bernstein coefficients on [lo, hi] of the polynomial with coefficients ``values`` on [0, 1]
            right = _split_at(values, lo)[1]
            return _split_at(right, (hi - lo) / (1 - lo) if lo < 1 else Fraction(0))[0]

        columns = list(zip(*c))                                   # per x index: values over s
        pieces = [segment(list(column), t0, t1) for column in columns]
        return [[pieces[j][i] for j in range(len(pieces))] for i in range(len(pieces[0]))]

    def _tight_x(self, arc, s_low, s_high, depth=24):
        """An x interval (knot coordinates) containing the zero set of the box over ``[s_low, s_high]``.

        Exact exclusion (a sub-box whose Bernstein coefficients share a strict
        sign holds no zero): the lowest and the highest non-excluded leaves at
        ``depth`` are found by descending into the outer child first; the
        interval between them contains every zero (sound, linear in ``depth``).
        """
        coefficients = self._sub_box(arc.box, arc.slicing, s_low, s_high)
        x0, x1 = (arc.box.v_range if arc.slicing == 0 else arc.box.u_range)

        def excluded(coeffs):
            flat = [value for row in coeffs for value in row]
            return all(value > 0 for value in flat) or all(value < 0 for value in flat)

        def halves(coeffs):
            left, right = zip(*(_split_at(list(row), Fraction(1, 2)) for row in coeffs))
            return [list(row) for row in left], [list(row) for row in right]

        def extreme(lowest: bool):
            # depth-first, outer child first: the first non-excluded leaf is the extreme one
            stack = [(Fraction(0), Fraction(1), coefficients, 0)]
            while stack:
                lo, hi, coeffs, level = stack.pop()
                if excluded(coeffs):
                    continue
                if level >= depth:
                    return lo if lowest else hi
                middle = (lo + hi) / 2
                left, right = halves(coeffs)
                inner, outer = ((middle, hi, right), (lo, middle, left)) if lowest else ((lo, middle, left), (middle, hi, right))
                stack.append((*inner, level + 1))
                stack.append((*outer, level + 1))
            return None

        low = extreme(True)
        if low is None:
            return None
        high = extreme(False)
        return x0 + low * (x1 - x0), x0 + high * (x1 - x0)

    def _ordered_root_rect(self, arc, root):
        """An end root's rectangle as ((s_lo, s_hi), (x_lo, x_hi)) in the arc's slicing order."""
        (u_lo, u_hi), (v_lo, v_hi) = self._root_rect(root)
        return ((u_lo, u_hi), (v_lo, v_hi)) if arc.slicing == 0 else ((v_lo, v_hi), (u_lo, u_hi))

    def _arc_rects(self, lower, upper):
        """Per overlapping arc: ``(arc, (s_low, s_high), (x_low, x_high))`` exact knot rectangles containing the curve."""
        arcs = self._arcs()
        a, b = sorted(float(value) for value in self._path_parameter(np.array([float(lower), float(upper)])))
        count = len(arcs)
        # evaluation computes path_t = (1 - t) a + t b (not monotone in floats: a few ulps of 1)
        # and tau = path_t * count - k; the scaled interval is widened by that before the arcs
        # are selected (review B1-P2 exact round 2)
        slack = Fraction(16 * _EPS) * (count + 1)
        scaled_low, scaled_high = Fraction(a) * count - slack, Fraction(b) * count + slack
        rects = []
        for k, arc in enumerate(arcs):
            if scaled_high < k or scaled_low > k + 1:
                continue
            tau0 = min(Fraction(1), max(Fraction(0), scaled_low - k))
            tau1 = min(Fraction(1), max(Fraction(0), scaled_high - k))
            s_values = sorted((arc.s0 + tau0 * (arc.s1 - arc.s0), arc.s0 + tau1 * (arc.s1 - arc.s0)))
            box_s = (arc.box.u_range if arc.slicing == 0 else arc.box.v_range)
            # the end roots' intervals (points outside the gap are end roots) are added below
            s_low = max(Fraction(box_s[0]), s_values[0])
            s_high = min(Fraction(box_s[1]), s_values[1])
            x_range = self._tight_x(arc, s_low, s_high) if s_high > s_low else None
            box_x = (arc.box.v_range if arc.slicing == 0 else arc.box.u_range)
            if x_range is None:
                x_range = box_x
            else:
                # defined points are midpoints of (at most) 2**-60 brackets of exact roots
                widen = Fraction(1, 2 ** 59) * (box_x[1] - box_x[0])
                x_range = (max(box_x[0], x_range[0] - widen), min(box_x[1], x_range[1] + widen))
            rects.append((arc, (s_low, s_high), x_range))
            # points that fall back to an end root (s outside the exact gap) sit in its rectangle; that
            # happens only within the root's isolating width of the arc end (in tau)
            span = abs(arc.s1 - arc.s0)
            for root, near in ((arc.first, tau0), (arc.second, 1 - tau1)):
                (s_lo, s_hi), _x = self._ordered_root_rect(arc, root)
                window = (s_hi - s_lo) / span + Fraction(16 * _EPS) if span else Fraction(1)
                if near <= window:
                    rects.append((arc, *self._ordered_root_rect(arc, root)))
        return rects

    def distance_lower_bound(self, lower, upper, target) -> float:
        """Exact lower bound of the distance from ``target`` to the curve on ``[lower, upper]``.

        Per arc: the curve lies in the exact sub-patch hull over its tight chart
        rectangle and in the slab ``|m . x - d| <= limit`` (its defined points are
        bisection midpoints, ``_exact_point``). For orthonormal ``m^``, ``n^``
        (``n`` in the plane), ``|P - X|^2 >= (m^.(P - X))^2 + (n^.(P - X))^2``;
        the first term is the distance to the slab, the second is bounded below
        by the minimum of ``n . P`` over hull ``∩`` slab (an exact linear
        programme over the hull's control points). A transversal patch
        piece meets the plane in a sliver of second-order width, so the bound
        converges quadratically; the box distance is the fallback. Rounded down.
        """
        from .nurbs import _sqrt_down
        x = [Fraction(float(value)) for value in np.asarray(target, dtype=float)]
        m, d, mm, _limit_squared, limit = self._plane_slab()
        # defined points lie in the slab |m . P - d| <= limit (``_exact_point``), not on the plane
        normal_term = max(Fraction(0), abs(sum(a * b for a, b in zip(m, x)) - d) - limit) ** 2 / mm
        net = _surface_net(self.surface)
        n = nn = None
        with np.errstate(over="ignore", invalid="ignore"):
            middle = 0.5 * (float(lower) + float(upper))
            probe = np.asarray(self.evaluate(middle), dtype=float) - np.asarray(target, dtype=float)
        if np.all(np.isfinite(probe)) and np.any(probe != 0.0):
            n = [Fraction(float(value)) for value in probe]
            along = sum(a * b for a, b in zip(n, m)) / mm
            n = [a - along * b for a, b in zip(n, m)]                   # exactly in the plane
            nn = sum(value * value for value in n)
        best = None
        for arc, s_range, x_range in self._arc_rects(lower, upper):
            u_interval, v_interval = (s_range, x_range) if arc.slicing == 0 else (x_range, s_range)
            hull = self.surface.knot_patch_hull(u_interval, v_interval, net)
            box_squared = Fraction(0)
            for k in range(3):
                low, high = min(point[k] for point in hull), max(point[k] for point in hull)
                gap = max(low - x[k], x[k] - high, Fraction(0))
                box_squared += gap * gap
            squared = box_squared
            if n is not None:
                if nn > 0:
                    sides = [sum(a * b for a, b in zip(m, point)) - d for point in hull]
                    values = [sum(a * b for a, b in zip(n, point)) for point in hull]
                    # vertices of hull ∩ slab: hull points inside, edges crossing either slab face
                    candidates = [values[i] for i in range(len(hull)) if -limit <= sides[i] <= limit]
                    for level in (-limit, limit):
                        for i in range(len(hull)):
                            for j in range(len(hull)):
                                if sides[i] < level < sides[j]:
                                    r = (level - sides[i]) / (sides[j] - sides[i])
                                    candidates.append(values[i] + r * (values[j] - values[i]))
                    if candidates:
                        margin = min(candidates) - sum(a * b for a, b in zip(n, x))
                        if margin > 0:
                            squared = max(squared, normal_term + margin * margin / nn)
                    else:
                        squared = max(squared, normal_term)           # hull misses the plane: impossible piece
            best = squared if best is None else min(best, squared)
        if best is None:
            return 0.0
        # evaluated points are the roundings of exact points (half an ulp per coordinate):
        # the bound is lowered by that so it also holds for them (review B1-P2 exact round 1)
        magnitude = max(abs(float(value)) for value in x) + float(_sqrt_down(best)) + 1.0
        rounding = 2 * math.sqrt(3.0) * math.ulp(magnitude)
        return max(0.0, _float_down(_sqrt_down(best)) - rounding)

    def bounds(self, lower=0.0, upper=1.0):
        lows, highs = [], []
        for arc, (s_low, s_high), x_range in self._arc_rects(lower, upper):
            enclosure = self._chart_box_enclosure((s_low, s_high), x_range, arc.slicing)
            lows.append(enclosure.lower)
            highs.append(enclosure.upper)
        return np.min(np.asarray(lows), axis=0), np.max(np.asarray(highs), axis=0)

    def end_box(self, end: int):
        """World enclosure (lower, upper) of the curve end ``end`` (0 or 1): its certified root."""
        arcs = self._arcs()
        a, b = self.domain
        path_t = a if end == 0 else b
        if path_t not in (0.0, 1.0):
            # an interior end (a sub-curve): the exact point enclosure at the parameter the
            # evaluation uses (exact bisection of G to 2**-60 of the box)
            u_range, v_range = self.chart_enclosure(float(end))
            enclosure = self._chart_box_enclosure(u_range, v_range, 0)
            return np.asarray(enclosure.lower), np.asarray(enclosure.upper)
        root = arcs[0].first if path_t == 0.0 else arcs[-1].second
        if root.axis == 0:
            u_range, v_range = (root.value, root.value), (root.lo, root.hi)
        else:
            u_range, v_range = (root.lo, root.hi), (root.value, root.value)
        enclosure = self._chart_box_enclosure(u_range, v_range, 0)
        return np.asarray(enclosure.lower), np.asarray(enclosure.upper)

    def breakpoints(self) -> tuple[float, ...]:
        count = len(self._arcs())
        a, b = self.domain
        values = [(k / count - a) / (b - a) for k in range(1, count)]
        return tuple(sorted(value for value in values if 0.0 < value < 1.0))

    # ---------------------------------------------------------- edge operations

    def subcurve(self, lower, upper) -> "PlaneNurbsCurve":
        values = self._path_parameter(np.array([float(lower), float(upper)]))
        if values[0] == values[1]:
            raise GeometryError("curve interval must have positive length")
        return replace(self, domain=(float(values[0]), float(values[1])))

    def transformed(self, matrix) -> "PlaneNurbsCurve":
        from .operations import _transform_surface
        moved_plane = _transform_surface(self.plane, np.asarray(matrix, dtype=float))
        return replace(self, plane_record=_plane_record(moved_plane), surface=self.surface.transformed(matrix))


def _de_casteljau(values: list, t: Fraction) -> Fraction:
    work = list(values)
    for _ in range(len(work) - 1):
        work = [(1 - t) * a + t * b for a, b in zip(work, work[1:])]
    return work[0]


def _split_at(values: list, t: Fraction):
    """De Casteljau split of Bernstein coefficients at ``t``: (left on [0, t], right on [t, 1])."""
    left, right, work = [values[0]], [values[-1]], list(values)
    for _ in range(len(values) - 1):
        work = [(1 - t) * a + t * b for a, b in zip(work, work[1:])]
        left.append(work[0])
        right.append(work[-1])
    return left, right[::-1]


_NETS: dict = {}


def _surface_net(surface):
    net = _NETS.get(surface)
    if net is None:
        if len(_NETS) >= 32:
            _NETS.clear()
        net = _NETS[surface] = surface.exact_net()
    return net


_ABSOLUTE: dict = {}


def _absolute_box(box):
    """The box with absolute coefficients (for the float evaluation error bound)."""
    cached = _ABSOLUTE.get(id(box))
    if cached is not None and cached[0] is box:
        return cached[1]
    from dataclasses import replace as _replace
    result = _replace(box, coefficients=tuple(tuple(abs(value) for value in row) for row in box.coefficients))
    if len(_ABSOLUTE) >= 512:
        _ABSOLUTE.clear()
    _ABSOLUTE[id(box)] = (box, result)
    return result


_SCALED: dict = {}


def _scaled_float_coefficients(box) -> np.ndarray:
    """Float coefficients of the box scaled exactly by a power of two (largest magnitude in [1/2, 1))."""
    cached = _SCALED.get(id(box))
    if cached is not None and cached[0] is box:
        return cached[1]
    largest = max(abs(value) for row in box.coefficients for value in row)
    if largest == 0:
        array = np.zeros((len(box.coefficients), len(box.coefficients[0])))
    else:
        shift = largest.numerator.bit_length() - largest.denominator.bit_length()
        scale = Fraction(2) ** (-shift) if shift >= 0 else Fraction(2) ** (-shift)
        array = np.asarray([[float(value * scale) for value in row] for row in box.coefficients])
    if len(_SCALED) >= 512:
        _SCALED.clear()
    _SCALED[id(box)] = (box, array)
    return array


def _exact_partials(arc, point):
    """Exact (G_s, G_x) of the box polynomial at an exact knot point (Bernstein derivatives)."""
    box = arc.box
    c = [list(row) for row in box.coefficients]            # [u][v]
    (u0, u1), (v0, v1) = box.u_range, box.v_range
    a, b = (point[0] - u0) / (u1 - u0), (point[1] - v0) / (v1 - v0)
    p, q = len(c) - 1, len(c[0]) - 1

    def value(rows, s, t):
        return _de_casteljau([_de_casteljau(list(row), t) for row in rows], s)

    g_u = Fraction(0) if p == 0 else p * value([[c[i + 1][j] - c[i][j] for j in range(q + 1)] for i in range(p)], a, b) \
        / (u1 - u0)
    g_v = Fraction(0) if q == 0 else q * value([[c[i][j + 1] - c[i][j] for j in range(q)] for i in range(p + 1)], a, b) \
        / (v1 - v0)
    return (g_u, g_v) if arc.slicing == 0 else (g_v, g_u)


def _exact_knot_partials(surface, net, u, v, box=None):
    """Exact ``(S_u, S_v)`` per knot unit at the knot point ``(u, v)``: the homogeneous derivative
    nets (``p (Pw_{i+1} - Pw_i) / (U_{i+p+1} - U_{i+1})`` on ``U[1:-1]``) and the quotient rule.

    The polynomial piece is the knot span holding ``box`` (the arc's box, so a point
    on a knot line is differentiated on the same side as its section polynomial;
    review B1-P2 exact round 3), else the span ``exact_point`` uses.
    """
    from .nurbs import _exact_basis, _exact_span
    p, q = surface.degree_u, surface.degree_v
    knots_u, knots_v = [Fraction(k) for k in surface.knots_u], [Fraction(k) for k in surface.knots_v]
    count_u, count_v = len(net), len(net[0])

    if box is None:
        side_u, side_v = u, v
    else:
        side_u, side_v = (sum(box.u_range) / 2, sum(box.v_range) / 2)  # interior of one knot span

    def combine(degree_u, ku, cu, degree_v, kv, cv, entry):
        span_u, span_v = _exact_span(degree_u, ku, cu, side_u), _exact_span(degree_v, kv, cv, side_v)
        basis_u, basis_v = _exact_basis(degree_u, ku, span_u, u), _exact_basis(degree_v, kv, span_v, v)
        total = [Fraction(0)] * 4
        for a, nu in enumerate(basis_u):
            for b, nv in enumerate(basis_v):
                if nu and nv:
                    value = entry(span_u - degree_u + a, span_v - degree_v + b)
                    total = [x + nu * nv * y for x, y in zip(total, value)]
        return total

    def difference(i, j, di, dj, degree, knots, index):
        width = knots[index + degree + 1] - knots[index + 1]
        if width == 0:
            return (Fraction(0),) * 4
        return tuple(degree * (x - y) / width for x, y in zip(net[i + di][j + dj], net[i][j]))

    value = combine(p, knots_u, count_u, q, knots_v, count_v, lambda i, j: net[i][j])
    zero = [Fraction(0)] * 4
    d_u = zero if p == 0 else combine(p - 1, knots_u[1:-1], count_u - 1, q, knots_v, count_v,
                                      lambda i, j: difference(i, j, 1, 0, p, knots_u, i))
    d_v = zero if q == 0 else combine(p, knots_u, count_u, q - 1, knots_v[1:-1], count_v - 1,
                                      lambda i, j: difference(i, j, 0, 1, q, knots_v, j))
    w = value[3]
    point = [value[k] / w for k in range(3)]
    return tuple(tuple((d[k] - point[k] * d[3]) / w for k in range(3)) for d in (d_u, d_v))
