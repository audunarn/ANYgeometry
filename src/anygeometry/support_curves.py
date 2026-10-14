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
_X_RESOLUTION = 2.0 ** -44                 # certified bracket width of evaluation (fraction of the box)


def _plane_record(plane) -> tuple:
    return tuple(tuple(float(value) for value in vector) for vector in (plane.origin, plane.u_vector, plane.v_vector))


@dataclass(frozen=True)
class _Arc:
    box: object
    slicing: int                    # 0: s = u, x = v; 1: s = v, x = u (knot coordinates)
    s0: float
    s1: float
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
                    return float(root.value)
                return float((root.lo + root.hi) / 2)

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

    @staticmethod
    def _bernstein_rows(box, s, slicing):
        """Float Bernstein coefficients in x of G(s, .) for each s (vectorised de Casteljau along s)."""
        c = np.asarray([[float(value) for value in row] for row in box.coefficients])     # (p+1, q+1), [u][v]
        if slicing == 1:
            c = c.T                                        # make the first index the slicing coordinate
        low, high = (box.u_range if slicing == 0 else box.v_range)
        t = (np.asarray(s, dtype=float) - float(low)) / (float(high) - float(low))
        work = np.repeat(c[None, :, :], len(t), axis=0)
        for _ in range(work.shape[1] - 1):
            work = (1 - t)[:, None, None] * work[:, :-1, :] + t[:, None, None] * work[:, 1:, :]
        return work[:, 0, :]                               # (n, degree_x + 1)

    @staticmethod
    def _exact_sign(box, slicing, s: float, tau_x: float) -> int:
        """Exact sign of the box polynomial at the float point (s, local x fraction ``tau_x``)."""
        c = box.coefficients
        a, b = (box.u_range if slicing == 0 else box.v_range)
        ts = (Fraction(s) - a) / (b - a)
        tx = Fraction(tau_x)
        if slicing == 0:
            column = [_de_casteljau([row[j] for row in c], ts) for j in range(len(c[0]))]
        else:
            column = [_de_casteljau(list(row), ts) for row in c]
        value = _de_casteljau(column, tx)
        return 0 if value == 0 else (1 if value > 0 else -1)

    def _solve_x(self, arc, s):
        """Certified x(s): returns knot x and a mask of points with no root of G(s, .) in the box.

        Bisection whose every sign is certified: the float value of the box
        polynomial carries the bound ``gamma * M`` (``M`` the same de Casteljau on
        the absolute coefficients, ``gamma = 8 (p + q + 4) eps`` covering the
        float coefficients and the two de Casteljau passes); where the bound does
        not decide, the sign is computed exactly (review B1-P2: unverified float
        bisection escaped the exact zero set's bounds). The returned x lies in a
        certified bracket of the exact root of G(s_float, .) (width ``2**-52`` of the
        box, or where the float bound stops deciding: refinement ends there). Where G(s, .) has one strict sign at both box
        sides (a float s just outside the arc's exact gap) the mask is set.
        """
        box = arc.box
        rows = self._bernstein_rows(box, s, arc.slicing)
        rows_abs = self._bernstein_rows(_absolute_box(box), s, arc.slicing)
        p, q = len(box.coefficients) - 1, len(box.coefficients[0]) - 1
        gamma = 8 * (p + q + 4) * _EPS
        degree = rows.shape[1] - 1

        def value(work_rows, tau):
            work = work_rows.copy()
            for _ in range(degree):
                work = (1 - tau)[:, None] * work[:, :-1] + tau[:, None] * work[:, 1:]
            return work[:, 0]

        def float_signs(tau):
            values = value(rows, tau)
            bound = gamma * value(rows_abs, tau)
            return np.where(values > bound, 1, np.where(values < -bound, -1, 0))

        def exact_signs(tau):
            result = float_signs(tau)
            for i in np.nonzero(result == 0)[0]:
                result[i] = self._exact_sign(box, arc.slicing, float(s[i]), float(tau[i]))
            return result

        n = len(s)
        low_sign = exact_signs(np.zeros(n))                # the bracket's ends: certified exactly
        high_sign = exact_signs(np.ones(n))
        no_root = (low_sign == high_sign) & (low_sign != 0)
        lo, hi = np.zeros(n), np.ones(n)
        exact_low = low_sign == 0
        exact_high = high_sign == 0
        active = ~(no_root | exact_low | exact_high)
        for _ in range(52):
            if not np.any(active):
                break
            middle = 0.5 * (lo + hi)
            middle_sign = float_signs(middle)
            # where the float bound cannot decide, the sign is computed exactly while the
            # bracket is wider than 2**-44 of the box; below that the bracket is kept as is
            for i in np.nonzero(active & (middle_sign == 0))[0]:
                if hi[i] - lo[i] > _X_RESOLUTION:
                    middle_sign[i] = self._exact_sign(box, arc.slicing, float(s[i]), float(middle[i]))
            hit = active & (middle_sign == 0) & (hi - lo > _X_RESOLUTION)
            active &= ~((middle_sign == 0) & ~hit)
            lo = np.where(hit, middle, lo)
            hi = np.where(hit, middle, hi)
            active &= ~hit
            towards_high = active & (middle_sign == low_sign)
            towards_low = active & (middle_sign != 0) & ~towards_high
            lo = np.where(towards_high, middle, lo)
            hi = np.where(towards_low, middle, hi)
        x_fraction = np.where(exact_low, 0.0, np.where(exact_high, 1.0, 0.5 * (lo + hi)))
        x_low, x_high = (box.v_range if arc.slicing == 0 else box.u_range)
        return float(x_low) + x_fraction * (float(x_high) - float(x_low)), no_root

    def _knot_points(self, t):
        arcs = self._arcs()
        path_t = self._path_parameter(t).ravel()
        index, tau, _count = self._locate(path_t)
        u = np.empty(len(path_t))
        v = np.empty(len(path_t))
        for k in np.unique(index):
            mask = index == k
            arc = arcs[k]
            arc_tau = tau[mask]
            s = arc.s0 + arc_tau * (arc.s1 - arc.s0)
            x, no_root = self._solve_x(arc, s)
            # the arc's ends are its roots; a float s outside the exact gap takes the nearest end root
            for end_tau, root, selector in ((0.0, arc.first, arc_tau < .5), (1.0, arc.second, arc_tau >= .5)):
                use = (arc_tau == end_tau) | (no_root & selector)
                if np.any(use):
                    point = tuple(float(value) for value in root.point())
                    s[use] = point[arc.slicing]
                    x[use] = point[1 - arc.slicing]
            if arc.slicing == 0:
                u[mask], v[mask] = s, x
            else:
                u[mask], v[mask] = x, s
        return u, v

    def chart(self, parameters) -> np.ndarray:
        u, v = self._knot_points(parameters)
        cu, cv = self._knot_to_chart(u, v)
        result = np.column_stack((np.clip(cu, 0.0, 1.0), np.clip(cv, 0.0, 1.0)))
        return result if np.ndim(parameters) else result[0]

    def evaluate(self, parameters) -> np.ndarray:
        chart = np.atleast_2d(self.chart(np.atleast_1d(parameters)))
        points = self.surface.evaluate_many(chart)
        return points if np.ndim(parameters) else points[0]

    def derivative(self, parameters) -> np.ndarray:
        arcs = self._arcs()
        t = np.atleast_1d(np.asarray(parameters, dtype=float))
        path_t = self._path_parameter(t)
        index, _tau, count = self._locate(path_t)
        chart = np.atleast_2d(self.chart(t))
        s_u, s_v = self.surface.derivatives_many(chart)
        (r0, r1), (q0, q1) = (tuple(map(float, item)) for item in self._chart_ranges())
        u, v = self._knot_points(t)
        result = np.empty((len(t), 3))
        a, b = self.domain
        for i in range(len(t)):
            arc = arcs[index[i]]
            ds = (arc.s1 - arc.s0) * count * (b - a)
            g_s, g_x = self._partials(arc, u[i], v[i])
            dx = -g_s / g_x * ds if g_x != 0 else 0.0
            du, dv = (ds, dx) if arc.slicing == 0 else (dx, ds)
            result[i] = s_u[i] * du / (r1 - r0) + s_v[i] * dv / (q1 - q0)
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
        c = np.asarray([[float(value) for value in row] for row in arc.box.coefficients])
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
        """Exact knot rectangle ``((u_lo, u_hi), (v_lo, v_hi))`` containing the curve point (Fractions).

        At an arc end, or where the float slicing coordinate falls outside the
        arc's exact gap (no root of ``G(s, .)`` in the box), the end root's
        certified rectangle; otherwise ``s`` exact and ``x`` bracketed by exact
        bisection (a sign change is established first; review B1-P2).
        """
        arcs = self._arcs()
        path_t = float(self._path_parameter(np.array([float(parameter)]))[0])
        index, tau, _count = self._locate(np.array([path_t]))
        arc = arcs[int(index[0])]
        tau = float(tau[0])

        def root_rect(root):
            if root.axis == 0:
                return (root.value, root.value), (root.lo, root.hi)
            return (root.lo, root.hi), (root.value, root.value)

        if tau == 0.0:
            return root_rect(arc.first)
        if tau == 1.0:
            return root_rect(arc.second)
        s = Fraction(float(arc.s0 + tau * (arc.s1 - arc.s0)))
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
            return root_rect(arc.first if tau < .5 else arc.second)
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
        x_rect = (x0 + lo * (x1 - x0), x0 + hi * (x1 - x0))
        return ((s, s), x_rect) if arc.slicing == 0 else (x_rect, (s, s))

    def _chart_box_enclosure(self, s_range, x_range, slicing):
        from .surface_capabilities import capabilities
        (r0, r1), (q0, q1) = self._chart_ranges()
        u_range, v_range = (s_range, x_range) if slicing == 0 else (x_range, s_range)
        chart_u = sorted(((u - r0) / (r1 - r0)) for u in u_range)
        chart_v = sorted(((v - q0) / (q1 - q0)) for v in v_range)
        box = ((max(0.0, _float_down(chart_u[0])), min(1.0, _float_up(chart_u[1]))),
               (max(0.0, _float_down(chart_v[0])), min(1.0, _float_up(chart_v[1]))))
        return capabilities(self.surface).enclosure(box)

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

    def _arc_rects(self, lower, upper):
        """Per overlapping arc: ``(arc, (s_low, s_high), (x_low, x_high))`` exact knot rectangles containing the curve."""
        arcs = self._arcs()
        a, b = sorted(float(value) for value in self._path_parameter(np.array([float(lower), float(upper)])))
        count = len(arcs)
        rects = []
        for k, arc in enumerate(arcs):
            t0, t1 = k / count, (k + 1) / count
            if t1 < a or t0 > b:
                continue
            tau0, tau1 = max(0.0, (a - t0) * count), min(1.0, (b - t0) * count)
            s_values = sorted((arc.s0 + tau0 * (arc.s1 - arc.s0), arc.s0 + tau1 * (arc.s1 - arc.s0)))
            box_s = (arc.box.u_range if arc.slicing == 0 else arc.box.v_range)
            slack = Fraction(4 * _EPS * max(abs(float(box_s[0])), abs(float(box_s[1])), 1.0))
            s_low = max(Fraction(box_s[0]), Fraction(s_values[0]) - slack)
            s_high = min(Fraction(box_s[1]), Fraction(s_values[1]) + slack)
            x_range = self._tight_x(arc, s_low, s_high) if s_high > s_low else None
            box_x = (arc.box.v_range if arc.slicing == 0 else arc.box.u_range)
            if x_range is None:
                x_range = box_x
            else:
                # evaluated points sit within the evaluation bracket of an exact root (2**-44 of the box)
                widen = Fraction(_X_RESOLUTION) * (box_x[1] - box_x[0])
                x_range = (max(box_x[0], x_range[0] - widen), min(box_x[1], x_range[1] + widen))
            rects.append((arc, (s_low, s_high), x_range))
        return rects

    def distance_lower_bound(self, lower, upper, target) -> float:
        """Exact lower bound of the distance from ``target`` to the curve on ``[lower, upper]``.

        Per arc: the curve lies in the exact sub-patch hull over its tight chart
        rectangle and in the plane ``m . x = d``. For orthonormal ``m^``, ``n^``
        (``n`` in the plane), ``|P - X|^2 >= (m^.(P - X))^2 + (n^.(P - X))^2``;
        the first term is the distance to the plane, the second is bounded below
        by the minimum of ``n . P`` over hull ``∩`` plane (an exact two-point
        linear programme over the hull's control points). A transversal patch
        piece meets the plane in a sliver of second-order width, so the bound
        converges quadratically; the box distance is the fallback. Rounded down.
        """
        from .nurbs import _sqrt_down
        x = [Fraction(float(value)) for value in np.asarray(target, dtype=float)]
        origin, u_vector, v_vector = ([Fraction(value) for value in vector] for vector in self.plane_record)
        m = (u_vector[1] * v_vector[2] - u_vector[2] * v_vector[1], u_vector[2] * v_vector[0] - u_vector[0] * v_vector[2],
             u_vector[0] * v_vector[1] - u_vector[1] * v_vector[0])
        mm = sum(value * value for value in m)
        d = sum(a * b for a, b in zip(m, origin))
        normal_term = (sum(a * b for a, b in zip(m, x)) - d) ** 2 / mm
        net = _surface_net(self.surface)
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
            with np.errstate(over="ignore", invalid="ignore"):
                middle = 0.5 * (float(lower) + float(upper))
                probe = np.asarray(self.evaluate(middle), dtype=float) - np.asarray(target, dtype=float)
            if np.all(np.isfinite(probe)) and np.any(probe != 0.0):
                n = [Fraction(float(value)) for value in probe]
                along = sum(a * b for a, b in zip(n, m)) / mm
                n = [a - along * b for a, b in zip(n, m)]               # exactly in the plane
                nn = sum(value * value for value in n)
                if nn > 0:
                    sides = [sum(a * b for a, b in zip(m, point)) - d for point in hull]
                    values = [sum(a * b for a, b in zip(n, point)) for point in hull]
                    candidates = [values[i] for i in range(len(hull)) if sides[i] == 0]
                    for i in range(len(hull)):
                        for j in range(len(hull)):
                            if sides[i] < 0 < sides[j]:
                                r = sides[i] / (sides[i] - sides[j])
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
        return _float_down(_sqrt_down(best))

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
