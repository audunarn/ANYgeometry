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
from contextvars import ContextVar
from .curve_parameter_matching import _cooperative_tick
import math

import numpy as np

from .errors import GeometryError
from ._cert_exact_runtime import fraction_kind as _fraction_kind
from .nurbs import NurbsSurface, _float_down, _float_up
from .plane_nurbs import PlaneNurbsNotCertified, charge_box, charge_split, depth_limit, plane_nurbs_section

_SECTIONS: dict = {}
_RUNTIME_CACHE_BANK = ContextVar('plane_nurbs_runtime_cache_bank', default=None)


def _runtime_cache(name, shared):
    isolated = _RUNTIME_CACHE_BANK.get()
    return shared if isolated is None else isolated.setdefault(name, {})
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

    def __post_init__(self):
        # stored as doubles: the record writes what a reader restores (review B1-P3)
        try:
            domain = tuple(float(value) for value in self.domain)
        except (TypeError, ValueError, OverflowError) as error:
            raise GeometryError("plane x NURBS curve domain must be two numbers") from error
        if len(domain) != 2:
            raise GeometryError("plane x NURBS curve domain must be two numbers")
        # the loader's rule, so every constructible curve also reloads (review B1-P3 round 2)
        if not all(0.0 <= value <= 1.0 for value in domain) or domain[0] == domain[1]:
            raise GeometryError("plane x NURBS curve domain must be two distinct values in [0, 1]")
        object.__setattr__(self, "domain", domain)

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
        cache = _runtime_cache('sections', _SECTIONS)
        key = (self.plane_record, self.surface, self.path)
        cached = cache.get(key)
        if cached is not None:
            return cached
        section = plane_nurbs_section(self.plane, self.surface)
        if section.coincident or not 0 <= self.path < len(section.paths):
            raise PlaneNurbsNotCertified("no such section path")
        # Root intervals refined (on copies: the section's roots are shared and recorded) until their
        # world image is within the curve's error bound: the arc limits (interval midpoints) and the
        # end points then lie within that bound of the true roots (review B1-P4a round 2).
        refined: dict = {}

        def root_of(ident):
            if ident not in refined:
                root = replace(section.roots[ident])
                for _ in range(self._MAX_BISECTIONS):
                    _cooperative_tick()
                    if root.exact or self._rect_ok(self._root_rect(root)):
                        break
                    root.refine()
                else:
                    raise PlaneNurbsNotCertified("section root not refined to the curve's error bound")
                refined[ident] = root
            return refined[ident]

        chain = [(section.boxes[box_index], 0 if section.boxes[box_index].kind == "v" else 1, root_of(first),
                  root_of(second)) for box_index, first, second in section.paths[self.path].arcs]
        # An arc starts at C*(s0), the zero at the rational arc limit s0 (or the root outside the gap):
        # the root interval is refined until the rectangle holding both the true root and C*(s0) is
        # within the error bound, for every arc using the root (review B1-P4a round 3).
        for _ in range(self._MAX_BISECTIONS):
            _cooperative_tick()
            loose = {id(root): root for box, slicing, start, finish in chain for root in (start, finish)
                     if not root.exact and not self._joint_ok(box, slicing, root)}
            if not loose:
                break
            for root in loose.values():
                root.refine()
        else:
            raise PlaneNurbsNotCertified("arc joint not refined to the curve's error bound")

        def s_of(root, slicing):
            if root.axis == slicing:
                return root.value
            return (root.lo + root.hi) / 2

        arcs = [_Arc(box, slicing, s_of(start, slicing), s_of(finish, slicing), start, finish)
                for box, slicing, start, finish in chain]
        result = tuple(arcs)
        if len(cache) >= 64:
            cache.clear()
        cache[key] = result
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
        cache = _runtime_cache('sections', _SECTIONS)
        cached = cache.get(key)
        if cached is not None:
            return cached
        origin, u_vector, v_vector = ([Fraction(value) for value in vector] for vector in self.plane_record)
        m = (u_vector[1] * v_vector[2] - u_vector[2] * v_vector[1], u_vector[2] * v_vector[0] - u_vector[0] * v_vector[2],
             u_vector[0] * v_vector[1] - u_vector[1] * v_vector[0])
        mm = sum(value * value for value in m)
        d = sum(a * b for a, b in zip(m, origin))
        # the surface's magnitude only: the plane's origin may sit anywhere along the plane and must
        # not inflate the error bound (review B1-P4a round 4)
        magnitude = max([1.0] + [abs(float(value)) for row in self.surface.controls for point in row for value in point])
        delta = Fraction(magnitude) / 2 ** 50
        # rational upper bound of sqrt(mm): (isqrt(n d) + 1) / d for mm = n / d
        root_up = Fraction(math.isqrt(mm.numerator * mm.denominator) + 1, mm.denominator)
        result = (m, d, mm, delta * delta * mm, delta * root_up)
        if len(cache) >= 64:
            cache.clear()
        cache[key] = result
        return result

    def _world_bound(self):
        """``(sigma2, u_width, v_width, delta2)``: the squared Jacobian bound of the whole surface (chart
        units, exact), the knot widths of the chart ranges and the squared error bound ``delta``
        (``2**-50`` of the surface magnitude, as the plane slab)."""
        key = ("bound", self.surface, self.plane_record)
        cache = _runtime_cache('sections', _SECTIONS)
        cached = cache.get(key)
        if cached is None:
            _m, _d, mm, limit_squared, _l = self._plane_slab()
            sigma2 = self.surface.patch_bounds(((0.0, 1.0), (0.0, 1.0)), _surface_net(self.surface))[2]
            (r0, r1), (q0, q1) = self._chart_ranges()
            cached = (Fraction(sigma2), abs(r1 - r0), abs(q1 - q0), limit_squared / mm)
            if len(cache) >= 64:
                cache.clear()
            cache[key] = cached
        return cached

    def _joint_ok(self, box, slicing, root) -> bool:
        """Whether the rectangle holding the true root and the arc's first point C*(s0) is within the
        error bound. ``s0`` is the root's value on a constant-s edge (C*(s0) is the root itself), else
        the midpoint of its interval: then C*(s0) is the box's zero at s0 (bracketed exactly) or, outside
        the gap, the root."""
        _cooperative_tick()
        rect = self._root_rect(root)
        if root.axis == slicing:
            return self._rect_ok(rect)
        s0 = (root.lo + root.hi) / 2
        c = box.coefficients
        if slicing == 0:
            a, b = box.u_range
            column = [_de_casteljau([row[j] for row in c], (s0 - a) / (b - a)) for j in range(len(c[0]))]
            x0, x1 = box.v_range
        else:
            a, b = box.v_range
            column = [_de_casteljau(list(row), (s0 - a) / (b - a)) for row in c]
            x0, x1 = box.u_range
        low, high = _de_casteljau(column, Fraction(0)), _de_casteljau(column, Fraction(1))
        if low != 0 and high != 0 and (low > 0) == (high > 0):
            return self._rect_ok(rect)                         # outside the gap: C*(s0) is the root
        lo, hi = Fraction(0), Fraction(1)
        if low == 0:
            hi = lo
        elif high == 0:
            lo = hi
        else:
            # bracket C*(s0) until its own rectangle meets the bound (else root refinement cannot help)
            for _ in range(self._MAX_BISECTIONS):
                _cooperative_tick()
                bracket = (x0 + lo * (x1 - x0), x0 + hi * (x1 - x0))
                if self._rect_ok(((s0, s0), bracket) if slicing == 0 else (bracket, (s0, s0))):
                    break
                middle = (lo + hi) / 2
                value = _de_casteljau(column, middle)
                if value == 0:
                    lo = hi = middle
                    break
                if (value > 0) == (low > 0):
                    lo = middle
                else:
                    hi = middle
        x_lo, x_hi = x0 + lo * (x1 - x0), x0 + hi * (x1 - x0)
        x_range = (min(x_lo, root.value), max(x_hi, root.value))
        s_range = (root.lo, root.hi)
        return self._rect_ok((s_range, x_range) if slicing == 0 else (x_range, s_range))

    def _rect_ok(self, rect) -> bool:
        """Whether every point of the surface over the knot rectangle lies within ``delta`` of every other
        (``sigma_max`` times the chart diagonal; the curve's world error bound, review B1-P4a round 2)."""
        sigma2, u_width, v_width, delta2 = self._world_bound()
        (u_lo, u_hi), (v_lo, v_hi) = rect
        return sigma2 * (((u_hi - u_lo) / u_width) ** 2 + ((v_hi - v_lo) / v_width) ** 2) <= delta2

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
    _INITIAL_ROOT_BISECTIONS = 60

    def _refine(self, rect, line_point, lo, hi, side_lo, side_hi):
        """Bisect ``[lo, hi]`` on a knot line (``line_point(x)`` the knot point) until the midpoint's
        world point lies in the plane slab; ``side_lo``/``side_hi`` the end sides (``None`` unknown).

        The point is the midpoint (a deterministic definition); exact zeros end it.
        Refuses when the slab is not reached (the line's sign change is lost). The bracket must also
        be within the curve's world error bound (``_rect_ok``): a slab alone does not bound the
        displacement along the curve (review B1-P4a round 2).
        """
        _m, _d, _mm, limit, _l = self._plane_slab()
        for _ in range(self._MAX_BISECTIONS):
            _cooperative_tick()
            middle = (lo + hi) / 2
            point = line_point(middle)
            world, side = self._world(point)
            if side * side <= limit and self._rect_ok(rect(lo, hi)):
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
            _cooperative_tick()
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
            for _ in range(self._INITIAL_ROOT_BISECTIONS):
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

    def _exact_locate(self, parameter):
        """Arc index and exact local parameter ``tau`` of a curve parameter: the exact affine map
        ``path = a + t (b - a)``, ``tau = path * count - k`` (the predicates use the same map; a float
        map moved the point along the curve by far more than the error bound, review B1-P4a round 4)."""
        count = len(self._arcs())
        a, b = (Fraction(value) for value in self.domain)
        t = min(Fraction(1), max(Fraction(0), Fraction(float(parameter))))
        scaled = (a + t * (b - a)) * count
        index = min(count - 1, max(0, math.floor(scaled)))
        return index, min(Fraction(1), max(Fraction(0), scaled - index))

    def _exact_points(self, t):
        """Per parameter: (arc, exact knot point (u, v), exact world point) - the definition of the curve's points."""
        arcs = self._arcs()
        result = []
        for value in np.atleast_1d(np.asarray(t, dtype=float)).ravel():
            index, tau = self._exact_locate(value)
            arc = arcs[index]
            _rect, point, world = self._exact_point(arc, tau)
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

    def chart_bounds(self):
        """Outward double chart box ``(low[2], high[2])`` of the curve (the exact arc rectangles of
        ``_arc_rects`` mapped to the chart), as ``NurbsCurve.bounds`` gives for a pcurve (B1-P4c)."""
        (r0, r1), (q0, q1) = self._chart_ranges()
        us, vs = [], []
        for arc, s_range, x_range in self._arc_rects(0.0, 1.0):
            u_range, v_range = (s_range, x_range) if arc.slicing == 0 else (x_range, s_range)
            us.extend((value - r0) / (r1 - r0) for value in u_range)
            vs.extend((value - q0) / (q1 - q0) for value in v_range)
        return (np.array([_float_down(min(us)), _float_down(min(vs))]),
                np.array([_float_up(max(us)), _float_up(max(vs))]))

    def chart_derivative(self, parameters) -> np.ndarray:
        """Chart derivative ``d(cu, cv)/dt`` (exact at the exact point, then rounded): the slicing rate
        and ``dx/ds = -G_s / G_x`` as in ``derivative``, divided by the chart ranges (B1-P4c)."""
        t = np.atleast_1d(np.asarray(parameters, dtype=float))
        count = len(self._arcs())
        (r0, r1), (q0, q1) = self._chart_ranges()
        a, b = self.domain
        rate = Fraction(count) * (Fraction(b) - Fraction(a))
        result = np.empty((len(t), 2))
        for i, (arc, point, _world) in enumerate(self._exact_points(t)):
            ds = (arc.s1 - arc.s0) * rate
            g_s, g_x = _exact_partials(arc, point)
            if g_x == 0:
                raise PlaneNurbsNotCertified("section derivative undefined (G_x = 0 in a regular box)")
            dx = -g_s / g_x * ds
            du, dv = (ds, dx) if arc.slicing == 0 else (dx, ds)
            result[i] = (float(du / (r1 - r0)), float(dv / (q1 - q0)))
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
        index, tau = self._exact_locate(parameter)
        rect, _point, _world = self._exact_point(self._arcs()[index], tau)
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
        interval between them contains every zero (sound, linear in ``depth``). Under an active
        qualification budget every examined sub-box and split is charged and ``depth`` is capped
        by the budget's depth limit (a shallower descent only widens the interval); an exhausted
        budget refuses (``PlaneNurbsNotCertified``).
        """
        depth = depth_limit(depth)
        coefficients = _integer_scaled(self._sub_box(arc.box, arc.slicing, s_low, s_high))
        x0, x1 = (arc.box.v_range if arc.slicing == 0 else arc.box.u_range)

        def excluded(coeffs):
            flat = [value for row in coeffs for value in row]
            return all(value > 0 for value in flat) or all(value < 0 for value in flat)

        def halves(coeffs):
            # only signs are used: integer coefficients and a uniform positive scale per split
            left, right = zip(*(_halve_scaled(row) for row in coeffs))
            return [list(row) for row in left], [list(row) for row in right]

        def extreme(lowest: bool):
            # depth-first, outer child first: the first non-excluded leaf is the extreme one
            stack = [(Fraction(0), Fraction(1), coefficients, 0)]
            while stack:
                lo, hi, coeffs, level = stack.pop()
                charge_box()
                if excluded(coeffs):
                    continue
                if level >= depth:
                    return lo if lowest else hi
                charge_split()
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

    def _graph_view(self, arc):
        """The shared primitive rectangle view of ``branch_predicates._Arc`` for the graph ``G1 = 0`` only
        (no second plane or weight polynomial is needed by the graph proofs used here)."""
        from .branch_predicates import _Arc
        view = _Arc.__new__(_Arc)
        box = arc.box
        view.curve, view.arc = self, arc
        view.u_box, view.v_box = box.u_range, box.v_range
        view.g1 = [list(row) for row in box.coefficients]
        view.g2 = view.w = None
        view.slicing = arc.slicing
        view.x_box = box.v_range if arc.slicing == 0 else box.u_range
        return view

    def _graph_x(self, arc, s_low, s_high, level, parent=None):
        """An x interval (knot coordinates) containing the section graph over ``[s_low, s_high]``, from the
        certified monotone-graph proof, or ``None`` when that proof is undecided (never a guess).

        ``parent`` (default the box's x range) must contain the graph over the piece. Where the ambient
        ``G_x`` has one strict sign and ``G_s`` a weak sign over ``piece x parent``, ``x(s)`` is monotone
        (``dx/ds = -G_s / G_x``) and the interval between exact brackets of the two ends contains it. A
        piece whose ``G_s`` straddles zero is bisected (at most ``depth_limit(3)`` levels, each bisection charged
        as a subdivision) and the halves united; the exact end-bracket bisection is a point refinement, not charged. The exact
        constant graph identity (``G1`` vanishing on the whole line ``x = x0`` with strict ``G_x``)
        gives ``[x0, x0]``. A bracket is ``None`` outside the arc's gap (end-root region): undecided.
        Every examined box is charged to the active qualification budget.
        """
        view = self._graph_view(arc)
        x_box = view.x_box
        s_box = view.u_box if view.slicing == 0 else view.v_box
        root_x = [self._ordered_root_rect(arc, root)[1] for root in (arc.first, arc.second)]
        if root_x[0][0] == root_x[0][1] == root_x[1][0] == root_x[1][1]:
            x0 = root_x[0][0]
            if parent is None or parent[0] <= x0 <= parent[1]:
                charge_box()
                restriction = view.over(view.g1, s_box, (x0, x0))
                du, dv = view.derivative_ranges(view.g1, s_box, (x0, x0))
                gx = dv if view.slicing == 0 else du
                if all(value == 0 for row in restriction for value in row) and (gx[0] > 0 or gx[1] < 0):
                    return (x0, x0)
        parent = x_box if parent is None else parent
        x_span, s_span = x_box[1] - x_box[0], s_box[1] - s_box[0]
        if s_high <= s_low:
            charge_box()
            return view.bracket(s_low, x_span / 2 ** 60)

        def attempt(lo, hi, splits):
            charge_box()
            g_u, g_v = view.derivative_ranges(view.g1, (lo, hi), parent)
            g_s, g_x = (g_u, g_v) if view.slicing == 0 else (g_v, g_u)
            if not (g_x[0] > 0 or g_x[1] < 0):
                return None
            if g_s[0] >= 0 or g_s[1] <= 0:
                charge_box()
                width = x_span * (hi - lo) / s_span / 64
                first, second = view.bracket(lo, width), view.bracket(hi, width)
                if first is None or second is None:
                    return None
                return min(first[0], second[0]), max(first[1], second[1])
            if splits <= 0:
                return None
            charge_split()                  # an actual spatial s-bisection (not an exact root-bracket step)
            middle = (lo + hi) / 2
            left = attempt(lo, middle, splits - 1)
            right = attempt(middle, hi, splits - 1) if left is not None else None
            if left is None or right is None:
                return None
            return min(left[0], right[0]), max(left[1], right[1])

        return attempt(s_low, s_high, depth_limit(3))

    def _certified_x(self, arc, s_low, s_high, level):
        """x interval containing the zero set over ``[s_low, s_high]``: the graph proof (box, then a coarse
        exclusion interval as parent), else the exclusion interval ``_tight_x`` at the full depth; ``None``
        only when exact exclusion shows no zero (or a zero-width piece is outside the gap)."""
        x = self._graph_x(arc, s_low, s_high, level)
        if x is not None or s_high <= s_low:
            return x
        coarse = self._tight_x(arc, s_low, s_high, depth=min(8, 24 + level))
        if coarse is None:
            return None
        x = self._graph_x(arc, s_low, s_high, level, parent=coarse)
        if x is not None:
            return x
        return self._tight_x(arc, s_low, s_high, depth=24 + level)

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
            _cooperative_tick()
            if scaled_high < k or scaled_low > k + 1:
                continue
            tau0 = min(Fraction(1), max(Fraction(0), scaled_low - k))
            tau1 = min(Fraction(1), max(Fraction(0), scaled_high - k))
            s_values = sorted((arc.s0 + tau0 * (arc.s1 - arc.s0), arc.s0 + tau1 * (arc.s1 - arc.s0)))
            box_s = (arc.box.u_range if arc.slicing == 0 else arc.box.v_range)
            # the end roots' intervals (points outside the gap are end roots) are added below
            s_low = max(Fraction(box_s[0]), s_values[0])
            s_high = min(Fraction(box_s[1]), s_values[1])
            # the x resolution follows the s range: a fixed 24 halvings of the box was far wider than a
            # short piece's curve (review B1-P4b round 2: 1.8e-7 boxes over 2e-12 of parameter)
            level = 0
            while level < 40 and (s_high - s_low) * 2 ** (level + 1) <= box_s[1] - box_s[0]:
                level += 1
            x_range = self._certified_x(arc, s_low, s_high, level)
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
        # capped at 2**1000 (still a lower bound) so huge exact distances convert to floats; the
        # rounding uses ulp(a + b + c) <= 2 (ulp a + ulp b + ulp c) without forming the float sum
        root = min(_sqrt_down(best), Fraction(2) ** 1000)
        low = _float_down(root)
        largest = max(abs(float(value)) for value in x)
        rounding = 4 * math.sqrt(3.0) * (math.ulp(largest) + math.ulp(low) + math.ulp(1.0))
        return max(0.0, low - rounding)

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
    kind = ().__class__.__class__
    rational = _fraction_kind(Fraction)
    canonical = rational is not None and len(work) > 1 and kind(t) is rational
    if canonical:
        for value in work:
            if kind(value) is not rational:
                canonical = False
                break
    if canonical and '_from_coprime_ints' in rational.__dict__:
        factory = rational._from_coprime_ints
        # One common denominator for the coefficients, then integer affine
        # recurrence. Each level adds exactly one factor of t._denominator.
        denominator = 1
        for a in work:
            d = a._denominator
            x, y = denominator, d
            while y:
                x, y = y, x % y
            denominator = denominator // x * d
        work = [a._numerator * (denominator // a._denominator) for a in work]
        p, q = t._numerator, t._denominator
        complement = q - p
        for _ in range(len(work) - 1):
            work = [complement * a + p * b for a, b in zip(work, work[1:])]
            denominator *= q
        numerator = work[0]
        x, y = numerator, denominator
        while y:
            x, y = y, x % y
        return factory(numerator // x, denominator // x)
    for _ in range(len(work) - 1):
        work = [(1 - t) * a + t * b for a, b in zip(work, work[1:])]
    return work[0]


def _integer_scaled(rows) -> list:
    """Bernstein rows (Fractions) times the least common denominator: integers with the same signs
    (sign tests are all the subdivision needs; CI tiers C1: rational splits dominated the audit)."""
    denominator = 1
    for row in rows:
        for value in row:
            denominator = math.lcm(denominator, Fraction(value).denominator)
    return [[int(Fraction(value) * denominator) for value in row] for row in rows]


def _halve_scaled(values: list):
    """De Casteljau split at 1/2 of integer Bernstein coefficients, both halves scaled by ``2**n``
    (``n`` the degree): sums only, the signs those of the exact halves."""
    n = len(values) - 1
    left, right, work = [values[0] << n], [values[-1] << n], list(values)
    for k in range(1, n + 1):
        work = [a + b for a, b in zip(work, work[1:])]          # scaled by 2**k
        left.append(work[0] << (n - k))
        right.append(work[-1] << (n - k))
    return left, right[::-1]


def _split_at(values: list, t: Fraction):
    """De Casteljau split of Bernstein coefficients at ``t``: (left on [0, t], right on [t, 1]).

    Exact, in integers: the values over a common denominator ``D`` and ``t = p / q``, each step
    ``(q - p) a + p b`` multiplies the scale by ``q``; every output is normalised once (CI tiers C1:
    the per-operation gcd of rational arithmetic dominated the exact section certificates)."""
    t = Fraction(t)
    p, q = t.numerator, t.denominator
    fractions = [Fraction(value) for value in values]
    denominator = 1
    for value in fractions:
        denominator = math.lcm(denominator, value.denominator)
    work = [value.numerator * (denominator // value.denominator) for value in fractions]
    left, right = [fractions[0]], [fractions[-1]]
    scale = denominator
    complement = q - p
    for _ in range(len(values) - 1):
        work = [complement * x + p * y for x, y in zip(work, work[1:])]
        scale *= q
        left.append(Fraction(work[0], scale))
        right.append(Fraction(work[-1], scale))
    return left, right[::-1]


_NETS: dict = {}


def _surface_net(surface):
    cache = _runtime_cache('nets', _NETS)
    net = cache.get(surface)
    if net is None:
        if len(cache) >= 32:
            cache.clear()
        net = cache[surface] = surface.exact_net()
    return net


_ABSOLUTE: dict = {}


def _absolute_box(box):
    """The box with absolute coefficients (for the float evaluation error bound)."""
    cache = _runtime_cache('absolute', _ABSOLUTE)
    cached = cache.get(id(box))
    if cached is not None and cached[0] is box:
        return cached[1]
    from dataclasses import replace as _replace
    result = _replace(box, coefficients=tuple(tuple(abs(value) for value in row) for row in box.coefficients))
    if len(cache) >= 512:
        cache.clear()
    cache[id(box)] = (box, result)
    return result


_SCALED: dict = {}


def _scaled_float_coefficients(box) -> np.ndarray:
    """Float coefficients of the box scaled exactly by a power of two (largest magnitude in [1/2, 1))."""
    cache = _runtime_cache('scaled', _SCALED)
    cached = cache.get(id(box))
    if cached is not None and cached[0] is box:
        return cached[1]
    largest = max(abs(value) for row in box.coefficients for value in row)
    if largest == 0:
        array = np.zeros((len(box.coefficients), len(box.coefficients[0])))
    else:
        shift = largest.numerator.bit_length() - largest.denominator.bit_length()
        scale = Fraction(2) ** (-shift) if shift >= 0 else Fraction(2) ** (-shift)
        array = np.asarray([[float(value * scale) for value in row] for row in box.coefficients])
    if len(cache) >= 512:
        cache.clear()
    cache[id(box)] = (box, array)
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


# ---------------------------------------------------------------------- schema-7 record (B1-P3)

# v2 (review B1-P4a round 3): section roots refined to the curve's world error bound; v1 records
# (unrefined roots) describe different arc limits and are refused, not migrated
PLANE_NURBS_ALGORITHM = "plane_nurbs_exact_v2"
_RECORD_FIELDS = {"type", "plane", "surface", "path", "domain", "branch"}
_BRANCH_FIELDS = {"algorithm", "roots", "arcs"}
_PLANE_FIELDS = {"type", "origin", "u_vector", "v_vector"}
_RATIONAL_DIGITS = 4096


def _rational_text(value: Fraction) -> str:
    """Canonical exact text of a rational: ``"n"`` or ``"n/d"`` (lowest terms, ``d > 1``)."""
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _rational_from_text(text: object, name: str) -> Fraction:
    """Parse a canonical rational text; any other spelling (or a non-string) is refused."""
    if not isinstance(text, str) or not 0 < len(text) <= _RATIONAL_DIGITS:
        raise GeometryError(f"{name} must be an exact rational text")
    head, slash, tail = text.partition("/")
    numerator_ok = head.lstrip("-").isdigit() and head.isascii() and head.count("-") <= 1
    denominator_ok = not slash or (tail.isdigit() and tail.isascii())
    if not numerator_ok or not denominator_ok:
        raise GeometryError(f"{name} must be an exact rational text")
    if slash and int(tail) == 0:
        raise GeometryError(f"{name} has a zero denominator")
    value = Fraction(int(head), int(tail)) if slash else Fraction(int(head))
    if _rational_text(value) != text:
        raise GeometryError(f"{name} is not in canonical form (lowest terms, no leading zeros)")
    return value


def _branch_record(curve: "PlaneNurbsCurve") -> dict:
    """The certified path: its roots in path order and the arcs between consecutive roots."""
    arcs = curve._arcs()
    ordered = [arcs[0].first] + [arc.second for arc in arcs]
    return {"algorithm": PLANE_NURBS_ALGORITHM,
            "roots": [[root.axis, *(_rational_text(value) for value in (root.value, root.lo, root.hi))]
                      for root in ordered],
            "arcs": [[arc.box.patch, [_rational_text(value) for value in (*arc.box.u_range, *arc.box.v_range)],
                      arc.slicing] for arc in arcs]}


def plane_nurbs_record(curve: "PlaneNurbsCurve") -> dict:
    """Schema-7 ``plane_nurbs_intersection`` record: the definition (plane, surface, path, domain)
    and the certified branch (roots and arc boxes as exact rationals), recomputed and compared on load."""
    from .nurbs import nurbs_record
    origin, u_vector, v_vector = (list(vector) for vector in curve.plane_record)
    return {"type": "plane_nurbs_intersection",
            "plane": {"type": "plane", "origin": origin, "u_vector": u_vector, "v_vector": v_vector},
            "surface": nurbs_record(curve.surface), "path": curve.path, "domain": list(curve.domain),
            "branch": _branch_record(curve)}


def plane_nurbs_from_record(record: object) -> "PlaneNurbsCurve":
    """Decode a ``plane_nurbs_intersection`` record and validate it against the recomputed section.

    The section is recomputed exactly from the plane and the surface (P1); the stored
    algorithm, root count and every root and arc must equal the recomputation, else
    the record is refused with the first differing index (tampering, or a section
    algorithm whose curve points would differ from the ones the document was made with).
    """
    from .nurbs import NurbsSurface, _exact, nurbs_from_record
    from .surfaces import Plane
    _exact(record, _RECORD_FIELDS, "plane_nurbs_intersection")
    if record["type"] != "plane_nurbs_intersection":
        raise GeometryError("not a plane_nurbs_intersection record")
    plane_data = record["plane"]
    _exact(plane_data, _PLANE_FIELDS, "plane_nurbs_intersection plane")
    if plane_data["type"] != "plane":
        raise GeometryError("plane_nurbs_intersection plane must be a plane record")
    try:
        plane = Plane(plane_data["origin"], plane_data["u_vector"], plane_data["v_vector"])
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"plane_nurbs_intersection plane is invalid: {error}") from error
    surface_data = record["surface"]
    # checked before decoding: only a flat nurbs_surface record (no nested records to recurse into)
    if not isinstance(surface_data, dict) or surface_data.get("type") != "nurbs_surface":
        raise GeometryError("plane_nurbs_intersection surface must be a nurbs_surface record")
    try:
        surface = nurbs_from_record(surface_data)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"plane_nurbs_intersection surface is invalid: {error}") from error
    if not isinstance(surface, NurbsSurface):
        raise GeometryError("plane_nurbs_intersection surface must be a nurbs_surface record")
    path = record["path"]
    if type(path) is not int or path < 0:
        raise GeometryError("plane_nurbs_intersection path must be a non-negative integer")
    domain = record["domain"]
    if (not isinstance(domain, list) or len(domain) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in domain)):
        raise GeometryError("plane_nurbs_intersection domain must be two numbers")
    try:
        domain = (float(domain[0]), float(domain[1]))
    except OverflowError as error:
        raise GeometryError("plane_nurbs_intersection domain must be two values in [0, 1]") from error
    if not all(0.0 <= value <= 1.0 for value in domain) or domain[0] == domain[1]:
        raise GeometryError("plane_nurbs_intersection domain must be two distinct values in [0, 1]")
    branch = record["branch"]
    _exact(branch, _BRANCH_FIELDS, "plane_nurbs_intersection branch")
    if branch["algorithm"] != PLANE_NURBS_ALGORITHM:
        raise GeometryError(f"unsupported plane x NURBS section algorithm {branch['algorithm']!r}")
    for name in ("roots", "arcs"):
        if not isinstance(branch[name], list):
            raise GeometryError(f"plane_nurbs_intersection branch {name} must be a list")
    for index, root in enumerate(branch["roots"]):
        if not isinstance(root, list) or len(root) != 4 or type(root[0]) is not int or root[0] not in (0, 1):
            raise GeometryError(f"plane_nurbs_intersection root {index} must be [axis, value, lo, hi]")
        for value in root[1:]:
            _rational_from_text(value, f"plane_nurbs_intersection root {index}")
    for index, arc in enumerate(branch["arcs"]):
        if (not isinstance(arc, list) or len(arc) != 3 or not isinstance(arc[1], list) or len(arc[1]) != 4
                or type(arc[0]) is not int or type(arc[2]) is not int or arc[2] not in (0, 1)):
            raise GeometryError(f"plane_nurbs_intersection arc {index} must be [patch, [u_lo, u_hi, v_lo, v_hi], slicing]")
        for value in arc[1]:
            _rational_from_text(value, f"plane_nurbs_intersection arc {index}")
    curve = PlaneNurbsCurve(_plane_record(plane), surface, path, domain)
    expected = _branch_record(curve)                     # recomputed: refusals of the section surface here
    for name, label in (("roots", "root"), ("arcs", "arc")):
        stored, computed = branch[name], expected[name]
        if len(stored) != len(computed):
            raise GeometryError(f"plane_nurbs_intersection has {len(stored)} {name}, "
                                f"the recomputed section {len(computed)}")
        for index, (left, right) in enumerate(zip(stored, computed)):
            if left != right:
                raise GeometryError(f"plane_nurbs_intersection {label} {index} differs from the recomputed section")
    return curve
