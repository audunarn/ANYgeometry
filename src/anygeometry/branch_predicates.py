"""Curve-level predicates on plane x NURBS section curves (generality slice B1-P4a, architecture 4.4).

A ``PlaneNurbsCurve`` is a chain of arcs, each the graph ``x(s)`` of the zero set of
``G1 = m . H - d1 W`` in a regular box of P1 (``G1`` strictly monotone in ``x``). A
second plane ``n . X = d`` gives ``G2 = n . H - d W`` on the same Bezier sub-patches;
along an arc ``h(s) = G2(s, x(s)) / W`` is the signed (scaled) distance to that plane.

The predicates are statements about the *mathematical* section curve ``C*`` of each
arc (review B1-P4a round 1): for a slicing value ``s`` in the arc's exact gap, the exact
zero of ``G1(s, .)`` in the box; outside the gap (between an arc limit ``s0``/``s1`` and
the true end root), the true end root. ``C*`` is continuous; P2's defined points (bracket
midpoints refined into the plane slab) lie within ``2**-60`` of the box and the slab of it,
far below any model tolerance, and callers compare with a tolerance anyway. Roots are
therefore roots of ``h`` on ``C*``; ``isolated`` means exactly one simple root on ``C*``.

Everything is exact: Bernstein coefficients of ``G1``, ``G2`` and ``W`` over knot
rectangles (de Casteljau splitting of the sub-patch coefficients), signs of ``G2`` at
arc points from exact brackets of the ``G1`` root, and monotonicity of ``h`` from the
interval of ``G2_s G1_x - G2_x G1_s`` over a rectangle that contains the arc piece
(``dh/ds`` has the sign of that expression times ``G1_x``, which has one sign in a
regular box). Undecided cases refuse (``PlaneNurbsNotCertified``).

- ``section_roots(curve, n, d)``: intervals of the curve parameter containing every root
  of ``h`` on the curve's domain; each either holds exactly one simple root
  (``isolated``) or is a tiny unresolved interval where ``h`` vanishes to the sign
  budget at an arc end or a domain end.
- ``certify_line_section(curve, start, end, shared, tolerance)``: whether the straight
  segment ``start -> end`` (in the curve's plane) and the curve meet only within
  ``tolerance`` of the ``shared`` points (``True``), certainly cross elsewhere
  (``False``), or undecided (``None``).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from .plane_nurbs import PlaneNurbsNotCertified, sub_patches_of
from .support_curves import PlaneNurbsCurve, _de_casteljau, _split_at

MAX_DEPTH = 48
MAX_PIECES = 20000                  # per call: a total work bound (review B1-P4a round 1)
_SIGN_STEPS = 200
_PATCHES: dict = {}


@dataclass(frozen=True)
class SectionRoot:
    """A curve-parameter interval (outward doubles) and a world enclosure of the curve over it."""
    t_lo: float
    t_hi: float
    lower: tuple
    upper: tuple
    isolated: bool                  # exactly one simple root of h inside


# ------------------------------------------------------------------ exact Bernstein pieces


def _segment(values, lo: Fraction, hi: Fraction):
    """Bernstein coefficients over [lo, hi] (fractions of [0, 1]) of the polynomial with ``values``."""
    if lo == hi:
        point = _de_casteljau(list(values), lo)
        return [point] * len(values)
    right = _split_at(list(values), lo)[1]
    return _split_at(right, (hi - lo) / (1 - lo) if lo < 1 else Fraction(0))[0]


def _rect(coefficients, u_box, v_box, u_range, v_range):
    """Coefficients ``[u][v]`` over the knot rectangle ``u_range x v_range`` inside the box ranges."""
    (a, b), (c, e) = u_box, v_box
    u0, u1 = (u_range[0] - a) / (b - a), (u_range[1] - a) / (b - a)
    v0, v1 = (v_range[0] - c) / (e - c), (v_range[1] - c) / (e - c)
    rows = [_segment(row, v0, v1) for row in coefficients]                      # along v
    columns = [_segment([row[j] for row in rows], u0, u1) for j in range(len(rows[0]))]   # along u
    return [[columns[j][i] for j in range(len(columns))] for i in range(len(columns[0]))]


def _range(coefficients):
    flat = [value for row in coefficients for value in row]
    return min(flat), max(flat)


def _derivative_ranges(coefficients, u_width, v_width):
    """Ranges of ``d/du`` and ``d/dv`` (knot units) over the rectangle the coefficients describe."""
    p, q = len(coefficients) - 1, len(coefficients[0]) - 1
    du = [p * (coefficients[i + 1][j] - coefficients[i][j]) / u_width
          for i in range(p) for j in range(q + 1)] or [Fraction(0)]
    dv = [q * (coefficients[i][j + 1] - coefficients[i][j]) / v_width
          for i in range(p + 1) for j in range(q)] or [Fraction(0)]
    return (min(du), max(du)), (min(dv), max(dv))


def _multiply(a, b):
    products = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return min(products), max(products)


def _patches(surface, n, d):
    key = (surface, tuple(n), d)
    cached = _PATCHES.get(key)
    if cached is None:
        cached = sub_patches_of(tuple(n), d, surface)
        if len(_PATCHES) >= 32:
            _PATCHES.clear()
        _PATCHES[key] = cached
    return cached


# ------------------------------------------------------------------ the engine


def _proportional(n, d, m, d1) -> bool:
    """Whether ``n . X = d`` and ``m . X = d1`` are the same plane (exact)."""
    first = (*n, d)
    second = (*m, d1)
    pivot = next(i for i, value in enumerate(second) if value != 0)
    ratio = Fraction(first[pivot]) / second[pivot]
    return ratio != 0 and all(Fraction(x) == ratio * y for x, y in zip(first, second))


class _Arc:
    """One arc of the curve with the second plane's and the weight's box polynomials."""

    def __init__(self, curve, arc, g2_patch, w_patch):
        self.curve, self.arc, box = curve, arc, arc.box
        self.u_box, self.v_box = box.u_range, box.v_range
        self.g1 = [list(row) for row in box.coefficients]
        self.g2 = _rect(g2_patch.coefficients, g2_patch.u_range, g2_patch.v_range, box.u_range, box.v_range)
        self.w = _rect(w_patch.coefficients, w_patch.u_range, w_patch.v_range, box.u_range, box.v_range)
        self.slicing = arc.slicing
        self.x_box = box.v_range if arc.slicing == 0 else box.u_range

    def uv(self, s_range, x_range):
        return (s_range, x_range) if self.slicing == 0 else (x_range, s_range)

    def over(self, coefficients, s_range, x_range):
        u_range, v_range = self.uv(s_range, x_range)
        return _rect(coefficients, self.u_box, self.v_box, u_range, v_range)

    def bracket(self, s, width):
        """An exact x interval (knot units) of at most ``width`` containing the arc's x at ``s``, or
        ``None`` when ``s`` lies outside the gap (an end root's region)."""
        g1 = self.column(self.g1, s)
        low, high = _de_casteljau(g1, Fraction(0)), _de_casteljau(g1, Fraction(1))
        x0, x1 = self.x_box
        if low == 0:
            return (x0, x0)
        if high == 0:
            return (x1, x1)
        if (low > 0) == (high > 0):
            return None
        lo, hi = Fraction(0), Fraction(1)
        while (hi - lo) * (x1 - x0) > width:
            middle = (lo + hi) / 2
            value = _de_casteljau(g1, middle)
            if value == 0:
                lo = hi = middle
                break
            if (value > 0) == (low > 0):
                lo = middle
            else:
                hi = middle
        return (x0 + lo * (x1 - x0), x0 + hi * (x1 - x0))

    def rectangle(self, s_lo, s_hi, level=0, parent=None):
        """A knot rectangle (s, x) containing ``C*`` for s in [s_lo, s_hi]; ``parent``: the x range of the
        enclosing piece (or ``None``).

        Where ``G1_s`` has one sign over the piece (with the parent's x range), ``x(s)`` is monotone
        (``dx/ds = -G1_s / G1_x``, ``G1_x`` of one sign in a regular box): the x range is the hull of
        exact brackets at the two ends, each refined to the piece's s width. Otherwise ``_tight_x``
        (exact exclusion, ``24 + level`` halvings capped at 40). Where the piece touches an end
        root's slicing interval, ``C*`` may sit at the true root (outside the gap): the root's
        bracket, refined to the piece's size, is included.
        """
        x_range = None
        if parent is not None and s_hi > s_lo:
            u_range, v_range = self.uv((s_lo, s_hi), parent)
            if u_range[1] > u_range[0] and v_range[1] > v_range[0]:
                g1 = self.over(self.g1, (s_lo, s_hi), parent)
                (g_u, g_v) = _derivative_ranges(g1, u_range[1] - u_range[0], v_range[1] - v_range[0])
                g_s = g_u if self.slicing == 0 else g_v
                if g_s[0] > 0 or g_s[1] < 0:
                    width = (s_hi - s_lo) / 64
                    first, second = self.bracket(s_lo, width), self.bracket(s_hi, width)
                    if first is not None and second is not None:
                        x_range = (min(first[0], second[0]), max(first[1], second[1]))
        if x_range is None and s_hi > s_lo:
            x_range = self.curve._tight_x(self.arc, s_lo, s_hi, depth=min(24 + level, 40))
        s_range = (s_lo, s_hi)
        target = s_hi - s_lo if s_hi > s_lo else (self.x_box[1] - self.x_box[0]) / 2 ** (60 + level)
        for root in (self.arc.first, self.arc.second):
            (rs, _rx) = self.curve._ordered_root_rect(self.arc, root)
            if rs[1] < s_lo or rs[0] > s_hi:
                continue                                        # C* cannot sit at this root here
            (bs, bx) = self.root_bracket(root, target)
            s_range = (min(s_range[0], bs[0]), max(s_range[1], bs[1]))
            x_range = bx if x_range is None else (min(x_range[0], bx[0]), max(x_range[1], bx[1]))
        if x_range is None:
            x_range = self.x_box
        return s_range, x_range

    def column(self, coefficients, s):
        """Coefficients along x of ``coefficients`` on the line s = const (exact)."""
        a, b = self.u_box if self.slicing == 0 else self.v_box
        t = (s - a) / (b - a)
        if self.slicing == 0:
            return [_de_casteljau([row[j] for row in coefficients], t) for j in range(len(coefficients[0]))]
        return [_de_casteljau(list(row), t) for row in coefficients]

    def sign_at(self, s):
        """Sign of G2 at the arc's point with slicing coordinate ``s`` (0: undecided to the budget)."""
        g1, g2 = self.column(self.g1, s), self.column(self.g2, s)
        low, high = _de_casteljau(g1, Fraction(0)), _de_casteljau(g1, Fraction(1))
        if low == 0 or high == 0:
            value = _de_casteljau(g2, Fraction(0) if low == 0 else Fraction(1))
            return (value > 0) - (value < 0)
        if (low > 0) == (high > 0):
            # outside the gap: C* sits at the end root whose slicing interval holds s
            for root in (self.arc.first, self.arc.second):
                (rs, _rx) = self.curve._ordered_root_rect(self.arc, root)
                if rs[0] <= s <= rs[1]:
                    return self.root_sign(root)
            raise PlaneNurbsNotCertified("section point outside its arc's gap and its end roots")
        lo, hi = Fraction(0), Fraction(1)
        for _ in range(_SIGN_STEPS):
            values = _segment(g2, lo, hi)
            if all(value > 0 for value in values):
                return 1
            if all(value < 0 for value in values):
                return -1
            middle = (lo + hi) / 2
            value = _de_casteljau(g1, middle)
            if value == 0:
                lo = hi = middle
                g2_value = _de_casteljau(g2, middle)
                return (g2_value > 0) - (g2_value < 0)
            if (value > 0) == (low > 0):
                lo = middle
            else:
                hi = middle
        return 0

    def _edge(self, root):
        """The root's edge line: ``(g1, g2, fixed, varying_is_s, lo, hi, box_lo, box_hi)`` with ``g1``/``g2``
        restricted to the line (Bernstein in the line's local parameter) and the root's local interval."""
        (s_lo, s_hi), (x_lo, x_hi) = self.curve._ordered_root_rect(self.arc, root)
        if s_lo == s_hi:                                       # on a constant-s edge: x varies
            g1, g2 = self.column(self.g1, s_lo), self.column(self.g2, s_lo)
            box_lo, box_hi = self.x_box
            fixed, varying_is_s, lo, hi = s_lo, False, x_lo, x_hi
        else:                                                  # on a constant-x edge: s varies
            a, b = self.x_box
            t = (x_lo - a) / (b - a)
            if self.slicing == 0:
                g1 = [_de_casteljau(list(row), t) for row in self.g1]
                g2 = [_de_casteljau(list(row), t) for row in self.g2]
            else:
                g1 = [_de_casteljau([row[j] for row in self.g1], t) for j in range(len(self.g1[0]))]
                g2 = [_de_casteljau([row[j] for row in self.g2], t) for j in range(len(self.g2[0]))]
            box_lo, box_hi = self.u_box if self.slicing == 0 else self.v_box
            fixed, varying_is_s, lo, hi = x_lo, True, s_lo, s_hi
        width = box_hi - box_lo
        return g1, g2, fixed, varying_is_s, (lo - box_lo) / width, (hi - box_lo) / width, box_lo, width

    def root_bracket(self, root, target):
        """An ordered (s, x) rectangle containing the true end root, its isolating interval refined
        (exact bisection by the sign of G1 on the edge) to at most ``target`` in knot units."""
        g1, _g2, fixed, varying_is_s, lo, hi, box_lo, width = self._edge(root)
        g1_lo = _de_casteljau(g1, lo)
        steps = 0
        while lo < hi and g1_lo != 0 and (hi - lo) * width > target and steps < _SIGN_STEPS:
            middle = (lo + hi) / 2
            value = _de_casteljau(g1, middle)
            if value == 0:
                lo = hi = middle
                break
            if (value > 0) == (g1_lo > 0):
                lo, g1_lo = middle, value
            else:
                hi = middle
            steps += 1
        if lo < hi and g1_lo == 0:
            hi = lo
        interval = (box_lo + lo * width, box_lo + hi * width)
        return (interval, (fixed, fixed)) if varying_is_s else ((fixed, fixed), interval)

    def root_sign(self, root):
        """Sign of G2 at the true end root (its bracket refined by the sign of G1); 0: undecided."""
        g1, g2, _fixed, _varying, lo, hi, _box_lo, _width = self._edge(root)

        def sign(lo, hi):
            values = _segment(g2, lo, hi)
            return 1 if all(value > 0 for value in values) else -1 if all(value < 0 for value in values) else 0

        g1_lo = _de_casteljau(g1, lo)
        if lo == hi or g1_lo == 0:
            return sign(lo, lo)
        for _ in range(_SIGN_STEPS):
            decided = sign(lo, hi)
            if decided:
                return decided
            middle = (lo + hi) / 2
            value = _de_casteljau(g1, middle)
            if value == 0:
                return sign(middle, middle)
            if (value > 0) == (g1_lo > 0):
                lo, g1_lo = middle, value
            else:
                hi = middle
        return 0

    def monotone(self, s_range, x_range) -> bool:
        """``dh/ds`` certified non-zero over the rectangle."""
        u_range, v_range = self.uv(s_range, x_range)
        u_width, v_width = u_range[1] - u_range[0], v_range[1] - v_range[0]
        if u_width == 0 or v_width == 0:
            return False
        g1, g2 = self.over(self.g1, s_range, x_range), self.over(self.g2, s_range, x_range)
        (g1_u, g1_v), (g2_u, g2_v) = _derivative_ranges(g1, u_width, v_width), _derivative_ranges(g2, u_width, v_width)
        g1_s, g1_x, g2_s, g2_x = (g1_u, g1_v, g2_u, g2_v) if self.slicing == 0 else (g1_v, g1_u, g2_v, g2_u)
        first, second = _multiply(g2_s, g1_x), _multiply(g2_x, g1_s)
        low, high = first[0] - second[1], first[1] - second[0]
        return low > 0 or high < 0

    def world(self, s_range, x_range):
        """Outward double enclosure of the surface over the rectangle (lower, upper)."""
        enclosure = self.curve._chart_box_enclosure(s_range, x_range, self.slicing)
        return tuple(enclosure.lower), tuple(enclosure.upper)


def _pieces(curve):
    """Per arc overlapping the curve's domain: (index, arc, s_lo, s_hi, lo_is_curve_end, hi_is_curve_end,
    lo_is_root, hi_is_root) with exact slicing limits in the arc's s order."""
    arcs = curve._arcs()
    count = len(arcs)
    a, b = (Fraction(value) for value in curve.domain)
    low, high = min(a, b) * count, max(a, b) * count
    result = []
    for k, arc in enumerate(arcs):
        tau0, tau1 = max(low, Fraction(k)) - k, min(high, Fraction(k + 1)) - k
        if tau0 >= tau1:
            continue                                       # no overlap of positive length
        s_a, s_b = arc.s0 + tau0 * (arc.s1 - arc.s0), arc.s0 + tau1 * (arc.s1 - arc.s0)
        result.append([k, arc, tau0, tau1, s_a, s_b])
    return result, count, a, b


def _engine(curve, n, d, classify, max_depth):
    """Run ``classify(arc_view, s_range, x_range, world, depth, signs)`` over every arc piece of the curve
    (exact subdivision in s); ``classify`` returns "done", "split" or raises."""
    from .nurbs import NurbsSurface
    surface: NurbsSurface = curve.surface
    m, d1, _mm, _limit, _l = curve._plane_slab()
    if _proportional(n, d, m, d1):
        raise PlaneNurbsNotCertified("the second plane is the curve's own plane (coincident)")
    g2_patches = _patches(surface, n, d)
    w_patches = _patches(surface, (Fraction(0),) * 3, Fraction(-1))
    pieces, count, a, b = _pieces(curve)
    examined = 0
    for k, arc, tau0, tau1, s_a, s_b in pieces:
        view = _Arc(curve, arc, g2_patches[arc.box.patch], w_patches[arc.box.patch])
        # signs at the piece ends: arc-end roots or interior slicing values
        def end_sign(tau, s, root):
            return view.root_sign(root) if tau in (0, 1) else view.sign_at(s)
        sign_a = end_sign(tau0, s_a, arc.first if tau0 == 0 else arc.second)
        sign_b = end_sign(tau1, s_b, arc.second if tau1 == 1 else arc.first)
        # depth is measured against the arc: a piece of a short sub-curve starts deep (an undecided
        # tiny piece is then reported or refused at once instead of being split as a full tree)
        span, width = abs(arc.s1 - arc.s0), abs(s_b - s_a)
        start = 0
        while start < max_depth - 1 and width * 2 ** (start + 1) <= span:
            start += 1
        stack = [(s_a, s_b, sign_a, sign_b, start, None)]
        while stack:
            s_from, s_to, sign_from, sign_to, depth, parent = stack.pop()
            examined += 1
            if examined > MAX_PIECES:
                raise PlaneNurbsNotCertified(f"section predicate budget of {MAX_PIECES} pieces exhausted")
            lo, hi = min(s_from, s_to), max(s_from, s_to)
            s_range, x_range = view.rectangle(lo, hi, depth, parent)
            context = _Context(view, k, count, a, b, s_from, s_to, s_range, x_range, sign_from, sign_to, depth)
            verdict = classify(context)
            if verdict == "done":
                continue
            if depth >= max_depth:
                raise PlaneNurbsNotCertified("section predicate undecided at depth (tangency or contact)")
            for fraction in (Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4)):
                middle = s_from + fraction * (s_to - s_from)
                sign_middle = view.sign_at(middle)
                if sign_middle != 0:
                    break
            stack.append((middle, s_to, sign_middle, sign_to, depth + 1, x_range))
            stack.append((s_from, middle, sign_from, sign_middle, depth + 1, x_range))


@dataclass
class _Context:
    view: _Arc
    k: int
    count: int
    a: Fraction
    b: Fraction
    s_from: Fraction
    s_to: Fraction
    s_range: tuple
    x_range: tuple
    sign_from: int
    sign_to: int
    depth: int

    def g2_range(self):
        return _range(self.view.over(self.view.g2, self.s_range, self.x_range))

    def w_max(self):
        return _range(self.view.over(self.view.w, self.s_range, self.x_range))[1]

    def world(self):
        return self.view.world(self.s_range, self.x_range)

    def t_interval(self):
        """Outward double interval of the curve parameter for s in [s_from, s_to]."""
        arc = self.view.arc
        values = []
        for s in (self.s_from, self.s_to):
            tau = (s - arc.s0) / (arc.s1 - arc.s0)
            path = (self.k + tau) / self.count
            values.append((path - self.a) / (self.b - self.a))
        low, high = min(values), max(values)
        return max(0.0, math.nextafter(float(low), -math.inf)), min(1.0, math.nextafter(float(high), math.inf))


# ------------------------------------------------------------------ predicates


def section_roots(curve: PlaneNurbsCurve, n, d, *, max_depth: int = MAX_DEPTH,
                  refine: int = 40) -> tuple[SectionRoot, ...]:
    """Intervals containing every root of ``n . C(t) = d`` on the curve (see the module docstring);
    isolated roots are refined by exact bisection to ``refine`` halvings of their arc piece."""
    n = tuple(Fraction(value) for value in n)
    d = Fraction(d)
    if not any(n):
        raise PlaneNurbsNotCertified("the second plane needs a non-zero normal")
    found: list[SectionRoot] = []

    def classify(context: _Context) -> str:
        low, high = context.g2_range()
        if low == high == 0:
            raise PlaneNurbsNotCertified("the second plane contains a piece of the curve (coincident)")
        if low > 0 or high < 0:
            return "done"                                      # no root: G2 has one sign over the piece
        signs = (context.sign_from, context.sign_to)
        if 0 not in signs and context.view.monotone(context.s_range, context.x_range):
            if signs[0] != signs[1]:
                if context.depth < min(refine, max_depth - 1):
                    return "split"                              # refine the isolated root
                t_lo, t_hi = context.t_interval()
                lower, upper = context.world()
                found.append(SectionRoot(t_lo, t_hi, lower, upper, True))
            return "done"
        if 0 in signs and context.depth >= max_depth - 1:
            t_lo, t_hi = context.t_interval()
            lower, upper = context.world()
            found.append(SectionRoot(t_lo, t_hi, lower, upper, False))
            return "done"
        return "split"

    _engine(curve, n, d, classify, max_depth)
    return tuple(sorted(found, key=lambda item: (item.t_lo, item.t_hi)))


def _plane_normal(curve: PlaneNurbsCurve):
    m, _d, _mm, _limit, _l = curve._plane_slab()
    return m


def certify_line_section(curve: PlaneNurbsCurve, start, end, shared=(), tolerance: float = 0.0,
                         *, max_depth: int = MAX_DEPTH):
    """Whether the segment ``start -> end`` (in the curve's plane) and the curve intersect only within
    ``tolerance`` of the ``shared`` points: ``True``; a certain intersection elsewhere on the segment:
    ``False``; undecided (a touching contact, an intersection near an unshared segment end): ``None``.

    Intersections are the roots of ``h`` (signed distance to the plane through the segment, normal
    to the curve's plane) whose points lie on the segment, as the existing line-pair predicates
    decide intersection points (not proximity). A curve piece is cleared when ``G2`` has one sign
    over it (no root), when its world box lies within the tolerance of a shared point, or when the
    box projects outside the segment (``lambda`` outside [0, 1]); a monotone piece with equal end
    signs has no root. A simple root (monotone piece, opposite end signs) whose box projects
    strictly inside the segment and is not near a shared point is an intersection. A piece smaller
    than ``tolerance / 16`` that is still undecided refuses (touching at the tolerance scale).
    """
    a = [Fraction(float(value)) for value in start]
    b = [Fraction(float(value)) for value in end]
    e = [y - x for x, y in zip(a, b)]
    m = _plane_normal(curve)
    n = (e[1] * m[2] - e[2] * m[1], e[2] * m[0] - e[0] * m[2], e[0] * m[1] - e[1] * m[0])
    if not any(n):
        return None
    d = sum(x * y for x, y in zip(n, a))
    tol = Fraction(float(tolerance))
    # precondition: the segment lies in the curve's plane within the tolerance (review B1-P4a round 1:
    # otherwise h's roots are only projected crossings)
    _m, d1, mm, _limit, _l = curve._plane_slab()
    if any((sum(x * y for x, y in zip(m, end_point)) - d1) ** 2 > tol * tol * mm for end_point in (a, b)):
        return None
    points = [[Fraction(float(value)) for value in point] for point in shared]
    ee = sum(value * value for value in e)
    crossing = [False]

    def corners(lower, upper):
        return [[Fraction(lower[k] if (index >> k) & 1 == 0 else upper[k]) for k in range(3)] for index in range(8)]

    def near_shared(box):
        return any(all(sum((c - p) ** 2 for c, p in zip(corner, point)) <= tol * tol for corner in box)
                   for point in points)

    def away_from_shared(lower, upper):
        # every point of the box farther than the tolerance from every shared point
        for point in points:
            gap = sum(max(Fraction(lo) - p, p - Fraction(hi), Fraction(0)) ** 2
                      for lo, hi, p in zip(lower, upper, point))
            if gap <= tol * tol:
                return False
        return True

    def classify(context: _Context) -> str:
        low, high = context.g2_range()
        if low == high == 0:
            raise PlaneNurbsNotCertified("the segment's plane contains a piece of the curve (coincident)")
        if low > 0 or high < 0:
            return "done"                                       # no root of h on this piece
        lower, upper = context.world()
        box = corners(lower, upper)
        if near_shared(box):
            return "done"
        lambdas = [sum((c - x) * y for c, x, y in zip(corner, a, e)) / ee for corner in box]
        if max(lambdas) < 0 or min(lambdas) > 1:
            return "done"                                       # roots here are off the segment
        signs = (context.sign_from, context.sign_to)
        if 0 not in signs and context.view.monotone(context.s_range, context.x_range):
            if signs[0] == signs[1]:
                return "done"                                   # monotone without a sign change: no root
            # (an isolated root not yet separated from the joints or the segment ends: refine)
            if min(lambdas) > 0 and max(lambdas) < 1 and away_from_shared(lower, upper):
                crossing[0] = True                              # a simple root on the segment, off the joints
                return "done"
        diameter = sum((Fraction(hi) - Fraction(lo)) ** 2 for lo, hi in zip(lower, upper))
        if tol > 0 and diameter * 256 <= tol * tol:
            raise PlaneNurbsNotCertified("line x section contact undecided at the tolerance scale")
        return "split"

    try:
        _engine(curve, n, d, classify, max_depth)
    except PlaneNurbsNotCertified:
        return False if crossing[0] else None
    return False if crossing[0] else True
