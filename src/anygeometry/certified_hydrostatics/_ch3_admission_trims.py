"""CH3a1: trims, corners and hints of a curved face (design CH3 section 2.3: C2, C3, H1, H2).

``admit_curved_face`` takes a face with a NURBS support and decides, in this order: the class and chart (C1, C4,
``_ch3_admission.classify_surface``); that the face has one simple chain of edges, every edge a
``PlaneNurbsCurve`` on the face's own support with equal ranges (C2, ``UNSUPPORTED_TRIM`` otherwise); the exact
graph ``r_k = N_k / D_k`` of every trim plane; the corner of every vertex (C3): a root ``u*`` of
``N_i D_j - N_j D_i`` with ``u_0 < u* < u_1`` and ``v_0 < r_i(u*) < v_1``, chosen by the vertex hint H1 (exactly one
candidate within ``delta_hint`` of the stored position, else ``TRIM_HINT_MISMATCH``); every edge's exact
``u``-interval between its corners, whole-edge range test and one-signed ``D_k`` (``graph_in_range``); and the
domain, path and direction hints H2 of the stored curve (``delta_chart``).

The stored vertex positions and edge domains are hints: they only select among exact candidates and are checked
against them; the certified solid uses the exact edges (design 3.1). Exact values pass through the bit-capped
``Ar``; every loop reserves a unit per element. The library evaluation of the stored curve for H2
(``PlaneNurbsCurve._knot_points``) is one native call per edge, charged three ``admission_element`` units before
it runs (a documented limit of the metering)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..curves import Straight
from ..errors import GeometryError
from ..nurbs import CurveOnSurface, NurbsSurface
from ..support_curves import PlaneNurbsCurve
from ._ch3_admission import RuledSupport, TrimPlane, classify_surface, exact_float, plane_graph
from ._ch3_algebraic import AlgebraicNumber, Enclosure, compare, isolate_roots, sign_at
from ._ch3_poly import Kernel, bernstein, padd, peval, pmul, pscale, psub, refusal
from ._ch3_range import graph_in_range
from ._errors import RefusalCategory
from ._work import Meter

__all__ = ["Corner", "CurvedTrim", "CurvedFace", "admit_curved_face", "admit_curved_faces", "rational_enclosure",
           "enclose_coordinates"]

UT = RefusalCategory.UNSUPPORTED_TRIM
MISMATCH = RefusalCategory.TRIM_HINT_MISMATCH


@dataclass(frozen=True)
class Corner:
    """A vertex of a curved face: the root ``number`` (``poly``/``ordinal`` identify it) of the incident trims'
    difference polynomial and ``hint_error``, a certified upper bound of the max-norm distance between the exact
    corner's world position and the stored vertex position (``<= delta_hint``)."""

    vertex: int
    number: AlgebraicNumber
    hint_error: Fraction

    @property
    def poly(self) -> tuple:
        return self.number.poly

    @property
    def ordinal(self) -> int:
        return self.number.ordinal


@dataclass(frozen=True)
class CurvedTrim:
    """One edge of a curved face's loop. ``start_vertex``/``end_vertex`` are the stored edge's ends (curve
    direction); ``u_lo < u_hi`` the exact ``u``-interval; ``curve_increasing``: the stored curve runs towards
    increasing ``u``; ``loop_increasing``: the loop direction does; ``range_method`` the whole-edge range test
    that decided (``"bernstein"`` or ``"exact"``); ``chart_errors`` the certified upper bounds ``(start, end,
    mid)`` of the H2 distances in knot units."""

    edge: int
    forward: bool
    plane: TrimPlane
    start_vertex: int
    end_vertex: int
    u_lo: AlgebraicNumber
    u_hi: AlgebraicNumber
    curve_increasing: bool
    loop_increasing: bool
    range_method: str
    chart_errors: tuple


@dataclass(frozen=True)
class CurvedFace:
    face: int
    support: RuledSupport
    trims: tuple            # CurvedTrim in loop order
    corners: dict           # vertex id -> Corner


# ------------------------------------------------------------------ enclosures of rational functions at a root

def _bounds(k: Kernel, p: tuple, lo, hi):
    low = high = None
    for c in bernstein(k, p, lo, hi):
        k.tick("bernstein_coefficient")
        if low is None or c < low:
            low = c
        if high is None or c > high:
            high = c
    return low, high


def rational_enclosure(k: Kernel, num: tuple, den: tuple, x) -> Enclosure:
    """An enclosure of ``num(x) / den(x)`` for a rational or algebraic ``x``; ``x`` is refined until ``den`` has
    one-signed Bernstein coefficients over its isolating interval. ``UNSUPPORTED_TRIM / denominator_vanishes``
    if ``den(x) = 0`` for a rational ``x``."""
    ar = k.ar
    while True:
        if isinstance(x, AlgebraicNumber) and x.is_rational:
            x = x.lo
        if not isinstance(x, AlgebraicNumber):
            d = peval(k, den, Fraction(x))
            if d == 0:
                raise refusal(UT, "the graph denominator vanishes at a corner", "denominator_vanishes")
            return Enclosure.exact(ar.div(peval(k, num, Fraction(x)), d))
        dl, dh = _bounds(k, den, x.lo, x.hi)
        if dl is not None and (dl > 0 or dh < 0):
            nl, nh = _bounds(k, num, x.lo, x.hi)
            quotients = [ar.div(n, d) for n in (nl, nh) for d in (dl, dh)]
            return Enclosure(min(quotients), max(quotients))
        x.refine(k)


def enclose_coordinates(k: Kernel, nums: tuple, den: tuple, x, width: Fraction) -> tuple:
    """Enclosures of ``num_c(x) / den(x)`` for each ``num_c``, ``x`` refined until each is at most ``width`` wide."""
    while True:
        k.tick("admission_element")
        encs = tuple(rational_enclosure(k, n, den, x) for n in nums)
        if all(e.width <= width for e in encs) or (not isinstance(x, AlgebraicNumber)) or x.is_rational:
            return encs
        x.refine(k)


def _distance(stored: Fraction, enc: Enclosure):
    """``(least, greatest)`` of ``|stored - t|`` over ``t`` in the enclosure."""
    far = max(abs(stored - enc.lo), abs(stored - enc.hi))
    near = Fraction(0) if enc.lo <= stored <= enc.hi else min(abs(stored - enc.lo), abs(stored - enc.hi))
    return near, far


def _delta(k: Kernel, bits: int, scale: Fraction) -> Fraction:
    k.ar.guard(bits)
    return k.ar.fit(Fraction(scale, 1 << bits))


# ------------------------------------------------------------------ C3: corners and H1

def _candidates(k: Kernel, support: RuledSupport, a: TrimPlane, b: TrimPlane) -> list:
    """Roots ``u*`` of ``N_a D_b - N_b D_a`` in the open ``u``-range with ``v_0 < r_a(u*) < v_1`` (exact)."""
    R = psub(k, pmul(k, a.N, b.D), pmul(k, b.N, a.D))
    if not R:
        raise refusal(UT, "two adjacent trims lie on the same graph", "coincident_trims")
    v0, v1 = support.v_range
    below = psub(k, a.N, pscale(k, a.D, v0))            # N - v0 D
    above = psub(k, pscale(k, a.D, v1), a.N)            # v1 D - N
    out = []
    for r in isolate_roots(k, R, support.u_range[0], support.u_range[1]):
        k.tick("admission_element")
        s = sign_at(k, a.D, r)
        if s == 0 or sign_at(k, below, r) * s <= 0 or sign_at(k, above, r) * s <= 0:
            continue
        out.append(r)
    return out


def _select_corner(k: Kernel, delta_hint_bits: int, support: RuledSupport, a: TrimPlane, b: TrimPlane,
                   vertex: int, stored: tuple) -> Corner:
    """H1: the unique candidate within ``delta_hint`` of ``stored`` per coordinate."""
    candidates = _candidates(k, support, a, b)
    if not candidates:
        raise refusal(UT, f"vertex {vertex} is not the corner of its two trims inside the range box",
                      "no_corner_in_range_box")
    biggest = max(abs(c) for c in stored)
    delta = _delta(k, delta_hint_bits, 1 + biggest)
    matches = []
    for r in candidates:
        k.tick("admission_element")
        encs = enclose_coordinates(k, a.E, a.M, r, delta / 4)
        spans = [_distance(s, e) for s, e in zip(stored, encs)]
        if all(far <= delta for _near, far in spans):
            matches.append((r, max(far for _near, far in spans)))
        elif all(near <= delta for near, _far in spans):
            raise refusal(MISMATCH, f"vertex {vertex}: a corner is not certified inside or outside delta_hint",
                          "ambiguous_hint")
    if len(matches) != 1:
        raise refusal(MISMATCH, f"vertex {vertex}: {len(matches)} corner candidates lie within delta_hint of the "
                      "stored position", "vertex_hint" if not matches else "ambiguous_candidates")
    return Corner(vertex, matches[0][0], matches[0][1])


# ------------------------------------------------------------------ H2

def _check_domain(k: Kernel, delta_chart_bits: int, curve: PlaneNurbsCurve, plane: TrimPlane, start: Corner,
                  end: Corner, u_lo: AlgebraicNumber, u_hi: AlgebraicNumber) -> tuple:
    """H2 (design 2.3): the stored curve's knot points at path parameters 0, 1/2, 1; returns the certified
    distance bounds ``(start, end, mid)``."""
    for _ in range(3):
        k.tick("admission_element")
    try:
        us, vs = curve._knot_points(np.array([0.0, 0.5, 1.0]))
    except (GeometryError, ArithmeticError, ValueError) as error:
        raise refusal(UT, "the stored curve cannot be evaluated: " + str(error), "curve_evaluation") from error
    points = [(exact_float(k, us[i]), exact_float(k, vs[i])) for i in range(3)]
    delta = _delta(k, delta_chart_bits, Fraction(1))
    errors = []
    for (pu, pv), corner, name in ((points[0], start, "start"), (points[2], end, "end")):
        # the exact corner (u*, N(u*) / D(u*)) enclosed to delta / 4 in both knot coordinates
        x = corner.number
        while True:
            k.tick("admission_element")
            ev = rational_enclosure(k, plane.N, plane.D, x)
            if (x.hi - x.lo <= delta / 8 or x.is_rational) and ev.width <= delta / 4:
                break
            x.refine(k)
        eu = x.interval()
        far = max(_distance(pu, eu)[1], _distance(pv, ev)[1])
        if far > delta:
            raise refusal(MISMATCH, f"the stored curve's {name} knot point is not within delta_chart of its vertex's "
                          "exact corner (domain, path or direction)", f"domain_{name}")
        errors.append(far)
    um, vm = points[1]
    if compare(k, u_lo, um) > 0 or compare(k, um, u_hi) > 0:
        raise refusal(MISMATCH, "the stored curve's midpoint lies outside the exact u-interval", "domain_mid")
    d = peval(k, plane.D, um)
    if d == 0:
        raise refusal(UT, "the graph denominator vanishes at the stored midpoint", "denominator_vanishes")
    mid = abs(vm - k.ar.div(peval(k, plane.N, um), d))
    if mid > delta:
        raise refusal(MISMATCH, "the stored curve's midpoint is off the exact graph", "domain_mid")
    errors.append(mid)
    return tuple(errors)


# ------------------------------------------------------------------ the face

def _trim_curve(model, edge_id: int, support: RuledSupport, surface: NurbsSurface):
    edge = model.edges.get(edge_id)
    if edge is None:
        raise refusal(RefusalCategory.INVALID_LOOP, f"the loop uses the unknown edge {edge_id}")
    curve = edge.curve
    if isinstance(curve, Straight):
        raise refusal(UT, f"edge {edge_id} of a curved face is straight", "straight_edge")
    if isinstance(curve, CurveOnSurface):
        raise refusal(UT, f"edge {edge_id} of a curved face is an iso-parametric curve on a surface",
                      "curve_on_surface")
    if not isinstance(curve, PlaneNurbsCurve):
        raise refusal(UT, f"edge {edge_id} of a curved face is a {type(curve).__name__}", "other_curve")
    if curve.surface.support_key() != support.key or curve.surface.u_range != surface.u_range \
            or curve.surface.v_range != surface.v_range:
        raise refusal(UT, f"edge {edge_id} lies on another support or range than its face", "support_mismatch")
    return edge, curve


def admit_curved_face(k: Kernel, model, face_id: int, policy) -> CurvedFace:
    """C1-C4, H1 and H2 for one face with a NURBS support (see the module docstring)."""
    face = model.faces.get(face_id)
    if face is None:
        raise refusal(RefusalCategory.INVALID_QUERY, f"face {face_id} is not in the model")
    surface = face.surface
    support = classify_surface(k, surface)
    if face.holes:
        raise refusal(RefusalCategory.INVALID_HOLES, f"curved face {face_id} has holes")
    items = face.loop
    if len(items) < 3:
        raise refusal(RefusalCategory.INVALID_LOOP, f"the loop of curved face {face_id} has fewer than three edges",
                      "loop_too_short")
    edges, curves, planes, starts, ends = [], [], [], [], []
    for item in items:
        k.tick("admission_element")
        edge, curve = _trim_curve(model, item.edge, support, surface)
        origin, u_vec, v_vec = ([exact_float(k, c) for c in vector] for vector in curve.plane_record)
        planes.append(plane_graph(k, support, tuple(origin), tuple(u_vec), tuple(v_vec)))
        edges.append(edge)
        curves.append(curve)
        starts.append(edge.start if item.forward else edge.end)
        ends.append(edge.end if item.forward else edge.start)
    count = len(items)
    if len(set(starts)) != count:
        raise refusal(RefusalCategory.INVALID_LOOP, f"a vertex repeats in the loop of curved face {face_id}")
    for i in range(count):
        k.tick("admission_element")
        if ends[i] != starts[(i + 1) % count]:
            raise refusal(RefusalCategory.INVALID_LOOP, f"the loop of curved face {face_id} is not a chain")
    corners = {}
    for i in range(count):
        k.tick("admission_element")
        vertex = starts[i]
        record = model.vertices.get(vertex)
        if record is None:
            raise refusal(RefusalCategory.INVALID_LOOP, f"the loop ends at the missing vertex {vertex}")
        stored = tuple(exact_float(k, c) for c in record.position)
        corners[vertex] = _select_corner(k, policy.delta_hint_bits, support, planes[i - 1], planes[i], vertex, stored)
    trims = []
    for i, item in enumerate(items):
        k.tick("admission_element")
        edge, curve, plane = edges[i], curves[i], planes[i]
        a, b = corners[edge.start], corners[edge.end]
        order = compare(k, a.number, b.number)
        if order == 0:
            raise refusal(UT, f"edge {item.edge} has equal corners at both ends", "degenerate_edge")
        lo, hi = (a.number, b.number) if order < 0 else (b.number, a.number)
        # both corners are roots in the open u-range (candidates), so [lo, hi] lies strictly inside it
        method = graph_in_range(k, plane.N, plane.D, lo, hi, support.v_range[0], support.v_range[1])
        errors = _check_domain(k, policy.delta_chart_bits, curve, plane, a, b, lo, hi)
        increasing = order < 0
        trims.append(CurvedTrim(item.edge, bool(item.forward), plane, edge.start, edge.end, lo, hi, increasing,
                                increasing == bool(item.forward), method, errors))
    return CurvedFace(face_id, support, tuple(trims), corners)


def admit_curved_faces(model, face_ids, meter: Meter, policy) -> dict:
    """``{face id: CurvedFace}`` for the given faces (each must have a NURBS support): the curved part of
    admission. Reserves one ``face_read`` unit per face."""
    k = Kernel(meter, policy.kernel_limits())
    result = {}
    for face_id in face_ids:
        meter.unit("face_read")
        result[int(face_id)] = admit_curved_face(k, model, int(face_id), policy)
    return result
