"""Early exact detector for coincident, overlapping and shared-boundary pairs (generic route only).

Returns ``None`` (silent: the generic atlas runs unchanged) unless one exact Fraction proof succeeds:

R1 ``COINCIDENT`` - same map. Equal degrees, clamped knot vectors with equal end knots; after exact Boehm
   insertion of the union of interior knots (homogeneous coordinates, each net divided by its first weight)
   the two nets are equal, and the selected rectangles overlap with positive area.  Then A(u,v) = B(u,v) on
   that rectangle, so the parameter zero set contains a two-dimensional piece.
R2 ``COPLANAR_OVERLAP`` - both surfaces have uniform weights and controls exactly equal to an affine function
   of their Greville abscissae (linear precision: the surface IS that affine map), both maps are
   non-degenerate, lie in one plane, and the images of the selected rectangles overlap with positive area
   (exact convex clipping in A's parameters).  The zero set again contains a two-dimensional piece.
R3 ``SHARED_BOUNDARY_CURVE`` - both surfaces select their full clamped knot domains and an edge curve of A
   equals an edge curve of B exactly (degree, knots, controls, weights; either direction) and is not a point.

R1/R2 are engine independent: no finite curve accounting (CERTIFIED) and no DISJOINT is correct for a
two-dimensional zero set.  R3 rests on the atlas's own rules: exclusions are sound, and a regular-graph
witness needs strict Krawczyk inclusion of its three unknowns in a region clipped to the selected supports.
A common edge point has one coordinate of EACH surface at a clipped bound, so for every free axis one
unknown is pinned at a region bound; no witness region can contain it and the cover never completes, i.e.
the previous result was UNCLASSIFIED as well (after the box budget).  Everything else stays silent.
"""
from __future__ import annotations

from fractions import Fraction

from ._source import UnclassifiedReason

__all__ = ["detect_coincidence"]

_MAX_INSERTIONS = 64            # R1 work bound per surface; beyond it the detector stays silent


def _clamped(knots, degree):
    return len(set(knots[:degree + 1])) == 1 and len(set(knots[-degree - 1:])) == 1


def _homogeneous(controls, weights):
    if weights is None:
        net = [[tuple(Fraction(x) for x in p) + (Fraction(1),) for p in row] for row in controls]
    else:
        net = [[tuple(Fraction(x) * Fraction(w) for x in p) + (Fraction(w),) for p, w in zip(row, wrow)]
               for row, wrow in zip(controls, weights)]
    first = net[0][0][3]
    if any(p[3] <= 0 for row in net for p in row):
        return None
    return [[tuple(x / first for x in p) for p in row] for row in net]


def _insert(knots, degree, points, x):
    """Exact Boehm insertion of interior knot ``x`` into one curve (homogeneous points)."""
    k = max(i for i, value in enumerate(knots) if value <= x)
    out = []
    for i in range(len(points) + 1):
        if i <= k - degree:
            out.append(points[i])
        elif i > k:
            out.append(points[i - 1])
        else:
            a = (x - knots[i]) / (knots[i + degree] - knots[i])
            out.append(tuple(a * p + (1 - a) * q for p, q in zip(points[i], points[i - 1])))
    return knots[:k + 1] + [x] + knots[k + 1:], out


def _refine(knots, degree, net, target, along_u):
    """Insert the knots of ``target`` missing from ``knots`` into every row (u) or column (v) of ``net``."""
    missing = []
    for value in sorted(set(target)):
        missing += [value] * max(0, target.count(value) - knots.count(value))
    if len(missing) > _MAX_INSERTIONS:
        return None, None
    # net[i][j]: a u-curve is a column (fixed j), a v-curve is a row (fixed i)
    lines = [list(col) for col in zip(*net)] if along_u else [list(row) for row in net]
    for x in missing:
        refined = [_insert(knots, degree, line, x) for line in lines]
        knots, lines = refined[0][0], [line for _, line in refined]
    net = [list(row) for row in zip(*lines)] if along_u else lines
    return knots, net


def _union(a, b):
    """Sorted knot multiset with the larger multiplicity of every value."""
    return [v for v in sorted(set(a) | set(b)) for _ in range(max(a.count(v), b.count(v)))]


def _overlap(range_a, range_b):
    return max(range_a[0], range_b[0]) < min(range_a[1], range_b[1])


def _same_map(first, second):
    pu, pv, ku, kv, ca, wa, ur_a, vr_a = first
    qu, qv, lu, lv, cb, wb, ur_b, vr_b = second
    if (pu, pv) != (qu, qv) or not (_overlap(ur_a, ur_b) and _overlap(vr_a, vr_b)):
        return False
    ku, kv, lu, lv = ([Fraction(x) for x in k] for k in (ku, kv, lu, lv))
    if not all(_clamped(k, p) for k, p in ((ku, pu), (kv, pv), (lu, pu), (lv, pv))):
        return False
    if (ku[0], ku[-1], kv[0], kv[-1]) != (lu[0], lu[-1], lv[0], lv[-1]):
        return False
    net_a, net_b = _homogeneous(ca, wa), _homogeneous(cb, wb)
    if net_a is None or net_b is None:
        return False
    tu, tv = _union(ku, lu), _union(kv, lv)
    ku2, net_a = _refine(ku, pu, net_a, tu, along_u=True)
    lu2, net_b = _refine(lu, pu, net_b, tu, along_u=True) if net_a is not None else (None, None)
    if net_a is None or net_b is None:
        return False
    kv2, net_a = _refine(kv, pv, net_a, tv, along_u=False)
    lv2, net_b = _refine(lv, pv, net_b, tv, along_u=False) if net_a is not None else (None, None)
    if net_a is None or net_b is None:
        return False
    return ku2 == lu2 and kv2 == lv2 and net_a == net_b


def _greville(knots, degree, count):
    return [sum(knots[i + 1:i + degree + 1], Fraction(0)) / degree for i in range(count)]


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _scale(s, a):
    return tuple(s * x for x in a)


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _affine(values):
    """``(O, E1, E2)`` with S(u, v) = O + u E1 + v E2 exactly, or ``None``."""
    pu, pv, ku, kv, controls, weights, _, _ = values
    if weights is not None and len({w for row in weights for w in row}) != 1:
        return None
    gu = _greville([Fraction(x) for x in ku], pu, len(controls))
    gv = _greville([Fraction(x) for x in kv], pv, len(controls[0]))
    if gu[-1] == gu[0] or gv[-1] == gv[0]:
        return None
    p = [[tuple(Fraction(x) for x in point) for point in row] for row in controls]
    e1 = _scale(1 / (gu[-1] - gu[0]), _sub(p[-1][0], p[0][0]))
    e2 = _scale(1 / (gv[-1] - gv[0]), _sub(p[0][-1], p[0][0]))
    o = _sub(p[0][0], _add(_scale(gu[0], e1), _scale(gv[0], e2)))
    for i, row in enumerate(p):
        for j, point in enumerate(row):
            if point != _add(o, _add(_scale(gu[i], e1), _scale(gv[j], e2))):
                return None
    if not any(_cross(e1, e2)):
        return None
    return o, e1, e2


def _clip(polygon, axis, bound, keep_above):
    out = []
    for k, current in enumerate(polygon):
        previous = polygon[k - 1]
        inside_c = current[axis] >= bound if keep_above else current[axis] <= bound
        inside_p = previous[axis] >= bound if keep_above else previous[axis] <= bound
        if inside_c != inside_p:
            t = (bound - previous[axis]) / (current[axis] - previous[axis])
            out.append(tuple(a + t * (b - a) for a, b in zip(previous, current)))
        if inside_c:
            out.append(current)
    return out


def _coplanar_overlap(first, second):
    fa, fb = _affine(first), _affine(second)
    if fa is None or fb is None:
        return False
    (oa, a1, a2), (ob, b1, b2) = fa, fb
    normal = _cross(a1, a2)
    if any(_cross(normal, _cross(b1, b2))) or _dot(normal, _sub(ob, oa)) != 0:
        return False
    g11, g12, g22 = _dot(a1, a1), _dot(a1, a2), _dot(a2, a2)
    det = g11 * g22 - g12 * g12                       # > 0: a1, a2 independent

    def chart(point):                                 # exact A-parameters of an in-plane point
        d = _sub(point, oa)
        r1, r2 = _dot(a1, d), _dot(a2, d)
        return ((g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det)

    (s0, s1), (t0, t1) = ([Fraction(x) for x in r] for r in second[6:8])
    polygon = [chart(_add(ob, _add(_scale(s, b1), _scale(t, b2)))) for s, t in ((s0, t0), (s1, t0), (s1, t1),
                                                                                  (s0, t1))]
    (u0, u1), (v0, v1) = ([Fraction(x) for x in r] for r in first[6:8])
    for axis, bound, above in ((0, u0, True), (0, u1, False), (1, v0, True), (1, v1, False)):
        polygon = _clip(polygon, axis, bound, above)
        if not polygon:
            return False
    area = sum((p[0] * q[1] - q[0] * p[1] for p, q in zip(polygon, polygon[1:] + polygon[:1])), Fraction(0))
    return area != 0


def _edges(values):
    """Exact edge curves ``(degree, knots, points)`` of a full-domain clamped surface (empty otherwise)."""
    pu, pv, ku, kv, controls, weights, ur, vr = values
    if not (_clamped(ku, pu) and _clamped(kv, pv)):
        return []
    if tuple(ur) != (ku[0], ku[-1]) or tuple(vr) != (kv[0], kv[-1]):
        return []
    w = weights if weights is not None else tuple((1.0,) * len(controls[0]) for _ in controls)
    pts = [[(tuple(Fraction(x) for x in p), Fraction(wt)) for p, wt in zip(row, wrow)]
           for row, wrow in zip(controls, w)]
    ku, kv = [Fraction(x) for x in ku], [Fraction(x) for x in kv]
    columns = [list(c) for c in zip(*pts)]
    edges = [(pv, kv, pts[0]), (pv, kv, pts[-1]), (pu, ku, columns[0]), (pu, ku, columns[-1])]
    out = []
    for degree, knots, points in edges:
        if len({p for p, _ in points}) < 2:
            continue                                  # collapsed edge: a point, not a curve
        out.append((degree, tuple(knots), tuple(points)))
        flipped = tuple(knots[0] + knots[-1] - k for k in reversed(knots))
        out.append((degree, flipped, tuple(reversed(points))))
    return out


def _shared_edge(first, second):
    edges_b = set(_edges(second))
    return bool(edges_b) and any(edge in edges_b for edge in _edges(first))


def detect_coincidence(source_a, source_b):
    """``(UnclassifiedReason, message)`` when proven exactly, else ``None``."""
    first, second = source_a.values, source_b.values
    if _same_map(first, second):
        return UnclassifiedReason.COINCIDENT, (
            "coincident surfaces: exactly the same map on a selected parameter rectangle of positive area; "
            "the intersection is two-dimensional and has no curve accounting")
    if _coplanar_overlap(first, second):
        return UnclassifiedReason.COPLANAR_OVERLAP, (
            "coplanar overlap: both surfaces are exact affine maps into one plane whose selected images overlap "
            "with positive area; the intersection is two-dimensional and has no curve accounting")
    if _shared_edge(first, second):
        return UnclassifiedReason.SHARED_BOUNDARY_CURVE, (
            "shared boundary curve: an edge curve of the first surface equals an edge curve of the second "
            "exactly; boundary contact on both supports is not certifiable by the generic cover")
    return None
