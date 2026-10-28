"""Exact whole-cell discrepancy bounds for total-degree<=3 polynomial supports (sufficient proof only).

A uniform-weight tensor NurbsSurface whose entire cell footprint occupies ONE original knot span per axis and has
TOTAL degree <= 3 in its normalized chart is, composed with the straight reference maps of a linear
triangle fan or a bilinear quad, a polynomial residual ``R = S(UV) - FE`` of known degree. Its exact power
coefficients use the unchanged exact quadratic identity or bounded local Cox-de Boor power conversion
with exact knot/range composition in the original chart (never an approximate fit). Cubics use ten triangular or
sixteen tensor Bernstein controls; quadratic hierarchical formulas remain restricted to quadratics. The corner charts
may be certified intervals (true algebraic source vertices): every Bernstein control of ``R`` is then
evaluated in outward-exact interval arithmetic, which encloses the control vectors of EVERY choice of
true corner points inside the boxes (dependency loss may widen, never underbound). The Euclidean norm of
any point of the cell is at most the largest control norm (convex hull). This module admits nothing by
itself: callers keep strict immersion, skeleton, node-precision, transport and callback gates, and fall
back to the general interval proof whenever a precondition is not proven here.
"""
from fractions import Fraction as Q

from .errors import GeometryError


# ---------------------------------------------------------------- exact interval helpers (Fractions)

def _point(x):
    x = Q(x)
    return x, x


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _sub(a, b):
    return a[0]-b[1], a[1]-b[0]


def _scale(a, k):
    k = Q(k)
    return (a[0]*k, a[1]*k) if k >= 0 else (a[1]*k, a[0]*k)


def _mul(a, b):
    products = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return min(products), max(products)


def _square(a):
    if a[0] >= 0:
        return a[0]*a[0], a[1]*a[1]
    if a[1] <= 0:
        return a[1]*a[1], a[0]*a[0]
    return Q(0), max(a[0]*a[0], a[1]*a[1])


def _cross(a, b):
    return _sub(_mul(a[0], b[1]), _mul(a[1], b[0]))


def _box(p):
    """A chart point given as a pair of numbers or a pair of (lo, hi) intervals, as a pair of intervals."""
    x, y = p
    return tuple((Q(c[0]), Q(c[1])) if isinstance(c, tuple) else _point(c) for c in (x, y))


def _inside(box):
    return all(0 <= lo <= hi <= 1 for lo, hi in box)


def _power_1d(f0, fh, f1):
    """Coefficients (c0, c1, c2) of the quadratic through f(0), f(1/2), f(1) in the power basis."""
    return f0, -3*f0+4*fh-f1, 2*f0-4*fh+2*f1


def _quadratic_coefficients(support):
    """Exact power coefficients ``C[axis][i][j]`` of ``u^i v^j`` (total degree <= 2), or ``None``.

    Requires the exact owner type, degrees 1..2 per axis, a clamped single Bezier span per axis, uniform
    weights (so the rational map is the polynomial of its controls) and, algebraically, vanishing
    ``u^2 v``, ``u v^2`` and ``u^2 v^2`` coefficients, read from exact evaluations on the 3x3 dyadic grid
    (which determine a bidegree-(2,2) polynomial uniquely). Anything unproven returns ``None``.
    """
    from .nurbs import NurbsSurface
    if type(support) is not NurbsSurface:
        return None
    p, q = support.degree_u, support.degree_v
    if not (type(p) is int and type(q) is int and 1 <= p <= 2 and 1 <= q <= 2):
        return None

    def bezier(knots, degree):
        knots = tuple(knots)
        return (len(knots) == 2*(degree+1) and len(set(knots[:degree+1])) == 1
                and len(set(knots[degree+1:])) == 1 and knots[0] < knots[-1])

    if not (bezier(support.knots_u, p) and bezier(support.knots_v, q)):
        return None
    controls = support.controls
    if len(controls) != p+1 or any(len(row) != q+1 for row in controls):
        return None
    weights = getattr(support, 'weights', None)
    if weights is not None and len({Q(w) for row in weights for w in row}) != 1:
        return None
    try:
        grid = [[tuple(Q(x) for x in support.exact_point(Q(i, 2), Q(j, 2))) for j in range(3)]
                for i in range(3)]
    except (GeometryError, ValueError, TypeError, ArithmeticError):
        return None                                     # an unproven evaluation is a refusal to fast-path
    table = []
    for axis in range(3):
        rows = [_power_1d(*(grid[i][j][axis] for j in range(3))) for i in range(3)]       # v powers per u node
        by_v = [_power_1d(*(rows[i][v_power] for i in range(3))) for v_power in range(3)]  # [v power][u power]
        # total degree <= 2 forbids u*v^2, u^2*v^2 (v power 2) and u^2*v (v power 1)
        if by_v[2][1] != 0 or by_v[2][2] != 0 or by_v[1][2] != 0:
            return None
        table.append(((by_v[0][0], by_v[1][0], by_v[2][0]),        # C[axis][i][j] = by_v[j][i]
                      (by_v[0][1], by_v[1][1], Q(0)),
                      (by_v[0][2], Q(0), Q(0))))
    return tuple(table)


def coefficients(support, boxes=None):
    """Exact total-degree<=3 coefficients on the complete corner hull; quadratic representation preserved."""
    quadratic = _quadratic_coefficients(support)
    if quadratic is not None:
        return quadratic
    from ._cubic_cell_bounds import coefficients as cubic_coefficients
    return cubic_coefficients(support, boxes)


def admissible(support):
    return coefficients(support) is not None


def _evaluate(table, u, v):
    """Interval enclosure of the support over the chart box ``u x v`` (term-wise; sound)."""
    if len(table[0]) == 4:
        powers = []
        for value in (u, v):
            sequence = [_point(1)]
            for _ in range(3):
                sequence.append(_mul(sequence[-1], value))
            powers.append(sequence)
        return tuple(_interval_sum(_scale(_mul(powers[0][i], powers[1][j]), c[i][j])
                                   for i in range(4) for j in range(4-i)) for c in table)
    uu, vv, uv = _square(u), _square(v), _mul(u, v)
    out = []
    for c in table:
        value = _point(c[0][0])
        for term, k in ((u, c[1][0]), (v, c[0][1]), (uu, c[2][0]), (uv, c[1][1]), (vv, c[0][2])):
            value = _add(value, _scale(term, k))
        out.append(value)
    return tuple(out)


def _interval_sum(values):
    result = _point(0)
    for value in values:
        result = _add(result, value)
    return result


def _residual(table, box, xyz):
    return tuple(_sub(s, _point(t)) for s, t in zip(_evaluate(table, *box), xyz))


def _largest(vectors):
    from .curvilinear_reference_maps import _norm_upper
    return max(_norm_upper(max(abs(lo), abs(hi)) for lo, hi in vector) for vector in vectors)


def _strict(value, orientation):
    """Whether the interval ``value`` is strictly of the sign ``orientation`` (+1 / -1)."""
    return value[0] > 0 if orientation > 0 else value[1] < 0


def _vadd(a, b):
    return tuple(_add(x, y) for x, y in zip(a, b))


def _vsub(a, b):
    return tuple(_sub(x, y) for x, y in zip(a, b))


def _vscale(a, k):
    return tuple(_scale(x, k) for x in a)


def triangle_residual_bound(support, uv, xyz, orientation=None, method=None):
    """Upper bound of ``|S(UV) - FE|`` over the whole linear triangle, or ``None`` when not proven.

    ``uv``: three chart points, each a pair of numbers or of (lo, hi) intervals (certified true vertices);
    ``xyz``: three literal exact targets. Both maps are affine in the barycentric coordinates, so for each
    choice of vertices the residual is a quadratic with Bernstein controls ``R_i`` at the vertices and
    ``2 R(m_ij) - (R_i + R_j)/2`` at the edge midpoints; interval evaluation encloses all choices.
    With ``orientation`` given the source UV wedge must be strictly of that sign for ALL choices."""
    boxes = tuple(_box(p) for p in uv)
    if not all(_inside(box) for box in boxes):
        return None
    table = coefficients(support, boxes)
    if table is None:
        return None
    xyz = tuple(tuple(Q(x) for x in p) for p in xyz)
    if orientation is not None:
        edge = tuple(_sub(b, a) for a, b in zip(boxes[0], boxes[1]))
        arm = tuple(_sub(c, a) for a, c in zip(boxes[0], boxes[2]))
        if not _strict(_cross(edge, arm), orientation):
            return None
    if len(table[0]) == 4:
        if method not in (None, 'bernstein'):
            return None
        from ._cubic_cell_bounds import residual_bound
        return residual_bound(table, boxes, xyz, False)
    vertex = [_residual(table, boxes[i], xyz[i]) for i in range(3)]
    controls, hierarchical = list(vertex), []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        middle_box = tuple(_scale(_add(a, b), Q(1, 2)) for a, b in zip(boxes[i], boxes[j]))
        middle_xyz = tuple((a+b)/2 for a, b in zip(xyz[i], xyz[j]))
        middle = _residual(table, middle_box, middle_xyz)
        controls.append(tuple(_sub(_scale(m, 2), _scale(_add(a, b), Q(1, 2)))
                              for m, a, b in zip(middle, vertex[i], vertex[j])))
        # hierarchical edge coefficient e_ij = 4 R(m_ij) - 2 R_i - 2 R_j
        hierarchical.append(_vsub(_vscale(middle, 4), _vscale(_vadd(vertex[i], vertex[j]), 2)))
    bernstein = _largest(controls)
    # Exact identity R = sum_i l_i R_i + sum_{i<j} l_i l_j e_ij for a quadratic residual (the difference
    # R - sum l_i R_i vanishes at the vertices, hence is a combination of the l_i l_j). With
    # sum_{i<j} l_i l_j <= 1/3 and convex weights: |R| <= max|R_i| + max|e_ij| / 3.
    second = _largest(vertex) + _largest(hierarchical)/3
    return {'bernstein': bernstein, 'hierarchical': second}.get(method, min(bernstein, second))


def _bilinear(corners, xi, eta):
    weights = ((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta)
    return tuple(_scale_sum(weights, [c[axis] for c in corners]) for axis in range(len(corners[0])))


def _scale_sum(weights, intervals):
    total = _point(0)
    for w, interval in zip(weights, intervals):
        total = _add(total, _scale(interval, w))
    return total


def quad_hierarchical_parts(values):
    """Exact hierarchical coefficients of a bidegree-(2,2) residual from its 3x3 dyadic grid values.

    ``values[i][j]`` is ``R(x_i, y_j)`` for nodes 0, 1/2, 1 (vectors of intervals). With
    ``B`` the bilinear interpolant of the four corner values the identity is
    ``R = B + x(1-x)[a0(1-y)+a1 y] + y(1-y)[b0(1-x)+b1 x] + x(1-x)y(1-y) c`` where
    ``a0 = 4R(.5,0)-2R(0,0)-2R(1,0)``, ``a1 = 4R(.5,1)-2R(0,1)-2R(1,1)``, ``b0 = 4R(0,.5)-2R(0,0)-2R(0,1)``,
    ``b1 = 4R(1,.5)-2R(1,0)-2R(1,1)`` and ``c = 16R(.5,.5)-4(sum of corners)-2(a0+a1+b0+b1)``.
    (R - B - the edge terms vanishes on the whole boundary of the square and has bidegree <= (2,2), so it is
    x(1-x)y(1-y) times a constant, fixed by the centre.)"""
    v = values
    corners = (v[0][0], v[2][0], v[0][2], v[2][2])

    def edge(mid, first, last):
        return _vsub(_vscale(mid, 4), _vscale(_vadd(first, last), 2))

    a0, a1 = edge(v[1][0], v[0][0], v[2][0]), edge(v[1][2], v[0][2], v[2][2])
    b0, b1 = edge(v[0][1], v[0][0], v[0][2]), edge(v[2][1], v[2][0], v[2][2])
    total = _vadd(_vadd(corners[0], corners[1]), _vadd(corners[2], corners[3]))
    edges = _vadd(_vadd(a0, a1), _vadd(b0, b1))
    c = _vsub(_vsub(_vscale(v[1][1], 16), _vscale(total, 4)), _vscale(edges, 2))
    return corners, (a0, a1), (b0, b1), c


def quad_residual_bound(support, uv, xyz, orientation=None, method=None):
    """Upper bound of ``|S(UVbil) - FEbil|`` over the whole reference square of a bilinear quad, or ``None``.

    ``uv``: four corner charts (numbers or (lo, hi) intervals) in reference order (-1,-1), (1,-1), (1,1),
    (-1,1); ``xyz``: four literal exact targets. For total-degree-2 ``S`` the residual has tensor degree
    <= 2 per reference coordinate for each choice of corners; its 3x3 Bernstein controls follow exactly
    from the dyadic grid values, evaluated in interval arithmetic over all choices. With ``orientation``
    the four corner turns (the affine bilinear Jacobian at the corners) must be strictly of that sign for
    ALL choices."""
    boxes = tuple(_box(p) for p in uv)
    if not all(_inside(box) for box in boxes):
        return None
    table = coefficients(support, boxes)
    if table is None:
        return None
    xyz = tuple(tuple(Q(x) for x in p) for p in xyz)
    if orientation is not None:
        for k in range(4):
            a, b, c = boxes[k], boxes[(k+1) % 4], boxes[(k+2) % 4]
            first = tuple(_sub(q, p) for p, q in zip(a, b))
            second = tuple(_sub(q, p) for p, q in zip(b, c))
            if not _strict(_cross(first, second), orientation):
                return None
    if len(table[0]) == 4:
        if method not in (None, 'bernstein'):
            return None
        from ._cubic_cell_bounds import residual_bound
        return residual_bound(table, boxes, xyz, True)
    nodes = (Q(0), Q(1, 2), Q(1))

    def target(xi, eta):
        return tuple(sum(w*p[k] for w, p in zip(((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta), xyz))
                     for k in range(3))

    values = [[_residual(table, _bilinear(boxes, xi, eta), target(xi, eta)) for eta in nodes] for xi in nodes]

    def bernstein(f0, fh, f1):
        middle = tuple(_sub(_scale(m, 2), _scale(_add(a, b), Q(1, 2))) for m, a, b in zip(fh, f0, f1))
        return f0, middle, f1

    along_eta = [bernstein(*values[i]) for i in range(3)]                  # per xi node, controls over eta
    controls = []
    for j in range(3):
        controls.extend(bernstein(*(along_eta[i][j] for i in range(3))))
    bernstein_bound = _largest(controls)
    # Hierarchical form (see quad_hierarchical_parts): 0 <= x(1-x), y(1-y) <= 1/4 and convex weights give
    # |R| <= max|corner| + max(|a0|,|a1|)/4 + max(|b0|,|b1|)/4 + |c|/16.
    corners, a, b, c = quad_hierarchical_parts(values)
    hierarchical = _largest(corners) + _largest(a)/4 + _largest(b)/4 + _largest([c])/16
    return {'bernstein': bernstein_bound, 'hierarchical': hierarchical}.get(method, min(bernstein_bound, hierarchical))
