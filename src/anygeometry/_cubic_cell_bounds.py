"""Exact total-degree-three cell residuals; no quadratic formulas reused."""
from fractions import Fraction as Q
from math import comb

from .polynomial_cell_bounds import _point, _add, _sub, _scale, _mul, _largest


def _charge():
    from .plane_nurbs import charge_box
    charge_box()


def coefficients(support, boxes=None):
    """Polynomial on ONE original span covering the entire mapped cell hull.

    Coefficients remain in the ORIGINAL normalized face chart, never in a
    silently rebased cell chart. Bounded local Cox-de Boor algebra uses only
    (p+1)*(q+1) controls; no whole-net knot insertion.
    """
    from .nurbs import NurbsSurface
    if type(support) is not NurbsSurface:
        return None
    p, q = support.degree_u, support.degree_v
    if type(p) is not int or type(q) is not int or not (1 <= p <= 3 and 1 <= q <= 3):
        return None
    rows, columns = len(support.controls), len(support.controls[0])
    if any(len(row) != columns for row in support.controls):
        return None
    _charge()  # before original-span search or local algebra
    if support.weights is not None:
        if len(support.weights) != rows or any(len(row) != columns for row in support.weights):
            return None
        weights = tuple(Q(w) for row in support.weights for w in row)
        if len(set(weights)) != 1 or any(w <= 0 for w in weights):
            return None
    bounds = ((Q(0), Q(1)), (Q(0), Q(1))) if boxes is None else tuple(
        (min(point[axis][0] for point in boxes), max(point[axis][1] for point in boxes)) for axis in range(2))
    local = []
    for knots, active, degree, count, chart in ((support.knots_u, support.u_range, p, rows, bounds[0]),
                                               (support.knots_v, support.v_range, q, columns, bounds[1])):
        knots = tuple(map(Q, knots))
        if (len(knots) != count+degree+1 or any(a > b for a, b in zip(knots, knots[1:]))
                or any(x != knots[degree] for x in knots[:degree+1])
                or any(x != knots[count] for x in knots[count:])
                or any(knots[degree+1:count].count(x) > degree for x in set(knots[degree+1:count]))):
            return None  # original knots must at least preserve C0 endpoint position
        a, b = map(Q, active)
        if a == b or not (knots[degree] <= min(a, b) <= max(a, b) <= knots[count]):
            return None
        mapped = sorted((a+(b-a)*chart[0], a+(b-a)*chart[1]))
        spans = [i for i in range(degree, count) if knots[i] < knots[i+1]
                 and knots[i] <= mapped[0] <= mapped[1] <= knots[i+1]]
        if len(spans) != 1:
            return None  # knot-crossing hull or degenerate ambiguous knot point
        span = spans[0]
        # Degree0 N_span=1 on the selected open original span. Recurrence
        # composes native u=a+(b-a)*U exactly; C0 extends endpoint positions.
        basis = {span: (Q(1),)}
        for d in range(1, degree+1):
            next_basis = {}
            for i in range(span-d, span+1):
                value = [Q(0)]*(d+1)
                for previous, denominator, constant, slope in (
                    (basis.get(i, ()), knots[i+d]-knots[i], a-knots[i], b-a),
                    (basis.get(i+1, ()), knots[i+d+1]-knots[i+1], knots[i+d+1]-a, a-b)):
                    if denominator:
                        for power, coefficient in enumerate(previous):
                            value[power] += coefficient*constant/denominator
                            value[power+1] += coefficient*slope/denominator
                next_basis[i] = tuple(value)
            basis = next_basis
        local.append(basis)
    table = []
    for axis in range(3):
        normalized = [[sum((Q(support.controls[i][j][axis])*u[r]*v[s]
                           for i, u in local[0].items() for j, v in local[1].items()), Q(0))
                       for s in range(q+1)] for r in range(p+1)]
        if any(normalized[i][j] != 0 for i in range(p+1) for j in range(q+1) if i+j > 3):
            return None
        table.append(tuple(tuple(normalized[i][j] if i <= p and j <= q else Q(0)
                                 for j in range(4)) for i in range(4)))
    return tuple(table)


def _plus(first, second):
    result = dict(first)
    for key, value in second.items():
        result[key] = _add(result.get(key, _point(0)), value)
    return {key: value for key, value in result.items() if value != _point(0)}


def _times(first, second):
    result = {}
    for (i, j), a in first.items():
        for (k, l), b in second.items():
            key = i+k, j+l
            result[key] = _add(result.get(key, _point(0)), _mul(a, b))
    return {key: value for key, value in result.items() if value != _point(0)}


def _map(values, quad):
    a, b, c = values[:3]
    if not quad:
        return {(0, 0): a, (1, 0): _sub(b, a), (0, 1): _sub(c, a)}
    d = values[3]
    return {(0, 0): a, (1, 0): _sub(b, a), (0, 1): _sub(d, a),
            (1, 1): _add(_sub(a, b), _sub(c, d))}


def residual_powers(table, boxes, xyz, quad):
    """Interval coefficients enclose EVERY fixed choice of true corner charts."""
    _charge()
    u, v = (_map([box[axis] for box in boxes], quad) for axis in range(2))
    powers = []
    for value in (u, v):
        sequence = [{(0, 0): _point(1)}]
        for _ in range(3):
            sequence.append(_times(sequence[-1], value))
        powers.append(sequence)
    result = []
    for axis in range(3):
        value = {}
        for i in range(4):
            for j in range(4-i):
                if table[axis][i][j]:
                    term = {key: _scale(coefficient, table[axis][i][j])
                            for key, coefficient in _times(powers[0][i], powers[1][j]).items()}
                    value = _plus(value, term)
        target = _map([_point(point[axis]) for point in xyz], quad)
        result.append(_plus(value, {key: _scale(coefficient, -1) for key, coefficient in target.items()}))
    return tuple(result)


def residual_bound(table, boxes, xyz, quad):
    power = residual_powers(table, boxes, xyz, quad)
    controls = []
    indices = ((i, j) for i in range(4) for j in range(4 if quad else 4-i))
    for i, j in indices:
        _charge()  # one examined residual control-vector interval box
        vector = []
        for axis in range(3):
            value = _point(0)
            for (a, b), coefficient in power[axis].items():
                if a <= i and b <= j:
                    weight = (Q(comb(i, a), comb(3, a))*Q(comb(j, b), comb(3, b)) if quad else
                              Q(comb(i, a)*comb(j, b), comb(3, a+b)*comb(a+b, a)))
                    value = _add(value, _scale(coefficient, weight))
            vector.append(value)
        controls.append(tuple(vector))
    return _largest(controls)
