"""Exact true-graph fan residuals on polynomial translational supports.

This consumes SourceSide's implicit graph, not PlaneNurbsCurve's rounded
point evaluator. All interval coefficients enclose every endpoint choice.
"""
from fractions import Fraction as Q
from math import comb

from .errors import GeometryError
from .polynomial_cell_bounds import _point, _add, _sub, _mul, _scale, _largest
from ._cubic_cell_bounds import coefficients, _plus, _times


def _constant(value):
    return {(0, 0): value}


def _scaled(poly, value):
    return {key: _scale(coefficient, value) for key, coefficient in poly.items()}


def _compose(values, poly):
    result = {}
    for value in reversed(values):
        result = _plus(_times(result, poly), _constant(_point(value)))
    return result


def _derivative(poly):
    return {(i, j-1): _scale(value, j) for (i, j), value in poly.items() if j}


def _controls(poly):
    """Tensor Bernstein degree3; explicit zero coefficients retain degree."""
    if any(i > 3 or j > 3 for i, j in poly):
        return None
    return tuple(tuple(sum_intervals(_scale(value, Q(comb(i, a), comb(3, a))*Q(comb(j, b), comb(3, b)))
                                    for (a, b), value in poly.items() if a <= i and b <= j)
                       for j in range(4)) for i in range(4))


def sum_intervals(values):
    result = _point(0)
    for value in values:
        result = _add(result, value)
    return result


def _split(net, axis):
    """Exact interval de Casteljau restriction, no new geometric target."""
    left, right = [[None]*4 for _ in range(4)], [[None]*4 for _ in range(4)]
    for fixed in range(4):
        rows = [net[k][fixed] if axis == 0 else net[fixed][k] for k in range(4)]
        lo, hi = [rows[0]], [rows[-1]]
        while len(rows) > 1:
            rows = [_scale(_add(a, b), Q(1, 2)) for a, b in zip(rows, rows[1:])]
            lo.append(rows[0]); hi.append(rows[-1])
        for k in range(4):
            if axis == 0: left[k][fixed], right[k][fixed] = lo[k], hi[3-k]
            else: left[fixed][k], right[fixed][k] = lo[k], hi[3-k]
    return tuple(map(tuple, left)), tuple(map(tuple, right))


def structural_fan_bound(support, boundary, centre_uv, xyz_side, centre_xyz, *,
                         orientation, surface_error, cancellation_check=None):
    """Sufficient exact (bound, examined) or None; general backend remains.

    One native arc, u-sliced, one original polynomial span over the whole
    chart, constant transverse plane coefficient. No immersion assertion.
    """
    from .regular_source_reference import SourceSide, SourceNode
    from .nurbs import NurbsSurface
    from .support_curves import PlaneNurbsCurve
    from .definition_binding import definition_checksum
    from .nurbs_runtime_binding import validate_section_runtime, _section_literal
    from .plane_nurbs import charge_box, charge_split, depth_limit
    if (type(support) is not NurbsSurface or type(boundary) is not SourceSide
            or type(boundary.curve) is not PlaneNurbsCurve or len(boundary.pieces) != 1
            or orientation not in (1, -1)):
        return None
    curve = boundary.curve
    if type(curve._INITIAL_ROOT_BISECTIONS) is not int or curve._INITIAL_ROOT_BISECTIONS != 60:
        raise GeometryError('structural fan native root law is not certified')
    if type(curve.surface) is not NurbsSurface or definition_checksum(curve.surface) != definition_checksum(support):
        return None
    first, last = boundary.start, boundary.end
    if type(first) is not SourceNode or type(last) is not SourceNode:
        return None
    for node in (first, last):
        if (type(node.box) is not tuple or len(node.box) != 2
                or any(type(interval) is not tuple or len(interval) != 2
                       or Q(interval[0]) > Q(interval[1]) for interval in node.box)):
            raise GeometryError('structural fan endpoint enclosure is malformed')
    lo, hi, arc, a, b = boundary.pieces[0]
    if lo != 0 or hi != 1 or a is not first or b is not last or arc.slicing != 0:
        return None
    charge_box()  # before native extraction and polynomial algebra
    validate_section_runtime(curve)
    arcs = curve._arcs()
    if not any(actual is arc for actual in arcs):
        raise GeometryError('structural fan arc is not the current native branch')
    # Detached definition and consumed-reference pins surround EVERY callback.
    signature = definition_checksum((support, boundary, centre_uv, xyz_side, centre_xyz))
    native = _section_literal(arc)
    surface, path = curve.surface, curve.path
    piece = boundary.pieces
    limit = Q(surface_error)
    def check():
        def unchanged():
            if (boundary.curve is not curve or curve.surface is not surface or curve.path is not path
                    or boundary.start is not first or boundary.end is not last or boundary.pieces is not piece
                    or type(curve._INITIAL_ROOT_BISECTIONS) is not int or curve._INITIAL_ROOT_BISECTIONS != 60
                    or definition_checksum((support, boundary, centre_uv, xyz_side, centre_xyz)) != signature
                    or _section_literal(arc) != native):
                raise GeometryError('structural fan operands changed during proof')
            validate_section_runtime(curve)
        unchanged()
        if cancellation_check is not None:
            cancelled = cancellation_check()
            unchanged()
            if cancelled:
                raise GeometryError('source-curved reference map proof cancelled')
    check()
    table = coefficients(support)
    if table is None:
        return None
    # S(U,V)=P(U)+D*V, in the ORIGINAL normalized support chart.
    if any(table[k][i][j] for k in range(3) for i in range(4) for j in range(1, 4)
           if (i, j) != (0, 1)):
        return None
    normal, offset, *_ = curve._plane_slab()
    denominator = sum(normal[k]*table[k][0][1] for k in range(3))
    if not denominator:
        return None
    graph = tuple(((offset if i == 0 else 0)-sum(normal[k]*table[k][i][0] for k in range(3)))/denominator
                  for i in range(4))
    u0, u1 = tuple(map(Q, first.box[0])), tuple(map(Q, last.box[0]))
    native_origin, native_end = map(Q, support.u_range)
    # Reversed range multiplication must use interval arithmetic, not ordering.
    native_u = _add(_point(native_origin), _mul(_point(native_end-native_origin),
                   (min(u0[0], u1[0]), max(u0[1], u1[1]))))
    if not min(arc.s0, arc.s1) <= native_u[0] <= native_u[1] <= max(arc.s0, arc.s1):
        return None
    u = {(0, 0): u0, (0, 1): _sub(u1, u0)}
    v = _compose(graph, u)
    # At t=0/1 SourceSide.box returns the WHOLE endpoint isolator, not
    # just f(U). Include that enclosure explicitly without changing q'.
    endpoint_error = [Q(0)]
    for node in (first, last):
        value = _compose(graph, _constant(tuple(map(Q, node.box[0])))).get((0, 0), _point(0))
        difference = _sub(tuple(map(Q, node.box[1])), value)
        endpoint_error.extend(difference)
    v_enclosed = _plus(v, _constant((min(endpoint_error), max(endpoint_error))))
    q = (u, v_enclosed)
    derivative = (_derivative(u), _derivative(v))
    radial = tuple(_plus(_constant(_point(c)), _scaled(x, -1)) for c, x in zip(centre_uv, q))
    wedge = _scaled(_plus(_times(derivative[0], radial[1]), _scaled(_times(derivative[1], radial[0]), -1)), orientation)
    one_minus_s = {(0, 0): _point(1), (1, 0): _point(-1)}
    r = tuple(_plus(_times(one_minus_s, x), {(1, 0): _point(c)}) for x, c in zip(q, centre_uv))
    residual = []
    for k in range(3):
        world = _plus(_compose(tuple(table[k][i][0] for i in range(4)), r[0]), _scaled(r[1], table[k][0][1]))
        edge = {(0, 0): _point(xyz_side[0][k]), (0, 1): _point(Q(xyz_side[1][k])-Q(xyz_side[0][k]))}
        target = _plus(_times(one_minus_s, edge), {(1, 0): _point(centre_xyz[k])})
        residual.append(_plus(world, _scaled(target, -1)))
    nets = tuple(_controls(p) for p in (*residual, *r, wedge))
    if any(net is None for net in nets):
        return None
    pending, maximum, examined = [(nets, 0)], Q(0), 0
    while pending:
        check()
        current, depth = pending.pop()
        charge_box(); examined += 1
        if examined > 32768:
            raise GeometryError('source-curved cell proof box budget exhausted')
        bound = _largest(tuple(current[k][i][j] for k in range(3)) for i in range(4) for j in range(4))
        inside = all(0 <= x[0] <= x[1] <= 1 for net in current[3:5] for row in net for x in row)
        positive = all(x[0] > 0 for row in current[5] for x in row)
        if positive and inside and bound <= limit:
            maximum = max(maximum, bound)
            continue
        if depth >= depth_limit(12):
            return None
        charge_split()
        children = tuple(_split(net, depth % 2) for net in current)
        pending.extend((tuple(child[side] for child in children), depth+1) for side in (0, 1))
    check()
    return maximum, examined
