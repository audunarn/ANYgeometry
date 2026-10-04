"""Conservative whole-triangle coverage in an owner-bound analytic chart.

Side events are exact rational polynomial candidates, not fitted world chords.
Unresolved common factors and overlapping root brackets refuse.  This validator
does not authorize mesh edits or certify discretization/element quality.
"""
from fractions import Fraction as F
from copy import deepcopy
from itertools import combinations
from numbers import Integral
import math

import numpy as np

from .analytic_roots import isolate_real_roots, _division
from .arrangement_geometry import LinePath, BezierPath
from .bezier_intersections import _powers, _resultant
from .cylinder_curve_events import _add, _scale, _multiply, _trim
from .errors import GeometryError
from .definition_binding import definition_checksum
from .extrusions import BezierDirectrix
from .material_regions import validate_material_surface_regions_binding
from .surfaces import Plane, ExtrudedSurface


_ROOT_TOLERANCE = 4 * np.finfo(float).eps


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _value(p, t):
    value = F(0)
    for coefficient in reversed(p):
        value = value*t + coefficient
    return value


def _compose(p, a, b):
    result = (F(0),)
    for coefficient in reversed(p):
        result = _add(_multiply(result, (a, b)), (coefficient,))
    return _trim(result)


def _vector_polynomial(curve):
    if isinstance(curve, LinePath):
        return tuple(_trim((F(a), F(b)-F(a))) for a, b in zip(curve.start, curve.end))
    if isinstance(curve, BezierPath):
        return _powers(curve)
    raise GeometryError('material cell coverage: unsupported polynomial trim')


def _projection(powers, vector):
    result = (F(0),)
    for row, coefficient in zip(powers, vector):
        result = _add(result, _scale(row, coefficient))
    return _trim(result)


def _frame(support):
    if type(support) is Plane:
        origin, u, v = (tuple(F(float(x)) for x in row)
                        for row in (support.origin, support.u_vector, support.v_vector))
        uu, uv, vv = _dot(u, u), _dot(u, v), _dot(v, v)
        determinant = uu*vv-uv*uv
        if determinant <= 0:
            raise GeometryError('material cell coverage: singular plane')
        return dict(kind='plane', origin=origin, u=u, v=v,
                    inverse=(tuple((vv*x-uv*y)/determinant for x, y in zip(u, v)),
                             tuple((uu*y-uv*x)/determinant for x, y in zip(u, v))))
    if type(support) is not ExtrudedSurface or type(support.directrix) is not BezierDirectrix:
        raise GeometryError('material cell coverage: unsupported support')
    controls = tuple(tuple(F(float(x)) for x in row) for row in support.directrix.controls)
    direction = tuple(F(float(x)) for x in support.vector)
    bounds = tuple(F(float(x)) for x in support.u_range)
    if not all(0 <= x <= 1 for x in bounds):
        raise GeometryError('material cell coverage: unsupported extrapolated directrix')
    dd = _dot(direction, direction)
    differences = [tuple(b-a for a, b in zip(first, second))
                   for first, second in zip(controls[:-1], controls[1:])]
    projection = None
    for axis in range(3):
        q = tuple(F(i == axis)*dd-direction[i]*direction[axis] for i in range(3))
        signs = [_dot(q, delta) for delta in differences]
        if all(x > 0 for x in signs) or all(x < 0 for x in signs):
            projection = q if signs[0] > 0 else tuple(-x for x in q)
            break
    if projection is None:
        raise GeometryError('material cell coverage: directrix injectivity is unresolved')
    offsets = [tuple(b-a for a, b in zip(controls[0], row)) for row in controls[1:]]
    normal = next((_cross(a, b) for a, b in combinations(offsets, 2)
                   if any(_cross(a, b))), None)
    if normal is None or any(_dot(normal, row) for row in offsets) or _dot(normal, direction) == 0:
        raise GeometryError('material cell coverage: exact profile plane is unresolved')
    powers = _powers(BezierPath(controls))
    return dict(kind='bezier', powers=powers, controls=controls, direction=direction,
                q=projection, normal=normal, rate=_dot(normal, direction),
                urange=bounds, vrange=tuple(F(float(x)) for x in support.v_range))


def _lift(frame, first, second):
    """Exact power coefficients; never round derived Bernstein controls."""
    u, v = first
    du, dv = second[0]-u, second[1]-v
    if frame['kind'] == 'plane':
        return tuple(_trim((o+u*x+v*y, du*x+dv*y))
                     for o, x, y in zip(frame['origin'], frame['u'], frame['v']))
    t0, t1 = frame['urange']; s0, s1 = frame['vrange']
    a, b = t0+(t1-t0)*u, (t1-t0)*du
    if not 0 <= a <= 1 or not 0 <= a+b <= 1:
        raise GeometryError('material cell coverage: side leaves certified directrix range')
    c, d = s0+(s1-s0)*v, (s1-s0)*dv
    return tuple(_add(_compose(row, a, b), (c*z, d*z))
                 for row, z in zip(frame['powers'], frame['direction']))


def _integer_root(value, degree):
    if value < 0:
        return None
    low, high = 0, 1 << ((value.bit_length()+degree-1)//degree)
    while low <= high:
        middle = (low+high)//2
        power = middle**degree
        if power == value:
            return middle
        if power < value:
            low = middle+1
        else:
            high = middle-1
    return None


def _affine_map(first, second):
    """Prove first(t) == second(a+b*t) coefficient by coefficient."""
    degree = max(map(len, second))-1
    if degree < 1 or max(map(len, first))-1 != degree:
        return None
    axis = next(i for i, row in enumerate(second) if len(row) == degree+1)
    p, q = first[axis], second[axis]
    if len(p) != degree+1:
        return None
    ratio = p[-1]/q[-1]
    negative = ratio < 0
    if negative and degree % 2 == 0:
        return None
    numerator = _integer_root(abs(ratio.numerator), degree)
    denominator = _integer_root(ratio.denominator, degree)
    if numerator is None or denominator is None or numerator == 0:
        return None
    root = F(numerator, denominator)*(-1 if negative else 1)
    for b in (root, -root) if degree % 2 == 0 else (root,):
        a = (p[-2]/b**(degree-1)-q[-2])/(degree*q[-1])
        if all(_trim(x) == _compose(y, a, b) for x, y in zip(first, second)):
            return a, b
    return None


def _quadric_polynomial(powers, support):
    matrix, linear, constant = support.exact_relative((0., 0., 0.))
    result = (constant,)
    for i in range(3):
        result = _add(result, _scale(powers[i], 2*linear[i]))
        for j in range(3):
            result = _add(result, _scale(_multiply(powers[i], powers[j]), matrix[i][j]))
    return _trim(result)


def _events(powers, curve, check):
    """A complete conservative side-event polynomial, or proved overlap."""
    check()
    if isinstance(curve, (LinePath, BezierPath)):
        other = _vector_polynomial(curve)
        return _polynomial_events(powers, other, check)
    from .branch_curves import BezierQuadricCurve
    from .quadric_curves import QuadricIntersectionCurve
    if isinstance(curve, (BezierQuadricCurve, QuadricIntersectionCurve)) and not curve._identity:
        raise GeometryError('material cell coverage: unsupported transformed branch trim')
    if isinstance(curve, BezierQuadricCurve):
        from .branch_events import world_quadric
        supports = (world_quadric(curve),)
    elif isinstance(curve, QuadricIntersectionCurve):
        from .quadric_events import branch_supports
        supports = branch_supports(curve)
    else:
        raise GeometryError('material cell coverage: unsupported trim family')
    for support in supports:
        polynomial = _quadric_polynomial(powers, support)
        if polynomial != (0,):
            return (polynomial,), False
    raise GeometryError('material cell coverage: unresolved quadric coincidence')


def _polynomial_events(powers, other, check):
    mapping = _affine_map(powers, other)
    if mapping is not None:
        a, b = mapping
        if 0 <= a <= 1 and 0 <= a+b <= 1:
            return (), True
        return ((a, b), (a-1, b)), False
    for x, y in combinations(range(len(powers)), 2):
        check()
        f = (_add(powers[x], (-other[x][0],)), *((-v,) for v in other[x][1:]))
        g = (_add(powers[y], (-other[y][0],)), *((-v,) for v in other[y][1:]))
        # A collapsed trim projection still supplies necessary equations.
        if len(f) == len(g) == 1:
            polynomial = next((row for row in (f[0], g[0]) if row != (0,)), None)
            if polynomial is not None:
                return (polynomial,), False
            continue
        polynomial = _resultant(f, g, check)
        if _trim(polynomial) != (0,):
            return (polynomial,), False
    raise GeometryError('material cell coverage: unresolved polynomial overlap')


def _roots(polynomials, check):
    intervals = []
    seen_polynomials = set()
    for polynomial in polynomials:
        check()
        polynomial = _trim(polynomial)
        if polynomial == (0,):
            raise GeometryError('material cell coverage: unresolved zero event polynomial')
        monic = tuple(value/polynomial[-1] for value in polynomial)
        if monic in seen_polynomials:
            continue
        seen_polynomials.add(monic)
        intervals.extend((root.lower, root.upper) for root in isolate_real_roots(
            polynomial, tolerance=_ROOT_TOLERANCE, interval=(F(0), F(1)),
            cancellation_check=lambda: (check() or False)))
    # Equal brackets from different polynomials need not enclose the same root.
    # Only an exact rational singleton is an equality certificate here.
    rational = {pair for pair in intervals if pair[0] == pair[1]}
    intervals = sorted([pair for pair in intervals if pair[0] != pair[1]] + list(rational))
    previous = None
    for lower, upper in intervals:
        if previous is not None and lower <= previous:
            raise GeometryError('material cell coverage: unresolved clustered events')
        if lower != upper and (lower == 0 or upper == 1):
            raise GeometryError('material cell coverage: unresolved endpoint event')
        previous = upper
    return intervals


def _chart_polynomial(frame, curve, check):
    """Prove the source trim equals its polynomial chart lift coefficientwise."""
    check()
    powers = _vector_polynomial(curve)
    if frame['kind'] == 'plane':
        offset = tuple(_add(row, (-o,)) for row, o in zip(powers, frame['origin']))
        uv = tuple(_projection(offset, row) for row in frame['inverse'])
        lifted = tuple(_add((o,), _add(_scale(uv[0], u), _scale(uv[1], v)))
                       for o, u, v in zip(frame['origin'], frame['u'], frame['v']))
        if lifted != powers:
            raise GeometryError('material cell coverage: trim is not exactly on support')
        return uv
    offset = tuple(_add(row, (-o,)) for row, o in zip(powers, frame['controls'][0]))
    s = _scale(_projection(offset, frame['normal']), 1/frame['rate'])
    base = tuple(_add(row, _scale(s, -d)) for row, d in zip(powers, frame['direction']))
    mapping = _affine_map(base, frame['powers'])
    if mapping is None and all(len(row) == 1 for row in base):
        polynomial = _add(_projection(frame['powers'], frame['q']),
                          (-_dot(tuple(row[0] for row in base), frame['q']),))
        roots = isolate_real_roots(polynomial, interval=(F(0), F(1)), tolerance=_ROOT_TOLERANCE,
                                  cancellation_check=lambda: (check() or False))
        if len(roots) == 1 and roots[0].lower == roots[0].upper:
            a = roots[0].lower
            if all(_value(row, a) == point[0] for row, point in zip(frame['powers'], base)):
                mapping = a, F(0)
    if mapping is None:
        raise GeometryError('material cell coverage: exact polynomial chart correspondence is unresolved')
    a, b = mapping
    if not 0 <= a <= 1 or not 0 <= a+b <= 1:
        raise GeometryError('material cell coverage: trim leaves certified directrix range')
    v0, v1 = frame['vrange']; u0, u1 = frame['urange']
    return (_trim(((a-u0)/(u1-u0), b/(u1-u0))),
            _scale(_add(s, (-v0,)), 1/(v1-v0)))


def _chart_loops(frame, domain, check):
    loops = tuple(tuple(_chart_polynomial(frame, path.curve, check) for path in loop)
                  for loop in domain.boundaries)
    if not loops or any(not loop for loop in loops):
        raise GeometryError('material cell coverage: empty boundary')
    for loop in loops:
        for first, second in zip(loop, (*loop[1:], loop[0])):
            check()
            if any(_value(a, F(1)) != b[0] for a, b in zip(first, second)):
                raise GeometryError('material cell coverage: boundary is not exactly closed')
    return loops


def _point_on_polynomial(powers, point, check):
    a, b = (_add(row, (-coordinate,)) for row, coordinate in zip(powers, point))
    while b != (0,):
        check()
        a, b = b, _division(a, b)[1]
    if a == (0,):
        return True
    return bool(isolate_real_roots(a, interval=(F(0), F(1)), tolerance=_ROOT_TOLERANCE,
                                  cancellation_check=lambda: (check() or False)))


def _interval_value(polynomial, lower, upper):
    low = high = F(0)
    for coefficient in reversed(polynomial):
        products = low*lower, low*upper, high*lower, high*upper
        low, high = min(products)+coefficient, max(products)+coefficient
    return low, high


def _loop_location(loop, point, check):
    """Exact winding: -1 outside, 0 boundary, 1 inside; uncertainty refuses."""
    if any(_point_on_polynomial(path, point, check) for path in loop):
        return 0
    # Shear the ray to avoid every trim endpoint. Each endpoint forbids at most
    # one slope, and exact boundary membership has already excluded equality.
    for slope in range(2*len(loop)+1):
        check()
        rows = []
        for u, v in loop:
            x = _add(u, (-point[0],))
            y = _add(_add(v, (-point[1],)), _scale(x, -F(slope)))
            rows.append((x, y))
        if all(_value(y, F(0)) and _value(y, F(1)) for _, y in rows):
            break
    else:
        raise GeometryError('material cell coverage: exact winding ray is unresolved')
    winding = 0
    for x, y in rows:
        intervals = _roots((y,), check)
        for index, (lower, upper) in enumerate(intervals):
            check()
            low, high = _interval_value(x, lower, upper)
            if low <= 0 <= high:
                raise GeometryError('material cell coverage: winding root sign is unresolved')
            if high < 0:
                continue
            previous = intervals[index-1][1] if index else F(0)
            following = intervals[index+1][0] if index+1 < len(intervals) else F(1)
            before, after = _value(y, (previous+lower)/2), _value(y, (upper+following)/2)
            winding += int(before < 0 < after)-int(after < 0 < before)
    return 1 if winding else -1


def _inside_exact(loops, point, check):
    return (_loop_location(loops[0], point, check) >= 0
            and all(_loop_location(loop, point, check) <= 0 for loop in loops[1:]))


def _orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _coordinate_fraction(value):
    """Keep explicit plain Fraction UV; other values retain binary64 meaning."""
    return value if type(value) is F else F(float(value))


def _triangle_rows(triangles_uv):
    """Detach finite real input, preserving explicit plain Fraction entries."""
    try:
        raw = np.array(triangles_uv, copy=True)
        if np.iscomplexobj(raw) or (raw.dtype == object and
                any(isinstance(value, (complex, np.complexfloating)) for value in raw.flat)):
            raise GeometryError('material cell coverage requires real finite (n,3,2) triangles')
        if raw.ndim != 3 or raw.shape[1:] != (3, 2):
            raise GeometryError('material cell coverage requires finite (n,3,2) triangles')
        if raw.dtype == object and any(type(value) is F for value in raw.flat):
            triangles = np.empty(raw.shape, dtype=object)
            for index,value in enumerate(raw.flat):
                triangles.flat[index] = _coordinate_fraction(value)
            return triangles
        triangles = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('material cell coverage requires finite (n,3,2) triangles') from error
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2) or not np.isfinite(triangles).all():
        raise GeometryError('material cell coverage requires finite (n,3,2) triangles')
    return triangles


def validate_material_surface_region_triangles(model, result, face, triangles_uv, *, cancellation_check=None):
    """Validate entire closed UV triangles against one bound material region.

    Success returns ``None``. Unsupported support/trim/overlap or unresolved
    root partitions raise GeometryError. Supports Plane and BezierDirectrix
    extrusions with exactly planar profile controls and a strictly monotone
    projection perpendicular to the ruling. Every trim must have an exact
    polynomial chart correspondence and every loop must close exactly.
    Branch trims, tolerance-only support coincidence, rounded fragment controls
    without a coefficient identity, and periodic supports refuse. Membership
    uses rational winding and interval sign proofs, not floating point samples.
    This is material coverage only, not mesh conformity, curve-error or quality.
    """
    def check():
        if cancellation_check is not None and cancellation_check('material cell coverage'):
            raise GeometryError('material cell coverage cancelled')
    triangles = _triangle_rows(triangles_uv)
    check()
    validate_material_surface_regions_binding(model, result, cancellation_check=cancellation_check)
    if not isinstance(face, Integral):
        if getattr(face, 'model_id', None) != model.model_id or getattr(face, 'kind', None) != 'face':
            raise GeometryError('material cell coverage requires a bound face')
        face = face.id
    if isinstance(face, (bool, np.bool_)):
        raise GeometryError('material cell coverage requires a bound face')
    selected = [region for region in result.regions if int(face) in {handle.id for handle in region.faces}]
    if len(selected) != 1:
        raise GeometryError('material cell coverage face has no unique bound region')
    _validate_domain_triangles(selected[0].domain, triangles, check)
    check()
    validate_material_surface_regions_binding(model, result, cancellation_check=cancellation_check)


def validate_material_surface_region_triangles_xyz(model, result, face, triangles_xyz,
                                                   *, cancellation_check=None):
    """Certify actual closed XYZ triangles in one bound Plane face's material.

    Inputs have shape (n,3,3), use document units and retain their literal
    binary64 values (plain Fraction entries retain exact rational values).
    The owner derives exact rational UV and proves its exact affine lift equals
    each supplied XYZ corner before applying the whole-cell material proof.
    No floating projection, tolerance-based support recovery or sampling proves
    containment. Noncoplanar cells, non-Plane carriers, explicit face
    parameterizations and regions spanning multiple faces refuse.

    This proves containment, not full coverage, source-reference transfer,
    connectivity, quality, solver admission or publication. The model and
    supplied cells are never modified. Unsupported exact trim proofs refuse.
    """
    validate_material_surface_regions_binding(model, result)
    def signature():
        return (model.model_id, model.revision, result.source.source_checksum,
                definition_checksum(result))

    entry_binding = signature()

    def validate_binding(*, callbacks=False):
        if signature() != entry_binding:
            raise GeometryError('XYZ material cell coverage entry binding changed')
        validate_material_surface_regions_binding(model, result,
            cancellation_check=cancellation_check if callbacks else None)
        if signature() != entry_binding:
            raise GeometryError('XYZ material cell coverage entry binding changed')

    if not isinstance(face, Integral):
        if getattr(face, 'model_id', None) != model.model_id or getattr(face, 'kind', None) != 'face':
            raise GeometryError('XYZ material cell coverage requires a bound face')
        face = face.id
    if isinstance(face, (bool, np.bool_)) or not isinstance(face, Integral):
        raise GeometryError('XYZ material cell coverage requires a bound face')
    face = int(face)
    validate_binding()
    selected = [region for region in result.regions
                if face in {handle.id for handle in region.faces}]
    if len(selected) != 1 or len(selected[0].faces) != 1 or face not in model.faces:
        raise GeometryError('XYZ material cell coverage requires one bound SOURCE face region')
    current = model.faces[face]
    if type(current.surface) is not Plane or current.parameterization is not None:
        raise GeometryError('XYZ material cell coverage requires an implicit Plane face')
    # Detach owner truth before input coercion or caller callbacks. A temporary
    # mutation/restoration of the live model must not change the proof's domain.
    domain = deepcopy(selected[0].domain)
    if type(domain.support) is not Plane:
        raise GeometryError('XYZ material cell coverage requires a Plane region')
    frame = _frame(domain.support)
    try:
        raw = np.array(triangles_xyz, copy=True)
        if (raw.ndim != 3 or raw.shape[1:] != (3, 3) or np.iscomplexobj(raw)
                or any(isinstance(value, (complex, np.complexfloating)) for value in raw.flat)):
            raise GeometryError('XYZ material cell coverage requires finite real (n,3,3) triangles')
        xyz = tuple(tuple(tuple(_coordinate_fraction(value) for value in corner)
                          for corner in triangle) for triangle in raw)
        if any(not math.isfinite(float(value)) for triangle in xyz
               for corner in triangle for value in corner):
            raise GeometryError('XYZ material cell coverage requires finite real (n,3,3) triangles')
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('XYZ material cell coverage requires finite real (n,3,3) triangles') from error
    validate_binding()

    def check():
        if cancellation_check is not None and cancellation_check('XYZ material cell coverage'):
            raise GeometryError('XYZ material cell coverage cancelled')

    check()
    uv = []
    for triangle in xyz:
        corners = []
        for point in triangle:
            check()
            offset = tuple(x-y for x, y in zip(point, frame['origin']))
            pair = tuple(_dot(offset, row) for row in frame['inverse'])
            lifted = tuple(origin+u*pair[0]+v*pair[1]
                           for origin, u, v in zip(frame['origin'], frame['u'], frame['v']))
            if lifted != point:
                raise GeometryError('XYZ material cell coverage requires exact Plane support correspondence')
            corners.append(pair)
        uv.append(tuple(corners))
    _validate_domain_triangles(domain, tuple(uv), check)
    check()
    validate_binding(callbacks=True)


def _validate_domain_triangles(domain, triangles, check):
    """Shared exact kernel; caller must separately authenticate domain semantics."""
    frame = _frame(domain.support)
    loops = _chart_loops(frame, domain, check)
    for values in triangles:
        check()
        triangle = tuple(tuple(_coordinate_fraction(x) for x in row) for row in values)
        orientation = _orient(*triangle)
        if orientation == 0:
            raise GeometryError('material cell coverage: degenerate triangle')
        if not all(_inside_exact(loops, row, check) for row in triangle):
            raise GeometryError('material cell coverage: triangle vertex outside material')
        hole_signs = []
        for loop in loops[1:]:
            check()
            low = high = tuple(row[0] for row in loop[0])
            signs = [tuple((1 if orientation > 0 else -1)*_orient(a, b, point) for point in (low, high))
                     for a, b in zip(triangle, (*triangle[1:], triangle[0]))]
            # A hole-boundary point strictly inside the child already proves
            # excluded material inside it, even if other hole parts cross sides.
            if all(min(pair) > 0 for pair in signs):
                raise GeometryError('material cell coverage: triangle encloses a hole')
            hole_signs.append(signs)
        for first, second in zip(triangle, (*triangle[1:], triangle[0])):
            _lift(frame, first, second)  # enforce the certified directrix range
            powers = tuple(_trim((a, b-a)) for a, b in zip(first, second))
            polynomials, covered = [], False
            for loop_index, loop in enumerate(loops):
                for path in loop:
                    events, overlap = _polynomial_events(powers, path, check)
                    roots = _roots(events, check)
                    if loop_index and (roots or overlap):
                        raise GeometryError('material cell coverage: hole contact is unresolved')
                    polynomials.extend(events)
                    covered |= overlap
            intervals = _roots(polynomials, check)
            if covered:
                continue
            # Each tested root-free component extends up to its exact algebraic
            # endpoint. Isolated roots are covered by closed-material continuity;
            # no contact bracket is silently treated as an untested open gap.
            previous = F(0)
            for lower, upper in (*intervals, (F(1), F(1))):
                if lower > previous:
                    witness = (previous+lower)/2
                    uv = tuple(a+witness*(b-a) for a, b in zip(first, second))
                    check()
                    if not _inside_exact(loops, uv, check):
                        raise GeometryError('material cell coverage: side outside or unresolved at material boundary')
                previous = upper
        # With no possible side/hole contact, every connected hole stays on one
        # side of the triangle boundary. Its exact polynomial endpoint suffices.
        for signs in hole_signs:
            check()
            if not any(max(pair) < 0 for pair in signs):
                raise GeometryError('material cell coverage: hole witness is unresolved')
