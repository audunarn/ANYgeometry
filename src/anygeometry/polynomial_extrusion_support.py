"""Exact stored-coefficient support correspondence, never trimmed material.

The proof concerns mathematical maps defined by finite stored coefficients. It
does not certify binary64 evaluator results, inversion, trim membership, a
partition, references, or mesh permissions. No point evaluation is used.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import comb

from .errors import GeometryError

COONS_POLYNOMIAL = 'coons_polynomial_v1'
BEZIER_EXTRUSION = 'bezier_extrusion_v1'
POLYNOMIAL_EXTRUSION_KINDS = (COONS_POLYNOMIAL, BEZIER_EXTRUSION)


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, F)):
        raise GeometryError('polynomial support requires finite real stored coefficients')
    try:
        return F(value)
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise GeometryError('polynomial support has invalid coefficients') from error


def _vector(row):
    if type(row) is not tuple or len(row) != 3:
        raise GeometryError('polynomial support vector must have three coefficients')
    return tuple(_q(x) for x in row)


def _controls(rows, charge):
    if type(rows) is not tuple or len(rows) < 2:
        raise GeometryError('polynomial support needs at least two controls')
    out = []
    for row in rows:
        charge()
        out.append(_vector(row))
    return tuple(out)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _power(controls, charge):
    degree = len(controls)-1
    coefficients = [[F(0)]*3 for _ in controls]
    for i, point in enumerate(controls):
        for j in range(degree-i+1):
            charge()
            factor = comb(degree, i)*comb(degree-i, j)*(-1)**j
            for axis in range(3):
                coefficients[i+j][axis] += factor*point[axis]
    while len(coefficients) > 1 and not any(coefficients[-1]):
        coefficients.pop()
    return tuple(tuple(row) for row in coefficients)


def _range(values):
    if type(values) is not tuple or len(values) != 2:
        raise GeometryError('polynomial support range must have two endpoints')
    pair = tuple(_q(value) for value in values)
    if pair[0] == pair[1]:
        raise GeometryError('polynomial support range is degenerate')
    return pair


def _regularity(controls, direction, charge):
    """Sufficient exact injectivity and rank-two proof on the whole unit strip.

    q.D=0 and every derivative Bernstein coefficient of q.B has one strict
    sign, so q.B is strictly monotone and B' is never parallel to D. Equality
    B(t)+sD=B(r)+zD consequently forces t=r and then s=z.
    """
    norm = _dot(direction, direction)
    if not norm:
        raise GeometryError('polynomial extrusion has zero ruling vector')
    degree = len(controls)-1
    for axis in range(3):
        charge()
        q = tuple((norm if i == axis else 0)-direction[i]*direction[axis]
                  for i in range(3))
        values = []
        for first, second in zip(controls, controls[1:]):
            charge()
            values.append(degree*_dot(q, _sub(second, first)))
        if all(x > 0 for x in values) or all(x < 0 for x in values):
            return q, tuple(values)
    raise GeometryError('polynomial extrusion regular/injective projection is unproved')


@dataclass(frozen=True)
class _Extrusion:
    controls: tuple
    power: tuple
    direction: tuple
    u_range: tuple
    v_range: tuple
    projection: tuple
    projected_derivative_coefficients: tuple


def _compile(kind, data, charge):
    if type(data) is not tuple:
        raise GeometryError('polynomial support snapshot is malformed')
    if kind == COONS_POLYNOMIAL:
        if len(data) != 2:
            raise GeometryError('implicit Coons coefficient snapshot is malformed')
        corners, sides = data
        if corners != (0, 1, 2, 3) or type(sides) is not tuple or len(sides) != 4:
            raise GeometryError('implicit Coons proof requires four single-edge sides')
        oriented = []
        kinds = []
        for side in sides:
            charge()
            if (type(side) is not tuple or len(side) != 4 or type(side[0]) is not int
                    or side[0] <= 0 or type(side[1]) is not bool
                    or side[2] not in ('straight', 'spline')):
                raise GeometryError('implicit Coons boundary snapshot is malformed')
            points = _controls(side[3], charge)
            if side[2] == 'straight' and len(points) != 2:
                raise GeometryError('implicit Coons Straight has nonaffine controls')
            oriented.append(points if side[1] else points[::-1])
            kinds.append(side[2])
        bottom, right, backwards_top, backwards_left = oriented
        top, left = backwards_top[::-1], backwards_left[::-1]
        if kinds[1] != 'straight' or kinds[3] != 'straight':
            raise GeometryError('implicit Coons connectors must be Straight')
        direction = _sub(top[0], bottom[0])
        lower, upper = _power(bottom, charge), _power(top, charge)
        if len(lower) != len(upper) or lower[1:] != upper[1:]:
            raise GeometryError('Coons boundaries are not exact polynomial translations')
        if (right != (bottom[-1], top[-1]) or left != (bottom[0], top[0])):
            raise GeometryError('Coons connector endpoints do not close exactly')
        # Endpoint closure and affine connectors cancel the entire bilinear
        # correction, leaving B(u)+vD as a polynomial identity.
        controls = bottom
        u_range = v_range = (F(0), F(1))
    elif kind == BEZIER_EXTRUSION:
        if len(data) != 6:
            raise GeometryError('Bezier extrusion coefficient snapshot is malformed')
        declared, actual, vector, actual_vector, u_values, v_values = data
        controls = _controls(actual, charge)
        if _controls(declared, charge) != controls or _vector(vector) != _vector(actual_vector):
            raise GeometryError('Bezier extrusion public/raw coefficient disagreement')
        direction = _vector(actual_vector)
        u_range, v_range = _range(u_values), _range(v_values)
    else:
        raise GeometryError('polynomial extrusion support snapshot is unavailable/unsupported')
    if min(u_range) < 0 or max(u_range) > 1:
        raise GeometryError('polynomial extrusion extrapolated directrix is unsupported')
    projection, derivatives = _regularity(controls, direction, charge)
    return _Extrusion(controls, _power(controls, charge), direction, u_range,
                      v_range, projection, derivatives)


def _multiple(vector, direction):
    axis = next(i for i, value in enumerate(direction) if value)
    scale = vector[axis]/direction[axis]
    if any(x != scale*y for x, y in zip(vector, direction)):
        raise GeometryError('polynomial support ruling/offset is not exactly parallel')
    return scale


def _pack(value):
    value = F(value)
    return value.numerator, value.denominator


def prove_polynomial_extrusion_support(source_kind, source_data, child_kind,
                                       child_data, *, charge=lambda: None):
    """Return exact child-unit-UV -> original-unit-UV support evidence.

    A deliberately sufficient recognizer: same/reversed directrix polynomial,
    a constant ruling offset and nonzero ruling scaling. Degree elevation is
    allowed through exact power coefficients. Arbitrary affine reparameterized
    directrices, sampled Coons and unproved regularity refuse. The transformed
    world data must themselves satisfy the coefficient identities.
    """
    source = _compile(source_kind, source_data, charge)
    child = _compile(child_kind, child_data, charge)
    scale = _multiple(child.direction, source.direction)
    if not scale:
        raise GeometryError('polynomial support child ruling collapsed')
    match = None
    for sign, controls in ((1, source.controls), (-1, source.controls[::-1])):
        candidate = _power(controls, charge)
        if len(candidate) == len(child.power) and candidate[1:] == child.power[1:]:
            shift = _multiple(_sub(child.power[0], candidate[0]), source.direction)
            match = sign, shift
            break
    if match is None:
        raise GeometryError('polynomial support carrier coefficients differ')
    sign, shift = match
    a, b = source.u_range
    c, d = source.v_range
    u0, u1 = child.u_range
    v0, v1 = child.v_range
    # Root carrier parameter = t or 1-t; root ruling = shift+scale*s.
    offset = F(0) if sign == 1 else F(1)
    alpha, beta = (offset+sign*u0-a)/(b-a), sign*(u1-u0)/(b-a)
    gamma, delta = (shift+scale*v0-c)/(d-c), scale*(v1-v0)/(d-c)
    if any(min(x, x+y) < 0 or max(x, x+y) > 1 for x, y in ((alpha, beta), (gamma, delta))):
        raise GeometryError('polynomial child parameter rectangle leaves original support patch')
    determinant = beta*delta
    if not determinant:
        raise GeometryError('polynomial support parameter correspondence collapsed')
    return dict(
        semantics='exact stored polynomial support map on closed unit parameter rectangle',
        child_to_original_uv=tuple(tuple(map(_pack, row)) for row in ((alpha, beta), (gamma, delta))),
        parameter_jacobian_determinant=_pack(determinant),
        orientation_sign=1 if determinant > 0 else -1,
        original_projection=tuple(map(_pack, source.projection)),
        original_projected_derivative_coefficients=tuple(map(_pack, source.projected_derivative_coefficients)),
        source_kind=source_kind, child_kind=child_kind,
        support_correspondence_qualified=True, trimmed_material_qualified=False,
        reference_mapping_qualified=False, floating_evaluation_preservation_qualified=False,
        meshing_permitted=False,
    )
