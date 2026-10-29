"""Rational elimination of a certified cylinder branch against any quadric."""
from fractions import Fraction
import math
import numpy as np

from .analytic_roots import isolate_real_roots
from .cylinder_curve_events import _quadratic, _add, _scale, _multiply

# Keyed by the identities of the branch's two immutable support descriptors and
# the exact bytes of its transform. Each entry holds both support objects, so
# their ids cannot be reused while the entry exists.
_BRANCH_POLYNOMIALS: dict = {}
_BRANCH_POLYNOMIALS_LIMIT = 4096


def _branch_polynomials(curve):
    """The quadric-independent part of ``quadric_roots``: the exact rational chart of the branch.

    Pure functions of ``curve.first``, ``curve.second`` and ``curve.transform``,
    computed with the same expressions as before; the rational values are exact,
    so sharing them between calls (and subcurves) cannot change any result.
    """
    first, second = curve.first, curve.second
    transform = np.asarray(curve.transform)
    key = (id(first), id(second), transform.tobytes())
    entry = _BRANCH_POLYNOMIALS.get(key)
    if entry is not None and entry[0] is first and entry[1] is second:
        return entry[2]
    origin=transform[:3,:3] @ first.origin+transform[:3,3]
    cosine=transform[:3,:3] @ (first.radius*np.asarray(first.radial_direction))
    sine=transform[:3,:3] @ (first.radius*np.cross(first.axis,first.radial_direction))
    direction=transform[:3,:3] @ first.axis
    direction.flags.writeable=False
    numerator=tuple(tuple(Fraction(float(value)) for value in (o+c,2*s,o-c))
                    for o,c,s in zip(origin,cosine,sine))
    radius=(1,0,1)
    radius2=_multiply(radius,radius)
    numerator_radius=tuple(_multiply(values,radius) for values in numerator)
    numerator_products={}           # (i, j) -> numerator[i]*numerator[j], formed on first nonzero use
    quadratic=_quadratic(curve,second,curve.transform)
    result=(direction,numerator,radius,radius2,numerator_radius,numerator_products,quadratic)
    if len(_BRANCH_POLYNOMIALS) >= _BRANCH_POLYNOMIALS_LIMIT:
        _BRANCH_POLYNOMIALS.clear()
    _BRANCH_POLYNOMIALS[key]=(first,second,result)
    return result


def quadric_roots(curve,matrix,linear,constant,*,tolerance=1e-10,cancellation_check=None):
    """Roots of x.M.x+2*linear.x+constant; None is a common branch proof.

    A zero resultant establishes an algebraic common factor. The immutable
    chart's absence of interior discriminant transitions makes its interior
    branch selector valid over the entire connected interval.
    """
    matrix=np.asarray(matrix);linear=np.asarray(linear)
    direction,numerator,radius,radius2,numerator_radius,numerator_products,(a,b,c)=_branch_polynomials(curve)
    d=Fraction(float(direction @ matrix @ direction))
    e=_scale(radius,Fraction(float(2*linear @ direction)))
    h=_scale(radius2,Fraction(float(constant)))
    # A term whose rational weight is zero adds the zero polynomial: skipping it
    # leaves every (already trimmed) coefficient value unchanged. A plane has a
    # zero matrix, so its nine quadratic products are never added.
    weights=2*(matrix @ direction)
    for i in range(3):
        weight=Fraction(float(weights[i]))
        if weight:
            e=_add(e,_scale(numerator[i],weight))
        weight=Fraction(float(2*linear[i]))
        if weight:
            h=_add(h,_scale(numerator_radius[i],weight))
        for j in range(3):
            weight=Fraction(float(matrix[i,j]))
            if weight:
                product=numerator_products.get((i,j))
                if product is None:
                    product=numerator_products[(i,j)]=_multiply(numerator[i],numerator[j])
                h=_add(h,_scale(product,weight))
    m=_add(_scale(h,a),_scale(c,-d))
    l=_add(_scale(e,a),_scale(b,-d))
    bh_ec=_add(_multiply(b,h),_scale(_multiply(e,c),-1))
    polynomial=_add(_multiply(m,m),_scale(_multiply(l,bh_ec),-1))
    def residual(parameter):
        point=curve.evaluate(parameter)
        return abs(float(point @ matrix @ point+2*linear @ point+constant))
    if polynomial==(0,):
        return None if residual(.5)<=tolerance else tuple(t for t in (0.,1.) if residual(t)<=tolerance)
    # Two exact bounded projective charts cover the real line and its point
    # at infinity. Isolating an enormous tangent parameter to an absolute
    # width needlessly magnifies integer arithmetic near the angular seam.
    # Both angular maps have derivative magnitude <=2 on [-1,1], preserving
    # the existing angular enclosure bound from the same root tolerance.
    roots=isolate_real_roots(polynomial,tolerance=4*np.finfo(float).eps,interval=(-1,1),
                            cancellation_check=cancellation_check)
    angles=[2*math.atan(root.witness) for root in roots]
    reciprocal=isolate_real_roots(tuple(reversed(polynomial)),tolerance=4*np.finfo(float).eps,
                                 interval=(-1,1),cancellation_check=cancellation_check)
    angles.extend(2*math.atan2(1.,root.witness) for root in reciprocal)
    if len(polynomial)<9:
        angles.append(math.pi)
    lower,upper=sorted((curve.start_angle,curve.start_angle+curve.sweep_angle))
    from .arrangement_geometry import _angle_parameter
    parameters=[]
    for angle in angles:
        for turn in range(math.ceil((lower-angle-1e-14)/math.tau),math.floor((upper-angle+1e-14)/math.tau)+1):
            parameter=min(1.,max(0.,_angle_parameter(curve,angle+turn*math.tau)))
            if residual(parameter)<=tolerance and not any(abs(parameter-old)<=64*np.finfo(float).eps
                                                         for old in parameters):
                parameters.append(parameter)
    return tuple(sorted(parameters))
