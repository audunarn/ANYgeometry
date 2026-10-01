"""Rational elimination of a certified cylinder branch against any quadric."""
from fractions import Fraction
import math
import numpy as np

from .analytic_roots import isolate_real_roots
from .cylinder_curve_events import _quadratic, _add, _scale, _multiply


def quadric_roots(curve,matrix,linear,constant,*,tolerance=1e-10,cancellation_check=None):
    """Roots of x.M.x+2*linear.x+constant; None is a common branch proof.

    A zero resultant establishes an algebraic common factor. The immutable
    chart's absence of interior discriminant transitions makes its interior
    branch selector valid over the entire connected interval.
    """
    matrix=np.asarray(matrix);linear=np.asarray(linear)
    transform=np.asarray(curve.transform)
    first=curve.first
    origin=transform[:3,:3] @ first.origin+transform[:3,3]
    cosine=transform[:3,:3] @ (first.radius*np.asarray(first.radial_direction))
    sine=transform[:3,:3] @ (first.radius*np.cross(first.axis,first.radial_direction))
    direction=transform[:3,:3] @ first.axis
    numerator=[tuple(Fraction(float(value)) for value in (o+c,2*s,o-c))
               for o,c,s in zip(origin,cosine,sine)]
    radius=(1,0,1)
    d=Fraction(float(direction @ matrix @ direction))
    e=_scale(radius,Fraction(float(2*linear @ direction)))
    h=_scale(_multiply(radius,radius),Fraction(float(constant)))
    for i in range(3):
        e=_add(e,_scale(numerator[i],Fraction(float(2*(matrix @ direction)[i]))))
        h=_add(h,_scale(_multiply(numerator[i],radius),Fraction(float(2*linear[i]))))
        for j in range(3):
            h=_add(h,_scale(_multiply(numerator[i],numerator[j]),Fraction(float(matrix[i,j]))))
    a,b,c=_quadratic(curve,curve.second,curve.transform)
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
