"""Analytic Green integrals in a stored affine plane basis.

Rational projection avoids platform-dependent least-squares rounding. Elliptic
primitives use guard precision and convergent alternating trigonometric series;
display samples and Gaussian witness stations are not area definitions.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import math

from .arrangement_geometry import LinePath, BezierPath
from .exact_curves import EllipticArc
from .errors import GeometryError


def _fraction_vector(values):
    return tuple(Fraction(float(value)) for value in values)


def _dot(first,second):
    return sum(a*b for a,b in zip(first,second))


def _cross(first,second):
    return first[0]*second[1]-first[1]*second[0]


def _decimal(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


@lru_cache(maxsize=32)
def _pi(precision):
    with localcontext() as context:
        context.prec=precision+12
        cutoff=Decimal(10)**(-precision-8)
        def atan_inverse(denominator):
            x=Decimal(1)/denominator
            power=x; total=x; index=1
            while True:
                power*=-x*x
                term=power/(2*index+1)
                total+=term
                if abs(term)<cutoff:return total
                index+=1
        return 16*atan_inverse(Decimal(5))-4*atan_inverse(Decimal(239))


def _sincos(value,precision):
    # Compute pi with enough absolute accuracy for argument reduction even for
    # large finite stored angles. Precision depends on input, not model counts.
    pi=_pi(precision+max(0,value.adjusted())+12)
    value=value%(2*pi)
    if value>pi:value-=2*pi
    if value<-pi:value+=2*pi
    square=-value*value
    sine_term=sine=value
    cosine_term=cosine=Decimal(1)
    cutoff=Decimal(10)**(-precision+12)
    index=1
    while True:
        sine_term*=square/((2*index)*(2*index+1))
        cosine_term*=square/((2*index-1)*(2*index))
        sine+=sine_term;cosine+=cosine_term
        if max(abs(sine_term),abs(cosine_term))<cutoff:return sine,cosine
        index+=1


def planar_loop_area(support,loop,tolerance):
    """Return an analytic area, or None for curves needing other integration."""
    if not all(isinstance(path.curve,(LinePath,BezierPath,EllipticArc)) for path in loop):
        return None
    first,second=map(_fraction_vector,(support.u_vector,support.v_vector))
    aa,ab,bb=_dot(first,first),_dot(first,second),_dot(second,second)
    determinant=aa*bb-ab*ab
    if determinant<=0:raise GeometryError('planar area basis is unusable')
    anchor=_fraction_vector(loop[0].curve.evaluate(0.))
    def projected(values,*,point=False):
        vector=_fraction_vector(values)
        if point:vector=tuple(a-b for a,b in zip(vector,anchor))
        a,b=_dot(first,vector),_dot(second,vector)
        return ((bb*a-ab*b)/determinant,(aa*b-ab*a)/determinant)
    rational=Fraction(0)
    ellipses=[]
    for path in loop:
        curve=path.curve
        if isinstance(curve,LinePath):
            rational+=_cross(projected(curve.start,point=True),projected(curve.end,point=True))/2
        elif isinstance(curve,BezierPath):
            controls=tuple(projected(point,point=True) for point in curve.controls)
            degree=len(controls)-1
            for i,point in enumerate(controls):
                for j in range(degree):
                    difference=tuple(b-a for a,b in zip(controls[j],controls[j+1]))
                    rational+=_cross(point,difference)*Fraction(
                        math.comb(degree,i)*math.comb(degree-1,j),
                        4*math.comb(2*degree-1,i+j))
        else:
            center=projected(curve.center,point=True)
            u,v=projected(curve.u_vector),projected(curve.v_vector)
            ellipses.append((_cross(center,v),_cross(center,u),_cross(u,v),
                             Fraction(curve.start_angle),Fraction(curve.sweep_angle)))
    if not ellipses:return float(rational)
    magnitude=max((abs(float(value)) for row in ellipses for value in row),default=1.)
    # Retain guard digits above the requested absolute convergence tolerance.
    precision=max(80,60+max(0,math.ceil(math.log10(magnitude)))
                     -min(0,math.floor(math.log10(tolerance))))
    with localcontext() as context:
        context.prec=precision
        result=_decimal(rational)
        for cv,cu,uv,start,sweep in ellipses:
            a,delta=_decimal(start),_decimal(sweep)
            sine_a,cosine_a=_sincos(a,precision)
            sine_b,cosine_b=_sincos(a+delta,precision)
            result+=(_decimal(cv)*(sine_b-sine_a)+_decimal(cu)*(cosine_b-cosine_a)
                     +_decimal(uv)*delta)/2
        return float(result)
