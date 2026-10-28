"""Literal, operation-local iso-curve hulls for boundary proof queries.

No mutable nurbs._ISO_CACHE payload is trusted. This uses the same exact basis
and knot-refinement primitives, with the existing float-map slack and world
evaluation margin, without modifying those shared numerical primitives.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import math
import numpy as np

from .errors import GeometryError
from .nurbs import (CurveOnSurface,NurbsCurve,NurbsSurface,_IDENTITY,_map,_exact_chart,
                    _exact_span,_exact_basis,_exact_refine,nurbs_record,nurbs_from_record)
from .nurbs_intersection_curve import operation_cached


class IsoHullUnavailable(GeometryError):
    """The existing affine iso-line sufficient conditions do not apply."""


def literal(value):
    kind = type(value)
    if kind is Q:
        return ('rational',value.numerator,value.denominator)
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise GeometryError('boundary proof literal needs exact string field names')
        return ('dict',tuple(sorted((key,literal(item)) for key,item in value.items())))
    if kind in (list,tuple):
        return (kind.__name__,tuple(literal(item) for item in value))
    if value is None:
        return ('None',None)
    if kind in (str,int,float,bool):
        if kind is float and not math.isfinite(value):
            raise GeometryError('boundary proof literal must be finite')
        return (kind.__name__,value)
    raise GeometryError('boundary proof literal contains an unsupported scalar or container type')


def curve_literal(curve):
    return literal(nurbs_record(curve))


def _q(value):
    return value.numerator,value.denominator


def _iso_key(curve):
    # Admission precedes lookup: serializers deliberately erase subclass and
    # unsupported also_on state, so a warmed base-class row cannot qualify it.
    if (type(curve) is not CurveOnSurface or type(curve.support) is not NurbsSurface
            or type(curve.pcurve) is not NurbsCurve
            or type(curve.also_on) is not tuple or curve.also_on
            or type(curve.transform) is not tuple
            or any(type(row) is not tuple or any(type(x) is not float for x in row)
                   for row in curve.transform)
            or curve.transform != _IDENTITY):
        raise IsoHullUnavailable('affine identity iso-curve proof is unavailable')
    return ('literal-affine-iso-v1',curve_literal(curve))


@operation_cached(_iso_key)
def _iso_payload(curve):
    if (type(curve) is not CurveOnSurface or type(curve.support) is not NurbsSurface
            or type(curve.pcurve) is not NurbsCurve or curve.transform != _IDENTITY):
        raise IsoHullUnavailable('affine identity iso-curve proof is unavailable')
    # Fresh public definitions, never evaluated buffers or global iso payloads.
    surface = nurbs_from_record(nurbs_record(curve.support))
    pcurve = nurbs_from_record(nurbs_record(curve.pcurve))
    if (pcurve.degree != 1 or len(pcurve.controls) != 2
            or pcurve.weights is not None and pcurve.weights[0] != pcurve.weights[1]):
        raise IsoHullUnavailable('iso-curve needs an affine two-control pcurve')
    (a0,b0),(a1,b1) = pcurve.controls
    fixed = 1 if b0 == b1 else 0 if a0 == a1 else None
    if fixed is None or (a0,b0) == (a1,b1):
        raise IsoHullUnavailable('pcurve is not a nonconstant iso-line')
    running = 1-fixed
    ranges = (surface.u_range,surface.v_range)
    chart_value = pcurve.controls[0][fixed]
    exact_fixed = _exact_chart(Q(chart_value),ranges[fixed])
    float_fixed = Q(float(_map(np.asarray([chart_value]),*ranges[fixed])[0]))
    if exact_fixed != float_fixed:
        raise IsoHullUnavailable('fixed exact and float-map iso parameters differ')
    net = surface.exact_net()
    degree = (surface.degree_u,surface.degree_v)[fixed]
    knots = tuple(map(Q,(surface.knots_u,surface.knots_v)[fixed]))
    count = len(net) if fixed == 0 else len(net[0])
    span = _exact_span(degree,list(knots),count,exact_fixed)
    basis = _exact_basis(degree,list(knots),span,exact_fixed)
    run_degree = (surface.degree_u,surface.degree_v)[running]
    run_knots = tuple(map(Q,(surface.knots_u,surface.knots_v)[running]))
    run_count = len(net) if running == 0 else len(net[0])
    points = []
    for k in range(run_count):
        total = [Q(0)]*4
        for offset,value in enumerate(basis):
            entry = net[span-degree+offset][k] if fixed == 0 else net[k][span-degree+offset]
            total = [x+value*y for x,y in zip(total,entry)]
        points.append(tuple(total))
    k0,k1 = map(Q,(pcurve.knots[1],pcurve.knots[2]))
    d0,d1 = map(Q,pcurve.domain)
    c0,c1 = (Q(pcurve.controls[i][running]) for i in range(2))
    r0,r1 = map(Q,ranges[running])
    sigma0 = r0+(c0+(c1-c0)*(d0-k0)/(k1-k0))*(r1-r0)
    sigma_delta = (c1-c0)*(d1-d0)/(k1-k0)*(r1-r0)
    eps = Q(1,1<<52)
    error_s = 4*Q(math.ulp(float(max(abs(d0),abs(d1)))))
    error_ratio = error_s/abs(k1-k0)+8*eps
    error_chart = abs(c1-c0)*error_ratio+8*eps*max(abs(c0),abs(c1),Q(1))
    slack = 2*(abs(r1-r0)*error_chart+4*Q(math.ulp(float(max(abs(r0),abs(r1))))))
    margin = tuple(16*(surface.degree_u+surface.degree_v+2)*eps*
        max(abs(Q(p[k])) for row in surface.controls for p in row) for k in range(3))
    return (run_degree,tuple(map(_q,run_knots)),tuple(tuple(map(_q,p)) for p in points),
            _q(sigma0),_q(sigma_delta),_q(slack),tuple(map(_q,sorted((r0,r1)))),tuple(map(_q,margin)))


@dataclass(frozen=True)
class IsoHull:
    degree: int
    knots: tuple
    points: tuple
    sigma0: object
    sigma_delta: object
    slack: object
    selected: tuple
    margin: tuple

    def hull(self,lower,upper):
        from .nurbs import _float_down,_float_up
        low,high = max(0.,_float_down(Q(lower))),min(1.,_float_up(Q(upper)))
        a,b = sorted((self.sigma0+Q(low)*self.sigma_delta,self.sigma0+Q(high)*self.sigma_delta))
        a,b = max(self.selected[0],a-self.slack),min(self.selected[1],b+self.slack)
        _,refined,active = _exact_refine(self.degree,list(self.knots),list(self.points),a,b)
        if any(refined[i][3] <= 0 for i in active):
            raise GeometryError('literal iso-curve homogeneous denominator is not positive')
        return tuple(tuple(x/refined[i][3] for x in refined[i][:3]) for i in active)


def prepare_iso_hull(curve):
    payload = _iso_payload(curve)  # only immutable integer data is cached
    degree,knots,points,start,delta,slack,selected,margin = payload
    return IsoHull(degree,tuple(Q(*x) for x in knots),tuple(tuple(Q(*x) for x in p) for p in points),
        Q(*start),Q(*delta),Q(*slack),tuple(Q(*x) for x in selected),tuple(Q(*x) for x in margin))
