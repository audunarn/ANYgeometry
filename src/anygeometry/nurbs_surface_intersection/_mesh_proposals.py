"""Owner-curvature station proposals; never chord or cell certification.

Every returned interval still goes through the ordinary boundary sampler and
its actual FE-chord proof. No inequality here classifies an actual violation.
"""
from fractions import Fraction as Q
import math
from types import SimpleNamespace

from ..errors import GeometryError
from ._source import (Limits,isolate_callback,NurbsSurfaceIntersectionCancelled as Cancelled,
                      NurbsSurfaceIntersectionIntegrityError as Integrity,
                      NurbsSurfaceIntersectionRefused as Refused)


def _binary(value):
    if type(value) not in (int,float,Q):
        raise GeometryError('curvature proposal parameters need literal real stations')
    try:
        point = Q(value)
        encoded = float(point)
    except (ValueError,OverflowError,TypeError) as cause:
        raise GeometryError('curvature proposal station is not finite binary64') from cause
    if not math.isfinite(encoded) or Q(encoded) != point or not 0 <= point <= 1:
        raise GeometryError('curvature proposal station must be exact binary64 in [0,1]')
    return Q(point.numerator,point.denominator)  # detach even Fraction inputs


def propose_c2_boundary_parameters(curve, parameters, *, chord_error,cancellation_check=None):
    """Return sorted binary64-exact source stations retaining every input.

    Sufficient C2 proposals target 3/4 of the supplied chord error, leaving
    proposal slack for stored endpoint corrections. This is a mesh proposal,
    not a changed geometry tolerance or an endpoint/cell discrepancy proof.
    Nonbinary mandatory breaks retain that interval for generic sampling.
    Every owner/cancellation/resource exception propagates; none is fallback.
    """
    from ..nurbs_intersection_curve import (NurbsSurfaceIntersectionCurve,
        intersection_curve_operation,intersection_curve_record)
    from .._boundary_curve_proofs import literal
    from ..plane_nurbs import charge_box,charge_split,depth_limit
    if type(curve) is not NurbsSurfaceIntersectionCurve:
        raise GeometryError('curvature proposals require the exact two-support owner curve')
    definition = literal(intersection_curve_record(curve))
    if type(chord_error) not in (int,float,Q):
        raise GeometryError('curvature proposal chord error must be a positive literal real')
    try:
        error = Q(chord_error)
    except (ValueError,OverflowError,TypeError) as cause:
        raise GeometryError('curvature proposal chord error must be finite') from cause
    if error <= 0:
        raise GeometryError('curvature proposal chord error must be positive')
    error = Q(error.numerator,error.denominator)
    callback = (None if cancellation_check is None else isolate_callback(
        lambda:cancellation_check('two-support curvature station proposal')))
    def check():
        if callback is not None and callback():
            raise Cancelled('curvature station proposal cancelled')
        if literal(intersection_curve_record(curve)) != definition:
            raise Integrity('curvature station proposal owner changed')
        return False
    with intersection_curve_operation(check):
        check(); curve._shape()
        limits = Limits.read(SimpleNamespace(**dict(curve.qualification_values)))
        maximum_depth = depth_limit(limits.max_subdivision_depth)
        stations = []
        for value in parameters:
            check()
            if len(stations) >= limits.max_trace_segments+1:
                raise Refused('curvature proposal input station budget exhausted')
            stations.append(_binary(value))
            check()
        if len(stations) < 2 or any(a >= b for a,b in zip(stations,stations[1:])):
            raise GeometryError('curvature proposals require increasing source stations')
        first,last = curve.domain
        count = len(curve.owner.segments)
        mandatory = tuple(sorted((Q(i,count)-first)/(last-first) for i in range(count+1)
                                 if 0 < (Q(i,count)-first)/(last-first) < 1))
        output = set(stations)
        pending = [(a,b,0) for a,b in reversed(tuple(zip(stations,stations[1:])))]
        queries = 0
        budget = error*Q(3,4)
        while pending:
            check(); curve._shape()
            a,b,depth = pending.pop()
            breaks = tuple(point for point in mandatory if a < point < b)
            if breaks:
                # Never apply a single C2 theorem across a derivative jump.
                if any(Q(float(point)) != point for point in breaks):
                    continue  # explicit no-single-C2-piece applicability
                cuts = (a,*breaks,b)
            else:
                if queries >= limits.max_boxes_per_pair:
                    raise Refused('curvature proposal enclosure budget exhausted')
                charge_box(); queries += 1
                bounds = curve.enclose(a,b)
                check(); curve._shape()
                if not bounds:
                    raise Refused('curvature proposal owner returned no interval enclosure')
                if len(bounds) != 1 or bounds[0].parameter_range != (a,b):
                    continue  # no single certified C2 piece for this request
                acceleration = bounds[0].world_second_derivative
                maximum = sum((max(abs(value.lo),abs(value.hi)) for value in acceleration),Q(0))
                if maximum*(b-a)**2/8 <= budget:
                    continue
                # Newly proposed stations may round to binary64. Requested
                # stations NEVER round; every subsequent child proof uses
                # these actual rounded endpoints, not the ideal halves.
                midpoint = Q(float((a+b)/2))
                if not a < midpoint < b:
                    raise Refused('curvature proposal interval has no interior binary64 station')
                cuts = (a,midpoint,b)
            if depth >= maximum_depth:
                raise Refused('curvature proposal subdivision depth exhausted')
            additions = set(cuts[1:-1])-output
            if len(output)+len(additions)-1 > limits.max_trace_segments:
                raise Refused('curvature proposal trace budget exhausted')
            for _ in cuts[1:-1]:
                charge_split()
            output.update(additions)
            pending.extend((lo,hi,depth+1) for lo,hi in reversed(tuple(zip(cuts,cuts[1:]))))
        check(); curve._shape()
        return tuple(Q(point.numerator,point.denominator) for point in sorted(output))
