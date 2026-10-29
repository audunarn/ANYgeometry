"""Declared pcurves for a literal two-support intersection owner."""
from dataclasses import dataclass
from fractions import Fraction as Q
import numpy as np

from .errors import GeometryError
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation


class IntersectionChartContactUndecided(GeometryError):
    """Only the queried boundary contact remained unresolved at its depth limit."""


@dataclass(frozen=True)
class IntersectionChart:
    section: NurbsSurfaceIntersectionCurve

    def evaluate(self,t):
        return self.section.chart(t)

    def derivative(self,t):
        return self.section.chart_derivative(t)

    def derivative_witness(self,t):
        """Estimate-only shortcut; mathematical refusal retains the old path."""
        from .nurbs_surface_intersection._curve_witness import FirstOrderWitnessNotQualified
        try:
            return self.section.chart_derivative_witness(t)
        except FirstOrderWitnessNotQualified:
            return self.derivative(t)

    def breakpoints(self):
        return self.section.breakpoints()

    def level_roots(self,axis,value):
        raise GeometryError('intersection pcurve level events require certified constrained-root clipping')

    def certified_polyline(self,tolerance,max_pieces=1<<16):
        from .nurbs_domain import _exact_segment_distance
        from .nurbs import _float_up,_sqrt_up
        from .surface_capabilities import capabilities
        sigma = Q(capabilities(self.section.surface).jacobian_bounds(((0.,1.),(0.,1.))).sigma_max)
        factor = _sqrt_up(Q(2))*sigma
        knots = (0.,*self.breakpoints(),1.)
        pending = list(zip(knots,knots[1:]))
        accepted = []
        with intersection_curve_operation():
            while pending:
                a,b = pending.pop()
                rect = self.section.chart_box(a,b)
                endpoints = [tuple(map(Q,p)) for p in self.evaluate(np.array([a,b]))]
                corners = [(x,y) for x in rect[0] for y in rect[1]]
                deviation = max(_exact_segment_distance(p,*endpoints) for p in corners)
                jets = self.section.enclose(a,b)
                if len(jets) == 1:
                    acceleration = (jets[0].first_chart_second_derivative,
                                    jets[0].second_chart_second_derivative)[self.section.chart_support]
                    magnitude = _sqrt_up(sum(max(abs(x.lo),abs(x.hi))**2 for x in acceleration))
                    endpoint_error = Q(0)
                    for t,witness in zip((a,b),endpoints):
                        box = self.section.chart_box(t,t)
                        endpoint_error = max(endpoint_error,max(_exact_segment_distance((x,y),witness,witness)
                            for x in box[0] for y in box[1]))
                    interpolation = magnitude*(Q(b)-Q(a))**2/8+endpoint_error
                    deviation = min(deviation,interpolation)
                bound = _float_up(factor*deviation)
                if bound <= tolerance:
                    accepted.append((a,b,bound))
                    continue
                if len(accepted)+len(pending) >= max_pieces or b-a <= 1e-12:
                    raise GeometryError('intersection pcurve polyline exhausted its certified piece budget')
                m = (a+b)/2
                pending.extend(((m,b),(a,m)))
            accepted.sort()
            points = self.evaluate(np.array([accepted[0][0],*(b for a,b,bound in accepted)]))
        return points,max(bound for a,b,bound in accepted)


def _closed_sign_contains(curves,support,point,clearance):
    """Sufficient canonical-owner sign proof; None means use winding."""
    if type(curves) not in (tuple,list) or not curves or any(type(c) is not NurbsSurfaceIntersectionCurve for c in curves):
        return None
    from .definition_binding import definition_checksum
    from .nurbs_surface_intersection import SourceDefinition
    snapshot = definition_checksum((curves,support))
    source = SourceDefinition.capture(support)
    first = curves[0]
    if not first.owner.closed:
        return None
    views=[]; intervals=[]
    for curve in curves:
        if (curve.owner != first.owner or curve.tolerance_values != first.tolerance_values
                or curve.qualification_values != first.qualification_values):
            return None
        view=curve.for_support(support)
        view._shape()       # issued canonical owner/config/runtime validation
        views.append(view)
        intervals.append(tuple(sorted(view.domain)))
    reach=Q(0)
    for a,b in sorted(intervals):
        if a != reach or b <= a:
            return None
        reach=b
    if reach != 1 or any(a.point_identity(1) != b.point_identity(0)
                        for a,b in zip(views,views[1:]+views[:1])):
        return None
    shape=views[0]._shape()
    answer=shape._context._closed_sign_membership(point,clearance)
    # Callbacks may have touched any occurrence or the external support.
    # Revalidate every original source/view before returning proof authority.
    for curve,view in zip(curves,views):
        curve._shape()
        view._shape()
    if definition_checksum((curves,support)) != snapshot or not source.matches(support):
        from .nurbs_surface_intersection import NurbsSurfaceIntersectionIntegrityError
        raise NurbsSurfaceIntersectionIntegrityError('membership loop or support changed during qualification')
    return answer


def contains_owner_loop(curves,support,point,clearance=Q(0)):
    with intersection_curve_operation():
        answer=_closed_sign_contains(curves,support,point,clearance)
        if answer is not None:
            return answer
        return _contains_owner_loop_winding(curves,support,point,clearance)


def _contains_owner_loop_winding(curves,support,point,clearance=Q(0)):
    """Winding via a convex interval homotopy which excludes the query point.

    Each true curve interval and its witness chord lie in one certified convex
    rectangle separated from the query's clearance square. Their homotopy
    cannot change winding. Unresolved boundary contact always refuses.
    """
    point = tuple(map(Q,point)); clearance = Q(clearance)
    vertices = []
    joint_witnesses = {}
    with intersection_curve_operation():
        for curve in curves:
            if isinstance(curve,NurbsSurfaceIntersectionCurve):
                curve = curve.for_support(support)
                knots = (0.,*curve.breakpoints(),1.)
                pending = [(a,b,0) for a,b in zip(knots,knots[1:])]
                accepted = []
                def joint(parameter):
                    identity = curve.point_identity(parameter)
                    if identity not in joint_witnesses:
                        joint_witnesses[identity] = tuple(map(Q,curve.chart(float(parameter))))
                    return joint_witnesses[identity]
                while pending:
                    a,b,depth = pending.pop()
                    box = curve.chart_box(a,b)
                    # One actual polygon vertex per declared source occurrence,
                    # shared across independently refined pieces AND edge seams.
                    ends = [joint(a),joint(b)]
                    box = tuple((min(row[0],*(p[k] for p in ends)),max(row[1],*(p[k] for p in ends)))
                                for k,row in enumerate(box))
                    if any(hi < point[k]-clearance or lo > point[k]+clearance for k,(lo,hi) in enumerate(box)):
                        accepted.append((a,ends[0]))
                        continue
                    if len(accepted)+len(pending) >= 65536:
                        raise GeometryError('intersection pcurve containment piece budget exhausted')
                    if depth >= 28:
                        raise IntersectionChartContactUndecided('intersection pcurve boundary contact is unresolved')
                    m = (a+b)/2
                    pending.extend(((m,b,depth+1),(a,m,depth+1)))
                vertices.extend(p for a,p in sorted(accepted))
            else:
                raise GeometryError('mixed intersection loop containment requires certified joint events')
    winding = 0
    for a,b in zip(vertices,vertices[1:]+vertices[:1]):
        orient = (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
        if a[1] <= point[1] < b[1] and orient > 0:
            winding += 1
        elif b[1] <= point[1] < a[1] and orient < 0:
            winding -= 1
    return winding != 0
