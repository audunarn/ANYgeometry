"""Declared chart access to implicit plane/NURBS section curves.

This is not a polynomial pcurve: chart level events are obtained by exact
restriction of the section polynomial on each certified graph arc.
"""
from dataclasses import dataclass
from fractions import Fraction as Q

from .errors import GeometryError
from .plane_nurbs import PlaneNurbsNotCertified, Root
from .exact_levels import _bernstein_to_power, _roots_in_unit
from .support_curves import PlaneNurbsCurve, _de_casteljau
import numpy as np


@dataclass(frozen=True)
class SectionLevelEvent:
    parameter_interval: tuple
    arc_index: int


@dataclass(frozen=True)
class SectionChart:
    section: PlaneNurbsCurve

    def evaluate(self, t):
        return self.section.chart(t)

    def derivative(self, t):
        return self.section.chart_derivative(t)

    def breakpoints(self):
        return self.section.breakpoints()

    def certified_polyline(self, tolerance, max_pieces=1 << 16):
        from .nurbs_section_faces import _Chart, _piece_rect, _corners
        from .nurbs_domain import _exact_segment_distance
        from .nurbs import _float_up, _sqrt_up
        from .surface_capabilities import capabilities
        section = self.section
        chart = _Chart(section.surface)
        sigma = capabilities(section.surface).jacobian_bounds(((0., 1.), (0., 1.))).sigma_max
        factor = _sqrt_up(Q(2))*Q(sigma)
        cuts = (0., *self.breakpoints(), 1.)
        pending = list(zip(cuts, cuts[1:]))
        accepted = []
        while pending:
            lo, hi = pending.pop()
            rect = _piece_rect(section, chart, lo, hi)
            ends = [tuple(map(Q, p)) for p in section.chart(np.array([lo, hi]))]
            deviation = max(_exact_segment_distance(p, *ends) for p in _corners(rect))
            endpoint_error = Q(0)
            for t, endpoint in zip((lo, hi), ends):
                knot_rect = section.chart_enclosure(t)
                end_rect = chart.rect(*knot_rect)
                endpoint_error = max(endpoint_error, max(_exact_segment_distance(p, endpoint, endpoint)
                                                         for p in _corners(end_rect)))
            bound = _float_up(factor*(deviation+endpoint_error))
            if bound <= tolerance:
                accepted.append((lo, hi, bound))
                continue
            if len(pending)+len(accepted) >= max_pieces or hi-lo <= 1e-12:
                raise GeometryError('certified section polyline exhausted its piece budget')
            middle = (lo+hi)/2
            pending.extend(((middle, hi), (lo, middle)))
        accepted.sort()
        parameters = [accepted[0][0], *(hi for lo, hi, bound in accepted)]
        return section.chart(np.array(parameters)), max(bound for lo, hi, bound in accepted)

    def level_events(self, axis, value):
        """All isolated level events in local curve parameter, or whole coincidence.

        Mixed coincident spans and unresolved clipping at algebraic joints
        explicitly refuse; they must never silently drop a crossing.
        """
        curve = self.section
        chart_range = (curve.surface.u_range, curve.surface.v_range)[axis]
        target = Q(chart_range[0]) + Q(value) * (Q(chart_range[1])-Q(chart_range[0]))
        arcs = curve._arcs()
        count = len(arcs)
        domain0, domain1 = map(Q, curve.domain)
        found, runs = [], 0
        for index, arc in enumerate(arcs):
            if axis == arc.slicing:
                if not min(arc.s0, arc.s1) <= target <= max(arc.s0, arc.s1):
                    continue
                values = [(target, target)]
            else:
                box = arc.box
                interval = box.v_range if axis == 1 else box.u_range
                if not interval[0] <= target <= interval[1]:
                    continue
                t = (target-interval[0])/(interval[1]-interval[0])
                coefficients = ([_de_casteljau(list(row), t) for row in box.coefficients]
                                if axis == 1 else
                                [_de_casteljau(list(column), t) for column in zip(*box.coefficients)])
                if not any(coefficients):
                    runs += 1
                    continue
                polynomial = _bernstein_to_power(coefficients)
                slicing_range = box.u_range if axis == 1 else box.v_range
                values = []
                for lo, hi, square_free in _roots_in_unit(polynomial):
                    root = Root(1-arc.slicing, target,
                                slicing_range[0]+lo*(slicing_range[1]-slicing_range[0]),
                                slicing_range[0]+hi*(slicing_range[1]-slicing_range[0]),
                                square_free, slicing_range)
                    for _ in range(256):
                        lower, upper = sorted((arc.s0, arc.s1))
                        if root.hi < lower or root.lo > upper:
                            break
                        if lower <= root.lo <= root.hi <= upper and root.hi-root.lo <= Q(1, 2**60):
                            values.append((root.lo, root.hi))
                            break
                        root.refine()
                    else:
                        raise PlaneNurbsNotCertified('section level event cannot be clipped at an arc joint')
            for lo, hi in values:
                local = sorted((((Q(index)+(s-arc.s0)/(arc.s1-arc.s0))/count-domain0)
                                /(domain1-domain0) for s in (lo, hi)))
                if local[1] < 0 or local[0] > 1:
                    continue
                if local[0] < 0 or local[1] > 1:
                    raise PlaneNurbsNotCertified('section level event straddles a trimmed endpoint')
                found.append(SectionLevelEvent(tuple(local), index))
        if runs:
            if runs == count:
                return None
            raise GeometryError('mixed level runs on an implicit section are not yet qualified')
        ordered = sorted(found, key=lambda event: event.parameter_interval)
        result = []
        for event in ordered:
            if result and event.parameter_interval[0] <= result[-1].parameter_interval[1]:
                previous = result[-1].parameter_interval
                if previous[0] == previous[1] == event.parameter_interval[0] == event.parameter_interval[1]:
                    continue  # Same exact rational station on an arc joint.
                raise PlaneNurbsNotCertified('section level event identities are not separated')
            result.append(event)
        return tuple(result)

    def level_roots(self, axis, value):
        """Approximate API stations, refusing distinct events that round together.

        Topology callers must retain level_events' isolators and arc provenance.
        """
        events = self.level_events(axis, value)
        if events is None:
            return None
        values = tuple(float(sum(event.parameter_interval)/2) for event in events)
        if any(a >= b for a, b in zip(values, values[1:])):
            raise PlaneNurbsNotCertified('distinct section level events have no separated float stations')
        return values
