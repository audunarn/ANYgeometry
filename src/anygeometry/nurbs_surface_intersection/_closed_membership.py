"""Private sufficient sign proof for one completely accounted interior zero loop.

Only the canonical prepared-curve closure constructs/uses this object. It is
not a receipt and accepting a caller-created instance grants no authority.
"""
from fractions import Fraction as Q


class ClosedSignMembership:
    def __init__(self, result, component_index, chart_bounds, tick):
        from ._source import IntersectionStatus, WorkCounters
        from ._records import StructuralCertificate, RationalProjectionCertificate
        from ._atlas import _Surface
        self.applicable = False
        self.tick = tick
        tick()
        if (result.status is not IntersectionStatus.CERTIFIED or not result.cover_complete
                or len(result.components) != 1 or component_index != 0
                or not result.components[0].closed
                or type(result.structural) not in (StructuralCertificate,RationalProjectionCertificate)
                or result.algorithm not in ('shared-affine-projection-scalar',
                                            'shared-separable-rational-projection-scalar')):
            return
        a,b = result.first_source.values,result.second_source.values
        if a[6:] != b[6:]:
            return
        component=result.components[0]
        roots=component.root_ids
        if (len(roots)!=len(component.arcs)+1 or roots[0]!=roots[-1]
                or len(set(roots[:-1]))!=len(component.arcs)
                or any((arc.first_root,arc.second_root)!=(roots[i],roots[i+1])
                       for i,arc in enumerate(component.arcs))):
            return
        # C1 through every active knot excludes a piecewise |F| contact even
        # if each open span separately has a nonzero derivative. This is a
        # sufficient restriction; unsupported continuity keeps old winding.
        for degree,knots,selected in ((a[0],a[2],a[6]),(a[1],a[3],a[7])):
            low,high = sorted(selected)
            if any(knots.count(k)>degree-1 for k in set(knots) if low<k<high):
                return
        from ._evaluate import _grid
        from ._curve_bounds import derivative_grid
        for arc in result.components[0].arcs:
            tick()
            box = result.boxes[arc.box_index]
            if box.kind not in ('u','v') or arc.slicing_axis != (0 if box.kind=='v' else 1):
                return
            # _grid permutes to [free][dependent]. A strict Bernstein hull
            # for the dependent derivative proves grad F !=0 over the whole
            # closed regular box, including every endpoint/knot occurrence.
            values = [entry[0] for row in derivative_grid(_grid(box),1) for entry in row]
            if not values or not (min(values)>0 or max(values)<0):
                return
        charts = chart_bounds(Q(0),Q(len(result.components[0].arcs)))
        if not all(0 < lo <= hi < 1 for support in charts for lo,hi in support):
            return
        self.supports = tuple(_Surface(source,WorkCounters()) for source in
                              (result.first_source,result.second_source))
        self.ranges = a[6:]
        # Equal ordered selected ranges map the normalized corner to the SAME
        # native parameters, including reversal. This is known outside because
        # the complete zero locus is strictly interior.
        points = tuple(s.surface.exact_point(Q(0),Q(0),net=s.net) for s in self.supports)
        coordinate = next((j for j,d in enumerate(result.structural.direction)
                           if d != 0 and points[1][j] != points[0][j]),None)
        if coordinate is None:
            return
        self.coordinate = coordinate
        self.outside_positive = points[1][coordinate] > points[0][coordinate]
        tick()
        self.applicable = True

    def contains(self, point, clearance):
        self.tick()
        if not self.applicable:
            return None
        # Conversions can execute user code. Recheck before proof work and at
        # issuance exactly as the context's other interval methods do.
        point = tuple(map(Q,point)); clearance = Q(clearance)
        self.tick()
        if len(point) != 2 or clearance < 0:
            return None
        chart = tuple((x-clearance,x+clearance) for x in point)
        if not all(0 <= lo <= hi <= 1 for lo,hi in chart):
            return None
        native = tuple(tuple(sorted((Q(a)+lo*(Q(b)-Q(a)),Q(a)+hi*(Q(b)-Q(a)))))
                       for (lo,hi),(a,b) in zip(chart,self.ranges))
        from ..plane_nurbs import charge_box
        hulls = []
        for support in self.supports:
            self.tick()
            charge_box()
            if len(support._hulls) >= 2048:
                support._hulls.clear()
            hulls.append(support.hull(native))
        j = self.coordinate
        lo = hulls[1][0][j]-hulls[0][1][j]
        hi = hulls[1][1][j]-hulls[0][0][j]
        self.tick()
        if lo > 0:
            return not self.outside_positive
        if hi < 0:
            return self.outside_positive
        return None
