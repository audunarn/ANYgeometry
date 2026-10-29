"""Approximate FIRST-order chart witnesses for advisory numerical integration.

These are not enclosure/certification APIs. Points come from the fresh issued
owner context; literal derivative coefficients and a separated implicit minor
provide a first-order value without constructing Hessians or world jets.
"""
from fractions import Fraction
import math

from ._bernstein import derivative_grid,grid_ranges,subbox
from ._intervals import Interval
from ._source import NurbsSurfaceIntersectionRefused as Refused


class FirstOrderWitnessNotQualified(Refused):
    """This advisory shortcut cannot provide a regular finite numeric witness."""


def _divide(a,b):
    if not b.excludes_zero():
        raise FirstOrderWitnessNotQualified('derivative witness denominator is not separated from zero')
    return a*Interval(1/b.hi,1/b.lo)


class FirstOrderChartWitness:
    """Preparation cache owned ONLY by one freshly issued evaluation context."""
    def __init__(self,result,component_index,tick):
        self.result,self.component_index,self.tick = result,component_index,tick
        self.scalar,self.pairs = {},{}
        self.surfaces = None

    def _scalar(self,arc_index,point):
        from ._evaluate import _grid,_RootCopy
        arc = self.result.components[self.component_index].arcs[arc_index]
        if arc_index not in self.scalar:
            box = self.result.boxes[arc.box_index]
            grid = _grid(box)  # [slicing][dependent], including native v slicing
            derivatives = tuple(derivative_grid(grid,k) for k in range(2))
            roots = tuple(_RootCopy(self.result.roots[k]) for k in (arc.first_root,arc.second_root))
            delta = roots[1].projection(arc.slicing_axis)-roots[0].projection(arc.slicing_axis)
            native = (box.u_range,box.v_range)
            slicing = native[arc.slicing_axis]
            self.scalar[arc_index] = (native,derivatives,delta.scale(1/(slicing[1]-slicing[0])))
        native,derivatives,speed = self.scalar[arc_index]
        if not speed.excludes_zero():
            raise FirstOrderWitnessNotQualified('derivative witness endpoint order is not separated')
        local = tuple(((lo-a)/(b-a),(hi-a)/(b-a)) for (lo,hi),(a,b) in zip(point.native_box,native))
        if arc.slicing_axis == 1:
            local = local[::-1]
        ds,dx = (grid_ranges(subbox(g,*local[0],*local[1]))[0] for g in derivatives)
        dependent = -_divide(ds,dx)*speed
        velocity = (speed,dependent) if arc.slicing_axis == 0 else (dependent,speed)
        return tuple(v.scale(b-a) for v,(a,b) in zip(velocity,native))*2

    def _generic(self,arc_index,point):
        from ._atlas import _Surface
        from ._certificate import _Pair
        from ._source import WorkCounters
        from ._curve_bounds import solve
        if self.surfaces is None:
            self.surfaces = tuple(_Surface(s,WorkCounters()) for s in (
                self.result.first_source,self.result.second_source))
        surfaces = self.surfaces
        arc = self.result.components[self.component_index].arcs[arc_index]
        witness = self.result.atlas.witnesses[arc.witness]
        if witness.spans not in self.pairs:
            knots = tuple(surfaces[j].spans[k][witness.spans[j][k]] for j in range(2) for k in range(2))
            pair = _Pair.from_pieces(*(surfaces[j].piece(*witness.spans[j]) for j in range(2)))
            self.pairs[witness.spans] = pair,knots
        pair,knots = self.pairs[witness.spans]
        free = witness.free_coordinate
        def projection(ident):
            root = self.result.roots[ident]
            old = self.result.atlas.witnesses[root.witness]
            j,k = divmod(free,2)
            a,b = surfaces[j].spans[k][old.spans[j][k]]
            lo,hi = root.region[free]
            return Interval(a+lo*(b-a),a+hi*(b-a))
        speed = (projection(arc.second_root)-projection(arc.first_root)).scale(1/(knots[free][1]-knots[free][0]))
        if not speed.excludes_zero():
            raise FirstOrderWitnessNotQualified('derivative witness endpoint order is not separated')
        local = tuple(((lo-a)/(b-a),(hi-a)/(b-a)) for (lo,hi),(a,b) in zip(point.native_box,knots))
        _value,jacobian = pair.value_and_jacobian(pair.enclose(local))
        columns = tuple(k for k in range(4) if k != free)
        try:
            dependent = solve([[row[k] for k in columns] for row in jacobian],
                              [-row[free]*speed for row in jacobian])
        except Refused as error:
            # This pure first-order solve has no callbacks or resource charges;
            # only its separated-minor failure becomes an advisory fallback.
            raise FirstOrderWitnessNotQualified(str(error)) from error
        velocity = tuple(speed if k == free else dependent[columns.index(k)] for k in range(4))
        return tuple(v.scale(b-a) for v,(a,b) in zip(velocity,knots))

    def evaluate(self,arc_index,point):
        self.tick()
        velocity = (self._generic(arc_index,point) if self.result.algorithm == 'generic-atlas'
                    else self._scalar(arc_index,point))
        ranges = self.result.first_source.values[-2:]+self.result.second_source.values[-2:]
        output = []
        for value,(a,b) in zip(velocity,ranges):
            scaled = value.scale(1/(Fraction(b)-Fraction(a)))
            try:
                lo,hi,witness = float(scaled.lo),float(scaled.hi),float(scaled.mid)
            except (OverflowError,ValueError) as error:
                raise FirstOrderWitnessNotQualified('derivative witness is not numerically representable') from error
            if not all(math.isfinite(x) for x in (lo,hi,witness)):
                raise FirstOrderWitnessNotQualified('derivative witness is not finite')
            if witness == 0. and scaled.excludes_zero():
                raise FirstOrderWitnessNotQualified('nonzero derivative witness underflows its numeric representation')
            output.append(witness)
        self.tick()
        return tuple(output)
