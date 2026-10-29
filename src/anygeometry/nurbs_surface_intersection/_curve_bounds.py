"""Continuous interval bounds from validated implicit equations, never samples."""
from dataclasses import dataclass
from fractions import Fraction

from ._bernstein import derivative_grid, subbox, grid_ranges, bezier_piece
from ._intervals import Interval, float_down, float_up
from ._continuation import _det
from ._source import NurbsSurfaceIntersectionRefused as Refused


def divide(a,b):
    if not b.excludes_zero():
        raise Refused("continuous derivative denominator is not separated from zero")
    return a*Interval(1/b.hi,1/b.lo)


def solve(matrix, rhs):
    determinant = _det(matrix)
    out = []
    for k in range(3):
        replaced = [[rhs[i] if j == k else matrix[i][j] for j in range(3)] for i in range(3)]
        out.append(divide(_det(replaced),determinant))
    return out


def ranges(grid, box):
    return grid_ranges(subbox(grid,*box[0],*box[1]))


def jet(grid,box):
    first = tuple(derivative_grid(grid,k) for k in range(2))
    return (ranges(grid,box),tuple(ranges(g,box) for g in first),
            tuple(tuple(ranges(derivative_grid(g,k),box) for k in range(2)) for g in first))


class _BoundPreparation:
    """Private per-issued-context coefficients; never a caller proof cache.

    Coefficient tuples stay closed over by the owner evaluator. No public
    interval or result object is retained as a cached mutable authority.
    """
    def __init__(self,result):
        from ._atlas import _Surface
        from ._source import WorkCounters
        self.supports = tuple(_Surface(s,WorkCounters()) for s in (result.first_source,result.second_source))
        self.scalar = {}
        self.world = {}
        self.pairs = {}
        self.jets = {}
        self.partial = {}

    def coefficients(self,grid):
        key = id(grid)
        stored = self.jets.get(key)
        if stored is None:
            first = tuple(derivative_grid(grid,k) for k in range(2))
            second = tuple(tuple(derivative_grid(g,k) for k in range(2)) for g in first)
            if len(self.jets) >= 2048:
                self.jets.clear()
            self.jets[key] = stored = (grid,first,second)  # retain grid identity
        return stored

    def ranges(self,grid,box):
        from ._bernstein import _blossom_axis
        u0,u1 = map(Fraction,box[0]); v0,v1 = map(Fraction,box[1])
        first = grid[0][0]
        if all(point == first for row in grid for point in row):
            return tuple(Interval(x) for x in first)
        # The free-coordinate restriction repeats through every Newton step.
        # Cache that exact first-axis polynomial, not an interval approximation.
        key = (id(grid),u0.numerator,u0.denominator,u1.numerator,u1.denominator)
        stored = self.partial.get(key)
        if stored is None:
            columns = tuple(_blossom_axis([grid[i][j] for i in range(len(grid))],u0,u1)
                            for j in range(len(grid[0])))
            count = 1 if u0 == u1 else len(columns[0])
            rows = tuple(tuple(columns[j][i] for j in range(len(columns))) for i in range(count))
            if len(self.partial) >= 2048:
                self.partial.clear()
            self.partial[key] = stored = (grid,rows)  # strong identity reference
        rows = stored[1]
        restricted = tuple(tuple(_blossom_axis(row,v0,v1)[:1] if v0 == v1
                                 else _blossom_axis(row,v0,v1)) for row in rows)
        return grid_ranges(restricted)

    def jet(self,grid,box):
        _,first,second = self.coefficients(grid)
        return (self.ranges(grid,box),tuple(self.ranges(g,box) for g in first),
                tuple(tuple(self.ranges(g,box) for g in row) for row in second))


def prepare_bound_evaluator(result,component_index):
    """Create a closed operation-local evaluator AFTER canonical validation."""
    prepared = None
    def evaluate(arc_index,parameter_range,local_range,tick,world_only=False,chart_only=False):
        nonlocal prepared
        if prepared is None:
            prepared = _BoundPreparation(result)
        return _bound_segment(result,component_index,arc_index,parameter_range,local_range,tick,prepared,world_only,chart_only)
    return evaluate


def world_jet(grid, box, velocity, acceleration, jet_evaluator=jet):
    value,d,h = jet_evaluator(grid,box)
    first = tuple(sum((d[i][k]*velocity[i] for i in range(2)),Interval(0)) for k in range(4))
    second = tuple(sum((d[i][k]*acceleration[i] for i in range(2)),Interval(0))+
        sum((h[i][j][k]*velocity[i]*velocity[j] for i in range(2) for j in range(2)),Interval(0))
        for k in range(4))
    w,wp,wpp = value[3],first[3],second[3]
    v = tuple(divide(first[k],w)-divide(value[k]*wp,w*w) for k in range(3))
    a = tuple(divide(second[k],w)-divide(first[k]*wp,w*w).scale(2)-
        divide(value[k]*wpp,w*w)+divide(value[k]*wp*wp,w*w*w).scale(2) for k in range(3))
    return v,a


@dataclass(frozen=True, slots=True)
class CurveSegmentBounds:
    parameter_range: tuple
    first_native: tuple
    second_native: tuple
    first_chart: tuple
    second_chart: tuple
    world_lo: tuple
    world_hi: tuple
    first_chart_derivative: tuple
    second_chart_derivative: tuple
    first_chart_second_derivative: tuple
    second_chart_second_derivative: tuple
    world_derivative: tuple
    world_second_derivative: tuple


def bound_segment(result, component_index, arc_index, parameter_range, local_range=(0,1), tick=lambda: None):
    """Unprepared reference implementation, retained for parity checks."""
    return _bound_segment(result,component_index,arc_index,parameter_range,local_range,tick,None)


def _bound_segment(result, component_index, arc_index, parameter_range, local_range,tick,prepared,world_only=False,chart_only=False):
    """Enclose a requested subarc using its complete validated chart domain.

    The complete topology supplies existence; interval Newton/Krawczyk then
    localizes every point over the requested free-coordinate interval.
    """
    positions_only = world_only or chart_only
    from ._topology import chart_box
    from ._atlas import _Surface
    from ._source import WorkCounters
    from ._evaluate import _grid, _RootCopy, _snap
    arc = result.components[component_index].arcs[arc_index]
    a,b = ((s.surface for s in prepared.supports) if prepared is not None else
           (s.rebuild() for s in (result.first_source,result.second_source)))
    local_jet = jet if prepared is None else prepared.jet
    local_ranges = ranges if prepared is None else prepared.ranges
    if result.algorithm != "generic-atlas":
        box = result.boxes[arc.box_index]
        if prepared is None:
            grid = _grid(box)
        else:
            if arc.box_index not in prepared.scalar:
                if len(prepared.scalar) >= 2048:
                    prepared.scalar.clear()
                prepared.scalar[arc.box_index] = _grid(box)
            grid = prepared.scalar[arc.box_index]
        axis = arc.slicing_axis
        roots = tuple(_RootCopy(result.roots[k]) for k in (arc.first_root,arc.second_root))
        delta = roots[1].projection(axis)-roots[0].projection(axis)
        sr = box.u_range if axis == 0 else box.v_range
        speed = delta.scale(1/(sr[1]-sr[0]))
        ia,ib = (r.projection(axis) for r in roots)
        start = ia.scale(1-local_range[0])+ib.scale(local_range[0])
        end = ia.scale(1-local_range[1])+ib.scale(local_range[1])
        sigma = Interval(max(Fraction(0),(min(start.lo,end.lo)-sr[0])/(sr[1]-sr[0])),
                         min(Fraction(1),(max(start.hi,end.hi)-sr[0])/(sr[1]-sr[0])))
        dependent = Interval(0,1)
        derivative = derivative_grid(grid,1) if prepared is None else prepared.coefficients(grid)[1][1]
        for _ in range(16):
            tick()
            middle = dependent.mid
            fm = local_ranges(grid,((sigma.lo,sigma.hi),(middle,middle)))[0]
            dx = local_ranges(derivative,((sigma.lo,sigma.hi),(dependent.lo,dependent.hi)))[0]
            newton = _snap(Interval(middle)-divide(fm,dx))
            lo,hi = max(dependent.lo,newton.lo),min(dependent.hi,newton.hi)
            if lo > hi:
                raise Refused("continuous scalar contraction lost its certified root")
            new = Interval(lo,hi)
            if new == dependent:
                break
            dependent = new
        if not positions_only:
            # _grid is [slicing][dependent], including when native slicing is v.
            value,d,h = local_jet(grid,((sigma.lo,sigma.hi),(dependent.lo,dependent.hi)))
            slope = -divide(d[0][0],d[1][0])
            dependent_speed = slope*speed
            dependent_acceleration = -divide(h[0][0][0]*speed*speed+
                h[0][1][0]*speed*dependent_speed+h[1][0][0]*dependent_speed*speed+
                h[1][1][0]*dependent_speed*dependent_speed,d[1][0])
            velocity = (speed,dependent_speed) if axis == 0 else (dependent_speed,speed)
            acceleration = (Interval(0),dependent_acceleration) if axis == 0 else (dependent_acceleration,Interval(0))
        full_native = (box.u_range,box.v_range)
        local = ((sigma.lo,sigma.hi),(dependent.lo,dependent.hi)) if axis == 0 else ((dependent.lo,dependent.hi),(sigma.lo,sigma.hi))
        native_a = native_b = tuple((l+x*(r-l),l+y*(r-l)) for (l,r),(x,y) in zip(full_native,local))
        if not positions_only:
            if prepared is None:
                grids = tuple(bezier_piece(s,s.exact_net(),box.u_range,box.v_range) for s in (a,b))
            else:
                if arc.box_index not in prepared.world:
                    if len(prepared.world) >= 2048:
                        prepared.world.clear()
                    prepared.world[arc.box_index] = tuple(bezier_piece(s.surface,s.net,box.u_range,box.v_range)
                                                          for s in prepared.supports)
                grids = prepared.world[arc.box_index]
            world_velocity,world_acceleration = world_jet(grids[0],local,velocity,acceleration,local_jet)
            native_velocity = tuple(v.scale(r[1]-r[0]) for v,r in zip(velocity,full_native))*2
            native_acceleration = tuple(v.scale(r[1]-r[0]) for v,r in zip(acceleration,full_native))*2
    else:
        from ._certificate import _Pair
        w = result.atlas.witnesses[arc.witness]
        supports = (tuple(_Surface(s,WorkCounters()) for s in (result.first_source,result.second_source))
                    if prepared is None else prepared.supports)
        knots = tuple(supports[j].spans[k][w.spans[j][k]] for j in range(2) for k in range(2))
        if prepared is None:
            pair = _Pair.from_pieces(*(supports[j].piece(*w.spans[j]) for j in range(2)))
        else:
            if arc.witness not in prepared.pairs:
                if len(prepared.pairs) >= 2048:
                    prepared.pairs.clear()
                prepared.pairs[arc.witness] = _Pair.from_pieces(*(supports[j].piece(*w.spans[j]) for j in range(2)))
            pair = prepared.pairs[arc.witness]
        region = w.region
        free = w.free_coordinate
        columns = tuple(k for k in range(4) if k != free)
        def root_projection(ident):
            root = result.roots[ident]
            old = result.atlas.witnesses[root.witness]
            old_knots = tuple(supports[j].spans[k][old.spans[j][k]] for j in range(2) for k in range(2))
            x,y = root.region[free]
            l,r = old_knots[free]
            return Interval(l+x*(r-l),l+y*(r-l))
        speed = (root_projection(arc.second_root)-root_projection(arc.first_root)).scale(1/(knots[free][1]-knots[free][0]))
        ia,ib = root_projection(arc.first_root),root_projection(arc.second_root)
        start = ia.scale(1-local_range[0])+ib.scale(local_range[0])
        end = ia.scale(1-local_range[1])+ib.scale(local_range[1])
        l,r = knots[free]
        phi = Interval(max(region[free][0],(min(start.lo,end.lo)-l)/(r-l)),
                       min(region[free][1],(max(start.hi,end.hi)-l)/(r-l)))
        from ._certificate import certify_prepared
        region = tuple((phi.lo,phi.hi) if k == free else x for k,x in enumerate(region))
        for _ in range(16):
            tick()
            cert = certify_prepared(pair,knots,free,[Interval(*region[k]) for k in columns],phi)
            if not cert.unknown_enclosure:
                raise Refused("continuous generic contraction lacks a root enclosure")
            contracted = list(region)
            for k,x in zip(columns,cert.unknown_enclosure):
                x = _snap(x)
                lo,hi = max(region[k][0],x.lo),min(region[k][1],x.hi)
                if lo > hi:
                    raise Refused("continuous generic contraction lost its certified root")
                contracted[k] = (lo,hi)
            contracted = tuple(contracted)
            if contracted == region:
                break
            region = contracted
        if not positions_only:
            _,jac = pair.value_and_jacobian(pair.enclose(region))
            minor = [[row[k] for k in columns] for row in jac]
            dependent = solve(minor,[-row[free]*speed for row in jac])
            velocity = tuple(speed if k == free else dependent[columns.index(k)] for k in range(4))
            ja,jb = local_jet(pair.a,region[:2]),local_jet(pair.b,region[2:])
            def factor(support,indices):
                v,d,h = (ja,jb)[support]
                if not indices:
                    return v
                return d[indices[0]] if len(indices)==1 else h[indices[0]][indices[1]]
            hessian = []
            for i in range(4):
                row = []
                for j in range(4):
                    ai = tuple(k for k in (i,j) if k<2)
                    bi = tuple(k-2 for k in (i,j) if k>=2)
                    av,bv = factor(0,ai),factor(1,bi)
                    row.append(tuple(av[k]*bv[3]-bv[k]*av[3] for k in range(3)))
                hessian.append(row)
            rhs = [-sum((hessian[i][j][k]*velocity[i]*velocity[j] for i in range(4) for j in range(4)),Interval(0)) for k in range(3)]
            dependent_acceleration = solve(minor,rhs)
            acceleration = tuple(Interval(0) if k == free else dependent_acceleration[columns.index(k)] for k in range(4))
        native = tuple((l+x*(r-l),l+y*(r-l)) for (x,y),(l,r) in zip(region,knots))
        native_a,native_b = native[:2],native[2:]
        if not positions_only:
            world_velocity,world_acceleration = world_jet(supports[0].piece(*w.spans[0])[0],region[:2],velocity[:2],acceleration[:2],local_jet)
            native_velocity = tuple(v.scale(r-l) for v,(l,r) in zip(velocity,knots))
            native_acceleration = tuple(v.scale(r-l) for v,(l,r) in zip(acceleration,knots))
    charts = (chart_box(native_a,a.u_range,a.v_range),chart_box(native_b,b.u_range,b.v_range))
    supports = (tuple(_Surface(s,WorkCounters()) for s in (result.first_source,result.second_source))
                if prepared is None else prepared.supports)
    if prepared is not None:
        for support in supports:
            if len(support._hulls) >= 2048:
                support._hulls.clear()
    hulls = (supports[0].hull(native_a),supports[1].hull(native_b))
    low = tuple(float_down(max(x,y)) for x,y in zip(hulls[0][0],hulls[1][0]))
    high = tuple(float_up(min(x,y)) for x,y in zip(hulls[0][1],hulls[1][1]))
    if any(x > y for x,y in zip(low,high)):
        raise Refused("continuous original-support hulls do not meet")
    if chart_only:
        return charts
    if world_only:
        return low,high
    selected = (a.u_range,a.v_range,b.u_range,b.v_range)
    chart_velocity = tuple(v.scale(1/(Fraction(r[1])-Fraction(r[0]))) for v,r in zip(native_velocity,selected))
    chart_acceleration = tuple(v.scale(1/(Fraction(r[1])-Fraction(r[0]))) for v,r in zip(native_acceleration,selected))
    return CurveSegmentBounds(parameter_range,native_a,native_b,*charts,low,high,
        chart_velocity[:2],chart_velocity[2:],chart_acceleration[:2],chart_acceleration[2:],
        world_velocity,world_acceleration)
