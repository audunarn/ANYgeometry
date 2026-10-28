"""Exact first-derivative intervals and injective projected NURBS charts."""
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
import itertools
import numpy as np

from .branch_predicates import _rect, _range, _derivative_ranges
from .errors import GeometryError
from .nurbs import NurbsSurface
from .plane_nurbs import sub_patches_of, charge_box, charge_split, depth_limit


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def mul(a, b):
    values = [x*y for x in a for y in b]
    return min(values), max(values)


def scale(a, value):
    return mul(a, (value, value))


def sub(a, b):
    return a[0]-b[1], a[1]-b[0]


@lru_cache(maxsize=64)
def _homogeneous(surface):
    return tuple(tuple(sub_patches_of(tuple(Q(i == axis) for i in range(3)), Q(0), surface))
                 for axis in range(3)) + (tuple(sub_patches_of((Q(0),)*3, Q(-1), surface)),)


def derivative_intervals(surface, box):
    """Exact chart-derivative component intervals over the complete box."""
    from .surfaces import Plane
    if isinstance(surface, Plane):
        return tuple(tuple((Q(float(x)), Q(float(x))) for x in vector) for vector in (surface.u_vector, surface.v_vector))
    if not isinstance(surface, NurbsSurface):
        raise GeometryError('exact metric intervals require a NURBS support')
    rows = _homogeneous(surface)
    chart_u, chart_v = (tuple(map(Q, r)) for r in (surface.u_range, surface.v_range))
    urange = sorted(chart_u[0]+Q(t)*(chart_u[1]-chart_u[0]) for t in box[0])
    vrange = sorted(chart_v[0]+Q(t)*(chart_v[1]-chart_v[0]) for t in box[1])
    collected = [[], []]
    for index, patch in enumerate(rows[3]):
        u = max(urange[0], patch.u_range[0]), min(urange[1], patch.u_range[1])
        v = max(vrange[0], patch.v_range[0]), min(vrange[1], patch.v_range[1])
        if u[0] >= u[1] or v[0] >= v[1]:
            continue
        values, derivatives = [], []
        for component in rows:
            coefficients = _rect(component[index].coefficients, patch.u_range, patch.v_range, u, v)
            values.append(_range(coefficients))
            derivatives.append(_derivative_ranges(coefficients, u[1]-u[0], v[1]-v[0]))
        w = values[3]
        if w[0] <= 0:
            raise GeometryError('NURBS metric denominator is not positive')
        inverse_w2 = Q(1)/(w[1]*w[1]), Q(1)/(w[0]*w[0])
        for axis in range(2):
            span = (chart_u, chart_v)[axis]
            factor = span[1]-span[0]
            collected[axis].append(tuple(scale(mul(sub(mul(derivatives[i][axis], w),
                mul(values[i], derivatives[3][axis])), inverse_w2), factor) for i in range(3)))
    if not collected[0]:
        raise GeometryError('NURBS derivative box has no positive area')
    return tuple(tuple((min(row[i][0] for row in pieces), max(row[i][1] for row in pieces))
                       for i in range(3)) for pieces in collected)


def _trace_derivative_intervals(surface, box):
    """Derivative enclosure on a closed trace line/point, not an immersion proof.

    Differentiate positive-area knot patches BEFORE restricting to a zero-width
    trace coordinate. Include both one-sided limits at every coincident knot.
    The public positive-area derivative_intervals contract is unchanged.
    """
    try:
        box=tuple(tuple(map(Q,pair)) for pair in box)
    except (TypeError,ValueError,OverflowError) as error:
        raise GeometryError('invalid trace derivative box') from error
    if len(box)!=2 or any(len(pair)!=2 for pair in box):
        raise GeometryError('invalid trace derivative box')
    if all(a<b for a,b in box):return derivative_intervals(surface,box)
    if any(not 0<=a<=b<=1 for a,b in box):
        raise GeometryError('trace derivative box leaves support chart')
    if type(surface) is not NurbsSurface:
        raise GeometryError('degenerate trace derivatives require a canonical NURBS support')
    rows=_homogeneous(surface)
    spans=tuple(tuple(map(Q,pair)) for pair in (surface.u_range,surface.v_range))
    ranges=tuple(tuple(sorted(a+(b-a)*t for t in pair)) for (a,b),pair in zip(spans,box))
    collected=[[],[]]
    for index,patch in enumerate(rows[3]):
        u=max(ranges[0][0],patch.u_range[0]),min(ranges[0][1],patch.u_range[1])
        v=max(ranges[1][0],patch.v_range[0]),min(ranges[1][1],patch.v_range[1])
        if u[0]>u[1] or v[0]>v[1]:continue
        uw=patch.u_range[1]-patch.u_range[0];vw=patch.v_range[1]-patch.v_range[0]
        if uw<=0 or vw<=0:raise GeometryError('degenerate native knot patch')
        values=[];derivatives=[]
        for component in rows:
            grid=component[index].coefficients;p=len(grid)-1;q=len(grid[0])-1
            du=[[p*(grid[i+1][j]-grid[i][j])/uw for j in range(q+1)] for i in range(p)] if p else [[Q(0)]*(q+1)]
            dv=[[q*(row[j+1]-row[j])/vw for j in range(q)] for row in grid] if q else [[Q(0)] for row in grid]
            values.append(_range(_rect(grid,patch.u_range,patch.v_range,u,v)))
            derivatives.append(tuple(_range(_rect(g,patch.u_range,patch.v_range,u,v)) for g in (du,dv)))
        w=values[3]
        if w[0]<=0:raise GeometryError('NURBS metric denominator is not positive')
        inverse_w2=Q(1)/(w[1]*w[1]),Q(1)/(w[0]*w[0])
        for axis,(a,b) in enumerate(spans):
            collected[axis].append(tuple(scale(mul(sub(mul(derivatives[i][axis],w),
                mul(values[i],derivatives[3][axis])),inverse_w2),b-a) for i in range(3)))
    if not collected[0]:raise GeometryError('trace derivative box has no native support')
    return tuple(tuple((min(row[i][0] for row in pieces),max(row[i][1] for row in pieces))
        for i in range(3)) for pieces in collected)


@dataclass(frozen=True)
class ProjectedChartCertificate:
    box: tuple
    projection: tuple
    subdivisions: int


def certify_projected_chart(surface, box=((Q(0), Q(1)), (Q(0), Q(1))), *, cancellation_check=None):
    """Prove a single world projection strongly monotone on this convex chart.

    A positive symmetric derivative part gives injectivity of the projection
    on the whole box, hence immersion and injectivity of the surface there.
    Different certificate subdivisions always use the SAME projection.
    """
    box = tuple(tuple(map(Q, interval)) for interval in box)
    centre = np.array([[float(sum(interval)/2) for interval in box]])
    su, sv = surface.derivatives_many(centre)
    jac = np.column_stack((su[0], sv[0]))
    candidates = []
    j3 = [[Q(float(jac[i, k])) for k in range(2)] for i in range(3)]
    gram = [[sum(j3[i][r]*j3[i][c] for i in range(3)) for c in range(2)] for r in range(2)]
    det = gram[0][0]*gram[1][1]-gram[0][1]*gram[1][0]
    if det:
        inverse = ((gram[1][1]/det, -gram[0][1]/det), (-gram[1][0]/det, gram[0][0]/det))
        candidates.append(tuple(tuple(sum(inverse[r][k]*j3[i][k] for k in range(2))
                                      for i in range(3)) for r in range(2)))
    for axes in itertools.combinations(range(3), 2):
        j = [[Q(float(jac[i, k])) for k in range(2)] for i in axes]
        det = j[0][0]*j[1][1]-j[0][1]*j[1][0]
        if not det:
            continue
        inverse = ((j[1][1]/det, -j[0][1]/det), (-j[1][0]/det, j[0][0]/det))
        candidates.append(tuple(tuple(sum(inverse[r][k]*Q(i == axes[k]) for k in range(2))
                                      for i in range(3)) for r in range(2)))
    for projection in candidates:
        pending, divisions, failed = [(box, 0)], 0, False
        while pending:
            if cancellation_check is not None and cancellation_check():
                raise GeometryError('NURBS chart metric certification cancelled')
            current, depth = pending.pop()
            charge_box()
            du, dv = derivative_intervals(surface, current)
            j = [[sum_intervals(scale(d[k], projection[row][k]) for k in range(3))
                  for d in (du, dv)] for row in range(2)]
            off = scale(add(j[0][1], j[1][0]), Q(1, 2))
            off2 = max(abs(off[0]), abs(off[1]))**2
            if j[0][0][0] > 0 and j[1][1][0] > 0 and j[0][0][0]*j[1][1][0] > off2:
                continue
            if depth >= depth_limit(8):
                failed = True
                break
            charge_split()
            divisions += 1
            u, v = current
            axis = 0 if u[1]-u[0] >= v[1]-v[0] else 1
            middle = sum(current[axis])/2
            for interval in ((current[axis][0], middle), (middle, current[axis][1])):
                child = list(current)
                child[axis] = interval
                pending.append((tuple(child), depth+1))
        if not failed:
            return ProjectedChartCertificate(box, projection, divisions)
    raise GeometryError('NURBS support immersion and injective projection are not certified')


def sum_intervals(values):
    result = (Q(0), Q(0))
    for value in values:
        result = add(result, value)
    return result
