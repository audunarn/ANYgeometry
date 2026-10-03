"""Exact completeness of a straight planar authored/current triangle partition.

Containment alone is insufficient. All children, exact areas and disjoint cell
interiors participate. No discretization, joint conformity or publication permit.
"""
from collections.abc import Mapping
from fractions import Fraction as F
from numbers import Integral

from .arrangement_geometry import LinePath
from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
from .authored_child_coverage import _literal_child_domain, _support_correspondence
from .authored_domain_coverage import _original_domain
from .errors import GeometryError
from .material_arrangement import MaterialDomain
from .material_cell_coverage import (_triangle_rows, _validate_domain_triangles,
                                    _frame, _chart_loops, _value, _orient)
from .surfaces import Plane


def _segments_meet(a, b, c, d):
    def on(a,b,p):
        return (_orient(a,b,p)==0 and all(min(x,y)<=z<=max(x,y) for x,y,z in zip(a,b,p)))
    ab=(_orient(a,b,c),_orient(a,b,d))
    cd=(_orient(c,d,a),_orient(c,d,b))
    return (ab[0]*ab[1]<0 and cd[0]*cd[1]<0 or
            on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b))


def _simple_area(domain, frame, check):
    if len(domain.boundaries)!=1 or any(type(p.curve) is not LinePath for p in domain.boundaries[0]):
        raise GeometryError('authored partition: only one straight outer loop is qualified')
    loop=_chart_loops(frame,domain,check)[0]
    points=tuple(tuple(_value(row,F(0)) for row in path) for path in loop)
    if len(points)<3 or len(set(points))!=len(points):
        raise GeometryError('authored partition: degenerate or repeated boundary vertex')
    edges=tuple(zip(points,(*points[1:],points[0])))
    for i,(a,b) in enumerate(edges):
        check()
        c=edges[(i+1)%len(edges)][1]
        if _orient(a,b,c)==0 and sum((y-x)*(z-y) for x,y,z in zip(a,b,c))<=0:
            raise GeometryError('authored partition: boundary backtracks')
        for j in range(i+1,len(edges)):
            check()
            if j==i+1 or i==0 and j==len(edges)-1:
                continue
            if _segments_meet(a,b,*edges[j]):
                raise GeometryError('authored partition: boundary is not simple')
    area=abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in edges))/2
    if not area:
        raise GeometryError('authored partition: boundary has zero area')
    return area


def _positive_overlap(first,second):
    # Convex triangle separating axes, exact rational signs. Contact along a
    # boundary is allowed; only positive-area interior intersection refuses.
    for triangle,other in ((first,second),(second,first)):
        for a,b in zip(triangle,(*triangle[1:],triangle[0])):
            if max(_orient(a,b,p) for p in other)<=0:
                return False
    return True


def _overlap_candidates(triangles,check):
    rows=[]
    for index,triangle in enumerate(triangles):
        check()
        low=tuple(min(p[axis] for p in triangle) for axis in (0,1))
        high=tuple(max(p[axis] for p in triangle) for axis in (0,1))
        rows.append((low[0],high[0],low[1],high[1],index))
    active=[]
    for low,high,bottom,top,index in sorted(rows):
        check()
        active=[row for row in active if row[0]>low]
        for _,y0,y1,previous in active:
            check()
            if min(top,y1)>max(bottom,y0):
                yield previous,index
        active.append((high,bottom,top,index))


def validate_prepared_authored_face_partition(model,correspondence,child_triangles_uv,
                                              *,cancellation_check=None):
    """Prove complete nonoverlapping triangle coverage of all current children.

    Mapping keys are every authenticated descendant ID; values are finite real
    (n,3,2) triangles in ORIGINAL UV. Exact coplanar Plane supports with one
    simple straight outer loop qualify. Holes, curved trims/supports and explicit
    parameterizations refuse. Success returns None and proves closed material
    equality of the supplied cells, every literal child, and the original face.
    It grants no node/constraint conformity, association, quality or publication
    permission. Queries never mutate the model; cancellation returns no proof.
    """
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)
    if not isinstance(child_triangles_uv,Mapping):
        raise GeometryError('authored partition requires a child-to-triangles mapping')
    rows={}
    for key,value in child_triangles_uv.items():
        if not isinstance(key,Integral) or isinstance(key,bool):
            raise GeometryError('authored partition requires integer child IDs')
        try:
            identifier=int(key)
        except (TypeError,ValueError,OverflowError) as error:
            raise GeometryError('authored partition requires usable integer child IDs') from error
        if identifier in rows:
            raise GeometryError('authored partition requires each child exactly once')
        rows[identifier]=_triangle_rows(value)
    if set(rows)!=set(correspondence.descendants):
        raise GeometryError('authored partition requires every authenticated child exactly once')
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)
    snapshots={key:_literal_child_domain(model,key) for key in sorted(rows)}
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)

    def check():
        if cancellation_check is not None and cancellation_check('authored partition coverage'):
            raise GeometryError('authored partition coverage cancelled')

    check()
    original=_original_domain(correspondence.authored_definition,check)
    if type(original.support) is not Plane:
        raise GeometryError('authored partition: only exact planar supports are qualified')
    frame=_frame(original.support)
    original_area=_simple_area(original,frame,check)
    triangles=[]
    child_area_total=F(0)
    for key,(face,domain) in snapshots.items():
        check()
        if type(face.surface) is not Plane or face.parameterization is not None:
            raise GeometryError('authored partition: literal child support is unqualified')
        _support_correspondence(original.support,face.surface,rows[key],check)
        literal=MaterialDomain(key,original.support,domain.boundaries)
        child_area=_simple_area(literal,frame,check)
        _validate_domain_triangles(original,rows[key],check)
        _validate_domain_triangles(literal,rows[key],check)
        area=F(0)
        for row in rows[key]:
            check()
            triangle=tuple(tuple(F(float(v)) for v in p) for p in row)
            signed=_orient(*triangle)
            if not signed:
                raise GeometryError('authored partition: degenerate cell')
            area+=abs(signed)/2
            triangles.append(triangle if signed>0 else triangle[::-1])
        if area!=child_area:
            raise GeometryError('authored partition: cells do not cover the literal child area')
        child_area_total+=child_area
    if child_area_total!=original_area:
        raise GeometryError('authored partition: literal child areas differ from original material')
    for first,second in _overlap_candidates(triangles,check):
        check()
        if _positive_overlap(triangles[first],triangles[second]):
            raise GeometryError('authored partition: cell interiors overlap')
    check()
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)
