"""Independent exact embedding predicates for the operational curved class.

This module uses only verifier-owned geometry/arithmetic. Unresolved separation
is an explicit refusal. In particular there is no assumed arc resultant.
"""
from fractions import Fraction as F
from functools import cmp_to_key
from ._ch3_verify_math import Rational as R,Interval as I,Root,add,sub,mul,pgcd,derivative,evaluate,compare,dot,cross,fail
from ._ch3_verify_geometry import plane,point_interval,middle,isign

def separate(a,b):return any(x.hi<y.lo or y.hi<x.lo for x,y in zip(a,b))
def ivector(p):return tuple(I(x) for x in p)

class Embedding:
    def __init__(self,rec):self.rec=rec;self.ar=rec.ar;self.cache={}
    def edge_box(self,eid):
        rec=self.rec
        if eid in rec.arcs:
            a,b=rec.arc_ends(eid);lo=min(a.lo,b.lo);hi=max(a.hi,b.hi)
            result=tuple(self.ar.range(x,lo,hi) for x in rec.arcs[eid]['curve'])
            if any(x is None for x in result):fail('rational edge box undecided','EMBEDDING_UNDECIDED')
            return result
        e=rec.edges[eid];a,b=rec.vertex(e.start),rec.vertex(e.end)
        return tuple(I(min(x.lo,y.lo),max(x.hi,y.hi)) for x,y in zip(a,b))
    def face_box(self,fid):
        if fid in self.cache:return self.cache[fid]
        rec=self.rec
        if fid in rec.supports:
            s=rec.supports[fid];rows=[]
            for v in s.vr:
                rows.append(tuple(self.ar.range(R(add(a,tuple(v*x for x in b)),s.W),*s.ur) for a,b in zip(s.H0,s.H1)))
            if any(x is None for row in rows for x in row):fail('ruled patch box undecided','EMBEDDING_UNDECIDED')
            box=tuple(I(min(row[i].lo for row in rows),max(row[i].hi for row in rows)) for i in range(3))
        else:
            boxes=[self.edge_box(oe.edge) for loop in (rec.faces[fid].loop,)+tuple(rec.faces[fid].holes) for oe in loop]
            box=tuple(I(min(b[i].lo for b in boxes),max(b[i].hi for b in boxes)) for i in range(3))
        self.cache[fid]=box;return box
    def normal(self,fid):
        p=plane(self.rec.faces[fid].surface);n=cross(p[1],p[2]);return n,dot(n,p[0])
    def vertex_sign(self,vid,n,d):
        rec=self.rec
        if vid in rec.corners:
            root,c=rec.corners[vid];r=dot(c,n)-R(d)
            return rec.root_sign(r.n,root)*rec.root_sign(r.d,root)
        value=dot(rec.positions[vid],n)-d;return (value>0)-(value<0)
    def common(self,a,b):
        rec=self.rec;ea={oe.edge for loop in (rec.faces[a].loop,)+tuple(rec.faces[a].holes) for oe in loop};eb={oe.edge for loop in (rec.faces[b].loop,)+tuple(rec.faces[b].holes) for oe in loop};edges=ea&eb
        va={v for e in ea for v in (rec.edges[e].start,rec.edges[e].end)};vb={v for e in eb for v in (rec.edges[e].start,rec.edges[e].end)}
        return edges,va&vb
    def point_on_segment(self,q,eid):
        rec=self.rec;e=rec.edges[eid];a,b=rec.vertex(e.start),rec.vertex(e.end)
        direction=tuple(y-x for x,y in zip(a,b));axis=max(range(3),key=lambda i:max(abs(direction[i].lo),abs(direction[i].hi)))
        # A segment lies on both incident rational planes. For this predicate
        # q is rational and is chosen off every endpoint/crossing breakpoint.
        for fid,_ in rec.edge_uses[eid]:
            if fid in rec.supports:fail('straight segment has an unsupported ruled incident face','EMBEDDING_UNDECIDED')
            n,d=self.normal(fid)
            if dot(q,n)!=d:return False
        low=min(a[axis].lo,b[axis].lo);high=max(a[axis].hi,b[axis].hi)
        if q[axis]<low or q[axis]>high:return False
        inner_low=min(a[axis].hi,b[axis].hi);inner_high=max(a[axis].lo,b[axis].lo)
        if inner_low<q[axis]<inner_high:return True
        if low==high==q[axis]:return True
        fail('segment endpoint containment undecided','EMBEDDING_UNDECIDED')
    def contains(self,fid,q):
        """Closed planar region membership of a rational point, by own ray counts."""
        rec=self.rec;face=rec.faces[fid];n,d=self.normal(fid)
        if dot(n,q)!=d:return False
        drop=max(range(3),key=lambda i:abs(n[i]));xaxis,yaxis=[i for i in range(3) if i!=drop]
        edges=[oe.edge for loop in (face.loop,)+tuple(face.holes) for oe in loop];vertices={v for eid in edges for v in (rec.edges[eid].start,rec.edges[eid].end)}
        # First detect boundary membership exactly, independently of the ray.
        for eid in edges:
            self.ar.tick('verify_planar_boundary')
            if eid not in rec.arcs:
                if self.point_on_segment(q,eid):return True
                continue
            c=rec.arcs[eid]['curve'];g=()
            for coordinate,value in zip(c,q):g=pgcd(g,(coordinate-R(value)).n)
            if len(g)>1:
                a,b=rec.arc_ends(eid);lo,hi=(a,b) if compare(a,b)<0 else (b,a)
                if any(compare(r,lo)>=0 and compare(r,hi)<=0 for r in self.ar.roots(g,lo.lo,hi.hi)):return True
        # A finite set of slopes avoids every vertex. This handles rays
        # through topological corners without a floating perturbation.
        slope=None
        for k in range(len(vertices)+2):
            yq=q[yaxis]+k*q[xaxis]
            projected={v:rec.vertex(v)[yaxis]+k*rec.vertex(v)[xaxis]-I(yq) for v in vertices}
            if all(z.lo>0 or z.hi<0 for z in projected.values()):slope=k;break
        if slope is None:fail('no certified off-vertex planar ray','EMBEDDING_UNDECIDED')
        parity=0;yq=q[yaxis]+slope*q[xaxis]
        for eid in edges:
            if eid in rec.arcs:
                c=rec.arcs[eid]['curve'];yr=c[yaxis]+R(slope)*c[xaxis]-R(yq);a,b=rec.arc_ends(eid);lo,hi=(a,b) if compare(a,b)<0 else (b,a)
                for root in self.ar.roots(yr.n,lo.lo,hi.hi):
                    if compare(root,lo)<=0 or compare(root,hi)>=0:continue
                    multiplicity=0;poly=yr.n
                    while rec.vanishes(poly,root):multiplicity+=1;poly=derivative(poly)
                    if multiplicity%2==0:continue
                    xr=c[xaxis]-R(q[xaxis]);side=rec.root_sign(xr.n,root)*rec.root_sign(xr.d,root)
                    if not side:return True
                    if side>0:parity+=1
            else:
                e=rec.edges[eid];a,b=rec.vertex(e.start),rec.vertex(e.end);ya=a[yaxis]+slope*a[xaxis]-I(yq);yb=b[yaxis]+slope*b[xaxis]-I(yq)
                if isign(ya)*isign(yb)<0:
                    xx=a[xaxis]+(b[xaxis]-a[xaxis])*ya/(ya-yb)-I(q[xaxis]);side=isign(xx)
                    if not side:return True
                    if side>0:parity+=1
        return bool(parity%2)
    def line_point(self,n,d,m,e,axis,t):
        # Cramer's rule for two plane equations and x_axis=t.
        unit=tuple(F(int(i==axis)) for i in range(3));det=dot(n,cross(m,unit))
        if not det:fail('plane intersection coordinate is singular','EMBEDDING_UNDECIDED')
        a=cross(m,unit);b=cross(unit,n);c=cross(n,m)
        return tuple((d*x+e*y+t*z)/det for x,y,z in zip(a,b,c))
    def boundary_points(self,fid,n,d,axis):
        """All intersections of a face's boundary with a rational plane."""
        rec=self.rec;points={}
        def vertex(vid):points[('v',vid)]=(rec.vertex(vid),{fid})
        for loop in (rec.faces[fid].loop,)+tuple(rec.faces[fid].holes):
            for oe in loop:
                eid=oe.edge;edge=rec.edges[eid]
                if eid in rec.arcs:
                    c=rec.arcs[eid]['curve'];r=dot(c,n)-R(d);a,b=rec.arc_ends(eid);lo,hi=(a,b) if compare(a,b)<0 else (b,a)
                    if not r.n:vertex(edge.start);vertex(edge.end);continue
                    for root in self.ar.roots(r.n,lo.lo,hi.hi):
                        if compare(root,lo)<0 or compare(root,hi)>0:continue
                        if compare(root,a)==0:vertex(edge.start)
                        elif compare(root,b)==0:vertex(edge.end)
                        else:points[('a',eid,tuple(root.identity()['poly']),root.identity()['ordinal'])]=(point_interval(c,root,self.ar),{fid})
                else:
                    sa=self.vertex_sign(edge.start,n,d);sb=self.vertex_sign(edge.end,n,d)
                    if not sa:vertex(edge.start)
                    if not sb:vertex(edge.end)
                    if sa*sb<0:
                        a,b=rec.vertex(edge.start),rec.vertex(edge.end);ra=dot(a,n)-I(d);rb=dot(b,n)-I(d);t=ra/(ra-rb);p=tuple(x+(y-x)*t for x,y in zip(a,b));points[('s',eid)]=(p,{fid})
        return points
    def planar_pair(self,a,b):
        rec=self.rec;na,da=self.normal(a);nb,db=self.normal(b);line=cross(na,nb);shared,shared_vertices=self.common(a,b)
        if not any(line):
            if dot(na,plane(rec.faces[b].surface)[0])!=da:return
            if separate(self.face_box(a),self.face_box(b)):return
            fail('overlapping coplanar faces require a separate exact cell proof','EMBEDDING_UNDECIDED')
        axis=max(range(3),key=lambda i:abs(line[i]));points=self.boundary_points(a,nb,db,axis)
        for key,(point,owners) in self.boundary_points(b,na,da,axis).items():
            if key in points:points[key][1].update(owners)
            else:points[key]=(point,owners)
        ordered=sorted(points.items(),key=lambda item:(item[1][0][axis].lo+item[1][0][axis].hi)/2)
        # A non-declared coincident boundary point is itself an intersection.
        for left,right in zip(ordered,ordered[1:]):
            x,y=left[1][0][axis],right[1][0][axis]
            if x.hi>=y.lo:fail('undeclared or unresolved coincident plane-line boundary points','EMBEDDING_UNDECIDED')
        gaps=[]
        for left,right in zip(ordered,ordered[1:]):
            x,y=left[1][0][axis],right[1][0][axis];t=(x.hi+y.lo)/2;q=self.line_point(na,da,nb,db,axis,t)
            on_shared=False
            for eid in shared:
                box=self.edge_box(eid)
                if box[axis].lo<t<box[axis].hi:on_shared=True;break
            if on_shared:gaps.append((True,True,True));continue
            ia=self.contains(a,q);ib=self.contains(b,q);gaps.append((ia,ib,False))
            if ia and ib:fail('planar face interiors intersect outside shared cells','SELF_INTERSECTING')
        for i,(key,(point,owners)) in enumerate(ordered):
            if key[0]=='v' and key[1] in shared_vertices:continue
            coordinate=point[axis]
            if any(self.edge_box(eid)[axis].lo<=coordinate.lo<=coordinate.hi<=self.edge_box(eid)[axis].hi for eid in shared):continue
            near=[]
            if i:near.append(gaps[i-1])
            if i<len(gaps):near.append(gaps[i])
            if (a in owners and any(x[1] for x in near)) or (b in owners and any(x[0] for x in near)):fail('undeclared planar endpoint/interior contact','SELF_INTERSECTING')
    def curved_inside(self,fid,u,v):
        rec=self.rec;parity=0
        for oe in rec.faces[fid].loop:
            a,b=rec.arc_ends(oe.edge);lo,hi=(a,b) if compare(a,b)<0 else (b,a)
            if compare(lo,u)<0 and compare(u,hi)<0:
                r=R(*rec.arcs[oe.edge]['graph']).at(u)
                if r==v:return None
                if r>v:parity+=1
        return bool(parity%2)
    def curved_inside_root(self,fid,root,graph):
        rec=self.rec;parity=0
        for oe in rec.faces[fid].loop:
            a,b=rec.arc_ends(oe.edge);lo,hi=(a,b) if compare(a,b)<0 else (b,a);difference=R(*rec.arcs[oe.edge]['graph'])-R(*graph)
            if compare(lo,root)<=0 and compare(root,hi)<=0 and rec.vanishes(difference.n,root):return None
            # Half-open ranges give the exact vertical-ray convention at a
            # corner's abscissa, with no numerical perturbation.
            if compare(lo,root)<=0 and compare(root,hi)<0:
                if rec.root_sign(difference.n,root)*rec.root_sign(difference.d,root)>0:parity+=1
        return bool(parity%2)
    def curved_planar(self,curved,planar):
        rec=self.rec;s=rec.supports[curved];n,d=self.normal(planar);graph=s.query_graph(n,d);num,den=graph;corners=[rec.corners[rec.edges[oe.edge].start if oe.forward else rec.edges[oe.edge].end][0] for oe in rec.faces[curved].loop];corners.sort(key=cmp_to_key(compare));lo,hi=corners[0],corners[-1];shared,shared_vertices=self.common(curved,planar)
        def inside_roots(p):return [r for r in self.ar.roots(p,lo.lo,hi.hi) if compare(r,lo)>=0 and compare(r,hi)<=0]
        if not den:
            if not num or inside_roots(num):fail('ruling or coplanar curved/planar embedding undecided','EMBEDDING_UNDECIDED')
            return
        if inside_roots(den):fail('curved/planar section pole in extent','EMBEDDING_UNDECIDED')
        curve=s.curve(graph);breaks=list(corners);hits=[]
        for oe in rec.faces[curved].loop:
            g=rec.arcs[oe.edge]['graph'];p=sub(mul(num,g[1]),mul(g[0],den));a,b=rec.arc_ends(oe.edge);a,b=(a,b) if compare(a,b)<0 else (b,a)
            if p:breaks.extend(r for r in inside_roots(p) if compare(r,a)>=0 and compare(r,b)<=0)
        # All planar boundary crossings must also partition the section.
        for loop in (rec.faces[planar].loop,)+tuple(rec.faces[planar].holes):
            for oe in loop:
                eid=oe.edge
                if eid in rec.arcs:
                    arc=rec.arcs[eid]
                    if arc['support'] is s:
                        g=arc['graph'];p=sub(mul(num,g[1]),mul(g[0],den))
                        a,b=rec.arc_ends(eid);a,b=(a,b) if compare(a,b)<0 else (b,a)
                        rr=inside_roots(p) if p else [a,b]
                        breaks.extend(rr);hits.extend((root,eid) for root in rr if compare(root,a)>=0 and compare(root,b)<=0)
                    elif not separate(self.face_box(curved),self.edge_box(eid)):fail('unresolved cross-support section/arc intersection','EMBEDDING_UNDECIDED')
                else:
                    other=next(fid for fid,_ in rec.edge_uses[eid] if fid!=planar)
                    if other in rec.supports:fail('straight edge in ruled boundary','EMBEDDING_UNDECIDED')
                    nn,dd=self.normal(other);p=(dot(curve,nn)-R(dd)).n
                    if p:
                        rr=inside_roots(p);breaks.extend(rr);hits.extend((root,eid) for root in rr)
                    elif not separate(self.face_box(curved),self.edge_box(eid)):fail('section coincides with an undeclared segment line','EMBEDDING_UNDECIDED')
        breaks.sort(key=cmp_to_key(compare));unique=[]
        for root in breaks:
            if not unique or compare(unique[-1],root):unique.append(root)
        planar_members=[]
        for a,b in zip(unique,unique[1:]):
            u=middle(a,b);v=R(*graph).at(u);inside=self.curved_inside(curved,u,v);on_shared=False
            for eid in shared:
                if eid not in rec.arcs:continue
                g=rec.arcs[eid]['graph'];aa,bb=rec.arc_ends(eid);aa,bb=(aa,bb) if compare(aa,bb)<0 else (bb,aa)
                if not sub(mul(num,g[1]),mul(g[0],den)) and compare(aa,u)<0 and compare(u,bb)<0:on_shared=True
            if on_shared:planar_members.append(True);continue
            if inside is None:fail('undeclared coincident curved/planar boundary','SELF_INTERSECTING')
            point=tuple(x.at(u) for x in curve);member=False if separate(ivector(point),self.face_box(planar)) else self.contains(planar,point);planar_members.append(member)
            if inside and member:fail('curved patch intersects planar face away from shared cells','SELF_INTERSECTING')
        # Root-only contacts cannot be erased by gap sampling. Shared corners
        # are exact identities; other unresolved contacts fail closed.
        for index,root in enumerate(unique):
            declared_vertex=False
            for vid in shared_vertices:
                if vid not in rec.corners or compare(root,rec.corners[vid][0]):continue
                corner_curve=rec.corners[vid][1]
                if all(rec.vanishes((x-y).n,root) for x,y in zip(curve,corner_curve)):declared_vertex=True
            if declared_vertex:continue
            on_shared=False
            for eid in shared:
                if eid not in rec.arcs:continue
                g=rec.arcs[eid]['graph'];aa,bb=rec.arc_ends(eid);aa,bb=(aa,bb) if compare(aa,bb)<0 else (bb,aa)
                if not sub(mul(num,g[1]),mul(g[0],den)) and compare(root,aa)>=0 and compare(root,bb)<=0:on_shared=True
            if on_shared:continue
            p=point_interval(curve,root,self.ar)
            if separate(p,self.face_box(planar)):continue
            inside=self.curved_inside_root(curved,root,graph)
            if inside is False:continue
            # A planar boundary tangency is a contact even when neither
            # neighboring gap is inside the planar region.
            for hit,eid in hits:
                if compare(root,hit):continue
                if eid in rec.arcs:fail('undeclared curved/arc boundary contact','SELF_INTERSECTING')
                e=rec.edges[eid];a,b=rec.vertex(e.start),rec.vertex(e.end);axis=max(range(3),key=lambda j:max(abs(b[j].lo-a[j].hi),abs(b[j].hi-a[j].lo)));lower=min(a[axis].lo,b[axis].lo);upper=max(a[axis].hi,b[axis].hi)
                if p[axis].hi<lower or p[axis].lo>upper:continue
                inner_low=min(a[axis].hi,b[axis].hi);inner_high=max(a[axis].lo,b[axis].lo)
                if inner_low<p[axis].lo<=p[axis].hi<inner_high:fail('undeclared curved/segment boundary contact','SELF_INTERSECTING')
                fail('curved/segment endpoint contact undecided','EMBEDDING_UNDECIDED')
            neighbors=[]
            if index:neighbors.append(planar_members[index-1])
            if index<len(planar_members):neighbors.append(planar_members[index])
            if any(neighbors):fail('undeclared curved boundary touches planar interior','SELF_INTERSECTING')
    def run(self):
        rec=self.rec
        for i,a in enumerate(rec.ids):
            for b in rec.ids[i+1:]:
                self.ar.tick('verify_embedding_pair')
                if separate(self.face_box(a),self.face_box(b)):continue
                ac=a in rec.supports;bc=b in rec.supports
                if ac and bc:fail('curved patch boxes overlap','EMBEDDING_UNDECIDED')
                if ac or bc:self.curved_planar(a if ac else b,b if ac else a)
                else:self.planar_pair(a,b)
