"""Reconstruction of ruled nets and oriented integration chains from live geometry."""
from fractions import Fraction as F
from functools import cmp_to_key
from math import comb,isqrt,lcm,gcd
from ._errors import CertifiedHydrostaticsRefusal,RefusalCategory
from ._ch3_verify_math import (Arithmetic,Rational as R,Interval as I,Root,add,sub,mul,scale,power,trim,pgcd,primitive,derivative,evaluate,compare,dot,cross,vsub,sign,fail,bounds)

def pair(x):x=F(x);return [x.numerator,x.denominator]
def pp(p):return [pair(x) for x in p]
def vec(v):return tuple(F(float(x)) if not isinstance(x,(int,F)) else F(x) for x in v)
def vp(v):return [pair(x) for x in v]
def plane(p):return tuple(vec(getattr(p,n)) for n in ('origin','u_vector','v_vector'))
def canonical_graph(g):
    n,d=g
    common=pgcd(n,d)
    from ._ch3_verify_math import divide
    n=divide(n,common)[0];d=divide(d,common)[0]
    den=lcm(*(x.denominator for x in n+d));ints=[int(x*den) for x in n+d];factor=gcd(*ints)
    if d[-1]<0:factor=-factor
    return scale(n,F(den,factor)),scale(d,F(den,factor))
def polynomial_sum(ps):
    a=()
    for p in ps:a=add(a,p)
    return a
def nullspace(rows,width):
    matrix=[list(map(F,row)) for row in rows];pivots=[];rank=0
    for col in range(width):
        pivot=next((i for i in range(rank,len(matrix)) if matrix[i][col]),None)
        if pivot is None:continue
        matrix[rank],matrix[pivot]=matrix[pivot],matrix[rank];scale_by=matrix[rank][col];matrix[rank]=[x/scale_by for x in matrix[rank]]
        for i in range(len(matrix)):
            if i!=rank and matrix[i][col]:q=matrix[i][col];matrix[i]=[x-q*y for x,y in zip(matrix[i],matrix[rank])]
        pivots.append(col);rank+=1
    result=[]
    for free in (i for i in range(width) if i not in pivots):
        v=[F(0)]*width;v[free]=1
        for row,pivot in enumerate(pivots):v[pivot]=-matrix[row][free]
        result.append(tuple(v))
    return result
def point_interval(c,u,ar):
    if isinstance(u,Root):
        u.refine(getattr(ar,"point_bits",140))
        return tuple(ar.range(x,u.lo,u.hi) for x in c)
    return tuple(I(x.at(u)) for x in c)
def hint_binds(c,root,position,ar,delta):
    """Prove the exact point is within delta; uncertainty adds no slack."""
    for _ in range(ar.policy['max_isolation_bits']):
        image=point_interval(c,root,ar)
        if all(x is not None for x in image):
            if all(x.width<delta/4 and max(abs(v-x.lo),abs(v-x.hi))<=delta for x,v in zip(image,position)):return True
            if any(v<x.lo-delta or v>x.hi+delta for x,v in zip(image,position)):return False
        root.bisect()
    fail('vertex hint precision exhausted','BUDGET_EXHAUSTED')

def middle(a,b):
    if compare(a,b)>0:a,b=b,a
    for _ in range(4096):
        ah=a.hi if isinstance(a,Root) else a;bl=b.lo if isinstance(b,Root) else b
        if ah<bl:return (ah+bl)/2
        if isinstance(a,Root):a.bisect()
        if isinstance(b,Root):b.bisect()
    fail('unable to find rational test point','BUDGET_EXHAUSTED')
def isign(x):
    if x.lo>0:return 1
    if x.hi<0:return -1
    if x.lo==x.hi==0:return 0
    fail('unresolved exact predicate','UNDECIDED_PREDICATE')

class Support:
    def __init__(self,s):
        self.degree=int(s.degree_u);self.su=(F(s.knots_u[self.degree]),F(s.knots_u[-self.degree-1]));self.sv=(F(s.knots_v[1]),F(s.knots_v[-2]));self.ur=vec(s.u_range);self.vr=vec(s.v_range)
        if s.degree_v!=1 or len(s.controls)!=self.degree+1:fail('unsupported ruled net','UNSUPPORTED_SURFACE')
        if self.degree not in (1,2) or tuple(map(F,s.knots_u))!=(self.su[0],)*(self.degree+1)+(self.su[1],)*(self.degree+1) or tuple(map(F,s.knots_v))!=(self.sv[0],)*2+(self.sv[1],)*2:fail('unsupported multiple-span ruled net','UNSUPPORTED_SURFACE')
        if not self.su[0]<=self.ur[0]<self.ur[1]<=self.su[1] or not self.sv[0]<=self.vr[0]<self.vr[1]<=self.sv[1]:fail('invalid ruled range','UNSUPPORTED_SURFACE')
        u0,u1=self.su;t=( -u0/(u1-u0),F(1)/(u1-u0));one=sub((F(1),),t)
        nets=[]
        for j in range(2):
            net=[]
            for k in range(4):
                terms=[]
                for i in range(self.degree+1):
                    w=F(s.weights[i][j]) if s.weights is not None else F(1);c=F(s.controls[i][j][k]) if k<3 else F(1)
                    terms.append(scale(mul(power(t,i),power(one,self.degree-i)),w*c*comb(self.degree,i)))
                net.append(polynomial_sum(terms))
            nets.append(tuple(net))
        self.A,self.B=nets
        if s.weights is not None and any(F(w)<=0 for row in s.weights for w in row):fail('nonpositive ruled weights','UNSUPPORTED_SURFACE')
        if self.A[3]!=self.B[3]:fail('weights vary along ruling','UNSUPPORTED_SURFACE')
        self.W=self.A[3];k0,k1=self.sv
        self.H1=tuple(scale(sub(b,a),1/(k1-k0)) for a,b in zip(self.A[:3],self.B[:3]))
        self.H0=tuple(sub(a,scale(b,k0)) for a,b in zip(self.A[:3],self.H1))
        rows=[]
        for j in range(self.degree+1):
            w=self.W[j] if j<len(self.W) else 0
            rows.append([(p[j] if j<len(p) else 0) for p in self.A[:3]]+[-w,0])
            delta=[sub(b,a) for a,b in zip(self.A[:3],self.B[:3])];rows.append([(p[j] if j<len(p) else 0) for p in delta]+[0,-w])
        charts=[v for v in nullspace(rows,5) if v[4]]
        if not charts:fail('ruled support has no injective chart projection','UNSUPPORTED_SURFACE')
        self.chart=charts[0]
        # C4: control polygon remains a non-collinear triple (or nonzero
        # segment) for every ruling coordinate, certified by Bernstein signs.
        controls=[]
        for row in s.controls:
            controls.append(tuple((F(row[0][i])-k0*(F(row[1][i])-F(row[0][i]))/(k1-k0),(F(row[1][i])-F(row[0][i]))/(k1-k0)) for i in range(3)))
        first=tuple(sub(b,a) for a,b in zip(controls[0],controls[1]))
        if self.degree==1:components=first
        else:
            second=tuple(sub(b,a) for a,b in zip(controls[0],controls[2]));components=(sub(mul(first[1],second[2]),mul(first[2],second[1])),sub(mul(first[2],second[0]),mul(first[0],second[2])),sub(mul(first[0],second[1]),mul(first[1],second[0])))
        if not any(lo>0 or hi<0 for lo,hi in (bounds(p,*self.sv) for p in components)):fail('ruled control rows do not certify chart injectivity','UNSUPPORTED_SURFACE')
    def graph(self,p):
        o,e,f=p;n=cross(e,f);d=dot(n,o)
        alpha=sub(polynomial_sum(scale(x,c) for x,c in zip(self.H0,n)),scale(self.W,d));beta=polynomial_sum(scale(x,c) for x,c in zip(self.H1,n))
        if not beta:fail('vertical trim graph','CURVED_SECTION_DEGENERACY')
        return canonical_graph((scale(alpha,-1),beta))
    def query_graph(self,n,d):
        alpha=sub(polynomial_sum(scale(x,c) for x,c in zip(self.H0,n)),scale(self.W,d));beta=polynomial_sum(scale(x,c) for x,c in zip(self.H1,n));return scale(alpha,-1),beta
    def curve(self,g):
        n,d=g;return tuple(R(add(mul(a,d),mul(b,n)),mul(self.W,d)) for a,b in zip(self.H0,self.H1))
    def canonical(self,sigma):return {'kind':'ruled','degree_u':self.degree,'net':[[pp(x) for x in net] for net in (self.A,self.B)],'span':[[pair(x) for x in span] for span in (self.su,self.sv)],'u_range':[pair(x) for x in self.ur],'v_range':[pair(x) for x in self.vr],'sigma_chart':sigma}

class Reconstruction:
    def __init__(self,model,face_ids,ar):
        self.model=model;self.ids=tuple(sorted(face_ids));self.ar=ar;self.faces={i:model.faces[i] for i in self.ids};self.edges={};self.positions={};self.supports={};self.arcs={};self.corners={};self.sigma={};self.edge_uses={};self.incidence={}
        for fid,f in self.faces.items():
            for loop in (f.loop,)+tuple(f.holes):
                if len(loop)<3 or len({oe.edge for oe in loop})!=len(loop):fail('invalid face loop','INVALID_LOOP')
                starts=[];ends=[]
                for oe in loop:
                    edge=model.edges[oe.edge];self.edges[edge.id]=edge;self.edge_uses.setdefault(edge.id,[]).append([fid,bool(oe.forward)])
                    starts.append(edge.start if oe.forward else edge.end);ends.append(edge.end if oe.forward else edge.start)
                    for v in (edge.start,edge.end):
                        self.positions[v]=vec(model.vertices[v].position);fs,es=self.incidence.setdefault(v,(set(),set()));fs.add(fid);es.add(edge.id)
                if ends!=starts[1:]+starts[:1] or len(set(starts))!=len(starts):fail('face loop is disconnected or repeats a corner','INVALID_LOOP')
            if hasattr(f.surface,'degree_u'):self.supports[fid]=Support(f.surface)
        for uses in self.edge_uses.values():
            if len(uses)!=2 or uses[0][0]==uses[1][0]:fail('shell does not use each edge twice','NOT_CLOSED')
            if uses[0][1]==uses[1][1]:fail('shell edge orientations disagree','INVERTED_ORIENTATION')
        for vid,(incident_faces,incident_edges) in self.incidence.items():
            adjacency={fid:set() for fid in incident_faces}
            for eid in incident_edges:
                a,b=(use[0] for use in self.edge_uses[eid]);adjacency[a].add(b);adjacency[b].add(a)
            reached=set();pending=[next(iter(incident_faces))]
            while pending:
                fid=pending.pop()
                if fid in reached:continue
                reached.add(fid);pending.extend(adjacency[fid]-reached)
            if reached!=incident_faces:fail('vertex link has disconnected shell fans','NON_MANIFOLD')
        # Each supported curved loop is expressed solely by plane/ruled graphs.
        for fid,s in self.supports.items():
            f=self.faces[fid]
            if f.holes:fail('curved holes are not admitted','CURVED_SECTION_DEGENERACY')
            for oe in f.loop:
                e=self.edges[oe.edge];c=e.curve
                if not hasattr(c,'plane_record'):fail('curved trim is not a graph section','UNSUPPORTED_TRIM')
                stored=Support(c.surface)
                if (stored.A,stored.B,stored.su,stored.sv,stored.ur,stored.vr)!=(s.A,s.B,s.su,s.sv,s.ur,s.vr):fail('edge support differs from face support','UNSUPPORTED_TRIM')
                p=tuple(vec(x) for x in c.plane_record);g=s.graph(p)
                self.arcs[e.id]={'support':s,'plane':p,'graph':g,'curve':s.curve(g),'domain':tuple(c.domain),'path':c.path}
            for previous,current in zip(f.loop[-1:]+f.loop[:-1],f.loop):
                e=self.edges[current.edge];vid=e.start if current.forward else e.end
                a=self.arcs[previous.edge]['graph'];b=self.arcs[current.edge]['graph'];p=sub(mul(a[0],b[1]),mul(b[0],a[1]));roots=ar.roots(p,*s.ur)
                matches=[]
                for root in roots:
                    root.refine(100);c=self.arcs[current.edge]['curve'];pi=point_interval(c,root,ar)
                    delta=F(1,2**ar.policy['delta_hint_bits'])*(1+max(map(abs,self.positions[vid])))
                    if hint_binds(c,root,self.positions[vid],ar,delta):matches.append(root)
                if len(matches)!=1:fail('corner does not bind to live vertex','VERIFICATION_FAILED','snapshot')
                root=matches[0]
                if vid in self.corners and compare(self.corners[vid][0],root):fail('inconsistent corner root','VERIFICATION_FAILED','snapshot')
                self.corners[vid]=(root,self.arcs[current.edge]['curve'])
            area=I(0)
            for oe in f.loop:
                a,b=self.arc_ends(oe.edge,oe.forward);g=self.arcs[oe.edge]['graph'];area-=ar.integrate(R(*g),a,b,F(1,10**18))
            self.sigma[fid]=isign(area)
            if not self.sigma[fid]:fail('zero signed chart loop area','INVALID_LOOP')
            self.simple_curved_loop(fid)
        for eid,arc in self.arcs.items():
            self.check_trim(eid)
        for eid,e in self.edges.items():
            if eid not in self.arcs and type(e.curve).__name__!='Straight':fail('unsupported nonsection edge','UNSUPPORTED_TRIM')
        for fid in self.ids:
            if fid not in self.supports:self.simple_planar_loop(fid)
        self.snapshot=self.canonical()
    def vanishes(self,p,root):
        if not isinstance(root,Root):return evaluate(p,root)==0
        if root.lo==root.hi:return evaluate(p,root.lo)==0
        g=pgcd(p,root.p)
        return len(g)>1 and bool(self.ar.roots(g,root.lo,root.hi))
    def root_sign(self,p,root):
        if self.vanishes(p,root):return 0
        if not isinstance(root,Root):return sign(evaluate(p,root))
        for _ in range(self.ar.policy['max_isolation_bits']):
            lo,hi=bounds(p,root.lo,root.hi)
            if lo>0:return 1
            if hi<0:return -1
            root.bisect()
        fail('single root sign refinement exhausted','BUDGET_EXHAUSTED')
    def check_trim(self,eid):
        arc=self.arcs[eid];s=arc['support'];a,b=self.arc_ends(eid);lo,hi=(a,b) if compare(a,b)<0 else (b,a);n,d=arc['graph'];g=R(n,d)
        if compare(lo,s.ur[0])<=0 or compare(hi,s.ur[1])>=0:fail('corner on or outside chart boundary','UNSUPPORTED_TRIM')
        extrema=sub(mul(derivative(n),d),mul(n,derivative(d)));points=[a,b]+[r for r in self.ar.roots(extrema,lo.lo,hi.hi) if compare(r,lo)>0 and compare(r,hi)<0]
        for root in points:
            ds=self.root_sign(d,root)
            if not ds or self.root_sign(sub(n,scale(d,s.vr[0])),root)*ds<=0 or self.root_sign(sub(scale(d,s.vr[1]),n),root)*ds<=0:fail('trim leaves ruled range box','UNSUPPORTED_TRIM')
        # H2 is a narrowly authorized black-box library hint check. The domain
        # fields parameterize the library path; they are not knot-u endpoints.
        import numpy as np
        self.ar.tick('verify_curve_hint',3)
        try:us,vs=self.edges[eid].curve._knot_points(np.array([0.,.5,1.]))
        except Exception as exc:raise CertifiedHydrostaticsRefusal(RefusalCategory.TRIM_HINT_MISMATCH,'stored curve knot hints cannot be evaluated',phase='verify') from exc
        us=vec(us);vs=vec(vs);delta=F(1,2**self.ar.policy['delta_chart_bits'])
        for index,root in ((0,a),(2,b)):
            root.refine(self.ar.policy['delta_chart_bits']+8);vi=self.ar.range(g,root.lo,root.hi)
            if max(abs(us[index]-root.lo),abs(us[index]-root.hi),abs(vs[index]-vi.lo),abs(vs[index]-vi.hi))>delta:fail('stored curve endpoint/direction differs from exact corner','TRIM_HINT_MISMATCH')
        if compare(lo,us[1])>=0 or compare(us[1],hi)>=0 or abs(vs[1]-g.at(us[1]))>delta:fail('stored curve midpoint differs from exact graph','TRIM_HINT_MISMATCH')
    def simple_curved_loop(self,fid):
        loop=self.faces[fid].loop;count=len(loop)
        for i,oe in enumerate(loop):
            a,b=self.arc_ends(oe.edge);lo,hi=(a,b) if compare(a,b)<0 else (b,a);g=self.arcs[oe.edge]['graph']
            if compare(lo,hi)==0:fail('zero length graph trim','INVALID_LOOP')
            if self.ar.roots(g[1],lo.lo if isinstance(lo,Root) else lo,hi.hi if isinstance(hi,Root) else hi):fail('trim graph contains a pole','INVALID_LOOP')
            for j in range(i+1,count):
                other=loop[j];c,d=self.arc_ends(other.edge);low,high=(c,d) if compare(c,d)<0 else (d,c);h=self.arcs[other.edge]['graph'];p=sub(mul(g[0],h[1]),mul(h[0],g[1]));adjacent=(j==i+1 or i==0 and j==count-1)
                left=lo if compare(lo,low)>0 else low;right=hi if compare(hi,high)<0 else high
                if compare(left,right)<0:
                    if not p:fail('coincident graph pieces','INVALID_LOOP')
                    aa=left.lo if isinstance(left,Root) else left;bb=right.hi if isinstance(right,Root) else right
                    for root in self.ar.roots(p,aa,bb):
                        if compare(root,left)>0 and compare(root,right)<0:fail('graph pieces cross','INVALID_LOOP')
                # Test BOTH endpoint/interior directions. Only the actual
                # shared topological vertex may supply an adjacent contact.
                e,f=self.edges[oe.edge],self.edges[other.edge]
                for endpoints,target,tl,th,source_ids,target_ids in (
                    ((a,b),(c,d),low,high,(e.start,e.end),(f.start,f.end)),
                    ((c,d),(a,b),lo,hi,(f.start,f.end),(e.start,e.end))):
                    for endpoint,vid in zip(endpoints,source_ids):
                        if compare(endpoint,tl)>=0 and compare(endpoint,th)<=0 and self.vanishes(p,endpoint):
                            shared=adjacent and any(vid==other_vid and compare(endpoint,t)==0 for t,other_vid in zip(target,target_ids))
                            if not shared:fail('endpoint contacts nonincident graph','INVALID_LOOP')
    def simple_planar_loop(self,fid):
        f=self.faces[fid];p=plane(f.surface);normal=cross(p[1],p[2]);offset=dot(normal,p[0]);drop=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=drop]
        if not any(normal):fail('degenerate planar support','INVALID_LOOP')
        def box(eid):
            if eid in self.arcs:
                a,b=self.arc_ends(eid);lo=min(a.lo,b.lo);hi=max(a.hi,b.hi);return tuple(self.ar.range(x,lo,hi) for x in self.arcs[eid]['curve'])
            e=self.edges[eid];a,b=self.vertex(e.start),self.vertex(e.end);return tuple(I(min(x.lo,y.lo),max(x.hi,y.hi)) for x,y in zip(a,b))
        def separated(a,b):return any(x.hi<y.lo or y.hi<x.lo for x,y in zip(a,b))
        for oe in f.loop:
            eid=oe.edge;e=self.edges[eid]
            if eid in self.arcs:
                if (dot(self.arcs[eid]['curve'],normal)-R(offset)).n:fail('section edge leaves planar face','INVALID_LOOP')
            for vid in (e.start,e.end):
                if vid in self.corners:
                    root,c=self.corners[vid]
                    if not self.vanishes((dot(c,normal)-R(offset)).n,root):fail('corner leaves planar face','INVALID_LOOP')
                elif dot(self.positions[vid],normal)!=offset:fail('vertex leaves planar face','INVALID_LOOP')
        for i,oe in enumerate(f.loop):
            eid=oe.edge;e=self.edges[eid]
            for other in f.loop[i+1:]:
                oid=other.edge;o=self.edges[oid];shared={e.start,e.end}&{o.start,o.end};ea=eid in self.arcs;oa=oid in self.arcs
                if not shared and separated(box(eid),box(oid)):continue
                if ea and oa:
                    # The admitted operational domain excludes unresolved arc
                    # intersections. Exact rational boxes may be subdivided;
                    # uncertainty is a refusal, never geometric sampling.
                    if shared:fail('adjacent independent arcs require an intersection proof','EMBEDDING_UNDECIDED')
                    a,b=self.arc_ends(eid);c,d=self.arc_ends(oid);queue=[(min(a.lo,b.lo),max(a.hi,b.hi),min(c.lo,d.lo),max(c.hi,d.hi))];attempt=0
                    while queue:
                        self.ar.tick('verify_arc_separation');attempt+=1
                        if attempt>self.ar.policy['max_pieces_per_integral']:fail('arc separation undecided','EMBEDDING_UNDECIDED')
                        al,ah,bl,bh=queue.pop();ba=tuple(self.ar.range(x,al,ah) for x in self.arcs[eid]['curve']);bb=tuple(self.ar.range(x,bl,bh) for x in self.arcs[oid]['curve'])
                        if None not in ba and None not in bb and separated(ba,bb):continue
                        if ah-al>=bh-bl:mid=(al+ah)/2;queue.extend(((al,mid,bl,bh),(mid,ah,bl,bh)))
                        else:mid=(bl+bh)/2;queue.extend(((al,ah,bl,mid),(al,ah,mid,bh)))
                elif ea or oa:
                    arcid,segid=(eid,oid) if ea else (oid,eid);segment=self.edges[segid];uses=self.edge_uses[segid];otherface=next(x[0] for x in uses if x[0]!=fid)
                    if otherface in self.supports:fail('straight edge incident to unsupported ruled trim','EMBEDDING_UNDECIDED')
                    support=plane(self.faces[otherface].surface);n=cross(support[1],support[2]);poly=(dot(self.arcs[arcid]['curve'],n)-R(dot(n,support[0]))).n
                    if not poly:fail('arc coincides with segment line','INVALID_LOOP')
                    a,b=self.arc_ends(arcid);lo,hi=(a,b) if compare(a,b)<0 else (b,a);roots=self.ar.roots(poly,lo.lo,hi.hi);sb=box(segid)
                    for root in roots:
                        if compare(root,lo)<0 or compare(root,hi)>0:continue
                        if any(compare(root,self.corners[v][0])==0 for v in shared if v in self.corners):continue
                        candidate=point_interval(self.arcs[arcid]['curve'],root,self.ar)
                        if not separated(candidate,sb):fail('nonincident segment/arc contact or unresolved endpoint','INVALID_LOOP')
                elif not shared:
                    a,b=self.vertex(e.start),self.vertex(e.end);c,d=self.vertex(o.start),self.vertex(o.end)
                    def orient(x,y,z):return (y[axes[0]]-x[axes[0]])*(z[axes[1]]-x[axes[1]])-(y[axes[1]]-x[axes[1]])*(z[axes[0]]-x[axes[0]])
                    signs=[isign(orient(a,b,c)),isign(orient(a,b,d)),isign(orient(c,d,a)),isign(orient(c,d,b))]
                    if signs[0]*signs[1]<=0 and signs[2]*signs[3]<=0:fail('planar segments cross or touch','INVALID_LOOP')
                else:
                    common=next(iter(shared));a=self.vertex(common);b=self.vertex(e.end if e.start==common else e.start);c=self.vertex(o.end if o.start==common else o.start)
                    turn=(b[axes[0]]-a[axes[0]])*(c[axes[1]]-a[axes[1]])-(b[axes[1]]-a[axes[1]])*(c[axes[0]]-a[axes[0]])
                    if turn.lo==turn.hi==0 and isign(dot(tuple(x-y for x,y in zip(b,a)),tuple(x-y for x,y in zip(c,a))))>=0:fail('adjacent segments overlap','INVALID_LOOP')
    def arc_ends(self,eid,forward=True):
        e=self.edges[eid];a=self.corners[e.start][0];b=self.corners[e.end][0];return (a,b) if forward else (b,a)
    def vertex(self,vid):
        if vid in self.corners:r,c=self.corners[vid];return point_interval(c,r,self.ar)
        return tuple(I(x) for x in self.positions[vid])
    def canonical(self):
        faces=[]
        for fid,f in self.faces.items():
            loops=[]
            for loop in (f.loop,)+tuple(f.holes):
                loops.append([[oe.edge,bool(oe.forward),self.edges[oe.edge].start if oe.forward else self.edges[oe.edge].end] for oe in loop])
            sup=self.supports[fid].canonical(self.sigma[fid]) if fid in self.supports else [vp(x) for x in plane(f.surface)]
            faces.append({'id':fid,'support':sup,'loops':loops})
        vertices=[]
        for vid,p in sorted(self.positions.items()):
            corner=None
            if vid in self.corners:corner={'corner':dict(self.corners[vid][0].identity(),hint=[float(x).hex() for x in self.model.vertices[vid].position])}
            vertices.append([vid,vp(p),corner])
        edges=[]
        for eid,e in sorted(self.edges.items()):
            section=None
            if eid in self.arcs:
                a=self.arcs[eid];section={'kind':'section','plane':[vp(x) for x in a['plane']],'domain_hint':[float(x).hex() for x in a['domain']],'path_hint':a['path'],'graph':[pp(x) for x in a['graph']]}
            edges.append([eid,e.start,e.end,self.edge_uses[eid],section])
        return {'snapshot_version':2,'faces':faces,'vertices':vertices,'edges':edges,'vertex_incidence':[[v,sorted(fs),sorted(es)] for v,(fs,es) in sorted(self.incidence.items())]}

def bpadd(a,b):
    n=max(len(a),len(b));return [ (a[i] if i<len(a) else R(0))+(b[i] if i<len(b) else R(0)) for i in range(n)]
def bpmul(a,b):
    r=[R(0) for _ in range(len(a)+len(b)-1)]
    for i,x in enumerate(a):
        for j,y in enumerate(b):r[i+j]=r[i+j]+x*y
    return r
def bpneg(a):return [-x for x in a]
def bpcross(a,b):return [bpadd(bpmul(a[1],b[2]),bpneg(bpmul(a[2],b[1]))),bpadd(bpmul(a[2],b[0]),bpneg(bpmul(a[0],b[2]))),bpadd(bpmul(a[0],b[1]),bpneg(bpmul(a[1],b[0])))]
def bpdot(a,b):
    r=[]
    for x,y in zip(a,b):r=bpadd(r,bpmul(x,y))
    return r
def antiderivative_at(p,g):
    r=R(0)
    for i,c in enumerate(p):r+=c*g**(i+1)/R(i+1)
    return r

class Computation:
    def __init__(self,rec,query,tolerance):
        self.rec=rec;self.ar=rec.ar;self.q=query;self.n=tuple(query.normal);self.d=query.offset;self.targets=({name:F(tolerance) for name in ('V','Jx','Jy','Jz','A','Mu','Mv','Iuu','Ivv','Iuv')} if not isinstance(tolerance,dict) else tolerance);ns=dot(self.n,self.n);k=max(160,(ns.denominator.bit_length()-ns.numerator.bit_length())//2+100);slo=F(isqrt((ns.numerator<<(2*k))//ns.denominator),2**k);divisor=max(32,len(rec.edges)*8);self.allowances={name:value*(slo if name in ('A','Mu','Mv','Iuu','Ivv','Iuv') else 1)/divisor for name,value in self.targets.items()};self.tol=min(self.allowances.values());self.values={x:I(0) for x in ('V','Jx','Jy','Jz','A','Mu','Mv','Iuu','Ivv','Iuv')};self.cap=[];self.crossings={};self.any_inside=False;self.any_outside=False
        self.o=tuple(query.waterplane_origin);self.axes=tuple(query.waterplane_axes);self.qcap=tuple(o+(self.d-dot(self.n,self.o))*n/dot(self.n,self.n) for o,n in zip(self.o,self.n))
        self.wanted=None
    def integrate(self,name,r,a,b):
        if self.wanted is None or name in self.wanted:self.accumulate(name,self.ar.integrate(r,a,b,self.allowances[name]))
    def accumulate(self,name,value):
        total=self.values[name]+value
        if total.width:
            # Dyadic outward rounding prevents denominator products from
            # growing with the number of independently enclosed terms.
            allowance=self.allowances[name]/64;k=max(160,getattr(self.ar,'point_bits',140))
            while F(1,2**k)>allowance:k+=32
            if k>self.ar.policy['max_bits']:fail('accumulation precision exhausted','BUDGET_EXHAUSTED')
            grid=2**k;total=I(F((total.lo*grid).__floor__(),grid),F(-(-total.hi*grid).__floor__(),grid))
        self.values[name]=total
    def plane_values(self,origin,c,a,b,cap=False):
        cp=tuple(x.derivative() for x in c);triple=dot(origin,cross(c,cp))
        self.integrate('V',triple/R(6),a,b)
        for name,x,o in zip(('Jx','Jy','Jz'),c,origin):
            if self.wanted is not None and name not in self.wanted:continue
            self.integrate(name,triple*(R(o)+R(2)*x)/R(24),a,b)
        if cap:
            z=tuple(x-R(q) for x,q in zip(c,self.qcap));da=dot(cross(z,cp),self.n)/R(2);uv=[dot(e,tuple(x-R(o) for x,o in zip(c,self.o))) for e in self.axes];quv=[dot(e,vsub(self.qcap,self.o)) for e in self.axes]
            fs=[R(1),(R(quv[0])+R(2)*uv[0])/R(3),(R(quv[1])+R(2)*uv[1])/R(3),(R(quv[0]**2)+R(2*quv[0])*uv[0]+R(3)*uv[0]**2)/R(6),(R(quv[1]**2)+R(2*quv[1])*uv[1]+R(3)*uv[1]**2)/R(6),(R(quv[0]*quv[1])+R(quv[0])*uv[1]+R(quv[1])*uv[0]+R(3)*uv[0]*uv[1])/R(6)]
            for name,f in zip(('A','Mu','Mv','Iuu','Ivv','Iuv'),fs):self.integrate(name,da*f,a,b)
    def segment_values(self,origin,a,b,cap=False):
        det=dot(tuple(I(x) for x in origin),cross(a,b));self.accumulate('V',det/F(6))
        for name,x,y,o in zip(('Jx','Jy','Jz'),a,b,origin):
            if self.wanted is None or name in self.wanted:self.accumulate(name,det*(I(o)+x+y)/F(24))
        if cap:
            da=dot(cross(tuple(x-I(q) for x,q in zip(a,self.qcap)),tuple(y-I(q) for y,q in zip(b,self.qcap))),self.n)/2
            au=[dot(e,tuple(x-I(o) for x,o in zip(a,self.o))) for e in self.axes];bu=[dot(e,tuple(x-I(o) for x,o in zip(b,self.o))) for e in self.axes];q=[dot(e,vsub(self.qcap,self.o)) for e in self.axes]
            factors=[I(1),(I(q[0])+au[0]+bu[0])/3,(I(q[1])+au[1]+bu[1])/3,(I(q[0]**2)+au[0]*au[0]+bu[0]*bu[0]+q[0]*(au[0]+bu[0])+au[0]*bu[0])/6,(I(q[1]**2)+au[1]*au[1]+bu[1]*bu[1]+q[1]*(au[1]+bu[1])+au[1]*bu[1])/6,(2*I(q[0]*q[1])+2*au[0]*au[1]+2*bu[0]*bu[1]+q[0]*(au[1]+bu[1])+q[1]*(au[0]+bu[0])+au[0]*bu[1]+au[1]*bu[0])/12]
            for name,f in zip(('A','Mu','Mv','Iuu','Ivv','Iuv'),factors):self.accumulate(name,da*f)
    def cut_arc(self,eid):
        a,b=self.rec.arc_ends(eid);lo,hi=(a,b) if compare(a,b)<0 else (b,a);c=self.rec.arcs[eid]['curve'];g=dot(c,self.n)-R(self.d);roots=self.ar.roots(g.n,self.rec.arcs[eid]['support'].ur[0],self.rec.arcs[eid]['support'].ur[1]);rr=[r for r in roots if compare(r,lo)>0 and compare(r,hi)<0]
        for i,r in enumerate(rr):
            if len(pgcd(g.n,derivative(g.n)))>1:
                if self.ar.roots(pgcd(g.n,derivative(g.n)),r.lo,r.hi):fail('tangent section','CURVED_SECTION_DEGENERACY')
            self.crossings[(eid,i)]=point_interval(c,r,self.ar)
        return rr
    def run(self):
        rec=self.rec
        for eid,arc in rec.arcs.items():
            if not (dot(arc['curve'],self.n)-R(self.d)).n:fail('edge lies in waterplane','CURVED_SECTION_DEGENERACY')
        for vid in rec.positions:
            if vid in rec.corners:
                root,c=rec.corners[vid]
                if rec.vanishes((dot(c,self.n)-R(self.d)).n,root):fail('plane through a vertex','CURVED_SECTION_DEGENERACY')
            p=rec.vertex(vid);s=isign(dot(p,self.n)-I(self.d))
            if not s:fail('plane through a vertex','CURVED_SECTION_DEGENERACY')
            self.any_inside|=s<0;self.any_outside|=s>0
        cuts={eid:self.cut_arc(eid) for eid in rec.arcs}
        if any(cuts.values()):self.any_inside=self.any_outside=True
        for fid,f in rec.faces.items():
            if fid in rec.supports:self.curved(fid,cuts)
            else:self.planar(fid,cuts)
        for piece in self.cap:
            if piece[0]=='arc':_,c,a,b,*_=piece;self.plane_values(self.qcap,c,a,b,True)
            else:_,a,b,*_=piece;self.segment_values(self.qcap,a,b,True)
        ns=dot(self.n,self.n);self.exact_values={name:(v.lo,None) for name,v in self.values.items() if v.width==0 and name in ('V','Jx','Jy','Jz')}
        for name in ('A','Mu','Mv','Iuu','Ivv','Iuv'):
            if self.values[name].width==0:self.exact_values[name]=(self.values[name].lo,ns)
        sn,sd=isqrt(ns.numerator),isqrt(ns.denominator)
        if sn*sn==ns.numerator and sd*sd==ns.denominator:sqrt=I(F(sn,sd))
        else:
            k=max(160,(ns.denominator.bit_length()-ns.numerator.bit_length())//2+100)
            while True:
                self.ar.tick('verify_sqrt');floor=F(isqrt((ns.numerator<<(2*k))//ns.denominator),2**k);sqrt=I(floor,floor+F(1,2**k))
                if floor and all(max(abs(self.values[name].lo),abs(self.values[name].hi))*sqrt.width/(floor*floor)<=self.tol for name in ('A','Mu','Mv','Iuu','Ivv','Iuv')):break
                k+=32
                if k>self.ar.policy['max_bits']:fail('square root refinement exhausted','BUDGET_EXHAUSTED')
        for name in ('A','Mu','Mv','Iuu','Ivv','Iuv'):self.values[name]=self.values[name]/sqrt
        occ='PARTIAL' if self.any_inside and self.any_outside else ('FULL' if self.any_inside else 'EMPTY')
        topology=self.topology() if self.cap else None
        return self.values,occ,topology
    def curved(self,fid,cuts):
        rec=self.rec;s=rec.supports[fid];f=rec.faces[fid];water=s.query_graph(self.n,self.d);wn,wd=water
        corners=[rec.corners[rec.edges[oe.edge].start if oe.forward else rec.edges[oe.edge].end][0] for oe in f.loop];corners.sort(key=cmp_to_key(compare));umin,umax=corners[0],corners[-1]
        def roots_in_extent(poly):return [root for root in self.ar.roots(poly,umin.lo,umax.hi) if compare(root,umin)>=0 and compare(root,umax)<=0]
        if not wd:
            if roots_in_extent(wn):fail('ruling in waterplane','CURVED_SECTION_DEGENERACY')
        elif roots_in_extent(wd):fail('water graph denominator zero','CURVED_SECTION_DEGENERACY')
        if not hasattr(s,'volume_terms'):
            P=[[R(a),R(b)] for a,b in zip(s.H0,s.H1)];Pu=[[x.derivative() for x in p] for p in P];Pv=[[p[1]] for p in P];triple=bpdot(P,bpcross(Pu,Pv));vol=[x/R(power(s.W,3)) for x in triple];mom=[[x/R(power(s.W,4)) for x in bpmul(triple,p)] for p in P];s.volume_terms=(vol,mom);s.boundary_primitives={}
        vol,mom=s.volume_terms
        def integrate_graph(g,a,b):
            key=tuple(tuple(p) for p in g)
            if key not in s.boundary_primitives:s.boundary_primitives[key]={}
            primitives=s.boundary_primitives[key]
            for name,terms,divisor in [('V',vol,3),*zip(('Jx','Jy','Jz'),mom,(4,4,4))]:
                if self.wanted is not None and name not in self.wanted:continue
                if name not in primitives:primitives[name]=-antiderivative_at(terms,R(*g))/R(divisor)
                self.integrate(name,primitives[name],a,b)
        graph_points=[]
        for oe in f.loop:
            eid=oe.edge;a,b=rec.arc_ends(eid,oe.forward);ordered=sorted(cuts[eid],key=cmp_to_key(compare),reverse=compare(a,b)>0);ends=[a]+ordered+[b];g=rec.arcs[eid]['graph'];c=rec.arcs[eid]['curve']
            for x,y in zip(ends,ends[1:]):
                t=middle(x,y)
                if dot(tuple(v.at(t) for v in c),self.n)<=self.d:integrate_graph(g,x,y)
            for ordinal,r in enumerate(cuts[eid]):graph_points.append((r,(eid,ordinal)))
        graph_points.sort(key=cmp_to_key(lambda a,b:compare(a[0],b[0])))
        for (a,aid),(b,bid) in zip(graph_points,graph_points[1:]):
            if compare(a,b)==0:continue
            t=middle(a,b);v=R(*water).at(t);parity=0
            for oe in f.loop:
                x,y=rec.arc_ends(oe.edge);lo,hi=(x,y) if compare(x,y)<0 else (y,x)
                if compare(lo,t)<=0 and compare(t,hi)<0 and R(*rec.arcs[oe.edge]['graph']).at(t)>v:parity+=1
            if parity%2:
                direction=-sign(evaluate(wd,t))*rec.sigma[fid];x,y=(a,b) if direction>0 else (b,a);ia,ib=(aid,bid) if direction>0 else (bid,aid);integrate_graph(water,x,y);self.cap.append(('arc',s.curve(water),y,x,ib,ia))
    def planar(self,fid,cuts):
        rec=self.rec;f=rec.faces[fid];origin=plane(f.surface)[0];crossings=[];getters={}
        if f.holes:fail('planar holes with curved shells unsupported','CURVED_SECTION_DEGENERACY')
        for oe in f.loop:
            eid=oe.edge;e=rec.edges[eid]
            if eid in rec.arcs:
                a,b=rec.arc_ends(eid,oe.forward);c=rec.arcs[eid]['curve'];rr=cuts[eid];ordered=sorted(rr,key=cmp_to_key(compare),reverse=compare(a,b)>0);ends=[a]+ordered+[b]
                for x,y in zip(ends,ends[1:]):
                    if dot(tuple(v.at(middle(x,y)) for v in c),self.n)<self.d:self.plane_values(origin,c,x,y)
                for j,r in enumerate(rr):
                    position=ordered.index(r);previous=a if position==0 else ordered[position-1]
                    before=dot(tuple(v.at(middle(previous,r)) for v in c),self.n)<self.d
                    crossings.append((self.crossings[(eid,j)],(eid,j),before));getters[(eid,j)]=lambda c=c,r=r:point_interval(c,r,self.ar)
            else:
                av,bv=(e.start,e.end) if oe.forward else (e.end,e.start);a,b=rec.vertex(av),rec.vertex(bv);sa=dot(a,self.n)-I(self.d);sb=dot(b,self.n)-I(self.d);ia=isign(sa)<0;ib=isign(sb)<0
                if ia and ib:self.segment_values(origin,a,b)
                elif ia!=ib:
                    t=sa/(sa-sb);p=tuple(x+(y-x)*t for x,y in zip(a,b));identity=(eid,0);self.crossings[identity]=p;crossings.append((p,identity,ia))
                    self.segment_values(origin,a,p) if ia else self.segment_values(origin,p,b)
                    def getter(av=av,bv=bv):
                        a,b=rec.vertex(av),rec.vertex(bv);sa=dot(a,self.n)-I(self.d);sb=dot(b,self.n)-I(self.d);t=sa/(sa-sb);return tuple(x+(y-x)*t for x,y in zip(a,b))
                    getters[identity]=getter
        if crossings:
            # Intersection with a plane is a line: parity pairs sorted crossings.
            nface=cross(*plane(f.surface)[1:]);line=cross(nface,self.n);axis=max(range(3),key=lambda i:abs(line[i]));crossings=certified_crossing_order(crossings,axis,self.ar,getters)
            for point,identity,_ in crossings:self.crossings[identity]=point
            if len(crossings)%2:fail('odd planar crossing count','CURVED_SECTION_DEGENERACY')
            for x,y in zip(crossings[::2],crossings[1::2]):
                a,aid,exit=x;b,bid,_=y
                if not exit:a,b,aid,bid=b,a,bid,aid
                self.segment_values(origin,a,b);self.cap.append(('segment',b,a,bid,aid))
    def topology(self):
        adjacency={};pieces={}
        for i,p in enumerate(self.cap):
            a,b=p[-2:]
            if a in adjacency:fail('ambiguous section connectivity','CURVED_SECTION_DEGENERACY')
            adjacency[a]=b;pieces[a]=p
        loops=[];unused=set(adjacency)
        while unused:
            first=min(unused);chain=[];key=first
            while key in unused:unused.remove(key);chain.append(key);key=adjacency[key]
            if key!=first:fail('open cap chain','CURVED_SECTION_DEGENERACY')
            identity=sorted(x[0] for x in chain);area=I(0)
            for endpoint in chain:
                piece=pieces[endpoint]
                if piece[0]=='arc':
                    _,c,a,b,*_=piece;cp=tuple(x.derivative() for x in c);integrand=dot(cross(tuple(x-R(q) for x,q in zip(c,self.qcap)),cp),self.n)/R(2);area+=self.ar.integrate(integrand,a,b,self.tol)
                else:
                    _,a,b,*_=piece;area+=dot(cross(tuple(x-I(q) for x,q in zip(a,self.qcap)),tuple(x-I(q) for x,q in zip(b,self.qcap))),self.n)/2
            orientation=isign(area)
            if orientation<=0:fail('hole nesting requires independent containment proof','EMBEDDING_UNDECIDED')
            loops.append({'identity':identity,'role':'outer'})
        loops.sort(key=lambda x:x['identity']);return {'loops':loops,'components':[{'outer':x['identity'],'holes':[]} for x in loops],'cut_points':[list(x) for x in sorted(self.crossings)]}


def certified_crossing_order(crossings,axis,ar,getters):
    """Sort by certified disjoint coordinate intervals, refining before use."""
    bits=getattr(ar,'point_bits',140)
    while True:
        crossings=[(getters[identity](),identity,exit) for _,identity,exit in crossings]
        crossings.sort(key=lambda x:(x[0][axis].lo+x[0][axis].hi)/2)
        if all(a[0][axis].hi<b[0][axis].lo for a,b in zip(crossings,crossings[1:])):return crossings
        if any(a[0][axis].lo==a[0][axis].hi==b[0][axis].lo==b[0][axis].hi for a,b in zip(crossings,crossings[1:])):fail('coincident planar crossings','CURVED_SECTION_DEGENERACY')
        bits*=2
        if bits>ar.policy['max_isolation_bits']:fail('planar crossing order refinement exhausted','BUDGET_EXHAUSTED')
        ar.tick('verify_crossing_order');ar.point_bits=bits
