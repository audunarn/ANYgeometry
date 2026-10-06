"""Whole affine attachment relations from detached, authenticated owner inputs.

Target face rectangles enclose incidence; they are never interpreted as paths.
Captured-real coordinates exclude floating evaluator roundoff.
"""
from fractions import Fraction as F

from .errors import GeometryError
from .authored_partition_coverage import _strictly_inside


def _q(v):
    return tuple(map(F,v))


def _add(a,b):
    return tuple(x+y for x,y in zip(a,b))


def _sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def _scale(a,s):
    return tuple(x*s for x in a)


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _cross(a,b):
    return tuple(a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1)))


def _lerp(a,b,t):
    return _add(a,_scale(_sub(b,a),t))


def _pack(x):
    return (x.numerator,x.denominator)


def _squared(a,b):
    d=_sub(a,b)
    return _dot(d,d)


def _union(spans):
    result=[]
    for a,b in sorted(spans):
        if b<a:
            raise GeometryError('attachment interval orientation is invalid')
        if result and a<=result[-1][1]:
            result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:
            result.append((a,b))
    return tuple(result)


class _Context:
    def __init__(self,original,current,descendants,edge_maps,use_maps,source_native,current_native,material_faces,proof):
        self.original,self.current=original,current
        self.proof=proof;self.check=proof.charge
        self.documents={False:original,True:current}
        self.vertices={side:{r['id']:_q(r['position']) for r in doc['vertices']} for side,doc in self.documents.items()}
        self.edges={side:{r['id']:r for r in doc['edges']} for side,doc in self.documents.items()}
        self.faces={side:{r['id']:r for r in doc['faces']} for side,doc in self.documents.items()}
        self.uses={side:{r['id']:r for r in doc['structural']['member_edge_uses']} for side,doc in self.documents.items()}
        self.members={side:{r['id']:r for r in doc['structural']['members']} for side,doc in self.documents.items()}
        self.supports={False:source_native,True:current_native}
        self.children=dict(descendants);self.material_faces=set(material_faces)
        self.edge_maps={};self.use_maps={};self.loop_cache={};self.frame_cache={}
        for row in edge_maps:
            self.check()
            if (row['classification'] in ('exact','bounded') and
                row['ancestor_model_id']==original['model_id'] and
                row['ancestor_revision']==original['revision'] and
                row['ancestor_source_checksum']==original['checksum']['value']):
                self.edge_maps.setdefault((row['ancestor_edge_id'],row['edge_id']),[]).append(row)
        for row in use_maps:
            if row['disposition'] in ('exact_native_map','bounded_native_map'):
                native=row['native_map'];source=self.uses[False].get(row['source_use_id'])
                use=self.uses[True].get(row['current_use_id'])
                if (source is not None and use is not None and source['member_id']==use['member_id']
                    and source['edge_id']==native['ancestor_edge_id'] and use['edge_id']==native['edge_id']
                    and native['ancestor_model_id']==original['model_id']
                    and native['ancestor_revision']==original['revision']
                    and native['ancestor_source_checksum']==original['checksum']['value']
                    and native['classification'] in ('exact','bounded')):
                    self.use_maps[row['current_use_id']]=row

    def line(self,side,edge):
        row=self.edges[side][edge]
        if row['curve']['type']!='straight':
            raise GeometryError('attachment coordinate proof requires a native Straight carrier')
        return self.vertices[side][row['start']],self.vertices[side][row['end']]

    def frame(self,side,face):
        key=(side,face)
        if key not in self.frame_cache:
            kind,data=self.supports[side].get(face,('unsupported',()))
            if kind!='plane':
                raise GeometryError('attachment target chart requires captured native Plane support')
            o,u,v=map(_q,data);n=_cross(u,v)
            gram=_dot(u,u)*_dot(v,v)-_dot(u,v)**2
            if gram<=0:
                raise GeometryError('attachment native Plane is singular')
            self.frame_cache[key]=(o,u,v,n,gram)
        return self.frame_cache[key]

    def uv(self,side,face,point):
        o,u,v,n,gram=self.frame(side,face);d=_sub(point,o)
        if _dot(d,n):
            raise GeometryError('attachment axis has no exact Plane incidence')
        du,dv=_dot(d,u),_dot(d,v);uv=_dot(u,v)
        return ((_dot(v,v)*du-uv*dv)/gram,(_dot(u,u)*dv-uv*du)/gram)

    def face_point(self,side,face,parameters):
        o,u,v,_,_=self.frame(side,face)
        return _add(o,_add(_scale(u,F(parameters[0][0])),_scale(v,F(parameters[1][0]))))

    def enclose_axis(self,side,face,parameters,a,b):
        for point in (a,b):
            uv=self.uv(side,face,point)
            if any(not F(interval[0])<=value<=F(interval[1]) for interval,value in zip(parameters,uv)):
                raise GeometryError('attachment target rectangle does not enclose its whole affine axis')

    def loops(self,side,face):
        key=(side,face)
        if key not in self.loop_cache:
            row=self.faces[side][face];loops=[]
            for loop in (row['loop'],*row['holes']):
                points=[];end=None
                for edge,forward in loop:
                    self.check();a,b=self.line(side,edge)
                    if not forward:a,b=b,a
                    if end is not None and a!=end:
                        raise GeometryError('attachment material loop is not closed')
                    points.append(self.uv(side,face,a));end=b
                if not points or self.uv(side,face,end)!=points[0]:
                    raise GeometryError('attachment material loop is not closed')
                loops.append(tuple(points))
            self.loop_cache[key]=tuple(loops)
        return self.loop_cache[key]

    @staticmethod
    def on(a,b,p):
        d=_sub(b,a);e=_sub(p,a)
        return d[0]*e[1]==d[1]*e[0] and all(min(x,y)<=z<=max(x,y) for x,y,z in zip(a,b,p))

    def contains(self,loops,p):
        boundary=any(self.on(a,b,p) for loop in loops for a,b in zip(loop,(*loop[1:],loop[0])))
        return boundary or (_strictly_inside(loops[0],p) and not any(_strictly_inside(h,p) for h in loops[1:]))

    def clip(self,side,face,a,b):
        loops=self.loops(side,face);p,q=self.uv(side,face,a),self.uv(side,face,b)
        if p==q:
            return ((F(0),F(1)),) if self.contains(loops,p) else ()
        d=_sub(q,p);events={F(0),F(1)}
        for loop in loops:
            for c,e in zip(loop,(*loop[1:],loop[0])):
                self.check();w=_sub(e,c);v=_sub(c,p);det=d[0]*w[1]-d[1]*w[0]
                if det:
                    t=(v[0]*w[1]-v[1]*w[0])/det;s=(v[0]*d[1]-v[1]*d[0])/det
                    if 0<=t<=1 and 0<=s<=1:events.add(t)
                elif v[0]*d[1]==v[1]*d[0]:
                    k=0 if d[0] else 1
                    for point in (c,e):
                        t=(point[k]-p[k])/d[k]
                        if 0<=t<=1:events.add(t)
        ordered=sorted(events)
        return _union((x,y) for x,y in zip(ordered,ordered[1:]) if self.contains(loops,_lerp(p,q,(x+y)/2)))

    def member_segments(self,side,member,interval):
        lo,hi=map(F,interval);segments=[]
        for key in self.members[side][member]['edge_use_ids']:
            self.check();use=self.uses[side][key];p,q=map(F,use['parent_range'])
            x,y=max(lo,p),min(hi,q)
            if y<x or (x==y and lo!=hi):continue
            if q<=p:raise GeometryError('attachment member use is singular')
            a,b=self.line(side,use['edge_id'])
            def point(t):
                local=(t-p)/(q-p)
                if use['orientation']!='forward':local=1-local
                return _lerp(a,b,local)
            if side:
                mapping=self.use_maps.get(key)
                if mapping is None:raise GeometryError('attachment member ancestry is unavailable')
                source=self.uses[False][mapping['source_use_id']]
                native=mapping['native_map'];u,v=(F(*z) for z in native['interval'])
                sp,sq=map(F,source['parent_range'])
                def source_parameter(t):
                    local=(t-p)/(q-p)
                    if use['orientation']!='forward':local=1-local
                    local=u+(v-u)*local
                    if source['orientation']!='forward':local=1-local
                    return sp+(sq-sp)*local
                sx,sy=source_parameter(x),source_parameter(y)
                sa,sb=self.line(False,source['edge_id'])
                def source_point(t):
                    local=(t-sp)/(sq-sp)
                    if source['orientation']!='forward':local=1-local
                    return _lerp(sa,sb,local)
                residual=max(_squared(point(x),source_point(sx)),_squared(point(y),source_point(sy)))
            else:
                sx,sy=x,y;residual=F(0)
            segments.append((x,y,sx,sy,point(x),point(y),residual))
        if _union((r[0],r[1]) for r in segments)!=((lo,hi),):
            raise GeometryError('attachment member interval is not completely covered')
        if lo==hi:
            points={r[4] for r in segments}
            if len(points)!=1:raise GeometryError('attachment station has ambiguous member incidence')
        return segments


def map_native_attachment_references(original,current,descendants,edge_maps,use_maps,member_ok,sheet_ok,
                                    source_native,current_native,material_faces,proof,part_ok=None,replacement_history=()):
    """Return complete source-to-current relations; unsupported rows refuse."""
    context=_Context(original,current,descendants,edge_maps,use_maps,source_native,current_native,material_faces,proof)
    check=proof.charge
    before={r['id']:r for r in original['structural']['attachments']}
    after={r['id']:r for r in current['structural']['attachments']}
    grouped={key:[] for key in before};untracked=[]
    replacement_children={(kind,key):tuple(map(tuple,children)) for kind,key,children in replacement_history}
    history_cache={}
    def history(root):
        if root not in history_cache:
            found={root};pending=[root]
            while pending:
                check()
                for child in replacement_children.get(pending.pop(),()):
                    if child not in found:found.add(child);pending.append(child)
            history_cache[root]=found
        return history_cache[root]
    for row in after.values():
        check();ancestry={key for kind,key in row['lineage'] if kind=='attachment' and key in before}
        if row['id'] in before:ancestry.add(row['id'])
        if len(ancestry)==1:grouped[next(iter(ancestry))].append(row)
        elif ancestry:untracked.append(row['id'])
    rows=[];current_sources={}
    for source in before.values():
        check();children=grouped[source['id']];evidence=[]
        try:
            if not children:raise GeometryError('original attachment has no authenticated descendants')
            tau=F(original['tolerance']['length']);tau2=tau*tau
            role=source['source_kind'];target=source['target_kind']
            relation_tau=F(source['tolerance_used']);relation_tau2=relation_tau*relation_tau
            if source['part_id'] is not None and not (part_ok or {}).get(source['part_id'],False):
                raise GeometryError('attachment Part owner semantics are unqualified')
            if source['sheet_id'] is not None and not sheet_ok.get(source['sheet_id'],False):
                raise GeometryError('attachment Sheet owner semantics are unqualified')
            stable={'kind','member_id','source_kind','metadata','connection_intent','evidence','max_residual',
                    'tolerance_used','part_id','sheet_id','provenance'}
            for child in children:
                if any(child[k]!=source[k] for k in stable):
                    raise GeometryError('attachment owner/payload/intent changed')
                if role!='member' and child['member_range']!=source['member_range']:
                    raise GeometryError('attachment non-Member source range changed')
                old=set(map(tuple,source['lineage']));new=set(map(tuple,child['lineage']))
                if not old<=new:raise GeometryError('attachment authored lineage was lost')
                allowed=old|{('attachment',source['id'])}
                for kind,key in new-old:
                    if (kind=='attachment' and key not in before and key in after and
                        ('attachment',source['id']) in set(map(tuple,after[key]['lineage']))):continue
                    if kind in ('face','edge') and ((kind,key) in history((target,source['target_id'])) or
                        (role in ('face','edge') and (kind,key) in history((role,source['source_id'])))):continue
                    if (kind,key) not in allowed:raise GeometryError('attachment has untracked lineage')
            role=source['source_kind'];target=source['target_kind']
            if role=='member' and not member_ok.get(source['member_id'],False):
                raise GeometryError('attachment source Member semantics are unqualified')
            if role=='sheet' and not sheet_ok.get(source['source_id'],False):
                raise GeometryError('attachment source Sheet semantics are unqualified')
            if target=='face' and role in ('member','vertex'):
                root=source['target_id'];faces=context.children.get(root,())
                if not set(faces)<=context.material_faces:
                    raise GeometryError('attachment target material descendants are unqualified')
                if any(child['target_kind']!='face' or child['target_id'] not in faces or child['source_id']!=source['source_id'] for child in children):
                    raise GeometryError('attachment target/source identity changed')
                point=all(F(a)==F(b) for a,b in source['target_parameters'])
                if point:
                    original_point=context.face_point(False,root,source['target_parameters'])
                    if not context.clip(False,root,original_point,original_point):
                        raise GeometryError('original attachment target point is outside material')
                    if role=='member':
                        original_segments=context.member_segments(False,source['member_id'],source['member_range'])
                        if source['member_range'][0]!=source['member_range'][1] or any(_squared(s[4],original_point)>relation_tau2 for s in original_segments):
                            raise GeometryError('original attachment member station is not incident to its face point')
                    elif _squared(context.vertices[False][source['source_id']],original_point)>relation_tau2:
                        raise GeometryError('original attachment vertex is not incident to its face point')
                    incident={face for face in faces if context.clip(True,face,original_point,original_point)}
                    if {c['target_id'] for c in children}!=incident or len(children)!=len(incident):
                        raise GeometryError('attachment point incident owner collection is incomplete or duplicated')
                    for child in children:
                        check()
                        if source['sheet_id'] is not None and not any(u['sheet_id']==source['sheet_id'] and
                            u['face_id']==child['target_id'] for u in current['structural']['face_uses']):
                            raise GeometryError('attachment target lost its explicit Sheet ownership')
                        if any(a!=b for a,b in child['target_parameters']):raise GeometryError('attachment point became a rectangle')
                        bound=_squared(original_point,context.face_point(True,child['target_id'],child['target_parameters']))
                        current_target=context.face_point(True,child['target_id'],child['target_parameters'])
                        if role=='vertex':
                            if context.vertices[False][source['source_id']]!=context.vertices[True][source['source_id']]:
                                raise GeometryError('attachment source vertex changed')
                            relation_bound=_squared(context.vertices[True][source['source_id']],current_target)
                        else:
                            if source['member_range'][0]!=source['member_range'][1] or child['member_range'][0]!=child['member_range'][1]:
                                raise GeometryError('attachment target point lacks a unique source station')
                            segments=context.member_segments(True,source['member_id'],child['member_range'])
                            if any(s[2]!=F(source['member_range'][0]) or s[3]!=s[2] for s in segments):
                                raise GeometryError('attachment original member station changed')
                            bound=max(bound,*(s[6] for s in segments))
                            relation_bound=max(_squared(s[4],current_target) for s in segments)
                        if relation_bound>relation_tau2:raise GeometryError('current attachment source-target incidence exceeds recorded tolerance')
                        if bound>tau2:raise GeometryError('attachment composed physical point residual exceeds length tolerance')
                        evidence.append(dict(current_attachment_id=child['id'],target_face_id=child['target_id'],
                            source_interval=tuple(map(lambda x:_pack(F(x)),source['member_range'])),squared_world_residual_bound=_pack(bound),
                            squared_relation_residual_bound=_pack(relation_bound)))
                else:
                    if role!='member':raise GeometryError('attachment axis source is unsupported')
                    original_segments=context.member_segments(False,source['member_id'],source['member_range'])
                    expected={face:[] for face in faces}
                    for segment in original_segments:
                        x,y,_,_,a,b,_=segment
                        if context.clip(False,root,a,b)!=((F(0),F(1)),):
                            raise GeometryError('original attachment source axis is not wholly material')
                        context.enclose_axis(False,root,source['target_parameters'],a,b)
                        for face in faces:
                            for lo,hi in context.clip(True,face,a,b):expected[face].append((x+(y-x)*lo,x+(y-x)*hi))
                    actual={face:[] for face in faces}
                    singleton_owners=set()
                    for child in children:
                        check();segments=context.member_segments(True,source['member_id'],child['member_range'])
                        singleton=child['member_range'][0]==child['member_range'][1]
                        if singleton and source['member_range'][0]!=source['member_range'][1]:
                            raise GeometryError('non-point attachment has an extra singleton owner incidence')
                        if singleton and child['target_id'] in singleton_owners:
                            raise GeometryError('attachment has duplicate singleton face-owner incidence')
                        if singleton:singleton_owners.add(child['target_id'])
                        if source['sheet_id'] is not None and not any(u['sheet_id']==source['sheet_id'] and
                            u['face_id']==child['target_id'] for u in current['structural']['face_uses']):
                            raise GeometryError('attachment target lost its explicit Sheet ownership')
                        bounds=[]
                        for x,y,sx,sy,a,b,residual in segments:
                            if sy<sx:raise GeometryError('attachment source interval traversal changed')
                            actual[child['target_id']].append((sx,sy))
                            if context.clip(True,child['target_id'],a,b)!=((F(0),F(1)),):
                                raise GeometryError('current attachment axis is not wholly material')
                            context.enclose_axis(True,child['target_id'],child['target_parameters'],a,b)
                            if residual>tau2:raise GeometryError('attachment composed affine residual exceeds length tolerance')
                            bounds.append(residual)
                        evidence.append(dict(current_attachment_id=child['id'],target_face_id=child['target_id'],
                            source_intervals=[tuple(map(_pack,(s[2],s[3]))) for s in segments],
                            squared_world_residual_bound=_pack(max(bounds,default=F(0)))))
                    if any(_union(actual[f])!=_union(expected[f]) for f in faces):
                        raise GeometryError('attachment material interval incidence is incomplete or overlaps wrong owners')
                    for spans in actual.values():
                        ordered=sorted(spans)
                        if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
                            raise GeometryError('attachment has duplicate overlapping owner incidence')
            elif target=='edge' and role in ('member','vertex','sheet'):
                expected=tuple(map(F,source['target_parameters'][0]));spans=[]
                source_a,source_b=context.line(False,source['target_id'])
                if expected[0]==expected[1]:
                    candidates=set()
                    for (ancestor,edge),mappings in context.edge_maps.items():
                        if ancestor!=source['target_id'] or len(mappings)!=1:continue
                        a,b=sorted(F(*x) for x in mappings[0]['interval'])
                        if a<=expected[0]<=b:candidates.add(('edge',edge))
                    if not candidates:raise GeometryError('point edge attachment has no authenticated intended owner')
                    pending=[('edge',source['target_id'])];visited=set();intended=None
                    while pending:
                        check();node=pending.pop()
                        if node in visited:continue
                        visited.add(node)
                        replacements=replacement_children.get(node,())
                        if replacements:
                            pending.extend(reversed(replacements))
                        elif node in candidates:
                            intended=node[1];break
                    if intended is None:raise GeometryError('point edge intended ownership lacks replacement-history authentication')
                    if len(children)!=1 or children[0]['target_id']!=intended:
                        raise GeometryError('point edge attachment violates deterministic first-child ownership')
                if role=='member':
                    original_paths=context.member_segments(False,source['member_id'],source['member_range'])
                    p0,p1=map(F,source['member_range'])
                    first=min(original_paths,key=lambda s:s[0])[4];last=max(original_paths,key=lambda s:s[1])[5]
                    endpoints=(_lerp(source_a,source_b,expected[0]),_lerp(source_a,source_b,expected[1]))
                    forward=max(_squared(first,endpoints[0]),_squared(last,endpoints[1]))
                    reverse=max(_squared(first,endpoints[1]),_squared(last,endpoints[0]))
                    direction=1 if forward<=reverse else -1
                    def target_parameter(p):
                        t=F(0) if p0==p1 else (p-p0)/(p1-p0)
                        if direction<0:t=1-t
                        return expected[0]+(expected[1]-expected[0])*t
                    if (p0==p1)!=(expected[0]==expected[1]):
                        raise GeometryError('attachment source and target incidence dimensions differ')
                    for segment in original_paths:
                        if max(_squared(segment[4],_lerp(source_a,source_b,target_parameter(segment[0]))),
                               _squared(segment[5],_lerp(source_a,source_b,target_parameter(segment[1]))))>relation_tau2:
                            raise GeometryError('original member-edge attachment has no whole affine incidence')
                elif role=='vertex':
                    if expected[0]!=expected[1] or _squared(context.vertices[False][source['source_id']],_lerp(source_a,source_b,expected[0]))>relation_tau2:
                        raise GeometryError('original vertex-edge attachment is not incident')
                else:
                    original_uses=[u for u in original['structural']['face_uses'] if u['sheet_id']==source['source_id']]
                    if not any(source['target_id']==edge for use in original_uses for face in (context.faces[False][use['face_id']],)
                        for loop in (face['loop'],*face['holes']) for edge,_ in loop):
                        raise GeometryError('original sheet-edge attachment lacks actual edge incidence')
                singleton_owners=set()
                for child in children:
                    check()
                    if child['target_kind']!='edge' or child['source_id']!=source['source_id']:
                        raise GeometryError('attachment edge/source identity changed')
                    maps=context.edge_maps.get((source['target_id'],child['target_id']),())
                    if len(maps)!=1:raise GeometryError('attachment target edge ancestry is ambiguous/unqualified')
                    mapping=maps[0];a,b=(F(*x) for x in mapping['interval']);u,v=map(F,child['target_parameters'][0])
                    if u==v:
                        if expected[0]!=expected[1]:
                            raise GeometryError('non-point edge attachment has an extra singleton owner incidence')
                        owner=(child['target_id'],u)
                        if owner in singleton_owners:
                            raise GeometryError('attachment has duplicate singleton edge-owner incidence')
                        singleton_owners.add(owner)
                    sx,sy=a+(b-a)*u,a+(b-a)*v;spans.append(tuple(sorted((sx,sy))))
                    ca,cb=context.line(True,child['target_id'])
                    bound=max(_squared(_lerp(source_a,source_b,sx),_lerp(ca,cb,u)),
                              _squared(_lerp(source_a,source_b,sy),_lerp(ca,cb,v)))
                    if role=='member':
                        segments=context.member_segments(True,source['member_id'],child['member_range'])
                        if any(s[6]>tau2 for s in segments):raise GeometryError('attachment source member world residual exceeds tolerance')
                        projected=_union(tuple(sorted((target_parameter(s[2]),target_parameter(s[3])))) for s in segments)
                        if projected!=(tuple(sorted((sx,sy))),):
                            raise GeometryError('attachment member and target ancestral intervals disagree')
                        q0,q1=map(F,child['member_range'])
                        relation_bound=F(0)
                        for segment in segments:
                            for q,point in ((segment[0],segment[4]),(segment[1],segment[5])):
                                t=F(0) if q0==q1 else (q-q0)/(q1-q0)
                                if (direction<0)!=(b<a):t=1-t
                                relation_bound=max(relation_bound,_squared(point,_lerp(ca,cb,u+(v-u)*t)))
                        if relation_bound>relation_tau2:raise GeometryError('current member-edge whole affine incidence exceeds recorded tolerance')
                        bound=max(bound,*(s[6] for s in segments))
                    elif role=='vertex':
                        if u!=v or sx!=expected[0] or sy!=expected[1] or context.vertices[False][source['source_id']]!=context.vertices[True][source['source_id']]:
                            raise GeometryError('attachment vertex incidence changed')
                        if _squared(context.vertices[True][source['source_id']],_lerp(ca,cb,u))>relation_tau2:
                            raise GeometryError('current vertex-edge attachment is not incident')
                    else:
                        current_uses=[use for use in current['structural']['face_uses'] if use['sheet_id']==child['source_id']]
                        if not any(child['target_id']==edge for use in current_uses for face in (context.faces[True][use['face_id']],)
                            for loop in (face['loop'],*face['holes']) for edge,_ in loop):
                            raise GeometryError('current sheet-edge attachment lacks actual edge incidence')
                    if bound>tau2:raise GeometryError('attachment edge composed physical residual exceeds length tolerance')
                    evidence.append(dict(current_attachment_id=child['id'],target_edge_id=child['target_id'],
                        source_interval=tuple(map(_pack,sorted((sx,sy)))),squared_world_residual_bound=_pack(bound)))
                if _union(spans)!=(expected,):raise GeometryError('attachment target edge interval coverage is incomplete')
                ordered=sorted(spans)
                if expected[0]!=expected[1] and any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
                    raise GeometryError('attachment target edge coverage overlaps')
            elif target in ('vertex','member') and role=='member':
                if any(child['target_kind']!=target or child['target_id']!=source['target_id'] or
                       child['source_id']!=source['source_id'] for child in children) or len(children)!=1:
                    raise GeometryError('attachment incoming entity relation ownership changed')
                child=children[0]
                if source['member_range'][0]!=source['member_range'][1] or child['member_range'][0]!=child['member_range'][1]:
                    raise GeometryError('incoming entity attachment currently requires a point station')
                original_source=context.member_segments(False,source['member_id'],source['member_range'])
                current_source=context.member_segments(True,source['member_id'],child['member_range'])
                if any(s[2]!=F(source['member_range'][0]) or s[3]!=s[2] for s in current_source):
                    raise GeometryError('attachment incoming source station map changed')
                if target=='vertex':
                    if source['target_parameters'] or child['target_parameters']:
                        raise GeometryError('vertex attachment has unexpected native parameters')
                    before_point=context.vertices[False][source['target_id']];after_point=context.vertices[True][source['target_id']]
                    target_bound=_squared(before_point,after_point)
                else:
                    if not member_ok.get(source['target_id'],False):raise GeometryError('attachment target Member semantics are unqualified')
                    if (source['target_parameters'][0][0]!=source['target_parameters'][0][1] or
                        child['target_parameters'][0][0]!=child['target_parameters'][0][1]):
                        raise GeometryError('incoming Member target requires a point station')
                    before_target=context.member_segments(False,source['target_id'],source['target_parameters'][0])
                    after_target=context.member_segments(True,source['target_id'],child['target_parameters'][0])
                    if any(s[2]!=F(source['target_parameters'][0][0]) or s[3]!=s[2] for s in after_target):
                        raise GeometryError('attachment incoming target station map changed')
                    before_point=before_target[0][4];after_point=after_target[0][4]
                    target_bound=max(s[6] for s in after_target)
                if any(_squared(s[4],before_point)>relation_tau2 for s in original_source) or any(
                        _squared(s[4],after_point)>relation_tau2 for s in current_source):
                    raise GeometryError('attachment incoming source-target station is not incident')
                bound=max(target_bound,*(s[6] for s in current_source))
                if bound>tau2:raise GeometryError('attachment incoming composed world residual exceeds tolerance')
                evidence.append(dict(current_attachment_id=child['id'],target_kind=target,target_id=child['target_id'],
                    squared_world_residual_bound=_pack(bound)))
            else:
                raise GeometryError('attachment role/target family has no whole coordinate proof')
            for child in children:current_sources[child['id']]=source['id']
            rows.append(dict(source_attachment_id=source['id'],current_attachment_ids=[c['id'] for c in children],
                classification='qualified_captured_real_relation',source_model_id=original['model_id'],
                source_revision=original['revision'],source_checksum=original['checksum']['value'],maps=evidence,
                floating_evaluation_preservation_qualified=False))
        except GeometryError as error:
            if error is proof.callback_error:raise
            rows.append(dict(source_attachment_id=source['id'],current_attachment_ids=[c['id'] for c in children],
                             classification='refused',refusal=str(error)))
    return rows,current_sources,untracked,context


def map_native_junction_references(original,current,attachment_maps,member_ok,sheet_ok,context):
    before=original['structural']['junctions'];after={r['id']:r for r in current['structural']['junctions']}
    mappings={r['source_attachment_id']:r for r in attachment_maps};result=[]
    for source in before:
        context.check();target=after.get(source['id'])
        try:
            if target is None:raise GeometryError('original Junction was deleted')
            stable=set(source)-{'attachment_ids','member_uses'}
            if any(source[k]!=target[k] for k in stable):raise GeometryError('Junction owner/payload/intent changed')
            expected=[]
            for key in source['attachment_ids']:
                row=mappings.get(key)
                if row is None or row['classification']=='refused':raise GeometryError('Junction has an unqualified original attachment')
                expected.extend(row['current_attachment_ids'])
            if len(target['attachment_ids'])!=len(set(target['attachment_ids'])) or set(target['attachment_ids'])!=set(expected):
                raise GeometryError('Junction attachment expansion is incomplete or has untracked extras')
            if any(not sheet_ok.get(key,False) for key in source['sheet_ids']):
                raise GeometryError('Junction Sheet owner semantics are unqualified')
            if len(source['member_uses'])!=len(target['member_uses']):raise GeometryError('Junction Member incidence inventory changed')
            evidence=[]
            for old,new in zip(source['member_uses'],target['member_uses']):
                if old['member_id']!=new['member_id'] or not member_ok.get(old['member_id'],False):
                    raise GeometryError('Junction Member owner changed or is unqualified')
                original_segments=context.member_segments(False,old['member_id'],old['member_range'])
                segments=context.member_segments(True,new['member_id'],new['member_range'])
                projected=_union(tuple(sorted((s[2],s[3]))) for s in segments)
                if projected!=(tuple(map(F,old['member_range'])),):raise GeometryError('Junction original Member interval changed')
                bound=max((s[6] for s in segments),default=F(0))
                if bound>F(original['tolerance']['length'])**2:raise GeometryError('Junction composed Member residual exceeds length tolerance')
                evidence.append(dict(member_id=old['member_id'],source_interval=tuple(map(lambda x:_pack(F(x)),old['member_range'])),
                    current_interval=tuple(map(lambda x:_pack(F(x)),new['member_range'])),squared_world_residual_bound=_pack(bound)))
            result.append(dict(source_junction_id=source['id'],current_junction_id=target['id'],
                               classification='qualified_captured_real_relation',member_maps=evidence))
        except GeometryError as error:
            if error is context.proof.callback_error:raise
            result.append(dict(source_junction_id=source['id'],classification='refused',refusal=str(error)))
    return result
