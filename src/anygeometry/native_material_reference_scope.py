"""Positive document-material census and complete geometry-owned native references.

Document Arc trims are exact circumcircle spans through their stored points.
Captured native functions/evaluator errors are distinct obligations. No external
consumer-reference, floating-evaluation or mesh permission is supplied.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cmp_to_key
import json
from weakref import WeakKeyDictionary, ref

from .authored_constraint_scope import _complete_native_reference_inventory
from .authored_partition_coverage import _segments_meet, _strictly_inside
from .cylinder_charts import _Proof, _Refusal
from .definition_binding import definition_checksum
from .edge_subcurve_preimages import (query_prepared_edge_subcurve_preimages,
    _arc_circle, _arc_positions, _unpack)
from .errors import GeometryError
from .native_arc_parameter_maps import NativeArcParameterMapPolicy, _digest, _frame, _entry
from .native_support_snapshots import capture_native_supports
from .polynomial_extrusion_support import (
    POLYNOMIAL_EXTRUSION_KINDS, prove_polynomial_extrusion_support,
)
from .prepared_model_scope import query_prepared_model_scope, validate_prepared_model_scope_binding


@dataclass(frozen=True, slots=True)
class NativeMaterialScopeRow:
    authored_face_id: int
    current_face_ids: tuple
    family: str
    classification: str
    document_material_qualified: bool
    physical_support_qualified: bool
    domain_evidence_json: str
    refusal: str | None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PreparedNativeMaterialReferenceScope:
    scope: object
    selected_root_ids: tuple
    outside_root_ids: tuple
    policy: NativeArcParameterMapPolicy
    native_support_digest: str
    edge_ancestry_digest: str
    material_rows: tuple
    native_arc_maps: tuple
    inventory_json: str
    geometry_native_reference_maps_qualified: bool
    document_material_qualified: bool
    opaque_semantics_qualified: bool = False
    external_reference_scope_qualified: bool = False
    floating_evaluation_preservation_qualified: bool = False
    meshing_permitted: bool = False
    work_counts: tuple = ()

    @property
    def inventory(self):
        return json.loads(self.inventory_json)


_issued = WeakKeyDictionary()


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _pack(x):
    x=F(x)
    return x.numerator,x.denominator


def _qvector(row):
    return tuple(F(x) for x in row)


def _sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _cross(a,b):
    return tuple(a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1)))


def _compile_document(document):
    uses={}
    for row in document['structural']['face_uses']:
        uses.setdefault(row['face_id'],[]).append(row)
    return dict(vertices={r['id']:_qvector(r['position']) for r in document['vertices']},
        edges={r['id']:r for r in document['edges']},faces={r['id']:r for r in document['faces']},
        face_uses=document['structural']['face_uses'],face_uses_by_face=uses)


def _replacement_snapshot(model):
    return tuple(sorted((old.kind,old.id,tuple((child.kind,child.id) for child in children))
                        for old,children in model.replacement_history().items()))


def _segment_candidates(segments,check):
    """Deterministic bounding-box sweep; duplicate geometry is indexed once."""
    boxes=sorted((min(a[0],b[0]),max(a[0],b[0]),min(a[1],b[1]),max(a[1],b[1]),i)
                 for i,(a,b) in enumerate(segments))
    active=[]
    for lo,hi,bottom,top,index in boxes:
        check()
        active=[row for row in active if row[0]>=lo]
        for _,y0,y1,previous in active:
            if min(top,y1)>=max(bottom,y0):
                check()
                yield previous,index
        active.append((hi,bottom,top,index))


def _plane_frame(support):
    origin,u,v=map(_qvector,support)
    uu,uv,vv=_dot(u,u),_dot(u,v),_dot(v,v)
    det=uu*vv-uv*uv
    if det<=0:
        raise GeometryError('singular native plane')
    return origin,u,v,_cross(u,v),(
        tuple((vv*x-uv*y)/det for x,y in zip(u,v)),
        tuple((uu*y-uv*x)/det for x,y in zip(u,v)))


def _planar_loops(document,face,frame,check):
    vertices,edges=document['vertices'],document['edges']
    origin,u,v,normal,inverse=frame
    result=[]
    for loop in (face['loop'],*face['holes']):
        points=[]
        ends=[]
        for edge_id,forward in loop:
            check()
            edge=edges[edge_id]
            if edge['curve']['type']!='straight':
                raise GeometryError('native planar material requires straight trims')
            a,b=(vertices[edge['start']],vertices[edge['end']])
            if not forward:
                a,b=b,a
            if _dot(normal,_sub(a,origin)) or _dot(normal,_sub(b,origin)):
                raise GeometryError('trim is not exactly on native plane')
            points.append(tuple(_dot(row,_sub(a,origin)) for row in inverse))
            ends.append((a,b))
        if any(a[1]!=b[0] for a,b in zip(ends,(*ends[1:],ends[0]))):
            raise GeometryError('document trim loop is not closed')
        result.append(tuple(points))
    return tuple(result)


def _orient(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _simple_loops(loops,check):
    if not loops:
        raise GeometryError('missing outer material loop')
    segments=[]
    for points in loops:
        if len(points)<3 or len(set(points))!=len(points):
            raise GeometryError('degenerate or repeated material loop vertex')
        edges=tuple(zip(points,(*points[1:],points[0])))
        area=sum(a[0]*b[1]-a[1]*b[0] for a,b in edges)/2
        if area==0:
            raise GeometryError('zero material winding area')
        for i,(a,b) in enumerate(edges):
            c=edges[(i+1)%len(edges)][1]
            if _orient(a,b,c)==0 and _dot(_sub(b,a),_sub(c,b))<=0:
                raise GeometryError('material boundary backtracks')
        for i,j in _segment_candidates(edges,check):
            if abs(i-j)==1 or {i,j}=={0,len(edges)-1}:
                continue
            if _segments_meet(*edges[i],*edges[j]):
                raise GeometryError('material boundary is not simple')
        segments.append(edges)
    for index,hole in enumerate(loops[1:],1):
        if not all(_strictly_inside(loops[0],p) for p in hole):
            raise GeometryError('hole is not strictly within outer material')
        for previous in range(index):
            for a,b in segments[previous]:
                for c,d in segments[index]:
                    check()
                    if _segments_meet(a,b,c,d):
                        raise GeometryError('material hole boundaries meet')
            if previous>0 and (_strictly_inside(loops[previous],hole[0]) or
                               _strictly_inside(hole,loops[previous][0])):
                raise GeometryError('material holes overlap or nest')
    return tuple(segments)


def _indicator(loops,point):
    return int(_strictly_inside(loops[0],point) and
               not any(_strictly_inside(hole,point) for hole in loops[1:]))


def _positive_census(domains,check):
    """Exact open-slab indicator census; every crossing is a slab breakpoint."""
    segments=[]
    for loops in domains:
        for row in _simple_loops(loops,check):
            segments.extend(row)
    segments=tuple(sorted({tuple(sorted(edge)) for edge in segments}))
    xs={p[0] for edge in segments for p in edge}
    for i,j in _segment_candidates(segments,check):
        a,b=segments[i];c,d=segments[j]
        ab,cd=_sub(b,a),_sub(d,c)
        determinant=ab[0]*cd[1]-ab[1]*cd[0]
        if determinant:
            offset=_sub(c,a)
            t=(offset[0]*cd[1]-offset[1]*cd[0])/determinant
            s=(offset[0]*ab[1]-offset[1]*ab[0])/determinant
            if 0<=t<=1 and 0<=s<=1:
                xs.add(a[0]+t*ab[0])
    cells=0
    for lo,hi in zip(sorted(xs),sorted(xs)[1:]):
        check()
        x=(lo+hi)/2
        ys=sorted({a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0]) for a,b in segments
                   if min(a[0],b[0])<x<max(a[0],b[0])})
        for bottom,top in zip(ys,ys[1:]):
            check()
            point=x,(bottom+top)/2
            source=_indicator(domains[0],point)
            multiplicity=sum(_indicator(child,point) for child in domains[1:])
            if multiplicity!=source:
                raise GeometryError('positive material indicator multiplicity differs from source')
            cells+=1
    return cells


def _cylinder_frame(support,proof):
    origin,a,r,c=map(_qvector,support[:4])
    radius,height,start,sweep=map(F,support[4:])
    determinant=_dot(r,_cross(c,a))
    if determinant==0 or radius<=0 or abs(sweep)>=2*proof.pi_bound().lo:
        raise GeometryError('finite injective native cylinder chart unavailable')
    # The public stored-dot projection needs these exact identities for this
    # axial/coaxial family; a general approximate orthogonal frame is refused.
    if _dot(a,a)!=1 or _dot(a,r)!=0 or _dot(a,c)!=0:
        raise GeometryError('stored-dot cylinder chart transversality unproved')
    ra,ca=_cross(r,a),_cross(c,a)
    trace=_dot(ra,ra)+_dot(ca,ca)
    gram=_dot(ra,ra)*_dot(ca,ca)-_dot(ra,ca)**2
    if gram<=0:
        raise GeometryError('native cylinder physical density is singular')
    low=proof.sqrt(proof.i(gram/trace)).lo
    high=proof.sqrt(proof.i(_dot(ra,ra))).hi+proof.sqrt(proof.i(_dot(ca,ca))).hi
    return origin,a,r,c,radius,(low,high)


def _ray(point,frame):
    origin,a,r,c=frame[:4]
    offset=_sub(point,origin);z=_dot(offset,a)
    radial=_sub(offset,tuple(z*x for x in a))
    x,y=_dot(radial,r),_dot(radial,c)
    if x==y==0:
        raise GeometryError('cylinder trim reaches projected chart axis')
    scale=abs(x) if x else abs(y)
    return (x/scale,y/scale),z


def _angle_compare(anchor,first,second):
    def half(ray):
        x=_dot(anchor,ray);y=anchor[0]*ray[1]-anchor[1]*ray[0]
        return int(y<0 or y==0 and x<0)
    a,b=half(first),half(second)
    if a!=b:
        return -1 if a<b else 1
    cross=first[0]*second[1]-first[1]*second[0]
    return -1 if cross>0 else 1 if cross<0 else 0


def _angular_anchor(document,face,frame):
    for edge_id,_ in face['loop']:
        edge=document['edges'][edge_id]
        if edge['curve']['type']=='arc':
            first=_ray(document['vertices'][edge['start']],frame)[0]
            last=_ray(document['vertices'][edge['end']],frame)[0]
            via=_ray(document['vertices'][edge['curve']['via_vertex']],frame)[0]
            return first if _angle_compare(first,via,last)<0 else last
    raise GeometryError('cylinder material needs a finite angular source span')


def _cylinder_loops(document,face,frame,angles,proof,check,anchor):
    vertices,edges=document['vertices'],document['edges']
    origin,a,r,c,radius,density=frame
    def coordinate(point):
        token,z=_ray(point,frame)
        if token not in angles:
            x=_dot(anchor,token);y=anchor[0]*token[1]-anchor[1]*token[0]
            angle=proof.atan2(y,x)
            angles[token]=proof.add(angle,proof.mul(2,proof.pi_bound())) if y<0 else angle
        return token,z
    result=[]
    for loop in (face['loop'],*face['holes']):
        points=[]
        endpoints=[]
        for edge_id,forward in loop:
            check()
            edge=edges[edge_id]
            p,q=vertices[edge['start']],vertices[edge['end']]
            if not forward:
                p,q=q,p
            first,last=coordinate(p),coordinate(q)
            kind=edge['curve']['type']
            if kind=='straight':
                if any(_cross(_sub(q,p),a)) or first[0]!=last[0]:
                    raise GeometryError('cylinder material needs exact axial straight trim')
            elif kind=='arc':
                via=vertices[edge['curve']['via_vertex']]
                center,rsq,normal=_arc_circle((p,via,q))
                radial_center=_sub(_sub(center,origin),tuple(_dot(_sub(center,origin),a)*x for x in a))
                if any(_cross(normal,a)) or _dot(radial_center,radial_center)>=rsq:
                    raise GeometryError('geometric Arc projection monotonicity is unproved')
                # The exact circumcircle strictly encloses the projection axis.
                # Hence its projected angular derivative has fixed sign:
                # rsq - sqrt(rsq)*||radial_center|| > 0. An invertible stored
                # radial/circumferential projection preserves this monotonicity.
                # Its chart image is exactly the horizontal endpoint-ray span,
                # even when a refitted circle center differs from its ancestor.
                middle=coordinate(via)
                if first[1]!=last[1] or middle[1]!=first[1]:
                    raise GeometryError('cylinder circumcircle trim is not axial-constant')
                before,after,via_angle=first[0],last[0],middle[0]
                if not (_angle_compare(anchor,before,via_angle)<0 and _angle_compare(anchor,via_angle,after)<0 or
                        _angle_compare(anchor,after,via_angle)<0 and _angle_compare(anchor,via_angle,before)<0):
                    raise GeometryError('cylinder geometric Arc lift crosses unresolved branch seam')
            else:
                raise GeometryError('unsupported native cylinder trim family')
            points.append(first)
            endpoints.append((p,q))
        if any(a[1]!=b[0] for a,b in zip(endpoints,(*endpoints[1:],endpoints[0]))):
            raise GeometryError('document circumcircle/straight loop is not closed')
        result.append(tuple(points))
    return tuple(result)


def _material(root,children,original,current,source_supports,current_supports,proof,frame_cache=None):
    check=proof.charge
    frame_cache={} if frame_cache is None else frame_cache
    def compile_frame(kind,data):
        key=(kind,data)
        if key not in frame_cache:
            frame_cache[key]=_plane_frame(data) if kind=='plane' else _cylinder_frame(data,proof)
        return frame_cache[key]
    original_faces=original['faces'];current_faces=current['faces']
    family,support=source_supports[root]
    evidence={'domain_semantics':'exact stored endpoint straight/circumcircle trims; positive outer-minus-holes',
              'source_native_support':support,'current_native_supports':[],
              'floating_evaluation_preservation_qualified':False,
              'source_face_use_occurrences':original['face_uses_by_face'].get(root,[]),
              'current_face_use_occurrences':[r for child in children for r in current['face_uses_by_face'].get(child,())]}
    try:
        if family=='plane':
            frame=compile_frame(family,support)
            curved=any(document['edges'][edge]['curve']['type']!='straight'
                for document,face in [(original,original_faces[root])]+[(current,current_faces[c]) for c in children]
                for loop in (face['loop'],*face['holes']) for edge,_ in loop)
            if curved:
                from .curved_planar_material import compile_loops,positive_chain_census
                compile_plane=lambda document,face:compile_loops(document,face,frame,proof)
            else:compile_plane=lambda document,face:_planar_loops(document,face,frame,check)
            domains=[compile_plane(original,original_faces[root])]
            for child in children:
                check()
                kind,child_support=current_supports[child]
                evidence['current_native_supports'].append((child,kind,child_support))
                if kind!='plane':
                    raise GeometryError('native child support family changed')
                cf=compile_frame(kind,child_support)
                if _dot(frame[3],_sub(cf[0],frame[0])) or _dot(frame[3],cf[1]) or _dot(frame[3],cf[2]):
                    raise GeometryError('native child plane is not exactly common support')
                domains.append(compile_plane(current,current_faces[child]))
            if curved:
                evidence.update(positive_chain_census(domains,proof))
                cells=None
            else:cells=_positive_census(domains,check)
            evidence.update(census_cells=cells,chart_coordinates='exact source native Plane UV',
                all_loops=[[[tuple(map(_pack,p.p if curved else p)) for p in loop] for loop in domain] for domain in domains])
            if curved:evidence['chart_coordinates']='axis-aligned physical Plane coordinates; actual line/circumcircle spans'
        elif family=='cylinder':
            frame=compile_frame(family,support)
            angles={}
            anchor=_angular_anchor(original,original_faces[root],frame)
            domains=[_cylinder_loops(original,original_faces[root],frame,angles,proof,check,anchor)]
            for child in children:
                check()
                kind,child_support=current_supports[child]
                evidence['current_native_supports'].append((child,kind,child_support))
                if kind!='cylinder':
                    raise GeometryError('native child support family changed')
                cf=compile_frame(kind,child_support)
                if cf[1:5]!=frame[1:5] or any(_cross(_sub(cf[0],frame[0]),frame[1])):
                    raise GeometryError('native cylinder support transition differs')
                domains.append(_cylinder_loops(current,current_faces[child],frame,angles,proof,check,anchor))
            ordered=sorted(angles,key=cmp_to_key(lambda a,b:_angle_compare(anchor,a,b)))
            ranks={token:F(i) for i,token in enumerate(ordered)}
            census=tuple(tuple(tuple((ranks[token],z) for token,z in loop) for loop in domain) for domain in domains)
            cells=_positive_census(census,check)
            evidence.update(census_cells=cells,chart_coordinates='certified ordered angle, exact axial z',
                native_projection='z=(P-O).a; phi=atan2((P-O-z*a).stored_c,(P-O-z*a).stored_r)',
                harmonic_density_coordinates='x=R*alpha, y=z; alpha is native harmonic angle, not stored-dot phi or symbolic rank',
                harmonic_density_bounds=tuple(map(_pack,frame[5])),
                physical_measure_argument='common injective support and equal pointwise material indicators; no density integration',
                angular_tokens=[(tuple(map(_pack,t)),tuple(map(_pack,(angles[t].lo,angles[t].hi)))) for t in ordered],
                all_rank_loops=[[[tuple(map(_pack,p)) for p in loop] for loop in domain] for domain in census],
                rank_coordinates_are_area_units=False)
            evidence['relative_ray_anchor']=tuple(map(_pack,anchor))
        elif family in POLYNOMIAL_EXTRUSION_KINDS:
            # This new evidence proves SUPPORT and signed chart correspondence
            # only. No trim census exists here for a Coons/Bezier wall or its
            # BQC boundaries. Never promote it to material/reference permission.
            support_rows=[]
            for child in children:
                check()
                kind,child_support=current_supports[child]
                evidence['current_native_supports'].append((child,kind,child_support))
                support_rows.append((child,prove_polynomial_extrusion_support(
                    family,support,kind,child_support,charge=check)))
            evidence.update(domain_semantics='trimmed material UNQUALIFIED; stored polynomial support identity only',
                polynomial_support_correspondence=support_rows,
                trimmed_partition_qualified=False,geometry_reference_mapping_qualified=False)
            return NativeMaterialScopeRow(root,children,family,'support_only',False,True,_json(evidence),
                'polynomial support correspondence does not establish trimmed material partition')
        else:
            raise GeometryError('unsupported native material support')
        return NativeMaterialScopeRow(root,children,family,'exact_document_material',True,True,_json(evidence),None)
    except _Refusal:
        raise
    except GeometryError as error:
        if error is proof.callback_error:
            raise
        return NativeMaterialScopeRow(root,children,family,'refused',False,False,_json(evidence),str(error))


def query_prepared_native_material_reference_scope(model, authored_face_ids, *,
        expected_revision=None,policy=None,cancellation_check=None):
    """One owner-issued whole-domain and geometry-reference batch, read-only."""
    if type(authored_face_ids) not in (tuple,list):
        raise GeometryError('native material scope requires finite list/tuple of roots')
    selected=tuple(authored_face_ids)
    if not selected or any(type(x) is not int or x<=0 for x in selected) or len(set(selected))!=len(selected):
        raise GeometryError('native material scope requires unique positive root IDs')
    selected=tuple(sorted(selected))
    if policy is not None and type(policy) is not NativeArcParameterMapPolicy:
        raise GeometryError('native scope needs a plain aggregate work policy')
    policy=NativeArcParameterMapPolicy() if policy is None else NativeArcParameterMapPolicy(policy.max_interval_operations)
    scope=query_prepared_model_scope(model,expected_revision=expected_revision)
    original_scope_digest=definition_checksum(scope)
    scope=deepcopy(scope)
    preimages=scope.face_preimages
    if preimages.authored_native_supports is None or preimages.current_native_supports is None:
        raise GeometryError('prospective actual native support capture is unavailable')
    if not set(selected)<=set(preimages.authored_face_ids):
        raise GeometryError('native material scope root is unavailable')
    actual=capture_native_supports(model)
    if actual!=preimages.current_native_supports:
        raise GeometryError('current actual native support binding changed')
    native_digest=definition_checksum(actual)
    replacements=_replacement_snapshot(model)
    edge_binding=query_prepared_edge_subcurve_preimages(model)
    edge_digest=definition_checksum(edge_binding)
    edges=deepcopy(edge_binding)
    original,current=scope.authored_document,scope.current_document
    original_index,current_index=_compile_document(original),_compile_document(current)
    proof=_Proof(policy,None)
    frames={}
    try:
        for row in edges.arc_records+edges.arc_alias_records:
            proof.charge()
            for definition in (row.ancestor.definition,row.current_definition):
                if definition not in frames:
                    proof.charge()
                    frames[definition]=_frame(definition)
    except _Refusal as error:
        raise GeometryError('native material/reference aggregate budget exhausted') from error
    def callback(phase):
        if cancellation_check is not None and cancellation_check(phase):
            raise GeometryError('native material/reference scope cancelled')
    proof.callback=callback
    source_supports={key:(kind,data) for key,kind,data in preimages.authored_native_supports}
    current_supports={key:(kind,data) for key,kind,data in actual}
    descendants=dict(preimages.face_descendants)
    try:
        proof.cancel('native material/reference scope')
        arc_maps=tuple(_entry(proof,row,frames[row.ancestor.definition],frames[row.current_definition])
                      for row in edges.arc_records+edges.arc_alias_records)
        edge_maps=[]
        for row in edges.records+edges.alias_records:
            proof.charge()
            exact=_unpack(row.squared_distance_bound)==0
            edge_maps.append(dict(edge_id=row.edge_id,ancestor_edge_id=row.ancestor.definition.edge_id,
                ancestor_model_id=str(row.ancestor.model_id),ancestor_revision=row.ancestor.revision,
                ancestor_source_checksum=row.ancestor.source_checksum,interval=row.interval,
                classification='exact' if exact else 'bounded',squared_residual_bound=row.squared_distance_bound))
        for row in arc_maps:
            edge_maps.append(dict(edge_id=row.edge_id,ancestor_edge_id=row.ancestor_edge_id,
                ancestor_model_id=str(row.ancestor_model_id),ancestor_revision=row.ancestor_revision,
                ancestor_source_checksum=row.ancestor_source_checksum,interval=tuple(map(_pack,row.interval)),
                classification=row.classification,residual_bound=_pack(row.residual_bound)))
        frame_cache={}
        material=tuple(_material(root,descendants[root],original_index,current_index,source_supports,current_supports,proof,frame_cache)
                       for root in selected)
        orientation_relations={}
        for root,children in preimages.face_descendants:
            kind,support=source_supports[root]
            for child in children:
                proof.charge()
                ck,cs=current_supports[child]
                same=False
                if kind==ck=='plane':
                    sn=_cross(_qvector(support[1]),_qvector(support[2]))
                    cn=_cross(_qvector(cs[1]),_qvector(cs[2]))
                    same=not any(_cross(sn,cn)) and _dot(sn,cn)>0
                elif kind==ck=='cylinder':
                    same=(support[1:5]==cs[1:5] and F(support[5])*F(support[7])*F(cs[5])*F(cs[7])>0)
                orientation_relations[root,child]=same
        inventory,outside,reference_ok=_complete_native_reference_inventory(original,current,
            preimages.face_descendants,set(selected),edge_maps,proof.charge,source_supports,current_supports,
            orientation_relations,{child for row in material if row.document_material_qualified for child in row.current_face_ids},proof,replacements,
            preimages.attachment_source_ids,preimages.epoch_derived_attachment_ids)
        inventory['replacement_history_snapshot']=replacements
        inventory['native_edge_maps']=edge_maps
        inventory['material_dispositions']=[(r.authored_face_id,r.classification,r.refusal) for r in material]
        if any(row.family in POLYNOMIAL_EXTRUSION_KINDS for row in material):
            # Keep the existing orientation/semantic remap predicates closed.
            # Signed maps are available in support evidence, but may not bypass
            # missing trimmed material and reference proofs.
            reference_ok=False
            inventory['polynomial_support_only_reference_refusal']=True
        proof.cancel('native material/reference scope final check')
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        raise GeometryError('native material/reference proof unavailable: '+str(error)) from error
    validate_prepared_model_scope_binding(model,scope)
    if (definition_checksum(scope)!=original_scope_digest or capture_native_supports(model)!=actual
            or _replacement_snapshot(model)!=replacements):
        raise GeometryError('native material/reference scope changed during query')
    if definition_checksum(query_prepared_edge_subcurve_preimages(model))!=edge_digest:
        raise GeometryError('native edge ancestry changed during query')
    result=PreparedNativeMaterialReferenceScope(scope,selected,outside,policy,native_digest,
        edge_digest,material,arc_maps,_json(inventory),reference_ok,
        all(row.document_material_qualified for row in material),work_counts=tuple(sorted(proof.counts.items())))
    issued=_issued.setdefault(model,{})
    key=id(result)
    issued[key]=(ref(result,lambda _:issued.pop(key,None)),_digest(result))
    return result


def validate_prepared_native_material_reference_scope_binding(model,receipt,*,cancellation_check=None):
    """Guard an owner-issued receipt, pinned before callbacks and after work."""
    issued=_issued.get(model,{}).get(id(receipt))
    if issued is None or issued[0]() is not receipt:
        raise GeometryError('native material/reference receipt was not issued to this owner')
    pinned=issued[1]
    if _digest(receipt)!=pinned:
        raise GeometryError('native material/reference receipt changed')
    snapshot=deepcopy(receipt)
    replacements=tuple((kind,key,tuple(tuple(child) for child in children))
                       for kind,key,children in snapshot.inventory['replacement_history_snapshot'])
    validate_prepared_model_scope_binding(model,snapshot.scope)
    if definition_checksum(capture_native_supports(model))!=snapshot.native_support_digest:
        raise GeometryError('native support binding changed')
    if definition_checksum(query_prepared_edge_subcurve_preimages(model))!=snapshot.edge_ancestry_digest:
        raise GeometryError('native ancestry binding changed')
    if _replacement_snapshot(model)!=replacements:
        raise GeometryError('native replacement-history binding changed')
    if cancellation_check is not None and cancellation_check('native material/reference scope binding'):
        raise GeometryError('native material/reference scope cancelled')
    validate_prepared_model_scope_binding(model,snapshot.scope)
    if (_digest(receipt)!=pinned or definition_checksum(capture_native_supports(model))!=snapshot.native_support_digest
            or definition_checksum(query_prepared_edge_subcurve_preimages(model))!=snapshot.edge_ancestry_digest
            or _replacement_snapshot(model)!=replacements):
        raise GeometryError('native material/reference binding changed during validation')
