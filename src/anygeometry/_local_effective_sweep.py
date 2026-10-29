"""Live analytic sweep unions; original regions/features are never merged away."""
from dataclasses import dataclass,replace
from fractions import Fraction as Q
from weakref import WeakKeyDictionary,ref
from .definition_binding import definition_checksum
from .errors import GeometryError
from .nurbs import NurbsCurve,CurveOnSurface
from .prepared_material_source_transfer import _state,_effective,_chart_map
from .nurbs_sweep_chart import has_sweep_chart,_decode,_tensor
from ._translational_pairs import _sweep
from .plane_nurbs import charge_box
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY


class EffectiveSweepNotCertified(GeometryError):pass


@dataclass(frozen=True,slots=True)
class EffectiveSweepMember:
    face_id: int
    profile_interval: tuple
    source_from_native: tuple
    connectors: tuple
    source_chart: object
    face_uses: tuple
    retained_features: tuple
    connector_transport_upper: Q


@dataclass(frozen=True,eq=False)
class EffectiveSweepUnion:
    source_model_id: object
    source_checksum: str
    profile: NurbsCurve
    translation: tuple
    members: tuple
    seams: tuple
    external_trim_loops: tuple
    closed: bool
    transport_upper: Q
    @property
    def face_ids(self):return tuple(member.face_id for member in self.members)


_ISSUED=WeakKeyDictionary()


def _recipe(source,face):
    record=source.faces[face]
    if has_sweep_chart(record):
        profile,vector,support=_decode(record)
        # Source union is a pre-cut full rectangle. Prepared children are
        # consumed only through the separate owner partition/transport receipt.
        rows=record.metadata['_nurbs_sweep_chart_v1'].get('connectors',())
        if len(rows)!=2:return None
        connectors=tuple(row[0] for row in rows)
    else:
        recognized=_sweep(source,face)
        if recognized is None:return None
        profile,vector,connectors=recognized
    if type(profile) is not NurbsCurve or len(connectors)!=2:return None
    if record.holes or len(record.corners)!=4:return None
    support=_effective(source,face)
    if definition_checksum(support)!=definition_checksum(_tensor(profile,vector)):
        raise EffectiveSweepNotCertified('sweep effective recipe changed')
    return profile,tuple(vector),tuple(connectors),support


def sweep_groups(source,faces):
    """Candidate grouping only, never exclusion or union authority."""
    groups={}
    for face in faces:
        recipe=_recipe(source,face)
        if recipe is None:continue
        profile,vector,_,_=recipe
        full=(profile.knots[profile.degree],profile.knots[len(profile.controls)])
        key=(definition_checksum(replace(profile,domain=full)),vector)
        groups.setdefault(key,[]).append(face)
    return tuple(tuple(group) for group in groups.values() if len(group)>1)


def query_effective_sweep_union(source,faces,*,cancellation_check=None):
    from .material_regions import query_material_surface_regions,validate_material_surface_regions_binding
    ids=tuple(faces)
    if not ids or any(type(f) is not int for f in ids) or len(ids)!=len(set(ids)):
        raise EffectiveSweepNotCertified('sweep union requires unique source faces')
    initial=_state(source);runtime=_runtime()
    if runtime!=_INITIAL_RUNTIME:raise EffectiveSweepNotCertified('sweep runtime context unsupported')
    used=0
    def check():
        nonlocal used
        used+=1
        if used>DEFAULT_INTERSECTION_QUALIFICATION_POLICY.max_boxes_per_pair:raise EffectiveSweepNotCertified('sweep union work budget exhausted')
        charge_box()
        if _state(source)!=initial or _runtime()!=runtime:raise EffectiveSweepNotCertified('sweep source changed')
        if cancellation_check is not None:
            cancelled=cancellation_check('effective sweep union')
            if _state(source)!=initial or _runtime()!=runtime:raise EffectiveSweepNotCertified('sweep changed during callback')
            if cancelled:raise EffectiveSweepNotCertified('sweep union cancelled')
    check();regions=query_material_surface_regions(source,ids,cancellation_check=cancellation_check)
    charts={chart.face.id:chart for region in regions.regions for chart in region.sources}
    members=[];common=None;vector=None;recipes={}
    for face in ids:
        check();recipe=_recipe(source,face)
        if recipe is None:raise EffectiveSweepNotCertified('source has no complete canonical sweep rectangle')
        profile,direction,connectors,support=recipe
        full=replace(profile,domain=(profile.knots[profile.degree],profile.knots[len(profile.controls)]))
        if common is None:common,vector=full,direction
        if definition_checksum(common)!=definition_checksum(full) or vector!=direction:
            raise EffectiveSweepNotCertified('sweep union lacks one exact profile/translation')
        chart=charts.get(face)
        if chart is None or not chart.face_uses:raise EffectiveSweepNotCertified('sweep union lacks structural occurrences')
        if definition_checksum(chart.support)!=definition_checksum(support):raise EffectiveSweepNotCertified('sweep chart support mismatch')
        # Recipe recognition proves four complete original sides; chart services
        # independently bind their actual world paths and connector transport.
        if len(chart.boundaries)!=1 or len(chart.boundaries[0])!=4:
            raise EffectiveSweepNotCertified('sweep union requires complete unholed rectangular trims')
        error=Q(0);features=[]
        for end,edge_id in enumerate(connectors):
            check()
            edge=source.edges[edge_id];exact=profile.exact_point(end)
            targets=(exact,tuple(x+d for x,d in zip(exact,vector)))
            actual=tuple(tuple(Q(float(x)) for x in source.vertex_position(v)) for v in (edge.start,edge.end))
            forward=tuple(b-a for a,b in zip(*actual))==vector
            if not forward:actual=actual[::-1]
            if tuple(b-a for a,b in zip(*actual))!=vector:raise EffectiveSweepNotCertified('connector translation changed')
            error=max(error,*(sum(abs(a-b) for a,b in zip(point,target)) for point,target in zip(actual,targets)))
            features.append((edge_id,edge.start,edge.end,definition_checksum(edge),tuple(sorted(source.tags_for(edge.ref)))))
        members.append(EffectiveSweepMember(face,tuple(map(Q,profile.domain)),_chart_map(source,face,support),
            connectors,chart,tuple(chart.face_uses),tuple(features),error))
        recipes[face]=profile
    members.sort(key=lambda m:min(m.profile_interval))
    intervals=[tuple(sorted(m.profile_interval)) for m in members]
    if any(a>=b for a,b in intervals) or any(left[1]!=right[0] for left,right in zip(intervals,intervals[1:])):
        raise EffectiveSweepNotCertified('sweep union profile ranges have gap or duplicate overlap')
    # No unselected incident source may disappear behind the analytic union.
    for member in members:
        for edge in member.connectors:
            if not set(source.faces_using_edge(edge))<=set(ids):raise EffectiveSweepNotCertified('sweep union omits connector incidence')
    a,b=intervals[0][0],intervals[-1][1]
    common=replace(common,domain=(float(a),float(b)))
    closed=common.exact_point(0)==common.exact_point(1)
    seams=[]
    pairs=list(zip(members,members[1:]))
    if closed:pairs.append((members[-1],members[0]))
    for first,second in pairs:
        check()
        left=first.connectors[1 if first.profile_interval[1]>first.profile_interval[0] else 0]
        right=second.connectors[0 if second.profile_interval[1]>second.profile_interval[0] else 1]
        if left!=right or set(source.faces_using_edge(left))!={first.face_id,second.face_id}:
            raise EffectiveSweepNotCertified('profile seam lacks unique actual shared connector')
        fp=recipes[first.face_id];sp=recipes[second.face_id]
        u=1 if first.profile_interval[1]>first.profile_interval[0] else 0
        v=0 if second.profile_interval[1]>second.profile_interval[0] else 1
        if fp.exact_point(u)!=sp.exact_point(v):raise EffectiveSweepNotCertified('profile seam exact anchors differ')
        seams.append((left,first.face_id,second.face_id,tuple(fp.exact_point(u))))
    tensor=_tensor(common,vector)
    lower=[];upper=[]
    for member in members:
        lo,hi=sorted(member.profile_interval)
        profile=replace(common,domain=(float(lo),float(hi)))
        lower.append(profile)
        translated=_tensor(profile,vector)
        upper.append(replace(profile,controls=tuple(row[1] for row in translated.controls),domain=(float(hi),float(lo))))
    if closed:loops=(tuple(lower),tuple(reversed(upper)))
    else:
        def connector(u,reverse=False):
            uv=((u,1.),(u,0.)) if reverse else ((u,0.),(u,1.))
            return CurveOnSurface(tensor,NurbsCurve(1,(0.,0.,1.,1.),uv))
        loops=((*lower,connector(1.),*reversed(upper),connector(0.,True)),)
    check();validate_material_surface_regions_binding(source,regions);check()
    result=EffectiveSweepUnion(source.model_id,initial[1],common,vector,tuple(members),tuple(seams),loops,closed,
        max(member.connector_transport_upper for member in members))
    _ISSUED[result]=(ref(source),initial,regions,definition_checksum(result),runtime)
    return result


def validate_effective_sweep_union(source,result,*,cancellation_check=None):
    from .material_regions import validate_material_surface_regions_binding
    entry=_ISSUED.get(result) if type(result) is EffectiveSweepUnion else None
    if entry is None or entry[0]() is not source:raise EffectiveSweepNotCertified('sweep union not issued for source')
    def guard():
        if _state(source)!=entry[1] or definition_checksum(result)!=entry[3] or _runtime()!=entry[4]:
            raise EffectiveSweepNotCertified('sweep union stale or changed')
    guard();validate_material_surface_regions_binding(source,entry[2],cancellation_check=cancellation_check)
    if cancellation_check is not None:
        guard();cancelled=cancellation_check('effective sweep union binding');guard()
        if cancelled:raise EffectiveSweepNotCertified('sweep union cancelled')
    guard()


def _runtime():
    from . import _translational_pairs as pairs,nurbs_sweep_chart as charts,prepared_material_source_transfer as transfer
    from . import nurbs
    functions=(nurbs.NurbsCurve,nurbs.NurbsSurface,NurbsCurve.__init__,NurbsCurve.__post_init__,
        nurbs.NurbsSurface.__init__,nurbs.NurbsSurface.__post_init__,CurveOnSurface.__init__,
        charts._qualified_sweep_chart_face,charts.source_sweep_chart_model,
        charts._SourceSweepChartView.__init__,charts._SourceSweepChartView.chart_face,
        charts.SweepChartModel.chart_edge,charts.SweepChartModel._connector_edge,
        _recipe,_sweep,pairs._sweep,_decode,charts._decode,_tensor,charts._tensor,
        _effective,transfer._effective,_chart_map,transfer._chart_map,NurbsCurve.exact_point,
        EffectiveSweepMember.__init__,EffectiveSweepUnion.__init__)
    return tuple((f,getattr(f,'__code__',None),definition_checksum(getattr(f,'__defaults__',None)),
        definition_checksum(getattr(f,'__kwdefaults__',None))) for f in functions)


_INITIAL_RUNTIME=_runtime()
