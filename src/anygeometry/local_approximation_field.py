"""Owner-issued local budgets for a bounded, explicitly supported source family.



Producer: globally monotone active-single-span NURBS/Plane sheets with

qualified trim stars; polynomial NN contact plans require actual realization.

The conservative whole-face infimum is used when no smaller footprint is proved.

"""

from dataclasses import dataclass

from fractions import Fraction as Q

from numbers import Real

import math

from weakref import WeakKeyDictionary,ref

from .errors import GeometryError

from .definition_binding import definition_checksum

from .nurbs import NurbsSurface,CurveOnSurface,_sqrt_down,_float_down

from .surfaces import Plane

from .arrangement_geometry import LinePath

from .trimmed_charts import query_trimmed_surface_charts,validate_trimmed_surface_charts_binding

from .prepared_material_source_transfer import _state,_effective,_chart_map

from ._local_curvature import bound_surface_curvature,LocalApproximationNotCertified

from ._local_neighborhood import certify_surface_star,projected_hull,separation_squared

from ._local_straight_clearance import clearance_of_straight_loops

from .nurbs_surface_intersection._bernstein import bezier_piece

from .nurbs_surface_intersection._intervals import Interval as I



_FIELDS=WeakKeyDictionary();_ALLOCATIONS=WeakKeyDictionary()

@dataclass(frozen=True,eq=False)

class LocalApproximationField:

    version: int

    model_id: object

    revision: int

    source_checksum: str

    face_ids: tuple

    relative_accuracy: Q

    scale_error: Q

    policy_floor: Q

    # face, authored-from-native chart map, ((native box, limit, curvature upper),...)

    face_tiles: tuple

    clearance_lower: Q

    # face, native chart enclosure, limit, source curve identity and interval

    boundary_tiles: tuple = ()

    contact_intent: bool = False

    contacts: tuple = ()

    spatial_ledger: tuple = ()
    span_certificates: tuple = ()
    # Live source material views; never reconstructed from serialized metadata.
    sweep_unions: tuple = ()



@dataclass(frozen=True,eq=False)

class LocalApproximationAllocation:

    fingerprint: tuple

    # Conservative whole-source-face limits are valid for every child footprint.

    face_limits: tuple





def _segment(curve):

    if type(curve) is LinePath:

        return tuple(tuple(Q(float(x)) for x in p) for p in (curve.start,curve.end))

    if type(curve) is CurveOnSurface:

        actual=curve._iso_world(fresh=True)

        if actual is not None and actual[0]==1 and len(actual[2])==2 and all(h[3]>0 for h in actual[2]):

            return tuple(tuple(curve.exact_point(t)) for t in (0,1))

    raise LocalApproximationNotCertified('local field trim-curve/contact adapter is not certified')





def _field_frontier(model,before,cancellation_check):

    if _state(model)!=before:raise LocalApproximationNotCertified('local field source changed before callback')

    if cancellation_check is not None:

        if cancellation_check('local approximation field'):

            raise LocalApproximationNotCertified('local field cancelled')

        if _state(model)!=before:raise LocalApproximationNotCertified('local field source changed')





def _canonical_line_operand(value):
    """Stored binary64 Plane boundary coordinates have exact float copies.

    This admits only the existing canonical LinePath container and numpy's
    binary64 scalar. Unknown/custom scalar classes retain detached fallback.
    """
    if type(value) is not LinePath:return value
    from numpy import float64
    def coordinates(row):
        if type(row) is not tuple:return row
        return tuple(float(x) if type(x) is float64 else x for x in row)
    return LinePath(coordinates(value.start),coordinates(value.end))


def _sweep_boundary_occurrences(model,union):
    """Cancel only authenticated internal connector uses, keeping source IDs.

    This constructs feature-clearance cycles, not replacement meshing topology.
    Every original member/connector remains in the live union and source model.
    """
    seams={row[0] for row in union.seams}
    outgoing={};count=0
    for member in union.members:
        uses={use.edge:use for use in model.faces[member.face_id].loop}
        for loop in member.source_chart.boundaries:
            for path in loop:
                if path.source_edge in seams:continue
                use=uses[path.source_edge]
                start=model.oriented_start_vertex(use);end=model.oriented_end_vertex(use)
                if start in outgoing:raise LocalApproximationNotCertified('sweep external boundary branches at an authored vertex')
                outgoing[start]=(end,member.face_id,path.curve,path.source_edge)
                count+=1
    loops=[]
    while outgoing:
        first=min(outgoing);cursor=first;rows=[]
        while True:
            item=outgoing.pop(cursor,None)
            if item is None:raise LocalApproximationNotCertified('sweep external material boundary has a gap')
            cursor,face,curve,edge=item;rows.append((face,curve,edge))
            if cursor==first:break
            if len(rows)>count:raise LocalApproximationNotCertified('sweep boundary traversal did not close')
        loops.append(tuple(rows))
    if not loops:raise LocalApproximationNotCertified('sweep union omitted all external material boundaries')
    return tuple(loops)


def query_local_approximation_field(model,faces,*,relative_accuracy,scale_error,policy_floor,cancellation_check=None,connect_intent=False):

    if type(connect_intent) is not bool:raise GeometryError('contact intent must be an explicit boolean')

    before=_state(model)

    from ._local_contact_stars import _ContactProposalRegistry

    contact_registry=_ContactProposalRegistry(model)

    contact_guard=contact_registry.guard

    contact_runtime_check=contact_registry._runtime_check

    cancellation_check=contact_registry.wrap_callback(cancellation_check)

    def positive(value):

        if isinstance(value,bool) or not isinstance(value,Real):

            raise LocalApproximationNotCertified('local field policy requires finite positive real values')

        try:

            if not math.isfinite(float(value)) or value<=0:raise ValueError()

            return Q(value)

        except (ValueError,TypeError,OverflowError,ZeroDivisionError):

            raise LocalApproximationNotCertified('local field policy requires finite positive real values') from None

    alpha,ceiling,floor=map(positive,(relative_accuracy,scale_error,policy_floor))

    if not 0<floor<=ceiling or alpha<=0:raise LocalApproximationNotCertified('invalid local field policy')

    charts=query_trimmed_surface_charts(model,faces,cancellation_check=cancellation_check)

    from ._detached_geometry_read import detach_numeric, NumericalBoundary, DetachedOperandUnavailable

    chart_pin=definition_checksum(charts)

    from ._local_effective_sweep import (sweep_groups,query_effective_sweep_union,validate_effective_sweep_union)
    sweep_unions=tuple(query_effective_sweep_union(model,group,cancellation_check=cancellation_check)
        for group in sweep_groups(model,tuple(c.face.id for c in charts.charts)))

    from ._local_contact_stars import _runtime

    runtime_pin=_runtime()

    from ._local_piecewise_surface_star import require_span_context
    def live_guard():
        require_span_context()
        for union in sweep_unions:validate_effective_sweep_union(model,union)

        contact_runtime_check();contact_guard()

        if (_state(model)!=before or definition_checksum(charts)!=chart_pin or _runtime()!=runtime_pin):

            raise LocalApproximationNotCertified('local field source, chart or runtime changed')

    boundary=NumericalBoundary(live_guard,cancellation_check,source=model,
                               error_type=LocalApproximationNotCertified)

    def numerical_operand(value):

        try: return detach_numeric(_canonical_line_operand(value))

        except DetachedOperandUnavailable:

            # Unknown data retains the original per-tick authority route.

            boundary._detached=False

            return value

    numeric_callback=boundary.tick

    def check():

        return numeric_callback()

    selected={chart.face.id for chart in charts.charts}

    inputs=[]
    plane_covers={}

    for chart in charts.charts:

        support=_effective(model,chart.face.id)

        if definition_checksum(support)!=definition_checksum(chart.support):

            raise LocalApproximationNotCertified('effective material support mismatch')

        loops_read=[]

        for loop in chart.boundaries:

            paths=[]

            for path in loop:

                if not set(model.faces_using_edge(path.source_edge))<=selected:

                    raise LocalApproximationNotCertified('local field selected closure omits incident source faces')

                paths.append((numerical_operand(path.curve),path.source_edge))

            loops_read.append(tuple(paths))

        chart_map=_chart_map(model,chart.face.id,support)
        detached_support=numerical_operand(support)
        if sweep_unions and type(support) is Plane:
            from .local_sweep_plane_contact import _plane_cover
            cover=_plane_cover(model,chart.face.id)
            (u0,u1),(v0,v1)=cover.chart_box
            if u0>=u1 or v0>=v1:raise LocalApproximationNotCertified('plane material chart cover is degenerate')
            chart_map=((u1-u0,Q(0),u0),(Q(0),v1-v0,v0))
            plane_covers[id(detached_support)]=cover
        inputs.append((chart.face.id,detached_support,tuple(loops_read),numerical_operand(chart_map)))

    inputs=tuple(inputs)

    boundary.guard()

    loops=[];curve_loops=[];curve_occurrences=[];bounds=[];curvatures=[];supports={}

    span_atlases={};atlas_pins=[];span_certificates=[]
    all_straight=True

    for face_id,support,read_loops,chart_map in inputs:

        check()

        supports[face_id]=support
        span_work=0;span_splits=0;atlas=None

        local=[]

        for loop in read_loops:

            segments=[];curve_loops.append(tuple(curve for curve,edge in loop))

            for curve,edge in loop:

                curve_occurrences.append((face_id,support,curve,edge))

                try:segments.append(_segment(curve))

                except LocalApproximationNotCertified:all_straight=False

            if len(segments)==len(loop):local.append(tuple(segments))

        local=tuple(local)

        loops.extend(local)

        if type(support) is NurbsSurface:

            from ._local_piecewise_surface_star import (SpanBudget,surface_span_atlas,
                certify_piecewise_surface_star,span_hull)
            span_budget=SpanBudget(check=check)
            atlas=surface_span_atlas(support,budget=span_budget)
            span_star=certify_piecewise_surface_star(atlas,((0,1),(0,1)),((0,1),(0,1)),budget=span_budget,
                source_boundary_sides=((0,0),(0,1),(1,0),(1,1)))
            hull=span_hull(atlas,((0,1),(0,1)),take=span_budget.take)
            span_atlases[id(support)]=atlas
            atlas_hash=definition_checksum(atlas)
            atlas_pins.append((atlas,atlas_hash))
            span_certificates.append((face_id,atlas.source_definition,atlas_hash,
                tuple(piece.chart_box for piece in atlas.pieces),
                (span_star.inner_box,span_star.outer_box,span_star.projection,
                 span_star.strong_monotonicity_lower,span_star.projection_norm_upper,
                 span_star.parameter_buffer,span_star.inner_to_outer_boundary_lower,
                 span_star.source_boundary_sides)))
            span_work=span_budget.used;span_splits=span_budget.splits
        elif type(support) is Plane:

            from ._local_curve_geometry import bound_curve_geometry

            boxes=[piece.world_box for loop in read_loops for curve,edge in loop

                for piece in bound_curve_geometry(curve,cancellation_check=numeric_callback).pieces]

            hull=tuple(I(min(box[k][0] for box in boxes),max(box[k][1] for box in boxes)) for k in range(3))

        else:raise LocalApproximationNotCertified('local field support adapter unsupported')

        bounds.append(hull)

        from dataclasses import replace as policy_replace
        from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        policy=DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        remaining=policy.max_boxes_per_pair-span_work
        if remaining<2:raise LocalApproximationNotCertified('spatial curvature box budget exhausted')
        curvature=bound_surface_curvature(support,cancellation_check=numeric_callback,
            policy=policy_replace(policy,max_boxes_per_pair=remaining),_atlas=atlas)
        curvature=policy_replace(curvature,boxes=curvature.boxes+span_work,
            subdivisions=curvature.subdivisions+span_splits)

        # Prospective bounded enrichment: non-flat tiles get two tensor bisections.

        # Every replacement is independently certified; flat charts keep one tile.

        # This changes bound tightness/coverage only, never the error policy.

        if any(tile.curvature_upper>0 for tile in curvature.tiles):

            from dataclasses import replace

            from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY

            from dataclasses import replace as policy_replace

            budget=DEFAULT_INTERSECTION_QUALIFICATION_POLICY

            leaves=[];used_boxes=curvature.boxes;used_splits=curvature.subdivisions

            for tile in curvature.tiles:

                (u0,u1),(v0,v1)=tile.chart_box

                us=tuple(u0+(u1-u0)*Q(i,4) for i in range(5))

                vs=tuple(v0+(v1-v0)*Q(i,4) for i in range(5))

                for i in range(4):

                    for j in range(4):

                        check()

                        remaining=budget.max_boxes_per_pair-used_boxes

                        if remaining<2:raise LocalApproximationNotCertified('spatial curvature box budget exhausted')

                        child=bound_surface_curvature(support,((us[i],us[i+1]),(vs[j],vs[j+1])),

                            policy=policy_replace(budget,max_boxes_per_pair=remaining),

                            cancellation_check=numeric_callback,_atlas=atlas)

                        used_boxes+=child.boxes;used_splits+=child.subdivisions

                        leaves.extend(child.tiles)

            curvature=replace(curvature,tiles=tuple(leaves),boxes=used_boxes,subdivisions=used_splits)

        curvatures.append((face_id,chart_map,curvature))

    if sweep_unions:
        # Rebuild only the feature-clearance cycles. Meshing retains all source
        # seams and per-member support/curvature/source-map records unchanged.
        grouped={f for union in sweep_unions for f in union.face_ids}
        feature_loops=[tuple((face,curve,edge) for curve,edge in loop)
            for face,support,read_loops,chart_map in inputs if face not in grouped for loop in read_loops]
        for union in sweep_unions:
            feature_loops.extend(tuple((face,numerical_operand(curve),edge) for face,curve,edge in loop)
                for loop in _sweep_boundary_occurrences(model,union))
        loops=[];curve_loops=[];curve_occurrences=[];all_straight=True
        for loop in feature_loops:
            curve_loops.append(tuple(curve for face,curve,edge in loop));segments=[]
            for face,curve,edge in loop:
                curve_occurrences.append((face,supports[face],curve,edge))
                try:segments.append(_segment(curve))
                except LocalApproximationNotCertified:all_straight=False
            loops.append(tuple(segments))
        boundary.guard()

    stars=()

    if all_straight:

        clearance=clearance_of_straight_loops(tuple(loops),check=check)

    else:

        from ._local_curve_stars import certify_curve_stars

        stars=certify_curve_stars(tuple(curve_loops),check=check)

        clearance=min(star.clearance_lower for star in stars)

    # Keep the conservative whole-face trim-feature ceiling. Inter-sheet

    # clearance is tile-local; boundary interval budgets are composed below.

    boundary_clearance=clearance

    from ._local_spatial_clearance import partition_surface_clearance

    spatial,contacts,spatial_work=partition_surface_clearance(model,supports,

        {row[0]:hull for row,hull in zip(inputs,bounds)},curvatures,

        connect_intent=connect_intent,check=numeric_callback,initial_work=sum(c.boxes for _,_,c in curvatures),contact_registry=contact_registry,numerical_boundary=boundary,span_atlases=span_atlases,plane_covers=plane_covers,sweep_unions=sweep_unions)

    face_tiles=[];ledger=[]

    for face,matrix,curvature in curvatures:

        rows=[]

        for box,upper,radius,gap,proof in spatial[face]:

            local_clearance=boundary_clearance if gap is None else min(boundary_clearance,_sqrt_down(gap))

            limit=min(ceiling,local_clearance/10)

            if radius is not None:limit=min(limit,alpha*radius)

            if limit<floor:raise LocalApproximationNotCertified('local geometry restriction is below numerical policy floor')

            rows.append((box,limit,upper));ledger.append((face,box,local_clearance,proof))

            clearance=min(clearance,local_clearance)

        face_tiles.append((face,matrix,tuple(rows)))

    boundary_tiles=[]

    if stars:

        from .prepared_material_curvilinear import _uv_interval

        for star in stars:

            face,support,curve,edge=curve_occurrences[star.curve_index]

            limit=min(ceiling,star.clearance_lower/10)

            if star.curvature_upper>0:limit=min(limit,alpha/star.curvature_upper)

            if limit<floor:raise LocalApproximationNotCertified('boundary restriction is below numerical policy floor')

            box=_uv_interval(support,curve,*star.inner)
            if id(support) in plane_covers:
                cover=plane_covers[id(support)]
                box=tuple(((a-lo)/(hi-lo),(b-lo)/(hi-lo)) for (a,b),(lo,hi) in zip(box,cover.chart_box))

            boundary_tiles.append((face,box,limit,(edge,definition_checksum(curve),star.inner)))

    check();validate_trimmed_surface_charts_binding(model,charts,cancellation_check=cancellation_check);check()

    result=LocalApproximationField(3,model.model_id,before[0],before[1],tuple(c.face.id for c in charts.charts),

        alpha,ceiling,floor,tuple(face_tiles),clearance,tuple(boundary_tiles),connect_intent,contacts,tuple(ledger),tuple(span_certificates),sweep_unions)

    checksum=definition_checksum(result)

    contact_runtime_check();contact_guard()  # No callbacks or public conversion after this gate.

    if _state(model)!=before:raise LocalApproximationNotCertified('local field source changed at issuance')

    if any(definition_checksum(atlas)!=pin for atlas,pin in atlas_pins):
        raise LocalApproximationNotCertified('local field span operands changed')
    boundary.finish()

    from ._local_piecewise_surface_star import require_span_context,span_runtime_signature
    require_span_context()
    from ._local_graph_clearance import runtime_signature as graph_runtime_signature
    _FIELDS[result]=(ref(model),before,charts,checksum,span_runtime_signature(),graph_runtime_signature())

    return result





def validate_local_approximation_field(model,result,*,cancellation_check=None):

    stored=_FIELDS.get(result) if type(result) is LocalApproximationField else None

    if stored is None or stored[0]() is not model:raise GeometryError('local field was not issued for source')
    from ._local_piecewise_surface_star import require_span_context,span_runtime_signature
    require_span_context()
    if span_runtime_signature()!=stored[4]:raise GeometryError('local field span runtime changed')
    from ._local_contact_stars import _check_contact_view_context
    _check_contact_view_context()
    from ._local_graph_clearance import runtime_signature as graph_runtime_signature
    if graph_runtime_signature()!=stored[5]:raise GeometryError('local field graph runtime changed')

    if cancellation_check is not None and cancellation_check('local field binding'):raise GeometryError('local field cancelled')

    validate_trimmed_surface_charts_binding(model,stored[2],cancellation_check=cancellation_check)

    from ._local_contact_stars import validate_local_contact_stars

    from ._local_effective_sweep import validate_effective_sweep_union
    for union in result.sweep_unions:validate_effective_sweep_union(model,union,cancellation_check=cancellation_check)
    for contact in result.contacts:validate_local_contact_stars(model,contact,cancellation_check=cancellation_check)

    for contact in result.contacts:validate_local_contact_stars(model,contact)
    for union in result.sweep_unions:validate_effective_sweep_union(model,union)

    _check_contact_view_context()
    if (_state(model)!=stored[1] or definition_checksum(result)!=stored[3]
            or span_runtime_signature()!=stored[4] or graph_runtime_signature()!=stored[5]):raise GeometryError('local field stale or changed')





def query_local_approximation_limit(model,field,face,*,cancellation_check=None):

    """Conservative infimum across WHOLE source face, hence every cell footprint."""

    validate_local_approximation_field(model,field,cancellation_check=cancellation_check)

    if field.contacts:raise GeometryError('conditional source budgets require actual prepared contact realization')

    return _planning_limit(field,face)





def _planning_limit(field,face):

    # Private proposal accessor; binding must realize contacts first.

    if hasattr(face,"model_id"):

        if face.model_id!=field.model_id or face.kind!="face":raise GeometryError("foreign local-field face")

        face=face.id

    rows=[rows for identifier,matrix,rows in field.face_tiles if identifier==face]

    if len(rows)!=1:raise GeometryError('source face absent from local field')

    return min([row[1] for row in rows[0]]+[limit for f,box,limit,identity in field.boundary_tiles if f==face])





def bind_local_approximation_field(source,prepared,field,regions,*,cancellation_check=None):

    from .local_field_transport import query_local_field_transport

    validate_local_approximation_field(source,field,cancellation_check=cancellation_check)

    transport=query_local_field_transport(source,prepared,regions,cancellation_check=cancellation_check)

    from .local_contact_transport import query_local_contact_transport

    contact_transport=transport

    if field.contacts:

        # Contact realization covers BOTH parents even for one-component numerical

        # generation. Source ancestry is authenticated, never inferred from IDs.

        from .local_field_transport import _partition

        from .material_regions import query_material_surface_regions

        _,_,mapping=_partition(source,prepared)

        parents={f for contact in field.contacts for f in contact.face_ids}

        descendants=tuple(sorted({child for parent,children in mapping if parent in parents for child in children}

            | {row.prepared_face.id for row in transport.footprints}))

        if not parents<={parent for parent,children in mapping if children}:

            raise GeometryError('prepared closure omits required contact parent')

        if set(descendants)!={row.prepared_face.id for row in transport.footprints}:

            contact_regions=query_material_surface_regions(prepared,descendants,cancellation_check=cancellation_check)

            contact_transport=query_local_field_transport(source,prepared,contact_regions,cancellation_check=cancellation_check)

    realized=tuple(query_local_contact_transport(source,prepared,contact,contact_transport,

        cancellation_check=cancellation_check) for contact in field.contacts)

    limits=tuple(sorted((row.prepared_face.id,_planning_limit(field,row.source_face.id)) for row in transport.footprints))

    fingerprint=(field.version,str(field.model_id),field.source_checksum,definition_checksum(field),definition_checksum(transport),tuple(definition_checksum(r) for r in realized))

    result=LocalApproximationAllocation(fingerprint,limits)

    _ALLOCATIONS[result]=(ref(source),ref(prepared),field,transport,definition_checksum(result),realized,contact_transport)

    validate_local_approximation_allocation(prepared,result,cancellation_check=cancellation_check)

    return result





def validate_local_approximation_allocation(prepared,result,*,cancellation_check=None):

    from .local_field_transport import validate_local_field_transport

    stored=_ALLOCATIONS.get(result) if type(result) is LocalApproximationAllocation else None

    if stored is None or stored[1]() is not prepared or stored[0]() is None:

        raise GeometryError('local allocation has no live owner authority')

    source=stored[0]()

    validate_local_approximation_field(source,stored[2],cancellation_check=cancellation_check)

    validate_local_field_transport(source,prepared,stored[3],cancellation_check=cancellation_check)

    from .local_contact_transport import validate_local_contact_transport

    for contact in stored[5]:validate_local_contact_transport(source,prepared,contact,cancellation_check=cancellation_check)

    for contact in stored[5]:validate_local_contact_transport(source,prepared,contact)

    validate_local_approximation_field(source,stored[2])

    validate_local_field_transport(source,prepared,stored[3])

    if stored[6] is not stored[3]:validate_local_field_transport(source,prepared,stored[6])

    if definition_checksum(result)!=stored[4]:raise GeometryError('local allocation changed')

    return result.fingerprint,result.face_limits





def local_approximation_field_fingerprint(field):

    """Owner-produced diagnostics/cache identity, never rehydrated authority."""

    stored=_FIELDS.get(field) if type(field) is LocalApproximationField else None

    if stored is None or stored[0]() is None:raise GeometryError('local field has no live source')

    validate_local_approximation_field(stored[0](),field)

    from ._local_contact_stars import local_contact_fingerprint

    contacts=tuple(local_contact_fingerprint(stored[0](),c) for c in field.contacts)

    return dict(version=field.version,definition_checksum=stored[3],source_checksum=field.source_checksum,

                source_model_id=str(field.model_id),clearance_face_ids=list(field.face_ids),

                contact_intent=field.contact_intent,conditional=bool(field.contacts),

                contact_pairs=[list(c.face_ids) for c in field.contacts],contacts=list(contacts),
                sweep_unions=[dict(face_ids=list(u.face_ids),seam_count=len(u.seams),
                    definition_checksum=definition_checksum(u)) for u in field.sweep_unions])





def prepare_local_cell_limit_query(prepared,allocation,*,_boundary=False):

    """Entry-validated immutable numerical view; caller owes callback/final guards."""

    validate_local_approximation_allocation(prepared,allocation)

    stored=_ALLOCATIONS[allocation]

    field,transport=stored[2],stored[6] if _boundary else stored[3]

    # All nested values below are immutable Fractions/ints/tuples. No live

    # dataclass/ndarray or mutable curve/kernel is closed over by the query.

    table={face:(tuple(tuple(Q(x) for x in row) for row in matrix),

                 tuple((tuple(tuple(Q(x) for x in pair) for pair in box),Q(limit))

                       for box,limit,curvature in rows))

           for face,matrix,rows in field.face_tiles}

    boundaries=tuple((face,tuple(tuple(Q(x) for x in pair) for pair in box),Q(limit))

        for face,box,limit,identity in field.boundary_tiles)

    maps={row.prepared_face.id:(row.source_face.id,

        tuple(tuple(Q(x) for x in part) for part in row.source_from_child)) for row in transport.footprints}

    def affine(matrix,box):

        return tuple((min(a*u+b*v+c for u in box[0] for v in box[1]),

                      max(a*u+b*v+c for u in box[0] for v in box[1])) for a,b,c in matrix)

    def inverse(matrix):

        (a,b,c),(d,e,f)=matrix;det=a*e-b*d

        if det==0:raise GeometryError('local field chart map is singular')

        return ((e/det,-b/det,(b*f-e*c)/det),(-d/det,a/det,(d*c-a*f)/det))

    def query(face,footprints):

        if face not in maps:raise GeometryError('local cell face outside issued transport')

        parent,to_authored=maps[face];from_native,tiles=table[parent]

        to_native=inverse(from_native)

        boxes=tuple(tuple(tuple(Q(x) for x in pair) for pair in box) for box in footprints)

        if not boxes or any(len(box)!=2 or any(len(pair)!=2 or pair[0]>pair[1] for pair in box) for box in boxes):

            raise GeometryError('local cell footprint enclosure malformed')

        mapped=tuple(affine(to_native,affine(to_authored,box)) for box in boxes)

        local_boundaries=tuple((box,limit) for f,box,limit in boundaries if f==parent)

        all_limit=min([limit for box,limit in tiles]+[limit for box,limit in local_boundaries])

        # An over-wide interval can exceed the chart despite a valid map.

        # Full-face infimum is still safe; later source-domain proof is required.

        if any(any(a<0 or b>1 for a,b in box) for box in mapped):return all_limit,('whole_face_outside_enclosure',mapped)

        selected=[]

        for box in mapped:

            overlaps=[limit for tile,limit in tiles if all(a<=d and c<=b for (a,b),(c,d) in zip(box,tile))]

            if not overlaps:raise GeometryError('local field footprint coverage is missing')

            selected.extend(overlaps)

            selected.extend(limit for tile,limit in local_boundaries if all(a<=d and c<=b for (a,b),(c,d) in zip(box,tile)))

        return min(selected),('intersected_tiles',mapped)

    return query





def prepare_local_boundary_limit_query(prepared,allocation):

    """Whole actual boundary interval, minimum over every issued incident face."""

    query=prepare_local_cell_limit_query(prepared,allocation,_boundary=True)

    transport=_ALLOCATIONS[allocation][6]

    charts={chart.face.id:chart for region in transport.regions.regions for chart in region.sources}

    def boundary(edge,a,b):

        from .arrangement_geometry import freeze_edge

        from .prepared_material_curvilinear import _uv_interval

        from .nurbs_sweep_chart import has_sweep_chart,chart_model

        values=[];records=[]

        incident=tuple(prepared.faces_using_edge(edge))

        if any(face not in charts for face in incident):

            raise GeometryError('local boundary budget omits an incident prepared face restriction')

        for face in incident:

            curve=(chart_model(prepared,face).edges[edge].curve if has_sweep_chart(prepared.faces[face])

                   else freeze_edge(prepared,edge))

            box=_uv_interval(charts[face].support,curve,a,b)

            value,record=query(face,(box,));values.append(value);records.append((face,record))

        if not values:raise GeometryError('local boundary has no issued incident face')

        return min(values),tuple(records)

    return boundary





def local_boundary_corner_parameters(curve):

    """Authored C0 rational knots must be literal mesh stations in local mode."""

    from .nurbs import NurbsCurve,CurveOnSurface

    from collections import Counter

    pc=curve.pcurve if type(curve) is CurveOnSurface else curve

    if type(pc) is not NurbsCurve:return ()

    a,b=map(Q,pc.domain)

    if a==b:raise GeometryError('collapsed local boundary parameter domain')

    values=[]

    for knot,multiplicity in Counter(map(Q,pc.knots)).items():

        t=(knot-a)/(b-a)

        if 0<t<1 and multiplicity>=pc.degree:

            if multiplicity>pc.degree:raise GeometryError('discontinuous local trim knot')

            if Q(float(t))!=t:raise GeometryError('local trim corner has no exact stored binary64 station')

            values.append(t)

    return tuple(sorted(values))

