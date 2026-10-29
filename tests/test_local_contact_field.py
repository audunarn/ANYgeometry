"""Real NN conditional source planning -> prepared allocation, never a mesh pass."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.local_approximation_field import (query_local_approximation_field,
    query_local_approximation_limit,bind_local_approximation_field,
    validate_local_approximation_field,validate_local_approximation_allocation,
    prepare_local_cell_limit_query,prepare_local_boundary_limit_query,
    local_approximation_field_fingerprint)
from anygeometry.local_field_transport import query_local_field_transport
from anygeometry.nurbs_intersection_curve import intersection_curve_operation,NurbsSurfaceIntersectionCurve
from test_nurbs_pair_public import model_pair


@pytest.fixture(scope='module')
def conditional():
    source,parents=model_pair()
    for face in parents:source.add_sheet((face,))
    field=query_local_approximation_field(source,parents,relative_accuracy=.001,
        scale_error=.01,policy_floor=1e-9,connect_intent=True)
    return source,parents,field


@pytest.fixture(scope='module')
def prepared_pair(conditional):
    source,parents,field=conditional
    prepared=owner.clone_prepared_geometry(source)
    with intersection_curve_operation():
        plan=owner.plan_intersections(prepared,parents,policy='connect')
        owner.apply_intersections(prepared,plan,policy='connect')
    regions=owner.query_material_surface_regions(prepared,tuple(prepared.faces))
    transport=query_local_field_transport(source,prepared,regions)
    return source,prepared,parents,field,regions,transport


def test_complete_local_source_partition_is_conditional(conditional):
    source,parents,field=conditional
    assert field.contacts and field.contact_intent
    identity=local_approximation_field_fingerprint(field)
    assert identity['conditional'] and identity['contact_pairs']==[list(parents)]
    for face,matrix,tiles in field.face_tiles:
        assert sum((box[0][1]-box[0][0])*(box[1][1]-box[1][0]) for box,_,_ in tiles)==1
        assert all(0<limit<=field.scale_error for _,limit,_ in tiles)
    assert any(kind=='conditional_contact' for _,_,_,proof in field.spatial_ledger for _,kind,_ in proof)
    assert all(len(proof)==1 for _,_,_,proof in field.spatial_ledger)
    for _,_,_,proof in field.spatial_ledger:
        for _,kind,record in proof:
            if kind=='conditional_contact':
                assert record.requires_realization and record.retained_partner_native_rectangles
    with pytest.raises(owner.GeometryError,match='realization'):
        query_local_approximation_limit(source,field,parents[0])
    clone=owner.clone_prepared_geometry(source)
    regions=owner.query_material_surface_regions(clone,parents)
    with pytest.raises(owner.GeometryError):bind_local_approximation_field(source,clone,field,regions)


def test_actual_connect_realizes_before_numerical_queries(prepared_pair):
    source,prepared,parents,field,regions,_=prepared_pair
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    validate_local_approximation_allocation(prepared,allocation)
    query=prepare_local_cell_limit_query(prepared,allocation)
    for face in prepared.faces:
        limit,_=query(face,(((0,1),(0,1)),))
        assert 0<limit<=field.scale_error


def test_one_parent_generation_keeps_all_shared_boundary_restrictions(prepared_pair):
    source,prepared,parents,field,_,transport=prepared_pair
    selected=tuple(r.prepared_face.id for r in transport.footprints if r.source_face.id==parents[0])
    regions=owner.query_material_surface_regions(prepared,selected)
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    assert {face for face,_ in allocation.face_limits}==set(selected)
    cell=prepare_local_cell_limit_query(prepared,allocation)
    foreign=next(f for f in prepared.faces if f not in selected)
    with pytest.raises(owner.GeometryError):cell(foreign,(((0,1),(0,1)),))
    boundary=prepare_local_boundary_limit_query(prepared,allocation)
    all_faces=prepare_local_cell_limit_query(prepared,allocation,_boundary=True)
    from anygeometry.prepared_material_curvilinear import _uv_interval
    charts={c.face.id:c for r in transport.regions.regions for c in r.sources}
    differences=[]
    for edge in prepared.edges.values():
        if type(edge.curve) is not NurbsSurfaceIntersectionCurve:continue
        value,records=boundary(edge.id,Q(0),Q(1))
        assert len(records)==4
        values={face:all_faces(face,(_uv_interval(charts[face].support,edge.curve.for_support(charts[face].support),0,1),))[0]
                for face in prepared.faces_using_edge(edge.id)}
        assert value==min(values.values())
        differences.append(len(set(values.values()))>1)
    assert any(differences), 'fixture must exercise a genuinely tighter incident-parent restriction'


@pytest.mark.parametrize('target',('intent','contacts','source'))
def test_conditional_binding_mutation_refused(conditional,target):
    source,parents,field=conditional
    obj,name=(source.faces[parents[0]].surface,'controls') if target=='source' else (field,'contact_intent' if target=='intent' else 'contacts')
    old=getattr(obj,name)
    bad=tuple(tuple((p[0],p[1],p[2]+1) for p in row) for row in old) if target=='source' else (False if target=='intent' else ())
    def mutate(phase):object.__setattr__(obj,name,bad)
    try:
        with pytest.raises(owner.GeometryError):validate_local_approximation_field(source,field,cancellation_check=mutate)
    finally:object.__setattr__(obj,name,old)


def test_proposal_input_conversion_cannot_mutate_source(conditional):
    from anygeometry._local_contact_stars import query_local_contact_proposal
    source,parents,field=conditional
    receipt=field.contacts[0];old=receipt.intent
    class Box:
        def __iter__(self):
            object.__setattr__(receipt,'intent','IMPRINT')
            return iter(((0,1),(0,1)))
    try:
        with pytest.raises(owner.GeometryError):query_local_contact_proposal(source,receipt,parents[0],Box())
    finally:object.__setattr__(receipt,'intent',old)


def test_spatial_clearance_retains_third_sheet_and_varies_on_flat_source():
    from anygeometry.nurbs import NurbsSurface
    from anygeometry._local_spatial_clearance import partition_surface_clearance
    from anygeometry._local_curvature import bound_surface_curvature
    from anygeometry._local_neighborhood import projected_hull
    from anygeometry.nurbs_surface_intersection._bernstein import bezier_piece
    knots=(0.,0.,1.,1.)
    supports={i:NurbsSurface(1,1,knots,knots,tuple(tuple((float(u),float(v),float(z(u))) for v in range(2)) for u in range(2)))
              for i,z in enumerate((lambda u:0,lambda u:1+u,lambda u:4))}
    hulls={i:projected_hull(bezier_piece(s,s.exact_net(),(Q(0),Q(1)),(Q(0),Q(1)))) for i,s in supports.items()}
    curvatures=tuple((i,(),bound_surface_curvature(s)) for i,s in supports.items())
    rows,contacts,_=partition_surface_clearance(None,supports,hulls,curvatures,connect_intent=False,check=lambda:None)
    assert contacts==()
    assert len({row[3] for row in rows[0]})>1
    assert all({other for other,_,_ in row[4]}==({0,1,2}-{face}) for face,tiles in rows.items() for row in tiles)
    assert all(sum((r[0][0][1]-r[0][0][0])*(r[0][1][1]-r[0][1][0]) for r in tiles)==1 for tiles in rows.values())



def test_no_intent_never_qualifies_a_proximity_contact(monkeypatch):
    from dataclasses import replace
    import anygeometry._local_spatial_clearance as spatial
    from anygeometry._local_curvature import bound_surface_curvature,LocalApproximationNotCertified
    from anygeometry._local_neighborhood import projected_hull
    from anygeometry.nurbs_surface_intersection._bernstein import bezier_piece
    source,faces=model_pair()
    supports={f:source.faces[f].surface for f in faces}
    hulls={f:projected_hull(bezier_piece(s,s.exact_net(),(Q(0),Q(1)),(Q(0),Q(1)))) for f,s in supports.items()}
    curvatures=tuple((f,(),bound_surface_curvature(s)) for f,s in supports.items())
    def forbidden(*args,**kwargs):raise AssertionError('proximity must not invent CONNECT intent')
    monkeypatch.setattr(spatial,'query_local_contact_stars',forbidden)
    monkeypatch.setattr(spatial,'DEFAULT_INTERSECTION_QUALIFICATION_POLICY',
        replace(spatial.DEFAULT_INTERSECTION_QUALIFICATION_POLICY,max_boxes_per_pair=32,max_subdivision_depth=2))
    with pytest.raises(LocalApproximationNotCertified):
        spatial.partition_surface_clearance(source,supports,hulls,curvatures,
            connect_intent=False,check=lambda:None)



def _search_fixture():
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.nurbs_surface_intersection._intervals import Interval as I
    from anygeometry._local_spatial_clearance import _hull,SpatialBudget
    surface=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,1.),(0.,1.,1.)),((1.,0.,2.),(1.,1.,2.))))
    unit=((Q(0),Q(1)),(Q(0),Q(1)))
    full=_hull(surface,unit,None,SpatialBudget(lambda:None))
    inner=(I(Q(0),Q(1,8)),I(Q(0),Q(1,8)),I(Q(0)))
    return surface,unit,full,inner


def test_retained_search_matches_exhaustive_bound_and_cover():
    import anygeometry._local_spatial_clearance as spatial
    from anygeometry._local_neighborhood import separation_squared
    surface,unit,full,inner=_search_fixture()
    budget=spatial.SpatialBudget(lambda:None)
    queue=[(unit,0)];reference=[];depth=6
    while queue:
        box,level=queue.pop();hull,directions=spatial._hull_data(surface,box,full,budget)
        gap=spatial._gap(inner,hull,directions)
        coarse=max(v.hi-v.lo for v in hull)>max(v.hi-v.lo for v in inner)
        if gap>0 and (not coarse or level>=depth):reference.append((box,gap));continue
        assert level<depth
        queue.extend((child,level+1) for child in spatial._split(box))
    actual=spatial._ordinary(inner,surface,full,spatial.SpatialBudget(lambda:None),depth,spatial._HullMemo())
    assert actual[0]==min(gap for _,gap in reference)
    assert sum((b[0][1]-b[0][0])*(b[1][1]-b[1][0]) for b,_ in actual[1])==1
    assert all(gap>=actual[0]>0 for _,gap in actual[1])
    assert len(actual[1])<=len(reference)


def test_hull_memo_keeps_exact_values_fresh_objects_and_charged_guards():
    import anygeometry._local_spatial_clearance as spatial
    surface,unit,full,_=_search_fixture();calls=[]
    budget=spatial.SpatialBudget(lambda:calls.append(True));memo=spatial._HullMemo()
    first=spatial._hull(surface,unit,full,budget,memo)
    expected=tuple((v.lo,v.hi) for v in first)
    assert len(memo.rows)==1
    object.__setattr__(first[0],'lo',Q(-100))  # Adversarial mutation cannot corrupt memo data.
    second=spatial._hull(surface,unit,full,budget,memo)
    assert tuple((v.lo,v.hi) for v in second)==expected
    assert budget.used==len(calls)==2 and second[0] is not first[0]
    old=surface.controls
    try:
        object.__setattr__(surface,'controls',tuple(tuple((x,y,z+1) for x,y,z in row) for row in old))
        changed=spatial._hull(surface,unit,full,budget,memo)
        assert changed[2].lo==expected[2][0]+1
    finally:object.__setattr__(surface,'controls',old)


def test_hull_memo_unknown_runtime_falls_back_and_cancel_still_propagates(monkeypatch):
    import anygeometry._local_spatial_clearance as spatial
    surface,unit,full,_=_search_fixture();memo=spatial._HullMemo();calls=[]
    original=spatial.bezier_piece
    def changed(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(spatial,'bezier_piece',changed)
        for _ in range(2):spatial._hull(surface,unit,full,spatial.SpatialBudget(lambda:None),memo)
        assert len(calls)==2 and memo.rows=={}
    with monkeypatch.context() as patch:
        patch.setattr(original,'__defaults__',(None,))
        spatial._hull(surface,unit,full,spatial.SpatialBudget(lambda:None),memo)
        assert memo.rows=={}
    budget=spatial.SpatialBudget(lambda:None)
    spatial._hull(surface,unit,full,budget,memo)
    class Stop(Exception):pass
    def stop():raise Stop('unchanged cancellation exception')
    budget.check=stop
    with pytest.raises(Stop):spatial._hull(surface,unit,full,budget,memo)


@pytest.mark.parametrize('weight_scale',(1.,7.))
def test_directional_slope_has_independent_nine_eighths_bound(weight_scale):
    import anygeometry._local_spatial_clearance as spatial
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.nurbs_surface_intersection._intervals import Interval as I
    from anygeometry._local_neighborhood import separation_squared
    surface,unit,full,_=_search_fixture()
    # Nonuniform rational weights; multiplying ALL weights preserves the map.
    weighted=NurbsSurface(surface.degree_u,surface.degree_v,surface.knots_u,surface.knots_v,
        surface.controls,weights=((weight_scale,2*weight_scale),(3*weight_scale,4*weight_scale)))
    inner=(I(Q(1,2),Q(1)),I(Q(0),Q(1)),I(Q(0)))
    hull,directions=spatial._hull_data(weighted,unit,full,spatial.SpatialBudget(lambda:None),spatial._HullMemo())
    assert separation_squared(inner,hull)==1
    # q=(-1,0,1), competitor projection=1, source max=-1/2:
    # (3/2)^2 / ||q||^2 = 9/8. No reference to the search implementation.
    assert spatial._gap(inner,hull,directions)==Q(9,8)


@pytest.mark.parametrize('last_height,expected',((0.,Q(0)),(4.,Q(9,8))))
def test_directional_projection_encloses_every_nonplanar_control(last_height,expected):
    import anygeometry._local_spatial_clearance as spatial
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.nurbs_surface_intersection._intervals import Interval as I
    surface,unit,full,_=_search_fixture()
    controls=(surface.controls[0],(surface.controls[1][0],(1.,1.,last_height)))
    nonplanar=NurbsSurface(1,1,surface.knots_u,surface.knots_v,controls)
    hull,directions=spatial._hull_data(nonplanar,unit,full,spatial.SpatialBudget(lambda:None))
    inner=(I(Q(1,2),Q(1)),I(Q(0),Q(1)),I(Q(0)))
    # The fourth point leaves the first-three-point plane. Its projection must
    # participate: the low case actually touches the source at (1,1,0).
    assert spatial._gap(inner,hull,directions)==expected


def test_degenerate_direction_and_nonpositive_weight_never_add_separation():
    import anygeometry._local_spatial_clearance as spatial
    grid=tuple(tuple((Q(2*i+j),Q(0),Q(1),Q(1)) for j in range(2)) for i in range(2))
    assert spatial._directions(grid)==()
    bad=((grid[0][0],(Q(1),Q(0),Q(1),Q(0))),grid[1])
    with pytest.raises(owner.GeometryError,match='positive'):spatial._directions(bad)


def test_field_frontier_keeps_full_none_check_and_both_callback_checks(monkeypatch):
    import anygeometry.local_approximation_field as field_api
    source,_=model_pair();original=field_api._state;before=original(source);calls=[]
    def observe(model):calls.append(True);return original(model)
    monkeypatch.setattr(field_api,'_state',observe)
    field_api._field_frontier(source,before,None)
    assert len(calls)==1
    calls.clear()
    field_api._field_frontier(source,before,lambda phase:False)
    assert len(calls)==2
    surface=next(iter(source.faces.values())).surface;old=surface.controls
    def mutate(phase):
        object.__setattr__(surface,'controls',tuple(tuple((x,y,z+1) for x,y,z in row) for row in old))
    try:
        with pytest.raises(owner.GeometryError,match='changed'):field_api._field_frontier(source,before,mutate)
        with pytest.raises(owner.GeometryError,match='changed'):field_api._field_frontier(source,before,None)
    finally:object.__setattr__(surface,'controls',old)


def test_field_final_issuance_refuses_contact_mutation_after_chart_validation(monkeypatch):
    """Real field construction; late mutation may not publish an issued field."""
    import anygeometry.local_approximation_field as field_api
    import anygeometry._local_contact_stars as contact
    source,parents=model_pair()
    for face in parents:source.add_sheet((face,))
    observed=[]
    original_validate=field_api.validate_trimmed_surface_charts_binding
    def late_mutation(*args,**kwargs):
        result=original_validate(*args,**kwargs)
        observed.extend(receipt for receipt,row in contact._ISSUED.items() if row[0]() is source)
        if observed:object.__setattr__(observed[0],'intent','IMPRINT')
        return result
    monkeypatch.setattr(field_api,'validate_trimmed_surface_charts_binding',late_mutation)
    before=set(field_api._FIELDS)
    with pytest.raises(owner.GeometryError,match='changed'):
        field_api.query_local_approximation_field(source,parents,relative_accuracy=.001,
            scale_error=.01,policy_floor=1e-9,connect_intent=True)
    assert observed and set(field_api._FIELDS)==before
