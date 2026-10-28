"""Source union and finite window premises; not full CONNECT acceptance."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryModel,GeometryError,Plane
from anygeometry.nurbs import NurbsCurve
from anygeometry._local_effective_sweep import query_effective_sweep_union,validate_effective_sweep_union
from anygeometry.local_sweep_plane_contact import certify_sweep_plane_window,validate_sweep_plane_window


def fixture():
    model=GeometryModel()
    curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.,0.),(0.,.5,1.),(0.,1.,0.)))
    points=model.add_points(tuple(tuple(float(x) for x in curve.exact_point(t)) for t in (Q(0),Q(1,2),Q(1))))
    edges=tuple(model.add_curve(a,b,replace(curve,domain=domain))
        for a,b,domain in zip(points,points[1:],((0.,.5),(.5,1.))))
    faces=tuple(model.extrude(edges,(5.,0.,0.)))
    model.add_sheet(faces)
    corners=model.add_points(((2.5,-2.,-2.),(2.5,2.,-2.),(2.5,2.,2.),(2.5,-2.,2.)))
    boundary=tuple(model.add_line(a,b) for a,b in zip(corners,corners[1:]+corners[:1]))
    plane=model.add_face(boundary,surface=Plane((2.5,0.,0.),(0.,1.,0.),(0.,0.,1.)))
    model.add_sheet((plane,))
    return model,faces,plane


def window(model,union,plane,**kwargs):
    return certify_sweep_plane_window(model,union,plane,(Q(1,4),Q(3,4)),
        ((0,1,0),(Q(1,5),0,1)),((Q(9,20),Q(11,20)),(Q(9,10),Q(1))),
        ((Q(3,8),Q(5,8)),(Q(4,5),Q(11,10))),**kwargs)


@pytest.fixture(scope='module')
def issued():
    model,faces,plane=fixture()
    union=query_effective_sweep_union(model,faces)
    result=window(model,union,plane)
    return model,faces,plane,union,result


def test_actual_shared_sweep_seam_and_source_plane_window(issued):
    model,faces,plane,union,result=issued
    validate_effective_sweep_union(model,union)
    validate_sweep_plane_window(model,result)
    assert union.face_ids==faces and len(union.seams)==1
    assert union.seams[0][1:3]==faces
    assert set(model.faces_using_edge(union.seams[0][0]))==set(faces)
    assert tuple(m.profile_interval for m in union.members)==((Q(0),Q(1,2)),(Q(1,2),Q(1)))
    assert union.transport_upper==0
    assert result.projection_margin>0 and result.sweep_containment_margin>0
    assert result.plane_containment_margin>0 and result.exterior_squared_lower>0
    assert result.retained_profile_intervals==((Q(0),Q(1,4)),(Q(3,4),Q(1)))
    assert result.requires_realization  # This is NOT a clearance or CONNECT receipt.
    matrix,offset=result.trace_law
    # Here the plane-intersection affine law is independently x=5/2,
    # y and z unchanged, for every profile point (including rational ones).
    assert matrix==((Q(0),Q(0),Q(0)),(Q(0),Q(1),Q(0)),(Q(0),Q(0),Q(1)))
    assert offset==(Q(5,2),Q(0),Q(0))


@pytest.mark.parametrize('kind',('duplicate','omitted'))
def test_source_union_refuses_incomplete_original_incidence(kind):
    model,faces,_=fixture()
    selection=(faces[0],faces[0]) if kind=='duplicate' else (faces[0],)
    with pytest.raises(GeometryError):query_effective_sweep_union(model,selection)


def test_source_and_window_copies_have_no_authority(issued):
    model,_,_,union,result=issued
    with pytest.raises(GeometryError):validate_effective_sweep_union(model,replace(union))
    with pytest.raises(GeometryError):validate_sweep_plane_window(model,replace(result))


@pytest.mark.parametrize('change',('plane','receipt','union'))
def test_late_callback_mutation_refused(issued,change):
    model,_,plane,union,result=issued
    target,name,bad=(model.faces[plane],'metadata',{'changed':True}) if change=='plane' else (
        (result,'retained_profile_intervals',()) if change=='receipt' else (union,'seams',()))
    old=getattr(target,name)
    def mutate(phase):object.__setattr__(target,name,bad)
    try:
        with pytest.raises(GeometryError):validate_sweep_plane_window(model,result,cancellation_check=mutate)
    finally:object.__setattr__(target,name,old)
    validate_sweep_plane_window(model,result)


def test_callback_identity_and_truthy_cancellation(issued):
    model,_,plane,union,_=issued
    token=GeometryError('external exact token')
    def stop(phase):raise token
    with pytest.raises(GeometryError) as caught:window(model,union,plane,cancellation_check=stop)
    assert caught.value is token
    with pytest.raises(GeometryError,match='cancelled'):window(model,union,plane,cancellation_check=lambda phase:True)


def test_rank_one_projection_and_material_exit_are_not_contact(issued):
    model,_,plane,union,_=issued
    with pytest.raises(GeometryError):
        certify_sweep_plane_window(model,union,plane,(Q(1,4),Q(3,4)),
            ((0,1,0),(Q(1,5),0,0)),((Q(9,20),Q(11,20)),(Q(2,5),Q(3,5))),
            ((Q(3,8),Q(5,8)),(Q(1,4),Q(3,4))))
    with pytest.raises(GeometryError,match='swept material'):
        certify_sweep_plane_window(model,union,plane,(Q(1,4),Q(3,4)),
            ((0,1,0),(Q(1,5),0,1)),((Q(9,20),Q(11,20)),(Q(9,10),Q(1))),
            ((Q(3,8),Q(5,8)),(Q(0),Q(2))))


def test_raw_sweep_material_query_retains_original_source():
    from anygeometry.trimmed_charts import query_trimmed_surface_charts,validate_trimmed_surface_charts_binding
    from anygeometry.serialization import to_dict
    from anygeometry.nurbs import NurbsSurface,CurveOnSurface
    from anygeometry.nurbs_sweep_chart import RECIPE
    from anygeometry.local_approximation_field import _sweep_boundary_occurrences
    model,faces,_=fixture();before=to_dict(model)
    charts=query_trimmed_surface_charts(model,faces)
    validate_trimmed_surface_charts_binding(model,charts)
    assert all(type(c.support) is NurbsSurface for c in charts.charts)
    assert all(type(path.curve) is CurveOnSurface for c in charts.charts for loop in c.boundaries for path in loop)
    union=query_effective_sweep_union(model,faces)
    loops=_sweep_boundary_occurrences(model,union)
    assert len(loops)==1 and len(loops[0])==6
    seam=union.seams[0][0]
    assert seam not in [edge for loop in loops for face,curve,edge in loop]
    assert seam in model.edges and set(model.faces_using_edge(seam))==set(faces)
    assert all(RECIPE not in model.faces[face].metadata and model.faces[face].parameterization is None for face in faces)
    assert to_dict(model)==before


def test_raw_sweep_source_change_and_fake_recipe_refuse():
    from anygeometry.trimmed_charts import query_trimmed_surface_charts
    from anygeometry.nurbs_sweep_chart import source_sweep_chart_model,RECIPE,_tensor
    from anygeometry._translational_pairs import _sweep
    model,faces,_=fixture();face=model.faces[faces[0]]
    profile,vector,_=_sweep(model,faces[0])
    old_parameterization,old_metadata=face.parameterization,face.metadata
    try:
        object.__setattr__(face,'parameterization',_tensor(profile,vector))
        object.__setattr__(face,'metadata',{RECIPE:{'fake':'not source authority'}})
        with pytest.raises(GeometryError):query_trimmed_surface_charts(model,faces)
    finally:
        object.__setattr__(face,'parameterization',old_parameterization)
        object.__setattr__(face,'metadata',old_metadata)
    # A nontranslation Coons chart must not inherit the exact sweep adapter.
    top=model.edges[face.loop[2].edge].curve;original=top.controls
    changed=list(original);x,y,z=changed[1];changed[1]=(x,y,z+.125)
    try:
        object.__setattr__(top,'controls',tuple(changed))
        assert source_sweep_chart_model(model,faces[0]) is None
        with pytest.raises(GeometryError):query_trimmed_surface_charts(model,faces)
    finally:object.__setattr__(top,'controls',original)


def test_raw_sweep_chart_callback_refusal_preserves_source():
    from anygeometry.trimmed_charts import query_trimmed_surface_charts
    from anygeometry.serialization import to_dict
    model,faces,_=fixture();before=to_dict(model)
    token=GeometryError('external chart callback token')
    def stop(phase):raise token
    with pytest.raises(GeometryError) as caught:query_trimmed_surface_charts(model,faces,cancellation_check=stop)
    assert caught.value is token
    with pytest.raises(GeometryError):query_trimmed_surface_charts(model,faces,cancellation_check=lambda phase:True)
    assert to_dict(model)==before



def _controlled_window_scope(model,union,callback=None):
    from anygeometry._detached_geometry_read import NumericalBoundary
    from anygeometry._local_spatial_clearance import SpatialBudget
    from anygeometry import local_sweep_plane_contact as contact
    from anygeometry.definition_binding import definition_checksum
    pin=definition_checksum(union);runtime=contact._runtime()
    def guard():
        validate_effective_sweep_union(model,union)
        if definition_checksum(union)!=pin or contact._runtime()!=runtime:
            raise GeometryError('window operand changed')
    boundary=NumericalBoundary(guard,callback,source=model)
    return boundary,SpatialBudget(boundary.tick)


def test_controlled_window_ticks_do_not_requalify_global_state(issued):
    import sys
    model,_,plane,union,_=issued
    boundary,budget=_controlled_window_scope(model,union)
    counts={'ticks':0,'internal_guards':0}
    def profile(frame,event,arg):
        if event!='call':return
        name=frame.f_code.co_qualname
        if name=='certify_sweep_plane_window.<locals>.check':counts['ticks']+=1
        if name=='certify_sweep_plane_window.<locals>.guard':
            if frame.f_back.f_code.co_name=='check':counts['internal_guards']+=1
    old=sys.getprofile()
    try:
        sys.setprofile(profile)
        result=window(model,union,plane,cancellation_check=budget.take,_numerical_boundary=boundary)
    finally:
        sys.setprofile(old);boundary.close()
    assert counts['ticks']>2 and budget.used>=counts['ticks']
    assert counts['internal_guards']==0
    validate_sweep_plane_window(model,result)


@pytest.mark.parametrize('mutation',('source','receipt','runtime','exception'))
def test_controlled_window_external_callback_frontiers(issued,mutation):
    from anygeometry import local_sweep_plane_contact as contact
    model,faces,plane,union,_=issued
    face=model.faces[faces[0]];metadata=face.metadata
    allowance=union.transport_upper;defaults=contact.I.scale.__defaults__
    token=GeometryError('original callback token');calls=[]
    def callback(phase):
        calls.append(phase)
        if mutation=='source':object.__setattr__(face,'metadata',dict(metadata,callback_mutated=True))
        elif mutation=='receipt':object.__setattr__(union,'transport_upper',allowance+1)
        elif mutation=='runtime':contact.I.scale.__defaults__=(object(),)
        else:raise token
        return False
    boundary,budget=_controlled_window_scope(model,union,callback)
    try:
        with pytest.raises(GeometryError) as caught:
            window(model,union,plane,cancellation_check=budget.take,_numerical_boundary=boundary)
        assert calls
        if mutation=='exception':assert caught.value is token
    finally:
        boundary.close()
        object.__setattr__(face,'metadata',metadata)
        object.__setattr__(union,'transport_upper',allowance)
        contact.I.scale.__defaults__=defaults



def _window_operation(issued,callback=None):
    from anygeometry import local_sweep_plane_contact as contact
    from anygeometry._detached_geometry_read import NumericalBoundary
    from anygeometry._local_spatial_clearance import SpatialBudget
    model,_,plane,union,_=issued;holder=[]
    def guard():
        validate_effective_sweep_union(model,union)
        if holder:holder[0].guard()
    boundary=NumericalBoundary(guard,callback,source=model)
    budget=SpatialBudget(boundary.tick)
    operation=contact._SweepCompetition(model,(union,),(plane,),None,budget,boundary)
    holder.append(operation)
    return operation,boundary


def _operation_window(operation,issued):
    model,_,plane,union,expected=issued
    return operation._make_window(union,plane,expected.profile_interval,expected.projection,
        expected.inner_box,expected.outer_box)


def test_operation_window_exact_public_equivalence_and_atomic_issuance(issued):
    from anygeometry.definition_binding import definition_checksum
    from anygeometry import local_sweep_plane_contact as contact
    operation,boundary=_window_operation(issued)
    try:
        actual=_operation_window(operation,issued)
        assert definition_checksum(actual)==definition_checksum(issued[-1])
        assert actual not in contact._WINDOWS
        before=dict(operation.curve_bounds)
        again=_operation_window(operation,issued)
        assert definition_checksum(again)==definition_checksum(actual)
        assert operation.curve_bounds.keys()==before.keys()
        assert all(operation.curve_bounds[k] is value for k,value in before.items())
        operation.finish()
        validate_sweep_plane_window(issued[0],actual)
        validate_sweep_plane_window(issued[0],again)
    finally:boundary.close()


@pytest.mark.parametrize('mutation',('source','proposal','cached_operand'))
def test_operation_failure_never_publishes_partial_windows(issued,mutation):
    from anygeometry import local_sweep_plane_contact as contact
    operation,boundary=_window_operation(issued)
    model,faces,_,_,_=issued;face=model.faces[faces[0]];metadata=face.metadata
    value=_operation_window(operation,issued)
    original=value.exterior_squared_lower
    key=next(iter(operation.curve_bounds));old_bound=operation.curve_bounds[key]
    try:
        if mutation=='source':object.__setattr__(face,'metadata',dict(metadata,changed=True))
        elif mutation=='proposal':object.__setattr__(value,'exterior_squared_lower',original+1)
        else:operation.curve_bounds[key]=None
        with pytest.raises(GeometryError):operation.finish()
        assert value not in contact._WINDOWS
    finally:
        boundary.close();object.__setattr__(face,'metadata',metadata)
        object.__setattr__(value,'exterior_squared_lower',original)
        operation.curve_bounds[key]=old_bound


@pytest.mark.parametrize('stop',('callback','budget','proposal_callback'))
def test_operation_ticks_preserve_cancellation_work_and_proposal_frontiers(issued,stop):
    from anygeometry import local_sweep_plane_contact as contact
    operation,boundary=_window_operation(issued)
    value=_operation_window(operation,issued);old=value.exterior_squared_lower
    try:
        if stop=='callback':boundary._callback=lambda phase:True
        elif stop=='budget':operation.budget.limit=operation.budget.used
        else:
            def mutate(phase):
                object.__setattr__(value,'exterior_squared_lower',old+1)
                return False
            boundary._callback=mutate
        with pytest.raises(GeometryError):_operation_window(operation,issued)
        assert all(window not in contact._WINDOWS for window in operation.pending)
    finally:
        boundary.close();object.__setattr__(value,'exterior_squared_lower',old)



def _actual_partition_operation(issued,kind):
    from itertools import product
    from anygeometry.prepared_material_source_transfer import _effective
    from anygeometry.curvilinear_reference_maps import _source_box
    from anygeometry.nurbs_surface_intersection._intervals import Interval as I
    model,faces,plane,union,_=issued
    operation,boundary=_window_operation(issued)
    supports={face:_effective(model,face) for face in faces}
    def hull(face,box):
        if face in supports:return tuple(I(*pair) for pair in _source_box(supports[face],box))
        cover=operation.covers[face]
        uv=tuple(tuple(a+(b-a)*t for t in pair) for (a,b),pair in zip(cover.chart_box,box))
        points=tuple(cover.frame.world(*point) for point in product(*uv))
        return tuple(I(min(p[k] for p in points),max(p[k] for p in points)) for k in range(3))
    operation.hull=hull
    box=((Q(3,4),Q(1)),(Q(49,100),Q(51,100)))
    other=faces[1] if kind=='continuation' else plane
    row=operation.query(faces[0],box,other)
    assert row is not None and row.excluded
    assert row.requires_realization==(kind=='plane')
    return operation,boundary,row,(faces[0],box,other)


@pytest.mark.parametrize('kind',('continuation','plane'))
def test_actual_partition_callback_mutation_is_never_issued(issued,kind):
    from anygeometry import local_sweep_plane_contact as contact
    operation,boundary,row,_=_actual_partition_operation(issued,kind)
    try:
        for name,bad in (('exterior_squared_lower',row.exterior_squared_lower+1),
                         ('retained',(('changed',),)),('excluded',()),
                         ('source_box',((Q(0),Q(0)),(Q(0),Q(0))))):
            old=getattr(row,name)
            def mutate(phase):
                object.__setattr__(row,name,bad)
                return False
            boundary._callback=mutate
            try:
                with pytest.raises(GeometryError,match='partition changed'):operation.budget.take()
                with pytest.raises(GeometryError):operation.finish()
                assert all(window not in contact._WINDOWS for window in operation.pending)
            finally:
                object.__setattr__(row,name,old);boundary._callback=None
        # The full nested continuation/window definition is also an operand.
        old=row.window.exterior_squared_lower
        def mutate_nested(phase):
            object.__setattr__(row.window,'exterior_squared_lower',old+1)
            return False
        boundary._callback=mutate_nested
        try:
            with pytest.raises(GeometryError):operation.budget.take()
            assert all(window not in contact._WINDOWS for window in operation.pending)
        finally:
            object.__setattr__(row.window,'exterior_squared_lower',old);boundary._callback=None
        operation.guard()
        token=GeometryError('original partition callback')
        def stop(phase):raise token
        boundary._callback=stop
        with pytest.raises(GeometryError) as caught:operation.budget.take()
        assert caught.value is token
        assert all(window not in contact._WINDOWS for window in operation.pending)
    finally:boundary.close()


def test_actual_partition_inventory_replacement_and_removal_refuse(issued):
    from anygeometry import local_sweep_plane_contact as contact
    operation,boundary,row,_=_actual_partition_operation(issued,'plane')
    key=next(iter(operation.rows));rows=operation.rows[key]
    try:
        for action in ('remove','replace','key'):
            def mutate(phase):
                if action=='remove':operation.rows.pop(key)
                elif action=='replace':operation.rows[key]=[]
                else:operation.rows[('foreign',)]=operation.rows.pop(key)
                return False
            boundary._callback=mutate
            try:
                with pytest.raises(GeometryError,match='inventory changed'):operation.budget.take()
                with pytest.raises(GeometryError):operation.finish()
                assert all(window not in contact._WINDOWS for window in operation.pending)
            finally:
                operation.rows.clear();operation.rows[key]=rows;boundary._callback=None
    finally:boundary.close()


@pytest.mark.parametrize('kind',('continuation','plane'))
def test_actual_partition_unchanged_calculation_and_final_guard(issued,kind):
    from anygeometry.definition_binding import definition_checksum
    operation,boundary,row,args=_actual_partition_operation(issued,kind)
    try:
        again=operation.query(*args)
        assert definition_checksum(again)==definition_checksum(row)
        operation.guard()
        issued_stars=operation.finish()
        assert bool(issued_stars)==(kind=='plane')
    finally:boundary.close()
