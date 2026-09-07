"""Owner evidence for the unchanged public quadratic CONNECT partition."""
from fractions import Fraction as Q
from dataclasses import replace
import numpy as np

import pytest

from anygeometry import (
    GeometryModel, apply_imprint, plan_imprint, query_intersection,
    query_trim_domain_relation, validate_trim_domain_binding,
    TrimInteriorRelation, TrimBoundaryContact,
)
from anygeometry.curves import Spline
from anygeometry import EntityRef, OrientedEdge, TolerancePolicy, Plane, TrimDomainError
from anygeometry.entities import Vertex
from anygeometry import trim_domains as td


def partition(shift=False):
    model = GeometryModel()
    if shift:
        model.add_plate(model.add_points(((100,100,0),(101,100,0),(101,101,0),(100,101,0))))
    support = model.add_plate(model.add_points(
        ((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0))))
    a, b, c = model.add_points(((.5, .5, 0), (1.5, 1.5, 0), (2.5, .5, 0)))
    spline = model.add_spline(a, (b,), c)
    wall = model.extrude((spline,), (0, 0, 1))[0]
    model.add_sheet((support,))
    model.add_sheet((wall,))
    result = query_intersection(model, model.handle('face', support), model.handle('face', wall))
    apply_imprint(model, plan_imprint(model, result, policy='connect'), policy='connect')
    parents = tuple(model.handle('face', r.id) for r in model.resolve_ref(EntityRef('face',support)))
    return model, parents, spline


def test_gate_one_exact_public_construction():
    model, parents, spline = partition()
    from anygeometry import EntityRef
    assert model.replacement_history()[EntityRef('face', 1)] == (
        EntityRef('face', 3), EntityRef('face', 4))
    assert 1 not in model.faces
    uses = [model.faces[p.id].loop for p in parents]
    shared = {u.edge for u in uses[0]} & {u.edge for u in uses[1]}
    assert shared == {5, 13, 14}
    assert all(next(u.forward for u in uses[0] if u.edge == e)
               != next(u.forward for u in uses[1] if u.edge == e) for e in shared)
    assert all(not model.faces[p.id].holes and model.faces[p.id].parameterization is None
               for p in parents)
    # Independent exact world-coordinate reconstruction, not a sampled polygon.
    xyz = lambda i: tuple(Q(float(x)) for x in model.vertex_position(i))
    edge = model.edges[spline]
    controls = [xyz(i) for i in (edge.start, *edge.curve.control_vertices, edge.end)]
    assert controls == [(Q(1,2), Q(1,2), 0), (Q(3,2), Q(3,2), 0),
                        (Q(5,2), Q(1,2), 0)]
    # x'(t)=2 strictly positive; y(t)=1/2+2t-2t² in [1/2,1].
    assert [2 * (b[0]-a[0]) for a,b in zip(controls, controls[1:])] == [2, 2]
    assert all(0 < x < 3 and 0 < y < 2 and z == 0 for x,y,z in controls)
    assert (xyz(11), xyz(12)) == ((0,Q(1,2),0), (3,Q(1,2),0))
    assert [(model.edges[i].start, model.edges[i].end) for i in (13,5,14)] == [
        (11,5), (5,7), (7,12)]
    assert [u.edge for u in uses[0]] == [11,1,10,13,5,14]
    assert [u.edge for u in uses[1]] == [9,3,12,14,5,13]
    # Cancelling that opposite chain leaves precisely the split rectangle sides.
    outer = [u for loop in uses for u in loop if u.edge not in shared]
    assert {u.edge for u in outer} == {1,3,9,10,11,12}
    assert {xyz(model.oriented_start_vertex(u)) for u in outer} == {
        (0,0,0),(3,0,0),(3,2,0),(0,2,0),(0,Q(1,2),0),(3,Q(1,2),0)}


def state(model):
    return repr(model.__dict__), tuple((i,v.position.tobytes()) for i,v in model.vertices.items())


def query(model, parents, **kwargs):
    return query_trim_domain_relation(model,*parents,expected_revision=model.revision,**kwargs)


def check(model,parents,positive=True):
    before = state(model)
    result = query(model,parents)
    assert result.complementary is positive, result.diagnostics
    if positive:
        assert result.relation is TrimInteriorRelation.DISJOINT_INTERIORS
        assert result.boundary is TrimBoundaryContact.CURVE
        assert result.algorithm == 'exact_quadratic_partition_crosscut_v1'
        assert len(result.shared_boundary)==3
    else:
        assert result.relation is TrimInteriorRelation.UNRESOLVED
    validate_trim_domain_binding(model,result,*parents,expected_revision=model.revision)
    assert state(model)==before
    return result


@pytest.mark.parametrize('order',[False,True])
@pytest.mark.parametrize('reverse',['none','loops','native'])
def test_public_partition_binding_and_orientation(order,reverse):
    m,p,e = partition()
    if order: p=tuple(reversed(p))
    if reverse=='loops':
        for h in p:
            f=m.faces[h.id]
            m._set_entity_unjournalled('face',h.id,replace(f,loop=tuple(
                OrientedEdge(u.edge,not u.forward) for u in reversed(f.loop))))
    elif reverse=='native':
        edge=m.edges[e]
        m._set_entity_unjournalled('edge',e,replace(edge,start=edge.end,end=edge.start))
        for h in p:
            f=m.faces[h.id]
            m._set_entity_unjournalled('face',h.id,replace(f,loop=tuple(
                OrientedEdge(u.edge,not u.forward) if u.edge==e else u for u in f.loop)))
    check(m,p)


@pytest.mark.parametrize('scale',[2.**-20,1.,2.**20])
@pytest.mark.parametrize('offset',[(0.,0.,0.),(2.**30,-2.**30,2.**30)])
def test_representable_translation_scale(scale,offset):
    m,p,e=partition()
    # Transform exact accepted records, including their support frames.
    for k,v in tuple(m.vertices.items()):
        m._set_entity_unjournalled('vertex',k,Vertex(k,np.asarray(offset)+scale*v.position))
    for k,f in tuple(m.faces.items()):
        if isinstance(f.surface,Plane):
            s=f.surface
            m._set_entity_unjournalled('face',k,replace(f,surface=Plane(
                np.asarray(offset)+scale*s.origin,scale*s.u_vector,scale*s.v_vector)))
    m.set_document_settings(tolerance=TolerancePolicy(
        length=1e-9*scale,merge_length=1e-7*scale,
        area=1e-18*scale**2,surface_residual=1e-8*scale))
    check(m,p)


@pytest.mark.parametrize('control',[(1.5,.75,0),(1.25,1.25,0),(1.75,.25,0)])
def test_independent_quadratic_derivative_and_hull_oracle(control):
    m,p,e=partition()
    m._set_entity_unjournalled('vertex',6,Vertex(6,control))
    # These sufficient signs certify every t, not a grid of samples.
    cx,cy,_=map(Q,control)
    assert 2*(cx-Q(1,2))>0 and 2*(Q(5,2)-cx)>0
    assert 0<cy<2 and 0<cx<3
    check(m,p)


@pytest.mark.parametrize('defect',[
    'missing','ambiguous','indirect','wider','history_budget','ref_budget',
    'retrace','outside','tangent','degenerate','nonplanar','cubic','hole',
    'orientation','duplicate','coincident','wrong_path','nonconvex','parameterization',
    'tiny_overlap','distinct_identity','nonfinite',
])
def test_hostile_partition_refuses_without_mutation(defect):
    m,p,e=partition(); f=m.faces[p[1].id]
    key=EntityRef('face',1)
    if defect=='missing': m._replacement_history.pop(key)
    elif defect=='ambiguous': m._replacement_history[EntityRef('face',99)]=m._replacement_history[key]
    elif defect=='indirect': m._replacement_history[EntityRef('face',99)]=(key,)
    elif defect=='wider': m._replacement_history[key]+= (EntityRef('face',2),)
    elif defect=='history_budget':
        for i in range(300): m._replacement_history[EntityRef('edge',1000+i)]=()
    elif defect=='ref_budget': m._replacement_history[EntityRef('edge',99)]=(EntityRef('edge',1),)*513
    elif defect in ('retrace','outside','tangent','degenerate','nonplanar','tiny_overlap'):
        pos={'retrace':(-1,1,0),'outside':(1.5,5,0),'tangent':(1.5,3.5,0),
             'degenerate':(1.5,.5,0),'nonplanar':(1.5,1.5,.01),
             'tiny_overlap':(1.5,3.5000000001,0)}[defect]
        m._set_entity_unjournalled('vertex',6,Vertex(6,pos))
    elif defect=='cubic': m._set_entity_unjournalled('edge',e,replace(m.edges[e],curve=Spline((6,6))))
    elif defect=='hole': m._set_entity_unjournalled('face',f.id,replace(f,holes=(f.loop,)))
    elif defect=='orientation': m._set_entity_unjournalled('face',f.id,replace(f,loop=tuple(
        OrientedEdge(u.edge,not u.forward) for u in reversed(f.loop))))
    elif defect=='duplicate': m._set_entity_unjournalled('face',f.id,replace(f,loop=f.loop+f.loop))
    elif defect=='coincident': m._set_entity_unjournalled('face',f.id,replace(f,loop=m.faces[p[0].id].loop))
    elif defect=='wrong_path': m._set_entity_unjournalled('face',f.id,replace(f,loop=f.loop[1:3]+f.loop[:1]+f.loop[3:]))
    elif defect=='nonconvex': m._set_entity_unjournalled('vertex',3,Vertex(3,(1,.25,0)))
    elif defect=='parameterization': m._set_entity_unjournalled('face',f.id,replace(f,parameterization=f.surface))
    elif defect=='distinct_identity':
        m._set_entity_unjournalled('edge',99,replace(m.edges[e],id=99))
        m._set_entity_unjournalled('face',f.id,replace(f,loop=tuple(
            OrientedEdge(99,u.forward) if u.edge==e else u for u in f.loop)))
    elif defect=='nonfinite':
        # Bypass immutable construction solely to inject malformed evidence.
        v=Vertex(6,(1,1,0)); object.__setattr__(v,'position',np.array([np.inf,1.,0.]))
        m._set_entity_unjournalled('vertex',6,v)
    check(m,p,False)


def test_between_samples_crossing_is_not_accepted():
    m,p,e=partition()
    # y=1/2+2(c-1/2)t(1-t) touches y=2 at t=1/2 for c=3.5;
    # c slightly larger crosses in a narrow interval around 1/2.
    c=Q(7,2)+Q(1,2**30)
    assert Q(1,4)+c/2>2
    m._set_entity_unjournalled('vertex',6,Vertex(6,(1.5,float(c),0)))
    check(m,p,False)


def test_freshness_forgery_and_busy():
    m,p,e=partition(); r=check(m,p)
    with pytest.raises(TrimDomainError):
        validate_trim_domain_binding(m,r,*reversed(p),expected_revision=m.revision)
    data=td._result_data(r); data['max_residual']=r.tolerance/2
    fake=td.TrimDomainResult(**data,digest=td._digest(data))
    before=state(m)
    with pytest.raises(TrimDomainError):
        validate_trim_domain_binding(m,fake,*p,expected_revision=m.revision)
    assert state(m)==before
    with m.transaction():
        before=state(m)
        with pytest.raises(TrimDomainError): query(m,p)
        assert state(m)==before
    m.add_point(100,100,100)
    with pytest.raises(TrimDomainError):
        validate_trim_domain_binding(m,r,*p,expected_revision=m.revision)
    other,op,_=partition()
    with pytest.raises(TrimDomainError): query(other,p)


@pytest.mark.parametrize('at',[1,5,12,20])
def test_cancellation_identity_and_purity(at):
    m,p,e=partition(); before=state(m); error=RuntimeError('cancel'); count=0
    def cancel(phase):
        nonlocal count
        count+=1
        if count==at: raise error
    with pytest.raises(RuntimeError) as caught: query(m,p,cancellation_check=cancel)
    assert caught.value is error and state(m)==before


def test_budget_and_late_operational_failure(monkeypatch):
    m,p,e=partition(); before=state(m)
    original=td._Proof.__init__
    def exhausted(self,callback):
        original(self,callback);self.counts['rational_operations']=200000
    with monkeypatch.context() as patch:
        patch.setattr(td._Proof,'__init__',exhausted)
        assert query(m,p).relation is TrimInteriorRelation.UNRESOLVED
    error=RuntimeError('late owner failure')
    def fail(*args): raise error
    with monkeypatch.context() as patch:
        patch.setattr(td,'_quadratic_partition',fail)
        with pytest.raises(RuntimeError) as caught: query(m,p)
        assert caught.value is error
    assert state(m)==before


def test_unrelated_geometry_locality():
    m,p,e=partition(); first=check(m,p)
    for i in range(100): m.add_point(100+i,100,100)
    second=check(m,p)
    assert dict(first.work_counts)==dict(second.work_counts)


def test_shifted_identity_and_external_radial_ownership():
    m,p,e=partition(shift=True)
    before=state(m)
    result=check(m,p)
    assert {h.id for h in result.shared_boundary} != {5,13,14}
    assert len(m.face_uses_using_edge(e))>=3
    assert state(m)==before


def test_bit_budget_unresolved(monkeypatch):
    m,p,_=partition();before=state(m)
    original=td._Proof.q
    def huge(self,value):
        return original(self,Q(1,1<<8193))
    monkeypatch.setattr(td._Proof,'q',huge)
    result=query(m,p)
    assert result.relation is TrimInteriorRelation.UNRESOLVED
    assert state(m)==before


def test_cancellation_during_fresh_binding():
    m,p,_=partition();r=check(m,p);before=state(m);error=RuntimeError('binding cancel')
    def cancel(phase): raise error
    with pytest.raises(RuntimeError) as caught:
        validate_trim_domain_binding(m,r,*p,expected_revision=m.revision,cancellation_check=cancel)
    assert caught.value is error and state(m)==before


def test_warm_and_cold_results_match():
    m,p,_=partition();cold=check(m,p)
    m.validate_topology()
    warm=check(m,p)
    assert warm.digest==cold.digest


def test_degenerate_other_support_refuses():
    m,p,_=partition();f=m.faces[p[1].id]
    # Constructing an invalid Plane publicly is prohibited; inject only to
    # exercise the owner handler against malformed committed-like records.
    plane=Plane((0,0,0),(1,0,0),(0,1,0))
    object.__setattr__(plane,'v_vector',np.array((2.,0.,0.)))
    m._set_entity_unjournalled('face',f.id,replace(f,surface=plane))
    check(m,p,False)
