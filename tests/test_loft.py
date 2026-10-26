import numpy as np
import pytest
from anygeometry import GeometryModel, GeometryError
from anygeometry.nurbs import NurbsCurve, NurbsSurface
from anygeometry.serialization import to_dict, from_dict
from anygeometry.surfaces import CoonsSurface
from anygeometry.curves import Arc, Straight
from anygeometry.loft import loft_profile_options, LoftNotCertified


def chain(model, points):
    ids = model.add_points(points)
    return tuple(model.add_line(a,b) for a,b in zip(ids,ids[1:]))


def test_skew_lines_roundtrip_and_source_immutability():
    g = GeometryModel()
    a = chain(g,[(0,0,0),(2,0,0)])
    b = chain(g,[(0,1,1),(3,1,2)])
    before = {i:g.edges[i] for i in (*a,*b)}
    face, = g.loft(a,b)
    assert all(g.edges[i] == edge for i,edge in before.items())
    assert isinstance(g.faces[face].surface,CoonsSurface)
    for candidate in (g,from_dict(to_dict(g))):
        np.testing.assert_allclose(candidate.face_point(face,.25,.4),(.6,.4,.5))


def test_unequal_edges_share_connector_and_have_local_maps():
    g = GeometryModel()
    a = chain(g,[(0,0,0),(1,0,0),(2,0,0)])
    b = chain(g,[(0,1,0),(2,1,0)])
    first,second = g.loft(a,b)
    common = {x.edge for x in g.faces[first].loop}&{x.edge for x in g.faces[second].loop}
    assert len(common) == 1
    np.testing.assert_allclose(g.face_point(first,1.,.5),g.face_point(second,0.,.5))
    np.testing.assert_allclose(g.face_point(first,.5,.5),(.5,.5,0))
    np.testing.assert_allclose(g.face_point(second,.5,.5),(1.5,.5,0))


def test_rational_profiles_have_straight_world_connections():
    g = GeometryModel()
    curves = [NurbsCurve(2,(0,0,0,1,1,1),((0,y,0),(1,y,z),(2,y,0)),weights)
              for y,z,weights in ((0,1,(1,.7,1)),(2,.5,(1,1.3,1)))]
    edges = []
    for c in curves:
        a,b = g.add_points(c.evaluate([0,1])); edges.append(g.add_curve(a,b,c))
    face, = g.loft((edges[0],),(edges[1],))
    for u in (.1,.3,.8):
        for v in (.2,.7):
            expected = (1-v)*curves[0].evaluate([u])[0]+v*curves[1].evaluate([u])[0]
            np.testing.assert_allclose(g.face_point(face,u,v),expected,rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(g.face_support_point(face,u,v),expected,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('kind',['crossed','closed','cancelled'])
def test_refusal_is_atomic(kind):
    g = GeometryModel()
    a = chain(g,[(0,0,0),(1,0,0)])
    b = chain(g,[(1,1,0),(0,1,0)])
    if kind == 'closed':
        a = chain(g,[(0,0,2),(1,0,2),(1,1,2)])
        a += (g.add_line(g.edges[a[-1]].end,g.edges[a[0]].start),)
    before = to_dict(g)
    with pytest.raises(GeometryError): g.loft(a,b,cancellation_check=(lambda: True) if kind=='cancelled' else None)
    assert to_dict(g) == before


def test_reverse_target_accepts_opposite_authored_direction():
    g = GeometryModel()
    a = chain(g,[(0,0,0),(1,0,0)])
    b = chain(g,[(1,1,0),(0,1,0)])
    face, = g.loft(a,b,reverse_second=True)
    np.testing.assert_allclose(g.face_point(face,.2,.5),(.2,.5,0))


def closed_chain(g,points):
    edges = chain(g,points)
    return (*edges,g.add_line(g.edges[edges[-1]].end,g.edges[edges[0]].start))


def test_closed_tapered_profiles_share_real_seam_and_roundtrip():
    g = GeometryModel()
    a = closed_chain(g,[(0,0,0),(2,0,0),(2,2,0),(0,2,0)])
    b = closed_chain(g,[(.5,.5,2),(1.5,.5,2),(1.5,1.5,2),(.5,1.5,2)])
    faces = g.loft(a,b)
    assert len(faces) == 4
    for i,face in enumerate(faces):
        other = faces[(i+1)%len(faces)]
        shared = {item.edge for item in g.faces[face].loop}&{item.edge for item in g.faces[other].loop}
        assert len(shared) == 1
        assert isinstance(g.edges[next(iter(shared))].curve,Straight)
    restored = from_dict(to_dict(g))
    for face in faces:
        np.testing.assert_allclose(restored.face_point(face,.3,.7),g.face_point(face,.3,.7))


def test_closed_automatic_seam_is_frozen_and_editable():
    g = GeometryModel()
    a = closed_chain(g,[(0,0,0),(2,0,0),(2,2,0),(0,2,0)])
    b = closed_chain(g,[(2,2,1),(0,2,1),(0,0,1),(2,0,1)])
    flags = loft_profile_options(g,a,b)
    assert flags['second_start'] == 2
    assert len(g.loft(a,b,**flags)) == 4


@pytest.mark.parametrize('flip',[False,True])
def test_flipped_chart_corners_match_topology(flip):
    g = GeometryModel()
    a = chain(g,[(0,0,0),(1,0,0)])
    b = chain(g,[(0,1,1),(1,1,2)])
    face, = g.loft(a,b,flip_normal=flip)
    for item,uv in zip(g.faces[face].loop,[(0,0),(1,0),(1,1),(0,1)]):
        edge = g.edges[item.edge]
        expected = g.vertex_position(edge.start if item.forward else edge.end)
        np.testing.assert_allclose(g.face_point(face,*uv),expected)


def test_cancellation_after_vertices_rolls_back_all_mutations():
    g = GeometryModel()
    a = chain(g,[(0,0,0),(1,0,0)])
    b = chain(g,[(0,1,0),(1,1,0)])
    before = to_dict(g); count = len(g.vertices); revision = g.revision
    with pytest.raises(GeometryError,match='cancelled'):
        g.loft(a,b,cancellation_check=lambda:len(g.vertices)>count)
    after = to_dict(g)
    assert g.revision == revision
    assert all(after['id_state'][key] >= value for key,value in before['id_state'].items())
    for document in (before,after):
        document.pop('checksum'); document.pop('id_state')
    assert after == before


def test_translated_arc_preserves_exact_authored_curve():
    g = GeometryModel(); edges=[]
    for z in (0,2):
        a,m,b = g.add_points([(1,0,z),(0,1,z),(-1,0,z)])
        edges.append(g.add_arc(a,m,b))
    before = [g.edges[i] for i in edges]
    flags = loft_profile_options(g,edges[:1],edges[1:])
    face, = g.loft(edges[:1],edges[1:],**flags)
    assert [g.edges[i] for i in edges] == before
    assert sum(isinstance(g.edges[item.edge].curve,Arc) for item in g.faces[face].loop) == 2
    for u in (.1,.3,.8):
        expected = g.sample_edge(edges[0],np.array([u]))[0]+np.array([0,0,.6])
        np.testing.assert_allclose(g.face_point(face,u,.3),expected)


def test_multispan_reversed_nurbs():
    g = GeometryModel()
    curve = NurbsCurve(1,(0,0,.5,1,1),((0,0,0),(1,0,.4),(2,0,0)),domain=(1.,0.))
    edges=[]
    for y in (0,2):
        c = NurbsCurve(curve.degree,curve.knots,tuple((x,y,z) for x,_,z in curve.controls),domain=curve.domain)
        a,b = g.add_points(c.evaluate([0,1]));edges.append(g.add_curve(a,b,c))
    faces=g.loft(edges[:1],edges[1:])
    assert len(faces)==2
    np.testing.assert_allclose(g.face_point(faces[0],.5,.5),(1.5,1,.2))


def test_genuinely_nonplanar_chain():
    g = GeometryModel()
    points = np.array([(0.,0.,0.),(1.,0.,.2),(2.,.3,.1),(3.,.2,.5)])
    assert np.linalg.det(points[1:]-points[0]) != 0
    a = chain(g,points); b = chain(g,points+(0,2,0))
    faces = g.loft(a,b)
    assert len(faces) == 3
    for i,face in enumerate(faces):
        np.testing.assert_allclose(g.face_point(face,.5,.5),(points[i]+points[i+1])/2+(0,1,0))


def test_public_rigid_transform_of_curved_loft():
    from anygeometry.editing import move_entities
    from anygeometry.entities import EntityRef
    g = GeometryModel(); edges=[]
    for y,w in ((0,.7),(2,1.3)):
        c=NurbsCurve(2,(0,0,0,1,1,1),((0,y,0),(1,y,.4),(2,y,0)),(1,w,1))
        a,b=g.add_points(c.evaluate([0,1]));edges.append(g.add_curve(a,b,c))
    face,=g.loft(edges[:1],edges[1:])
    samples=[(.1,.2),(.6,.7),(.9,.3)]
    before=[g.face_point(face,*uv) for uv in samples]
    matrix=np.array([[0.,-1.,0.,3.],[1.,0.,0.,-2.],[0.,0.,1.,5.],[0.,0.,0.,1.]])
    move_entities(g,[EntityRef('face',face)],matrix)
    for uv,point in zip(samples,before):
        np.testing.assert_allclose(g.face_support_point(face,*uv),matrix[:3,:3]@point+matrix[:3,3],atol=1e-12)


@pytest.mark.parametrize('kind',['branch','repeat','disconnected','coincident'])
def test_invalid_profile_topology_refuses_without_mutation(kind):
    g=GeometryModel();a=chain(g,[(0,0,0),(1,0,0),(2,0,0)])
    b=chain(g,[(0,1,0),(2,1,0)])
    if kind=='branch':
        tip=g.add_point(1,-1,0);a+= (g.add_line(g.edges[a[0]].end,tip),)
    elif kind=='repeat': a+=(a[0],)
    elif kind=='disconnected': a+=(chain(g,[(3,0,0),(4,0,0)])[0],)
    else: b=a
    before=to_dict(g)
    with pytest.raises(GeometryError):g.loft(a,b)
    assert to_dict(g)==before


@pytest.mark.parametrize('distance',[2.,1e30])
def test_unrepresentable_boundary_refuses_under_authored_tolerance(distance):
    from anygeometry.tolerance import TolerancePolicy
    g=GeometryModel(tolerance=TolerancePolicy(curve_fit_residual=1e-30,relative_length=1e-40))
    edges=[]
    for y in (0,distance):
        if y==0:
            a,m,b=g.add_points(((0,y,0),(1,y,.4),(2,y,0)))
            edges.append(g.add_spline(a,(m,),b))
        else:
            a,m,n,b=g.add_points(((0,y,0),(.7,y,.2),(1.4,y,.3),(2,y,0)))
            edges.append(g.add_spline(a,(m,n),b))
    before=to_dict(g)
    with pytest.raises(LoftNotCertified,match='boundary error'):
        g.loft(edges[:1],edges[1:])
    assert to_dict(g)==before


def test_expanded_authored_profile_reference_is_explicitly_blocked():
    from anygeometry import EntityRef
    g=GeometryModel();a=chain(g,[(0,0,0),(2,0,0)]);b=chain(g,[(0,1,0),(2,1,0)])
    g.features.capture_baseline(g)
    g.features.append('geometry.split_edge',inputs={'edge':(EntityRef('edge',a[0]),)},parameters={'fraction':.5})
    loft=g.features.append('geometry.loft',inputs={'first_edges':(EntityRef('edge',a[0]),),
                                                'second_edges':(EntityRef('edge',b[0]),)})
    report=g.regenerate_features()
    result=next(item for item in report.features if item.feature_id==loft.feature_id)
    assert result.state=='blocked'
    assert 'changed topology' in result.diagnostic
    assert not result.outputs and not g.faces


@pytest.mark.parametrize('end',[(1.,0.),(1.,1.)],ids=['iso','diagonal'])
def test_curve_on_surface_profiles_refuse_without_crashing_or_mutation(end):
    from anygeometry.nurbs import CurveOnSurface
    g=GeometryModel()
    support=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
                         (((0.,0.,0.),(0.,1.,0.)),((2.,0.,0.),(2.,1.,0.))))
    curve=CurveOnSurface(support,NurbsCurve(1,(0.,0.,1.,1.),((0.,0.),end)))
    a,b=g.add_points(curve.evaluate([0.,1.]));first=(g.add_curve(a,b,curve),)
    second=chain(g,[(0,2,1),(2,2,1)])
    before=to_dict(g)
    for action in (lambda:g.loft(first,second),lambda:loft_profile_options(g,first,second)):
        with pytest.raises(GeometryError,match='curve-on-surface profiles'):
            action()
        assert to_dict(g)==before
    with pytest.raises(GeometryError,match='cancelled'):
        g.loft(first,second,cancellation_check=lambda:True)
    assert to_dict(g)==before
