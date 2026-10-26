"""Ordinary modeling profiles, including actual sketch-like circular wires."""
import math
import numpy as np
import pytest

from anygeometry import GeometryModel,GeometryError
from anygeometry.curves import Arc,Straight
from anygeometry.loft import loft_profile_options
from anygeometry.serialization import to_dict,from_dict


def circle(g,radius=1.,center=(0.,0.,0.),tilt=0.,pieces=4):
    center=np.asarray(center);angle=math.radians(tilt)
    x=np.array([math.cos(angle),0.,-math.sin(angle)]);y=np.array([0.,1.,0.])
    # Four exact cardinal positions and arithmetic via points match sketch v2.
    if pieces==4:
        vectors=(x,y,-x,-y);via=[(vectors[i]+vectors[(i+1)%4])/math.sqrt(2) for i in range(4)]
    else:
        assert pieces==2;vectors=(x,-x);via=(y,-y)
    vertices=[g.add_point(*(center+radius*v)) for v in vectors]
    mids=[g.add_point(*(center+radius*v)) for v in via]
    return tuple(g.add_arc(vertices[i],mids[i],vertices[(i+1)%pieces]) for i in range(pieces))


@pytest.mark.parametrize('radius,offset,tilt,pieces',[
    (1.,(0,0,2),0,4),(.8,(0,0,2),0,4),(4.,(0,0,3),0,4),
    (.8,(.2,.1,3),20,4),(.8,(.2,.1,3),0,2),
])
def test_circle_loft_keeps_arcs_and_shared_seam(radius,offset,tilt,pieces):
    g=GeometryModel();first=circle(g);second=circle(g,radius,offset,tilt,pieces)
    source={i:g.edges[i] for i in (*first,*second)}
    flags=loft_profile_options(g,first,second);faces=g.loft(first,second,**flags)
    assert len(faces)>=4
    assert all(g.edges[i]==edge for i,edge in source.items())
    for i,face in enumerate(faces):
        edges={item.edge for item in g.faces[face].loop}
        assert sum(isinstance(g.edges[e].curve,Arc) for e in edges)==2
        shared=edges&{item.edge for item in g.faces[faces[(i+1)%len(faces)]].loop}
        assert len(shared)==1 and isinstance(g.edges[shared.pop()].curve,Straight)
    restored=from_dict(to_dict(g))
    for face in faces:
        np.testing.assert_allclose(restored.face_point(face,.3,.6),g.face_point(face,.3,.6))


def test_circle_manual_seam_twist_and_auto_reversed_authorship():
    g=GeometryModel();a=circle(g);b=circle(g,center=(0,0,2))
    shifted=b[2:]+b[:2]
    flags=loft_profile_options(g,a,shifted)
    assert flags['second_start']==2
    assert len(g.loft(a,shifted,**flags))==4
    assert len(g.loft(a,b,second_start=1))==4


def test_large_radius_ratio_is_independent_of_global_plane_orientation():
    g=GeometryModel();angle=math.radians(27)
    a=circle(g,tilt=27);b=circle(g,radius=4.,center=(3*math.sin(angle),0.,3*math.cos(angle)),tilt=27)
    assert len(g.loft(a,b,**loft_profile_options(g,a,b)))==4


@pytest.mark.parametrize('kind',['coincident','reversed','cancel'])
def test_circle_invalid_or_cancelled_loft_is_atomic(kind):
    g=GeometryModel();a=circle(g);b=circle(g,center=(0,0,0 if kind=='coincident' else 2))
    before=to_dict(g);vertices=len(g.vertices)
    with pytest.raises(GeometryError):
        g.loft(a,b,reverse_second=kind=='reversed',
               cancellation_check=(lambda:len(g.vertices)>vertices) if kind=='cancel' else None)
    after=to_dict(g)
    if kind=='cancel':
        for payload in (before,after):payload.pop('checksum');payload.pop('id_state')
    assert after==before


def test_scaled_quarter_arc_is_not_approximated_by_nurbs():
    g=GeometryModel();edges=[]
    for radius,z in ((1,0),(.8,2)):
        a,m,b=g.add_points([(radius,0,z),(radius/math.sqrt(2),radius/math.sqrt(2),z),(0,radius,z)])
        edges.append(g.add_arc(a,m,b))
    faces=g.loft(edges[:1],edges[1:])
    assert len(faces)==1
    assert sum(isinstance(g.edges[item.edge].curve,Arc) for item in g.faces[faces[0]].loop)==2


@pytest.mark.parametrize('count',[3,4,5])
def test_rotated_polygon_boundaries_stay_straight(count):
    g=GeometryModel();profiles=[]
    for z,scale,angle in ((0.,1.,0.),(3.,.8,math.radians(20))):
        points=[(scale*math.cos(t)*math.cos(angle),scale*math.sin(t),z-scale*math.cos(t)*math.sin(angle)) for t in np.arange(count)*math.tau/count]
        ids=g.add_points(points);profiles.append(tuple(g.add_line(ids[i],ids[(i+1)%count]) for i in range(count)))
    faces=g.loft(*profiles,**loft_profile_options(g,*profiles))
    assert faces
    assert all(isinstance(g.edges[item.edge].curve,Straight) for face in faces for item in g.faces[face].loop)


def test_tilted_face_boundaries_keep_relative_length_correspondence():
    g=GeometryModel();profiles=[]
    for points in ([(0,0,0),(2,0,0),(2,2,0),(0,2,0)],
                   [(.2,.1,3),(1.8,.1,2.8),(1.8,1.7,2.8),(.2,1.7,3)]):
        ids=g.add_points(points)
        profiles.append(tuple(g.add_line(ids[i],ids[(i+1)%4]) for i in range(4)))
    faces=g.loft(*profiles,**loft_profile_options(g,*profiles))
    assert len(faces)>4  # Different edge lengths retain separate corner stations.
    assert all(isinstance(g.edges[item.edge].curve,Straight) for f in faces for item in g.faces[f].loop)


def test_periodic_circle_error_bounds_include_full_turn_and_open_bridge():
    from dataclasses import replace
    from fractions import Fraction
    from anygeometry.loft_circles import circle_profile,_periodic_errors,_tau_interval
    g=GeometryModel();profile=circle_profile(g,circle(g))
    bounded=_periodic_errors(profile)
    assert bounded.value_error>profile.value_error
    low,high=_tau_interval()
    assert low<high and not low<=Fraction(math.tau)<=high
    ev=Fraction(1,1000)
    open_profile=replace(profile,closed=False,span=math.pi,value_error=ev,derivative_error=Fraction(0))
    assert _periodic_errors(open_profile).derivative_error>=2*ev/(low-Fraction(math.pi))
    with pytest.raises(GeometryError,match='less than one full turn'):
        _periodic_errors(replace(open_profile,span=7.))


def test_circle_feature_seam_edit_preserves_stable_roles():
    from anygeometry import EntityRef
    from anygeometry.surfaces import CoonsSurface
    g=GeometryModel();a=circle(g);b=circle(g,radius=.8,center=(0,0,2))
    g.features.capture_baseline(g)
    feature=g.features.append('geometry.loft',inputs={
        'first_edges':tuple(EntityRef('edge',e) for e in a),
        'second_edges':tuple(EntityRef('edge',e) for e in b)})
    assert g.regenerate_features().success
    old=dict(g.features.get(feature.feature_id).outputs)
    assert all(isinstance(g.faces[r.id].surface,CoonsSurface) for r in old.values() if r.kind=='face')
    g.features.update(feature.feature_id,parameters={'second_start':1})
    assert g.regenerate_features().success
    new=dict(g.features.get(feature.feature_id).outputs)
    assert old.keys()==new.keys()
    assert all(isinstance(g.faces[r.id].surface,CoonsSurface) for r in new.values() if r.kind=='face')
    for slot,reference in old.items():
        assert g.resolve_ref(reference)==(new[slot],)


def test_parallel_fast_path_checks_actual_arc_frame_plane():
    from dataclasses import replace
    from fractions import Fraction
    from anygeometry.loft_circles import circle_profile,_parallel_admission
    g=GeometryModel();a=circle_profile(g,circle(g));b=circle_profile(g,circle(g,center=(0,0,2)))
    assert _parallel_admission(a,b,Fraction(0))
    # Keep all authored vertices coplanar, but perturb an actual stored frame.
    frame=replace(b.frames[0],center=b.frames[0].center+np.array([0.,0.,1e-12]))
    altered=replace(b,frames=[frame,*b.frames[1:]])
    assert not _parallel_admission(a,altered,Fraction(0))


def test_legacy_analytic_support_transition_is_loft_only():
    from dataclasses import replace
    from anygeometry import EntityRef
    from anygeometry.features import _bind_topology_roles
    from anygeometry.surfaces import Cone
    g=GeometryModel();a=circle(g);b=circle(g,radius=.8,center=(0,0,2))
    old=g.loft(a,b)[0];new=g.loft(a,b)[0]
    # A prior saved aligned loft could carry this canonical analytic support.
    g.set_face_surface(old,Cone(np.zeros(3),np.array([0.,0.,1.]),
        np.array([1.,0.,0.]),1.,.8,2.,0.,math.pi/2))
    oldref,newref=EntityRef('face',old),EntityRef('face',new)
    with pytest.raises(GeometryError,match='changes face structure'):
        _bind_topology_roles(oldref,g,newref,g,{})
    bindings={EntityRef('edge',a.edge):(EntityRef('edge',b.edge),)
              for a,b in zip(g.faces[old].loop,g.faces[new].loop)}
    _bind_topology_roles(oldref,g,newref,g,bindings,_stable_face_edges=True)
    assert bindings[oldref]==(newref,)
