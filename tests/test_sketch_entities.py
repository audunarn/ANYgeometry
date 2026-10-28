from dataclasses import replace
import numpy as np
import pytest
from anygeometry import (GeometryModel, SketchDefinition, SketchEntity, SketchWire,
    SketchConstraint, plane_sketch_plane, upgrade_sketch, solve_sketch_edit,
    materialize_sketch_on_plane, sample_sketch_entity, set_sketch_radius,
    sketch_wire_edge_keys, apply_sketch_cut, round_sketch_corner, GeometryError)

PLANE=plane_sketch_plane((0,0,0),(0,0,1),(1,0,0))

@pytest.mark.parametrize('reverse', [False, True])
def test_arc_and_chord_closed_sketch_and_plate_cut(reverse):
    endpoints = ('a','c') if reverse else ('c','a')
    definition = SketchDefinition(
        points={'a':(2.,5.),'via':(5.,8.),'c':(8.,5.)}, version=2,
        entities=(SketchEntity('arc','arc',('a','via','c')),
                  SketchEntity('chord','line',endpoints)),
        wires=(SketchWire('opening',('arc','chord'),True),))
    geometry, face = plate()
    outputs = materialize_sketch_on_plane(geometry, PLANE, definition)
    keys = sketch_wire_edge_keys(definition)[0]
    assert len(keys) == 2
    apply_sketch_cut(geometry, face, [[outputs[key].id for key in keys]])
    assert len(geometry.faces[face].holes[0]) == 3
    assert all(outputs[key].id in geometry.edges for key in keys)
    assert not geometry.validate_topology()

def plate():
    g=GeometryModel(); f=g.add_plate(g.add_points(((0,0,0),(10,0,0),(10,10,0),(0,10,0))))
    return g,f

def circle():
    return SketchDefinition({'c':(5,5),'r':(6,5)},version=2,
        entities=(SketchEntity('circle','circle',('c','r')),),wires=(SketchWire('hole',('circle',),True),))

def test_legacy_upgrade_roundtrip_aliases():
    d=SketchDefinition({'a':(1,1),'b':(2,1),'c':(2,2)},('a','b','c'))
    assert 'version' not in d.to_parameters()
    u=upgrade_sketch(d)
    assert SketchDefinition.from_parameters(u.to_parameters())==u
    assert sketch_wire_edge_keys(d)==(('profile/edge/0','profile/edge/1','profile/edge/2'),)
    g=GeometryModel(); out=materialize_sketch_on_plane(g,PLANE,u)
    assert out['profile/edge/1']==out['entity/legacy-1/edge/0']

def test_constrained_drag_and_fixed_diagnostics():
    d=SketchDefinition({'a':(0,0),'b':(2,0)},version=2,
      constraints=(SketchConstraint('fixed','a',id='anchor'),SketchConstraint('distance','a','b',2,id='length')))
    r=solve_sketch_edit(d,PLANE,drag_targets={'b':(2,2)})
    assert r.converged and r.degrees_of_freedom==1
    assert r.points['a']==(0,0)
    assert np.linalg.norm(r.points['b'])==pytest.approx(2)
    bad=replace(d,constraints=d.constraints+(SketchConstraint('fixed','b'),SketchConstraint('distance','a','b',3)))
    r=solve_sketch_edit(bad,PLANE)
    assert not r.converged and r.points==bad.points and r.conflicts

def test_circle_cut_preserves_identity_and_holes():
    g,f=plate(); d=circle(); outputs=materialize_sketch_on_plane(g,PLANE,d)
    edges=[outputs[k].id for k in sketch_wire_edge_keys(d)[0]]
    result=apply_sketch_cut(g,f,[edges]); assert result['face'].id==f
    assert len(g.faces[f].holes)==1
    with pytest.raises(GeometryError): apply_sketch_cut(g,f,[edges])
    assert len(g.faces[f].holes)==1
    assert len(sample_sketch_entity(set_sketch_radius(d,'circle',2),'circle'))>30

@pytest.mark.parametrize('kind',['spline','nurbs'])
def test_closed_spline_shared_endpoint_and_cut(kind):
    g,f=plate()
    d=SketchDefinition({'a':(3,3),'b':(7,3),'c':(7,7),'d':(3,7)},version=2,
       entities=(SketchEntity('curve',kind,('a','b','c','d'),closed=True),),wires=(SketchWire('hole',('curve',),True),))
    out=materialize_sketch_on_plane(g,PLANE,d)
    edges=[out[k].id for k in sketch_wire_edge_keys(d)[0]]
    assert len(g.order_loop(edges))==4
    apply_sketch_cut(g,f,[edges]); assert len(g.faces[f].holes)==1

def test_fillet_and_shared_point_identity():
    d=SketchDefinition({'a':(1,1),'b':(3,1),'c':(3,3)},('a','b','c'),closed=False)
    u=upgrade_sketch(d); r=round_sketch_corner(u,'legacy-0','legacy-1',.3)
    assert 'b' in r.points and r.wires[0].id=='profile'
    assert r.entities[0].id not in ('legacy-0','legacy-1')
    assert len(sample_sketch_entity(r,r.entities[0]))>3
    g=GeometryModel(); out=materialize_sketch_on_plane(g,PLANE,u)
    assert g.edges[out['entity/legacy-0/edge/0'].id].end==g.edges[out['entity/legacy-1/edge/0'].id].start

def test_fillet_retires_incident_output_roles_but_preserves_corner_point():
    g=GeometryModel(); g.features.capture_baseline(g)
    d=upgrade_sketch(SketchDefinition({'a':(1,1),'b':(3,1),'c':(3,3),'d':(1,3)},('a','b','c','d')))
    params=d.to_parameters(); params['plane']={'origin':[0,0,0],'normal':[0,0,1],'x_axis':[1,0,0]}
    feature=g.features.append('geometry.sketch.plane',parameters=params)
    report=g.regenerate_features(); assert report.success,report
    rounded=round_sketch_corner(d,'legacy-0','legacy-1',.3)
    params=rounded.to_parameters(); params['plane']={'origin':[0,0,0],'normal':[0,0,1],'x_axis':[1,0,0]}
    g.features.update(feature.feature_id,parameters=params)
    report=g.regenerate_features(); assert report.success,report
    outputs=g.features.get(feature.feature_id).outputs
    assert 'profile/edge/0' not in outputs and 'profile/edge/1' not in outputs
    assert not any(k.startswith('wire/') for k in outputs)
    assert all(k in outputs for k in sketch_wire_edge_keys(rounded)[0])
    assert tuple(g.vertex_position(outputs['point/b'].id))==(3,1,0)

def test_coincident_coordinates_do_not_merge_unrelated_point_ids():
    d=SketchDefinition({'a':(1,1),'b':(2,1),'c':(2,1),'d':(3,1)},version=2,
       entities=(SketchEntity('e1','line',('a','b')),SketchEntity('e2','line',('c','d'))))
    g=GeometryModel(); out=materialize_sketch_on_plane(g,PLANE,d)
    assert out['point/b']!=out['point/c']

def test_curve_feature_regeneration_and_persistence():
    from anygeometry import FeatureOutputRef,to_dict,from_dict
    g,f=plate(); g.features.capture_baseline(g)
    d=SketchDefinition({'a':(3,3),'b':(7,3),'c':(7,7),'d':(3,7)},version=2,
       entities=(SketchEntity('curve','nurbs',('a','b','c','d'),closed=True),),wires=(SketchWire('hole',('curve',),True),))
    params=d.to_parameters(); params['plane']={'origin':[0,0,0],'normal':[0,0,1],'x_axis':[1,0,0]}
    feature=g.features.append('geometry.sketch.plane',parameters=params)
    assert g.regenerate_features().success
    keys=sketch_wire_edge_keys(d)[0]
    cut=g.features.append('geometry.sketch.cut',parameters={'wire_sizes':[len(keys)],'wire_ids':['hole'],'sketch_feature_id':feature.feature_id},
        inputs={'target':(g.faces[f].ref,), 'profile':tuple(FeatureOutputRef(feature.feature_id,key,'edge') for key in keys)})
    report=g.regenerate_features(); assert report.success,report
    restored=from_dict(to_dict(g)); assert len(restored.faces[f].holes)==1
    changed=dict(params); changed['points']=dict(params['points']); changed['points']['b']=[6.5,3]
    g.features.update(feature.feature_id,parameters=changed)
    report=g.regenerate_features(); assert report.success,report
    assert len(g.faces[f].holes)==1

def test_cut_rejects_touching_and_outside_and_preserves_existing_holes():
    g,f=plate()
    first=circle(); out=materialize_sketch_on_plane(g,PLANE,first)
    apply_sketch_cut(g,f,[[out[k].id for k in sketch_wire_edge_keys(first)[0]]])
    second=replace(first,points={'c':(2,2),'r':(2.5,2)})
    out=materialize_sketch_on_plane(g,PLANE,second)
    apply_sketch_cut(g,f,[[out[k].id for k in sketch_wire_edge_keys(second)[0]]]); assert len(g.faces[f].holes)==2
    bad=replace(first,points={'c':(.5,5),'r':(1.5,5)})
    out=materialize_sketch_on_plane(g,PLANE,bad)
    with pytest.raises(GeometryError): apply_sketch_cut(g,f,[[out[k].id for k in sketch_wire_edge_keys(bad)[0]]])
    assert len(g.faces[f].holes)==2

def test_nurbs_cut_preserves_existing_disjoint_circular_hole():
    g,f=plate(); first=replace(circle(),points={'c':(2,2),'r':(2.5,2)})
    out=materialize_sketch_on_plane(g,PLANE,first)
    apply_sketch_cut(g,f,[[out[k].id for k in sketch_wire_edge_keys(first)[0]]])
    d=SketchDefinition({'a':(5,5),'b':(8,5),'c':(8,8),'d':(5,8)},version=2,
       entities=(SketchEntity('curve','nurbs',('a','b','c','d'),closed=True),),wires=(SketchWire('hole',('curve',),True),))
    out=materialize_sketch_on_plane(g,PLANE,d)
    apply_sketch_cut(g,f,[[out[k].id for k in sketch_wire_edge_keys(d)[0]]])
    assert len(g.faces[f].holes)==2

def test_radius_dimension_remains_driving_and_fixed_radius_conflict():
    d=set_sketch_radius(circle(),'circle',2)
    r=solve_sketch_edit(d,PLANE,drag_targets={'r':(8,8)})
    assert r.converged
    assert np.linalg.norm(np.asarray(r.points['r'])-r.points['c'])==pytest.approx(2)
    d=replace(circle(),constraints=(SketchConstraint('fixed','c'),SketchConstraint('fixed','r')))
    changed=set_sketch_radius(d,'circle',2)
    r=solve_sketch_edit(changed,PLANE)
    assert not r.converged

def test_explicit_coincidence_unifies_topology_identity():
    d=SketchDefinition({'a':(1,1),'b':(2,1),'c':(2,1),'d':(3,1)},version=2,
       entities=(SketchEntity('e1','line',('a','b')),SketchEntity('e2','line',('c','d'))),
       constraints=(SketchConstraint('coincident','b','c'),))
    g=GeometryModel(); out=materialize_sketch_on_plane(g,PLANE,d)
    assert out['point/b']==out['point/c']
    assert g.edges[out['entity/e1/edge/0'].id].end==g.edges[out['entity/e2/edge/0'].id].start

def test_fit_spline_missing_scipy_is_actionable_but_controls_do_not_require_it(monkeypatch):
    import builtins
    original=builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name.startswith('scipy'): raise ImportError('scipy intentionally unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    d=SketchDefinition({'a':(0,0),'b':(1,1),'c':(2,0)},version=2,
        entities=(SketchEntity('curve','nurbs',('a','b','c')),))
    assert len(sample_sketch_entity(d,'curve'))>2
    fit=replace(d,entities=(replace(d.entities[0],kind='spline'),))
    with pytest.raises(GeometryError,match=r'ANYgeometry\[sketch\]'):
        sample_sketch_entity(fit,'curve')
