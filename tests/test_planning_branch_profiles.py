"""Prospective profiles require exact declarative inputs before exclusions."""
from dataclasses import replace
import types
import numpy as np
import pytest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_executor import PlannerSourceChanged


def api():
    from anygeometry import _planning_profiles
    return _planning_profiles


def test_plane_face_selectors_are_proved_without_constructor_call(monkeypatch):
    from anygeometry._planning_canaries import _fixture
    from anygeometry.material_arrangement import MaterialDomain
    model,_=_fixture()
    profile=api().BranchProfiles()
    monkeypatch.setattr(MaterialDomain,'from_model',classmethod(lambda *args:pytest.fail('constructor executed')))
    with pytest.raises(PlannerSourceChanged):profile.face(model,next(iter(model.faces)))


def test_actual_domain_support_not_nominal_face_label_controls_pair():
    from anygeometry._planning_canaries import _fixture
    from anygeometry.material_arrangement import MaterialDomain
    model,_=_fixture();ids=list(model.faces)
    first=MaterialDomain.from_model(model,ids[0]);second=MaterialDomain.from_model(model,ids[1])
    profile=api().BranchProfiles()
    assert profile.pair(first,second,route=None,shadow=None)=='plane-cylinder-pair'
    with pytest.raises(QualificationDecline):profile.pair(replace(first,support=object()),second,route=None,shadow=None)
    with pytest.raises(QualificationDecline):profile.pair(first,second,route=object(),shadow=None)
    with pytest.raises(QualificationDecline):profile.pair(first,second,route=None,shadow={})


def test_declarative_plane_face_guard_preserves_model():
    from anygeometry._planning_canaries import _fixture
    model,_=_fixture();profile=api().BranchProfiles();face=next(iter(model.faces))
    before=dict(vars(model))
    assert profile.face(model,face)=='unchanged-plane-cylinder-support'
    assert all(vars(model)[key] is value for key,value in before.items())


@pytest.mark.parametrize('shape',[(3,),(2,3),(2,2,2),()])
def test_default_norm_profile_covers_exact_float64_any_dimension(shape):
    profile=api().BranchProfiles()
    assert profile.norm(np.ones(shape,dtype=np.float64),{})=='norm-default-real'


@pytest.mark.parametrize('value,kwargs',[(np.array([1],dtype=np.int64),{}),
    (np.array([float('nan')]),{}),(np.ones((2,3)),{'axis':1}),
    (np.ones(3),{'ord':2}),(np.ones(3),{'keepdims':True})])
def test_norm_other_variants_refuse(value,kwargs):
    with pytest.raises(QualificationDecline):api().BranchProfiles().norm(value,kwargs)


def test_json_profiles_bind_exact_effective_kwargs_and_closed_tree():
    profile=api().BranchProfiles()
    value={'x':[1,2.5,None,True,'text']}
    assert profile.json(value,dict(sort_keys=True,separators=(',',':'),allow_nan=False))=='json-stage'
    assert profile.json(value,dict(sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False))=='json-canonical'
    for candidate in ({'x':object()},{1:'key'},{'x':float('inf')}):
        with pytest.raises(QualificationDecline):profile.json(candidate,dict(sort_keys=True,separators=(',',':'),allow_nan=False))
    with pytest.raises(QualificationDecline):profile.json(value,dict(sort_keys=True,separators=(',',':'),allow_nan=False,default=lambda x:0))


def test_profile_detects_json_method_rebinding(monkeypatch):
    import json
    profile=api().BranchProfiles()
    monkeypatch.setattr(json.JSONEncoder,'encode',lambda self,x:'foreign')
    with pytest.raises(PlannerSourceChanged):profile.json({},dict(sort_keys=True,separators=(',',':'),allow_nan=False))


def test_profile_rejects_array_subclass_without_hooks():
    class Foreign(np.ndarray):
        def __array_function__(self,*args,**kwargs):pytest.fail('override executed')
    value=np.ones(3).view(Foreign)
    with pytest.raises(QualificationDecline):api().BranchProfiles().norm(value,{})


def test_initial_same_module_function_substitution_refuses(monkeypatch):
    from anygeometry import batch_intersections as batch
    monkeypatch.setattr(batch,'_pair_paths',batch._pair_paths_exact)
    with pytest.raises(QualificationDecline):api().BranchProfiles()


def test_non_native_endian_nan_is_not_read_as_native_finite_bytes():
    value=np.array([float('nan')],dtype='>f8' if np.little_endian else '<f8')
    assert value.dtype.type is np.float64 and not value.dtype.isnative
    with pytest.raises(QualificationDecline):api().BranchProfiles().norm(value,{})


def test_initial_json_native_replacement_refuses(monkeypatch):
    import json
    monkeypatch.setattr(json.encoder,'c_make_encoder',lambda *args:None)
    with pytest.raises(QualificationDecline):api().BranchProfiles()


def test_initial_surface_descriptor_replacement_does_not_execute(monkeypatch):
    from anygeometry.entities import Face
    monkeypatch.setattr(Face,'surface',property(lambda self:pytest.fail('foreign property executed')))
    with pytest.raises(QualificationDecline):api().BranchProfiles()


@pytest.mark.parametrize('which',['plane','coons','extruded','builtin'])
def test_initial_predicate_alias_or_builtin_shadow_refuses(monkeypatch,which):
    from anygeometry import intersections,nurbs_sweep_chart,extruded_supports,surfaces
    if which=='plane':monkeypatch.setattr(intersections,'Plane',surfaces.Cylinder)
    elif which=='coons':monkeypatch.setattr(nurbs_sweep_chart,'CoonsSurface',surfaces.Plane)
    elif which=='extruded':monkeypatch.setattr(extruded_supports,'ExtrudedSurface',surfaces.Plane)
    else:monkeypatch.setattr(intersections,'isinstance',lambda *args:pytest.fail('shadow called'),raising=False)
    with pytest.raises(QualificationDecline):api().BranchProfiles()


def test_face_namespace_custom_mapping_never_executes():
    from anygeometry._planning_canaries import _fixture
    model,_=_fixture();profile=api().BranchProfiles();identifier=next(iter(model.faces))
    class Foreign:
        def __getitem__(self,key):pytest.fail('mapping hook executed')
    model.faces=types.MappingProxyType(Foreign())
    with pytest.raises(QualificationDecline):profile.face(model,identifier)
