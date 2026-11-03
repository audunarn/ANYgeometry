"""Completed selector guards connect to an actual callable manifest and call."""
import pytest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_executor import PlannerSourceChanged


@pytest.mark.parametrize('which',['chart','extruded','plane'])
def test_guarded_selector_capture_and_actual_call(which):
    from anygeometry._planning_profiles import BranchProfiles
    from anygeometry._planning_canaries import _fixture
    from anygeometry import nurbs_sweep_chart,extruded_supports,intersections
    model,_=_fixture();identifier=next(iter(model.faces));face=model.faces[identifier]
    profile=BranchProfiles()
    function,args,expected={
        'chart':(nurbs_sweep_chart.has_sweep_chart,(face,),False),
        'extruded':(extruded_supports.extruded_support,(model,identifier),None),
        'plane':(intersections._qualified_face_plane,(model,identifier),face.surface)}[which]
    captured=profile.capture(function,*args)
    assert captured.manifest.nodes<100
    assert captured() is expected


def test_guarded_selector_checks_argument_domain_again_before_call():
    from anygeometry._planning_profiles import BranchProfiles
    from anygeometry._planning_canaries import _fixture
    from anygeometry.nurbs_sweep_chart import has_sweep_chart
    model,_=_fixture();face=next(iter(model.faces.values()))
    captured=BranchProfiles().capture(has_sweep_chart,face)
    old=face.surface
    try:
        object.__setattr__(face,'surface',object())
        with pytest.raises(PlannerSourceChanged):captured()
    finally:object.__setattr__(face,'surface',old)


def test_uncompleted_profile_does_not_exclude_dependency_body():
    from anygeometry._planning_profiles import BranchProfiles
    from anygeometry.batch_intersections import _pair_paths_exact
    with pytest.raises(QualificationDecline):BranchProfiles().capture(_pair_paths_exact,object(),object())
