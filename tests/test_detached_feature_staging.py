"""Disposable staging may reuse topology without weakening public atomic replay."""
import pytest
from anygeometry import GeometryModel,GeometryError,to_dict
from anygeometry.features import builtin_feature_registry


def source_model():
    model=GeometryModel()
    model.add_point(-1,0,0)
    model.features.capture_baseline(model)
    first=model.features.append("geometry.point",parameters={"position":[1,0,0]})
    assert model.regenerate_features().success
    return model,first.feature_id


def semantic(model):
    payload=to_dict(model)
    for key in ("model_id","revision","checksum"):payload.pop(key,None)
    return payload


def test_appended_stage_reuses_topology_and_matches_atomic_ids_and_state(monkeypatch):
    source,_=source_model();before=to_dict(source)
    counters=source.id_state();counters["vertex"]=100
    source.reserve_id_state(counters);before=to_dict(source)
    stage=source.clone();control=source.clone()
    added=stage.features.append("geometry.point",parameters={"position":[2,0,0]})
    control.features.append("geometry.point",parameters={"position":[2,0,0]})
    expected=control.regenerate_features();assert expected.success
    def no_clone(*args,**kwargs):raise AssertionError("duplicated staging clone")
    monkeypatch.setattr(stage,"clone",no_clone)
    report=stage.regenerate_features_detached(appended_feature_ids=(added.feature_id,))
    assert report==expected
    assert semantic(stage)==semantic(control)
    assert stage.features.get(added.feature_id).outputs["point"].id==100
    assert to_dict(source)==before
    # External record/snapshot changes still cannot alias either owner.
    snapshot=stage.design_snapshot()
    snapshot["features"]["baseline"]["vertices"].clear()
    snapshot["features"]["records"][-1].parameters["position"][0]=99
    assert semantic(stage)==semantic(control) and to_dict(source)==before


@pytest.mark.parametrize("change",["edit","suppress","missing_ids","wrong_ids"])
def test_other_feature_edits_keep_atomic_clone_path(monkeypatch,change):
    stage,identifier=source_model();calls=[]
    if change=="edit":stage.features.update(identifier,parameters={"position":[3,0,0]});ids=(identifier,)
    elif change=="suppress":stage.features.set_suppressed(identifier);ids=(identifier,)
    else:
        added=stage.features.append("geometry.point",parameters={"position":[3,0,0]})
        ids=() if change=="missing_ids" else (added.feature_id+1,)
    original=stage.clone
    def clone(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(stage,"clone",clone)
    report=stage.regenerate_features_detached(appended_feature_ids=ids)
    assert report.success and calls


def test_late_refusal_keeps_public_atomicity_and_source_isolation():
    source,_=source_model();source_before=to_dict(source)
    for detached in (False,True):
        stage=source.clone()
        one=stage.features.append("geometry.point",parameters={"position":[3,0,0]})
        two=stage.features.append("geometry.point",parameters={"position":[4,0,0,0]})
        before=to_dict(stage)
        report=(stage.regenerate_features_detached(appended_feature_ids=(one.feature_id,two.feature_id))
                if detached else stage.regenerate_features())
        assert not report.success
        if detached:assert len(stage.vertices)>len(before["vertices"])
        else:assert to_dict(stage)==before
        assert to_dict(source)==source_before


def test_mutable_payload_is_revalidated_before_staging_execution():
    stage,_=source_model()
    added=stage.features.append("geometry.point",parameters={"position":[3,0,0]})
    before=set(stage.vertices)
    stage.features._records[-1].parameters["position"][0]=float("nan")
    with pytest.raises(GeometryError,match="finite"):
        stage.regenerate_features_detached(appended_feature_ids=(added.feature_id,))
    assert set(stage.vertices)==before


def test_cancelled_disposable_execution_cannot_mutate_source():
    class Cancelled(BaseException):pass
    source,_=source_model();before=to_dict(source);stage=source.clone()
    registry=builtin_feature_registry()
    def cancel(geometry,feature,inputs):
        geometry.add_point(9,0,0)
        raise Cancelled()
    registry.register("test.cancel",cancel,replay_mode="additive")
    added=stage.features.append("test.cancel")
    with pytest.raises(Cancelled):
        stage.regenerate_features_detached(registry,appended_feature_ids=(added.feature_id,))
    assert to_dict(source)==before


def test_explicit_mutating_suffix_keeps_atomic_clone_path(monkeypatch):
    from anygeometry import EntityRef
    stage,_=source_model();registry=builtin_feature_registry();calls=[]
    def execute(geometry,feature,inputs):
        return {"point":EntityRef("vertex",geometry.add_point(9,0,0))}
    registry.register("test.mutating",execute,replay_mode="mutating")
    added=stage.features.append("test.mutating")
    original=stage.clone
    def clone(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(stage,"clone",clone)
    report=stage.regenerate_features_detached(registry,appended_feature_ids=(added.feature_id,))
    assert report.success and calls==[True]
