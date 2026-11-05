"""Runtime observation units do not qualify native/parallel execution."""
import json
from types import SimpleNamespace
import pytest
from _planning_runtime_observation import capture_witness, retain_observation


def test_witness_requires_actual_controller_poll(monkeypatch,tmp_path):
    from test_planning_r8_controller import _finalization_fixture
    from anygeometry import _planning_controller as controller
    from anygeometry._planning_executor import PlannerCancelled
    _,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    seen=[]
    def callback():
        seen.append(capture_witness())
        return seen[-1]['captured']
    policy.cancellation_check=callback
    with pytest.raises(PlannerCancelled):
        controller.run(None,lambda:pytest.fail('unexpected replay'),lambda bridge:plan,policy=policy,mode='exact')
    assert [row['captured'] for row in seen]==[False,True]
    assert all(row['exact_poll'] for row in seen)
    assert seen[-1]['manifest_sha256']=='manifest'


def test_same_named_foreign_poll_does_not_attest_capture():
    def poll():
        source=manifest=bridge=object()
        return callback()
    def callback():return capture_witness()
    assert poll()==dict(exact_poll=False,captured=False)


@pytest.mark.parametrize('status',['parallel','serial','terminal'])
def test_full_receipt_retained_without_claim_or_filter(tmp_path,monkeypatch,status):
    monkeypatch.delenv('CI',raising=False)
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    receipt=dict(disposition=status,reason='exact unknown reason',stages=[],manifest_sha256='m')
    records={'parallel_planning':[receipt]}
    with retain_observation(request,records,label='warm8-warm',witness=[dict(captured=False)]):pass
    got=json.loads((tmp_path/'runtime-observation-warm8-warm.json').read_text())
    assert got['status']=='UNVALIDATED' and got['records']==records
    assert got['witness']==[dict(captured=False)]
    assert got['parallel_qualified'] is False


def test_retention_error_never_masks_original_callback_error(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.delenv('CI',raising=False)
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    original=RuntimeError('same callback exception')
    def refused(*args,**kwargs):raise OSError('write refused')
    monkeypatch.setattr(Path,'open',refused)
    with pytest.raises(RuntimeError) as caught:
        with retain_observation(request,{},label='callback-exception'):raise original
    assert caught.value is original
    assert any('retention failed' in note for note in original.__notes__)


@pytest.mark.parametrize('fault',['oversized','foreign','symlink','label'])
def test_observation_rejects_unowned_or_unbounded_metadata(tmp_path,monkeypatch,fault,request):
    from _planning_runtime_observation import FILE_LIMIT
    monkeypatch.delenv('CI',raising=False)
    base=tmp_path;records={}
    if fault=='oversized':records={'value':'x'*FILE_LIMIT}
    elif fault=='foreign':records={'value':object()}
    elif fault=='symlink':
        # Portable predicate injection: no OS symlink privilege/qualification.
        from pathlib import Path
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda self:self==base or original(self))
        request.node.user_properties.append(('execution_scope','filesystem-predicate-injection-unit'))
    request=SimpleNamespace(getfixturevalue=lambda name:base)
    with pytest.raises((ValueError,TypeError)):
        with retain_observation(request,records,label='unknown' if fault=='label' else 'callback-code'):pass


def test_ci_copy_preserves_exact_owned_bytes(tmp_path,monkeypatch):
    import _planning_runtime_observation as helper
    monkeypatch.setenv('CI','true')
    monkeypatch.setattr(helper,'_ROOT',tmp_path)
    owned=tmp_path/'test';owned.mkdir()
    request=SimpleNamespace(getfixturevalue=lambda name:owned)
    with retain_observation(request,{'parallel_planning':[{'disposition':'serial','reason':'unelided'}]},label='warm8-cold'):pass
    targets=list((tmp_path/'ci-isolation'/'runtime-observations').glob('*/observation.json'))
    assert len(targets)==1
    assert targets[0].read_bytes()==(owned/'runtime-observation-warm8-cold.json').read_bytes()


def test_initial_decline_is_failed_positive_with_full_receipt(monkeypatch,tmp_path):
    from test_planning_r8_controller import _finalization_fixture
    from anygeometry import _planning_controller as controller,_planning_manifest
    from anygeometry._planning_qualification import QualificationDecline
    from anygeometry.runtime_diagnostics import collect_runtime_records
    monkeypatch.delenv('CI',raising=False)
    _,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    def decline(**kwargs):raise QualificationDecline('exact initial decline')
    monkeypatch.setattr(_planning_manifest,'capture_loaded_manifest',decline)
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    seen=[]
    def callback():
        seen.append(capture_witness())
        return False
    policy.cancellation_check=callback
    with collect_runtime_records() as records:
        with retain_observation(request,records,label='warm8-warm',witness=seen):
            result=controller.run(None,lambda:plan,lambda bridge:pytest.fail('not admitted'),policy=policy,mode='exact')
        assert result is plan
        receipt=records['parallel_planning'][-1]
        with pytest.raises(AssertionError):assert receipt['disposition']=='parallel',receipt
    assert all(not row['captured'] for row in seen)
    evidence=json.loads((tmp_path/'runtime-observation-warm8-warm.json').read_text())
    assert evidence['records']['parallel_planning'][-1]['reason']=='exact initial decline'
    assert evidence['records']['parallel_planning'][-1]['disposition']=='serial'


def test_actual_callback_test_body_uses_captured_poll(monkeypatch,tmp_path):
    # Execute the actual body, not the immutable-profile decorator: component only.
    import inspect
    from test_planning_r8_controller import (_finalization_fixture,
        test_real_public_callback_after_capture_is_terminal as actual_test)
    from anygeometry import _planning_controller as controller,batch_intersections as bi,_planning_canaries
    monkeypatch.delenv('CI',raising=False)
    _,plan,_=_finalization_fixture(monkeypatch,tmp_path)
    model=SimpleNamespace(faces={1:None})
    monkeypatch.setattr(_planning_canaries,'_fixture',lambda:(model,None))
    def plan_component(value,faces,*,policy,_parallel_control):
        assert value is model and faces==(1,) and _parallel_control is None
        return controller.run(model,lambda:pytest.fail('unexpected serial replay'),lambda bridge:plan,policy=policy,mode='exact')
    monkeypatch.setattr(bi,'plan_intersections',plan_component)
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    inspect.unwrap(actual_test)(monkeypatch,tmp_path,'cancel',request)
    evidence=json.loads((tmp_path/'runtime-observation-callback-cancel.json').read_text())
    assert [row['captured'] for row in evidence['witness']]==[False,True]
    terminal=evidence['records']['parallel_planning'][-1]
    assert terminal['disposition']=='terminal' and terminal['retired'] is True


@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_retention_terminal_preserves_original_in_context(tmp_path,monkeypatch,kind):
    from pathlib import Path
    monkeypatch.delenv('CI',raising=False)
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    original=RuntimeError('same primary callback failure')
    terminal=kind('retention terminal')
    def interrupted(*args,**kwargs):raise terminal
    monkeypatch.setattr(Path,'open',interrupted)
    with pytest.raises(kind) as caught:
        with retain_observation(request,{},label='callback-exception'):raise original
    assert caught.value is terminal
    assert caught.value.__context__ is original


def test_existing_observation_target_is_not_overwritten(tmp_path,monkeypatch):
    monkeypatch.delenv('CI',raising=False)
    target=tmp_path/'runtime-observation-callback-code.json'
    target.write_bytes(b'original owned bytes')
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    with pytest.raises(ValueError,match='already exists'):
        with retain_observation(request,{},label='callback-code'):pass
    assert target.read_bytes()==b'original owned bytes'


def test_exclusive_creation_rejects_racing_target(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.delenv('CI',raising=False)
    original=Path.open
    target=tmp_path/'runtime-observation-callback-code.json'
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path)
    def racing(path,mode='r',*args,**kwargs):
        if path==target and mode=='xb':
            with original(path,'wb') as stream:stream.write(b'competing owned bytes')
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,'open',racing)
    with pytest.raises(FileExistsError):
        with retain_observation(request,{},label='callback-code'):pass
    assert target.read_bytes()==b'competing owned bytes'


def test_both_observation_writes_use_exclusive_creation(tmp_path,monkeypatch):
    from pathlib import Path
    import _planning_runtime_observation as helper
    monkeypatch.setenv('CI','true');monkeypatch.setattr(helper,'_ROOT',tmp_path)
    base=tmp_path/'owned';base.mkdir()
    original=Path.open;writes=[]
    def observed(path,mode='r',*args,**kwargs):
        if any(flag in mode for flag in 'wax'):writes.append(mode)
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,'open',observed)
    request=SimpleNamespace(getfixturevalue=lambda name:base)
    with retain_observation(request,{},label='callback-code'):pass
    assert writes==['xb','xb']
