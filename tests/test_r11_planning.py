"""R11 controller hygiene and cost-gate contracts (no geometry/native execution)."""
from types import SimpleNamespace
import pytest
from anygeometry import _planning_controller as controller
from anygeometry.runtime_diagnostics import collect_runtime_records
from _planning_numpy_profile_contract import numpy_profile_test


def _policy(**kw):
    base=dict(max_predicates=None,max_candidate_pairs=None,cancellation_check=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _forbid_setup(monkeypatch):
    monkeypatch.setattr(controller,'_storage',lambda:pytest.fail('setup before routing'))
    monkeypatch.setattr(controller,'freeze_model',lambda m:pytest.fail('freeze before routing'))


@pytest.mark.parametrize('estimate',[(None,'unknown'),(0,'tiny'),(97,'sector'),(240,'r7'),('x','bad')])
def test_cheap_or_unknown_estimate_routes_serial_before_setup(monkeypatch,estimate):
    _forbid_setup(monkeypatch)
    monkeypatch.setattr(controller,'_estimate_cost',lambda model,operands=None:estimate)
    sentinel=object()
    with collect_runtime_records() as records:
        assert controller.run(None,lambda:sentinel,lambda b:pytest.fail('parallel'),
            policy=_policy(),mode='exact') is sentinel
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='serial' and 'routed serial' in receipt['reason']


def test_unsupported_policy_exits_before_estimate(monkeypatch):
    monkeypatch.setattr(controller,'_estimate_cost',lambda model,operands=None:pytest.fail('estimate'))
    sentinel=object()
    for kwargs in (dict(mode='fast'),dict(eligible=False)):
        args=dict(mode='exact');args.update(kwargs)
        assert controller.run(None,lambda:sentinel,None,policy=_policy(),**args) is sentinel
    assert controller.run(None,lambda:sentinel,None,policy=_policy(max_predicates=1),mode='exact') is sentinel


def test_estimate_is_passive_and_refuses_unknown_models():
    class Fake:
        @property
        def faces(self):raise AssertionError('property touched')
    assert controller._estimate_cost(Fake())[0] is None
    assert controller._estimate_cost(None)[0] is None


def test_estimate_counts_faces_of_real_model_without_properties():
    from anygeometry.model import GeometryModel
    model=GeometryModel(model_id='00000000-0000-0000-0000-000000000002')
    units,reason=controller._estimate_cost(model)
    assert units==0 and 'faces=0' in reason
    assert controller._ROUTING_THRESHOLD>240   # sector60/r7 scale stays serial
    assert controller._ROUTING_THRESHOLD<=561  # N2c scale may parallelize


def _run_fixture(monkeypatch,tmp_path):
    from test_planning_production_allowance import _fixture
    work,plan,policy,clock,events,seen,parallel=_fixture(monkeypatch,tmp_path)
    lease=controller.memory_profile().begin();checks=[]
    lease.check=lambda:checks.append(len(checks))
    return plan,policy,clock,checks


def test_memory_check_count_clock_wait_and_boundary_gating(monkeypatch,tmp_path):
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path);seen=[]
    def parallel(bridge):
        poll=bridge.poll;base=len(checks)
        for _ in range(100):poll()                       # cheap: no clock, count below 2048
        seen.append(('cheap',len(checks)-base));base=len(checks)
        poll.wait();seen.append(('wait',len(checks)-base));base=len(checks)
        poll.boundary();seen.append(('boundary',len(checks)-base));base=len(checks)
        clock[0]+=.03;poll();seen.append(('clock',len(checks)-base));base=len(checks)
        for _ in range(2047):poll()
        seen.append(('below-count',len(checks)-base));poll()
        seen.append(('count',len(checks)-base))
        return plan
    controller.run(None,lambda:pytest.fail('serial'),parallel,policy=policy,mode='exact')
    assert seen==[('cheap',0),('wait',1),('boundary',1),('clock',1),('below-count',0),('count',1)]


def test_callback_cancels_every_poll_without_full_verification(monkeypatch,tmp_path):
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path)
    verified=[];calls=[]
    monkeypatch.setattr(controller,'verify_sources',lambda source:verified.append(1))
    policy.cancellation_check=lambda:calls.append(1) or False
    out=[]
    def parallel(bridge):
        base=(len(calls),len(verified),len(checks))
        for _ in range(10):bridge.poll()
        out.append((len(calls)-base[0],len(verified)-base[1],len(checks)-base[2]))
        bridge.poll.boundary()
        out.append((len(calls)-base[0],len(verified)-base[1]))
        return plan
    controller.run(None,lambda:pytest.fail('serial'),parallel,policy=policy,mode='exact')
    assert out==[(10,0,0),(11,1)]   # callback each poll; verification only at boundary


def test_callback_cancel_and_exception_are_terminal_without_replay(monkeypatch,tmp_path):
    from anygeometry._planning_executor import PlannerCancelled
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path)
    policy.cancellation_check=lambda:True
    with collect_runtime_records() as records,pytest.raises(PlannerCancelled):
        controller.run(None,lambda:pytest.fail('serial'),lambda b:b.poll() or plan,policy=policy,mode='exact')
    assert records['parallel_planning'][-1]['disposition']=='terminal'


def test_callback_exception_is_preserved_and_terminal(monkeypatch,tmp_path):
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path)
    original=RuntimeError('callback boom');state=[0]
    def callback():
        state[0]+=1
        if state[0]>1:raise original
        return False
    policy.cancellation_check=callback
    with collect_runtime_records() as records,pytest.raises(BaseException) as caught:
        controller.run(None,lambda:pytest.fail('serial'),lambda b:b.poll() or plan,policy=policy,mode='exact')
    assert caught.value is original or getattr(caught.value,'error',None) is original
    assert records['parallel_planning'][-1]['disposition']=='terminal'


def test_model_mutation_detected_only_at_explicit_boundary_and_is_terminal(monkeypatch,tmp_path):
    from anygeometry import _planning_model_state
    from anygeometry._planning_executor import PlannerSourceChanged
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path)
    policy.cancellation_check=lambda:False
    changed=[False];cheap=[]
    monkeypatch.setattr(_planning_model_state,'production_snapshot',
        lambda model:{'sha256':'changed'} if changed[0] else {'sha256':'model'})
    def parallel(bridge):
        changed[0]=True
        bridge.poll();cheap.append('ordinary poll did not verify')
        bridge.poll.boundary()
        return plan
    with collect_runtime_records() as records,pytest.raises(PlannerSourceChanged):
        controller.run(None,lambda:pytest.fail('serial'),parallel,policy=policy,mode='exact')
    assert cheap and records['parallel_planning'][-1]['disposition']=='terminal'


def test_wait_poll_checks_memory_each_iteration_without_verification(monkeypatch,tmp_path):
    from anygeometry import _planning_host as host
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path)
    verified=[];monkeypatch.setattr(controller,'verify_sources',lambda s:verified.append(1))
    out=[]
    def parallel(bridge):
        base=(len(checks),len(verified))
        # Ordinary predicate polls (batch check_cancellation -> _stages.check_deadline):
        # throttled, never every call.
        for _ in range(100):bridge.check_deadline()
        out.append((len(checks)-base[0],len(verified)-base[1]))
        for _ in range(2048):bridge.check_deadline()
        out.append((len(checks)-base[0],len(verified)-base[1]))
        # Actual file/pipe wait callback: memory check on every iteration, no verification.
        base=(len(checks),len(verified))
        for _ in range(5):bridge.check_wait()
        out.append((len(checks)-base[0],len(verified)-base[1]))
        # The session wait path is handed check_wait (not check_deadline) by finish().
        base=(len(checks),len(verified))
        class Session:
            def finish(self,*,poll=None):
                assert poll == bridge.check_wait
                for _ in range(4):poll()
        bridge.session=Session()
        bridge.finish()
        bridge.session=None
        out.append((len(checks)-base[0],len(verified)-base[1]))
        return plan
    controller.run(None,lambda:pytest.fail('serial'),parallel,policy=policy,mode='exact')
    # 2 explicit boundaries in finish() (full checks incl. 1 verification each)
    # + 4 forced wait iterations.
    assert out==[(0,0),(1,0),(5,0),(6,2)]


def test_receipt_failure_replay_stops_child_memory_polling(monkeypatch,tmp_path):
    from pathlib import Path
    from test_planning_r8_controller import _finalization_fixture
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    lease=controller.memory_profile().begin();checks=[];at_return=[]
    lease.check=lambda:checks.append(1)
    original=Path.write_text
    def write(path,*args,**kwargs):
        if path.name=='receipt.json':raise OSError('receipt unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',write)
    def parallel(bridge):
        at_return.append(len(checks));return plan
    sentinel=object()
    with collect_runtime_records() as records:
        assert controller.run(None,lambda:sentinel,parallel,policy=policy,mode='exact') is sentinel
    assert records['parallel_planning'][-1]['disposition']=='serial-receipt-failure'
    assert len(checks)==at_return[0]


def test_unretired_receipt_failure_replay_is_not_attempted_as_child_free(monkeypatch,tmp_path):
    """Sticky retirement error stays terminal; no replay or publication."""
    from anygeometry._planning_executor import PlannerCleanupError
    plan,policy,clock,checks=_run_fixture(monkeypatch,tmp_path)
    cleanup=PlannerCleanupError('sticky cleanup')
    from anygeometry import _planning_host as host
    def close(self):
        self.retired=False;self._retirement_error=cleanup;raise cleanup
    monkeypatch.setattr(controller._ProductionBridge,'close',close)
    clock[0]=0.
    with collect_runtime_records() as records,pytest.raises(PlannerCleanupError) as caught:
        controller.run(None,lambda:pytest.fail('replay'),lambda b:plan,policy=policy,mode='exact')
    assert caught.value is cleanup
    assert records['parallel_planning'][-1]['disposition']=='terminal'


def _row(width):
    return ('a'*64,tuple(('edge',index,float(index),b'x',frozenset((index,None,True))) for index in range(width)))


def test_authority_digest_is_exact_typed_and_deterministic():
    from anygeometry import _planning_manifest as m
    first=m._document_authority_digest(_row(5),[],[])
    assert first==m._document_authority_digest(_row(5),[],[])
    assert first[0]=='authority-sha256' and len(first[1])==64
    # typed: True/1, 1/1.0, str/bytes never collide
    digests={m._document_authority_digest(v,[],[])[1] for v in (True,1,1.0,'1',b'1',(1,),frozenset((1,)))}
    assert len(digests)==7
    # frozenset order independent, tuple order dependent
    assert (m._document_authority_digest(frozenset((1,2,3)),[],[])==
        m._document_authority_digest(frozenset((3,2,1)),[],[]))
    assert m._document_authority_digest((1,2),[],[])!=m._document_authority_digest((2,1),[],[])
    assert m._document_authority_digest(_row(5),[],[])!=m._document_authority_digest(_row(6),[],[])


def test_large_authority_has_independent_bound_and_one_callable_node():
    from anygeometry import _planning_manifest as m
    from anygeometry._planning_qualification import QualificationDecline
    big=_row(7000)             # ~63k authority nodes, far over any callable share
    digest=m._document_authority_digest(big,[],[])
    assert digest[2]>50000
    assert len(repr(digest))<200           # callable graph receives only a fixed-size digest
    with pytest.raises(QualificationDecline,match='80000'):
        m._document_authority_digest(_row(9000),[],[])
    with pytest.raises(QualificationDecline,match='80000'):
        m._document_authority_digest(_row(10),[],[],limit=5)


def test_authority_refuses_unsupported_cyclic_and_mutated_values():
    from anygeometry import _planning_manifest as m
    from anygeometry._planning_qualification import QualificationDecline
    for bad in (object(),{'a':1},[1],{1},1+2j,('ok',[1]),bytearray(b'x')):
        with pytest.raises(QualificationDecline):m._document_authority_digest(bad,[],[])
    deep=()
    for _ in range(300):deep=(deep,)
    with pytest.raises(QualificationDecline,match='depth'):m._document_authority_digest(deep,[],[])
    class Sub(tuple):pass
    with pytest.raises(QualificationDecline):m._document_authority_digest(Sub((1,)),[],[])  # exact type only


def test_authority_types_retain_owner_references_and_refuse_foreign_types():
    from anygeometry import _planning_manifest as m
    from anygeometry._planning_qualification import QualificationDecline
    references=[];identities=[]
    m._document_authority_digest(('x',m.QualificationDecline),references,identities)
    assert references==[m.QualificationDecline] and identities==[('authority-type',id(m.QualificationDecline))]
    class Local:pass
    with pytest.raises(QualificationDecline,match='owner'):
        m._document_authority_digest((Local,),[],[])


def test_authority_mismatch_changes_digest_for_any_row_mutation():
    from anygeometry import _planning_manifest as m
    base=_row(4);digest=m._document_authority_digest(base,[],[])
    changed=(base[0][:-1]+'b',base[1])
    assert m._document_authority_digest(changed,[],[])!=digest
    swapped=(base[0],base[1][:2]+(('edge',2,2.5,b'x',frozenset((2,None,True))),)+base[1][3:])
    assert m._document_authority_digest(swapped,[],[])!=digest


@pytest.mark.slow
@numpy_profile_test
def test_production_manifest_authority_is_one_node_independent_and_row_bound():
    """Real capture: large authority rows stop consuming callable budget; alterations differ."""
    from anygeometry import _planning_manifest as m, serialization
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_qualification import QualificationDecline
    from anygeometry._planning_executor import PlannerSourceChanged
    model,_=_fixture();serialization.to_dict(model)
    registry=serialization._qualified_documents
    original=registry[model]
    # capture_loaded_manifest() freeze-normalizes (regenerates) the registry row, so
    # synthetic rows are installed and captured with LoadedManifest directly.
    def capture(row):
        registry[model]=row
        return m.LoadedManifest(model=model)
    def records(captured,role):
        return [t for t in captured.roles[role]._last_tokens if len(t)>2 and t[1]=='request-document-authority']
    try:
        base=capture(original)
        for role in ('constructor','host'):
            assert len(records(base,role))==1       # exactly one fixed record per role
        again=capture(original)
        assert {r:again.roles[r].digest for r in again.roles}=={r:base.roles[r].digest for r in base.roles}
        altered=capture((original[0][:-1]+('0' if original[0][-1]!='0' else '1'),original[1]))
        for role in ('constructor','host'):
            assert len(records(altered,role))==1 and records(altered,role)!=records(base,role)
            assert altered.roles[role].digest!=base.roles[role].digest
        assert altered.callables.digest==base.callables.digest   # common role carries no document authority
        # ~60k authority nodes would have exhausted the shared 80k callable budget.
        large=capture((original[0],(original[1],tuple(range(60000)))))
        for role in ('constructor','host'):
            assert large.roles[role].digest not in (base.roles[role].digest,altered.roles[role].digest)
            assert len(records(large,role))==1 and len(repr(records(large,role)[0]))<300
            assert large.roles[role].graph==base.roles[role].graph
        assert large.callables.digest==base.callables.digest
        with pytest.raises(QualificationDecline,match='80000'):
            capture((original[0],(original[1],tuple(range(80001)))))
        with pytest.raises(QualificationDecline):capture((original[0],(original[1],[1])))   # unsupported value
        del registry[model]
        with pytest.raises(QualificationDecline,match='absent'):m.LoadedManifest(model=model)
        # Post-capture mutation of the live row is refused by the retained check.
        live=capture(original);live.verify()
        registry[model]=(original[0],(original[1],1))
        with pytest.raises((QualificationDecline,PlannerSourceChanged)):live.verify()
    finally:
        registry[model]=original
    # Public wrapper: same admitted document (normalized by freeze) agrees.
    first=m.capture_loaded_manifest(model=model);second=m.capture_loaded_manifest(model=model)
    assert {r:first.roles[r].digest for r in first.roles}=={r:second.roles[r].digest for r in second.roles}


def test_requested_scope_not_container_size_drives_actual_selector(monkeypatch):
    """Actual (unpatched) estimate: a large container with a small requested scope stays serial."""
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_qualification import QualificationDecline
    model,_=_fixture();faces=tuple(model.faces)
    assert len(faces)>=2
    handles=tuple(sorted((model.handle('face',f) for f in faces),key=lambda h:h.key))
    assert controller._estimate_cost(model,handles)[0]==len(faces)
    assert controller._estimate_cost(model,handles[:1])[0]==1
    assert controller._estimate_cost(model)[0]==len(model.faces)
    monkeypatch.setattr(controller,'_ROUTING_THRESHOLD',len(faces))   # private threshold only
    policy=_policy();sentinel=object()
    # Small scope in a model at/above threshold: serial before setup.
    monkeypatch.setattr(controller,'_storage',lambda:pytest.fail('setup for small requested scope'))
    with collect_runtime_records() as records:
        assert controller.run(model,lambda:sentinel,lambda b:pytest.fail('parallel'),policy=policy,
            mode='exact',route_operands=handles[:1]) is sentinel
    assert 'requested faces=1' in records['parallel_planning'][-1]['reason']
    # Large requested scope is eligible: it reaches setup (refused here, replayed serial).
    monkeypatch.setattr(controller,'_storage',lambda:(_ for _ in ()).throw(QualificationDecline('scope admitted')))
    with collect_runtime_records() as records:
        assert controller.run(model,lambda:sentinel,lambda b:pytest.fail('parallel'),policy=policy,
            mode='exact',route_operands=handles) is sentinel
    assert records['parallel_planning'][-1]['reason']=='scope admitted'


def test_non_face_or_foreign_operands_are_unknown_and_serial():
    from anygeometry._planning_canaries import _fixture
    model,_=_fixture()
    assert controller._estimate_cost(model,[object()])[0] is None
    assert controller._estimate_cost(model,(object(),))[0] is None
    assert controller._estimate_cost(model,(model.handle('face',tuple(model.faces)[0]),object()))[0] is None


def test_memory_gate_constants():
    assert controller._MEMORY_POLL_INTERVAL==2048
    assert controller._MEMORY_SECONDS==.025
