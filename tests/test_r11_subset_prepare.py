"""Bounded-fake contracts of tools/r11_subset_prepare.py (no model, no meshing, no workers)."""
import contextlib
import dataclasses
import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    name = 'r11_subset_prepare_under_test'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / 'r11_subset_prepare.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()
FULL = list(range(1, 11))
SELECTED = (2, 4, 6)


@dataclasses.dataclass
class FakeClosure:
    working_model: object
    source_to_work: dict


class Handle:
    def __init__(self, kind, identifier):
        self.kind, self.id = kind, identifier

    def __hash__(self):
        return hash((self.kind, self.id))

    def __eq__(self, other):
        return (self.kind, self.id) == (other.kind, other.id)


class Geometry:
    faces = {i: None for i in FULL}

    def handle(self, kind, identifier):
        return Handle(kind, identifier)


def make_world(log, *, working_faces=None, mapped=None, alias_mismatch=False, mesher_returns=False,
               plan_counts=None, forward=None, construct=None):
    """Fake owner namespaces whose originals log what runs; the mesher original must NEVER run."""
    working = SimpleNamespace(faces={i: None for i in (SELECTED if working_faces is None else working_faces)})
    evidence = []
    calls = {'prepare': 0}

    def original_closure(geometry, handles, **kwargs):
        log.append(('closure', tuple(sorted(h.id for h in handles)), dict(kwargs)))
        ids = SELECTED if mapped is None else mapped
        return FakeClosure(working, {Handle('face', i): Handle('face', i) for i in ids})

    plan = SimpleNamespace(content_checksum='plan-1')

    def original_prepare(model, **kwargs):
        count = 1 if plan_counts is None else plan_counts[calls['prepare']]
        calls['prepare'] += 1
        log.append(('prepare', os.environ.get(tool.ENV), dict(kwargs)))
        for _ in range(count):
            evidence.append({'plan_checksum': 'plan-1', 'plan': plan})
        return SimpleNamespace(report='r')

    def original_mesher(*args, **kwargs):
        log.append('MESH')                                                   # must never be reached

    other_prepare = (lambda *a, **k: None) if alias_mismatch else original_prepare
    namespaces = dict(
        project=SimpleNamespace(extract_model_closure=original_closure, prepare_structural_connectivity=original_prepare),
        backend=SimpleNamespace(prepare_structural_connectivity=other_prepare, generate_hybrid_mesh_result=original_mesher),
        preparation=SimpleNamespace(prepare_structural_connectivity=original_prepare),
        hybrid=SimpleNamespace(generate_hybrid_mesh_result=original_mesher))

    @contextlib.contextmanager
    def collectors():
        evidence.clear()
        yield (evidence, {'t2b_pairs': 3, 'x_time_ns': 9}, {'parallel_planning': [{
            'reason': 'routed serial: tiny', 'routing': 'routed serial: tiny', 'disposition': 'serial', 'retired': True,
            'stages': []}]})

    class Project:
        _mesh_preparation_cache = None

        def __init__(self):
            if construct is not None:
                construct(namespaces)

        def generate_mesh(self, **kwargs):
            ns = namespaces
            log.append(('generate', dict(kwargs)))
            handles = tuple(Handle('face', i) for i in FULL)                  # the project asks for the WHOLE model
            closure = ns['project'].extract_model_closure(Geometry(), handles, include_structural_closure=True, include_features=False)
            ns['project'].prepare_structural_connectivity(
                closure.working_model, source_model_id='m', source_revision=1, member_ids=(), cancellation_check=None)
            log.append('post-prep')
            options = dict(target_size=0.25, order='linear', strategy='native', layout_policy='adaptive',
                           face_ids=tuple(sorted(closure.working_model.faces)))
            options.update(forward or {})
            geometry = options.pop('geometry', closure.working_model)
            for name in [k for k, v in options.items() if v is Ellipsis]:
                del options[name]
            ns['hybrid'].generate_hybrid_mesh_result(geometry, **options)
            if mesher_returns:
                return 'mesh'
            log.append('AFTER-MESHER-STOP')

    def hooks_factory(recorder):
        return tool.Hooks(recorder, original_closure=original_closure, to_dict=lambda m: {'doc': sorted(m.faces)},
                          collectors=collectors, **namespaces)
    return namespaces, Project, hooks_factory, original_closure, original_prepare, original_mesher, collectors


def test_select_case_is_fail_closed():
    ids = list(range(1, 98))
    data = dict(subsets=dict(elbow_sector60=ids))
    faces = list(range(1, tool.FULL_FIXTURE_FACES + 1))
    assert tool.select_case(data, 'elbow_sector60', total_faces=faces) == tuple(ids)
    bad = {
        'wrong count': dict(subsets=dict(elbow_sector60=ids[:-1])), 'duplicate': dict(subsets=dict(elbow_sector60=ids[:-1] + [1])),
        'not int': dict(subsets=dict(elbow_sector60=ids[:-1] + ['x'])), 'absent id': dict(subsets=dict(elbow_sector60=ids[:-1] + [5000])),
        'missing': dict(subsets={}), 'not a list': dict(subsets=dict(elbow_sector60='x')), 'no subsets': {},
    }
    for name, data_bad in bad.items():
        with pytest.raises(tool.HarnessError):
            tool.select_case(data_bad, 'elbow_sector60', total_faces=faces)
    with pytest.raises(tool.HarnessError):
        tool.select_case(data, 'elbow_sector61', total_faces=faces)
    with pytest.raises(tool.HarnessError):
        tool.select_case(data, 'elbow_sector60', total_faces=faces[:500])
    with pytest.raises(tool.HarnessError):
        tool.select_case(data, 'elbow_sector60', total_faces=faces, sheets=1)
    with pytest.raises(tool.HarnessError):
        tool.select_case(data, 'elbow_sector60', total_faces=faces, members=2)
    assert tool.select_case(dict(subsets=dict(canary3=[1, 2, 3])), 'canary3', total_faces=[1, 2, 3]) == (1, 2, 3)


def test_both_aliases_are_hooked_during_the_row_and_restored_afterwards():
    log = []
    ns, Project, factory, closure, prepare, mesher, _ = make_world(log)
    seen = {}

    class Probe(Project):
        def generate_mesh(self, **kwargs):
            seen['project_prepare'] = ns['project'].prepare_structural_connectivity
            seen['backend_prepare'] = ns['backend'].prepare_structural_connectivity
            seen['hybrid'] = ns['hybrid'].generate_hybrid_mesh_result
            seen['backend_hybrid'] = ns['backend'].generate_hybrid_mesh_result
            return super().generate_mesh(**kwargs)
    row, recorder = tool.run_row('cold', Probe, factory, SELECTED)
    assert seen['project_prepare'] is seen['backend_prepare'] and seen['project_prepare'] is not prepare
    assert seen['hybrid'] is seen['backend_hybrid'] and seen['hybrid'] is not mesher
    assert ns['project'].prepare_structural_connectivity is prepare and ns['backend'].prepare_structural_connectivity is prepare
    assert ns['project'].extract_model_closure is closure and ns['hybrid'].generate_hybrid_mesh_result is mesher
    assert ns['backend'].generate_hybrid_mesh_result is mesher

    class Boom(Project):
        def generate_mesh(self, **kwargs):
            raise RuntimeError('boom')
    with pytest.raises(RuntimeError):
        tool.run_row('cold', Boom, factory, SELECTED)
    assert ns['project'].prepare_structural_connectivity is prepare and ns['hybrid'].generate_hybrid_mesh_result is mesher


def test_stop_is_a_baseexception_before_any_meshing_and_orders_the_metrics():
    assert issubclass(tool.StopStage, BaseException) and not issubclass(tool.StopStage, Exception)
    log = []
    _, Project, factory, *_ = make_world(log)
    row, recorder = tool.run_row('cold', Project, factory, SELECTED)
    assert 'MESH' not in log and 'AFTER-MESHER-STOP' not in log
    assert row['events'] == ['generate_mesh_call', 'closure_entry', 'closure_exit', 'prep_entry', 'prep_exit', 'mesher_entry']
    t = recorder.t
    assert t['generate_mesh_call'] <= t['closure_entry'] <= t['closure_exit'] <= t['prep_entry'] <= t['prep_exit'] <= t['mesher_entry']
    assert row['generate_call_to_mesher_entry_seconds'] >= row['prep_entry_to_mesher_entry_seconds'] >= row['prep_seconds'] >= 0
    assert row['initial_document_capture_seconds_inside_timer'] <= row['prep_seconds']
    assert row['preparation_calls'] == row['plans_observed'] == 1 and row['meshing_calls'] == 0
    assert recorder.working_doc == {'doc': list(SELECTED)}
    closure_call = next(entry for entry in log if type(entry) is tuple and entry[0] == 'closure')
    assert closure_call[1] == SELECTED and closure_call[2] == dict(include_structural_closure=False, include_features=False)
    generate_call = next(entry for entry in log if type(entry) is tuple and entry[0] == 'generate')
    assert generate_call[1] == tool.GENERATE_ARGS == dict(target_size=0.25, strategy='quad_first', order='linear',
        layout_policy='adaptive', quad_t3_minimum_angle=15, quad_t3_angle_action='error')
    assert row['forwarded_settings']['strategy'] == 'native' and row['forwarded_face_ids'] == list(SELECTED)
    _, Returning, factory2, *_ = make_world([])
    class NoStop(Returning):
        def generate_mesh(self, **kwargs):
            return 'mesh'
    with pytest.raises(tool.HarnessError):
        tool.run_row('cold', NoStop, factory2, SELECTED)


def test_the_timestamp_is_taken_before_the_mesher_arguments_are_inspected(monkeypatch):
    order = []
    original = tool.Recorder.inspect_mesher

    def spy(self, args, kwargs):
        order.append(('inspect', 'mesher_entry' in self.t, self.events[-1]))
        return original(self, args, kwargs)
    monkeypatch.setattr(tool.Recorder, 'inspect_mesher', spy)
    _, Project, factory, *_ = make_world([])
    tool.run_row('cold', Project, factory, SELECTED)
    assert order == [('inspect', True, 'mesher_entry')]


@pytest.mark.parametrize('forward,needle', [
    (dict(geometry=SimpleNamespace(faces={2: None, 4: None, 6: None})), 'prepared working model'),
    (dict(face_ids=Ellipsis), 'face_ids'), (dict(face_ids=(2, 4)), 'face_ids'),
    (dict(target_size=0.3), 'target_size'), (dict(order='quadratic'), 'order'), (dict(strategy='quad_first'), 'strategy'),
    (dict(layout_policy='existing'), 'layout_policy'), (dict(target_size=Ellipsis), 'target_size')])
def test_mesher_entry_contract_violations_are_refused_after_the_stop_and_never_mesh(forward, needle):
    log = []
    _, Project, factory, *_ = make_world(log, forward=forward)
    with pytest.raises(tool.HarnessError, match='mesher-entry contract') as caught:
        tool.run_row('cold', Project, factory, SELECTED)
    assert needle in str(caught.value) and 'MESH' not in log and 'AFTER-MESHER-STOP' not in log


@pytest.mark.parametrize('kwargs', [dict(working_faces=(2, 4, 6, 8)), dict(working_faces=(2, 4)), dict(mapped=(2, 4, 8))])
def test_subset_scope_violations_refuse_before_preparation(kwargs):
    log = []
    _, Project, factory, *_ = make_world(log, **kwargs)
    with pytest.raises(tool.HarnessError, match='exactly the selected'):
        tool.run_row('cold', Project, factory, SELECTED)
    assert not any(type(entry) is tuple and entry[0] == 'prepare' for entry in log)


def test_alias_identity_mismatch_refuses_before_anything_is_installed():
    log = []
    ns, Project, factory, closure, prepare, mesher, _ = make_world(log, alias_mismatch=True)
    with pytest.raises(tool.HarnessError, match='cached prepare aliases'):
        tool.run_row('cold', Project, factory, SELECTED)
    assert ns['project'].extract_model_closure is closure and ns['hybrid'].generate_hybrid_mesh_result is mesher


def test_row_requires_an_initially_empty_preparation_cache():
    _, Project, factory, *_ = make_world([])
    class Warm(Project):
        _mesh_preparation_cache = object()
    with pytest.raises(tool.HarnessError, match='_mesh_preparation_cache'):
        tool.run_row('warm', Warm, factory, SELECTED)


@pytest.mark.parametrize('alias,needle', [('closure', 'closure alias invoked during Project construction'),
                                          ('prepare', 'preparation alias invoked during Project construction'),
                                          ('mesher', 'constructor reached the mesher stop')])
def test_hooks_are_active_during_project_construction_and_refuse_any_alias_use(alias, needle):
    """The hooks are installed BEFORE the Project is built; a constructor touching an alias is stopped/refused before any
    real preparation/mesher work, and everything is restored."""
    log = []

    def construct(ns):
        if alias == 'closure':
            ns['project'].extract_model_closure(Geometry(), (), include_structural_closure=True, include_features=False)
        elif alias == 'prepare':
            ns['project'].prepare_structural_connectivity(SimpleNamespace(faces={}))
        else:
            ns['hybrid'].generate_hybrid_mesh_result(SimpleNamespace(faces={}))
    ns, Project, factory, closure, prepare, mesher, _ = make_world(log, construct=construct)
    with pytest.raises(tool.HarnessError, match=needle):
        tool.run_row('cold', Project, factory, SELECTED)
    assert not [entry for entry in log if entry == 'MESH' or (type(entry) is tuple and entry[0] in ('prepare', 'closure'))]
    assert ns['project'].extract_model_closure is closure and ns['project'].prepare_structural_connectivity is prepare
    assert ns['hybrid'].generate_hybrid_mesh_result is mesher and ns['backend'].generate_hybrid_mesh_result is mesher


def _session_api(log, collectors, prepare):
    return SimpleNamespace(
        from_dict=lambda doc: SimpleNamespace(faces={i: None for i in doc['doc']}), collectors=collectors,
        prepare=lambda model, **kwargs: (log.append(('control', os.environ.get(tool.ENV))) or prepare(model, **kwargs)),
        source_edge_paths=lambda closure: {}, source_work_mapping=lambda closure, edge_paths=None: {},
        export=lambda recorder, directory, label: (log.append(('export', label)) or
                                                   ({'application': 'a', 'plan': 'p', 'report': 'r', 'diagnostics': 'd'}, {'plan': label})))


@pytest.mark.parametrize('mode', ['serial', 'default'])
def test_session_runs_both_timed_rows_before_exports_and_controls(monkeypatch, tmp_path, mode):
    monkeypatch.delenv(tool.ENV, raising=False)
    log = []
    _, Project, factory, closure, prepare, mesher, collectors = make_world(log)
    result = tool.run_session(None, _session_api(log, collectors, prepare), mode=mode, directory=tmp_path, selected=SELECTED,
                              make_project=Project, hooks_factory=factory)
    stops = [i for i, entry in enumerate(log) if entry == 'post-prep']
    first_export = next(i for i, entry in enumerate(log) if type(entry) is tuple and entry[0] in ('export', 'control'))
    assert len(stops) == 2 and first_export > stops[1]
    assert [row['label'] for row in result['rows']] == ['cold', 'warm'] and result['timed_preparations'] == 2
    assert 'row_pair_identical' not in result and 'distinct UUIDs' in result['cold_vs_warm']
    controls = [entry for entry in log if type(entry) is tuple and entry[0] == 'control']
    if mode == 'serial':
        assert controls == [] and result['unmeasured_preparations'] == 0 and 'identity_verdict' not in result
    else:
        assert [c[1] for c in controls] == ['0', '1', '0', '1'] and result['unmeasured_preparations'] == 4
        assert result['identity_verdict'] is True and all(c['genuine_serial_retired_route'] for c in result['controls'])
        assert all(c['routing_identical_absent_vs_1'] and c['1']['plans_observed'] == 1 for c in result['controls'])
    assert os.environ.get(tool.ENV) is None


@pytest.mark.parametrize('counts,needle', [([1, 1, 0, 1, 1, 1], 'observed 0 plans'), ([1, 1, 1, 2, 1, 1], 'observed 2 plans'),
                                           ([1, 1, 1, 1, 1, 0], 'observed 0 plans')])
def test_every_identity_control_requires_exactly_one_observed_plan(monkeypatch, tmp_path, counts, needle):
    monkeypatch.delenv(tool.ENV, raising=False)
    _, Project, factory, closure, prepare, mesher, collectors = make_world([], plan_counts=counts)
    api = _session_api([], collectors, prepare)
    with pytest.raises(tool.HarnessError, match=needle):
        tool.run_session(None, api, mode='default', directory=tmp_path, selected=SELECTED, make_project=Project, hooks_factory=factory)


def test_public_setting_is_restored_and_diagnostics_use_declared_families():
    os.environ[tool.ENV] = '0'
    try:
        with tool.public_setting(None):
            assert tool.ENV not in os.environ
        assert os.environ[tool.ENV] == '0'
        with tool.public_setting('1'):
            assert os.environ[tool.ENV] == '1'
        assert os.environ[tool.ENV] == '0'
    finally:
        os.environ.pop(tool.ENV, None)
    counts = {'t2b_pairs': 3, 't2b_x_time_ns': 7, 'cache_hits': 4, 'intersection_pair_a': 2.0, 'other': 1}
    kept, excluded = tool.deterministic_diagnostics(counts)
    assert kept == {'t2b_pairs': 3, 'intersection_pair_a': 2.0} and excluded == ['cache_hits', 'other', 't2b_x_time_ns']
    assert tool.diagnostic_identity({'t2b_a': 2}) != tool.diagnostic_identity({'t2b_a': 2.0})


def test_source_hashes_refuse_unreadable_inputs_and_exit_code_blocks_changed_sources(tmp_path):
    readable = tmp_path / 'a.py'
    readable.write_text('x')
    assert list(tool.source_hashes([readable])) == [str(readable)]
    with pytest.raises(tool.HarnessError, match='unreadable source'):
        tool.source_hashes([readable, tmp_path / 'missing.py'])           # never two equal "unreadable" strings
    assert tool.exit_code(True, True, False) == 0 and tool.exit_code(True, None, False) == 0
    assert tool.exit_code(True, True, True) == 1 and tool.exit_code(False, True, False) == 1 and tool.exit_code(True, False, False) == 1


# --------------------------------------------------------------------- exact exports with registered fixture types
def _fixture_module(monkeypatch, prefix):
    module = types.ModuleType(prefix + '._r11_subset_fixture')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _register(module, cls):
    cls.__module__, cls.__qualname__ = module.__name__, cls.__name__
    setattr(module, cls.__name__, cls)
    return cls


def test_exact_plan_document_has_the_historical_registered_type_guard(monkeypatch):
    module = _fixture_module(monkeypatch, 'anygeometry')

    @dataclasses.dataclass
    class Policy:
        mode: str
        cancellation_check: object

    @dataclasses.dataclass
    class Plan:
        operands: tuple
        label: str
        policy: Policy
    _register(module, Policy), _register(module, Plan)
    record = lambda value: value if isinstance(value, (str, int, float, bool, type(None), list, tuple)) else repr(value)
    plan = Plan(operands=tuple(range(5)), label='x', policy=Policy('exact', lambda: None))
    text = tool.exact_plan_document(plan, record)
    assert '"operands"' in text and '"label"' in text and '"mode"' in text and 'cancellation_check' not in text
    assert text.index('"operands"') < text.index('"label"') < text.index('"policy"')
    with pytest.raises(ValueError, match='component bound'):
        tool.exact_plan_document(Plan(operands=tuple(range(100001)), label='x', policy=Policy('exact', None)), record)
    # foreign plan / foreign policy / unbound identity / non-dataclass are all refused
    @dataclasses.dataclass
    class Foreign:
        operands: tuple
    @dataclasses.dataclass
    class ForeignPolicy:
        mode: str
    for bad in (Foreign(()), SimpleNamespace(), Plan(operands=(), label='x', policy=ForeignPolicy('exact'))):
        with pytest.raises(ValueError, match='unregistered qualification plan record'):
            tool.exact_plan_document(bad, record)
    module.Plan = type('Plan', (), {})                                  # the bound name no longer is this class
    with pytest.raises(ValueError, match='unregistered'):
        tool.exact_plan_document(plan, record)


def _owner_types(monkeypatch):
    module = _fixture_module(monkeypatch, 'anyfem')

    @dataclasses.dataclass(frozen=True)
    class Conn:
        first: str
        reused: bool
        relations: tuple = ()

    @dataclasses.dataclass
    class Rep:
        source_model_id: str
        source_revision: int
        working_model_id: uuid.UUID
        connections: list
        diagnostics: list
        source_to_working: dict
        source_edge_paths: dict
        scale: float = 1.0
    _register(module, Conn), _register(module, Rep)
    return Conn, Rep, tool.registered_owner_types((Rep, Conn))


def _report(Conn, Rep, **changes):
    values = dict(source_model_id='m', source_revision=3, working_model_id=uuid.UUID(int=7),
                  connections=[Conn('face:1', True, ('a', 'b')), Conn('face:2', False)], diagnostics=['d'],
                  source_to_working={'face:1': ('face:5',)}, source_edge_paths={4: ((9, True), (10, False))}, scale=0.0)
    values.update(changes)
    return Rep(**values)


def test_owner_record_encodes_every_declared_field_and_is_type_uuid_and_sign_sensitive(monkeypatch):
    Conn, Rep, registered = _owner_types(monkeypatch)
    base = tool.owner_record(_report(Conn, Rep), registered)
    assert tool.owner_record(_report(Conn, Rep), registered) == base                       # deterministic
    for field in dataclasses.fields(Rep):                                                   # every field is encoded by name, in order
        assert '"%s"' % field.name in base
    assert base.index('"source_model_id"') < base.index('"connections"') < base.index('"scale"')
    assert '"relations"' in base and '"reused"' in base and 'face:2' in base               # nested connection fields
    changes = dict(
        revision=dict(source_revision=4), uuid=dict(working_model_id=uuid.UUID(int=8)), nested=dict(connections=[Conn('face:1', True, ('a', 'c')), Conn('face:2', False)]),
        nested_bool=dict(connections=[Conn('face:1', False, ('a', 'b')), Conn('face:2', False)]),
        order=dict(connections=[Conn('face:2', False), Conn('face:1', True, ('a', 'b'))]),
        diag=dict(diagnostics=['e']), mapping=dict(source_to_working={'face:1': ('face:6',)}),
        tuple_vs_list=dict(source_to_working={'face:1': ['face:5']}),
        edge=dict(source_edge_paths={4: ((9, False), (10, False))}), signed_zero=dict(scale=-0.0), int_vs_float=dict(scale=0),
        key_type=dict(source_edge_paths={'4': ((9, True), (10, False))}))
    records = {name: tool.owner_record(_report(Conn, Rep, **change), registered) for name, change in changes.items()}
    assert all(record != base for record in records.values()) and len(set(records.values())) == len(records)
    assert tool.owner_record(_report(Conn, Rep, source_to_working={1: 'a', '1': 'b'}), registered) != \
        tool.owner_record(_report(Conn, Rep, source_to_working={'1': 'a', 1: 'b'}), registered)   # typed mapping keys


def _export_world(monkeypatch, behavior, working_doc=None):
    """Bounded fakes for the ACTUAL export_bundle: registered plan/report types, a fake recorder, no ANYfem."""
    import json
    owner = _fixture_module(monkeypatch, 'anygeometry')

    @dataclasses.dataclass
    class Policy:
        mode: str
        cancellation_check: object

    @dataclasses.dataclass
    class Plan:
        operands: tuple
        policy: Policy
    _register(owner, Policy), _register(owner, Plan)
    fem = _fixture_module(monkeypatch, 'anyfem')

    class Rep:
        pass
    Rep = dataclasses.dataclass(type('Rep', (), {'__annotations__': {'source_revision': int}, 'source_revision': 3}))
    if behavior == 'valid':
        Rep.to_dict = lambda self: {'source_revision': self.source_revision, 'scale': 1.5}
    elif behavior == 'noncallable':
        Rep.to_dict = 5
    elif behavior == 'raising':
        def boom(self):
            raise RuntimeError('to_dict exploded')
        Rep.to_dict = boom
    elif behavior == 'nan':
        Rep.to_dict = lambda self: {'x': float('nan')}
    elif behavior == 'unserializable':
        Rep.to_dict = lambda self: {'x': object()}
    _register(fem, Rep)
    registered = tool.registered_owner_types((Rep,))
    recorder = SimpleNamespace(evidence=[{'plan_checksum': 'c', 'plan': Plan((1, 2), Policy('exact', None))}], report=Rep(),
                               working=SimpleNamespace(faces={1: None}), counts={'t2b_x': 1, 'y_time_ns': 4},
                               records={'parallel_planning': [{'disposition': 'serial'}]},
                               working_doc={'model_id': 'uuid-1', 'faces': [1]} if working_doc is None else working_doc)
    record = lambda value: value if isinstance(value, (str, int, float, bool, type(None), list, tuple)) else repr(value)
    call = lambda directory, label='cold': tool.export_bundle(recorder, lambda m: {'doc': sorted(m.faces)}, record, registered,
                                                              directory, label)
    return recorder, call


def test_export_bundle_requires_the_owner_public_observable_and_persists_the_initial_document(monkeypatch, tmp_path):
    import hashlib
    import json
    recorder, call = _export_world(monkeypatch, 'valid')
    bundle, digests = call(tmp_path)
    assert json.loads(bundle['report_to_dict']) == {'source_revision': 3, 'scale': 1.5}      # a real public observable
    initial = tmp_path / 'cold-initial-working-document.json'
    text = json.dumps(recorder.working_doc, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert initial.read_text(encoding='utf-8') == text and 'uuid-1' in text                    # UUID included, not stripped
    assert digests['initial_working_document'] == hashlib.sha256(text.encode()).hexdigest()
    assert 'initial_working_document' not in bundle                                            # evidence, not a comparison field
    for suffix in ('plan', 'application', 'report', 'report_to_dict', 'diagnostics', 'raw-diagnostics', 'runtime-records'):
        assert (tmp_path / ('cold-%s.json' % suffix)).is_file()
    # a control recorder carries no initial document: nothing extra is written, the row artifact is referenced instead
    recorder.working_doc = None
    control_bundle, control_digests = call(tmp_path, 'cold-control-1')
    assert 'initial_working_document' not in control_digests and not (tmp_path / 'cold-control-1-initial-working-document.json').exists()
    assert control_bundle == bundle


@pytest.mark.parametrize('behavior,cause', [('missing', None), ('noncallable', None), ('raising', RuntimeError),
                                            ('nan', ValueError), ('unserializable', TypeError)])
def test_export_bundle_fails_closed_when_the_owner_observable_is_unavailable(monkeypatch, tmp_path, behavior, cause):
    recorder, call = _export_world(monkeypatch, behavior)
    with pytest.raises(tool.HarnessError, match='to_dict') as caught:
        call(tmp_path)
    if cause is None:
        assert caught.value.__cause__ is None
    else:
        assert type(caught.value.__cause__) is cause                                          # chained to the real cause
    assert not (tmp_path / 'cold-report_to_dict.json').exists()                               # no comparable failure string exists


def test_export_bundle_refuses_a_non_strict_initial_document(monkeypatch, tmp_path):
    recorder, call = _export_world(monkeypatch, 'valid', working_doc={'x': float('nan')})
    with pytest.raises(tool.HarnessError, match='initial working document'):
        call(tmp_path)


def test_session_records_the_initial_document_artifact_path_and_controls_reference_it(monkeypatch, tmp_path):
    monkeypatch.delenv(tool.ENV, raising=False)
    _, Project, factory, closure, prepare, mesher, collectors = make_world([])
    result = tool.run_session(None, _session_api([], collectors, prepare), mode='default', directory=tmp_path, selected=SELECTED,
                              make_project=Project, hooks_factory=factory)
    for row, control in zip(result['rows'], result['controls']):
        expected = str(tmp_path / ('%s-initial-working-document.json' % row['label']))
        assert row['initial_working_document_path'] == expected and control['initial_working_document'] == expected


def test_owner_record_refuses_foreign_unknown_cyclic_and_oversized_values(monkeypatch):
    Conn, Rep, registered = _owner_types(monkeypatch)

    @dataclasses.dataclass
    class Foreign:
        value: int
    for bad in (Foreign(1), _report(Conn, Rep, diagnostics=[Foreign(1)]), _report(Conn, Rep, diagnostics=[object()]),
                _report(Conn, Rep, diagnostics=[{1, object()}]), _report(Conn, Rep, scale=1 + 2j)):
        with pytest.raises(tool.HarnessError, match='unsupported or unregistered'):
            tool.owner_record(bad, registered)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(tool.HarnessError, match='cyclic'):
        tool.owner_record(_report(Conn, Rep, diagnostics=cycle), registered)
    with pytest.raises(tool.HarnessError, match='value bound'):
        tool.owner_record(_report(Conn, Rep, diagnostics=list(range(100001))), registered)
    with pytest.raises(tool.HarnessError, match='unregistered owner report type'):
        tool.registered_owner_types((Foreign,))
    with pytest.raises(tool.HarnessError, match='unregistered owner report type'):
        tool.registered_owner_types((Rep,), prefix='anygeometry.')                           # wrong owner prefix
    sys.modules[Rep.__module__].Rep = type('Rep', (), {})                                    # unbound identity
    with pytest.raises(tool.HarnessError, match='unregistered owner report type'):
        tool.registered_owner_types((Rep,))
