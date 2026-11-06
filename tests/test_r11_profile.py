"""R11 probe/profile tooling contracts: single process, shared builders, no children."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest
from _planning_numpy_profile_contract import numpy_profile_test

ROOT =Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_builders_feed_the_real_session_consumers(tmp_path):
    """The probe's packets come from the same builders the normal route and child validators use."""
    from anygeometry._planning_executor import freeze_model, pack, write_packet, source_files
    from anygeometry._planning_session import (_setup, build_session_command, build_session_setup,
        build_worker_command, validate_command)
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_values import value_codec_contract
    model, _ = _fixture()
    packet = freeze_model(model)
    limits = dict(payload_limit=4_000_000, walk_limit=100000)
    setup = build_session_setup(packet, source_files(), workers=8, batch_size=None, limits=limits,
        startup_concurrency=8, numerical_manifest=dict(value_codec=value_codec_contract()))
    assert set(setup) == {'model', 'sources', 'workers', 'batch_size', 'limits', 'startup_concurrency', 'numerical_manifest'}
    plain = build_session_setup(packet, {}, workers=8, batch_size=None, limits=limits)
    assert set(plain) == {'model', 'sources', 'workers', 'batch_size', 'limits'}
    initial = pack(setup, 'session')
    write_packet(tmp_path / 'session.json', initial)
    import time
    read, digest = _setup(tmp_path, limits, time.monotonic() + 30, decode=False)   # real child-side schema check
    assert digest == initial['sha256'] and read['workers'] == 8
    tasks = [dict(kind='pair', first=1, second=2, tolerance=.1, face_tolerances=[.1, .1], cost=1.)]
    body = build_session_command(digest, 0, digest, 'pair', tasks, seen=())
    assert validate_command(body, session=digest, sequence=0, previous=digest, seen=()) == 'pair'
    worker = build_worker_command(digest, 0, digest, 'pair', [[0, tasks[0]]], 0, pack(body, 'session-command')['sha256'])
    assert validate_command(worker, session=digest, sequence=0, previous=digest, seen=(), worker=0) == 'pair'
    with pytest.raises(Exception):
        build_session_command(digest, 0, digest, 'face', tasks, seen=('face',))    # grammar still enforced


class _Stop(BaseException):
    pass


def _planner_folder(tmp_path, monkeypatch, *, production, corrupt=False):
    from types import SimpleNamespace
    from anygeometry import _planning_code
    from anygeometry import _planning_session as s
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model, pack, write_packet, source_files
    from anygeometry._planning_values import value_codec_contract
    model, _ = _fixture()
    packet = freeze_model(model)
    if corrupt:
        packet = dict(packet, sha256='0' * 64)
    limits = dict(payload_limit=4_000_000, walk_limit=100000)
    manifest = dict(value_codec=value_codec_contract(), common='c', constructor='k', host='h',
                    infrastructure_host='i', resource={}) if production else None
    setup = s.build_session_setup(packet, source_files(), workers=8, batch_size=None, limits=limits,
                                  startup_concurrency=8 if production else 1, numerical_manifest=manifest)
    write_packet(tmp_path / 'session.json', pack(setup, 'session'))
    if production:
        write_packet(tmp_path / 'admission.json', pack(manifest, 'package-admission'))
        monkeypatch.setattr(_planning_code, 'InfrastructurePin',
                            lambda role: SimpleNamespace(native={}, digest='p', nodes=1, verify=lambda: None))
    return s, limits


def test_production_planner_thaws_the_model_once_after_infrastructure_binding(tmp_path, monkeypatch):
    import time
    s, limits = _planner_folder(tmp_path, monkeypatch, production=True)
    calls = []
    real = s.thaw_model

    def counting(packet, **kw):
        calls.append(1)
        raise _Stop()
    monkeypatch.setattr(s, 'thaw_model', counting)
    with pytest.raises(_Stop):
        s.session_planner(tmp_path, time.monotonic() + 60, {}, {}, {}, limits, ())
    assert calls == [1]                                    # the first (and only) decode is model_context
    phases = s._read_initialization(tmp_path, __import__('os').getpid())['phases']
    assert 'infrastructure' in phases and 'model_context' not in phases and 'setup_and_initial_thaw' in phases
    assert real is not counting


def test_invalid_model_is_still_refused_in_production_and_legacy(tmp_path, monkeypatch):
    import time
    from anygeometry._planning_qualification import QualificationDecline
    s, limits = _planner_folder(tmp_path, monkeypatch, production=True, corrupt=True)
    with pytest.raises(QualificationDecline):
        s.session_planner(tmp_path, time.monotonic() + 60, {}, {}, {}, limits, ())
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    s, limits = _planner_folder(legacy, monkeypatch, production=False, corrupt=True)
    with pytest.raises(QualificationDecline):
        s.session_planner(legacy, time.monotonic() + 60, {}, {}, {}, limits, ())


def test_legacy_planner_keeps_its_upfront_decode_validation(tmp_path, monkeypatch):
    import time
    s, limits = _planner_folder(tmp_path, monkeypatch, production=False)
    calls = []
    monkeypatch.setattr(s, 'thaw_model', lambda packet, **kw: (calls.append(1), (_ for _ in ()).throw(_Stop()))[1])
    with pytest.raises(_Stop):
        s.session_planner(tmp_path, time.monotonic() + 60, {}, {}, {}, limits, ())
    assert calls == [1]
    # Worker/_setup(decode=False) remains decode-free; decode=True still validates.
    monkeypatch.setattr(s, 'thaw_model', lambda packet, **kw: calls.append(2))
    s._setup(tmp_path, limits, time.monotonic() + 60, decode=False)
    assert calls == [1]
    s._setup(tmp_path, limits, time.monotonic() + 60, decode=True)
    assert calls == [1, 2]


def test_instruction_cache_cap_is_storage_only_and_identity_guarded(monkeypatch):
    from anygeometry import _planning_manifest as m
    from anygeometry._planning_qualification import QualificationDecline
    assert m._INSTRUCTION_CACHE_SLOTS >= 4096

    def sample(x):
        return [i for i in range(x) if i % 2]
    code = sample.__code__
    full = tuple(m._instructions(code))
    cached = m._ImmutableEvidence()
    ops = cached.instructions(code)
    assert ops == full == m._ImmutableEvidence(cache_traversal=False).instructions(code)
    assert cached.instructions(code) is ops and cached.instruction_slots >= len(full) > 0
    monkeypatch.setattr(m, '_INSTRUCTION_CACHE_SLOTS', 1)         # cap exhausted: still complete, just uncached
    tiny = m._ImmutableEvidence()
    assert tiny.instructions(code) == full and tiny.instruction_cache == {} and tiny.instruction_slots == 0
    cached.instruction_cache[(id(code), True)] = (sample.__code__.replace(), ())
    with pytest.raises(QualificationDecline):
        cached.instructions(code)                                 # recycled identity refused
    cached.begin()
    assert cached.instruction_cache == {} and cached.instruction_slots == 0


@pytest.mark.slow
@numpy_profile_test
def test_cached_uncached_and_old_cap_role_graphs_are_equivalent(monkeypatch, request):
    from anygeometry import _planning_manifest as m, serialization
    from anygeometry._planning_canaries import _fixture
    model, _ = _fixture()
    serialization.to_dict(model)

    def summary(manifest):
        return (manifest.callables.digest, {r: x.digest for r, x in manifest.roles.items()}, manifest.wire['sha256'],
                manifest.callables.graph, {r: x.graph for r, x in manifest.roles.items()})
    base = m.LoadedManifest(model=model)
    base.verify()
    uncached = m.LoadedManifest(model=model, _cache_traversal=False)
    monkeypatch.setattr(m, '_INSTRUCTION_CACHE_SLOTS', 4096)
    old = m.LoadedManifest(model=model)
    assert summary(base) == summary(uncached) == summary(old)


def test_estimator_never_calls_the_model_id_property(monkeypatch):
    from anygeometry import _planning_controller as controller
    from anygeometry._planning_canaries import _fixture
    from anygeometry.model import GeometryModel
    model, _ = _fixture()
    faces = tuple(model.faces)
    handles = tuple(sorted((model.handle('face', f) for f in faces), key=lambda h: h.key))
    def forbidden(self):
        raise AssertionError('model_id property called by the cost estimator')
    monkeypatch.setattr(GeometryModel, 'model_id', property(forbidden))
    assert controller._estimate_cost(model, handles)[0] == len(faces)
    assert controller._estimate_cost(model, handles[:1])[0] == 1


def _session_bridge(tmp_path, results_by_kind, *, limits=None):
    """Real _SessionBridge.run path with an in-process fake session (no spawn)."""
    from types import SimpleNamespace
    from anygeometry import _planning_host as host
    from anygeometry._planning_values import value_codec_contract
    control = SimpleNamespace(production=True, workers=8, startup_concurrency=8, timeout=100, reuse_workers=True,
        directory=tmp_path, reports=[], limits=dict(limits or dict(payload_limit=4_000_000, walk_limit=100000)),
        lease=SimpleNamespace(observe_phase=lambda *a: None), tree_options=lambda: {},
        numerical_manifest=dict(common='c', constructor='k', host='h', infrastructure_host='i', resource={},
                                value_codec=value_codec_contract()))
    bridge = host._SessionBridge({'sha256': 'model'}, control, poll=lambda: None)
    class Session:
        tree = SimpleNamespace(memory_snapshot=lambda: None)
        def submit_stage(self, kind, packet, tasks, *, poll=None):
            from anygeometry._planning_stages import dispatch
            return dict(results=list(results_by_kind[kind]), request_sha256='r' * 64, startup=None, transport_bytes=None,
                        worker_reports=[], sampling_sha256='c' * 64, task_layout=dispatch(list(tasks), 8)[1])
    bridge.session = Session()
    return bridge


def _wire(value):
    from anygeometry._planning_values import encode_value, _wire_json
    return _wire_json(encode_value(value, codec='v2'))


PAIR_TASKS = [dict(kind='pair', first=1, second=2, tolerance=.1, face_tolerances=[.1, .1], cost=1.)]
FACE_TASKS = [dict(kind='face', face=1, inputs=None, tolerance=.1, area_tolerance=.1, seam='baseline', cost=1.)]
PAIR_VALUE = ((1.5, 2.5), [3.0])
FACE_VALUE = ((4.5,), [5.0])


def _face_tasks():
    from anygeometry._planning_values import encode_value
    # Like batch_intersections' stage_faces: default (expanded) codec; transport_tasks_v2 compacts it.
    return [dict(FACE_TASKS[0], inputs=encode_value(([], [])))]


def _run_stage(bridge, kind, tasks, value):
    out = bridge.run(kind, tasks, lambda task: value)
    bridge.session = None
    return out


def _count_decodes(monkeypatch):
    from anygeometry import _planning_host as host
    real = host.decode_stage_values
    calls = []
    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(host, 'decode_stage_values', counting)
    return calls, real


def test_session_bridge_second_decode_after_retirement_still_validates(tmp_path, monkeypatch):
    """Original behavior: run decodes once, deferred verification decodes+validates again (both stages)."""
    from anygeometry._planning_values import decode_stage_values
    calls, _ = _count_decodes(monkeypatch)
    results = {'pair': [_wire(PAIR_VALUE)], 'face': [_wire(FACE_VALUE)]}
    bridge = _session_bridge(tmp_path, results)
    first = _run_stage(bridge, 'pair', PAIR_TASKS, PAIR_VALUE)
    assert first == decode_stage_values(results['pair'], codec='v2')    # same typed values as the original decoder
    assert len(calls) == 1
    bridge.session = _session_bridge(tmp_path, results).session
    _run_stage(bridge, 'face', _face_tasks(), FACE_VALUE)
    assert len(calls) == 2
    bridge.verify_pending()                                   # real deferred path: samples + second decodes
    assert len(calls) == 4
    assert [report['stage'] for report in bridge.control.reports] == ['pair', 'face']


def test_second_decode_refuses_tampered_result_bytes(tmp_path, monkeypatch):
    from anygeometry._planning_qualification import QualificationDecline
    calls, _ = _count_decodes(monkeypatch)
    bridge = _session_bridge(tmp_path, {'pair': [_wire(PAIR_VALUE)]})
    _run_stage(bridge, 'pair', PAIR_TASKS, PAIR_VALUE)
    candidate = bridge.pending[0][3]
    candidate['results'][0] = ' ' + candidate['results'][0]    # same JSON value, noncanonical bytes
    with pytest.raises(QualificationDecline, match='noncanonical'):
        bridge.verify_pending()
    assert len(calls) == 2


def test_failed_first_decode_is_refused(tmp_path, monkeypatch):
    from anygeometry._planning_qualification import QualificationDecline
    calls, _ = _count_decodes(monkeypatch)
    bridge = _session_bridge(tmp_path, {'pair': [' ' + _wire(PAIR_VALUE)]})
    with pytest.raises(QualificationDecline):
        bridge.run('pair', PAIR_TASKS, lambda task: PAIR_VALUE)
    assert len(calls) == 1


@pytest.mark.slow
def test_probe_fixture_small_is_single_process_and_reports_actual_packets(tmp_path):
    probe = _tool('r11_probe')
    from anygeometry._planning_canaries import _fixture
    model, _ = _fixture()
    result = probe.run_probe(model, tuple(model.faces), tmp_path, thaw_equivalence=True)
    assert not result['errors'], result['errors']
    assert result['single_process'] is True and result['children_spawned'] == 0
    assert (tmp_path / 'r11-probe.json').is_file()
    setup = result['session_setup']
    assert setup['payload_utf8_bytes'] > 0 and setup['envelope_bytes'] > setup['payload_utf8_bytes'] and setup['walked_nodes'] > 0
    assert result['manifest']['constructor'] and result['manifest']['host']
    assert result['manifest']['authority_constructor'] and result['manifest']['authority_nodes']
    # Full constructed replies (production builders + real verify_prepared_stage), every component bounded.
    assert result['stages']
    for stage in result['stages']:
        assert stage['errors'] == [] and stage['within'] and stage['task_count'] > 0
        assert len(stage['worker_commands']) == len(stage['worker_replies']) == len(stage['compact_reports_utf8_bytes']) == 8
        reply = stage['session_reply']
        assert reply['exact'] is False and reply['envelope_bytes'] > reply['payload_utf8_bytes']
        assert reply['payload_utf8_bytes'] >= stage['result_value_utf8_total'] + stage['diagnostics_canonical_bytes']
        assert all(c['within'] for c in stage['components']) and len(stage['components']) == 2 + 16
    assert result['ceilings'] == dict(callable_nodes=80000, authority_nodes=80000, model_value_nodes=100000,
        payload_bytes=4_000_000, envelope_bytes=8_100_000)
    # Real reset_for_batch simulation in the dedicated process; the worker-setup gap stays named and open.
    reset = result['reset_simulation']
    total_tasks = sum(stage['task_count'] for stage in result['stages'])
    assert reset['ledger_constructed'] and reset['reset_calls'] >= total_tasks and reset['registered_stores'] > 0
    assert reset['pins_match_manifest'] and reset['role'] in ('worker', 'probe-process-no-role')
    assert reset['determinism'] and all(value is True for value in reset['determinism'].values())
    assert result['worker_setup']['bound_proven'] is True and result['worker_setup']['cap']['payload_bytes'] == 524288
    assert result['thaw_equivalence']['constructor_equal'] and result['thaw_equivalence']['host_equal']
    assert result['stage_coverage']['captured'] == result['stage_coverage']['proven'] and result['stage_coverage']['captured']
    assert all(result['session_setup']['within'].values())
    terminal = result['terminal']
    assert terminal['within'] is True and terminal['errors'] == [] and len(terminal['worker_results']) == 8
    assert result['binding']['workers'] == 8 and len(result['binding']['sources_sha256']) == 64
    assert result['feasibility'] == 'passed' and result['missing'] == [] and result['complete'] is True


def _synthetic_proof_inputs():
    from anygeometry._planning_values import transport_tasks_v2, value_codec_contract
    tasks = [dict(PAIR_TASKS[0], first=a, second=b, cost=float(a)) for a, b in ((1, 2), (1, 3), (2, 3), (2, 4), (3, 4))]
    stage = dict(kind='pair', tasks=tasks, transported=list(transport_tasks_v2(tasks)),
                 encoded=[_wire(PAIR_VALUE) for _ in tasks],
                 diagnostics=[dict(counters={'pairs': 3, 'clip_time_ns': 12345}, records={'note': ['x']}) for _ in tasks])
    return stage, dict(session='1' * 64, sequence=0, previous='1' * 64, seen=(), workers=8, packet={'sha256': 'f' * 64},
                       numerical=dict(value_codec=value_codec_contract(), infrastructure_host='e' * 64),
                       infrastructure_native={'binary_sha256': 'd' * 64})


def test_full_reply_builders_are_used_and_validated_by_the_production_verifier():
    probe = _tool('r11_probe')
    stage, kwargs = _synthetic_proof_inputs()
    proof = probe.stage_proof(stage, **kwargs)
    assert proof['errors'] == [] and proof['within'] is True and len(proof['components']) == 18
    reply = proof['session_reply']
    # values + diagnostics + metadata all accounted for, escaped envelope larger than payload
    assert reply['payload_utf8_bytes'] >= proof['result_value_utf8_total'] + proof['diagnostics_canonical_bytes'] > 0
    assert all(r['payload_utf8_bytes'] > 0 for r in proof['worker_replies']) and len(proof['worker_replies']) == 8


def test_worst_case_bound_dominates_realistic_metadata_and_float_repr_is_maximal():
    probe = _tool('r11_probe')
    candidates = [1.2345678901234567e-308, 1.7976931348623157e+308, 2.2250738585072014e-308, 5e-324, 0.1 + 0.2,
                  123456789.12345678, 1e22, 1.2345678901234567e+300]
    assert len(repr(probe.FLOAT_WORST)) == max(len(repr(x)) for x in candidates) == 23
    stage, kwargs = _synthetic_proof_inputs()
    worst = probe.build_stage_reply(stage, worst=True, **kwargs)
    real = probe.build_stage_reply(stage, worst=False, **kwargs)
    for a, b in zip(worst[1]['worker_replies'], real[1]['worker_replies']):
        assert a['payload_utf8_bytes'] > b['payload_utf8_bytes']
    assert probe.measure_packet('w', worst[2], 'session-reply')['payload_utf8_bytes'] > \
        probe.measure_packet('r', real[2], 'session-reply')['payload_utf8_bytes']


def test_bound_compares_against_an_accepted_actual_reply(tmp_path):
    probe = _tool('r11_probe')
    from anygeometry._planning_executor import pack
    stage, kwargs = _synthetic_proof_inputs()
    proof = probe.stage_proof(stage, **kwargs)
    _, _, accepted = probe.build_stage_reply(stage, worst=False, **kwargs)     # emulates a real accepted run
    path = tmp_path / 'reply-0.json'
    path.write_text(json.dumps(pack(accepted, 'session-reply')), encoding='utf-8')
    comparison = probe.compare_actual(proof, path)
    assert comparison['payload_ok'] and comparison['envelope_ok'] and comparison['compact_ok']
    bigger = dict(accepted, infrastructure=dict(accepted['infrastructure'], padding='z' * 5_000_000))
    path.write_text(json.dumps(pack(bigger, 'session-reply', limit=10**8)), encoding='utf-8')
    assert probe.compare_actual(proof, path)['payload_ok'] is False       # an unbounded surprise is detected


def test_added_reply_metadata_field_or_oversize_makes_the_proof_fail(monkeypatch):
    probe = _tool('r11_probe')
    from anygeometry import _planning_session as session
    stage, kwargs = _synthetic_proof_inputs()
    real = session.build_worker_report
    monkeypatch.setattr(session, 'build_worker_report', lambda binding, **kw: dict(real(binding, **kw), extra_metadata='x'))
    refused = probe.stage_proof(stage, **kwargs)
    assert refused['within'] is False and any(e.startswith('refused:') for e in refused['errors'])
    monkeypatch.undo()
    limited = probe.stage_proof(stage, payload_limit=1000, **kwargs)
    assert limited['within'] is False and any(e.startswith('limit:') for e in limited['errors'])
    result = dict(errors=[], stages=[limited])
    assert probe.verdict(result) == ('refused', [])


def test_reply_metadata_bounds_are_enforced_by_the_production_check():
    from anygeometry._planning_session import reply_bounds, check_blas_report
    from anygeometry._planning_qualification import QualificationDecline
    REPLY_BOUNDS = reply_bounds()
    ok = [dict(user_api='blas', filepath='C:\\x.dll', num_threads=1)]
    check_blas_report(ok)
    # Unicode (Windows paths), non-BMP, quotes, backslashes and control characters are valid metadata.
    for good in ('caf\u00e9', '\U0010FFFF' * 1024, '"\\\n\t\x00' * 100, 'C:\\Users\\\u00c5udun\\x.dll'):
        check_blas_report([dict(ok[0], filepath=good)])
    check_blas_report([dict(ok[0], version='\U0010FFFF' * 64, prefix='"' * 64)])
    for bad in ([], ok * (REPLY_BOUNDS['blas_entries'] + 1), [dict(ok[0], filepath='x' * 1025)],
                [dict(ok[0], filepath='\U0010FFFF' * 1025)], [dict(ok[0], version='x' * 65)],
                [dict(ok[0], unknown='x')], [dict(ok[0], num_threads=-1)], [dict(ok[0], prefix=5)], 'x'):
        with pytest.raises(QualificationDecline):
            check_blas_report(bad)


def test_old_worker_reports_without_timing_are_still_accepted():
    probe = _tool('r11_probe')
    stage, kwargs = _synthetic_proof_inputs()
    body, _, session_body = probe.build_stage_reply(stage, worst=False, **kwargs)
    for report in session_body['worker_reports']:
        report.pop('task_timing')
    probe.verify_with_production_schema(stage, body, session_body, workers=8, packet=kwargs['packet'],
        numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'])


def test_packet_escaping_worst_case_is_measured_on_the_real_wire_form():
    probe = _tool('r11_probe')
    row = probe.measure_packet('e', {'a': '\u00e9\U0001F600"\\'}, 'tasks')
    payload = json.dumps({'a': '\u00e9\U0001F600"\\'}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert row['payload_utf8_bytes'] == len(payload.encode('utf-8')) and row['payload_utf8_bytes'] > 20


def test_worst_blas_uses_the_12_chars_per_codepoint_json_form_and_dominates_every_valid_string():
    probe = _tool('r11_probe')
    from anygeometry._planning_session import reply_bounds, check_blas_report
    REPLY_BOUNDS = reply_bounds()
    assert len(json.dumps(probe.worst_string(10))) == 2 + 10 * 12          # two surrogate escapes per U+10FFFF
    worst = probe.worst_blas()
    check_blas_report(worst)                                              # the placeholder is itself admissible
    assert len(worst) == REPLY_BOUNDS['blas_entries']
    worst_len = len(json.dumps(worst))
    for sample in ('"', '\\', '\x00', 'é', '€', '\U0001F600'):  # every other codepoint class is no larger
        entry = {k: sample * (REPLY_BOUNDS['blas_filepath_chars'] if k == 'filepath' else REPLY_BOUNDS['blas_string_chars'])
                 for k in worst[0] if k != 'num_threads'}
        entry['num_threads'] = 1
        assert len(json.dumps([entry] * REPLY_BOUNDS['blas_entries'])) <= worst_len


def test_reset_simulation_runs_real_reset_before_each_task_with_gate_semantics(monkeypatch):
    probe = _tool('r11_probe')
    from types import SimpleNamespace
    from anygeometry import _planning_stages as stages
    events = []
    class Ledger:
        def reset_for_batch(self, *, owned_worker):
            assert owned_worker is True
            events.append('reset')
            return {'a': 2, 'b': 3}
    def fake_stage_task(context, task):
        events.append('task')
        return _wire(PAIR_VALUE)
    monkeypatch.setattr(stages, 'stage_task', fake_stage_task)
    tasks = [dict(PAIR_TASKS[0], first=a, second=a + 1, cost=1.) for a in (1, 2, 3)]
    recorder = probe.Recorder(None, dict(payload_limit=4_000_000, walk_limit=100000), 8, ledger=Ledger())
    recorder.run('pair', tasks, None)
    assert events == ['reset', 'task'] * 3 + ['reset', 'task']             # + determinism re-execution (small case)
    assert recorder.reset_calls == 4 and recorder.reset_total == 20 and recorder.determinism == {'pair': True}
    assert len(recorder.stages) == 1 and len(recorder.stages[0]['encoded']) == 3     # one computation per task


def test_reset_simulation_detects_nondeterministic_diagnostics(monkeypatch):
    probe = _tool('r11_probe')
    from anygeometry import _planning_stages as stages
    from anygeometry.runtime_diagnostics import _increment
    calls = []
    def fake_stage_task(context, task):
        calls.append(1)
        _increment('drift_counter', len(calls))          # differs between the original and the re-execution
        return _wire(PAIR_VALUE)
    monkeypatch.setattr(stages, 'stage_task', fake_stage_task)
    class Ledger:
        def reset_for_batch(self, *, owned_worker):
            return {}
    recorder = probe.Recorder(None, dict(payload_limit=4_000_000, walk_limit=100000), 8, ledger=Ledger())
    recorder.run('pair', [dict(PAIR_TASKS[0])], None)
    assert recorder.determinism == {'pair': False}
    result = _full_result(reset_simulation=dict(ledger_constructed=True, reset_calls=2, pins_match_manifest=True,
                                                determinism=recorder.determinism))
    assert probe.verdict(result) == ('conditional', ['diagnostics.ledger_reset_semantics'])


def test_ledger_construction_failure_records_the_exact_invariant_and_uses_the_same_api(monkeypatch):
    probe = _tool('r11_probe')
    from types import SimpleNamespace
    from anygeometry import _planning_ledger as ledger_api, _planning_qualification as q
    from anygeometry._planning_qualification import QualificationDecline
    built = []
    def fake(model, *, resource_pin=None, role=None):
        built.append(role)
        if role == 'worker':
            raise QualificationDecline('worker resource absence invariant')
        return 'ledger'
    monkeypatch.setattr(ledger_api, 'package_worker_ledger', fake)
    monkeypatch.setattr(q, 'resource_file_pin', lambda native: dict(native))
    ledger, role, error = probe.build_probe_ledger(SimpleNamespace(model=object()), {'binary_sha256': 'x'})
    assert (ledger, role, error) == ('ledger', 'probe-process-no-role', 'worker resource absence invariant')
    assert built == ['worker', None]


def _full_result(**overrides):
    """A result that satisfies every verdict gate; tests flip one condition at a time."""
    result = dict(errors=[], stages=[dict(within=True)],
                  session_setup=dict(within=dict(payload=True, envelope=True, walk=True)),
                  reset_simulation=dict(ledger_constructed=True, reset_calls=3, pins_match_manifest=True, determinism={'pair': True}),
                  thaw_equivalence=dict(constructor_equal=True, host_equal=True, common_equal=True),
                  stage_coverage=dict(captured=['pair'], proven=['pair']), terminal=dict(within=True, errors=[]),
                  worker_setup=dict(bound_proven=True))
    result.update(overrides)
    return result


def test_no_flag_can_upgrade_missing_evidence_and_passed_requires_every_proof():
    probe = _tool('r11_probe')
    with pytest.raises(TypeError):
        probe.run_probe(None, (), '.', accept_diagnostic_semantics=True)
    with pytest.raises(SystemExit):
        probe.main(['--fixture-small', '--output', '.', '--accept-diagnostic-semantics'])
    assert probe.verdict(_full_result()) == ('passed', [])
    # every required condition independently flipped (false OR absent) fails closed with its own NAME
    flips = {
        'session_setup.within': [dict(session_setup=dict(within=dict(payload=True, envelope=False, walk=True))), dict(session_setup={}),
                                 dict(session_setup=dict(within=dict(payload=True, envelope=True)))],
        'reset_simulation.pins_match_manifest': [dict(reset_simulation=dict(ledger_constructed=True, reset_calls=3, determinism={})),
                                                 dict(reset_simulation=dict(ledger_constructed=True, reset_calls=3,
                                                                            pins_match_manifest=False, determinism={}))],
        'diagnostics.ledger_reset_semantics': [dict(reset_simulation=dict(pins_match_manifest=True, ledger_constructed=False)),
                                               dict(reset_simulation=dict(pins_match_manifest=True, ledger_constructed=True, reset_calls=0))],
        'thaw_equivalence': [dict(thaw_equivalence=None), dict(thaw_equivalence=dict(constructor_equal=True, host_equal=True)),
                             dict(thaw_equivalence=dict(constructor_equal=True, host_equal=False, common_equal=True)),
                             dict(thaw_equivalence=dict(constructor_equal=True, host_equal=True, common_equal=1))],
        'stage_coverage': [dict(stage_coverage=None), dict(stage_coverage=dict(captured=[], proven=[])),
                           dict(stage_coverage=dict(captured=['pair', 'face'], proven=['pair']))],
        'terminal_wire_proof': [dict(terminal=None), dict(terminal=dict(within=False, errors=[])),
                                dict(terminal=dict(within=True, errors=['refused: x']))],
        probe.WORKER_SETUP_GAP: [dict(worker_setup={}), dict(worker_setup=dict(bound_proven=False))],
    }
    for name, variants in flips.items():
        for variant in variants:
            status, missing = probe.verdict(_full_result(**variant))
            assert status == 'conditional' and name in missing, (name, variant, status, missing)
    # a probe run without thaw equivalence (the --no-thaw-equivalence path) can never qualify
    assert probe.verdict(_full_result(thaw_equivalence=None)) == ('conditional', ['thaw_equivalence'])
    # named conditions survive a generic error label; limit overruns stay refusals; no stages is incomplete
    status, missing = probe.verdict(_full_result(errors=[dict(step='x', error='y')], thaw_equivalence=None))
    assert status == 'incomplete' and 'thaw_equivalence' in missing
    assert probe.verdict(_full_result(stages=[dict(within=False)]))[0] == 'refused'
    assert probe.verdict(_full_result(stages=[]))[0] == 'incomplete'


def test_half_specified_comparison_arguments_are_rejected_before_any_work(tmp_path):
    probe = _tool('r11_probe')
    for kwargs in (dict(compare_reply='reply.json'), dict(compare_stage='pair')):
        with pytest.raises(ValueError, match='together'):
            probe.run_probe(None, (), tmp_path, **kwargs)           # raises before touching the (None) model
    assert not (tmp_path / 'r11-probe.json').exists()
    for half in (['--compare-reply', 'reply.json'], ['--compare-stage', 'pair']):
        with pytest.raises(SystemExit):
            probe.main(['--fixture-small', '--output', str(tmp_path)] + half)
    assert not (tmp_path / 'r11-probe.json').exists()


def _capture_directory(tmp_path, probe, *, tamper=None):
    """Emulate a real accepted small run with a CORRECT chain: every packet written with the production builders
    (realistic metadata) and chained exactly like session_planner/session_worker (worker previous = worker setup sha,
    session sha = setup packet sha). Tampers re-pack, so checksums are always valid: only content binding can fail."""
    from anygeometry._planning_child import with_process_seconds
    from anygeometry._planning_executor import pack
    from anygeometry._planning_session import (build_infrastructure_record, build_planner_result, build_session_command,
        build_session_setup, build_worker_command, build_worker_final, build_worker_setup)
    stage, kwargs = _synthetic_proof_inputs()
    limits = dict(payload_limit=4_000_000, walk_limit=100000)
    pin = {'binary_sha256': 'd' * 64, 'pin': 'resource'}
    native = kwargs['infrastructure_native']
    numerical = dict(kwargs['numerical'], resource=native)
    setup = build_session_setup(kwargs['packet'], {}, workers=8, batch_size=None, limits=limits, startup_concurrency=8,
                                numerical_manifest=numerical)
    session = pack(setup, 'session')['sha256']
    kwargs = dict(kwargs, session=session, previous=session)
    proof = probe.stage_proof(stage, **kwargs)
    stage_sha = proof['parent_command']['sha256']
    base = probe.measure_packet('worker setup', build_worker_setup(session, 0, kwargs['packet'], {}, {}, 8, None), 'session-worker')
    binding = dict(session_sha256=session, workers=8, startup_concurrency=8, model_sha256=kwargs['packet']['sha256'],
                   model_packet_sha256=probe._h(kwargs['packet']), sources_sha256=probe._h({}), numerical_sha256=probe._h(numerical),
                   limits_sha256=probe._h(limits), resource_pin_sha256=probe._h(pin), resource=native,
                   value_codec=numerical['value_codec'], numerical_manifest=numerical)
    terminal = probe.terminal_proof(('pair',), [stage_sha], session=session, previous=stage_sha, workers=8, packet=kwargs['packet'],
        numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'])
    result = dict(session_setup=probe.measure_packet('setup', setup, 'session'), stages=[proof], workers=8, binding=binding,
                  worker_setup=probe.worker_setup_bound(base), terminal=terminal)
    body, packets, session_body = probe.build_stage_reply(stage, worst=False, **kwargs)
    root = tmp_path / 'session'
    root.mkdir()
    def write(path, value, kind):
        path.write_text(json.dumps(pack(value, kind)), encoding='utf-8')
        return pack(value, kind)['sha256']
    write(root / 'session.json', setup, 'session')
    valid_infra = dict(session_body['infrastructure'])      # ONE planner infrastructure record shared by every stage and the final
    if tamper == 'chain':
        body = dict(body, previous='9' * 64)               # valid checksum after re-pack, wrong chain content
    if tamper == 'task':                                   # changed task input; this packet's own checksum is re-computed
        body = dict(body, tasks=[dict(body['tasks'][0], tolerance=0.5)] + list(body['tasks'][1:]))
    if tamper == 'infra_resource':                         # same-length stale resource hash in the P infrastructure record
        session_body = dict(session_body, infrastructure=dict(session_body['infrastructure'],
                            resource=dict(native, binary_sha256='f' * 64)))
    write(root / 'command-0.json', body, 'session-command')
    finish = build_session_command(session, 1, stage_sha, 'finish', [], seen=('pair',))
    if tamper == 'finish_tasks':                           # same bytes ({} vs []), chain hashes recomputed downstream
        finish = dict(finish, tasks={})
    if tamper == 'finish_unknown':
        finish = dict(finish, extra='x')
    finish_sha = write(root / 'command-1.json', finish, 'session-command')
    finals = []
    for index, (command, report) in enumerate(packets['worker_bodies']):
        folder = root / ('worker-%d' % index)
        folder.mkdir()
        wsetup = build_worker_setup(session, index, kwargs['packet'], {}, {'resource_pin': pin}, 8, None)
        if tamper == 'model' and index == 2:
            wsetup = dict(wsetup, model=dict(kwargs['packet'], sha256='e' * 64))
        wsha = write(folder / 'session.json', wsetup, 'session-worker')
        command = dict(command, previous=wsha)
        csha = write(folder / 'command-0.json', command, 'session-worker-command')
        report = dict(report, worker_setup_sha256=wsha, request_sha256=csha)
        # the aggregate publishes the compact form of exactly this worker report (as session_planner does)
        session_body['worker_reports'][index] = dict(session_body['worker_reports'][index], worker_setup_sha256=wsha, request_sha256=csha)
        if tamper == 'rows' and index == 0:
            report = dict(report, rows=[[row[0], row[1] + ' '] for row in report['rows']])
        if tamper == 'stable' and index == 0:
            report = dict(report, diagnostics=[[o, dict(d, counters=dict(d['counters'], pairs=4))] for o, d in report['diagnostics']])
        if tamper == 'oversize_diag' and index == 0:
            report = dict(report, diagnostics=[[o, dict(d, records=dict(d['records'], pad=['z' * 2000]))] for o, d in report['diagnostics']])
        if tamper == 'time_counter' and index == 0:       # variable wall-time observables stay within their bounds
            report = dict(report, diagnostics=[[o, dict(d, counters=dict(d['counters'], clip_time_ns=987654321))] for o, d in report['diagnostics']])
            session_body['worker_reports'][index] = dict(session_body['worker_reports'][index], diagnostics=report['diagnostics'])
        if tamper == 'stage_guard' and index == 1:           # same-length stale binary hash, valid re-packed checksum
            report = dict(report, resource_guard=dict(report['resource_guard'], binary_sha256='f' * 64))
        if tamper != 'missing' or index != 3:
            write(folder / 'reply-0.json', report, 'session-worker-reply')
        fcommand = build_worker_command(session, 1, csha, 'finish', [], index, finish_sha)
        if tamper == 'worker_finish_tasks' and index == 2:
            fcommand = dict(fcommand, tasks={})
        fsha = write(folder / 'command-1.json', fcommand, 'session-worker-command')
        final = with_process_seconds(build_worker_final(
            dict(session_sha256=session, worker=index, worker_setup_sha256=wsha, request_sha256=fsha),
            model_sha256=kwargs['packet']['sha256'], stages=('pair',), gate_closures=1,
            blas_threads=[dict(num_threads=1)], pid=12345,         # the stage replies publish pid 12345 (verified per index)
            resource_guard=dict(controller_loaded=False, native_adapter_loaded=False, native_core_loaded=False,
                                binary_sha256=pin['binary_sha256']), value_codec=kwargs['numerical']['value_codec']), 0.5)
        finals.append(final)
        written = final
        if tamper == 'final_guard' and index == 2:
            written = dict(final, resource_guard=dict(final['resource_guard'], binary_sha256='f' * 64))
        if tamper == 'codec' and index == 2:
            written = dict(final, value_codec=dict(final['value_codec'], schema_sha256='0' * 64))
        if tamper == 'final_pid' and index == 1:
            written = dict(final, pid=0)
        if tamper == 'final_decode' and index == 1:
            written = dict(final, model_decode_count=2)
        if tamper == 'final_blas' and index == 1:
            written = dict(final, blas_threads=[dict(num_threads=2)])
        if tamper == 'init_error_final' and index == 4:
            written = dict(final, initialization_evidence_error='OSError: x')
        if tamper != 'terminal_missing' or index != 5:
            write(folder / 'result.json', written, 'worker-result')
        if written is not final:
            finals[-1] = written                          # the planner echoes what that worker published
    write(root / 'reply-0.json', session_body, 'session-reply')
    infra = dict(valid_infra)      # same digest/nodes/references/resource/codec as the stage reply (P-only consistency)
    echoed = list(finals)
    if tamper == 'planner_final':                         # valid-looking changed echo of an otherwise valid final
        echoed[3] = dict(echoed[3], process_seconds=0.75)
    planner = with_process_seconds(build_planner_result(session, finish_sha, ('pair',), infra, [stage_sha], echoed,
        [] if tamper == 'prepared' else None, None), 0.7)
    if tamper == 'init_error_planner':
        planner = dict(planner, initialization_evidence_error='OSError: y')
    write(root / 'result.json', planner, 'result')
    return result, root


def _capture_directory_sizes_only_unused(tmp_path, probe, *, tamper=None):
    """(superseded by _capture_directory; kept unreferenced)"""
    from anygeometry._planning_executor import pack
    from anygeometry._planning_session import build_session_setup, build_worker_setup
    stage, kwargs = _synthetic_proof_inputs()
    setup = build_session_setup(kwargs['packet'], {}, workers=8, batch_size=None, limits=dict(payload_limit=4_000_000, walk_limit=100000))
    base = probe.measure_packet('worker setup', build_worker_setup(kwargs['session'], 0, kwargs['packet'], {}, {}, 8, None), 'session-worker')
    result = dict(session_setup=probe.measure_packet('setup', setup, 'session'), stages=[probe.stage_proof(stage, **kwargs)],
                  worker_setup=probe.worker_setup_bound(base), workers=8)
    body, packets, session_body = probe.build_stage_reply(stage, worst=False, **kwargs)
    root = tmp_path / 'session'
    root.mkdir()
    def write(path, value, kind):
        path.write_text(json.dumps(pack(value, kind)), encoding='utf-8')
    write(root / 'session.json', setup, 'session')
    write(root / 'command-0.json', body, 'session-command')
    write(root / 'reply-0.json', session_body, 'session-reply')
    for index, (command, report) in enumerate(packets['worker_bodies']):
        folder = root / ('worker-%d' % index)
        folder.mkdir()
        if tamper == 'rows' and index == 0:
            report = dict(report, rows=[[row[0], row[1] + ' '] for row in report['rows']])
        if tamper == 'stable' and index == 0:
            report = dict(report, diagnostics=[[o, dict(d, counters=dict(d['counters'], pairs=4))] for o, d in report['diagnostics']])
        if tamper == 'oversize_diag' and index == 0:
            report = dict(report, diagnostics=[[o, dict(d, records=dict(d['records'], pad=['z' * 2000]))] for o, d in report['diagnostics']])
        if tamper == 'time_counter' and index == 0:       # variable wall-time counters stay within the int63 bound
            report = dict(report, diagnostics=[[o, dict(d, counters=dict(d['counters'], clip_time_ns=987654321))] for o, d in report['diagnostics']])
        write(folder / 'command-0.json', command, 'session-worker-command')
        if tamper != 'missing' or index != 3:
            write(folder / 'reply-0.json', report, 'session-worker-reply')
        write(folder / 'session.json', build_worker_setup(kwargs['session'], index, kwargs['packet'], {}, {'small': 'reference'}, 8, None),
              'session-worker')
    return result, root


def test_compare_capture_validates_actual_packets_against_the_proof(tmp_path):
    probe = _tool('r11_probe')
    result, root = _capture_directory(tmp_path, probe)
    comparison = probe.compare_capture(result, root)
    assert comparison['all_within'] is True
    names = [row['name'] for row in comparison['rows']]
    assert 'session setup' in names and 'pair session-reply' in names and sum('worker-' in n and 'reply' in n for n in names) == 8
    replies = [row for row in comparison['rows'] if row['name'].endswith('reply') and 'worker-' in row['name']]
    assert all(row['rows_equal_simulated'] and row['diagnostics_within_simulated'] for row in replies)
    assert all(row['actual']['payload'] <= row['bound']['payload'] and row['actual']['walk'] <= row['bound']['walk']
               for row in comparison['rows'] if row['bound'])


def test_compare_capture_rejects_changed_rows_and_oversize_packets(tmp_path):
    probe = _tool('r11_probe')
    result, root = _capture_directory(tmp_path, probe, tamper='rows')
    assert probe.compare_capture(result, root)['all_within'] is False
    clean = tmp_path / 'clean'
    clean.mkdir()
    result, root = _capture_directory(clean, probe)
    path = root / 'worker-1' / 'command-0.json'
    from anygeometry._planning_executor import pack
    packet = json.loads(path.read_text())
    body = json.loads(packet['payload'])
    body['padding'] = 'z' * 5_000_000
    path.write_text(json.dumps(pack(body, 'session-worker-command', limit=10**8)), encoding='utf-8')
    comparison = probe.compare_capture(result, root)
    assert comparison['all_within'] is False
    assert [row['within'] for row in comparison['rows'] if row['name'].endswith('worker-1 command')] == [False]


def test_profile_packet_capture_copies_raw_packets_read_only_before_cleanup(tmp_path):
    import shutil
    profile = _tool('r11_profile')
    work = tmp_path / 'run' / 'work'
    (work / 'session' / 'worker-0').mkdir(parents=True)
    (work / 'session' / 'session.json').write_text('{"a":1}')
    (work / 'session' / 'worker-0' / 'reply-0.json').write_text('{"b":2}')
    (work / 'session' / 'note.txt').write_text('ignored')
    destination = tmp_path / 'captured'
    real = shutil.rmtree
    with profile.capture_session_packets(destination):
        shutil.rmtree(work)
    assert shutil.rmtree is real and not work.exists()
    captured = destination / 'session'
    assert (captured / 'session.json').read_text() == '{"a":1}' and (captured / 'worker-0' / 'reply-0.json').exists()
    assert not (captured / 'note.txt').exists()
    manifest = json.loads((destination / 'capture-manifest.json').read_text())
    assert set(manifest) == {'session.json', str(Path('worker-0') / 'reply-0.json')}
    import os
    import stat
    assert not os.stat(captured / 'session.json').st_mode & stat.S_IWRITE


R11_METADATA_NAMES = ('_REPLY_BOUNDS', '_WORKER_EXPECTED_CAP', '_TIMING_RECORD_FIELDS', '_BLAS_STRING_FIELDS', '_TIMING_FIELDS')


def test_r11_metadata_declarations_are_immutable_and_accessors_return_fresh_mappings():
    from anygeometry import _planning_session as session
    for name in R11_METADATA_NAMES:
        assert type(getattr(session, name)) in (tuple, frozenset), name              # immutable containers only
    for name in R11_METADATA_NAMES[:3]:                                             # (key, scalar) pairs
        assert all(type(item) is tuple and len(item) == 2 and type(item[0]) is str and type(item[1]) in (int, str)
                   for item in getattr(session, name)), name
    for name in ('REPLY_BOUNDS', 'WORKER_EXPECTED_CAP', 'TIMING_RECORD_FIELDS'):
        assert not hasattr(session, name)                                            # no mutable-mapping roots remain
    for accessor, key in ((session.reply_bounds, 'pid_max'), (session.worker_expected_cap, 'payload_bytes'),
                          (session.timing_record_fields, 'arrangement_snap_fallback')):
        first = accessor()
        original = first[key]
        first[key] = 'mutated'
        assert accessor()[key] == original                                           # mutating a copy changes nothing


@pytest.mark.slow
@numpy_profile_test
def test_r11_metadata_admits_through_the_real_prepared_legacy_ledger_and_mutable_standins_are_refused(monkeypatch, request):
    """Scope: ACTUAL legacy ledger preparation, not native-worker qualification. Mirrors what the legacy session child
    does before it builds its ledger: import the session closure, run the real ``_planning_child.prepare`` (real
    canaries + constructor/task lazy closure) under threadpool_limits(1), THEN loaded_geometry_ledger(trusted_image=
    True) + prepare(owned_worker=True). Runtime requirement: OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1 set in the
    environment BEFORE the test process starts (the child module records them at import and prepare() refuses
    otherwise). The two-census coverage guard is deliberate and unchanged: nothing here mocks or retries it."""
    import runpy
    import sys
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from anygeometry import _planning_session as session                    # session closure imported first, as the roles do
    from anygeometry import _planning_ledger as ledger_api, _planning_qualification as q
    from anygeometry._planning_qualification import QualificationDecline
    # Production runs _planning_child.py BY FILE as a script (never as the package module anygeometry._planning_child),
    # so its namespace (scipy/threadpoolctl/... globals) is not an audited anygeometry.* root. Load it the same way
    # under a NON-anygeometry, non-__main__ name; importing it as a package module would add an artificial root.
    assert 'anygeometry._planning_child' not in sys.modules, \
        'requires a fresh host: anygeometry._planning_child was already imported by another test/cohort'
    namespace = runpy.run_path(str(Path(session.__file__).with_name('_planning_child.py')), run_name='r11_legacy_preparation')
    assert 'anygeometry._planning_child' not in sys.modules and 'r11_legacy_preparation' not in sys.modules
    with threadpool_limits(limits=1):
        canaries, goldens = namespace['prepare'](production=False)           # the real legacy producer closure
        generated = q.generated_runtime_modules()
        assert set(canaries) == {'pair', 'face'} and set(goldens) == set(canaries)
        ledger = ledger_api.loaded_geometry_ledger(trusted_image=True)       # the real legacy admission, default roots
        ledger.prepare(owned_worker=True)                                    # real reset of the registered declined caches
        ledger.validate()                                                    # and the real census validation
        assert generated == q.generated_runtime_modules()                    # session_planner's own post-prepare guard
        prefix = 'anygeometry._planning_session.'
        stored = {path[len(prefix):] for path in ledger.rows if path.startswith(prefix)}
        assert not stored & (set(R11_METADATA_NAMES) | {'REPLY_BOUNDS', 'WORKER_EXPECTED_CAP', 'TIMING_RECORD_FIELDS'})
        # a mutable stand-in for a declaration is still refused by the unchanged ledger (census not weakened)
        monkeypatch.setattr(session, '_REPLY_BOUNDS', dict(session._REPLY_BOUNDS))
        with pytest.raises(QualificationDecline) as caught:
            ledger_api.loaded_geometry_ledger(trusted_image=True)
        assert '_REPLY_BOUNDS' in str(caught.value), str(caught.value)


def test_worker_expected_cap_is_exact_and_overcap_declines_with_a_typed_error():
    from anygeometry._planning_session import worker_expected_cap, check_worker_expected_bound
    from anygeometry._planning_qualification import QualificationDecline
    assert worker_expected_cap() == dict(payload_bytes=524288, walk_nodes=16384)
    at_cap = {'k': 'x' * (524288 - len('{"k":""}'))}
    assert check_worker_expected_bound(at_cap) == (524288, 3)
    with pytest.raises(QualificationDecline, match='eligibility cap'):
        check_worker_expected_bound({'k': 'x' * (524288 - len('{"k":""}') + 1)})
    assert check_worker_expected_bound({'k': [0] * 16381})[1] == 16384                 # dict + key + list + 16381 items
    with pytest.raises(QualificationDecline, match='nodes=16385'):
        check_worker_expected_bound({'k': [0] * 16382})
    # multi-byte content counts canonical (ensure_ascii) bytes, so a non-BMP codepoint costs 12
    with pytest.raises(QualificationDecline):
        check_worker_expected_bound({'k': '\U0010FFFF' * (524288 // 12 + 1)})
    for bad in (float('nan'), {1: object()}, {'k': {1, 2}}):
        with pytest.raises(QualificationDecline):
            check_worker_expected_bound(bad)


def test_worker_setup_bound_formula_covers_the_worst_expected_reference_including_wrapping():
    probe = _tool('r11_probe')
    from anygeometry._planning_session import build_worker_setup
    base = probe.measure_packet('b', build_worker_setup('1' * 64, 0, {'sha256': 'f' * 64}, {'a.py': 'b' * 64}, {}, 8, None),
                                'session-worker')
    bound = probe.worker_setup_bound(base)
    assert bound['bound_proven'] is True
    limit = bound['bound']
    # payload worst: exactly the cap; envelope worst: every payload quote/backslash doubles; walk worst: 16384 nodes.
    quotes = {'k': '"' * ((524288 - len('{"k":""}')) // 2)}
    nodes = {'k': [0] * 16381}
    for expected in (quotes, nodes, {}):
        row = probe.measure_packet('w', build_worker_setup('1' * 64, 7, {'sha256': 'f' * 64}, {'a.py': 'b' * 64}, expected, 8, None),
                                   'session-worker')
        assert row['payload_utf8_bytes'] <= limit['payload_utf8_bytes']
        assert row['envelope_bytes'] <= limit['envelope_bytes'] and row['walked_nodes'] <= limit['walked_nodes']
    full = probe.measure_packet('w', build_worker_setup('1' * 64, 0, {'sha256': 'f' * 64}, {'a.py': 'b' * 64}, quotes, 8, None),
                                'session-worker')
    assert full['payload_utf8_bytes'] == limit['payload_utf8_bytes']          # the exact-cap reference attains the payload bound
    assert full['envelope_bytes'] - base['envelope_bytes'] <= 2 * (524288 - 2)
    walk = probe.measure_packet('w', build_worker_setup('1' * 64, 0, {'sha256': 'f' * 64}, {'a.py': 'b' * 64}, nodes, 8, None),
                                'session-worker')
    assert walk['walked_nodes'] == limit['walked_nodes']
    assert probe.worker_setup_bound(base, payload_limit=100)['bound_proven'] is False      # a limit that does not fit refuses


def test_compare_capture_fails_for_oversized_unstable_or_missing_evidence_and_passes_for_variable_time(tmp_path):
    probe = _tool('r11_probe')
    cases = {}
    for tamper in ('oversize_diag', 'stable', 'missing', 'time_counter', None):
        folder = tmp_path / str(tamper)
        folder.mkdir()
        result, root = _capture_directory(folder, probe, tamper=tamper)
        cases[tamper] = (result, probe.compare_capture(result, root))
    assert cases[None][1]['all_within'] is True and cases['time_counter'][1]['all_within'] is True
    for name in ('oversize_diag', 'stable', 'missing'):
        assert cases[name][1]['all_within'] is False
    reply0 = next(r for r in cases['oversize_diag'][1]['rows'] if r['name'] == 'pair worker-0 reply')
    assert reply0['diagnostics_within_simulated'] is False
    stable0 = next(r for r in cases['stable'][1]['rows'] if r['name'] == 'pair worker-0 reply')
    assert stable0['diagnostics_stable_equal'] is False and stable0['diagnostics_within_simulated'] is True
    missing = [r for r in cases['missing'][1]['rows'] if r.get('missing')]
    assert [r['name'] for r in missing] == ['pair worker-3 reply']
    # a failing supplied comparison can never yield 'passed'
    ok = _full_result()
    assert cases[None][1]['binding'] == dict(errors=[], ok=True)
    assert probe.verdict(dict(ok, compare_capture=cases[None][1])) == ('passed', [])
    assert probe.verdict(dict(ok, compare_capture=cases['stable'][1])) == ('refused', ['comparison.compare_capture'])
    assert probe.verdict(dict(ok, compare_actual=dict(payload_ok=True, envelope_ok=False, compact_ok=True))) == \
        ('refused', ['comparison.compare_actual'])


def test_binding_rejects_tampered_chain_model_missing_terminal_and_unsupported_preparation(tmp_path):
    probe = _tool('r11_probe')
    expectations = {'chain': 'chain', 'model': 'worker-2 setup binding', 'terminal_missing': 'worker-5',
                    'prepared': 'planner terminal binding'}
    for tamper, needle in expectations.items():
        folder = tmp_path / tamper
        folder.mkdir()
        result, root = _capture_directory(folder, probe, tamper=tamper)
        comparison = probe.compare_capture(result, root)       # every packet has a VALID checksum (re-packed)
        assert comparison['all_within'] is False, tamper
        assert any(needle in error for error in comparison['binding']['errors']), (tamper, comparison['binding']['errors'])
    # stale same-length hashes: a setup whose sources differ only in a same-length hash value is not accepted
    folder = tmp_path / 'stale'
    folder.mkdir()
    result, root = _capture_directory(folder, probe)
    result['binding'] = dict(result['binding'], sources_sha256=probe._h({'x.py': 'a' * 64}))
    errors = probe.compare_capture(result, root)['binding']['errors']
    assert any('sources differs' in error for error in errors)


def test_protocol_verifiers_reject_stale_guards_codec_finals_init_errors_and_task_drift(tmp_path):
    probe = _tool('r11_probe')
    clean = tmp_path / 'clean'
    clean.mkdir()
    result, root = _capture_directory(clean, probe)
    baseline = probe.compare_capture(result, root)
    assert baseline['binding'] == dict(errors=[], ok=True) and baseline['all_within'] is True     # valid emulation passes
    expectations = {
        'stage_guard': 'pair worker-1 reply resource/codec',          # same-length stale binary hash, valid checksum
        'final_guard': 'worker-2 final resource/codec',
        'infra_resource': 'pair infrastructure',                      # stage infrastructure.resource stale hash
        'codec': 'worker-2 final resource/codec',                     # value codec drift
        'planner_final': 'planner worker_reports[3]',                 # valid-looking changed echoed final
        'final_pid': 'finish workers', 'final_decode': 'finish workers', 'final_blas': 'finish workers',
        'init_error_final': 'worker-4 terminal carries initialization_evidence_error',
        'init_error_planner': 'planner terminal carries initialization_evidence_error',
        'task': 'parent command tasks differ',
    }
    for tamper, needle in expectations.items():
        folder = tmp_path / tamper
        folder.mkdir()
        result, root = _capture_directory(folder, probe, tamper=tamper)
        comparison = probe.compare_capture(result, root)             # every packet re-packed: checksums are valid
        assert comparison['all_within'] is False, tamper
        assert any(needle in error for error in comparison['binding']['errors']), (tamper, comparison['binding']['errors'])
        assert probe.verdict(dict(_full_result(), compare_capture=comparison)) == ('refused', ['comparison.compare_capture'])


def test_repacked_null_scalar_or_list_bodies_are_refused_for_every_captured_packet_kind(tmp_path):
    from anygeometry._planning_executor import pack
    probe = _tool('r11_probe')
    result, root = _capture_directory(tmp_path, probe)
    assert probe.compare_capture(result, root)['all_within'] is True
    targets = (('session.json', 'session'), ('command-0.json', 'session-command'), ('reply-0.json', 'session-reply'),
               ('command-1.json', 'session-command'), ('result.json', 'result'),
               ('worker-1/session.json', 'session-worker'), ('worker-1/command-0.json', 'session-worker-command'),
               ('worker-1/reply-0.json', 'session-worker-reply'), ('worker-1/command-1.json', 'session-worker-command'),
               ('worker-1/result.json', 'worker-result'))
    for name, kind in targets:
        path = root / name
        original = path.read_bytes()
        for value in (None, 5, 'text', [1]):
            path.write_text(json.dumps(pack(value, kind)), encoding='utf-8')          # valid PP2 checksum, wrong body type
            comparison = probe.compare_capture(result, root)
            assert comparison['all_within'] is False, (name, value)
            assert any('must be a JSON object' in error and path.name in error for error in comparison['binding']['errors']), \
                (name, value, comparison['binding']['errors'])
        path.write_bytes(original)
    assert probe.compare_capture(result, root)['all_within'] is True


def test_every_captured_command_including_finish_passes_the_production_grammar(tmp_path):
    probe = _tool('r11_probe')
    expectations = {'finish_tasks': 'finish command grammar refused', 'finish_unknown': 'finish command grammar refused',
                    'worker_finish_tasks': 'finish worker-2 command grammar refused'}
    for tamper, needle in expectations.items():
        folder = tmp_path / tamper
        folder.mkdir()
        result, root = _capture_directory(folder, probe, tamper=tamper)         # chain hashes are recomputed downstream
        comparison = probe.compare_capture(result, root)
        errors = comparison['binding']['errors']
        assert comparison['all_within'] is False, tamper
        assert any(needle in error for error in errors), (tamper, errors)
        assert not any('command chain' in error for error in errors), (tamper, errors)   # the chain itself is consistent


def test_protocol_comparison_refuses_malformed_packets_instead_of_skipping_them(tmp_path):
    probe = _tool('r11_probe')
    result, root = _capture_directory(tmp_path, probe)
    (root / 'worker-1' / 'result.json').write_text('not json', encoding='utf-8')
    (root / 'reply-0.json').write_text('{"schema":"x"}', encoding='utf-8')
    errors = probe.compare_capture(result, root)['binding']['errors']
    assert any('worker-1/result.json unreadable' in error for error in errors)
    assert any('reply-0.json unreadable' in error for error in errors)
    again = tmp_path / 'again'
    again.mkdir()
    result, root = _capture_directory(again, probe)
    bare = dict(result, binding={k: v for k, v in result['binding'].items() if k != 'numerical_manifest'})
    assert any('protocol binding identities' in e for e in probe.compare_capture(bare, root)['binding']['errors'])


def test_terminal_proof_uses_shared_builders_validates_finish_reports_and_refuses_preparation():
    probe = _tool('r11_probe')
    stage, kwargs = _synthetic_proof_inputs()
    proof = probe.terminal_proof(('pair', 'face'), ['a' * 64, 'b' * 64], session=kwargs['session'], previous='c' * 64, workers=8,
        packet=kwargs['packet'], numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'])
    assert proof['within'] is True and proof['errors'] == [] and len(proof['components']) == 1 + 8 + 8 + 1
    assert 'prepared_entries=None' in proof['scope'] and len(proof['worker_results']) == 8
    refused = probe.terminal_proof(('pair',), ['a' * 64], session=kwargs['session'], previous='c' * 64, workers=8,
        packet=kwargs['packet'], numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'],
        preparation=[{'kind': 'pair'}])
    assert refused['within'] is False and 'alternative preparation' in refused['errors'][0]
    limited = probe.terminal_proof(('pair',), ['a' * 64], session=kwargs['session'], previous='c' * 64, workers=8,
        packet=kwargs['packet'], numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'],
        payload_limit=1000)
    assert limited['within'] is False and any(e.startswith('limit:') for e in limited['errors'])
    # no stage seen => a finish command is not a valid session command: refused, not claimed
    assert probe.terminal_proof((), [], session=kwargs['session'], previous='c' * 64, workers=8, packet=kwargs['packet'],
        numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'])['within'] is False


def test_terminal_verifiers_enforce_the_declared_metadata_ceilings():
    from anygeometry._planning_qualification import QualificationDecline
    from anygeometry._planning_session import _verify_finish_workers
    good = dict(status='passed', worker=0, stages=['pair'], gate_closures=1, model_sha256='m', model_decode_count=1, pid=5,
                blas_threads=[dict(num_threads=1)], process_seconds=0.5)
    _verify_finish_workers([good], ['pair'], 1, 'm')
    for bad in (dict(good, pid=2**32), dict(good, blas_threads=[dict(num_threads=1, filepath='x' * 1025)]),
                dict(good, blas_threads=[dict(num_threads=1)] * 9), dict(good, process_seconds=10**100),
                dict(good, process_seconds=float('inf')), dict(good, process_seconds=-1.0), dict(good, process_seconds=True)):
        with pytest.raises(QualificationDecline):
            _verify_finish_workers([bad], ['pair'], 1, 'm')


def test_bounded_duration_is_one_validator_that_never_overflows():
    from anygeometry._planning_session import bounded_duration
    assert all(bounded_duration(v) for v in (0, 1, 2**63 - 1, 0.0, 1.5, 1e308))
    for bad in (-1, 2**63, 10**100, -0.0 - 1, float('inf'), float('nan'), True, False, '1', None, 1 + 0j):
        assert bounded_duration(bad) is False                  # a huge int is refused BEFORE any float conversion
    assert bounded_duration(-0.0) is True


def test_known_record_timing_field_is_bounded_and_excluded_only_by_name():
    probe = _tool('r11_probe')
    from anygeometry._planning_session import check_diagnostic_timing, stable_diagnostics
    from anygeometry._planning_qualification import QualificationDecline
    base = dict(counters={'a': 1, 'x_time_ns': 5},
                records={'arrangement_snap_fallback': [dict(face_id=3, moved_edges=2, max_move=0.25, seconds=0.0123)],
                         'other': [dict(seconds=7.5, value=1)]})
    varied = dict(base, counters={'a': 1, 'x_time_ns': 99999},
                  records=dict(base['records'], arrangement_snap_fallback=[dict(face_id=3, moved_edges=2, max_move=0.25, seconds=9.0)]))
    assert stable_diagnostics(base) == stable_diagnostics(varied)               # timing variation alone is stable
    for name, edit in (('face', dict(face_id=4)), ('moved', dict(moved_edges=3)), ('move', dict(max_move=0.5))):
        changed = dict(base, records=dict(base['records'], arrangement_snap_fallback=[dict(base['records']['arrangement_snap_fallback'][0], **edit)]))
        assert stable_diagnostics(changed) != stable_diagnostics(base), name     # any non-time content still differs
    unknown = dict(base, records=dict(base['records'], other=[dict(seconds=8.5, value=1)]))
    assert stable_diagnostics(unknown) != stable_diagnostics(base)              # no blanket stripping of other 'seconds'
    check_diagnostic_timing(base)
    for bad in (-1.0, float('nan'), 10**100, 'x', None, True):
        with pytest.raises(QualificationDecline):
            check_diagnostic_timing(dict(records={'arrangement_snap_fallback': [dict(seconds=bad)]}))
    with pytest.raises(QualificationDecline):
        check_diagnostic_timing(dict(records={'arrangement_snap_fallback': [dict(face_id=1)]}))      # field required
    # bound form: the known field takes the worst float repr; non-time content is untouched
    bound = probe.bound_diagnostics(base)
    assert bound['records']['arrangement_snap_fallback'][0]['seconds'] == probe.FLOAT_WORST
    assert bound['records']['other'] == base['records']['other'] and bound['counters']['x_time_ns'] == 2**63 - 1
    assert len(json.dumps(bound)) >= len(json.dumps(varied))
    # the production verifier rejects a malformed known timing field in a worker diagnostic
    stage, kwargs = _synthetic_proof_inputs()
    stage = dict(stage, diagnostics=[dict(d, records=dict(d['records'], arrangement_snap_fallback=[dict(seconds=0.5)])) for d in stage['diagnostics']])
    assert probe.stage_proof(stage, **kwargs)['within'] is True
    stage = dict(stage, diagnostics=[dict(d, records=dict(d['records'], arrangement_snap_fallback=[dict(seconds=-2.0)])) for d in stage['diagnostics']])
    assert probe.stage_proof(stage, **kwargs)['within'] is False


def test_missing_worker_setup_packet_fails_the_comparison(tmp_path):
    probe = _tool('r11_probe')
    result, root = _capture_directory(tmp_path, probe)
    (root / 'worker-5' / 'session.json').unlink()
    comparison = probe.compare_capture(result, root)
    assert comparison['all_within'] is False
    assert [r['name'] for r in comparison['rows'] if r.get('missing')] == ['worker-5 setup']


def test_reply_metadata_numeric_and_list_limits_are_enforced_by_the_production_verifier():
    import copy
    probe = _tool('r11_probe')
    from anygeometry._planning_qualification import QualificationDecline
    stage, kwargs = _synthetic_proof_inputs()
    body, _, session_body = probe.build_stage_reply(stage, worst=False, **kwargs)
    def verify(mutate=None):
        candidate = copy.deepcopy(session_body)
        if mutate:
            mutate(candidate['worker_reports'][0])
        probe.verify_with_production_schema(stage, body, candidate, workers=8, packet=kwargs['packet'],
            numerical=kwargs['numerical'], infrastructure_native=kwargs['infrastructure_native'])
    verify()
    mutations = {
        'counter above int63': lambda w: w['diagnostics'][0][1]['counters'].__setitem__('clip_time_ns', 2**63),
        'pid above 32 bit': lambda w: w.__setitem__('pid', 2**32),
        'reset_count above int63': lambda w: w.__setitem__('reset_count', 2**63),
        'encoded bytes differ from rows': lambda w: w['task_timing'].__setitem__('encoded_utf8_bytes', w['task_timing']['encoded_utf8_bytes'] + 1),
        'task_seconds length': lambda w: w['task_seconds'].pop(),
        'negative duration': lambda w: w['boundary_seconds'].__setitem__(0, -1.0),
        'nan duration': lambda w: w.__setitem__('assignment_seconds', float('nan')),
        'bool duration': lambda w: w.__setitem__('startup_seconds', True),
        'blas string': lambda w: w['blas_threads'][0].__setitem__('filepath', 'x' * 1025),
        'task_timing string': lambda w: w['task_timing'].__setitem__('total_wall', 'x'),
    }
    for name, mutate in mutations.items():
        with pytest.raises(QualificationDecline):
            verify(mutate)
    # valid decoded diagnostics at the int63 cap, and a non-BMP/backslash/quote BLAS path, stay accepted
    verify(lambda w: w['diagnostics'][0][1]['counters'].__setitem__('clip_time_ns', 2**63 - 1))
    verify(lambda w: w['blas_threads'][0].__setitem__('filepath', 'C:\\"\U0010FFFF' * 100))


def test_model_loader_extracts_project_geometry_and_refuses_ambiguity(tmp_path):
    probe = _tool('r11_probe')
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import to_dict
    document = to_dict(_fixture()[0])
    def write(name, value):
        path = tmp_path / name
        raw = json.dumps(value).encode() if not isinstance(value, bytes) else value
        path.write_bytes(raw)
        return path, hashlib_sha(raw)
    import hashlib
    hashlib_sha = lambda raw: hashlib.sha256(raw).hexdigest()
    path, digest = write('plain.json', document)
    model, actual = probe.load_model(path, digest)
    assert actual == digest and to_dict(model)['checksum'] == document['checksum']
    path, digest = write('project.anyfem', dict(name='project', geometry=document, other={'x': 1}))
    model, actual = probe.load_model(path, digest)           # the hash binds the whole project file
    assert actual == digest and to_dict(model)['checksum'] == document['checksum']
    with pytest.raises(SystemExit, match='sha256 mismatch'):
        probe.load_model(path, '0' * 64)
    for name, value, message in (('ambiguous', dict(document, geometry=document), 'ambiguous'),
                                 ('bad_geometry', dict(geometry=[1]), 'malformed project geometry'),
                                 ('no_checksum_geometry', dict(geometry={'a': 1}), 'malformed project geometry'),
                                 ('neither', {'a': 1}, 'neither'), ('array', [1, 2], 'JSON object'),
                                 ('garbage', b'not json', 'not valid JSON')):
        path, digest = write(name, value)
        with pytest.raises(SystemExit, match=message):
            probe.load_model(path, digest)


def test_probe_preserves_partial_output_on_failure(tmp_path, monkeypatch):
    probe = _tool('r11_probe')
    from anygeometry._planning_canaries import _fixture
    from anygeometry import _planning_manifest
    model, _ = _fixture()
    monkeypatch.setattr(_planning_manifest, 'capture_loaded_manifest', lambda **kw: (_ for _ in ()).throw(RuntimeError('boom')))
    result = probe.run_probe(model, tuple(model.faces), tmp_path, thaw_equivalence=False)
    saved = json.loads((tmp_path / 'r11-probe.json').read_text())
    assert result['complete'] is False and saved['complete'] is False
    assert 'model_packet' in saved and any('boom' in row['error'] for row in saved['errors'])


def test_probe_packet_measure_and_walk_counts():
    probe = _tool('r11_probe')
    # containers count themselves plus one per dict key; scalars count once:
    # top(1+2 keys) + 'x' + list + 1 + 2 + dict(1+1 key) + 3 == 10
    assert probe.walk_nodes({'a': [1, 2, {'b': 3}], 'c': 'x'}) == 10
    row = probe.measure_packet('x', {'a': [1, 2, 3]}, 'tasks', payload_limit=4)
    assert row['payload_utf8_bytes'] == len('{"a":[1,2,3]}') and row['within']['payload'] is False
    assert row['envelope_bytes'] > row['payload_utf8_bytes']


def test_profile_case_selection_and_bounds():
    profile = _tool('r11_profile')
    data = dict(document={}, subsets=dict(elbow_r7=[3, 1, 2, 2]))
    assert profile.select_case(data, 'elbow_r7') == (1, 2, 3)
    assert profile.select_case(dict(subsets=[dict(case='elbow_sector60', faces=[5])]), 'elbow_sector60') == (5,)
    for bad in (dict(subsets={}), dict(subsets=dict(elbow_r7=[])), dict(subsets=dict(elbow_r7=[0])),
                dict(subsets=dict(elbow_r7='x')), dict()):
        with pytest.raises(SystemExit):
            profile.select_case(bad, 'elbow_r7')
    a = dict(count=2, kernel_wall=1., total_wall=2., max_kernel_wall=.7)
    b = dict(count=3, kernel_wall=3., total_wall=4., max_kernel_wall=.9)
    receipt = dict(costs=dict(infrastructure=[dict(workers=[dict(task_timing=a), dict(task_timing=b), dict()])]))
    assert profile.timing_summary(receipt) == dict(tasks=5, kernel_wall=4., total_wall=6., max_kernel_wall=.9)
    kept, excluded = profile.deterministic_diagnostics(
        {'a': 1, 'intersection_arrangements_time_ns': 5, 'edge_cache_hits': 3, 'b': 2})
    assert kept == {'a': 1, 'b': 2} and excluded == ['edge_cache_hits', 'intersection_arrangements_time_ns']


def test_profile_classifies_declined_forced_run_as_incomplete():
    profile = _tool('r11_profile')
    ok = profile.classify('forced-parallel', dict(disposition='parallel', retired=True, forced_selector='x',
        stages=[dict(stage='pair', accepted=3, total=3, accepted_weight=3., total_weight=3.)]))
    assert ok['complete'] is True and ok['accepted'][0]['accepted'] == 3
    for receipt in (dict(disposition='serial', retired=True, reason='declined', stages=[]),
                    dict(disposition='terminal', retired=True, reason='x', stages=[]),
                    dict(disposition='parallel', retired=False, stages=[dict(stage='pair')]), None):
        row = profile.classify('forced-parallel', receipt)
        assert row['complete'] is False
    assert profile.classify('serial', None)['complete'] is True
    assert profile.classify('forced-parallel', dict(disposition='serial', retired=True, reason='declined', stages=[]))['reason'] == 'declined'


def test_forced_route_skips_only_the_cost_gate(monkeypatch):
    from types import SimpleNamespace
    from anygeometry import _planning_controller as controller
    from anygeometry._planning_qualification import QualificationDecline
    from anygeometry.runtime_diagnostics import collect_runtime_records
    policy = SimpleNamespace(cancellation_check=None, max_predicates=None, max_candidate_pairs=None)
    monkeypatch.setattr(controller, '_estimate_cost', lambda model, operands=None: (0, 'tiny'))
    monkeypatch.setattr(controller, '_storage', lambda: (_ for _ in ()).throw(QualificationDecline('reached setup')))
    with collect_runtime_records() as records:
        assert controller.run(None, lambda: 'serial', lambda b: pytest.fail('parallel'), policy=policy, mode='exact') == 'serial'
    assert 'routed serial' in records['parallel_planning'][-1]['reason']                 # default still gated
    with collect_runtime_records() as records:
        with controller._forced_route('r11-test'):
            assert controller.run(None, lambda: 'serial', lambda b: pytest.fail('parallel'), policy=policy, mode='exact') == 'serial'
    receipt = records['parallel_planning'][-1]
    assert receipt['reason'] == 'reached setup' and receipt['forced_selector'] == 'r11-test'
    # unsupported policy still exits before everything, forced or not
    with controller._forced_route('r11-test'):
        bad = SimpleNamespace(cancellation_check=None, max_predicates=1, max_candidate_pairs=None)
        monkeypatch.setattr(controller, '_storage', lambda: pytest.fail('setup for unsupported policy'))
        assert controller.run(None, lambda: 'serial', None, policy=bad, mode='exact') == 'serial'
    with pytest.raises(QualificationDecline):
        with controller._forced_route(''):
            pass
    assert controller._FORCE_ROUTE.get() is None


def test_stage_task_reports_separated_kernel_and_encode_timing():
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_stages import WorkerModel, stage_task
    model, _ = _fixture()
    faces = tuple(model.faces)
    tolerance = model.tolerance.effective_length(5.)
    context = WorkerModel(freeze_model(model))
    value = stage_task(context, dict(kind='pair', first=faces[0], second=faces[1], tolerance=tolerance,
        face_tolerances=[tolerance, tolerance], cost=1.))
    timing = context.last_timing
    assert set(timing) == {'setup_wall', 'setup_cpu', 'kernel_wall', 'kernel_cpu', 'encode_wall', 'encode_cpu', 'encoded_utf8_bytes'}
    assert all(v >= 0 for v in timing.values()) and timing['kernel_wall'] > 0 and timing['setup_wall'] > 0
    assert timing['encoded_utf8_bytes'] == len(value.encode('utf-8'))


def test_timing_aggregate_schema_is_fixed_size_and_gate_initializes_it():
    from anygeometry import _planning_batch as batch
    from anygeometry import _planning_session as session
    expected = {'count', 'total_wall', 'total_cpu', 'setup_wall', 'setup_cpu', 'kernel_wall', 'kernel_cpu',
        'encode_wall', 'encode_cpu', 'bookkeeping_wall', 'bookkeeping_cpu', 'encoded_utf8_bytes',
        'max_total_wall', 'max_kernel_wall'}
    assert session._TIMING_FIELDS == expected
    # Gate source initializes exactly the same field set (checked without admitting a gate).
    import inspect
    init = inspect.getsource(batch.BatchWorkerGate.__init__)
    assert all(name + '=' in init for name in expected)
    huge = {name: 0. for name in expected}
    huge.update(count=10**6, encoded_utf8_bytes=10**9)
    assert len(json.dumps(huge)) < 600          # wire volume independent of task count


def test_prepared_stage_timing_validation_is_optional_strict_and_dict_only():
    from anygeometry import _planning_session as session
    import inspect
    source = inspect.getsource(session.verify_prepared_stage)
    assert "worker.get('task_timing')" in source and "timing is not None" in source   # old reports omit it
    assert "type(timing) is not dict" in source and "timing['count']!=len(assigned)" in source   # no per-task lists
