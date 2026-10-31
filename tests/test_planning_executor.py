"""PP2 regressions: absent on the parent, no stand-in production gate changes."""
import importlib
import json
import pytest
from anygeometry._planning_qualification import QualificationDecline


def api():
    return importlib.import_module('anygeometry._planning_executor')


def test_transport_checksum_and_schema():
    a = api()
    packet = a.pack({'value': [-0.0, 0.0, 2.0]}, 'model')
    assert a.unpack(packet, 'model') == {'value': [-0.0, 0.0, 2.0]}
    bad = dict(packet, payload=packet['payload'] + ' ')
    with pytest.raises(QualificationDecline, match='checksum'):
        a.unpack(bad, 'model')
    with pytest.raises(QualificationDecline, match='kind'):
        a.unpack(packet, 'result')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {'x': object()}])
def test_transport_refuses_non_json(value):
    with pytest.raises(QualificationDecline):
        api().pack(value, 'model')


def test_duplicate_keys_decline():
    a = api()
    import hashlib
    text = '{"x":1,"x":2}'
    packet = a.pack({}, 'model')
    packet.update(payload=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(QualificationDecline, match='duplicate'):
        a.unpack(packet, 'model')


@pytest.mark.parametrize('pools', [[], [{'num_threads': 2}], [{'num_threads': True}]])
def test_wrong_thread_count_declines(pools):
    with pytest.raises(QualificationDecline, match='thread'):
        api().verify_threads(pools)


def test_default_assignment_and_smaller_k():
    a = api()
    assert a.assignment_size(None, 600) == 600
    assert a.assignment_size(None, 0) == 1
    assert a.assignment_size(256, 600) == 256
    for bad in (True, 0, -1, 'whole'):
        with pytest.raises(QualificationDecline):
            a.assignment_size(bad, 600)


def test_pool_off_does_not_inspect_model_or_start(monkeypatch, tmp_path):
    a = api()
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING', raising=False)
    monkeypatch.setattr(a, 'freeze_model', lambda m: pytest.fail('OFF inspected model'))
    seen = []
    result = a.execute(object(), ('pair', 'face'), lambda task: seen.append(task) or task,
                       directory=tmp_path/'unused')
    assert result['mode'] == 'serial' and result['results'] == ('pair', 'face')
    assert seen == ['pair', 'face'] and not (tmp_path/'unused').exists()


@pytest.mark.parametrize('reason', ['transport checksum', 'thread count', 'worker canary',
    'worker boundary', 'worker exit', 'planner exit', 'result coverage', 'unsupported model',
    'source changed', 'deadline'])
def test_refusal_retires_then_original_serial(reason, monkeypatch, tmp_path):
    a = api()
    events = []
    class Fake:
        def __init__(self, *args, **kwargs): events.append('start')
        def submit(self, *args, **kwargs): raise QualificationDecline(reason)
        def close(self): events.append('close')
    monkeypatch.setattr(a, 'OwnedPlanner', Fake)
    result = a.execute(object(), ('pair',), lambda t: events.append('serial') or 's',
                       directory=tmp_path/'run', opt_in=True)
    assert events == ['start', 'close', 'serial']
    assert result['mode'] == 'serial' and reason in result['decline_reason']


def test_cancel_is_not_serial_replay(monkeypatch, tmp_path):
    a = api()
    events = []
    class Fake:
        def __init__(self, *args, **kwargs): pass
        def submit(self, *args, **kwargs): raise KeyboardInterrupt()
        def close(self): events.append('close')
    monkeypatch.setattr(a, 'OwnedPlanner', Fake)
    with pytest.raises(KeyboardInterrupt):
        a.execute(object(), ('pair',), lambda t: pytest.fail('cancel replayed'),
                  directory=tmp_path/'run', opt_in=True)
    assert events == ['close']


def test_cleanup_failure_prevents_fallback(monkeypatch, tmp_path):
    a = api()
    class Fake:
        def __init__(self, *args, **kwargs): pass
        def submit(self, *args, **kwargs): raise QualificationDecline('worker')
        def close(self): raise RuntimeError('cleanup failed')
    monkeypatch.setattr(a, 'OwnedPlanner', Fake)
    with pytest.raises(RuntimeError, match='cleanup'):
        a.execute(object(), ('pair',), lambda t: pytest.fail('cleanup replayed'),
                  directory=tmp_path/'run', opt_in=True)


def test_bad_result_coverage_declines():
    a = api()
    for result in (['a'], ['a', 'b'], [[0, 'a'], [0, 'b']], [[0, 'a'], [2, 'b']]):
        with pytest.raises(QualificationDecline):
            a.reduce_results(result, 2)
    assert a.reduce_results([[1, 'b'], [0, 'a']], 2) == ('a', 'b')


def test_model_transport_and_nonfield_refusal():
    a = api()
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import from_dict, to_dict
    model, _ = _fixture()
    # A cold reconstructed model, without warm canary-domain caches.
    model = from_dict(to_dict(model))
    packet = a.freeze_model(model)
    restored = a.thaw_model(packet)
    assert to_dict(restored) == to_dict(model)
    model.extra_behavior = lambda: None
    with pytest.raises(QualificationDecline):
        a.freeze_model(model)


def test_sampling_seed_binds_dispatch_plan_not_only_model(monkeypatch, tmp_path):
    a = api()
    candidate = dict(results=('value',), task_layout=((0, 0, 'pair'),),
                     model_sha256='a'*64, request_sha256='b'*64)
    class Fake:
        def __init__(self, *args, **kwargs): pass
        def submit(self, *args, **kwargs): return candidate
        def close(self): pass
    monkeypatch.setattr(a, 'OwnedPlanner', Fake)
    result = a.execute(object(), ('pair',), lambda _: 'value', directory=tmp_path/'unused',
        opt_in=True, expected_ci=('plan', 'apply'), observed_ci=lambda _: ('plan', 'apply'))
    assert result['mode'] == 'parallel'
    assert result['sampling']['plan_digest'] == candidate['request_sha256']
