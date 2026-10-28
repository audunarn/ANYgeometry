"""Publication failures restore live state, rather than only a topology snapshot."""
import pytest

from anygeometry import GeometryModel
from anygeometry import translational_imprint as owner


def fixture():
    model = GeometryModel()
    model.add_point(0., 0., 0.)
    staged = model.clone(preserve_identity=True)
    staged.add_point(1., 2., 3.)
    return model, staged


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_checksum_failure_restores_content_counters_views_and_receipt(monkeypatch, failure):
    model, staged = fixture()
    before = owner._checksum(model)
    revision = model.revision
    ids = dict(model._next_id)
    versions = dict(model._entity_versions)
    view = model.vertices
    previous = (object(), revision, before, ())
    owner._RECEIPTS[model] = previous
    events = []
    model._change_hooks.append(events.append)
    real_checksum = owner._checksum

    def fail_published(candidate):
        if candidate is model:
            raise failure('injected final validation failure')
        return real_checksum(candidate)

    monkeypatch.setattr(owner, '_checksum', fail_published)
    with pytest.raises(failure):
        owner._publish_qualified(model, staged, object(), real_checksum(staged))
    assert real_checksum(model) == before
    assert model.revision == revision
    assert model._next_id == ids
    assert model._entity_versions == versions
    assert tuple(view) == tuple(model.vertices)
    assert owner._RECEIPTS[model] is previous
    assert events == []


def test_restore_failure_after_mutation_rolls_back(monkeypatch):
    model, staged = fixture()
    before = owner._checksum(model)
    revision = model.revision
    restore = model.restore_topology

    def failing(snapshot):
        restore(snapshot)
        raise RuntimeError('injected restore failure after publication')

    monkeypatch.setattr(model, 'restore_topology', failing)
    with pytest.raises(RuntimeError):
        owner._publish_qualified(model, staged, object(), owner._checksum(staged))
    assert owner._checksum(model) == before
    assert model.revision == revision
    assert model not in owner._RECEIPTS


def test_success_notifies_only_with_live_receipt():
    model, staged = fixture()
    events = []
    receipt = object()
    model._change_hooks.append(lambda change: events.append((change, owner._RECEIPTS[model][0])))
    owner._publish_qualified(model, staged, receipt, owner._checksum(staged))
    assert owner._checksum(model) == owner._checksum(staged)
    assert len(events) == 1 and events[0][1] is receipt


def test_registration_cancellation_restores_prior_state():
    model, staged = fixture()
    before = owner._checksum(model)
    revision = model.revision
    ids = dict(model._next_id)
    def cancel(*args):
        raise KeyboardInterrupt('injected cancellation')
    with pytest.raises(KeyboardInterrupt):
        owner._publish_qualified(model, staged, object(), owner._checksum(staged),
                                 register=cancel)
    assert owner._checksum(model) == before
    assert model.revision == revision
    assert model._next_id == ids
    assert model not in owner._RECEIPTS


def test_observer_cannot_veto_commit_after_another_observer_saw_it():
    model, staged = fixture()
    observed = []
    model._change_hooks.append(lambda change: observed.append((model.revision, tuple(model.vertices))))
    def cancel(change):
        raise KeyboardInterrupt('observer cancellation after commit')
    model._change_hooks.append(cancel)
    owner._publish_qualified(model, staged, object(), owner._checksum(staged))
    assert observed == [(model.revision, tuple(model.vertices))]
    assert owner._checksum(model) == owner._checksum(staged)
