"""The public default is serial; only explicit1 enters the experimental hook."""
import pytest

from anygeometry import GeometryModel
from anygeometry import batch_intersections as batch, _planning_host as host
from anygeometry._planning_code import exact_numerical_record
from anygeometry.serialization import from_dict, to_dict


SETTING = 'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING'


def fixture():
    model = GeometryModel(model_id='777f6179-d87c-4b5d-964a-4e70a2fc52a8')
    for points in (((0.,0.,0.),(2.,0.,0.),(2.,2.,0.),(0.,2.,0.)),
                   ((0.,1.,-1.),(2.,1.,-1.),(2.,1.,1.),(0.,1.,1.))):
        model.add_sheet((model.add_plate(model.add_points(points)),))
    return to_dict(model)


@pytest.mark.parametrize('setting', [None, '0', 'other'])
def test_default_and_non_opt_in_values_use_real_serial_plan_and_apply(monkeypatch, setting):
    document = fixture()
    expected, actual = from_dict(document), from_dict(document)
    monkeypatch.setenv(SETTING, '0')
    reference = batch.plan_intersections(expected, tuple(expected.handle('sheet', i) for i in expected.sheets), policy='connect')
    reference_outcome = batch.apply_intersections(expected, reference, policy='connect')
    if setting is None:
        monkeypatch.delenv(SETTING, raising=False)
    else:
        monkeypatch.setenv(SETTING, setting)
    def forbidden(*args, **kwargs):
        pytest.fail('serial request entered experimental planner setup')
    monkeypatch.setattr(host, 'qualified_plan', forbidden)
    plan = batch.plan_intersections(actual, tuple(actual.handle('sheet', i) for i in actual.sheets), policy='connect')
    outcome = batch.apply_intersections(actual, plan, policy='connect')
    assert exact_numerical_record(plan) == exact_numerical_record(reference)
    assert exact_numerical_record(outcome) == exact_numerical_record(reference_outcome)
    assert to_dict(actual) == to_dict(expected)


def test_explicit_one_preserves_public_experimental_hook(monkeypatch):
    model = from_dict(fixture())
    observed = []
    def qualified(*args, **kwargs):
        observed.append((args[0], kwargs['mode']))
        return args[2]()                    # actual serial callable, no fake plan
    monkeypatch.setattr(host, 'qualified_plan', qualified)
    monkeypatch.setenv(SETTING, '1')
    plan = batch.plan_intersections(model, tuple(model.handle('sheet', i) for i in model.sheets), policy='connect')
    assert observed == [(model, 'exact')]
    assert plan.content_checksum


def test_default_serial_keeps_original_callback_exception(monkeypatch):
    model = from_dict(fixture())
    monkeypatch.delenv(SETTING, raising=False)
    original = RuntimeError('caller callback exception')
    def callback():
        raise original
    def forbidden(*args, **kwargs):
        pytest.fail('default serial entered experimental planner setup')
    monkeypatch.setattr(host, 'qualified_plan', forbidden)
    with pytest.raises(RuntimeError) as caught:
        batch.plan_intersections(model, tuple(model.faces),
            policy=batch.IntersectionBatchPolicy(cancellation_check=callback))
    assert caught.value is original
