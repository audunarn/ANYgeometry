import pytest
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics, _timed_phase


def test_timing_is_observational_nested_and_retained_on_exception():
    with collect_runtime_diagnostics() as outer:
        with pytest.raises(ValueError):
            with collect_runtime_diagnostics() as inner:
                with _timed_phase('failed_phase'):
                    raise ValueError('refused')
        assert inner['failed_phase_time_ns'] > 0
        assert outer['failed_phase_time_ns'] == inner['failed_phase_time_ns']
    with collect_runtime_diagnostics() as fresh:
        assert 'failed_phase_time_ns' not in fresh
