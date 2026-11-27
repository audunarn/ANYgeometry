"""The CI driver budget follows the job timeout of the macOS Intel shards (owner, 2026-11-26).

Run 45843638335: with the Intel Mac pytest job raised to 45 minutes, shard 2/16 still stopped at
32 minutes with "shared CI driver budget exhausted", because ``PhaseBudget`` had its own fixed
1800 s budget. Owner decision: the driver reads its budget from ``ANYGEOMETRY_CI_DRIVER_SECONDS``,
which the workflow sets to 2640 s for macos-15-intel only; every other platform keeps 1800 s, and
the validation cap is 2700 s. A missing variable means 1800 s; a malformed one is refused.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import ci_isolated_tests as ci  # noqa: E402

WORKFLOW = ROOT / '.github' / 'workflows' / 'ci.yml'


def test_missing_variable_keeps_the_30_minute_driver_budget():
    assert ci.driver_budget_seconds({}) == 1800


@pytest.mark.parametrize('value, seconds', [('2640', 2640), ('1800', 1800), ('2700', 2700), ('1', 1), ('02640', 2640), ('0001', 1), ('0' * 5000 + '1', 1)])
def test_the_variable_sets_the_driver_budget(value, seconds):
    assert ci.driver_budget_seconds({'ANYGEOMETRY_CI_DRIVER_SECONDS': value}) == seconds
    assert ci.PhaseBudget(seconds).seconds == seconds


@pytest.mark.parametrize('value', ['', '0', '0000', '02701', '-5', '2701', '3600', '2640.0', ' 2640', '2640s', 'abc', '+2640', '０', '0' * 5000, '0' * 5000 + '2701', '9' * 5000])
def test_a_malformed_or_out_of_range_variable_is_refused(value):
    with pytest.raises(ValueError, match='CI driver budget'):
        ci.driver_budget_seconds({'ANYGEOMETRY_CI_DRIVER_SECONDS': value})


@pytest.mark.parametrize('seconds', [0, 2701, 1800.0, True])
def test_phase_budget_refuses_values_past_the_cap(seconds):
    with pytest.raises(ValueError, match='invalid CI driver budget'):
        ci.PhaseBudget(seconds)


def test_the_workflow_raises_the_driver_budget_only_for_intel_mac_pytest_shards():
    text = WORKFLOW.read_text(encoding='utf-8')
    line = "ANYGEOMETRY_CI_DRIVER_SECONDS: ${{ matrix.os == 'macos-15-intel' && '2640' || '1800' }}"
    assert text.count('ANYGEOMETRY_CI_DRIVER_SECONDS') == 1 and text.count(line) == 1
    regular = re.search(r'(?ms)^  pytest:\n(.*?)(?=^  [\w-]+:\n)', text).group(1)
    assert line in regular.split('    steps:\n', 1)[0]
    assert "timeout-minutes: ${{ matrix.os == 'macos-15-intel' && 50 || 30 }}" in regular
    # Owner decision 2026-11-28 (option 1): the Intel Mac job is 50 minutes, the budget stays 2640 s.
    # Measured (runs 46021054104, 46047881135; PR #58 attempt 1): setup before the test step up to 144 s,
    # steps after it up to 67 s; the 45-minute job left 60 s, so the job cap fired before the driver's
    # typed deadline. 50 minutes leaves 360 s.
    assert 50 * 60 - 2640 == 360


def test_main_reads_the_variable(monkeypatch, tmp_path):
    seen = []

    class Stop(Exception):
        pass

    def budget(seconds=1800):
        seen.append(seconds)
        raise Stop
    monkeypatch.setattr(ci, 'PhaseBudget', budget)
    monkeypatch.setenv('ANYGEOMETRY_CI_DRIVER_SECONDS', '2640')
    monkeypatch.delenv('PYTEST_ADDOPTS', raising=False)
    with pytest.raises(Stop):
        ci.main(['--profile', 'full', '--evidence-dir', str(tmp_path / 'evidence')])
    assert seen == [2640]
