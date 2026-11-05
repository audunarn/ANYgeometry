"""Pure workflow metadata: preserve runtime coverage while expanding one lane."""
import ast
from collections import Counter
import json
from pathlib import Path
import re

WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml'


def job(text, name):
    match = re.search(r'^  ' + re.escape(name) + r':\n(.*?)(?=^  [\w-]+:|\Z)', text, re.M | re.S)
    assert match is not None, name
    return match.group(1)


def axis(block, name):
    value = re.search(r'^        ' + re.escape(name) + r': \[([^\n]*)\]$', block, re.M)
    assert value is not None, name
    if name == 'os':
        return [part.strip() for part in value.group(1).split(',')]
    return ast.literal_eval('[' + value.group(1) + ']')


def test_qualified_matrix_is_separate_and_preserves_all_runtime_assignments():
    text = WORKFLOW.read_text(encoding='utf-8')
    regular, qualified = job(text, 'pytest'), job(text, 'pytest-qualified')
    assert 'include:' not in regular and 'include:' not in qualified
    assert set(re.findall(r'^        ([\w-]+):', regular.split('    steps:\n')[0], re.M)) == {'os', 'python-version', 'shard', 'exclude'}
    assert set(re.findall(r'^        ([\w-]+):', qualified.split('    steps:\n')[0], re.M)) == {'os', 'python-version', 'shard'}
    systems = ['windows-latest', 'ubuntu-latest', 'macos-15', 'macos-15-intel']
    versions = ['3.11', '3.12', '3.13', '3.14']
    assert axis(regular, 'os') == systems
    assert axis(regular, 'python-version') == versions
    assert axis(regular, 'shard') == list(range(1, 17))
    exclusion = re.search(r'^        exclude:\n(.*?)(?=^    steps:)', regular, re.M | re.S)
    assert exclusion is not None
    assert exclusion.group(1).strip() == '- os: windows-latest\n            python-version: "3.14"'
    assert axis(qualified, 'os') == ['windows-latest']
    assert axis(qualified, 'python-version') == ['3.14']
    assert axis(qualified, 'shard') == list(range(1, 129))
    assert 'exclude:' not in qualified
    rows = [(system, version, shard) for system in systems for version in versions
            for shard in range(1, 17) if (system, version) != ('windows-latest', '3.14')]
    qrows = [('windows-latest', '3.14', shard) for shard in range(1, 129)]
    assert len(rows) == 240 and len(qrows) == 128
    assert max(len(rows), len(qrows)) <= 256
    assert len(set(rows + qrows)) == 368
    counts = Counter((system, version) for system, version, _ in rows + qrows)
    assert set(counts) == {(system, version) for system in systems for version in versions}
    assert all(value == (128 if pair == ('windows-latest', '3.14') else 16) for pair, value in counts.items())


def test_qualified_setup_test_and_artifact_steps_are_an_inverse_only_count_change():
    text = WORKFLOW.read_text(encoding='utf-8')
    regular, qualified = job(text, 'pytest'), job(text, 'pytest-qualified')
    assert regular.split('    steps:\n', 1)[1].strip() == qualified.split('    steps:\n', 1)[1].replace('/128', '/16').strip()
    for block in (regular, qualified):
        assert 'timeout-minutes: 30' in block
        assert 'fail-fast: false' in block
        assert "ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS: ${{ matrix.os == 'windows-latest' && matrix.python-version == '3.14' && '1' || '0' }}" in block
        assert "'numpy==2.4.6' 'scipy==1.17.1'" in block
        assert 'python tools/ci_planning_profile.py --preflight --report planning-profile.json' in block
        assert 'python tools/ci_isolated_tests.py --maxfail=1 --durations=20 --junitxml=kernel-results.xml' in block
        assert 'name: kernel-runtime-${{ matrix.os }}-${{ matrix.python-version }}-${{ matrix.shard }}' in block
        assert 'planning-profile.json' in block and 'ci-isolation/' in block
    assert 'ANYGEOMETRY_TEST_SHARD: ${{ matrix.shard }}/128' in qualified


def test_aggregate_requires_both_matrices_and_original_exact_coverage_and_profile_gate():
    gate = job(WORKFLOW.read_text(encoding='utf-8'), 'pytest-shards')
    assert 'needs: [pytest, pytest-qualified, minimum-dependencies]' in gate
    for variable, dependency in [('PYTEST_RESULT', 'pytest'), ('QUALIFIED_RESULT', 'pytest-qualified'),
                                 ('MINIMUM_RESULT', 'minimum-dependencies')]:
        assert variable + ': ${{ needs.' + dependency + '.result }}' in gate
        assert 'if [ "$' + variable + '" != success ]; then' in gate
    assert 'python tools/ci_shards.py verify lanes/kernel-runtime-$os-$py-*/shard-*.json' in gate
    assert 'python tools/ci_shards.py verify lanes/minimum-dependencies-runtime-*/shard-*.json' in gate
    assert 'python tools/ci_planning_profile.py lanes/kernel-runtime-windows-latest-3.14-*/kernel-results.xml' in gate


def test_sixty_four_reports_require_every_assigned_junit_result(tmp_path, monkeypatch):
    # Synthetic CI metadata only: no test body or numerical owner is executed.
    monkeypatch.syspath_prepend(str(WORKFLOW.parents[2] / 'tools'))
    import ci_shards
    selected = [f'tests/test_metadata.py::case_{i}' for i in range(64)]
    assignment = ci_shards.assign(selected, 64, {'tests': {}, 'rest_mean': {'windows': .1}})
    assert set(assignment) == set(selected) and set(assignment.values()) == set(range(1, 65))
    reports = []
    for index in range(1, 65):
        folder = tmp_path / str(index)
        folder.mkdir()
        node = next(node for node, owner in assignment.items() if owner == index)
        report = folder / f'shard-{index}.json'
        report.write_text(json.dumps(dict(index=index, count=64, selected=selected, ran=[node], slow=[])))
        name = node.split('::')[1]
        (folder / 'kernel-results.xml').write_text(
            f'<testsuites><testsuite><testcase classname="tests.test_metadata" name="{name}" time="0"/></testsuite></testsuites>')
        reports.append(report)
    assert ci_shards.verify(reports) == []
    assert any('do not cover 1..N exactly once' in error for error in ci_shards.verify(reports[:-1]))
    (reports[-1].parent / 'kernel-results.xml').unlink()
    assert any('no kernel-results.xml' in error for error in ci_shards.verify(reports))
