"""A generic refusal must never replace required positive qualification CI."""
import importlib.util
from pathlib import Path

import pytest


def api():
    path = Path(__file__).resolve().parents[1] / 'tools/ci_planning_profile.py'
    spec = importlib.util.spec_from_file_location('ci_planning_profile', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(tmp_path, *, scope='original-profile-assertions', failed=False,
                 contract='numpy-context-immutable-images-v1', extra=''):
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    root = Element('testsuite')
    for name in api().REQUIRED_PROFILE_CASES:
        case = SubElement(root, 'testcase', classname=name.rsplit('::', 1)[0], name=name.rsplit('::', 1)[1])
        props = SubElement(case, 'properties')
        for key, value in [('numpy_profile_contract', contract), ('execution_scope', scope)]:
            SubElement(props, 'property', name=key, value=value)
        if failed:
            SubElement(case, 'failure', message='retained failure')
        if extra:
            SubElement(props, 'property', name='postcapture_qualification', value=extra)
    path = tmp_path/'results.xml'
    ElementTree(root).write(path)
    return path


def test_complete_positive_profile_is_required(tmp_path):
    report = write_report(tmp_path)
    assert api().verify_junit([report]) == 22


@pytest.mark.parametrize('change', ['refusal', 'failure', 'contract', 'false-qualification'])
def test_refusal_failure_or_wrong_contract_never_counts_as_positive(tmp_path, change):
    values = {'refusal': dict(scope='initial-profile-refusal'), 'failure': dict(failed=True),
              'contract': dict(contract='other'), 'false-qualification': dict(extra='False')}
    report = write_report(tmp_path, **values[change])
    with pytest.raises(ValueError):
        api().verify_junit([report])


def test_missing_or_duplicate_positive_report_refuses(tmp_path):
    report = write_report(tmp_path)
    with pytest.raises(ValueError):
        api().verify_junit([])
    with pytest.raises(ValueError):
        api().verify_junit([report, report])
