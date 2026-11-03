"""Require positive planner checks on the existing immutable runtime profile.

Generic CI lanes test initial refusal on other runtime images. They cannot
replace these positive checks or certify worker/postcapture execution.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


CONTRACT = 'numpy-context-immutable-images-v1'
REQUIRED_PROFILE_CASES = frozenset([
    'test_planning_r8_integration::test_production_ledger_has_strict_contexts_and_no_external_anchor',
    'test_planning_r8_integration::test_package_numpy_context_retains_getter_identity_without_anchor',
    'test_planning_r8_integration::test_cached_and_uncached_complete_role_graphs_are_identical',
    *('test_planning_r8_integration::test_public_eight_worker_canary_matches_serial['+warm+']'
      for warm in ('False', 'True')),
    'test_planning_numpy_context::test_capture_preserves_exact_effective_numpy_context',
    *('test_planning_numpy_context::test_effective_numpy_setting_change_is_terminal['+kind+']'
      for kind in ('divide', 'over', 'under', 'invalid', 'buffer')),
    'test_planning_numpy_context::test_loaded_manifest_binds_numpy_context_with_small_owned_root',
    'test_planning_numpy_context::test_loaded_manifest_callback_drift_is_terminal',
    *('test_planning_host_connected::test_actual_plane_pair_face_workload_under_unchanged_default_host['+warm+']'
      for warm in ('False', 'True')),
    'test_planning_loaded_manifest::test_real_cold_and_warm_callable_capture_preserves_host_caches',
    'test_planning_loaded_manifest::test_effective_setting_and_native_binary_are_bound',
    'test_planning_resource_lifetime::test_demanded_and_eager_syntax_preserve_complete_role_graph',
    *('test_planning_r8_controller::test_real_public_callback_after_capture_is_terminal['+action+']'
      for action in ('model', 'code', 'exception', 'cancel')),
])


def verify_junit(paths):
    seen = set()
    for path in paths:
        for case in ET.parse(path).iter('testcase'):
            key = case.get('classname', '').rsplit('.', 1)[-1]+'::'+case.get('name', '')
            if key not in REQUIRED_PROFILE_CASES:
                continue
            if key in seen:
                raise ValueError('duplicate required planning profile case: '+key)
            properties = [(item.get('name'), item.get('value')) for item in case.findall('./properties/property')]
            if (any(case.find(kind) is not None for kind in ('failure', 'error', 'skipped')) or
                    ('numpy_profile_contract', CONTRACT) not in properties or
                    ('execution_scope', 'original-profile-assertions') not in properties or
                    any(name == 'postcapture_qualification' and value.lower() == 'false'
                        for name, value in properties) or
                    any(name == 'execution_scope' and value.endswith('refusal')
                        for name, value in properties)):
                raise ValueError('required planning profile case did not run positive assertions: '+key)
            seen.add(key)
    missing = REQUIRED_PROFILE_CASES-seen
    if missing:
        raise ValueError('missing required planning profile cases: '+', '.join(sorted(missing)))
    return len(seen)


def preflight():
    if sys.platform != 'win32' or sys.version_info[:3] != (3, 14, 2):
        raise ValueError('positive planning lane requires Windows CPython3.14.2')
    required = os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS') == '1'
    if os.environ.get('CI') == 'true' and not required:
        raise ValueError('qualified CI lane did not require positive test execution')
    import numpy as np
    import scipy
    if np.__version__ != '2.4.6' or scipy.__version__ != '1.17.1':
        raise ValueError('positive planning lane numerical dependency versions differ')
    # Select the checkout before the first ANY import, including local checks
    # where an older global installation may exist. This is a source-CI tool.
    source = Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(source))
    import anygeometry
    if Path(anygeometry.__file__).resolve() != source/'anygeometry/__init__.py':
        raise ValueError('positive planning lane imported another geometry checkout')
    from anygeometry._planning_numpy_context import capture_numpy_context
    from anygeometry._planning_native import identity
    context = capture_numpy_context(package_only=True)
    context.verify()
    native = identity()
    return dict(contract=CONTRACT, python=list(sys.version_info[:3]), numpy=np.__version__,
                scipy=scipy.__version__, context=context.binding, native=native,
                required_positive_cases=len(REQUIRED_PROFILE_CASES), positive_assertions_required=required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('junit', nargs='*', type=Path)
    args = parser.parse_args()
    record = preflight() if args.preflight else dict(positive_cases=verify_junit(args.junit))
    if args.report:
        args.report.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
