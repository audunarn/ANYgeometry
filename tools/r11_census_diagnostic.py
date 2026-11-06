"""R11 paired OBSERVATIONAL census diagnostic (not a test, not a qualification, never marks anything passed).

Purpose: tell apart the causes of ``build_ledger``'s combined guard
``census.truncated or set(fresh) != set(objects) or any(fresh[p] is not v ...)`` ('ledger creation live coverage
mismatch') by running the SAME setup against two source trees (approved base and candidate), each in its OWN fresh
process:

  1. all four native thread variables must already be '1' BEFORE any import (refused otherwise);
  2. ``<source-root>/src`` is put first on sys.path BEFORE anygeometry is imported; the origin is verified;
  3. ambient imports of the failed test: pytest + the numpy-profile helper (optionally by importing a bootstrap test
     file, import-only: its functions are defined, never executed);
  4. ``anygeometry._planning_session`` is imported, then ``_planning_child.py`` is loaded BY FILE with
     ``runpy.run_path(..., run_name='r11_legacy_preparation')`` (non-anygeometry, non-__main__) and its real
     ``prepare(production=False)`` runs under ``threadpool_limits(1)``;
  5. the default module roots (every ``anygeometry``/``anygeometry.*`` entry of ``sorted(sys.modules.items())``) are
     captured as ``loaded_geometry_ledger`` does and ``_planning_ledger._census`` is invoked EXACTLY twice with that
     same tuple and the original argument semantics of ``loaded_geometry_ledger`` / ``build_ledger``:
         census 1: _census(modules, objects, omit_cache_payloads=False)
         census 2: _census(modules, fresh, definitions=definitions, omit_cache_payloads=False)
     ``loaded_geometry_ledger`` / ``build_ledger`` are NOT called; nothing is validated, admitted, retried, reset,
     garbage-collected or widened; no bound is changed.

Observation method: ``sys.setprofile`` is installed ONLY around each ``_census`` call. The callback reacts to 'return'
events of exactly two code objects (identity comparison): the nested ``visit`` of ``_planning_purity.inventory`` (first
frame whose ``count > max_nodes`` or ``depth > max_depth``: the inventory charges BEFORE atom/alias detection, so this
is the crossing node) and ``inventory`` itself (final count/truncated/max_nodes/max_depth). It reads scalar frame
locals only (never repr/getattr of inspected instances; only ``type(value)`` names) and never wraps or replaces the
walker. Observer impact: the profile hook adds per-call overhead (so durations are NOT speed evidence and the walk is
slower than unobserved), it is per-thread, restored in ``finally``, and it is identical for baseline and candidate.

Usage (one process per source tree, thread variables exported by the parent first):
  python tools/r11_census_diagnostic.py --source-root C:/abs/checkout --output C:/abs/evidence/census-<label>.json
         [--bootstrap-test-file C:/abs/tests/test_r11_profile.py]
Exit 0 = a COMPLETE observation was written (it says nothing about acceptance); 1 = partial (stage/exception failure,
partial results saved); 2 = refused before any import (arguments/environment).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
SAMPLE = 10


def sha256_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        return 'unreadable: ' + type(error).__name__


def type_name(value):
    """Passive type label: type.__getattribute__ on the type object; no hook of the inspected instance is invoked."""
    kind = type(value)
    try:
        return '%s.%s' % (type.__getattribute__(kind, '__module__'), type.__getattribute__(kind, '__qualname__'))
    except Exception:
        return '?'


class Report:
    """Incrementally saved result (partial results survive any failure)."""
    def __init__(self, path):
        self.path = Path(path)
        self.data = dict(schema='R11-census-diagnostic-v1', observational_only=True, qualification='NOT qualified: observation only',
                         complete=False, stages=[], errors=[])

    def save(self):
        temporary = self.path.with_name(self.path.name + '.tmp')
        temporary.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=str), encoding='utf-8')
        temporary.replace(self.path)

    def stage(self, name, function):
        started = time.perf_counter()
        row = dict(name=name, status='running')
        self.data['stages'].append(row)
        try:
            value = function()
            row['status'] = 'ok'
            return value
        except BaseException as error:           # recorded; the caller decides whether later stages can run
            row['status'] = 'failed'
            row['error'] = '%s: %s' % (type(error).__name__, error)
            self.data['errors'].append(dict(stage=name, error=row['error']))
            if not isinstance(error, Exception):
                raise
            return None
        finally:
            row['seconds'] = time.perf_counter() - started
            self.save()


def roots_snapshot():
    """Default roots exactly as loaded_geometry_ledger selects them, in the same sorted order."""
    items = [(name, module) for name, module in sorted(sys.modules.items())
             if name == 'anygeometry' or name.startswith('anygeometry.')]
    return tuple(module for _, module in items), [dict(name=name, id=id(module), module_type=type_name(module))
                                                  for name, module in items]


def prim(value):
    """Primitive-only field: ints/floats/bools/None/str pass through; anything else is replaced by a type label."""
    return value if type(value) in (int, float, bool, str, type(None)) else '<' + type_name(value) + '>'


def prim_state(state):
    return {key: (prim(value) if type(value) is not dict else {k: prim(v) if type(v) is not list else v for k, v in value.items()})
            for key, value in state.items()}


def make_observer():
    """Build (observer, state) BEFORE the pair so no closure/state is allocated between the two walks.

    The hook reacts to 'return' events of exactly two code objects (identity compare): the nested ``visit`` of
    ``_planning_purity.inventory`` and ``inventory`` itself; it reads scalar frame locals and, for the crossing node,
    only ``type(value)`` labels."""
    from anygeometry._planning_purity import inventory
    visit_code = next((c for c in inventory.__code__.co_consts if type(c) is types.CodeType and c.co_name == 'visit'), None)
    inventory_code = inventory.__code__
    state = dict(first_crossing=None, inventory=None, observer_errors=0, visit_code_found=visit_code is not None,
                 seconds_observed=None)

    def observer(frame, event, arg):
        if event != 'return':
            return
        code = frame.f_code
        if code is visit_code:
            if state['first_crossing'] is None:
                try:
                    local = frame.f_locals
                    count, depth, max_nodes, max_depth = local['count'], local['depth'], local['max_nodes'], local['max_depth']
                    if count > max_nodes or depth > max_depth:
                        state['first_crossing'] = dict(path=prim(local['path']), depth=prim(depth), count=prim(count),
                            max_nodes=prim(max_nodes), max_depth=prim(max_depth),
                            crossed=[name for name, hit in (('nodes', count > max_nodes), ('depth', depth > max_depth)) if hit],
                            value_type=type_name(local['value']))
                except Exception:
                    state['observer_errors'] += 1
        elif code is inventory_code:
            try:
                local = frame.f_locals
                state['inventory'] = dict(count=prim(local['count']), truncated=prim(local['truncated']),
                                          max_nodes=prim(local['max_nodes']), max_depth=prim(local['max_depth']))
            except Exception:
                state['observer_errors'] += 1
    return observer, state


def call_observed(census, observer, state, modules, *args, **kwargs):
    """Run one ``_census`` call under the prebuilt profile hook (hook restored in ``finally``)."""
    previous = sys.getprofile()
    started = time.perf_counter()
    sys.setprofile(observer)
    try:
        result = census(modules, *args, **kwargs)
    finally:
        sys.setprofile(previous)
    state['seconds_observed'] = time.perf_counter() - started
    return result


def store_rows(objects):
    return {path: dict(type=type_name(value), id=id(value)) for path, value in objects.items()}


def compare(objects, fresh):
    """original-only / fresh-only paths, identity changes on shared paths (``is``), same-object relocations."""
    original_only = sorted(set(objects) - set(fresh))
    fresh_only = sorted(set(fresh) - set(objects))
    changed = sorted(path for path in set(objects) & set(fresh) if fresh[path] is not objects[path])
    identity_changes = [dict(path=path, original=dict(type=type_name(objects[path]), id=id(objects[path])),
                             fresh=dict(type=type_name(fresh[path]), id=id(fresh[path]))) for path in changed]
    by_id_original, by_id_fresh = {}, {}
    for path, value in objects.items():
        by_id_original.setdefault(id(value), []).append(path)
    for path, value in fresh.items():
        by_id_fresh.setdefault(id(value), []).append(path)
    relocations = [dict(id=identity, original_paths=sorted(by_id_original[identity]), fresh_paths=sorted(by_id_fresh[identity]))
                   for identity in sorted(set(by_id_original) & set(by_id_fresh))
                   if set(by_id_original[identity]) != set(by_id_fresh[identity])]
    return dict(original_only=original_only, fresh_only=fresh_only, identity_changes=identity_changes,
                same_object_relocations=relocations, original_count=len(objects), fresh_count=len(fresh),
                guard_would_fire=bool(original_only or fresh_only or identity_changes))


def report_summary(report, observation, objects, seconds):
    return dict(truncated=report.truncated, store_count=len(report.stores), unknown_count=len(report.unknowns),
                alias_count=len(report.aliases), handle_count=len(objects),
                handles_cover_stores=set(objects) == {store.path for store in report.stores},
                inventory_return=observation['inventory'], first_bound_crossing=observation['first_crossing'],
                observer_errors=observation['observer_errors'], visit_code_found=observation['visit_code_found'],
                census_seconds_observed=seconds,
                stores=[dict(path=s.path, kind=s.kind, decision=s.decision, reason=s.reason) for s in report.stores],
                unknowns=[list(u) for u in report.unknowns], aliases=[list(a) for a in report.aliases])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-test-file')
    args = parser.parse_args(argv)
    source_root, output = Path(args.source_root), Path(args.output)
    if not source_root.is_absolute() or not (source_root / 'src' / 'anygeometry' / '__init__.py').is_file():
        print('refused: --source-root must be an absolute checkout containing src/anygeometry'); return 2
    if not output.is_absolute() or str(output.resolve()).lower().startswith(str(source_root.resolve()).lower() + os.sep):
        print('refused: --output must be an absolute path OUTSIDE the source root'); return 2
    if args.bootstrap_test_file and not Path(args.bootstrap_test_file).is_absolute():
        print('refused: --bootstrap-test-file must be absolute'); return 2
    wrong = {name: os.environ.get(name) for name in THREAD_VARIABLES if os.environ.get(name) != '1'}
    if wrong:
        print('refused: thread variables must be "1" before any import: %s' % wrong); return 2
    if 'anygeometry' in sys.modules:
        print('refused: anygeometry already imported in this process'); return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    source_src = str((source_root / 'src').resolve())
    sys.path.insert(0, source_src)
    tests_dir = str((Path(args.bootstrap_test_file).parent if args.bootstrap_test_file else source_root / 'tests').resolve())
    if tests_dir not in sys.path:
        sys.path.insert(1, tests_dir)
    report = Report(output)
    data = report.data
    data.update(source_root=str(source_root), argv=dict(source_root=args.source_root, output=args.output,
                bootstrap_test_file=args.bootstrap_test_file), thread_variables={n: os.environ[n] for n in THREAD_VARIABLES},
                python=sys.version, platform=sys.platform)
    ledger_state = {}

    def ambient():
        import pytest                                              # noqa: F401  (ambient import of the failed test)
        if args.bootstrap_test_file:                               # import-only: functions are defined, never executed
            spec = importlib.util.spec_from_file_location('r11_census_bootstrap', args.bootstrap_test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            from _planning_numpy_profile_contract import numpy_profile_test   # noqa: F401
        data['pytest_version'] = getattr(pytest, '__version__', None)

    def import_session():
        import anygeometry
        from anygeometry import _planning_session, _planning_ledger, _planning_purity, _planning_qualification  # noqa: F401
        origin = Path(anygeometry.__file__).resolve()
        if Path(source_src) not in origin.parents:
            raise RuntimeError('anygeometry imported from %s, not from %s' % (origin, source_src))
        data['anygeometry_origin'] = str(origin)
        data['hashes'] = {name: sha256_file(path) for name, path in (
            ('tool', __file__), ('bootstrap_test_file', args.bootstrap_test_file or ''),
            ('numpy_profile_helper', Path(tests_dir) / '_planning_numpy_profile_contract.py'),
            ('_planning_ledger', Path(source_src) / 'anygeometry' / '_planning_ledger.py'),
            ('_planning_purity', Path(source_src) / 'anygeometry' / '_planning_purity.py'),
            ('_planning_child', Path(source_src) / 'anygeometry' / '_planning_child.py'),
            ('_planning_session', Path(source_src) / 'anygeometry' / '_planning_session.py'))}
        import numpy, scipy, threadpoolctl
        data['versions'] = dict(numpy=numpy.__version__, scipy=scipy.__version__, threadpoolctl=threadpoolctl.__version__)
        modules, roots = roots_snapshot()
        data['roots_after_import'] = roots

    def prepare_and_observe():
        """ONE threadpool_limits(1) context spans the real prepare AND both censuses (as in the failed test). Nothing
        is summarized, compared, snapshotted-between or saved between the two walks; every full structure is built
        after both calls. Automatic GC is neither forced nor suppressed."""
        import runpy
        from threadpoolctl import threadpool_info, threadpool_limits
        from anygeometry import _planning_ledger as ledger_api
        if 'anygeometry._planning_child' in sys.modules:
            raise RuntimeError('anygeometry._planning_child already imported as a package module')
        child_file = Path(source_src) / 'anygeometry' / '_planning_child.py'
        started = time.perf_counter()
        namespace = runpy.run_path(str(child_file), run_name='r11_legacy_preparation')
        data['child_load_seconds'] = time.perf_counter() - started
        observer1, state1 = make_observer()          # observers are built BEFORE the pair so nothing is allocated between walks
        observer2, state2 = make_observer()
        first = second = omitted_first = omitted_second = None
        with threadpool_limits(limits=1):
            data['threadpool_effective'] = [dict(internal_api=prim(p.get('internal_api')), num_threads=prim(p.get('num_threads')),
                                                 filepath=prim(p.get('filepath')), version=prim(p.get('version')))
                                            for p in threadpool_info()]
            started = time.perf_counter()
            namespace['prepare'](production=False)
            data['prepare_seconds'] = time.perf_counter() - started
            modules, roots_before = roots_snapshot()   # the ONE input tuple, taken right after prepare (as loaded_geometry_ledger does)
            data['sys_modules_count'] = len(sys.modules)
            data['child_imported_as_package'] = 'anygeometry._planning_child' in sys.modules
            data['run_name_in_sys_modules'] = 'r11_legacy_preparation' in sys.modules
            data['census_input_modules'] = roots_before
            objects, fresh, definitions = {}, {}, {}
            try:
                (first, omitted_first) = call_observed(ledger_api._census, observer1, state1, modules, objects, omit_cache_payloads=False)
                (second, omitted_second) = call_observed(ledger_api._census, observer2, state2, modules, fresh,
                                                         definitions=definitions, omit_cache_payloads=False)
            except Exception as error:                 # keep whatever primitive observations exist, then fail the stage
                data['pair_failure'] = dict(error='%s: %s' % (type(error).__name__, error), census_1_state=prim_state(state1),
                                            census_2_state=prim_state(state2), census_1_finished=first is not None)
                raise
            _, roots_after = roots_snapshot()            # after the pair, never between the walks
        data['roots_changed_over_pair'] = roots_before != roots_after
        data['roots_after_pair'] = roots_after
        # ---- everything below runs AFTER both censuses -------------------------------------------------------
        data['census_1'] = report_summary(first, state1, objects, state1['seconds_observed'])
        data['census_1']['omitted'] = sorted(omitted_first)
        data['stores_1'] = store_rows(objects)
        data['census_2'] = report_summary(second, state2, fresh, state2['seconds_observed'])
        data['census_2']['omitted'] = sorted(omitted_second)
        data['stores_2'] = store_rows(fresh)
        data['definitions_2'] = {str(identity): sorted(paths) for identity, paths in definitions.items()}
        data['comparison'] = compare(objects, fresh)
        data['combined_guard_terms'] = dict(
            census_2_truncated=bool(second.truncated), handle_set_differs=set(fresh) != set(objects),
            identity_differs=any(fresh[p] is not v for p, v in objects.items() if p in fresh),
            census_1_truncated=bool(first.truncated))
        data['combined_guard_would_fire'] = bool(second.truncated or data['combined_guard_terms']['handle_set_differs']
                                                 or data['combined_guard_terms']['identity_differs'])

    if report.stage('ambient_imports', ambient) is None and report.data['stages'][-1]['status'] == 'failed':
        report.save(); return 1
    report.stage('import_session', import_session)
    if report.data['stages'][-1]['status'] != 'ok':
        return 1
    report.stage('prepare_and_observe_two_censuses', prepare_and_observe)
    data['complete'] = report.data['stages'][-1]['status'] == 'ok' and not data['errors']
    report.save()
    one = data.get('census_1', {})
    two = data.get('census_2', {})
    print(json.dumps(dict(complete=data['complete'], output=str(output), qualification=data['qualification'],
        census_1=dict(truncated=one.get('truncated'), count=(one.get('inventory_return') or {}).get('count'), stores=one.get('store_count')),
        census_2=dict(truncated=two.get('truncated'), count=(two.get('inventory_return') or {}).get('count'), stores=two.get('store_count')),
        first_bound_crossing=[one.get('first_bound_crossing'), two.get('first_bound_crossing')],
        combined_guard_would_fire=data.get('combined_guard_would_fire'),
        original_only=len(data.get('comparison', {}).get('original_only', [])),
        fresh_only=len(data.get('comparison', {}).get('fresh_only', [])),
        identity_changes=len(data.get('comparison', {}).get('identity_changes', [])),
        sample=dict(original_only=data.get('comparison', {}).get('original_only', [])[:SAMPLE],
                    fresh_only=data.get('comparison', {}).get('fresh_only', [])[:SAMPLE]))))
    return 0 if data['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
