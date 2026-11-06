"""R11 subset PREPARATION-TO-ACTUAL-MESHER-ENTRY harness (measurement tool; never meshes, never plans the full model).

What it times: the REAL ``anyfem.model.project.Project.generate_mesh`` of a NEW Project on the full fixture document,
called with the explicit historical settings (target_size=.25, strategy='quad_first', order='linear',
layout_policy='adaptive', quad_t3_minimum_angle=15, quad_t3_angle_action='error'), with only these observation/restriction
hooks, installed BEFORE the Project is constructed (aliases refuse/stop while ``armed`` is False) and restored in ``finally``:
  * ``project.extract_model_closure`` -> a real-closure wrapper that calls the ORIGINAL function with ONLY the selected face
    handles and ``include_structural_closure=False`` (the historical subset profile; the fixture has 0 sheets/members) and
    asserts the exact SOURCE face keys and working face count BEFORE preparation (no whole-model fallback; working ids are
    not assumed to equal source ids);
  * the cached ``prepare_structural_connectivity`` aliases of ``anyfem.model.project`` and ``anyfem.native_meshing_backend``
    -> a recorder that captures the exact initial working document (UUID included, serialization cost INSIDE the timer and
    reported) and then calls the ORIGINAL (all real planning/apply/report/lineage/clone work is preserved);
  * the cached ``generate_hybrid_mesh_result`` aliases of ``anymesher.hybrid`` and ``anyfem.native_meshing_backend``
    -> a stop whose FIRST instruction takes the mesher-entry timestamp; it then records the geometry and forwarded kwargs and
    raises ``StopStage`` (a BaseException). Violations (geometry is not the prepared working model, settings differ from
    target_size .25 / order linear / strategy native / layout adaptive, face_ids != ALL prepared working faces) are recorded
    and refused by the harness AFTER the stop: nothing is ever meshed.
No force/private parallel route is used: the public setting is '0' (serial process) or absent (default process).
Cold then warm rows run in ONE process, each with a NEW Project and closure and an initially empty
``_mesh_preparation_cache``. Heavy diagnostic export happens only after BOTH timed rows (and after the default-process
identity controls). Targets: prep_entry -> mesher_entry (primary), generate_mesh call -> mesher_entry (includes closure).
The tool produces raw per-row data and an identity verdict; the parent adjudicates speed.

Default process only, after both timed rows: each initial working document is replayed through the ORIGINAL
``prepare_structural_connectivity`` twice (public setting '0' and '1') with the same kwargs; each control must observe
EXACTLY ONE plan like the timed rows; plan/application/report/deterministic diagnostics must equal the timed row, and the
setting-1 / timed-absent routing receipts must be identical genuine serial-retired small routes with no children.
Cold and warm are NOT compared with each other (their working documents carry distinct UUIDs by construction).
Runtime records (timing/execution data) are retained in raw artifacts and are NOT part of the deterministic comparison.

Usage:
  python tools/r11_subset_prepare.py --fixture-json F --case elbow_r7 --mode serial|default --output DIR [--model-sha256 H]
  python tools/r11_subset_prepare.py --canary3 --mode default --output DIR        # tiny 3-face dry check (no fixture)
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import struct
import sys
import time
import uuid
from pathlib import Path

ENV = 'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING'
EXPECTED_COUNTS = {'elbow_sector60': 97, 'elbow_r7': 240, 'canary3': 3}
FULL_FIXTURE_FACES = 1083
DIAGNOSTIC_PREFIXES = ('batch_face_pairs_', 'intersection_pair_', 't2b_', 'planning_mode:', 'ring_split_',
                       'arrangement_snap_', 'arrangement_seam_')
GENERATE_ARGS = dict(target_size=0.25, strategy='quad_first', order='linear', layout_policy='adaptive',
                     quad_t3_minimum_angle=15, quad_t3_angle_action='error')
EXPECTED_FORWARDED = dict(target_size=0.25, order='linear', strategy='native', layout_policy='adaptive')
WALK_LIMIT = 100000
BYTE_LIMIT = 64 * 2 ** 20


class StopStage(BaseException):
    """Raised at the mesher entry; deliberately NOT an Exception."""


class HarnessError(Exception):
    """Fail-closed guard violation (subset scope, alias identity, ordering, evidence grammar)."""


# ------------------------------------------------------------------------------------------------ fixture / subset
def select_case(data, case, *, total_faces=None, sheets=0, members=0):
    """Validate the fixed case id, exact count, ids and uniqueness; never the full model."""
    if case not in EXPECTED_COUNTS:
        raise HarnessError('unknown case %r (allowed: %s)' % (case, sorted(EXPECTED_COUNTS)))
    subsets = data.get('subsets') if type(data) is dict else None
    ids = subsets.get(case) if type(subsets) is dict else None
    if type(ids) is not list or any(type(i) is not int or i < 1 for i in ids):
        raise HarnessError('fixture has no valid integer face-id list for %s' % case)
    if len(ids) != EXPECTED_COUNTS[case] or len(set(ids)) != len(ids):
        raise HarnessError('%s requires exactly %d unique face ids, got %d (%d unique)'
                           % (case, EXPECTED_COUNTS[case], len(ids), len(set(ids))))
    if total_faces is not None:
        if not set(ids) <= set(total_faces):
            raise HarnessError('%s selects face ids absent from the fixture document' % case)
        if len(ids) >= len(total_faces) and case != 'canary3':      # the explicit tiny canary IS a 3-face model
            raise HarnessError('refusing a selection that is the whole model')
        if case != 'canary3' and (len(total_faces) != FULL_FIXTURE_FACES or sheets or members):
            raise HarnessError('fixture is not the 1083-face / 0 sheets / 0 members source (%d faces, %d sheets, %d members)'
                               % (len(total_faces), sheets, members))
    return tuple(sorted(ids))


# ------------------------------------------------------------------------------------------------------- hooks
def _plain(value):
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if type(value) in (tuple, list) and all(type(v) is int for v in value):
        return list(value)
    return '<' + type(value).__qualname__ + '>'


class Recorder:
    def __init__(self, selected):
        self.selected = tuple(selected)
        self.clock = time.perf_counter
        self.events = []
        self.armed = False
        self.closure_calls = self.prep_calls = self.mesher_calls = 0
        self.closure = self.working = self.report = self.working_doc = None
        self.prep_kwargs = None
        self.evidence = self.counts = self.records = None
        self.violations = []
        self.forwarded = None
        self.t = {}
        self.capture_seconds = None
        self.working_face_ids = None

    def mark(self, name):
        self.t[name] = self.clock()
        self.events.append(name)

    def inspect_mesher(self, args, kwargs):
        """Record the geometry and forwarded settings; NEVER raises (violations are refused after the stop)."""
        try:
            geometry = args[0] if args else kwargs.get('geometry', kwargs.get('model'))
            if not self.armed:
                self.violations.append('mesher reached during Project construction')
            if self.working is None or geometry is not self.working:
                self.violations.append('forwarded geometry is not the recorded prepared working model')
            else:
                forwarded_ids = kwargs.get('face_ids')
                if forwarded_ids is None or sorted(forwarded_ids) != sorted(geometry.faces):
                    self.violations.append('forwarded face_ids are not exactly all prepared working faces')
            for key, expected in EXPECTED_FORWARDED.items():
                if key not in kwargs or kwargs[key] != expected or type(kwargs[key]) is not type(expected):
                    self.violations.append('forwarded %s=%r != %r' % (key, _plain(kwargs.get(key)), expected))
            self.forwarded = {key: _plain(value) for key, value in kwargs.items()}
        except Exception as error:
            self.violations.append('mesher inspection failed: %s' % type(error).__name__)


class Hooks:
    """Install/restore the closure wrapper, both prepare aliases and both mesher aliases on injected namespaces."""
    def __init__(self, recorder, *, project, backend, preparation, hybrid, original_closure, to_dict, collectors):
        self.r, self.project, self.backend, self.preparation, self.hybrid = recorder, project, backend, preparation, hybrid
        self.original_closure, self.to_dict, self.collectors = original_closure, to_dict, collectors
        self.saved = []

    def _set(self, namespace, name, value):
        self.saved.append((namespace, name, getattr(namespace, name)))
        setattr(namespace, name, value)

    def __enter__(self):
        prepare = self.preparation.prepare_structural_connectivity
        result = self.hybrid.generate_hybrid_mesh_result
        if not (self.project.prepare_structural_connectivity is prepare is self.backend.prepare_structural_connectivity):
            raise HarnessError('cached prepare aliases do not initially match the original')
        if self.backend.generate_hybrid_mesh_result is not result:
            raise HarnessError('cached mesher aliases do not initially match anymesher.hybrid')
        if self.project.extract_model_closure is not self.original_closure:
            raise HarnessError('project closure alias is not the original extract_model_closure')
        try:
            recorder = self.r

            def closure_wrapper(geometry, handles, **kwargs):
                if not recorder.armed:
                    raise HarnessError('closure alias invoked during Project construction')
                recorder.mark('closure_entry')
                recorder.closure_calls += 1
                if recorder.closure_calls != 1:
                    raise HarnessError('closure extracted more than once')
                if set(kwargs) != {'include_structural_closure', 'include_features'}:
                    raise HarnessError('unexpected closure arguments %s' % sorted(kwargs))
                subset = tuple(geometry.handle('face', i) for i in recorder.selected)
                closure = self.original_closure(geometry, subset, include_structural_closure=False, include_features=False)
                work = closure.working_model
                # SOURCE keys and the working COUNT: working ids need not equal source ids (gapped sector ids).
                mapped = sorted(h.id for h in closure.source_to_work if h.kind == 'face')
                if mapped != list(recorder.selected) or len(work.faces) != len(recorder.selected):
                    raise HarnessError('working closure is not exactly the selected %d faces (mapped %d, working %d)'
                                       % (len(recorder.selected), len(mapped), len(work.faces)))
                recorder.closure, recorder.working = closure, work
                recorder.working_face_ids = sorted(work.faces)
                recorder.mark('closure_exit')
                return closure

            def prepare_wrapper(working, **kwargs):
                if not recorder.armed:
                    raise HarnessError('preparation alias invoked during Project construction')
                recorder.mark('prep_entry')
                recorder.prep_calls += 1
                if recorder.prep_calls != 1 or working is not recorder.working or len(working.faces) != len(recorder.selected):
                    raise HarnessError('preparation did not receive the exact selected working closure once')
                started = recorder.clock()
                recorder.working_doc = self.to_dict(working)         # exact initial document, UUID included, inside the timer
                recorder.capture_seconds = recorder.clock() - started
                recorder.prep_kwargs = {k: v for k, v in kwargs.items() if k != 'cancellation_check'}
                recorder.cancellation_given = kwargs.get('cancellation_check') is not None
                with self.collectors() as observed:
                    report = prepare(working, **kwargs)
                recorder.evidence, recorder.counts, recorder.records = observed
                recorder.report = report
                recorder.mark('prep_exit')
                return report

            def stop(*args, **kwargs):
                recorder.t['mesher_entry'] = recorder.clock()        # FIRST instruction
                recorder.events.append('mesher_entry')
                recorder.mesher_calls += 1
                recorder.inspect_mesher(args, kwargs)
                raise StopStage()

            self._set(self.project, 'extract_model_closure', closure_wrapper)
            self._set(self.project, 'prepare_structural_connectivity', prepare_wrapper)
            self._set(self.backend, 'prepare_structural_connectivity', prepare_wrapper)
            self._set(self.hybrid, 'generate_hybrid_mesh_result', stop)
            self._set(self.backend, 'generate_hybrid_mesh_result', stop)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        while self.saved:
            namespace, name, value = self.saved.pop()
            setattr(namespace, name, value)
        return False


# ------------------------------------------------------------------------------------------------------- one row
def run_row(label, make_project, hooks_factory, selected):
    """One timed row. Hooks are installed BEFORE the Project is constructed; the generate_mesh timer starts after it."""
    recorder = Recorder(selected)
    with hooks_factory(recorder):
        try:
            project = make_project()
        except StopStage as error:
            raise HarnessError('%s: the Project constructor reached the mesher stop' % label) from error
        if recorder.events or recorder.closure_calls or recorder.prep_calls or recorder.mesher_calls:
            raise HarnessError('%s: the Project constructor invoked a hooked alias' % label)
        if getattr(project, '_mesh_preparation_cache', None) is not None:
            raise HarnessError('%s: _mesh_preparation_cache is not initially None' % label)
        recorder.armed = True
        recorder.mark('generate_mesh_call')
        try:
            project.generate_mesh(**GENERATE_ARGS)
        except StopStage:
            pass
        else:
            raise HarnessError('%s: generate_mesh returned without reaching the mesher stop' % label)
    t = recorder.t
    if recorder.violations:
        raise HarnessError('%s: mesher-entry contract violated: %s' % (label, '; '.join(recorder.violations)))
    if recorder.mesher_calls != 1 or recorder.prep_calls != 1 or recorder.closure_calls != 1:
        raise HarnessError('%s: expected 1 closure, 1 preparation and 1 mesher entry, saw %d/%d/%d'
                           % (label, recorder.closure_calls, recorder.prep_calls, recorder.mesher_calls))
    if recorder.events != ['generate_mesh_call', 'closure_entry', 'closure_exit', 'prep_entry', 'prep_exit', 'mesher_entry']:
        raise HarnessError('%s: unexpected event order %s' % (label, recorder.events))
    plans = list(recorder.evidence or [])
    if len(plans) != 1:
        raise HarnessError('%s: expected exactly one observed plan, saw %d' % (label, len(plans)))
    if getattr(project, '_mesh_preparation_cache', None) is not None:
        raise HarnessError('%s: the cache was populated although no mesh was generated' % label)
    row = dict(label=label, selected_faces=len(selected), working_faces=len(recorder.working_face_ids),
               events=list(recorder.events), generate_mesh_arguments=dict(GENERATE_ARGS),
               forwarded_settings=recorder.forwarded,
               forwarded_face_ids=recorder.forwarded.get('face_ids') if recorder.forwarded else None,
               prep_entry_to_mesher_entry_seconds=t['mesher_entry'] - t['prep_entry'],
               generate_call_to_mesher_entry_seconds=t['mesher_entry'] - t['generate_mesh_call'],
               closure_seconds=t['closure_exit'] - t['closure_entry'],
               pre_closure_seconds=t['closure_entry'] - t['generate_mesh_call'],
               closure_to_prep_seconds=t['prep_entry'] - t['closure_exit'],
               prep_seconds=t['prep_exit'] - t['prep_entry'],
               post_prep_to_mesher_seconds=t['mesher_entry'] - t['prep_exit'],
               initial_document_capture_seconds_inside_timer=recorder.capture_seconds,
               plan_checksum=plans[0].get('plan_checksum'),
               preparation_calls=recorder.prep_calls, plans_observed=len(plans), meshing_calls=0)
    return row, recorder


# --------------------------------------------------------------------------------- exact typed export / comparison
def registered_owner_types(classes, prefix='anyfem.'):
    """Validate the owner dataclass types the offline report exporter may encode: module prefix, actual bound
    module+qualname identity and declared dataclass fields."""
    registered = {}
    for cls in classes:
        owner = sys.modules.get(cls.__module__)
        bound = owner
        for part in cls.__qualname__.split('.'):
            bound = None if bound is None else vars(bound).get(part)
        if not cls.__module__.startswith(prefix) or bound is not cls or '__dataclass_fields__' not in vars(cls):
            raise HarnessError('unregistered owner report type %s.%s' % (cls.__module__, cls.__qualname__))
        registered[cls] = (cls.__module__, cls.__qualname__)
    return registered


def owner_record(value, registered):
    """OFFLINE exact typed evidence of ANYfem report objects (the production exact_numerical_record is unchanged and
    rejects them). Every declared dataclass field in order; ints/strs/bools/None, float64 BITS (signed zeros kept),
    UUIDs, tuples/lists (type kept), sets, dicts with typed keys; unknown, unregistered or cyclic values refuse.
    Bounded: <=100000 values, <=64 MiB JSON."""
    count = 0
    active = set()

    def encode(record):
        return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)

    def rec(item):
        nonlocal count
        count += 1
        if count > WALK_LIMIT:
            raise HarnessError('exact owner evidence value bound')
        typ = type(item)
        if item is None:
            return ['none']
        if typ in (bool, int, str):
            return [typ.__name__, item]
        if typ is float:
            return ['float64', struct.pack('>d', item).hex()]
        if typ is bytes:
            return ['bytes', item.hex()]
        if typ is uuid.UUID:
            return ['uuid', item.hex]
        identity = id(item)
        if identity in active:
            raise HarnessError('cyclic exact owner evidence')
        active.add(identity)
        try:
            if typ in (tuple, list):
                return [typ.__name__, [rec(part) for part in item]]
            if typ in (set, frozenset):
                return [typ.__name__, sorted((rec(part) for part in item), key=encode)]
            if typ is dict:
                return ['dict', sorted(([rec(key), rec(part)] for key, part in item.items()), key=encode)]
            if typ in registered:
                module, name = registered[typ]
                return ['record', module, name,
                        [[field.name, rec(object.__getattribute__(item, field.name))] for field in dataclasses.fields(typ)]]
            raise HarnessError('unsupported or unregistered exact owner evidence type %s.%s' % (typ.__module__, typ.__qualname__))
        finally:
            active.remove(identity)
    text = encode(rec(value))
    if len(text) > BYTE_LIMIT:
        raise HarnessError('exact owner evidence byte bound')
    return text


def exact_plan_document(plan, exact_numerical_record):
    """Chunked typed grammar ported from the historical gate_stage.exact_plan_document (redesign/gate_stage.py lines
    223-281) INCLUDING its registered-type guard for the plan and its policy: every declared plan field in order,
    policy.cancellation_check the sole excluded execution field; each sequence element encoded independently
    (<=100000 components, <=64 MiB ASCII JSON). Production limits are NOT changed."""
    from dataclasses import fields

    def registered_type(value):
        typ = type(value)
        module, name = typ.__module__, typ.__qualname__
        owner = sys.modules.get(module)
        bound = owner
        for part in name.split('.'):
            bound = None if bound is None else vars(bound).get(part)
        if (not module.startswith('anygeometry.') or bound is not typ or '__dataclass_fields__' not in vars(typ)):
            raise ValueError('unregistered qualification plan record')
        return module, name
    plan_type = registered_type(plan)
    values = []
    for field in fields(type(plan)):
        value = object.__getattribute__(plan, field.name)
        if field.name == 'policy':
            policy_type = registered_type(value)
            value = (policy_type, tuple((item.name, object.__getattribute__(value, item.name))
                                        for item in fields(type(value)) if item.name != 'cancellation_check'))
        values.append((field.name, value))
    chunks, size, components = [], 0, len(values)

    def emit(text):
        nonlocal size
        size += len(text)
        if size > BYTE_LIMIT:
            raise ValueError('exact plan artifact byte bound')
        chunks.append(text)

    def leaf(value):
        emit(json.dumps(exact_numerical_record(value), sort_keys=True, separators=(',', ':'), allow_nan=False))

    def field_value(value):
        nonlocal components
        if type(value) not in (tuple, list):
            leaf(value)
            return
        components += len(value)
        if components > WALK_LIMIT:
            raise ValueError('exact plan artifact component bound')
        emit('["' + type(value).__name__ + '",[')
        for index, part in enumerate(value):
            if index:
                emit(',')
            leaf(part)
        emit(']]')
    if components > WALK_LIMIT:
        raise ValueError('exact plan artifact component bound')
    emit('["tuple",[')
    leaf(plan_type)
    emit(',["tuple",[')
    for index, (name, value) in enumerate(values):
        if index:
            emit(',')
        emit('["tuple",[')
        leaf(name)
        emit(',')
        field_value(value)
        emit(']]')
    emit(']]]]')
    return ''.join(chunks)


def deterministic_diagnostics(counts):
    """Same declared outcome families as r11_profile / the historical gate; wall-time counters excluded and recorded."""
    kept = {k: v for k, v in dict(counts).items() if k.startswith(DIAGNOSTIC_PREFIXES) and not k.endswith('_time_ns')}
    return kept, sorted(k for k in dict(counts) if k not in kept)


def diagnostic_identity(counts):
    """Serialized (type- and sign-preserving) identity of the deterministic outcome counters."""
    return json.dumps(deterministic_diagnostics(counts)[0], sort_keys=True, separators=(',', ':'), allow_nan=False)


def export_bundle(recorder, to_dict, exact_numerical_record, registered, directory, label):
    """Plan/application/report/diagnostics of one preparation (deterministic comparison set) plus the retained raw
    runtime records (timing/execution data, NOT deterministic): built and written ONLY after the timed rows."""
    directory = Path(directory)
    plans = list(recorder.evidence or [])
    if len(plans) != 1:
        raise HarnessError('%s: expected exactly one observed plan, saw %d' % (label, len(plans)))
    # The owner's public observable is REQUIRED: a callable report.to_dict whose successful result serializes as strict
    # JSON. Failures are never turned into comparable strings (equal errors must not pass identity).
    report_to_dict = getattr(recorder.report, 'to_dict', None)
    if not callable(report_to_dict):
        raise HarnessError('%s: owner report has no callable to_dict (a public observable is required)' % label)
    try:
        owner_observable = json.dumps(report_to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
    except Exception as error:
        raise HarnessError('%s: owner report.to_dict failed or is not strictly JSON serializable: %s: %s'
                           % (label, type(error).__name__, error)) from error
    initial_text = None
    if getattr(recorder, 'working_doc', None) is not None:      # timed rows only; controls reference the row artifact
        try:
            initial_text = json.dumps(recorder.working_doc, sort_keys=True, separators=(',', ':'), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise HarnessError('%s: the captured initial working document is not strictly JSON serializable' % label) from error
    bundle = dict(plan=exact_plan_document(plans[0]['plan'], exact_numerical_record),
                  application=json.dumps(to_dict(recorder.working), sort_keys=True, separators=(',', ':'), allow_nan=False),
                  report=owner_record(recorder.report, registered),
                  report_to_dict=owner_observable,
                  diagnostics=diagnostic_identity(recorder.counts))
    digests = {}
    for name, text in bundle.items():
        (directory / ('%s-%s.json' % (label, name))).write_text(text, encoding='utf-8')
        digests[name] = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if initial_text is not None:
        # Exact captured initial working document (UUID included) survives process exit; it is evidence, NOT part of
        # the comparison bundle. Serialization happens here, after the timed rows, never inside a timer.
        (directory / ('%s-initial-working-document.json' % label)).write_text(initial_text, encoding='utf-8')
        digests['initial_working_document'] = hashlib.sha256(initial_text.encode('utf-8')).hexdigest()
    (directory / ('%s-raw-diagnostics.json' % label)).write_text(
        json.dumps(dict(counts=dict(recorder.counts), excluded_from_deterministic=deterministic_diagnostics(recorder.counts)[1]),
                   sort_keys=True, default=str), encoding='utf-8')
    (directory / ('%s-runtime-records.json' % label)).write_text(
        json.dumps(dict(classification='timing/execution data: retained raw, NOT part of the deterministic comparison',
                        records=recorder.records), sort_keys=True, default=str), encoding='utf-8')
    return bundle, digests


# --------------------------------------------------------------------------------------------- identity controls
@contextlib.contextmanager
def public_setting(value):
    """The public planning selector: '0'/'1' or absent (None). Restored exactly."""
    previous = os.environ.get(ENV)
    if value is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous


def run_control(setting, row_recorder, api):
    """Replay the row's exact initial working document through the ORIGINAL prepare (outside any timed window)."""
    working = api.from_dict(row_recorder.working_doc)
    closure = dataclasses.replace(row_recorder.closure, working_model=working)
    with public_setting(setting), api.collectors() as observed:
        report = api.prepare(working, cancellation_check=None, **row_recorder.prep_kwargs)
    evidence, counts, records = observed
    if len(evidence) != 1:
        raise HarnessError('control (setting %s) observed %d plans, expected exactly one' % (setting, len(evidence)))
    # Only the two exact Project-owner enrichments (project.generate_mesh, after preparation), using the captured
    # closure and the control working model:
    report.source_edge_paths = api.source_edge_paths(closure)
    report.source_to_working = api.source_work_mapping(closure, edge_paths=report.source_edge_paths)
    report.source_to_working = api.source_work_mapping(closure, edge_paths=report.source_edge_paths)
    control = Recorder(row_recorder.selected)
    control.evidence, control.counts, control.records, control.report, control.working = evidence, counts, records, report, working
    receipts = records.get('parallel_planning', []) if hasattr(records, 'get') else []
    return control, (receipts[-1] if receipts else None)


def routing_summary(receipt):
    if receipt is None:
        return None
    return dict(reason=receipt.get('reason'), routing=receipt.get('routing'), disposition=receipt.get('disposition'),
                retired=receipt.get('retired'), stages=receipt.get('stages'), forced=receipt.get('forced_selector'))


def genuine_serial_route(summary):
    return bool(summary and summary['disposition'] == 'serial' and summary['retired'] is True and summary['stages'] == []
                and summary['forced'] is None and str(summary['reason']).startswith('routed serial'))


# ------------------------------------------------------------------------------------------------------- session
def run_session(rows_spec, api, *, mode, directory, selected, make_project, hooks_factory):
    """Cold then warm timed rows, THEN exports, THEN (default mode) the unmeasured identity controls."""
    rows, recorders = [], []
    for label in ('cold', 'warm'):
        row, recorder = run_row(label, make_project, hooks_factory, selected)
        rows.append(row)
        recorders.append(recorder)
    result = dict(mode=mode, rows=rows, timed_preparations=sum(row['preparation_calls'] for row in rows),
                  controls=[], unmeasured_preparations=0,
                  cold_vs_warm='not compared: the cold and warm working documents carry distinct UUIDs by construction')
    exports = {}
    for row, recorder in zip(rows, recorders):                      # exports only after BOTH timed rows
        bundle, digests = api.export(recorder, directory, row['label'])
        exports[row['label']] = (bundle, digests)
        row['export_sha256'] = digests
        row['initial_working_document_path'] = str(Path(directory) / ('%s-initial-working-document.json' % row['label']))
    if mode == 'default':
        for row, recorder in zip(rows, recorders):
            controls = {}
            for setting in ('0', '1'):
                control, receipt = run_control(setting, recorder, api)
                result['unmeasured_preparations'] += 1
                bundle, digests = api.export(control, directory, '%s-control-%s' % (row['label'], setting))
                controls[setting] = dict(digests=digests, routing=routing_summary(receipt), plans_observed=len(control.evidence),
                                         equal_to_timed=bundle == exports[row['label']][0])
            # The timed row itself ran with the public setting ABSENT: its own receipt is the absent-route evidence
            # (2 timed + 4 unmeasured preparations in total, no further replay).
            receipts = (recorder.records or {}).get('parallel_planning', []) if hasattr(recorder.records or {}, 'get') else []
            absent_route = routing_summary(receipts[-1] if receipts else None)
            controls['absent_timed_routing'] = absent_route
            controls['initial_working_document'] = row['initial_working_document_path']   # the controls replay THIS exact document
            routes = [absent_route, controls['1']['routing']]
            controls['routing_identical_absent_vs_1'] = routes[0] == routes[1]
            controls['genuine_serial_retired_route'] = all(genuine_serial_route(r) for r in routes)
            controls['all_identical'] = (all(controls[k]['equal_to_timed'] for k in ('0', '1'))
                                         and controls['routing_identical_absent_vs_1'] and controls['genuine_serial_retired_route'])
            result['controls'].append(dict(row=row['label'], **controls))
        result['identity_verdict'] = all(c['all_identical'] for c in result['controls'])
    return result


# ----------------------------------------------------------------------------------------------------------- CLI
def source_hashes(paths):
    """sha256 of every watched source; an unreadable file REFUSES (never two equal 'unreadable' strings)."""
    digests = {}
    for path in paths:
        try:
            digests[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as error:
            raise HarnessError('unreadable source %s: %s' % (path, type(error).__name__)) from error
    return digests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--fixture-json')
    parser.add_argument('--model-sha256')
    parser.add_argument('--case', choices=('elbow_sector60', 'elbow_r7'))
    parser.add_argument('--canary3', action='store_true', help='explicit tiny 3-face dry check (no fixture file)')
    parser.add_argument('--mode', required=True, choices=('serial', 'default'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if bool(args.canary3) == bool(args.fixture_json):
        parser.error('exactly one of --fixture-json (+--case) or --canary3 is required')
    if args.fixture_json and not args.case:
        parser.error('--case is required with --fixture-json')
    if args.mode == 'serial':
        os.environ[ENV] = '0'
    else:
        os.environ.pop(ENV, None)                                   # absent: the default process
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from anygeometry.serialization import from_dict, to_dict
    from anygeometry._planning_code import exact_numerical_record
    from anygeometry.runtime_diagnostics import collect_runtime_diagnostics, collect_runtime_records
    from anyfem._preparation_context import _collect_preparation_evidence
    import anyfem.model.project as project_module
    import anyfem.native_meshing_backend as backend_module
    import anyfem.structural_preparation as preparation_module
    import anymesher.hybrid as hybrid_module
    from anyfem.model.project import Project
    if args.canary3:
        from anygeometry._planning_canaries import _fixture
        document = to_dict(_fixture()[0])
        case = 'canary3'
        fixture_sha = None
        data = dict(document=document, subsets=dict(canary3=sorted(from_dict(document).faces)[:3]))
    else:
        raw = Path(args.fixture_json).read_bytes()
        fixture_sha = hashlib.sha256(raw).hexdigest()
        if args.model_sha256 and args.model_sha256.lower() != fixture_sha:
            print('refused: fixture sha256 mismatch'); return 2
        data = json.loads(raw)
        document, case = data['document'], args.case
    probe = from_dict(document)
    selected = select_case(data, case, total_faces=sorted(probe.faces), sheets=len(probe.sheets), members=len(probe.members))
    del probe
    registered = registered_owner_types((preparation_module.StructuralPreparationReport, preparation_module.PreparedConnection))
    watched = [Path(project_module.__file__), Path(preparation_module.__file__), Path(backend_module.__file__),
               Path(hybrid_module.__file__), Path(__file__)]
    before = source_hashes(watched)                                 # unreadable input source => refuse

    def make_project():
        return Project(geometry=from_dict(document))

    @contextlib.contextmanager
    def collectors():
        with _collect_preparation_evidence() as evidence, collect_runtime_diagnostics() as counts, \
                collect_runtime_records() as records:
            yield (evidence, counts, records)

    original_closure = project_module.extract_model_closure

    def hooks_factory(recorder):
        return Hooks(recorder, project=project_module, backend=backend_module, preparation=preparation_module,
                     hybrid=hybrid_module, original_closure=original_closure, to_dict=to_dict, collectors=collectors)

    api = argparse.Namespace(
        from_dict=from_dict, collectors=collectors, prepare=preparation_module.prepare_structural_connectivity,
        source_edge_paths=preparation_module.source_edge_paths, source_work_mapping=preparation_module.source_work_mapping,
        export=lambda recorder, directory, label: export_bundle(recorder, to_dict, exact_numerical_record, registered, directory, label))
    summary = dict(schema='R11-subset-prepare-v2', case=case, mode=args.mode, selected_faces=len(selected),
                   fixture_sha256=fixture_sha, public_setting='0' if args.mode == 'serial' else 'absent', complete=False,
                   qualification='NOT an acceptance verdict: raw rows + identity only', before_hashes=before)
    path = output / ('r11-subset-prepare-%s-%s.json' % (case, args.mode))
    changed = True
    try:
        from threadpoolctl import threadpool_info
        import anygeometry, anyfem, anymesher, numpy
        summary.update(python=sys.version, origins=dict(anygeometry=str(Path(anygeometry.__file__)), anyfem=str(Path(anyfem.__file__)),
                       anymesher=str(Path(anymesher.__file__))), numpy=numpy.__version__,
                       threadpool=[dict(internal_api=p.get('internal_api'), num_threads=p.get('num_threads'),
                                        filepath=p.get('filepath')) for p in threadpool_info()])
        summary['result'] = run_session(None, api, mode=args.mode, directory=output, selected=selected,
                                        make_project=make_project, hooks_factory=hooks_factory)
        summary['complete'] = True
    except BaseException as error:
        summary['error'] = '%s: %s' % (type(error).__name__, error)
        if not isinstance(error, Exception):
            raise
    finally:
        try:
            summary['after_hashes'] = source_hashes(watched)
            changed = summary['after_hashes'] != before
        except HarnessError as error:                               # an unreadable source after the run counts as changed
            summary['after_hashes_error'] = str(error)
            changed = True
        summary['sources_changed_during_run'] = changed
        path.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str), encoding='utf-8')
    verdict = (summary.get('result') or {}).get('identity_verdict')
    print(json.dumps(dict(complete=summary['complete'], error=summary.get('error'), identity_verdict=verdict,
                          sources_changed_during_run=changed,
                          rows=[{k: r[k] for k in ('label', 'prep_entry_to_mesher_entry_seconds',
                                                   'generate_call_to_mesher_entry_seconds')}
                                for r in (summary.get('result') or {}).get('rows', [])], output=str(path))))
    return exit_code(summary['complete'], verdict, changed)


def exit_code(complete, verdict, changed):
    """Success only for a complete run whose identity is not False and whose watched sources did not change."""
    return 0 if complete and verdict is not False and not changed else 1


if __name__ == '__main__':
    sys.exit(main())
