"""Full owner-backed diagnosis, explicit coverage and lossless report export.

Diagnosis is information, never edit authority. The strict owner audit remains
intact in the report, including its classifications, policy and clean semantics.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import sys
from time import perf_counter
from threading import Event
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from .cleaner import CleanerOptions, CleanerReport, _polygon_data, _require_coordinate_contract, review_geometry
from .curves import Straight
from .entities import EntityRef
from .spatial import AABB, AABBTree
from .strict_audit import strict_audit


@dataclass(frozen=True)
class DiagnosisFinding:
    id: str
    kind: str
    refs: tuple[EntityRef, ...]
    message: str
    value: float | None = None
    repairable: bool = False
    severity: str = 'warning'
    measurements: tuple[tuple[str, float], ...] = ()
    tolerance: float | None = None
    action: str = 'Review the geometry and intended topology before editing.'
    owner_issue: dict | None = None

    def to_dict(self):
        return {'id': self.id, 'kind': self.kind, 'refs': [asdict(item) for item in self.refs],
                'message': self.message, 'value': self.value, 'repairable': self.repairable,
                'severity': self.severity, 'measurements': dict(self.measurements),
                'tolerance': self.tolerance, 'action': self.action,
                'owner_issue': deepcopy(self.owner_issue)}


@dataclass(frozen=True)
class DiagnosisCheck:
    name: str
    status: str
    detail: str
    candidates: int = 0


@dataclass(frozen=True)
class DiagnosisReport:
    model_id: str
    revision: int
    options: CleanerOptions
    findings: tuple[DiagnosisFinding, ...]
    checks: tuple[DiagnosisCheck, ...]
    coverage: tuple[str, ...]
    timings: tuple[tuple[str, float], ...]
    quality: CleanerReport | None = None
    audit: object | None = None
    source: dict | None = None

    @property
    def complete(self):
        return all(item.status == 'completed' for item in self.checks)

    @property
    def clean(self):
        return self.complete and self.audit_clean and not any(item.severity in ('warning', 'error', 'blocker') for item in self.findings)

    @property
    def audit_clean(self):
        return bool(self.audit is not None and self.audit.clean)

    @property
    def counts(self):
        return {'findings': len(self.findings), 'by_kind': dict(sorted(Counter(item.kind for item in self.findings).items())),
                'by_severity': dict(sorted(Counter(item.severity for item in self.findings).items())),
                'checks': len(self.checks), 'incomplete_checks': sum(item.status != 'completed' for item in self.checks),
                'candidates': sum(item.candidates for item in self.checks)}

    def to_dict(self):
        quality = None if self.quality is None else {
            'model_id': self.quality.model_id, 'revision': self.quality.revision,
            'options': asdict(self.quality.options), 'complete': self.quality.complete,
            'coverage': list(self.quality.coverage),
            'findings': [asdict(item) for item in self.quality.findings]}
        return {'schema': 'anygeometry.diagnosis.v1', 'model_id': self.model_id,
                'revision': self.revision, 'source': deepcopy(self.source), 'options': asdict(self.options),
                'complete': self.complete, 'clean': self.clean, 'audit_clean': self.audit_clean,
                'counts': self.counts, 'checks': [asdict(item) for item in self.checks],
                'coverage': list(self.coverage), 'timings_seconds': dict(self.timings),
                'findings': [item.to_dict() for item in self.findings], 'quality': quality,
                'owner_audit': None if self.audit is None else self.audit.to_dict()}

    def to_markdown(self):
        lines = ['# Geometry diagnosis', '', f'Model: {self.model_id}; revision: {self.revision}',
                 f'Complete: {self.complete}; clean: {self.clean}; owner audit clean: {self.audit_clean}',
                 f'Findings: {len(self.findings)}', '', '## Checks', '']
        lines.extend(f'- {check.name}: {check.status}; {check.detail}; candidates={check.candidates}' for check in self.checks)
        lines.extend(['', '## Coverage', ''])
        lines.extend('- ' + item for item in self.coverage)
        lines.extend(['', '## Source, options and counts', '', '```json',
                      json.dumps({'source': self.source, 'options': asdict(self.options), 'counts': self.counts}, sort_keys=True, indent=2),
                      '```', '', '## Findings', ''])
        for item in self.findings:
            refs = ', '.join(str(entity) for entity in item.refs) or 'model'
            lines.extend([f'### {item.severity.upper()} {item.kind}: {refs}', '',
                          item.message, f'ID: {item.id}', f'Action: {item.action}',
                          f'Measurements: {dict(item.measurements)}; tolerance: {item.tolerance}', ''])
            if item.owner_issue is not None:
                lines.extend(['Owner classification (lossless):', '```json',
                              json.dumps(item.owner_issue, sort_keys=True, ensure_ascii=False), '```', ''])
        provenance = None if self.audit is None else self.audit.to_dict()
        if provenance is not None:
            provenance.pop('issues', None)  # Every issue is printed losslessly above.
        lines.extend(['## Owner audit provenance', '', '```json',
                      json.dumps(provenance, sort_keys=True, ensure_ascii=False, indent=2), '```', ''])
        lines.extend(['## Timings (seconds)', '', json.dumps(dict(self.timings), sort_keys=True), ''])
        return '\n'.join(lines)


class _Cancelled(Exception):
    pass


def _parallel_candidate_model(model):
    return len(model.faces) >= 256 or len(model.edges) >= 2048


def _segment_gap(model, first, second):
    """Minimum of endpoint projections and a bounded interior stationary pair.

    For nonparallel straight segments an interior minimum solves the two
    orthogonality equations; otherwise the minimum is on a segment endpoint.
    Existing owner projection handles every boundary case, including parallels.
    """
    first_edge, second_edge = model.edges[first], model.edges[second]
    p0, p1 = model.vertex_position(first_edge.start), model.vertex_position(first_edge.end)
    q0, q1 = model.vertex_position(second_edge.start), model.vertex_position(second_edge.end)
    distances = [model.closest_edge_point(second, point)[2] for point in (p0, p1)]
    distances.extend(model.closest_edge_point(first, point)[2] for point in (q0, q1))
    first_length, second_length = math.dist(p0, p1), math.dist(q0, q1)
    if first_length <= 0 or second_length <= 0:
        raise ValueError('Degenerate straight segment has no supported near-gap metric.')
    u, v = (p1 - p0) / first_length, (q1 - q0) / second_length
    parameters, _, rank, _ = np.linalg.lstsq(np.column_stack((u, -v)), q0 - p0, rcond=None)
    if rank == 2 and 0 <= parameters[0] <= first_length and 0 <= parameters[1] <= second_length:
        distances.append(math.dist(p0 + parameters[0] * u, q0 + parameters[1] * v))
    result = min(distances)
    if not math.isfinite(result):
        raise ValueError('Nonfinite segment-gap measurement.')
    return result

def _gap_findings(model, threshold, checkpoint):
    """Spatially bounded point-edge and point-planar-face gap diagnostics."""
    supported = []
    unsupported = []
    incident_faces = {}
    for edge_id, edge in sorted(model.edges.items()):
        checkpoint()
        if isinstance(edge.curve, Straight):
            supported.append(('edge', edge_id))
        else:
            unsupported.append(EntityRef('edge', edge_id))
    for face_id, face in sorted(model.faces.items()):
        checkpoint()
        if _polygon_data(model, face) is not None:
            supported.append(('face', face_id))
            incident_faces[face_id] = {model.edges[item.edge].start for loop in (face.loop,) + face.holes for item in loop}
            incident_faces[face_id].update(model.edges[item.edge].end for loop in (face.loop,) + face.holes for item in loop)
        else:
            unsupported.append(EntityRef('face', face_id))
    items = []
    for key, bounds in zip(supported, model.entity_bounds_many(supported)):
        checkpoint()
        if bounds is None:
            unsupported.append(EntityRef(*key))
        else:
            items.append((key, AABB(tuple(bounds[:3]), tuple(bounds[3:]))))
    tree = AABBTree(items)
    findings = []
    candidates = 0
    for vertex_id in sorted(model.vertices):
        checkpoint()
        point = model.vertex_position(vertex_id)
        keys = tree.query(AABB.around_point(point).expanded(threshold)).keys
        for kind, identifier in keys:
            checkpoint()
            if kind == 'edge':
                edge = model.edges[identifier]
                if vertex_id in (edge.start, edge.end) or identifier in model.control_edges_using_vertex(vertex_id):
                    continue
                candidates += 1
                projected, parameter, distance = model.closest_edge_point(identifier, point)
            else:
                if vertex_id in incident_faces[identifier]:
                    continue
                candidates += 1
                projected, parameter, distance = model.project_to_face(identifier, point)
            # Exact/coincident defects remain classified by strict_audit.
            if model.tolerance.length < distance <= threshold:
                entity = EntityRef(kind, identifier)
                code = 'near_point_edge' if kind == 'edge' else 'near_point_face'
                findings.append(DiagnosisFinding(f'{code}:vertex{vertex_id},{kind}{identifier}', code,
                    (EntityRef('vertex', vertex_id), entity),
                    'Separated geometry is within the configured near-gap threshold; connection may be intentional or missing.',
                    float(distance), False, 'warning', (('gap_m', float(distance)),), threshold,
                    'Review separation and connection intent. Do not stitch automatically.'))
    # The same fresh spatial index filters segment pairs. Shared endpoint
    # topology is already connected and strict_audit checks its validity.
    edge_bounds = {key[1]: bounds for key, bounds in items if key[0] == 'edge'}
    for first in sorted(edge_bounds):
        checkpoint()
        first_edge = model.edges[first]
        for kind, second in tree.query(edge_bounds[first].expanded(threshold), kinds=('edge',)).keys:
            checkpoint()
            if second <= first:
                continue
            second_edge = model.edges[second]
            if {first_edge.start, first_edge.end} & {second_edge.start, second_edge.end}:
                continue
            candidates += 1
            distance = _segment_gap(model, first, second)
            if model.tolerance.length < distance <= threshold:
                refs = (EntityRef('edge', first), EntityRef('edge', second))
                regions = sorted(set(model.faces_using_edge(first)) | set(model.faces_using_edge(second)))
                refs += tuple(EntityRef('face', face) for face in regions)
                findings.append(DiagnosisFinding(f'near_edges:edge{first},edge{second}', 'near_edges', refs,
                    'Separated straight segments/planar-region boundaries are within the near-gap threshold.',
                    distance, False, 'warning', (('gap_m', distance),), threshold,
                    'Review intended separation/connectivity; do not stitch automatically.'))
    return findings, candidates, tuple(sorted(set(unsupported), key=lambda item: (item.kind, item.id)))


def _feature_findings(model):
    findings = []
    try:
        model.features.validate_persistence(model)
    except Exception as error:
        findings.append(DiagnosisFinding('feature:validation', 'invalid_feature_history', (), str(error),
            severity='blocker', action='Repair the feature-history definition or materialization before accepting diagnosis.'))
    records = model.features.records
    for record in records:
        if not record.suppressed and record.state not in ('ok', 'active', 'frozen'):
            findings.append(DiagnosisFinding(f'feature:{record.feature_id}:state', 'unresolved_feature', (),
                f'Feature {record.feature_id} ({record.name}) is {record.state}: {record.diagnostic or "no qualified materialization"}',
                severity='blocker', action='Resolve feature intent through supported feature commands.'))
        if record.suppressed:
            continue
        for key, anchors in sorted(record.inputs.items()):
            for index, anchor in enumerate(anchors):
                if not model.features.resolve(anchor, model):
                    refs = (anchor,) if isinstance(anchor, EntityRef) else ()
                    findings.append(DiagnosisFinding(f'feature:{record.feature_id}:input:{key}:{index}', 'unresolved_feature_input', refs,
                        f'Feature {record.feature_id} input {key}[{index}] cannot be resolved.', severity='blocker'))
        for key, anchor in sorted(record.outputs.items()):
            if not model.resolve_ref(anchor):
                findings.append(DiagnosisFinding(f'feature:{record.feature_id}:output:{key}', 'unresolved_feature_output', (anchor,),
                    f'Feature {record.feature_id} output {key} cannot be resolved.', severity='blocker'))
    return findings, len(records)

def diagnose_geometry(model, options: CleanerOptions | None = None, *, cancelled=None, progress=None) -> DiagnosisReport:
    """Evaluate every declared diagnostic; defects are distinct from coverage.

    Callback signature: progress(phase: str, fraction: float). Cancellation is
    cooperative between phases and near-gap candidates; the existing strict
    audit is indivisible and cancellation is checked immediately after it.
    """
    options = CleanerOptions() if options is None else options
    if not isinstance(options, CleanerOptions):
        raise TypeError('options must be CleanerOptions')
    started = perf_counter()
    identity, revision = str(model.model_id), model.revision
    source = {'units': model.units, 'coordinate_transform': None if model.coordinate_transform is None else model.coordinate_transform.tolist(),
              'vertices': len(model.vertices), 'edges': len(model.edges), 'faces': len(model.faces),
              'members': len(model.members), 'sheets': len(model.sheets), 'feature_records': len(model.features)}
    source['owner_audit_measurements'] = 'Raw source-coordinate frame and document units; not converted to SI.'
    source['quality_measurements'] = 'SI only when metre/identity coordinate contract is accepted.'
    findings, checks, timings = [], [], []
    quality = audit = None
    coverage = ['Full ANYgeometry strict audit: topology/structural validity, manifold/orientation, ownership, attachments/junctions, orphan controls and spatial classifications.',
                'Quality: small/short/sliver geometry, exact polygon duplicates, vertex proximity and topological components.',
                'Feature intent: public persistence validation, materialization status and every input/output reference; no regeneration.',
                'Near gaps: spatially filtered point-straight-edge, straight-segment pairs and point-supported-planar-face minimum distances.',
                'For bounded planar polygons, positive minimum separation is covered by vertex-face and boundary edge-edge candidates; zero-distance crossings/overlaps belong to strict audit. Curved domains remain unsupported/incomplete.',
                'Existing strict audit is indivisible; cancellation is observed before/after it. Timings are nondeterministic; other report content is deterministic.']
    coverage.append('When near_gaps_overlapped is true, gap work on a detached clone overlaps strict child classification; phase durations overlap and total is the actual critical-path wall time.')
    gap_stop = Event()
    gap_executor = gap_future = None
    gap_state = {}
    def cancellation_requested():
        if gap_stop.is_set():
            return True
        if cancelled is not None and cancelled():
            gap_stop.set()
            return True
        return False
    def checkpoint():
        if cancellation_requested():
            raise _Cancelled()
    def gap_checkpoint():
        if gap_stop.is_set():
            raise _Cancelled()
    def start_gap_during_classification():
        nonlocal gap_executor, gap_future
        if not coordinate_ok:
            return
        try:
            clone_started = perf_counter()
            detached = model.clone(preserve_identity=True)
            timings.append(('near_gaps_clone', perf_counter() - clone_started))
            def evaluate_gap():
                started = perf_counter()
                try:
                    return _gap_findings(detached, options.near_vertex, gap_checkpoint)
                finally:
                    gap_state['elapsed'] = perf_counter() - started
            gap_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='geometry-near-gaps')
            gap_future = gap_executor.submit(evaluate_gap)
            source['audit_execution']['near_gaps_overlapped'] = True
        except Exception as error:
            # A gap observer failure must not abort independent strict work.
            gap_state['start_error'] = error
    def phase(name, fraction):
        checkpoint()
        if progress is not None:
            progress(name, fraction)
    coordinate_ok = True
    try:
        _require_coordinate_contract(model)
        checks.append(DiagnosisCheck('coordinate_contract', 'completed', 'Metre coordinates and absent/identity transform.'))
    except Exception as error:
        coordinate_ok = False
        checks.append(DiagnosisCheck('coordinate_contract', 'unsupported', str(error)))
        findings.append(DiagnosisFinding('coordinate_contract', 'unsupported_coordinates', (), str(error), severity='blocker', action='Convert document coordinates explicitly; strict owner audit below uses the source coordinate frame.'))
    try:
        phase('quality', 0.)
        t = perf_counter()
        try:
            if coordinate_ok:
                # Full diagnosis retains all quality findings, independent of the
                # fast repair-review cap. No viewport or fast-review truncation.
                quality = replace(review_geometry(model, replace(options, max_vertex_pairs=sys.maxsize)))
                for item in quality.findings:
                    severity = 'info' if item.kind == 'unowned_edge' else 'warning'
                    if item.kind.startswith('unsupported'):
                        severity = 'blocker'
                    tolerance = {'short_edge': options.short_edge, 'small_face': options.small_face,
                                 'sliver_face': options.sliver_quality, 'near_vertices': options.near_vertex}.get(item.kind)
                    findings.append(DiagnosisFinding(item.id, item.kind, item.refs, item.message, item.value,
                        item.repairable, severity, () if item.value is None else (('value', item.value),), tolerance))
                checks.append(DiagnosisCheck('quality', 'completed' if quality.complete else 'unsupported',
                                             'All findings retained; see quality coverage for unsupported cases.'))
            else:
                checks.append(DiagnosisCheck('quality', 'unsupported', 'Coordinate contract prevents SI quality measurements.'))
        except _Cancelled:
            raise
        except Exception as error:
            checks.append(DiagnosisCheck('quality', 'failed', f'{type(error).__name__}: {error}'))
            findings.append(DiagnosisFinding('quality:failure', 'diagnostic_failure', (),
                f'{type(error).__name__}: {error}', severity='blocker'))
        timings.append(('quality', perf_counter() - t))
        phase('feature_intent', .15)
        t = perf_counter()
        try:
            feature_findings, feature_count = _feature_findings(model)
            findings.extend(feature_findings)
            checks.append(DiagnosisCheck('feature_intent', 'incomplete' if feature_findings else 'completed',
                'Feature persistence, state, input and output references checked without regeneration.', feature_count))
        except _Cancelled:
            raise
        except Exception as error:
            checks.append(DiagnosisCheck('feature_intent', 'failed', f'{type(error).__name__}: {error}'))
            findings.append(DiagnosisFinding('feature_intent:failure', 'diagnostic_failure', (),
                f'{type(error).__name__}: {error}', severity='blocker'))
        timings.append(('feature_intent', perf_counter() - t))
        phase('strict_audit', .2)
        t = perf_counter()
        try:
            execution = {'route': 'serial', 'effective_workers': 1, 'status': 'completed', 'near_gaps_overlapped': False}
            source['audit_execution'] = execution
            if _parallel_candidate_model(model):
                from .audit_parallel import strict_audit_parallel
                audit = strict_audit_parallel(model, workers=8, cancelled=cancellation_requested, execution=execution,
                                             classification_started=start_gap_during_classification)
            else:
                audit = strict_audit(model)
            timings.extend(('strict_' + name, value) for name, value in execution.pop('_timings', {}).items())
            timings.append(('strict_audit', perf_counter() - t))
            for issue in audit.issues:
                raw = issue.to_dict()
                digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
                measurements = tuple((name, float(getattr(issue, name))) for name in
                                     ('measured_gap', 'measured_angle', 'overlap_length', 'overlap_area') if getattr(issue, name) is not None)
                findings.append(DiagnosisFinding('audit:' + issue.code.value + ':' + digest, issue.code.value,
                    tuple(EntityRef(entity.kind, entity.id) for entity in issue.entities), issue.message,
                    issue.measured_gap, False, issue.severity.name.lower(), measurements, issue.tolerance_used,
                    issue.recommended_action or 'Review the owner classification and intended model topology.', raw))
            status = 'completed' if audit.completed and audit.verified and audit.metrics.classification_complete else 'incomplete'
            if any(issue.code.value in ('unsupported_candidate', 'capability_missing', 'unclassified_candidate', 'unverified_classification', 'check_failed') for issue in audit.issues):
                status = 'incomplete'
            checks.append(DiagnosisCheck('strict_audit', status, 'Owner completion/classification provenance retained verbatim.', audit.metrics.candidate_count))
        except _Cancelled:
            raise
        except Exception as error:
            checks.append(DiagnosisCheck('strict_audit', 'failed', f'{type(error).__name__}: {error}'))
            findings.append(DiagnosisFinding('strict_audit:failure', 'diagnostic_failure', (),
                f'{type(error).__name__}: {error}', severity='blocker'))
            source.get('audit_execution', {})['status'] = 'failed'
            if not any(key == 'strict_audit' for key, value in timings):
                timings.append(('strict_audit', perf_counter() - t))
        phase('near_gaps', .8)
        t = perf_counter()
        try:
            if coordinate_ok:
                if 'start_error' in gap_state:
                    raise gap_state['start_error']
                if gap_future is None:
                    gap_findings, candidates, unsupported = _gap_findings(model, options.near_vertex, checkpoint)
                else:
                    while not gap_future.done():
                        checkpoint()
                        wait((gap_future,), timeout=.05)
                    checkpoint()
                    gap_findings, candidates, unsupported = gap_future.result()
                findings.extend(gap_findings)
                checks.append(DiagnosisCheck('near_gaps', 'unsupported' if unsupported else 'completed',
                    'Point-edge, edge-edge and planar-region gap coverage; unsupported entities: ' + ', '.join(map(str, unsupported)), candidates))
            else:
                checks.append(DiagnosisCheck('near_gaps', 'unsupported', 'Coordinate contract prevents SI distance checks.'))
        except _Cancelled:
            raise
        except Exception as error:
            checks.append(DiagnosisCheck('near_gaps', 'failed', f'{type(error).__name__}: {error}'))
            findings.append(DiagnosisFinding('near_gaps:failure', 'diagnostic_failure', (),
                f'{type(error).__name__}: {error}', severity='blocker'))
        timings.append(('near_gaps', gap_state.get('elapsed', perf_counter() - t)))
        phase('finished', 1.)
    except _Cancelled:
        checks.append(DiagnosisCheck('cancelled', 'cancelled', 'User cancellation; remaining phases not evaluated.'))
        findings.append(DiagnosisFinding('cancelled', 'diagnosis_cancelled', (), 'Diagnosis cancelled; partial findings retained.', severity='blocker'))
    except Exception as error:
        checks.append(DiagnosisCheck('diagnostic_failure', 'failed', f'{type(error).__name__}: {error}'))
        findings.append(DiagnosisFinding('diagnostic_failure', 'diagnostic_failure', (), f'{type(error).__name__}: {error}', severity='blocker'))
    finally:
        gap_stop.set()
        if gap_executor is not None:
            # Cooperative gap checkpoints ensure no background work survives
            # a returned report, including cancellation and phase failures.
            gap_executor.shutdown(wait=True, cancel_futures=True)
            if 'elapsed' in gap_state and not any(name == 'near_gaps' for name, value in timings):
                timings.append(('near_gaps', gap_state['elapsed']))
    completed_names = {check.name for check in checks}
    for name in ('quality', 'feature_intent', 'strict_audit', 'near_gaps'):
        if name not in completed_names:
            checks.append(DiagnosisCheck(name, 'not_evaluated', 'Earlier cancellation/failure prevented this phase.'))
    if str(model.model_id) != identity or model.revision != revision:
        checks.append(DiagnosisCheck('source_consistency', 'failed', 'Source changed during diagnosis; discard stale results.'))
    else:
        checks.append(DiagnosisCheck('source_consistency', 'completed', 'Source identity and committed revision unchanged.'))
    timings.append(('total', perf_counter() - started))
    return DiagnosisReport(identity, revision, options, tuple(sorted(findings, key=lambda item: item.id)),
                           tuple(checks), tuple(coverage), tuple(timings), quality, audit, source)



