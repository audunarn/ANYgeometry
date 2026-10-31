"""Full loaded-store coverage for the PP1c engineering boundary.

Definition metadata is protected, not cleared. Reviewed numerical cache roots
are disposable only inside a fresh owned worker, before fresh task input creation.
Unknown stores and active contexts decline before any mutation. This policy is
bound to the pinned source/runtime and finite canaries, not a universal proof.
"""
import contextvars
import functools
import json
import sys
import time
import weakref

from ._planning_purity import inventory
from ._planning_qualification import QualificationDecline, reset_declined

_NUMERICAL_ROOTS = frozenset((
    '_GAUSS_LEGENDRE', '_BRANCH_CONSTANTS', '_PI_CACHE', '_W_FACTORS', '_ISO_CACHE',
    '_SECTION_CANONICAL', '_ISSUED_CONTEXTS', '_SECTIONS', '_NETS', '_ABSOLUTE',
    '_SCALED', '_MEMO', '_FIRST_CHOICE', '_PATCHES', '_BRANCH_POLYNOMIALS'))
_FIXED_ROOTS = (
    'anygeometry.backend._RUNTIME_', 'anygeometry.backend._TRUE',
    'anygeometry.batch_intersections.SEAM_POLICY_VERSIONS',
    'anygeometry.batch_intersections._CARRIER_FIELDS', 'anygeometry.batch_intersections._CARRIER_ROWS',
    'anygeometry.identity._ENTITY_KIND_ORDER', 'anygeometry.nurbs._CURVE_FIELDS',
    'anygeometry.nurbs._SURFACE_FIELDS', 'anygeometry.nurbs._ON_SURFACE_FIELDS',
    'anygeometry.nurbs_surface_intersection._topology._PLANE',
    'anygeometry.prepared_mixed_sheet_network._SURFACE_FAMILIES',
    'anygeometry.quadric_algebra._ONE_PLUS_X2', 'anygeometry.serialization._RECORD_VERSIONS',
    'anygeometry.serialization._CURRENT_REQUIRED_FIELDS', 'anygeometry.serialization._CURRENT_OPTIONAL_FIELDS',
    'anygeometry.serialization._V3_REQUIRED_FIELDS', 'anygeometry.serialization._RULED_FIELDS',
    'anygeometry.serialization._QUADRIC_FIELDS', 'anygeometry.serialization._ELLIPTIC_QUADRIC_FIELDS',
    'anygeometry.serialization._QUADRIC_CURVE_FIELDS', 'anygeometry.serialization._BEZIER_CURVE_FIELDS',
    'anygeometry.support_curves._RECORD_FIELDS', 'anygeometry.support_curves._BRANCH_FIELDS',
    'anygeometry.support_curves._PLANE_FIELDS', 'anygeometry.surfaces._LSTSQ_GUFUNC',
    'anygeometry._planning_',
)
_DEFINITION_PARTS = ('__annotations__', '__annotations_cache__', '__dataclass_fields__',
                     '__annotate_func__', '__kwdefaults__', '__all__', '__conditional_annotations__',
                     '_field_defaults')
_ACTIONS = frozenset(('reset', 'covered-by-reset', 'pinned-definition', 'require-empty', 'decline'))


def check_coverage(report, rows):
    expected = {r.path: r.kind for r in report.stores}
    if report.truncated or set(rows) != set(expected):
        raise QualificationDecline('ledger coverage is incomplete/truncated')
    for path, row in rows.items():
        if row['kind'] != expected[path] or row['action'] not in _ACTIONS:
            raise QualificationDecline('ledger coverage/type decision mismatch: ' + path)


class Ledger:
    def __init__(self, report, objects, rows):
        check_coverage(report, rows)
        self.report, self.objects, self.rows = report, objects, rows
        self.resets = {p: objects[p] for p, row in rows.items() if row['action'] == 'reset'}
        self.costs = []

    def prepare(self, *, owned_worker):
        start = time.perf_counter()
        check_coverage(self.report, self.rows)
        # Refusal checks precede all mutations, including independent caches.
        for path, row in self.rows.items():
            if row['action'] == 'decline':
                raise QualificationDecline('declined ledger store: ' + path)
            if row['action'] == 'require-empty':
                value = self.objects[path]
                if type(value) is contextvars.ContextVar:
                    try:
                        value = value.get()
                    except LookupError:
                        value = None
                    if path == 'anygeometry.arrangement_geometry._JUNCTION_BOUNDS' and value is False:
                        # Declared default False, guarded by a restoring context
                        # manager: inactive switch, not an active work/budget token.
                        continue
                    if value is not None and not (type(value) in (tuple, dict, list, set) and len(value) == 0):
                        raise QualificationDecline('active context declines: ' + path)
                elif len(value):
                    raise QualificationDecline('active recursion/default state declines: ' + path)
        counts = reset_declined(self.resets, owned_worker=owned_worker)
        self.costs.append(time.perf_counter() - start)
        return counts

    def record(self):
        return dict(stores=list(self.rows.values()), aliases=list(self.report.aliases),
                    unknowns=list(self.report.unknowns), truncated=self.report.truncated,
                    reset_seconds=list(self.costs))


def build_ledger(report, objects):
    if set(objects) != {r.path for r in report.stores}:
        raise QualificationDecline('ledger handle coverage mismatch')
    rows = {}
    reset_roots = []
    # First choose concrete cache roots. Nested weak-store implementation dicts
    # are covered by the owner's clear operation, never independently cleared.
    for row in report.stores:
        path, value = row.path, objects[row.path]
        parts = path.split('.')
        kind = type(value)
        if path.startswith('anygeometry._planning_'):
            action = 'pinned-definition'
        elif kind in (functools._lru_cache_wrapper, weakref.WeakKeyDictionary,
                      weakref.WeakValueDictionary, weakref.WeakSet):
            action = 'reset'
        elif parts[-1] in _NUMERICAL_ROOTS and kind in (dict, list, set):
            action = 'reset'
        elif kind is contextvars.ContextVar or '.__repr__.cell[' in path or '.deepcopy.__defaults__' in path:
            action = 'require-empty'
        elif any(part in _DEFINITION_PARTS for part in parts) or path.startswith(_FIXED_ROOTS):
            action = 'pinned-definition'
        else:
            action = 'decline'
        if action == 'reset':
            reset_roots.append(path)
        rows[path] = dict(path=path, kind=row.kind, action=action,
                         reason='PP1c reviewed source/runtime policy; unknowns refuse')
    for path, row in rows.items():
        for parent in reset_roots:
            if path != parent and (path.startswith(parent + '.') or path.startswith(parent + '[')):
                row.update(action='covered-by-reset', parent=parent)
                break
    return Ledger(report, objects, rows)


def loaded_geometry_ledger():
    modules = tuple(m for name, m in sorted(sys.modules.items())
                    if name == 'anygeometry' or name.startswith('anygeometry.'))
    objects = {}
    report = inventory(modules, store_objects=objects)
    return build_ledger(report, objects)
