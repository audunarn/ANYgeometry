import dataclasses
import enum
import types
import typing

import pytest

from anygeometry import _planning_ledger as l
from anygeometry import _planning_purity as p
from anygeometry import _planning_qualification as q


def make_ledger(module):
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    return l.build_ledger(report, objects, modules=(module,))


def test_field_structural_digest_detects_changed_default():
    field = dataclasses.field(default=0.0, metadata={'unit': 'm'})
    first = l.content_digest({'x': field})
    field.default = -0.0
    assert first != l.content_digest({'x': field})


def test_last_field_metadata_cannot_hide_behind_temporary_identity_reuse():
    fields = [dataclasses.field(default=i) for i in range(50)]
    before = l.content_digest(fields)
    fields[-1].default = 'changed'
    assert before != l.content_digest(fields)


@pytest.mark.parametrize('one,two', [(list[int], list[float]),
                                    (typing.List[int], typing.List[float]),
                                    (int | None, float | None)])
def test_typing_metadata_digest_is_structural(one, two):
    assert l.content_digest(one) == l.content_digest(one)
    assert l.content_digest(one) != l.content_digest(two)


def test_enum_metadata_digest_checks_member_state():
    class Choice(enum.Enum):
        A = 1
    before = l.content_digest(Choice.A)
    Choice.A.extra = 2
    assert before != l.content_digest(Choice.A)


def test_named_sentinels_accepted_but_arbitrary_objects_decline():
    import numpy as np
    from anygeometry.transactions import _MISSING
    assert l.content_digest((dataclasses.MISSING, np._NoValue, _MISSING))
    with pytest.raises(q.QualificationDecline):
        l.content_digest(object())


def test_named_sentinel_slot_state_is_checked(monkeypatch):
    before = l.content_digest(dataclasses._FIELD)
    monkeypatch.setattr(dataclasses._FIELD, 'name', 'changed-marker-name')
    assert before != l.content_digest(dataclasses._FIELD)


def test_dataclass_annotation_body_is_exact_metadata_and_guarded():
    module = types.ModuleType('anygeometry.pp1e_metadata_fixture')
    exec('class C:\n    x: int = 1', module.__dict__)
    module.C = dataclasses.dataclass(module.C)
    registry = make_ledger(module)
    assert not [row for row in registry.rows.values() if row['action'] == 'decline']
    registry.prepare(owned_worker=True)
    module.C.__dataclass_fields__['x'].default = 2
    with pytest.raises(q.QualificationDecline, match='protected'):
        registry.prepare(owned_worker=True)


def test_exact_constant_container_children_are_guarded():
    module = types.ModuleType('anygeometry.quadric_algebra')
    module._ONE_PLUS_X2 = [[1], [1, 0, 1]]
    registry = make_ledger(module)
    assert all(row['action'] == 'pinned-definition' for row in registry.rows.values())
    registry.prepare(owned_worker=True)
    module._ONE_PLUS_X2[1][0] = 2
    with pytest.raises(q.QualificationDecline, match='protected'):
        registry.prepare(owned_worker=True)


def test_runtime_globals_require_empty_and_are_never_cleared():
    module = types.ModuleType('anygeometry.backend')
    module._RUNTIME_PYTHON_GLOBALS = {'keep': 'implementation binding'}
    registry = make_ledger(module)
    assert registry.rows[module.__name__ + '._RUNTIME_PYTHON_GLOBALS']['action'] == 'require-empty'
    with pytest.raises(q.QualificationDecline):
        registry.prepare(owned_worker=True)
    assert module._RUNTIME_PYTHON_GLOBALS == {'keep': 'implementation binding'}


def test_property_metadata_checks_documentation():
    descriptor = property(lambda self: 1, doc='before')
    before = l.content_digest(descriptor)
    descriptor.__doc__ = 'after'
    assert before != l.content_digest(descriptor)


def test_fraction_metadata_is_exact():
    from fractions import Fraction
    assert l.content_digest(Fraction(1, 3)) == l.content_digest(Fraction(2, 6))
    assert l.content_digest(Fraction(1, 3)) != l.content_digest(Fraction(1, 4))


@pytest.mark.parametrize('kind', ['audit', 'qualification', 'frozen', 'range', 'sheet'])
def test_static_record_defaults_are_structural(kind):
    from anygeometry.audit import AuditMetrics
    from anygeometry.predicates import IntersectionQualificationPolicy
    from anygeometry.structural import FrozenMetadata, ParameterRange, SheetTopologyPolicy
    pairs = {
        'audit': (AuditMetrics(), AuditMetrics(candidate_count=1)),
        'qualification': (IntersectionQualificationPolicy(), IntersectionQualificationPolicy(max_boxes_per_pair=1)),
        'frozen': (FrozenMetadata(), FrozenMetadata({'name': 'changed'})),
        'range': (ParameterRange(0., 1.), ParameterRange(.5, 1.)),
        'sheet': (SheetTopologyPolicy(), SheetTopologyPolicy(connectivity='allow_disconnected')),
    }
    one, two = pairs[kind]
    assert l.content_digest(one) != l.content_digest(two)


def test_namedtuple_annotation_closure_is_bound_and_guarded():
    module = types.ModuleType('anygeometry.pp1e_namedtuple_fixture')
    module.C = typing.NamedTuple('Probe', [('x', int)])
    registry = make_ledger(module)
    cells = [cell.cell_contents for cell in vars(module.C)['__annotate_func__'].__closure__]
    paths = [path for path, obj in registry.objects.items() if any(obj is cell for cell in cells)]
    assert len(paths) == 2
    assert all(registry.rows[path]['action'] == 'pinned-definition' for path in paths)
    cells[0]['x'] = str
    with pytest.raises(q.QualificationDecline, match='protected'):
        registry.prepare(owned_worker=True)
