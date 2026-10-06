"""Whole captured-function proofs; display stations are never acceptance evidence."""
from dataclasses import replace
from fractions import Fraction as F
import gc
import os
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from anygeometry import (GeometryError, apply_intersections, plan_intersections,
    query_prepared_native_arc_parameter_maps as query,
    validate_prepared_native_arc_parameter_maps_binding as validate)
from anygeometry.cylinder_charts import _Proof, _Refusal
from anygeometry.native_arc_parameter_maps import NativeArcParameterMapPolicy as ProofPolicy
from anygeometry.native_arc_parameter_maps import NativeArcFrame, _bound, _issued
from anygeometry.edge_subcurve_preimages import _edge_ancestry_definition, _reseal_arc_occurrence
from anygeometry.definition_binding import definition_checksum
import anygeometry
import test_cylinder_boundary_correspondence as cylinder_owner
import test_edge_subcurve_preimages as owner


def test_effective_source_runtime():
    assert Path(anygeometry.__file__).resolve() == Path(__file__).resolve().parents[1] / 'src/anygeometry/__init__.py'
    assert all(os.environ[x] == '1' for x in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    if os.environ.get('NATIVE_ARC_RUNTIME_EVIDENCE'):
        Path(os.environ['NATIVE_ARC_RUNTIME_EVIDENCE']).write_text(json.dumps({
            'executable': sys.executable, 'python_version': sys.version,
            'source_origin': str(Path(anygeometry.__file__).resolve()),
            'numpy_version': version('numpy'), 'pytest_version': version('pytest'),
            'threads': {x:os.environ[x] for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        }, indent=2))


def test_public_plan_apply_actual_arc_splits():
    from anygeometry.generators.structural import cylinder
    model = cylinder(1., 1., circumferential_segments=3)
    roots = set(model.edges)
    model.add_plate(model.add_points(((2., .5, -.5), (2., .5, 1.5),
                                      (-2., .5, 1.5), (-2., .5, -.5))))
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    binding = query(model)
    assert binding.requested_edge_ids is None
    assert binding.selected_edge_ids == tuple(sorted({r.edge_id for r in binding.records+binding.alias_records}))
    split = [row for row in binding.records if row.edge_id not in roots]
    ancestry = {row.edge_id:row for row in model._edge_subcurve_preimages_receipt[0].arc_records}
    for row in binding.records:
        source = ancestry[row.edge_id].ancestor
        assert (row.ancestor_model_id, row.ancestor_revision, row.ancestor_source_checksum) == (
            source.model_id, source.revision, source.source_checksum)
    assert split and any(row.interval != (F(0), F(1)) for row in split)
    assert all(row.classification in ('bounded', 'refused') for row in split)
    assert any(row.classification == 'bounded' for row in split)
    assert all(row.residual_bound <= row.tolerance for row in split if row.classification == 'bounded')
    validate(model, binding)


def test_unsplit_identity_and_receipt_forgery():
    model, _ = cylinder_owner.panel(0.)
    binding = query(model)
    assert binding.records and all(r.classification == 'exact' and r.residual_bound == 0 for r in binding.records)
    validate(model, binding)
    with pytest.raises(GeometryError, match='not issued'):
        validate(model, replace(binding))
    original = binding.records
    object.__setattr__(binding, 'records', ())
    with pytest.raises(GeometryError, match='changed'):
        validate(model, binding, cancellation_check=lambda _: object.__setattr__(binding, 'records', original))
    object.__setattr__(binding, 'records', original)
    validate(model, binding)


def test_mutate_restore_uses_detached_frames_and_entry_receipt():
    model, _ = cylinder_owner.panel(0.)
    expected = query(model)
    source = model._edge_subcurve_preimages_receipt[0]
    rows = source.arc_records
    def callback(_):
        object.__setattr__(source, 'arc_records', ())
        object.__setattr__(source, 'arc_records', rows)
        return False
    actual = query(model, cancellation_check=callback)
    assert actual.records == expected.records
    validate(model, actual, cancellation_check=callback)


def test_callback_ancestry_forgery_even_repaired_checksum_refuses():
    model, _ = cylinder_owner.panel(0.)
    receipt = model._edge_subcurve_preimages_receipt
    binding = receipt[0]
    def callback(_):
        changed = replace(binding, arc_records=())
        model._edge_subcurve_preimages_receipt = (changed, definition_checksum(changed), receipt[2])
        return False
    with pytest.raises(GeometryError, match='changed'):
        query(model, cancellation_check=callback)
    model._edge_subcurve_preimages_receipt = receipt


@pytest.mark.parametrize('validator', (False, True))
def test_cancellation_true_and_exception_identity(validator):
    model, _ = cylinder_owner.panel(0.)
    binding = query(model)
    action = (lambda cb: validate(model, binding, cancellation_check=cb)) if validator else (
        lambda cb: query(model, cancellation_check=cb))
    with pytest.raises(GeometryError, match='cancelled'):
        action(lambda _: True)
    marker = _Refusal('caller marker')
    def stop(_):
        raise marker
    with pytest.raises(_Refusal) as caught:
        action(stop)
    assert caught.value is marker


def test_budget_is_aggregate_and_has_no_partial_receipt():
    model, _ = cylinder_owner.panel(0.)
    with pytest.raises(GeometryError, match='budget'):
        query(model, policy=ProofPolicy(max_interval_operations=1))
    assert not _issued.get(model)


def test_selected_batch_only_compiles_requested_frames():
    model, _ = cylinder_owner.panel(0.)
    edge = model._edge_subcurve_preimages_receipt[0].arc_records[0].edge_id
    selected = query(model, edge_ids=[edge], policy=ProofPolicy(max_interval_operations=3))
    assert selected.selected_edge_ids == (edge,)
    validate(model, selected)
    with pytest.raises(GeometryError, match='budget'):
        query(model, policy=ProofPolicy(max_interval_operations=3))


def test_hashes_are_constant_per_batch(monkeypatch):
    import anygeometry.edge_subcurve_preimages as ancestry
    original = ancestry._serialized_model_state
    counts = []
    def counted(model):
        counts.append(1)
        return original(model)
    monkeypatch.setattr(ancestry, '_serialized_model_state', counted)
    model, _ = cylinder_owner.panel(0.)
    counts.clear()
    one = query(model, edge_ids=[model._edge_subcurve_preimages_receipt[0].arc_records[0].edge_id])
    assert one.requested_edge_ids == one.selected_edge_ids
    first = len(counts)
    counts.clear()
    query(model)
    assert len(counts) == first == 4
    validate(model, one)


def test_repeated_real_refits_and_reversed_alias_interval():
    model, _, _ = owner.fixture()
    arc = model.add_arc(*model.add_points(((8., 0., 0.), (9., 1., 0.), (10., 0., 0.))))
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    draft = owner.capture(model, allow_seed=True)
    candidate = model.clone(preserve_identity=True)
    def split(edge, station):
        parent = _edge_ancestry_definition(candidate, edge)
        _, children = candidate.split_edge(edge, station)
        owner.record(draft, edge, station, children, 1e-9, model=candidate, parent_definition=parent)
        return children
    left, right = split(arc, .31)
    children = (*split(left, .47), right)
    ancestry = owner.commit(model, candidate, draft, plan)
    result = query(model)
    rows = [r for r in result.records if r.ancestor_edge_id == arc]
    assert {r.edge_id for r in rows} == set(children)
    assert any(r.classification == 'bounded' for r in rows)
    row = next(r for r in ancestry.arc_records if r.ancestor.definition.edge_id == arc)
    definition = row.current_definition
    reversed_definition = replace(definition, start=definition.end, end=definition.start,
        positions=definition.positions[::-1])
    reversed_row = _reseal_arc_occurrence(row, reversed_definition)
    assert reversed_row.interval == row.interval[::-1]
    from anygeometry.native_arc_parameter_maps import _entry, _frame
    reverse = _entry(_Proof(ProofPolicy(), None), reversed_row,
        _frame(row.ancestor.definition), _frame(reversed_definition))
    assert reverse.interval == tuple(F(*x) for x in row.interval[::-1])
    assert reverse.residual_bound < F(1, 10**9)


@pytest.mark.parametrize('sweep', (F(2), F(-1), F(7)))
def test_frequency_direction_and_extra_turn_counterexamples(sweep):
    source = NativeArcFrame((F(0),)*3, (F(1),F(0),F(0)), (F(0),F(1),F(0)), F(1))
    child = replace(source, sweep=sweep)
    bound = _bound(_Proof(ProofPolicy(), None), source, child, (F(0), F(1)))
    # Independent analytic bound for orthogonal unit harmonics is <=2;
    # frequency discrepancy is nonzero even with identical circle image.
    assert bound > F(1, 10**9)


def test_independent_translation_oracle_and_exact_rational_products():
    source = NativeArcFrame((F(0),)*3, (F(1),F(0),F(0)), (F(0),F(1),F(0)), F(1))
    child = replace(source, center=(F(3),F(4),F(0)))
    bound = _bound(_Proof(ProofPolicy(), None), source, child, (F(0), F(1)))
    assert F(5) <= bound < F(5)+F(1,2**65)
    from anygeometry.native_arc_parameter_maps import _frame
    from anygeometry.curves import arc_frame
    model, _ = cylinder_owner.panel(.123)
    definition = model._edge_subcurve_preimages_receipt[0].arc_records[0].current_definition
    raw = arc_frame(*(tuple(float(F(*v)) for v in p) for p in definition.positions))
    captured = _frame(definition)
    assert captured.cosine == tuple(F(raw.radius)*F(float(v)) for v in raw.e1)


def test_public_reversed_alias_after_authenticated_split():
    from anygeometry.edge_subcurve_preimages import _record_edge_subcurve_unification
    model, _, _ = owner.fixture()
    start, via, end = model.add_points(((8.,0.,0.), (9.,1.,0.), (10.,0.,0.)))
    canonical = model.add_arc(start, via, end)
    duplicate = model.add_arc(end, via, start)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    draft = owner.capture(model, allow_seed=True)
    candidate = model.clone(preserve_identity=True)
    _record_edge_subcurve_unification(draft, canonical, (duplicate,), model=candidate)
    before = _edge_ancestry_definition(candidate, canonical)
    _, children = candidate.split_edge(canonical, .37)
    owner.record(draft, canonical, .37, children, 1e-9, model=candidate, parent_definition=before)
    owner.commit(model, candidate, draft, plan)
    binding = query(model, edge_ids=list(children))
    validate(model, binding)
    assert len(binding.alias_records) == 2
    assert all(r.interval[0] > r.interval[1] for r in binding.alias_records)
    assert all(r.classification == 'bounded' for r in binding.alias_records)


def test_negative_sweep_equivalent_representation_and_phase_counterexample():
    source = NativeArcFrame((F(0),)*3, (F(1),F(0),F(0)), (F(0),F(1),F(0)), F(-1))
    same = replace(source, sweep=F(1), sine=tuple(-x for x in source.sine))
    assert _bound(_Proof(ProofPolicy(), None), source, same, (F(0),F(1))) == 0
    phase_bound = _bound(_Proof(ProofPolicy(), None), source, source, (F(1,2),F(3,2)))
    assert phase_bound > F(1,10)


def test_invalid_frame_is_typed(monkeypatch):
    import anygeometry.native_arc_parameter_maps as native
    model, _ = cylinder_owner.panel(0.)
    def broken(_):
        raise ValueError('frame failure')
    monkeypatch.setattr(native, '_frame', broken)
    with pytest.raises(native.NativeArcParameterMapError, match='frame unavailable'):
        query(model)


def test_invalid_requests_stale_and_weak_issued_receipts():
    model, _ = cylinder_owner.panel(0.)
    for edges in ((x for x in (1,)), [True], [1,1], [0]):
        with pytest.raises(GeometryError):
            query(model, edge_ids=edges)
    binding = query(model)
    key = id(binding)
    del binding
    gc.collect()
    assert key not in _issued.get(model, {})
    binding = query(model)
    model.add_point(9., 9., 9.)
    with pytest.raises(GeometryError):
        validate(model, binding)
