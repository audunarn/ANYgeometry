"""Exact owner-receipt reuse of ``validate_prepared_material_curvilinear_cells(previous=...)`` (UNRUN).

All geometry is the real owner ``square_cells`` fixture. The full source-cell proof is observed by wrapping
the real ``certify_source_cells`` with a counter that delegates to it; nothing is mocked or replaced.
"""
from dataclasses import replace

import pytest

from anygeometry import (GeometryError, query_prepared_material_boundary_chords,
    validate_prepared_material_curvilinear_cells, validate_prepared_material_curvilinear_cells_binding)
from anygeometry.serialization import to_dict
from test_prepared_material_curvilinear import square_cells


@pytest.fixture
def proofs(monkeypatch):
    import anygeometry.source_cell_complex as complex_module
    real, calls = complex_module.certify_source_cells, []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(complex_module, 'certify_source_cells', counting)
    return calls


def issue(model, regions, chords, cells, **extra):
    return validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001, **extra)


def test_exact_identity_reuses_the_proof_and_issues_a_new_valid_receipt(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    assert len(proofs) == 1
    second = issue(model, regions, chords, cells, previous=first)
    assert len(proofs) == 1                                   # no second full source-cell proof
    assert second is not first
    assert second.cells == cells and second.surface_error == first.surface_error
    for name in ('reference_nodes', 'reference_uv_by_cell', 'source_map_bounds', 'boundary_correspondence'):
        assert getattr(second, name) == getattr(first, name)
    validate_prepared_material_curvilinear_cells_binding(model, regions, second)
    validate_prepared_material_curvilinear_cells_binding(model, regions, first)


def test_default_path_without_previous_always_runs_the_full_proof(proofs):
    model, regions, chords, cells = square_cells()
    issue(model, regions, chords, cells)
    issue(model, regions, chords, cells)
    assert len(proofs) == 2


def test_changed_error_or_new_chord_receipt_runs_the_full_proof(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    other_error = validate_prepared_material_curvilinear_cells(model, regions, chords, cells,
                                                               surface_error=.002, previous=first)
    assert len(proofs) == 2 and other_error.surface_error == .002
    same_values = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', c.edge.id), 0., 1.) for c in chords.intervals), chord_error=.01)
    assert same_values is not chords                          # equal content is not the same identity
    again = issue(model, regions, same_values, cells, previous=first)
    assert len(proofs) == 3
    validate_prepared_material_curvilinear_cells_binding(model, regions, again)


def test_changed_literal_cells_with_previous_are_not_reused(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    changed = replace(cells[0], xyz=((.01, 0., 0.), *cells[0].xyz[1:]))
    with pytest.raises(GeometryError, match='literal target'):
        issue(model, regions, chords, (changed,), previous=first)
    assert len(proofs) == 2                                   # the full proof ran and refused
    validate_prepared_material_curvilinear_cells_binding(model, regions, first)


def forged_copy(receipt, *_):
    return replace(receipt)


def tampered(receipt, *_):
    object.__setattr__(receipt, 'surface_error', .5)
    return receipt


LIVE_FOREIGN = []                       # keeps the other model alive: wrong-model authority, not a dead weakref


def foreign(_receipt, model, *_):
    fixture = square_cells()
    LIVE_FOREIGN.append(fixture)
    other_model, other_regions, other_chords, other_cells = fixture
    return issue(other_model, other_regions, other_chords, other_cells)


@pytest.mark.parametrize('make', (forged_copy, tampered, foreign))
def test_unauthentic_previous_is_refused_before_any_callback(make, proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    bad = make(first, model)
    counted = len(proofs)                                     # the foreign fixture's own proof is not counted against reuse
    before, phases = to_dict(model), []
    with pytest.raises(GeometryError):
        issue(model, regions, chords, cells, previous=bad, cancellation_check=lambda phase: phases.append(phase))
    assert phases == [] and to_dict(model) == before and len(proofs) == counted
    LIVE_FOREIGN.clear()


def test_stale_previous_after_a_source_change_is_refused_before_any_callback(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    model.add_point(5., 5., 5.)                               # the revision of the issued receipt is stale
    phases = []
    with pytest.raises(GeometryError):
        issue(model, regions, chords, cells, previous=first, cancellation_check=lambda p: phases.append(p))
    assert phases == []


@pytest.mark.parametrize('mutation', ('previous', 'source'))
def test_mutation_at_a_callback_refuses_reuse(mutation, proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    done = []

    def callback(phase):
        if not done:
            done.append(phase)
            if mutation == 'previous':
                object.__setattr__(first, 'surface_error', .5)
            else:
                model.add_point(7., 7., 7.)
        return False

    with pytest.raises(GeometryError):
        issue(model, regions, chords, cells, previous=first, cancellation_check=callback)
    assert len(proofs) == 1                                   # no proof ran after the mutation was seen


def test_same_revision_runtime_kernel_poison_refuses_previous_before_any_callback(proofs):
    """Pattern of ``test_native_evaluation_kernel_and_point_cache_are_not_geometry_authority``: the evaluated
    kernel of a pcurve is replaced by another rational curve's kernel; serialization and revision are unchanged."""
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    curve = model.edges[next(iter(model.edges))].curve.pcurve
    other = replace(curve, weights=(1., 2.))
    prepared, counted, phases = curve._prepared, len(proofs), []
    serialized, revision = to_dict(model), model.revision
    object.__setattr__(curve, '_prepared', other._prepared)
    try:
        assert to_dict(model) == serialized and model.revision == revision      # invisible to public state
        with pytest.raises(GeometryError):
            issue(model, regions, chords, cells, previous=first, cancellation_check=lambda p: phases.append(p))
        assert phases == [] and len(proofs) == counted
        assert curve._prepared is other._prepared                 # the refusal did not touch the poisoned kernel
    finally:
        object.__setattr__(curve, '_prepared', prepared)
    validate_prepared_material_curvilinear_cells_binding(model, regions, first)
    again = issue(model, regions, chords, cells, previous=first)    # restored: exact reuse is available again
    assert len(proofs) == counted and again is not first


def test_changed_uv_with_unchanged_xyz_enters_the_full_proof_and_refuses(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    counted = len(proofs)
    uv = cells[0].uv
    moved = replace(cells[0], uv=(*uv[:3], (uv[3][0], .5)))      # xyz untouched
    assert moved.xyz == cells[0].xyz
    with pytest.raises(GeometryError):
        issue(model, regions, chords, (moved,), previous=first)
    assert len(proofs) == counted + 1                             # a real proof attempt, no reuse
    validate_prepared_material_curvilinear_cells_binding(model, regions, first)


def test_changed_global_node_ids_run_the_full_proof_and_are_never_reused(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    counted = len(proofs)
    shifted = (replace(cells[0], node_ids=tuple(i + 10 for i in cells[0].node_ids)),)
    assert shifted[0].uv == cells[0].uv and shifted[0].xyz == cells[0].xyz
    try:
        second = issue(model, regions, chords, shifted, previous=first)
    except GeometryError:
        second = None                                            # a typed refusal is also a full-proof outcome
    assert len(proofs) == counted + 1                             # the proof ran: identity is not inferred
    if second is not None:
        assert second is not first and second.cells == shifted and second.cells != first.cells
        validate_prepared_material_curvilinear_cells_binding(model, regions, second)
    validate_prepared_material_curvilinear_cells_binding(model, regions, first)


def test_cancellation_during_reuse_issues_nothing_and_keeps_previous_valid(proofs):
    model, regions, chords, cells = square_cells()
    first = issue(model, regions, chords, cells)
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        issue(model, regions, chords, cells, previous=first,
              cancellation_check=lambda phase: phase == 'curvilinear source-cell qualification complete')
    assert to_dict(model) == before and len(proofs) == 1
    validate_prepared_material_curvilinear_cells_binding(model, regions, first)
