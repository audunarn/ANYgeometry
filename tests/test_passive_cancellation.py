"""Owner-issued passive ``CancellationSignal`` (UNRUN definitions, real ``square_cells`` owner proofs)."""
import copy
import pickle
import threading

import pytest

from anygeometry import (CancellationSignal, GeometryError, is_cancellation_signal,
    query_prepared_material_boundary_chords, validate_prepared_material_curvilinear_cells,
    validate_prepared_material_curvilinear_cells_binding)
from anygeometry.serialization import to_dict
import anygeometry.prepared_material_curvilinear as owner
from test_prepared_material_curvilinear import square_cells


def issue(model, regions, chords, cells, **extra):
    return validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001, **extra)


def test_signal_identity_state_and_thread_safety():
    signal = CancellationSignal()
    assert is_cancellation_signal(signal) and signal.cancelled is False and signal('any phase') is False
    with pytest.raises(TypeError):
        CancellationSignal(lambda: True)                         # the public constructor takes no arguments
    assert not hasattr(signal, '__dict__')                       # no mutable attribute or callback field
    with pytest.raises(AttributeError):
        signal.callback = lambda phase: True
    worker = threading.Thread(target=signal.cancel)
    worker.start()
    worker.join()
    assert signal.cancelled is True and signal() is True and signal('x') is True


def test_reinitialization_cannot_uncancel_a_live_signal():
    signal = CancellationSignal()
    signal.cancel()
    with pytest.raises(GeometryError):
        signal.__init__()
    assert is_cancellation_signal(signal) and signal.cancelled is True and signal() is True


def test_only_the_exact_issued_token_is_recognized():
    signal = CancellationSignal()
    assert not is_cancellation_signal(lambda phase=None: False)
    assert not is_cancellation_signal(threading.Event())
    assert not is_cancellation_signal(None) and not is_cancellation_signal(True)
    with pytest.raises(TypeError):
        type('Forged', (CancellationSignal,), {})                # no subclasses
    forged = object.__new__(CancellationSignal)                  # never issued: no registry entry
    assert not is_cancellation_signal(forged)
    with pytest.raises(GeometryError):
        forged()
    with pytest.raises(GeometryError):
        forged.cancel()
    with pytest.raises((GeometryError, TypeError)):
        copy.copy(signal)
    with pytest.raises((GeometryError, TypeError)):
        copy.deepcopy(signal)
    with pytest.raises(TypeError):
        pickle.dumps(signal)


def test_precancelled_signal_refuses_the_actual_proof_and_changes_nothing():
    model, regions, chords, cells = square_cells()
    before = to_dict(model)
    signal = CancellationSignal()
    signal.cancel()
    with pytest.raises(GeometryError, match='cancelled'):
        issue(model, regions, chords, cells, cancellation_check=signal)
    with pytest.raises(GeometryError, match='cancelled'):
        query_prepared_material_boundary_chords(model, regions,
            tuple((model.handle('edge', c.edge.id), 0., 1.) for c in chords.intervals),
            chord_error=.01, cancellation_check=signal)
    assert to_dict(model) == before
    receipt = issue(model, regions, chords, cells)                # the model is still fully usable
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)


def test_cancellation_during_the_actual_proof_issues_no_receipt(monkeypatch):
    """The wrapper below cancels the signal and then DELEGATES to the real source-cell proof."""
    import anygeometry.source_cell_complex as complex_module
    model, regions, chords, cells = square_cells()
    before, signal, real = to_dict(model), CancellationSignal(), complex_module.certify_source_cells

    def cancel_then_prove(*args, **kwargs):
        signal.cancel()
        return real(*args, **kwargs)

    monkeypatch.setattr(complex_module, 'certify_source_cells', cancel_then_prove)
    with pytest.raises(GeometryError, match='cancelled'):
        issue(model, regions, chords, cells, cancellation_check=signal)
    assert to_dict(model) == before
    monkeypatch.undo()
    validate_prepared_material_curvilinear_cells_binding(model, regions, issue(model, regions, chords, cells))


def test_uncancelled_signal_gives_a_valid_receipt_and_reuse_keeps_its_status_checks():
    model, regions, chords, cells = square_cells()
    signal = CancellationSignal()
    first = issue(model, regions, chords, cells, cancellation_check=signal)
    validate_prepared_material_curvilinear_cells_binding(model, regions, first, cancellation_check=signal)
    second = issue(model, regions, chords, cells, cancellation_check=signal, previous=first)
    assert second is not first
    validate_prepared_material_curvilinear_cells_binding(model, regions, second)
    signal.cancel()
    with pytest.raises(GeometryError, match='cancelled'):
        issue(model, regions, chords, cells, cancellation_check=signal, previous=second)
    validate_prepared_material_curvilinear_cells_binding(model, regions, second)   # previous stays valid


def test_forged_signal_object_has_no_passive_authority():
    model, regions, chords, cells = square_cells()
    forged = object.__new__(CancellationSignal)
    with pytest.raises(GeometryError):
        issue(model, regions, chords, cells, cancellation_check=forged)


def test_ordinary_callbacks_keep_full_mutation_pinning():
    model, regions, chords, cells = square_cells()
    before = to_dict(model)

    def mutate(phase):
        model.add_point(5., 5., 5.)
        return False

    with pytest.raises(GeometryError):
        issue(model, regions, chords, cells, cancellation_check=mutate)
    assert to_dict(model) != before                              # the caller's own edit is the caller's
    # a fresh fixture: an ordinary non-mutating callback still runs the pinned path
    model, regions, chords, cells = square_cells()
    validate_prepared_material_curvilinear_cells_binding(model, regions,
        issue(model, regions, chords, cells, cancellation_check=lambda phase: False))


def test_passive_signal_skips_per_frontier_pinning_that_ordinary_callbacks_pay(monkeypatch):
    """Structural check: a delegating counter on the owner's ``_checksum`` (actual work still done)."""
    counts = []
    real = owner._checksum

    def counting(model):
        counts.append(1)
        return real(model)

    monkeypatch.setattr(owner, '_checksum', counting)
    model, regions, chords, cells = square_cells()
    counts.clear()
    issue(model, regions, chords, cells, cancellation_check=CancellationSignal())
    passive = len(counts)
    counts.clear()
    issue(model, regions, chords, cells, cancellation_check=lambda phase: False)
    ordinary = len(counts)
    assert passive < ordinary
