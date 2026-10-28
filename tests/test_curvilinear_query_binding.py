"""Owner curvilinear query binding (``prepared_material_curvilinear._checksum``).

``_checksum`` recomputes the complete persisted encoding and reverse-incidence binding on every call
(``serialization._qualified_model_state``) and captures the actual native snapshot. For models whose edges
are all Straight/NURBS/CurveOnSurface/PlaneNurbsCurve with no features and only supported native rows, a
locally qualified (checksum, native snapshot) pair is reused; the first local qualification, any change to
either, and every other model (Arc, Spline, other families, features, unsupported rows) run the full public
``to_dict`` validation. These tests use the real ``square_cells`` owner proofs; counters delegate to the
actual serialization qualification.
"""
from dataclasses import replace

import pytest

from anygeometry import (GeometryError, validate_prepared_material_curvilinear_cells,
    validate_prepared_material_curvilinear_cells_binding)
from anygeometry.serialization import to_dict
import anygeometry.prepared_material_curvilinear as owner
import anygeometry.serialization as serialization
from test_prepared_material_curvilinear import square_cells


def issue(model, regions, chords, cells, **extra):
    return validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001, **extra)


@pytest.fixture
def qualifications(monkeypatch):
    real, calls = serialization._qualify_document, []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serialization, '_qualify_document', counting)
    return calls


def test_owner_checksum_is_the_content_fingerprint_of_public_serialization():
    model, regions, chords, cells = square_cells()
    assert owner._checksum(model) == to_dict(model)['checksum']['value']


def test_same_content_reuses_the_valid_qualification_and_changed_content_requalifies(qualifications):
    model, regions, chords, cells = square_cells()
    owner._checksum(model)
    warmed = len(qualifications)
    owner._checksum(model)
    owner._checksum(model)
    assert len(qualifications) == warmed                           # the complete encoding ran; no requalification
    original = model._units
    model._units = original + '-edited'                            # same revision, different persisted content
    try:
        owner._checksum(model)
        assert len(qualifications) > warmed                        # helper requalification and full native proof
    finally:
        model._units = original
    assert owner._checksum(model) == to_dict(model)['checksum']['value']


@pytest.mark.parametrize('edit', ('units', 'position'))
def test_raw_same_revision_public_edit_rejects_an_existing_receipt(edit):
    model, regions, chords, cells = square_cells()
    receipt = issue(model, regions, chords, cells)
    revision = model.revision
    if edit == 'units':
        original = model._units
        model._units = original + '-edited'
        restore = lambda: setattr(model, '_units', original)
    else:
        vertex = next(iter(model.vertices.values()))
        original = vertex.position
        object.__setattr__(vertex, 'position', original + 0.01)
        restore = lambda: object.__setattr__(vertex, 'position', original)
    try:
        assert model.revision == revision                           # revision alone would not notice
        with pytest.raises(GeometryError):
            validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    finally:
        restore()
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)   # restored: valid again


def test_reverse_incidence_tamper_after_warm_state_forces_requalification(qualifications):
    model, regions, chords, cells = square_cells()
    owner._checksum(model)
    warmed = len(qualifications)
    key = max(model._face_structural_uses, default=0) + 1000
    model._face_structural_uses[key] = ()                         # derived index outside the persisted checksum
    try:
        try:
            owner._checksum(model)
        except GeometryError:
            pass                                                  # a refusal is an acceptable requalification
        assert len(qualifications) > warmed
    finally:
        del model._face_structural_uses[key]
    owner._checksum(model)


def unselected_cylinder_fixture():
    from test_prepared_material_source_transfer import prepared_square
    from anygeometry.surfaces import Cylinder
    source, prepared, regions, chords, receipt = prepared_square(unselected_cylinder=True)
    cylinder = next(face.surface for face in prepared.faces.values() if type(face.surface) is Cylinder)
    return prepared, regions, chords, receipt, cylinder


@pytest.mark.parametrize('basis', ('zero', 'wrong'))
def test_poisoned_unselected_cylinder_native_basis_is_refused_after_warm_state(basis, qualifications):
    import numpy as np
    prepared, regions, chords, receipt, cylinder = unselected_cylinder_fixture()
    owner._checksum(prepared)                                      # warm: valid state
    warmed = len(qualifications)
    owner._checksum(prepared)
    # this fixture holds Arc edges, which are not cache-safe: the conservative fallback runs the full
    # public validation (and therefore a qualification) on every call
    assert len(qualifications) > warmed
    original = cylinder._circumferential
    poisoned = np.zeros_like(original) if basis == 'zero' else np.asarray(cylinder.axis, dtype=float).copy()
    before = serialization._serialized_model_state(prepared)['checksum']['value']
    object.__setattr__(cylinder, '_circumferential', poisoned)
    phases = []
    try:
        # the persisted encoding is unchanged: only the native snapshot and full validation can notice
        assert serialization._serialized_model_state(prepared)['checksum']['value'] == before
        with pytest.raises(GeometryError):
            owner._checksum(prepared)
        assert len(qualifications) > warmed                        # changed native state forced requalification
        with pytest.raises(GeometryError):
            validate_prepared_material_curvilinear_cells_binding(prepared, regions, receipt)
        with pytest.raises(GeometryError):
            validate_prepared_material_curvilinear_cells_binding(
                prepared, regions, receipt, cancellation_check=lambda phase: phases.append(phase) or False)
        assert phases == []                                        # refused before any caller callback
    finally:
        object.__setattr__(cylinder, '_circumferential', original)
    validate_prepared_material_curvilinear_cells_binding(prepared, regions, receipt)   # restored: old receipt valid


def test_native_poison_applied_before_the_first_local_qualification_is_refused():
    import numpy as np
    prepared, regions, chords, receipt, cylinder = unselected_cylinder_fixture()
    owner._NATIVE_QUALIFIED.pop(prepared, None)                    # no local qualification exists yet
    serialization._qualified_model_state(prepared)                 # the global helper alone is already warm
    original = cylinder._circumferential
    object.__setattr__(cylinder, '_circumferential', np.zeros_like(original))
    try:
        with pytest.raises(GeometryError):
            owner._checksum(prepared)
    finally:
        object.__setattr__(cylinder, '_circumferential', original)
    assert owner._checksum(prepared)


def arc_sector_model():
    """The NURBS square plus an unselected explicit-Plane quarter-disk sector bounded by an Arc."""
    import math
    from anygeometry import Plane
    model, regions, chords, cells = square_cells()
    start = model.add_point(11., 0., 0.)
    via = model.add_point(10. + math.sqrt(.5), math.sqrt(.5), 0.)
    end = model.add_point(10., 1., 0.)
    centre = model.add_point(10., 0., 0.)
    arc = model.add_arc(start, via, end)
    closing = [model.add_line(end, centre), model.add_line(centre, start)]
    model.add_face([arc, *closing], surface=Plane((10., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    return model, arc


def test_arc_edges_use_full_public_validation_on_every_call(qualifications):
    model, arc = arc_sector_model()
    model.arc_frame(arc)                                           # fills the evaluated arc cache
    owner._checksum(model)
    counts = [len(qualifications)]
    for _ in range(2):
        owner._checksum(model)
        counts.append(len(qualifications))
    assert counts[1] > counts[0] and counts[2] > counts[1]         # the fallback qualifies each call


def test_poisoned_cached_arc_frame_is_refused_like_public_to_dict():
    import copy
    import numpy as np
    model, arc = arc_sector_model()
    model.arc_frame(arc)
    assert owner._checksum(model)                                  # a valid warm state
    stamp, frame = model._arc_cache[arc]
    original_entry = model._arc_cache[arc]
    before = serialization._serialized_model_state(model)['checksum']['value']
    poisoned = copy.copy(frame)
    object.__setattr__(poisoned, 'center', np.asarray(frame.center, dtype=float) + np.array([0., 0., .5]))
    model._arc_cache[arc] = (stamp, poisoned)                      # same stamp, evaluated centre now off-plane
    try:
        assert serialization._serialized_model_state(model)['checksum']['value'] == before
        with pytest.raises(GeometryError):
            to_dict(model)
        with pytest.raises(GeometryError):
            owner._checksum(model)
    finally:
        model._arc_cache[arc] = original_entry
    assert owner._checksum(model)


def test_arc_and_legacy_spline_edges_are_not_cache_safe_but_nurbs_squares_are():
    from anygeometry.native_support_snapshots import capture_native_supports
    model, regions, chords, cells = square_cells()
    assert owner._native_cache_safe(model, capture_native_supports(model))
    a, b, c, d = (model.add_point(x, 3., 0.) for x in (20., 21., 22., 23.))
    model.add_spline(a, (b, c), d)
    assert not owner._native_cache_safe(model, capture_native_supports(model))
    model2, arc = arc_sector_model()
    assert not owner._native_cache_safe(model2, capture_native_supports(model2))


def test_nonempty_feature_history_or_unsupported_native_rows_use_full_validation_every_call(qualifications):
    model, regions, chords, cells = square_cells()
    from anygeometry.native_support_snapshots import capture_native_supports
    native = capture_native_supports(model)
    assert owner._native_cache_safe(model, native)                  # empty features, supported rows
    assert not owner._native_cache_safe(model, native + ((999, 'unsupported', ()),))
    assert not owner._native_cache_safe(model, native + ((999, 'unsupported_parameterization', ()),))


def test_native_kernel_poison_persisting_past_an_ordinary_callback_refuses_before_proof_reads():
    model, regions, chords, cells = square_cells()
    curve = model.edges[next(iter(model.edges))].curve.pcurve
    other = replace(curve, weights=(1., 2.))
    prepared, fired = curve._prepared, []

    def poison(phase):
        if not fired:
            fired.append(phase)
            object.__setattr__(curve, '_prepared', other._prepared)
        return False

    try:
        with pytest.raises(GeometryError):
            issue(model, regions, chords, cells, cancellation_check=poison)
    finally:
        object.__setattr__(curve, '_prepared', prepared)
    assert fired
    validate_prepared_material_curvilinear_cells_binding(model, regions, issue(model, regions, chords, cells))


def test_callback_source_edit_and_cancellation_keep_authority_unlaundered():
    model, regions, chords, cells = square_cells()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        issue(model, regions, chords, cells, cancellation_check=lambda phase: True)
    assert to_dict(model) == before
    edits = []

    def edit(phase):
        if not edits:
            edits.append(model.add_point(7., 7., 7.))
        return False

    with pytest.raises(GeometryError):
        issue(model, regions, chords, cells, cancellation_check=edit)
    assert edits                                                    # the caller's own edit stays the caller's
