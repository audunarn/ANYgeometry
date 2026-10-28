"""Actual owner pre-mesh transport; no cell receipts or generation stubs."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import (GeometryError, Plane, clone_prepared_geometry,
    plan_plane_split, apply_plane_split, query_material_surface_regions,
    qualify_model_closure_source, to_dict, from_dict)
from anygeometry.local_field_transport import (
    query_local_field_transport, validate_local_field_transport)
from test_closure_source_transfer import source_fixture, closure_fixture


def fixture(*, split=False, closure=False):
    if closure:
        source, _, _, captured = closure_fixture()
        qualify_model_closure_source(source, captured)
        prepared = captured.working_model
    else:
        source, _, _ = source_fixture()
        prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    if split:
        plan = plan_plane_split(prepared, list(prepared.faces),
            Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        application = apply_plane_split(prepared, plan, policy='imprint')
        assert len(application.parent_children[0][1]) == 2
    regions = query_material_surface_regions(prepared, list(prepared.faces))
    return source, prepared, regions


@pytest.mark.parametrize('closure', (False, True))
@pytest.mark.parametrize('split', (False, True))
def test_real_owner_clone_or_closure_and_qualified_split(split, closure):
    source, prepared, regions = fixture(split=split, closure=closure)
    before = to_dict(source), to_dict(prepared)
    result = query_local_field_transport(source, prepared, regions)
    validate_local_field_transport(source, prepared, result)
    assert len(result.footprints) == (2 if split else 1)
    assert {row.prepared_face.id for row in result.footprints} == set(prepared.faces)
    assert {row.source_face.id for row in result.footprints} == set(source.faces)
    for row in result.footprints:
        chart = next(c for c in regions.regions[row.region_index].sources if c.face == row.prepared_face)
        assert row.child_domain is chart.domain
        assert row.child_domain.boundaries == chart.boundaries
        assert row.source_coordinate_system == 'authored_query_uv'
        assert row.face_uses == chart.face_uses
        assert row.source_from_child == ((Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)))
    assert (to_dict(source), to_dict(prepared)) == before


def test_same_model_needs_real_regions_and_keeps_footprints():
    source, _, _ = source_fixture()
    regions = query_material_surface_regions(source, list(source.faces))
    result = query_local_field_transport(source, source, regions)
    validate_local_field_transport(source, source, result)
    assert all(row.prepared_face == row.source_face for row in result.footprints)


def test_missing_split_child_refused_without_any_cell_proof():
    source, prepared, _ = fixture(split=True)
    partial = query_material_surface_regions(prepared, [next(iter(prepared.faces))])
    with pytest.raises(GeometryError, match='omits required material descendants'):
        query_local_field_transport(source, prepared, partial)


@pytest.mark.parametrize('copier', (lambda m: m.clone(), lambda m: from_dict(to_dict(m))))
def test_same_bytes_without_owner_route_refused(copier):
    source, prepared, _ = fixture()
    unrelated = copier(prepared)
    regions = query_material_surface_regions(unrelated, list(unrelated.faces))
    with pytest.raises(GeometryError):
        query_local_field_transport(source, unrelated, regions)


def test_copied_receipt_and_other_model_refused():
    source, prepared, regions = fixture()
    result = query_local_field_transport(source, prepared, regions)
    with pytest.raises(GeometryError, match='not issued'):
        validate_local_field_transport(source, prepared, replace(result))
    with pytest.raises(GeometryError, match='not issued'):
        validate_local_field_transport(source.clone(), prepared, result)


@pytest.mark.parametrize('operand', ('source', 'prepared', 'regions', 'receipt'))
@pytest.mark.parametrize('during_callback', (False, True))
def test_definition_mutation_is_refused_before_or_after_callback(operand, during_callback):
    source, prepared, regions = fixture()
    result = query_local_field_transport(source, prepared, regions)
    if operand in ('source', 'prepared'):
        model = source if operand == 'source' else prepared
        target = model.faces[next(iter(model.faces))]
        name, bad = 'metadata', {'raw_same_revision_change': True}
    elif operand == 'regions':
        target, name, bad = regions.regions[0], 'material_area', 999.
    else:
        target, name, bad = result, 'footprints', ()
    original = getattr(target, name)
    calls = []
    def callback(phase):
        calls.append(phase)
        if phase == 'local field transport binding complete':
            object.__setattr__(target, name, bad)
        return False
    try:
        if not during_callback:
            object.__setattr__(target, name, bad)
        with pytest.raises(GeometryError):
            validate_local_field_transport(source, prepared, result, cancellation_check=callback)
        if not during_callback:
            assert not calls
    finally:
        object.__setattr__(target, name, original)


def test_query_callback_mutation_and_cancellation_do_not_issue_authority():
    source, prepared, regions = fixture()
    target = regions.regions[0]
    original = target.material_area
    def mutate(phase):
        if phase == 'local field transport qualification complete':
            object.__setattr__(target, 'material_area', 999.)
        return False
    try:
        with pytest.raises(GeometryError, match='regions changed'):
            query_local_field_transport(source, prepared, regions, cancellation_check=mutate)
    finally:
        object.__setattr__(target, 'material_area', original)
    before = to_dict(source), to_dict(prepared)
    with pytest.raises(GeometryError, match='cancelled'):
        query_local_field_transport(source, prepared, regions, cancellation_check=lambda phase: True)
    assert before == (to_dict(source), to_dict(prepared))


def test_ambiguous_owner_partition_refused(monkeypatch):
    # Corrupt the producer ledger only to exercise refusal; no successful proof is mocked.
    import anygeometry.prepared_material_source_transfer as owner
    source, prepared, regions = fixture()
    route = owner._LINKS[prepared][0]
    parent, children = route[3][0]
    malformed = (*route[:3], ((parent, children), (parent, children)), route[4])
    monkeypatch.setitem(owner._LINKS, prepared, (malformed,))
    with pytest.raises(GeometryError, match='malformed'):
        query_local_field_transport(source, prepared, regions)

def test_closed_split_retains_hole_and_shared_trace_in_actual_child_domains():
    from test_plane_split import bowl
    source, face, plane = bowl()
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    application = apply_plane_split(prepared,
        plan_plane_split(prepared, [face], plane), policy='imprint')
    regions = query_material_surface_regions(prepared, list(prepared.faces))
    result = query_local_field_transport(source, prepared, regions)
    validate_local_field_transport(source, prepared, result)
    assert sorted(len(row.child_domain.boundaries) for row in result.footprints) == [1, 2]
    for trace in application.trace_edges:
        occurrences = sum(path.source_edge == trace for row in result.footprints
            for loop in row.child_domain.boundaries for path in loop)
        assert occurrences == 2


def test_callback_exception_propagates_without_model_mutation():
    source, prepared, regions = fixture()
    before = to_dict(source), to_dict(prepared)
    class Stop(BaseException):
        pass
    def stop(phase):
        raise Stop()
    with pytest.raises(Stop):
        query_local_field_transport(source, prepared, regions, cancellation_check=stop)
    assert before == (to_dict(source), to_dict(prepared))


@pytest.mark.parametrize('rational', (False, True))
def test_qualified_sweep_split_uses_declared_authored_chart_map(rational):
    from test_nurbs_sweep_chart import sweep, cropped_rational_sweep
    from anygeometry.nurbs_sweep_chart import authored_query_uv
    from anygeometry.prepared_material_source_transfer import _effective
    source, parent, *_ = cropped_rational_sweep() if rational else sweep()
    prepared = clone_prepared_geometry(source, new_preparation_epoch=True)
    plane = Plane((2.5, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    apply_plane_split(prepared, plan_plane_split(prepared, [parent], plane), policy='imprint')
    regions = query_material_surface_regions(prepared, list(prepared.faces))
    result = query_local_field_transport(source, prepared, regions)
    validate_local_field_transport(source, prepared, result)
    assert len(result.footprints) == 2
    probe = (Q(1, 4), Q(3, 4))
    expected = authored_query_uv(source, parent, _effective(source, parent), (probe,))[0]
    for row in result.footprints:
        mapped = tuple(a*probe[0]+b*probe[1]+c for a, b, c in row.source_from_child)
        assert mapped == tuple(Q(v) for v in expected)
        assert row.source_coordinate_system == 'authored_query_uv'
        assert row.source_face.id == parent

@pytest.mark.parametrize('passive', (False, True), ids=('absent', 'owner-signal'))
def test_inner_region_validation_preserves_absent_or_passive_callback_identity(monkeypatch, passive):
    import anygeometry.local_field_transport as transport
    from anygeometry.cancellation import CancellationSignal
    source, prepared, regions = fixture()
    signal = CancellationSignal() if passive else None
    actual = transport.validate_material_surface_regions_binding
    observed = []
    def observe(model, material, *, cancellation_check=None):
        observed.append(cancellation_check)
        assert cancellation_check is signal
        return actual(model, material, cancellation_check=cancellation_check)
    monkeypatch.setattr(transport, 'validate_material_surface_regions_binding', observe)
    result = query_local_field_transport(source, prepared, regions, cancellation_check=signal)
    validate_local_field_transport(source, prepared, result, cancellation_check=signal)
    assert len(observed) == 2


@pytest.mark.parametrize('operand', ('source', 'receipt'))
def test_ordinary_callback_mutation_at_real_inner_frontier_remains_guarded(operand):
    source, prepared, regions = fixture()
    result = query_local_field_transport(source, prepared, regions)
    if operand == 'source':
        target = source.faces[next(iter(source.faces))]
        name, bad = 'metadata', {'inner_callback_change': True}
    else:
        target, name, bad = result, 'footprints', ()
    original = getattr(target, name)
    touched = []
    def mutate(phase):
        if phase == 'trimmed surface chart binding':
            touched.append(phase)
            object.__setattr__(target, name, bad)
        return False
    try:
        with pytest.raises(GeometryError):
            validate_local_field_transport(source, prepared, result, cancellation_check=mutate)
        assert touched
    finally:
        object.__setattr__(target, name, original)


def test_passive_cancellation_after_inner_validation_refuses_before_issuance(monkeypatch):
    import anygeometry.local_field_transport as transport
    from anygeometry.cancellation import CancellationSignal
    source, prepared, regions = fixture()
    signal = CancellationSignal()
    actual = transport.validate_material_surface_regions_binding
    issued_before = tuple(transport._ISSUED)  # Retain prior objects across this check.
    reached = []
    def cancel_after_real_validation(model, material, *, cancellation_check=None):
        assert cancellation_check is signal
        actual(model, material, cancellation_check=cancellation_check)
        reached.append(True)
        signal.cancel()
    monkeypatch.setattr(transport, 'validate_material_surface_regions_binding', cancel_after_real_validation)
    with pytest.raises(GeometryError, match='cancelled'):
        query_local_field_transport(source, prepared, regions, cancellation_check=signal)
    assert reached and signal.cancelled
    assert tuple(transport._ISSUED) == issued_before
