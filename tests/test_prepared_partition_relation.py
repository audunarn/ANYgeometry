"""Live complete owner-partition world-disjointness regressions; owner UNRUN."""
from dataclasses import replace
import pytest

from anygeometry import (GeometryError, Plane, clone_prepared_geometry,
    plan_plane_split, apply_plane_split, query_prepared_partition_disjoint_interiors,
    has_current_prepared_partition_lineage, CancellationSignal, to_dict, from_dict)


def genuine_partition():
    from test_prepared_material_curvilinear import square_cells
    source,_,_,_ = square_cells()
    prepared = clone_prepared_geometry(source,new_preparation_epoch=True)
    result = apply_plane_split(prepared,plan_plane_split(prepared,list(prepared.faces),
        Plane((.6,0.,0.),(0.,1.,0.),(0.,0.,1.))),policy='imprint')
    children = dict(result.parent_children)[next(iter(source.faces))]
    assert len(children) == 2
    return source,prepared,tuple(prepared.handle('face',child) for child in children)


def test_genuine_complete_partition_and_ancestor_carrying_recipe_clone_prove_world_disjoint_interiors():
    source,prepared,pair = genuine_partition()
    before = to_dict(prepared)
    assert query_prepared_partition_disjoint_interiors(prepared,*pair) is True
    assert to_dict(prepared) == before
    recipe = clone_prepared_geometry(prepared)
    recipe_pair = tuple(recipe.handle('face',handle.id) for handle in pair)
    assert query_prepared_partition_disjoint_interiors(recipe,*recipe_pair) is True
    assert source.faces and prepared.faces  # retain the genuine live ancestor


def test_unknown_loaded_and_ordinary_equal_document_clones_have_no_partition_authority():
    source,prepared,pair = genuine_partition()
    for unknown in (prepared.clone(),from_dict(to_dict(prepared))):
        handles = tuple(unknown.handle('face',handle.id) for handle in pair)
        assert query_prepared_partition_disjoint_interiors(unknown,*handles) is False


def test_direct_split_without_prior_genuine_clone_has_no_new_partition_registry():
    from test_prepared_material_curvilinear import square_cells
    model,_,_,_ = square_cells()
    apply_plane_split(model,plan_plane_split(model,list(model.faces),
        Plane((.6,0.,0.),(0.,1.,0.),(0.,0.,1.))),policy='imprint')
    pair = tuple(model.handle('face',face) for face in model.faces)
    assert len(pair) == 2
    assert query_prepared_partition_disjoint_interiors(model,*pair) is False


def test_same_support_cross_parent_overlapping_faces_are_not_exempted():
    from test_prepared_material_curvilinear import square_cells
    source,_,_,_ = square_cells()
    face = next(iter(source.faces.values()))
    duplicate = source.add_face([use.edge for use in face.loop],surface=face.surface)
    prepared = clone_prepared_geometry(source,new_preparation_epoch=True)
    pair = (prepared.handle('face',face.id),prepared.handle('face',duplicate))
    assert query_prepared_partition_disjoint_interiors(prepared,*pair) is False


@pytest.mark.parametrize('operand',['source','prepared'])
def test_raw_same_revision_source_or_prepared_changes_are_stale_errors(operand):
    source,prepared,pair = genuine_partition()
    model = source if operand == 'source' else prepared
    vertex = model.vertices[next(iter(model.vertices))]
    original = vertex.position
    moved = original.copy(); moved[0] += 1.e-10
    try:
        object.__setattr__(vertex,'position',moved)
        with pytest.raises(GeometryError):
            query_prepared_partition_disjoint_interiors(prepared,*pair)
    finally:
        object.__setattr__(vertex,'position',original)


@pytest.mark.parametrize('phase',['prepared partition relation binding','prepared partition injectivity',
                                  'prepared partition relation complete'])
def test_callback_mutation_at_each_binding_and_global_proof_frontier_refuses(phase):
    source,prepared,pair = genuine_partition()
    original = pair[0].id
    def mutate(current):
        if current == phase:
            object.__setattr__(pair[0],'id',pair[1].id)
        return False
    try:
        with pytest.raises(GeometryError,match='changed during callback'):
            query_prepared_partition_disjoint_interiors(prepared,*pair,cancellation_check=mutate)
    finally:
        object.__setattr__(pair[0],'id',original)


def test_passive_cancellation_and_arbitrary_baseexception_propagate_without_geometry_change():
    source,prepared,pair = genuine_partition()
    before = to_dict(prepared)
    signal = CancellationSignal(); signal.cancel()
    with pytest.raises(GeometryError,match='cancelled'):
        query_prepared_partition_disjoint_interiors(prepared,*pair,cancellation_check=signal)
    error = KeyboardInterrupt('relation callback stop')
    def callback(phase):
        raise error
    with pytest.raises(KeyboardInterrupt) as caught:
        query_prepared_partition_disjoint_interiors(prepared,*pair,cancellation_check=callback)
    assert caught.value is error and to_dict(prepared) == before


def test_global_injectivity_zero_box_budget_is_not_silently_unknown():
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    source,prepared,pair = genuine_partition()
    before = to_dict(prepared)
    token = CURRENT_WORK.set(Work(0,32))
    try:
        # Complete topology/native preflight can exhaust the same zero-box
        # budget before reaching the metric kernel. Either phase must refuse,
        # never quietly report unknown/disjoint and never mutate the model.
        with pytest.raises(GeometryError):
            query_prepared_partition_disjoint_interiors(prepared,*pair)
    finally:
        CURRENT_WORK.reset(token)
    assert to_dict(prepared) == before


def test_registered_global_injectivity_unresolved_is_not_a_support_identity_exemption(monkeypatch):
    from anygeometry import nurbs_metric
    source,prepared,pair = genuine_partition()
    def unresolved(*args,**kwargs):
        raise GeometryError('NURBS support immersion and injective projection are not certified')
    monkeypatch.setattr(nurbs_metric,'certify_projected_chart',unresolved)
    assert query_prepared_partition_disjoint_interiors(prepared,*pair) is False


def test_injectivity_callback_error_matching_kernel_refusal_text_keeps_exception_identity():
    source,prepared,pair = genuine_partition()
    error = GeometryError('NURBS support immersion and injective projection are not certified')
    def callback(phase):
        if phase == 'prepared partition injectivity':
            raise error
        return False
    with pytest.raises(GeometryError) as caught:
        query_prepared_partition_disjoint_interiors(prepared,*pair,cancellation_check=callback)
    assert caught.value is error


def test_wrong_model_and_malformed_handles_refuse():
    source,prepared,pair = genuine_partition()
    with pytest.raises(GeometryError,match='model-bound'):
        query_prepared_partition_disjoint_interiors(prepared,source.handle('face',next(iter(source.faces))),pair[1])
    malformed = replace(pair[0])
    object.__setattr__(malformed,'id',float(malformed.id))
    with pytest.raises(GeometryError,match='model-bound'):
        query_prepared_partition_disjoint_interiors(prepared,malformed,pair[1])


def test_optional_lineage_readiness_keeps_current_partition_without_batch_receipt():
    from anygeometry import has_current_intersection_preparation
    source, prepared, pair = genuine_partition()
    before = to_dict(source), to_dict(prepared)
    assert not has_current_intersection_preparation(prepared)
    assert has_current_prepared_partition_lineage(prepared) is True
    assert query_prepared_partition_disjoint_interiors(prepared, *pair) is True
    assert (to_dict(source), to_dict(prepared)) == before
    for unknown in (prepared.clone(), from_dict(to_dict(prepared))):
        assert has_current_prepared_partition_lineage(unknown) is False


@pytest.mark.parametrize('operand', ['source', 'prepared'])
def test_optional_readiness_after_public_revision_preserves_strict_stale_proof(operand):
    from anygeometry.prepared_material_source_transfer import _LINKS
    source, prepared, pair = genuine_partition()
    links = _LINKS[prepared]
    (source if operand == 'source' else prepared).add_point(100, 100, 100)
    before = to_dict(source), to_dict(prepared)
    assert has_current_prepared_partition_lineage(prepared) is False
    assert (to_dict(source), to_dict(prepared)) == before
    assert _LINKS[prepared] is links
    with pytest.raises(GeometryError, match='stale'):
        query_prepared_partition_disjoint_interiors(prepared, *pair)


@pytest.mark.parametrize('operand', ['source', 'prepared'])
def test_readiness_cannot_treat_same_revision_raw_change_as_unavailable(operand):
    source, prepared, _pair = genuine_partition()
    model = source if operand == 'source' else prepared
    vertex = model.vertices[next(iter(model.vertices))]
    original = vertex.position
    moved = original.copy(); moved[0] += 1.e-10
    try:
        object.__setattr__(vertex, 'position', moved)
        with pytest.raises(GeometryError):
            has_current_prepared_partition_lineage(prepared)
    finally:
        object.__setattr__(vertex, 'position', original)


@pytest.mark.parametrize('operand', ['source', 'prepared'])
def test_readiness_refuses_revision_counter_without_an_owner_commit(operand):
    source, prepared, _pair = genuine_partition()
    model = source if operand == 'source' else prepared
    revision = model.revision
    try:
        model._revision += 1
        with pytest.raises(GeometryError, match='stale'):
            has_current_prepared_partition_lineage(prepared)
    finally:
        model._revision = revision


@pytest.mark.parametrize('advanced', [None, 'source', 'prepared'])
@pytest.mark.parametrize('which', ['advanced', 'other'])
def test_readiness_preserves_native_refusal_even_when_other_operand_was_edited(advanced, which):
    from anygeometry import Cylinder
    from test_prepared_material_source_transfer import prepared_square
    source, prepared, _regions, _chords, _cell = prepared_square(unselected_cylinder=True)
    if advanced is not None:
        (source if advanced == 'source' else prepared).add_point(100, 100, 100)
    poisoned = source if advanced in (None, 'source') else prepared
    if which == 'other':
        poisoned = prepared if poisoned is source else source
    support = next(face.surface for face in poisoned.faces.values() if isinstance(face.surface, Cylinder))
    original = support._circumferential
    before = to_dict(poisoned)
    try:
        object.__setattr__(support, '_circumferential', original.copy() * (1 + 2**-40))
        assert to_dict(poisoned) == before
        with pytest.raises(GeometryError):
            has_current_prepared_partition_lineage(prepared)
    finally:
        object.__setattr__(support, '_circumferential', original)


@pytest.mark.parametrize('operand', ['source', 'prepared'])
def test_readiness_refuses_raw_current_content_after_a_legitimate_revision(operand):
    source, prepared, _pair = genuine_partition()
    model = source if operand == 'source' else prepared
    model.add_point(100, 100, 100)
    face = next(iter(model.faces))
    original = model._faces[face]
    try:
        model._faces[face] = replace(original, metadata={'raw_after_commit': True})
        with pytest.raises(GeometryError):
            has_current_prepared_partition_lineage(prepared)
    finally:
        model._faces[face] = original


def test_readiness_refuses_raw_mutation_by_a_commit_observer():
    source, prepared, _pair = genuine_partition()
    face = next(iter(prepared.faces))
    def raw_observer(_change):
        prepared._faces[face] = replace(prepared.faces[face], metadata={'raw_observer': True})
    prepared.add_change_hook(raw_observer)
    prepared.add_point(100, 100, 100)
    with pytest.raises(GeometryError):
        has_current_prepared_partition_lineage(prepared)


@pytest.mark.parametrize('operand', ['source', 'prepared'])
@pytest.mark.parametrize('publication', ['transaction', 'feature', 'settings', 'topology', 'design'])
def test_complete_commit_witness_covers_each_owner_publication_path(operand, publication):
    import anygeometry.prepared_material_source_transfer as transport
    source, prepared, _pair = genuine_partition()
    model = source if operand == 'source' else prepared
    if publication == 'transaction':
        model.add_point(100, 100, 100)
    elif publication == 'feature':
        model.features.append('vendor.optional', suppressed=True)
    elif publication == 'settings':
        model.set_document_settings(local_origin=(1., 0., 0.))
    else:
        candidate = model.clone()
        candidate.add_point(100, 100, 100)
        if publication == 'topology':
            model.restore_topology(candidate.topology_snapshot())
        else:
            model.restore_design(candidate.design_snapshot())
    committed, failure = transport._PARTITION_COMMITS[model]
    assert failure is None and committed == transport._state(model)
    assert has_current_prepared_partition_lineage(prepared) is False
    face = next(iter(model.faces))
    model._faces[face] = replace(model.faces[face], metadata={'raw_after_publication': publication})
    with pytest.raises(GeometryError, match='commit witness'):
        has_current_prepared_partition_lineage(prepared)


def test_read_and_reclone_do_not_overwrite_an_existing_commit_witness():
    import anygeometry.prepared_material_source_transfer as transport
    source, prepared, _pair = genuine_partition()
    source.add_point(100, 100, 100)
    witness = transport._PARTITION_COMMITS[source]
    face = next(iter(source.faces))
    source._faces[face] = replace(source.faces[face], metadata={'raw_before_reclone': True})
    to_dict(source)
    cloned = clone_prepared_geometry(source, new_preparation_epoch=True)
    assert cloned.faces and transport._PARTITION_COMMITS[source] is witness
    with pytest.raises(GeometryError, match='commit witness'):
        has_current_prepared_partition_lineage(prepared)


def test_commit_witness_registry_does_not_retain_models():
    import gc
    from weakref import ref
    import anygeometry.prepared_material_source_transfer as transport
    source, prepared, pair = genuine_partition()
    assert source in transport._PARTITION_COMMITS and prepared in transport._PARTITION_COMMITS
    source_ref, prepared_ref = ref(source), ref(prepared)
    del source, prepared, pair
    gc.collect()
    assert source_ref() is None and prepared_ref() is None


def test_untracked_owner_edits_never_capture_partition_state(monkeypatch):
    from anygeometry import GeometryModel
    import anygeometry.prepared_material_source_transfer as transport
    model = GeometryModel()
    def forbidden(_model):
        raise AssertionError('untracked edit captured lineage state')
    monkeypatch.setattr(transport, '_state', forbidden)
    model.add_point(1, 2, 3)
    model.features.append('vendor.optional', suppressed=True)
    model.set_document_settings(local_origin=(1., 0., 0.))
    candidate = model.clone()
    candidate.add_point(4, 5, 6)
    model.restore_topology(candidate.topology_snapshot())
    candidate.add_point(7, 8, 9)
    model.restore_design(candidate.design_snapshot())
    assert model not in transport._PARTITION_COMMITS
    assert not getattr(model, '_prepared_partition_lineage_tracked', False)


def test_failed_commit_capture_preserves_commit_and_records_unusable_reason(monkeypatch):
    import anygeometry.prepared_material_source_transfer as transport
    source, prepared, _pair = genuine_partition()
    original = transport._state
    error = GeometryError('forced owner commit witness capture failure')
    observed = []
    def capture(model):
        if model is prepared and model._notifying_hooks:
            raise error
        return original(model)
    monkeypatch.setattr(transport, '_state', capture)
    prepared.add_change_hook(lambda change: observed.append(change))
    point = prepared.add_point(100, 100, 100)
    assert point in prepared.vertices and observed
    state, reason = transport._PARTITION_COMMITS[prepared]
    assert state is None and 'forced owner commit witness capture failure' in reason
    with pytest.raises(GeometryError, match='commit witness'):
        has_current_prepared_partition_lineage(prepared)


@pytest.mark.parametrize('malformed', ['row_shape', 'duplicate_children'])
@pytest.mark.parametrize('obsolete', [False, True])
def test_readiness_does_not_downgrade_malformed_live_authority(malformed, obsolete):
    from anygeometry.prepared_material_source_transfer import _LINKS
    source, prepared, _pair = genuine_partition()
    if obsolete:
        prepared.add_point(100, 100, 100)
    original = _LINKS[prepared]
    row = original[0]
    broken = row[:-1] if malformed == 'row_shape' else (
        *row[:3], ((row[3][0][0], row[3][0][1] * 2),), row[4])
    try:
        _LINKS[prepared] = (broken,)
        with pytest.raises(GeometryError, match='malformed'):
            has_current_prepared_partition_lineage(prepared)
    finally:
        _LINKS[prepared] = original


def test_readiness_grants_no_exemption_from_later_callback_mutation_or_cancellation():
    source, prepared, pair = genuine_partition()
    assert has_current_prepared_partition_lineage(prepared)
    signal = CancellationSignal(); signal.cancel()
    with pytest.raises(GeometryError, match='cancelled'):
        query_prepared_partition_disjoint_interiors(prepared, *pair, cancellation_check=signal)
    before = to_dict(prepared)
    def edit(phase):
        if phase == 'prepared partition relation complete':
            prepared.add_point(100, 100, 100)
        return False
    with pytest.raises(GeometryError, match='stale|changed'):
        query_prepared_partition_disjoint_interiors(prepared, *pair, cancellation_check=edit)
    assert prepared.revision > before['revision']  # The caller edit happened; its proof was refused.
