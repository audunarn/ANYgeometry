"""Live complete owner-partition world-disjointness regressions; owner UNRUN."""
from dataclasses import replace
import pytest

from anygeometry import (GeometryError, Plane, clone_prepared_geometry,
    plan_plane_split, apply_plane_split, query_prepared_partition_disjoint_interiors,
    CancellationSignal, to_dict, from_dict)


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
