"""Live owner qualification of a renumbered ``ModelClosure`` as the immediate source route (UNRUN).

Real flat NURBS square patch with unrelated geometry prepended (so the closure renumbers vertices, edges
and the face), a Sheet, and the real owner cell/transfer receipts. Nothing is mocked.
"""
from dataclasses import replace
from fractions import Fraction as Q
from types import MappingProxyType

import pytest

from anygeometry import (GeometryError, GeometryModel, CurvilinearMaterialCell, extract_model_closure,
    model_closure_from_dict, model_closure_to_dict, qualify_model_closure_source, query_material_surface_regions,
    query_prepared_material_boundary_chords, validate_prepared_material_curvilinear_cells, query_prepared_material_source_transfer)
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.serialization import to_dict
import anygeometry.prepared_material_source_transfer as transfer

UV = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))


class Cancelled(BaseException):
    """A cancellation-style exception that is not an ``Exception``."""


def source_fixture(orientation=None):
    model = GeometryModel()
    a, b = model.add_points(((9., 9., 9.), (8., 8., 8.)))
    model.add_line(a, b)                                          # unrelated geometry: ids shift
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    vertices = model.add_points(tuple((*p, 0.) for p in UV))
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, UV[(i + 1) % 4])))) for i, p in enumerate(UV)]
    face = model.add_face(edges, surface=support)
    if orientation is None:
        model.add_sheet((face,))
    else:
        model.add_sheet((face,), orientations=(orientation,))
    return model, face, edges


def closure_fixture(orientation=None):
    source, face, edges = source_fixture(orientation)
    closure = extract_model_closure(source, [source.handle('face', face)])
    return source, face, edges, closure


def work_receipt(work):
    """The real owner chord and cell receipts on the (renumbered) working model."""
    wface = next(iter(work.faces))
    regions = query_material_surface_regions(work, [wface])
    loop = [use.edge for use in work.faces[wface].loop]
    chords = query_prepared_material_boundary_chords(work, regions,
        tuple((work.handle('edge', edge), 0., 1.) for edge in loop), chord_error=.01)
    xyz = tuple((*p, 0.) for p in UV)
    cells = (CurvilinearMaterialCell(work.handle('face', wface), (0, 1, 2, 3), UV, xyz,
                                     tuple((i, i, True) for i in range(4))),)
    return validate_prepared_material_curvilinear_cells(work, regions, chords, cells, surface_error=.001)


def no_route(source, work, receipt):
    with pytest.raises(GeometryError):
        query_prepared_material_source_transfer(source, work, (receipt,))


def test_closure_is_actually_renumbered():
    source, face, edges, closure = closure_fixture()
    wface = next(iter(closure.working_model.faces))
    assert wface != face or set(closure.working_model.edges) != set(edges)


def test_qualified_closure_gives_real_source_transfer_receipt_with_correct_faces_edges_and_stations():
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    receipt = work_receipt(work)
    no_route(source, work, receipt)                               # no prior qualification: no authority
    assert qualify_model_closure_source(source, closure) is closure
    transferred = query_prepared_material_source_transfer(source, work, (receipt,))
    wface = next(iter(work.faces))
    assert transferred.source_to_prepared_faces == ((source.handle('face', face), (work.handle('face', wface),)),)
    assert {row[1].id for row in transferred.edge_parameter_maps} == set(edges)
    assert all((row[2], row[3]) == (Q(0), Q(1)) for row in transferred.edge_parameter_maps)
    assert {c.face.id for c in transferred.source_cells} == {face}
    for edge_row in transferred.edge_parameter_maps:                 # prepared edge maps to the right source edge
        assert closure.work_to_source[work.handle('edge', edge_row[0].id)] == edge_row[1]
    assert transferred.boundary_station_correspondence


def test_exact_requalification_keeps_the_route_entry_and_previously_issued_receipts_valid():
    from anygeometry.prepared_material_source_transfer import validate_prepared_material_source_transfer_binding
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    qualify_model_closure_source(source, closure)
    first = transfer._LINKS[work]
    receipt = work_receipt(work)
    issued = query_prepared_material_source_transfer(source, work, (receipt,))
    qualify_model_closure_source(source, closure)                   # full independent proof again
    assert transfer._LINKS[work] is first and len(first) == 1       # identity preserved, nothing accumulated
    validate_prepared_material_source_transfer_binding(source, work, issued, cell_receipts=(receipt,))


@pytest.mark.parametrize('field, value', (
    ('source_handles', (1,)), ('source_handles', [1]), ('source_handles', 'x'),
    ('source_to_work', []), ('work_to_source', []), ('work_to_source', None)))
def test_malformed_closure_fields_raise_a_typed_error_before_any_callback(field, value):
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    bad = replace(closure, **{field: value})
    phases = []
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, bad, cancellation_check=lambda phase: phases.append(phase))
    assert phases == [] and transfer._LINKS.get(work) is None


def test_wrong_model_selection_handle_raises_a_typed_error():
    source, face, edges, closure = closure_fixture()
    other, other_face, _edges = source_fixture()
    bad = replace(closure, source_handles=(other.handle('face', other_face),))
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, bad)
    assert transfer._LINKS.get(closure.working_model) is None


def test_forged_maps_and_selection_are_refused_and_leave_no_lineage():
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    items = sorted(closure.work_to_source.items())
    vertices = [(w, s) for w, s in items if w.kind == 'vertex']
    (w0, s0), (w1, s1) = vertices[:2]
    forged = dict(closure.work_to_source)
    forged[w0], forged[w1] = s1, s0                               # swapped identities
    bad = replace(closure, work_to_source=MappingProxyType(forged),
                  source_to_work=MappingProxyType({s: w for w, s in forged.items()}))
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, bad)
    wrong_selection = replace(closure, source_handles=())
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, wrong_selection)
    assert transfer._LINKS.get(work) is None


def test_edited_work_public_or_native_same_state_is_refused():
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    work.add_point(5., 5., 5.)                                    # public edit
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, closure)
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    curve = next(iter(work.edges.values())).curve.pcurve
    other = replace(curve, weights=(1., 2.))
    prepared = curve._prepared
    object.__setattr__(curve, '_prepared', other._prepared)       # native-only, same revision/serialization
    try:
        with pytest.raises(GeometryError):
            qualify_model_closure_source(source, closure)
    finally:
        object.__setattr__(curve, '_prepared', prepared)
    assert transfer._LINKS.get(work) is None


def test_stale_and_wrong_source_are_refused():
    source, face, edges, closure = closure_fixture()
    other, *_ = source_fixture()
    with pytest.raises(GeometryError):
        qualify_model_closure_source(other, closure)
    source.add_point(5., 5., 5.)
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, closure)


def test_cancellation_and_callback_mutation_are_atomic_and_keep_prior_lineage():
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    qualify_model_closure_source(source, closure)
    prior = transfer._LINKS[work]
    before_source, before_work = to_dict(source), to_dict(work)
    with pytest.raises(GeometryError, match='cancelled'):
        qualify_model_closure_source(source, closure,
            cancellation_check=lambda phase: phase == 'closure source qualification complete')
    assert transfer._LINKS[work] is prior and to_dict(source) == before_source and to_dict(work) == before_work
    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, closure, cancellation_check=lambda phase: source.add_point(7., 7., 7.) and False)
    assert transfer._LINKS[work] is prior


def test_transport_loses_authority_and_independent_requalification_restores_it():
    source, face, edges, closure = closure_fixture()
    qualify_model_closure_source(source, closure)
    loaded = model_closure_from_dict(model_closure_to_dict(closure))
    receipt = work_receipt(loaded.working_model)
    no_route(source, loaded.working_model, receipt)               # a loaded closure holds no authority
    qualify_model_closure_source(source, loaded)
    transferred = query_prepared_material_source_transfer(source, loaded.working_model, (receipt,))
    assert {row[1].id for row in transferred.edge_parameter_maps} == set(edges)

@pytest.mark.parametrize('phase', ('closure source qualification', 'closure source reconstruction',
                                   'closure source qualification complete'))
def test_working_model_object_substitution_at_any_callback_grants_no_lineage(phase):
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    before_source, before_work = to_dict(source), to_dict(work)
    impostor = work.clone(preserve_identity=True)                  # same UUID and equal definitions
    assert impostor is not work and impostor.model_id == work.model_id

    def callback(current):
        if current == phase:
            object.__setattr__(closure, 'working_model', impostor)
        return False

    with pytest.raises(GeometryError):
        qualify_model_closure_source(source, closure, cancellation_check=callback)
    assert transfer._LINKS.get(work) is None and transfer._LINKS.get(impostor) is None
    assert to_dict(source) == before_source and to_dict(work) == before_work


def test_qualified_working_model_composes_through_a_genuine_prepared_clone():
    from anygeometry import clone_prepared_geometry
    source, face, edges, closure = closure_fixture()
    qualify_model_closure_source(source, closure)
    clone = clone_prepared_geometry(closure.working_model)         # genuine owner clone boundary
    receipt = work_receipt(clone)
    transferred = query_prepared_material_source_transfer(source, clone, (receipt,))
    assert transferred.source_to_prepared_faces[0][0] == source.handle('face', face)
    assert {row[1].id for row in transferred.edge_parameter_maps} == set(edges)


@pytest.mark.parametrize('failure', (RuntimeError('injected'), Cancelled()))
@pytest.mark.parametrize('with_prior', (False, True))
def test_failure_after_actual_publication_restores_the_exact_prior_lineage(monkeypatch, failure, with_prior):
    source, face, edges, closure = closure_fixture()
    work = closure.working_model
    if with_prior:
        qualify_model_closure_source(source, closure)
    prior = transfer._LINKS.get(work)
    before_source, before_work = to_dict(source), to_dict(work)
    real = transfer._register_source_closure

    def publish_then_fail(*args):
        real(*args)                                            # the actual registration runs ...
        if not with_prior:
            assert transfer._LINKS.get(work) is not prior      # ... and publishes (an exact prior is kept)
        raise failure                                          # ... and then fails or is cancelled

    monkeypatch.setattr(transfer, '_register_source_closure', publish_then_fail)
    with pytest.raises(type(failure)) as raised:
        qualify_model_closure_source(source, closure)
    assert raised.value is failure                             # the original exception is not masked
    assert transfer._LINKS.get(work) is prior                  # exact prior object (or still absent)
    assert to_dict(source) == before_source and to_dict(work) == before_work


@pytest.mark.parametrize('orientation', (1, -1))
def test_face_use_and_coedge_orientation_correspond_for_forward_and_reversed_sheets(orientation):
    source, face, edges, closure = closure_fixture(orientation)
    qualify_model_closure_source(source, closure)
    work = closure.working_model
    (src_id, src_use), (work_id, work_use) = next(iter(source.face_uses.items())), next(iter(work.face_uses.items()))
    assert replace(work_use, id=src_use.id, sheet_id=src_use.sheet_id, face_id=src_use.face_id,
                   loops=src_use.loops) == src_use
    assert len(work.coedges) == len(source.coedges) == 4
    for work_coedge_id, work_coedge in work.coedges.items():
        source_handle = closure.work_to_source[work.handle('coedge', work_coedge_id)]
        source_coedge = source.coedges[source_handle.id]
        assert replace(work_coedge, id=source_coedge.id, face_use_id=source_coedge.face_use_id,
                       edge_id=source_coedge.edge_id) == source_coedge       # forward flag etc. preserved


def test_face_use_and_sheet_definitions_are_corresponded_by_the_closure_not_inferred():
    source, face, edges, closure = closure_fixture()
    qualify_model_closure_source(source, closure)
    work = closure.working_model
    assert len(source.sheets) == len(work.sheets) == 1 and len(source.face_uses) == len(work.face_uses) == 1
    (src_id, src_use), (work_id, work_use) = next(iter(source.face_uses.items())), next(iter(work.face_uses.items()))
    # every field other than the renumbered ids/loops (including orientation) must be equal
    assert replace(work_use, id=src_use.id, sheet_id=src_use.sheet_id, face_id=src_use.face_id,
                   loops=src_use.loops) == src_use
    assert closure.work_to_source[work.handle('face_use', work_id)] == source.handle('face_use', src_id)
