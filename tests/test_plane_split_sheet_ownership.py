"""Sheet/FaceUse correspondence through public plan_plane_split / apply_plane_split (UNRUN definitions).

The closed-loop bowl cut (disk + outer region) and its known second vertical cut come from
tests/test_plane_split.py. The transfer itself is the model's existing structural replacement; these tests
prove it is reached, bound by the plan, rolled back with publication, and that unsupported ownership refuses
before the caller changes.
"""
from dataclasses import replace

import pytest

from anygeometry import GeometryError, Plane, apply_plane_split, plan_plane_split
from anygeometry.serialization import from_dict, to_dict
from test_plane_split import bowl


def sheet_model(orientation=1, metadata=False):
    model, face, plane = bowl()
    sheet = model.add_sheet((face,), orientations=(orientation,))
    if metadata:                                  # meaningful, nonempty structural metadata through the owner store
        with model.transaction():
            part_id = model.sheets[sheet].part_id
            model._put_structural('part', replace(model.parts[part_id], name='wall assembly',
                                                  metadata={'role': 'assembly'}))
            model._put_structural('sheet', replace(model.sheets[sheet], name='skin', metadata={'layer': 2}))
            use_id = next(iter(model.face_uses))
            model._put_structural('face_use', replace(model.face_uses[use_id], metadata={'use': 'outer'}))
    return model, face, plane, sheet


def state_of(model):
    """The exact non-numpy state of the topology snapshot (a deterministic capture of the caller)."""
    snap = model.topology_snapshot()
    return (model.revision, snap['structural'], snap['structural_ids'], snap['ids'],
            tuple(sorted((k, tuple(v.tolist())) for k, v in snap['vertex_state'].items())),
            tuple(sorted((k, v[0], v[1]) for k, v in snap['edge_state'].items())),
            tuple(sorted((k, v[0], v[1], v[2], v[3]) for k, v in snap['face_state'].items())),
            tuple(sorted(snap['groups'])), tuple(sorted(map(str, snap['tags']))))


def structural_ok(model):
    assert not model.validate_topology() and not model._validate_structural()
    for use in model.face_uses.values():
        assert use.face_id in model.faces
        assert all(len(loop) >= 3 for loop in use.loops)                  # every loop keeps >= 3 coedges
    owned = sorted(use.face_id for use in model.face_uses.values())
    assert owned == sorted(model.faces)                                    # one FaceUse per face, none dropped


@pytest.mark.parametrize('orientation', (1, -1))
def test_closed_loop_split_transfers_sheet_uses_with_identities_orientation_and_metadata(orientation):
    model, face, plane, sheet = sheet_model(orientation, metadata=True)
    old_use_id, old_use = next(iter(model.face_uses.items()))
    assert old_use.metadata.to_dict() == {'use': 'outer'}                  # the fixture really is nonempty
    assert model.sheets[sheet].metadata.to_dict() == {'layer': 2} and model.sheets[sheet].name == 'skin'
    part_before, sheet_before = dict(model.parts), model.sheets[sheet]
    plan = plan_plane_split(model, [face], plane)
    result = apply_plane_split(model, plan, policy='imprint')
    children = result.parent_children[0][1]
    assert len(children) == 2 and face not in model.faces
    structural_ok(model)
    assert set(model.sheets) == {sheet} and model.sheets[sheet].part_id == sheet_before.part_id
    assert {k: v for k, v in model.parts.items()} == part_before          # part identity and definition kept
    assert replace(model.sheets[sheet], face_use_ids=sheet_before.face_use_ids) == sheet_before
    assert set(model.sheets[sheet].face_use_ids) == set(model.face_uses)
    uses = {use.face_id: use for use in model.face_uses.values()}
    assert uses[children[0]].id == old_use_id                              # the first child keeps the original use
    assert {use.orientation for use in uses.values()} == {old_use.orientation}
    assert all(use.metadata.to_dict() == {'use': 'outer'} for use in uses.values())
    assert all(use.sheet_id == sheet for use in uses.values())
    assert model.sheets[sheet].name == 'skin' and model.sheets[sheet].metadata.to_dict() == {'layer': 2}
    assert model.parts[sheet_before.part_id].name == 'wall assembly'
    assert model.parts[sheet_before.part_id].metadata.to_dict() == {'role': 'assembly'}


def test_save_and_load_reconstruct_the_structural_tables_and_a_second_cut_continues_the_transfer():
    model, face, plane, sheet = sheet_model(metadata=True)
    apply_plane_split(model, plan_plane_split(model, [face], plane), policy='imprint')
    document = to_dict(model)
    loaded = from_dict(document)
    assert to_dict(loaded)['structural'] == document['structural']       # persisted tables verified first
    structural_ok(loaded)
    vertical = Plane((.2, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    result = apply_plane_split(loaded, plan_plane_split(loaded, tuple(loaded.faces), vertical), policy='imprint')
    assert len(loaded.faces) == 4 and len(result.parent_children) == 2    # the RELOADED model was cut
    structural_ok(loaded)
    assert set(loaded.sheets) == {sheet} and len(loaded.face_uses) == 4
    assert all(use.metadata.to_dict() == {'use': 'outer'} for use in loaded.face_uses.values())
    assert to_dict(model) == document                                      # the original was not touched


def test_same_revision_raw_sheet_edit_makes_the_plan_stale():
    model, face, plane, sheet = sheet_model()
    plan = plan_plane_split(model, [face], plane)
    revision = model.revision
    model._structural_store('sheet')[sheet] = replace(model.sheets[sheet], name='raw edit')
    assert model.revision == revision
    before = to_dict(model)
    with pytest.raises(GeometryError, match='stale'):
        apply_plane_split(model, plan, policy='imprint')
    assert to_dict(model) == before


class Cancelled(BaseException):
    """A cancellation-style failure that is not an ``Exception``."""


@pytest.mark.parametrize('failure', (RuntimeError('injected publication failure'), Cancelled()))
def test_publication_failure_rolls_back_geometry_and_structural_tables(monkeypatch, failure):
    import anygeometry.prepared_material_source_transfer as transfer
    model, face, plane, sheet = sheet_model(metadata=True)
    before, links = state_of(model), transfer._LINKS.get(model)
    document = to_dict(model)
    plan = plan_plane_split(model, [face], plane)
    assert state_of(model) == before                                       # planning never touches the caller

    def failing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(transfer, '_publish_source_lineage', failing)
    with pytest.raises(type(failure)) as raised:
        apply_plane_split(model, plan, policy='imprint')
    assert raised.value is failure                                         # the identical exception propagates
    assert state_of(model) == before and to_dict(model) == document        # tables, revision and IDs unchanged
    assert face in model.faces and len(model.face_uses) == 1 and transfer._LINKS.get(model) is links
    monkeypatch.undo()
    apply_plane_split(model, plan, policy='imprint')                       # the unchanged plan still applies
    structural_ok(model)


def test_two_structural_owners_refuse_before_any_caller_change():
    """The second sheet makes the document intentionally invalid (``to_dict`` itself refuses), so the exact
    non-qualifying state is captured with the public topology snapshot instead."""
    model, face, plane, sheet = sheet_model()
    model.add_sheet((face,))                                               # two FaceUses now own the face
    assert len(model._face_structural_uses[face]) == 2
    with pytest.raises(GeometryError):
        to_dict(model)                                                     # the invalid input is real
    before = state_of(model)
    with pytest.raises(GeometryError, match='multiple structural owners'):
        plan_plane_split(model, [face], plane)
    assert state_of(model) == before


def test_members_still_refuse_before_any_caller_change():
    model, face, plane, sheet = sheet_model()
    model.add_member([model.faces[face].loop[0].edge])
    before = state_of(model)
    with pytest.raises(GeometryError, match='structural reference remapping'):
        plan_plane_split(model, [face], plane)
    assert state_of(model) == before


def test_unowned_face_split_is_unchanged_by_the_sheet_gate():
    model, face, plane = bowl()
    result = apply_plane_split(model, plan_plane_split(model, [face], plane), policy='imprint')
    assert len(model.faces) == 2 and not model.face_uses and not model.sheets and result.trace_edges
