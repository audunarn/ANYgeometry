"""Source-state reuse retains complete content, incidence and actual native checks."""
import pytest

from anygeometry import GeometryError, to_dict
from anygeometry import serialization
from anygeometry import prepared_material_source_transfer as transport
from test_prepared_material_curvilinear import square_cells


def test_repeated_state_reads_qualify_once_but_encode_and_capture_every_time(monkeypatch):
    model, *_ = square_cells()
    serialization._qualified_documents.pop(model, None)
    calls = {'qualify': 0, 'encode': 0, 'native': 0}
    for module, name, key in (
        (serialization, '_qualify_document', 'qualify'),
        (serialization, '_serialized_model_state', 'encode'),
        (transport, 'capture_native_supports', 'native'),
    ):
        original = getattr(module, name)
        def measured(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, measured)
    states = []
    for _ in range(4):
        encoded = calls['encode']
        states.append(transport._state(model))
        assert calls['encode'] > encoded
    assert all(state == states[0] for state in states)
    assert calls['qualify'] == 1
    assert calls['native'] == 4
    # Public serialization still performs its original qualification.
    to_dict(model)
    to_dict(model)
    assert calls['qualify'] == 3


def test_transaction_refused_even_with_warm_qualification():
    model, *_ = square_cells()
    transport._state(model)
    with model.transaction():
        with pytest.raises(GeometryError, match='committed'):
            transport._state(model)


def test_raw_same_revision_vertex_change_cannot_reuse_qualification():
    model, *_ = square_cells()
    transport._state(model)
    revision = model.revision
    vertex = next(iter(model.vertices.values()))
    original = vertex.position
    changed = original.copy()
    changed[2] += 1.
    try:
        object.__setattr__(vertex, 'position', changed)
        assert model.revision == revision
        with pytest.raises(GeometryError):
            transport._state(model)
    finally:
        object.__setattr__(vertex, 'position', original)


@pytest.mark.parametrize('malformed', (False, True))
def test_reverse_incidence_is_checked_on_warm_state(malformed):
    model, *_ = square_cells()
    face = next(iter(model.faces))
    model.add_sheet((face,))  # Geometric faces alone do not issue coedges.
    assert model._edge_coedges
    assert all(model._edge_coedges[use.edge] for use in model.faces[face].loop)
    transport._state(model)
    original = model._edge_coedges
    changed = [] if malformed else dict(original)
    if not malformed:
        changed[next(iter(changed))] = {999999999}
    try:
        object.__setattr__(model, '_edge_coedges', changed)
        with pytest.raises(GeometryError):
            transport._state(model)
    finally:
        object.__setattr__(model, '_edge_coedges', original)


def test_actual_curve_runtime_is_checked_after_warm_qualification():
    model, *_ = square_cells()
    curve = next(iter(model.edges.values())).curve.pcurve
    curve.evaluate(.375)
    transport._state(model)
    original = curve._point_values
    key, value = next(iter(original.items()))
    poisoned = value.copy()
    poisoned.flat[0] += .25
    try:
        object.__setattr__(curve, '_point_values', {**original, key: poisoned})
        with pytest.raises(GeometryError):
            transport._state(model)
    finally:
        object.__setattr__(curve, '_point_values', original)


def test_nonserialized_native_coefficients_remain_in_every_state():
    from anygeometry.surfaces import Cylinder
    from test_cylinder_ruling_batch import panel
    model, *_ = square_cells()
    support, loop = panel(model, (10., 0., 0.), (3., 4.))
    model.add_face_from_loop(loop, surface=support)
    assert type(support) is Cylinder
    state = transport._state(model)
    checksum = serialization._serialized_model_state(model)['checksum']['value']
    original = support._circumferential
    try:
        object.__setattr__(support, '_circumferential', original.copy() * (1 + 2**-40))
        assert serialization._serialized_model_state(model)['checksum']['value'] == checksum
        # This snapshot is evidence, not a validator receipt: the changed
        # native state must be visible to the enclosing binding comparison.
        assert transport._state(model) != state
    finally:
        object.__setattr__(support, '_circumferential', original)
