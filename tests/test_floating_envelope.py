import numpy as np
import pytest

from anygeometry import floating_envelope as fe
from anygeometry.errors import GeometryError


class FakeGeometry:
    """Minimal public-surface stand-in (vertex positions + face corner ids) for pure topology tests."""
    def __init__(self, verts, faces):
        self.v, self.f = verts, faces
    def vertex_position(self, v):
        return np.array(self.v[v], float)
    def face_corner_vertices(self, f):
        return tuple(self.f[f])


CUBE = {0: (0, 0, 0), 1: (2, 0, 0), 2: (2, 2, 0), 3: (0, 2, 0), 4: (0, 0, 2), 5: (2, 0, 2), 6: (2, 2, 2), 7: (0, 2, 2)}
FACES = {10: (0, 1, 2, 3), 11: (4, 5, 6, 7), 12: (0, 1, 5, 4), 13: (1, 2, 6, 5), 14: (2, 3, 7, 6), 15: (3, 0, 4, 7)}


def test_envelope_is_oriented_outward_and_identity_bound_regardless_of_input_orientation():
    mixed = dict(FACES); mixed[12] = FACES[12][::-1]; mixed[14] = FACES[14][::-1]
    a = fe.build_envelope(FakeGeometry(CUBE, FACES), list(FACES))
    b = fe.build_envelope(FakeGeometry(CUBE, mixed), list(FACES))
    assert a['volume'] == pytest.approx(8.0) and b['volume'] == pytest.approx(8.0)
    assert a['faces'] == b['faces'] and a['identity'] == b['identity']      # propagated orientation is canonical


def test_open_shell_missing_face_and_nonplanar_are_refused_not_repaired():
    with pytest.raises(GeometryError, match='not closed'):
        fe.build_envelope(FakeGeometry(CUBE, FACES), [10, 12, 13, 14, 15])
    bent = dict(CUBE); bent[6] = (2, 2, 2.5)
    with pytest.raises(GeometryError, match='planar'):
        fe.build_envelope(FakeGeometry(bent, FACES), list(FACES))


def test_coordinate_coincident_but_distinct_vertices_are_not_welded():
    verts = dict(CUBE); verts[8] = verts[0]
    faces = dict(FACES); faces[15] = (3, 8, 4, 7)
    with pytest.raises(GeometryError, match='not closed'):
        fe.build_envelope(FakeGeometry(verts, faces), list(FACES))
