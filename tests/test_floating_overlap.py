import numpy as np
import pytest

from anygeometry import floating_overlap as ov
from anygeometry.errors import GeometryError

SQ = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], float)     # normal +z


def test_wet_clip_keeps_orientation_and_exact_area():
    side = np.array([[0, 0, -1], [4, 0, -1], [4, 0, 1], [0, 0, 1]], float)   # normal -y
    wet = ov.wet_polygon(side, 0.0)
    assert wet is not None and ov.polygon_area(wet) == pytest.approx(4.0)
    assert np.allclose(ov.area_vector(wet) / 4.0, ov.area_vector(side) / ov.polygon_area(side))
    assert ov.wet_polygon(side + [0, 0, 5], 0.0) is None


def test_panelize_conserves_area_and_orientation():
    pieces = ov.panelize(SQ, 0.7)
    assert sum(ov.polygon_area(p) for p in pieces) == pytest.approx(4.0, rel=1e-12)
    assert all(ov.polygon_plane(p)[0] @ [0, 0, 1] > 1 - 1e-12 for p in pieces)
    with pytest.raises(GeometryError):
        ov.panelize(SQ, 0.0)


def test_overlap_matrix_is_exactly_conservative_for_nonmatching_partitions():
    src = [np.array(t) for t in ov.fan_triangles(SQ)]                      # 2 triangles
    tgt = ov.panelize(SQ, 0.5)                                             # 16 cells, none aligned with the diagonal
    r, c, a = ov.overlap_matrix(src, tgt)
    rep = ov.coverage_report([ov.polygon_area(s) for s in src], [ov.polygon_area(t) for t in tgt], r, c, a)
    assert rep["source_uncovered_rel"] < 1e-12 and rep["target_unmatched_rel"] < 1e-12
    # force conservation: sum over targets of (overlap * p_source) equals sum of source panel force
    p = np.array([3.0, -1.0])
    mapped = np.zeros(len(tgt)); np.add.at(mapped, c, a * p[r])
    assert mapped.sum() == pytest.approx(sum(p[i] * ov.polygon_area(s) for i, s in enumerate(src)), rel=1e-12)


def test_overlap_zero_for_other_plate_opposite_orientation_or_offset_plane():
    tri = np.array(ov.fan_triangles(SQ)[0])
    assert ov.overlap_area(tri, SQ[::-1]) == 0.0
    assert ov.overlap_area(tri, SQ + [0, 0, 0.1]) == 0.0
    wall = np.array([[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]], float)
    assert ov.overlap_area(tri, wall) == 0.0


def test_nonplanar_polygon_is_refused():
    bad = SQ.copy(); bad[2, 2] = 0.5
    with pytest.raises(GeometryError):
        ov.polygon_plane(bad)
