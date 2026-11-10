"""Tests for anygeometry.shell_detail (written without being executed in the authoring session)."""
import copy

import numpy as np
import pytest

from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.operations import punch_hole, trim_face
from anygeometry.serialization import to_dict
from anygeometry.shell_detail import ShellDetailError, derive_shell_detail_box
from anygeometry.surfaces import Plane


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class Frame:
    """A local (x, y, normal) frame placed in space."""

    def __init__(self, rotation=None, shift=(0.0, 0.0, 0.0)):
        self.R = np.eye(3) if rotation is None else rotation
        self.shift = np.asarray(shift, dtype=float)
        self.normal = self.R @ np.array([0.0, 0.0, 1.0])

    def place(self, xy):
        return self.R @ np.array([xy[0], xy[1], 0.0]) + self.shift

    def points(self, xys):
        return np.array([self.place(p) for p in xys])


def plate(frame, outer, holes=()):
    """One planar face, outer loop counter-clockwise in the local frame."""

    model = GeometryModel()

    def loop(xys):
        ids = [model.add_point(*frame.place(p)) for p in xys]
        return [model.add_line(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]

    support = Plane(frame.place(outer[0]), frame.R @ np.array([10.0, 0.0, 0.0]), frame.R @ np.array([0.0, 10.0, 0.0]))
    outer_edges = loop(outer)
    face = model.add_face_from_loop([OrientedEdge(e, True) for e in outer_edges], surface=support)
    if holes:
        # The owner's hole API is operations.trim_face(model, face, inner_loops).
        trim_face(model, face, [[OrientedEdge(e, True) for e in loop(h)] for h in holes])
    return model, face


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
RECT = [(3.0, 3.0), (6.0, 3.0), (6.0, 5.0), (3.0, 5.0)]  # counter-clockwise


FRAMES = {
    "axis": Frame(),
    "rotated": Frame(_rotation((1.0, 2.0, 3.0), 0.7), (1.5, -2.0, 4.0)),
}


@pytest.fixture(params=sorted(FRAMES))
def frame(request):
    return FRAMES[request.param]


def derive(model, face, frame, xys, **kwargs):
    kwargs.setdefault("director", frame.normal)
    kwargs.setdefault("thickness", 0.2)
    return derive_shell_detail_box(model, face, frame.points(xys), **kwargs)


def test_box_roles_offset_and_no_mutation(frame):
    model, face = plate(frame, SQUARE)
    revision, checksum = model.revision, to_dict(model)["checksum"]["value"]
    detail, region_id = derive(model, face, frame, RECT, reference_offset=0.25, thickness=0.1)
    assert (model.revision, to_dict(model)["checksum"]["value"]) == (revision, checksum)
    assert not model.solid_regions
    assert detail is not model and detail.model_id != model.model_id
    snapshot = detail.solid_region_snapshot(region_id)
    base = frame.points(RECT)
    heights = (np.asarray(snapshot.coordinates) - base[[0, 1, 2, 3, 0, 1, 2, 3]]) @ frame.normal
    np.testing.assert_allclose(heights, [0.2] * 4 + [0.3] * 4, atol=1e-12)
    assert snapshot.units == "m" and snapshot.volume == pytest.approx(3 * 2 * 0.1)
    assert set(snapshot.face_groups) >= {"start", "end", "side_0", "side_1", "side_2", "side_3"}


@pytest.mark.parametrize("clockwise", [False, True])
@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_side_i_stands_on_input_edge_i(frame, clockwise, shift):
    model, face = plate(frame, SQUARE)
    xys = list(reversed(RECT)) if clockwise else list(RECT)
    xys = xys[shift:] + xys[:shift]
    detail, region_id = derive(model, face, frame, xys)
    snapshot = detail.solid_region_snapshot(region_id)
    coords = np.asarray(snapshot.coordinates)
    footprint = frame.points(xys)
    for i in range(4):
        (group,) = snapshot.face_groups[f"side_{i}"]
        loop = snapshot.face_loops[list(snapshot.face_refs).index(group)]
        # Projected onto the shell plane, the side's corners are exactly input edge i's end points.
        bases = set()
        for r in loop:
            height = (coords[r] - footprint[0]) @ frame.normal
            bases.add(tuple(np.round(coords[r] - height * frame.normal, 9)))
        assert bases == {tuple(np.round(footprint[i], 9)), tuple(np.round(footprint[(i + 1) % 4], 9))}
    # The start and end faces are the two caps.
    (start,), (end,) = snapshot.face_groups["start"], snapshot.face_groups["end"]
    heights = {name: (coords[list(snapshot.face_loops[list(snapshot.face_refs).index(g)])] - footprint[0]) @ frame.normal
               for name, g in (("start", start), ("end", end))}
    np.testing.assert_allclose(heights["start"], -0.1, atol=1e-12)
    np.testing.assert_allclose(heights["end"], 0.1, atol=1e-12)


def test_deterministic_identity_and_lineage(frame):
    model, face = plate(frame, SQUARE)
    first = derive(model, face, frame, RECT)
    second = derive(model, face, frame, RECT)
    assert first[1] == second[1]
    other = derive(model, face, frame, RECT, reference_offset=0.5)
    assert other[1] != first[1]
    info = first[0].solid_regions[first[1]].provenance.to_dict()["shell_detail"]
    assert info["source_model_id"] == str(model.model_id) and info["face_id"] == face
    assert info["source_revision"] == model.revision
    assert info["source_checksum"] == to_dict(model)["checksum"]["value"]
    assert info["footprint"] == frame.points(RECT).tolist()
    assert len(info["footprint_signature"]) == 64


def test_units_must_be_metres():
    model, face = plate(Frame(), SQUARE)
    model.set_document_settings(units="mm")
    with pytest.raises(ShellDetailError) as error:
        derive_shell_detail_box(model, face, Frame().points(RECT), director=(0, 0, 1), thickness=0.2)
    assert error.value.code == "DETAIL_UNITS"


@pytest.mark.parametrize("director", [(0, 0, -1), (1, 0, 0), (0, 0, 0), (0, 0.01, 1)])
def test_director_must_match_authoritative_normal(director):
    model, face = plate(Frame(), SQUARE)
    with pytest.raises(GeometryError):
        derive_shell_detail_box(model, face, Frame().points(RECT), director=director, thickness=0.2)


@pytest.mark.parametrize("thickness", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_thickness_validated(thickness):
    model, face = plate(Frame(), SQUARE)
    with pytest.raises(GeometryError):
        derive_shell_detail_box(model, face, Frame().points(RECT), director=(0, 0, 1), thickness=thickness)


def test_footprint_must_be_rectangle_on_plane():
    f = Frame()
    model, face = plate(f, SQUARE)
    for bad in ([(3, 3), (6, 3), (6, 5), (3, 6)],  # trapezoid
                [(3, 3), (6, 5), (6, 3), (3, 5)],  # bow tie
                [(3, 3), (3, 3), (6, 5), (3, 5)]):  # repeated corner
        with pytest.raises(GeometryError):
            derive_shell_detail_box(model, face, f.points(bad), director=(0, 0, 1), thickness=0.2)
    lifted = f.points(RECT) + (0, 0, 0.5)
    with pytest.raises(GeometryError):
        derive_shell_detail_box(model, face, lifted, director=(0, 0, 1), thickness=0.2)
    with pytest.raises(GeometryError):
        derive_shell_detail_box(model, face, f.points(RECT)[:3], director=(0, 0, 1), thickness=0.2)


@pytest.mark.parametrize("xys", [
    [(-1, 3), (6, 3), (6, 5), (-1, 5)],      # crosses the outer rim
    [(0, 3), (6, 3), (6, 5), (0, 5)],        # touches the rim
    [(11, 3), (13, 3), (13, 5), (11, 5)],    # wholly outside
])
def test_must_be_strictly_inside_face(frame, xys):
    model, face = plate(frame, SQUARE)
    with pytest.raises(GeometryError):
        derive(model, face, frame, xys)


L_SHAPE = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


def test_concave_rim_excluded(frame):
    model, face = plate(frame, L_SHAPE)
    derive(model, face, frame, [(1, 1), (3, 1), (3, 9), (1, 9)])  # inside the vertical arm
    with pytest.raises(GeometryError):  # spans the notch
        derive(model, face, frame, [(1, 1), (8, 1), (8, 8), (1, 8)])
    with pytest.raises(GeometryError):  # corner pokes into the re-entrant corner region
        derive(model, face, frame, [(3, 3), (5, 3), (5, 5), (3, 5)])


def test_holes_excluded(frame):
    hole = [(4, 4), (4, 6), (6, 6), (6, 4)]  # clockwise, as a hole loop
    model, face = plate(frame, SQUARE, holes=[hole])
    derive(model, face, frame, [(1, 1), (3, 1), (3, 3), (1, 3)])  # beside the hole
    with pytest.raises(GeometryError):  # surrounds the hole, no boundary contact
        derive(model, face, frame, [(2, 2), (8, 2), (8, 8), (2, 8)])
    with pytest.raises(GeometryError):  # inside the hole
        derive(model, face, frame, [(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)])
    with pytest.raises(GeometryError):  # crosses the hole rim
        derive(model, face, frame, [(3, 4.5), (5, 4.5), (5, 5.5), (3, 5.5)])


@pytest.mark.parametrize("xys", [
    [(1, 1), (3, 1), (3, 3), (1, 3)],                      # far from the circular hole
    [(1.0, 4.0), (4.0, 4.0), (4.0, 6.0), (1.0, 6.0)],      # side x = 4 is tangent to the radius-1 hole at (4, 5)
    [(1.0, 4.0), (4.0 + 1e-9, 4.0), (4.0 + 1e-9, 6.0), (1.0, 6.0)],  # grazes the hole between any fixed samples
    [(1.0, 4.0), (3.9999, 4.0), (3.9999, 6.0), (1.0, 6.0)],          # clear by 1e-4: still refused, not sampled
    [(4.2, 4.2), (5.8, 4.2), (5.8, 5.8), (4.2, 5.8)],      # wholly inside the curved hole
])
def test_curved_trim_edge_is_refused_not_sampled(frame, xys):
    model, face = plate(frame, SQUARE)
    punch_hole(model, face, frame.place((5.0, 5.0)), 1.0)
    revision = model.revision
    with pytest.raises(ShellDetailError) as error:
        derive(model, face, frame, xys)
    assert error.value.code == "DETAIL_CURVED_BOUNDARY"
    assert model.revision == revision


def test_curved_outer_rim_is_refused():
    model = GeometryModel()
    ids = [model.add_point(*p) for p in [(10, 0, 0), (0, 10, 0), (-10, 0, 0), (0, -10, 0)]]
    via = [model.add_point(*p) for p in [(7.0710678118654755, 7.0710678118654755, 0), (-7.0710678118654755, 7.0710678118654755, 0),
                                         (-7.0710678118654755, -7.0710678118654755, 0), (7.0710678118654755, -7.0710678118654755, 0)]]
    arcs = [model.add_arc(ids[i], via[i], ids[(i + 1) % 4]) for i in range(4)]
    face = model.add_face_from_loop([OrientedEdge(e, True) for e in arcs],
                                    surface=Plane((0, 0, 0), (10, 0, 0), (0, 10, 0)))
    with pytest.raises(ShellDetailError) as error:
        derive_shell_detail_box(model, face, [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
                                director=(0, 0, 1), thickness=0.2)
    assert error.value.code == "DETAIL_CURVED_BOUNDARY"


def test_straight_hole_contact_by_a_hair_is_still_refused(frame):
    """Exact straight edges: a footprint side within the tolerance of a hole edge is refused."""

    hole = [(4, 4), (4, 6), (6, 6), (6, 4)]
    model, face = plate(frame, SQUARE, holes=[hole])
    with pytest.raises(ShellDetailError) as error:
        derive(model, face, frame, [(1, 3), (4.0 - 1e-12, 3), (4.0 - 1e-12, 7), (1, 7)])
    assert error.value.code == "DETAIL_BOUNDARY"


def _default_face_plate(frame, outer=SQUARE):
    """Project-style: a four-line loop and ``add_face`` without a surface (a topology-backed planar Coons)."""

    model = GeometryModel()
    ids = [model.add_point(*frame.place(p)) for p in outer]
    edges = [model.add_line(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    return model, model.add_face(edges)


def test_default_planar_coons_face_is_accepted(frame):
    from anygeometry.surfaces import CoonsSurface

    model, face = _default_face_plate(frame)
    assert isinstance(model.faces[face].support_surface, CoonsSurface)
    revision = model.revision
    detail, region_id = derive(model, face, frame, RECT, reference_offset=0.25, thickness=0.1)
    assert model.revision == revision
    snapshot = detail.solid_region_snapshot(region_id)
    base = frame.points(RECT)
    heights = (np.asarray(snapshot.coordinates) - base[[0, 1, 2, 3, 0, 1, 2, 3]]) @ frame.normal
    np.testing.assert_allclose(sorted(heights), [0.2] * 4 + [0.3] * 4, atol=1e-9)
    # The footprint still has to be inside the material.
    with pytest.raises(GeometryError):
        derive(model, face, frame, [(-1, 3), (6, 3), (6, 5), (-1, 5)])


def test_planar_coons_off_plane_footprint_is_refused():
    f = Frame()
    model, face = _default_face_plate(f)
    with pytest.raises(GeometryError):
        derive_shell_detail_box(model, face, f.points(RECT) + (0, 0, 0.5), director=(0, 0, 1), thickness=0.2)


def test_nonplanar_and_crossed_coons_are_refused():
    model = GeometryModel()
    ids = [model.add_point(*p) for p in [(0, 0, 0), (10, 0, 0), (10, 10, 1.0), (0, 10, 0)]]  # warped
    edges = [model.add_line(ids[i], ids[(i + 1) % 4]) for i in range(4)]
    warped = model.add_face(edges)
    with pytest.raises(ShellDetailError) as error:
        derive_shell_detail_box(model, warped, Frame().points(RECT), director=(0, 0, 1), thickness=0.2)
    assert error.value.code == "DETAIL_FACE"
    crossed = GeometryModel()
    ids = [crossed.add_point(*p) for p in [(0, 0, 0), (10, 10, 0), (10, 0, 0), (0, 10, 0)]]  # bow tie
    try:
        face = crossed.add_face([crossed.add_line(ids[i], ids[(i + 1) % 4]) for i in range(4)])
    except GeometryError:
        return  # refused at construction: also fail-closed
    with pytest.raises(GeometryError):
        derive_shell_detail_box(crossed, face, Frame().points(RECT), director=(0, 0, 1), thickness=0.2)


def test_unknown_face_and_type_errors():
    model, face = plate(Frame(), SQUARE)
    for bad in (face + 99, True, "1", 1.5):
        with pytest.raises(GeometryError):
            derive_shell_detail_box(model, bad, Frame().points(RECT), director=(0, 0, 1), thickness=0.2)
    with pytest.raises(GeometryError):
        derive_shell_detail_box(object(), face, Frame().points(RECT), director=(0, 0, 1), thickness=0.2)


def test_clone_is_independent_and_shallow_copy_shares_stores():
    """Documents the contract: only ``clone()`` is a supported way to copy a model."""

    model, face = plate(Frame(), SQUARE)
    deep = model.clone(include_features=False)
    assert deep.model_id != model.model_id and deep.revision == model.revision
    deep.add_point(50.0, 50.0, 50.0)
    assert len(deep.vertices) == len(model.vertices) + 1
    shallow = copy.copy(model)
    assert shallow._vertices is model._vertices  # shared mutable store: unsupported
    assert shallow._solid_regions is model._solid_regions
