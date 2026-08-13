from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import anygeometry.model as model_module
from anygeometry import CoonsSurface, GeometryError, GeometryModel, Plane
from anygeometry.entities import OrientedEdge
from anygeometry.operations import punch_hole
from anygeometry.predicates import (
    IntersectionComponent,
    IntersectionKind,
    IntersectionQuality,
    IntersectionResult,
)


def _warped_face(offset: float = 0.0) -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    boundary = geometry.add_points(
        tuple(
            np.asarray(point, dtype=float) + offset
            for point in (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.5, 1.0),
                (1.0, 1.0, 0.0),
                (0.5, 1.0, 1.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.5, 1.0),
            )
        )
    )
    face = geometry.add_face(
        geometry.add_polyline(boundary, close=True), corners=(0, 2, 4, 6)
    )
    return geometry, face


def _expected_warped_outer_uv() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (1.0, 1.0),
            (0.5, 1.0),
            (0.0, 1.0),
            (0.0, 0.5),
        )
    )


def test_planar_bow_tie_remains_rejected_atomically() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        )
    )
    edges = geometry.add_polyline(vertices, close=True)
    revision = geometry.revision

    with pytest.raises(GeometryError, match="self-intersects"):
        geometry.add_face_from_loop(
            tuple(OrientedEdge(edge, True) for edge in edges),
            surface=Plane(
                np.zeros(3),
                np.asarray((1.0, 0.0, 0.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
        )

    assert geometry.faces == {}
    assert geometry.revision == revision
    assert geometry.validate_topology() == ()


@pytest.mark.parametrize("offset", (0.0, 1.0e12))
def test_valid_warped_outer_uses_exact_uv_without_generic_inverse(
    monkeypatch: pytest.MonkeyPatch, offset: float
) -> None:
    def forbidden_inverse(*_args: object, **_kwargs: object) -> tuple[float, float]:
        raise AssertionError("topology construction boundary used generic inversion")

    monkeypatch.setattr(model_module, "closest_uv", forbidden_inverse)
    geometry, face = _warped_face(offset)

    loops = geometry.face_trim_loops_uv(face, curve_samples=11)

    assert len(loops) == 1
    assert loops[0] == pytest.approx(_expected_warped_outer_uv())
    assert geometry._validate_face_geometry(face) == []  # noqa: SLF001
    assert geometry.face_point(face, 0.5, 0.5) == pytest.approx(
        (offset + 0.5, offset + 0.5, offset + 2.0)
    )


def test_outer_uv_preserves_loop_order_lengths_and_stored_edge_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.75, 0.7),
            (0.0, 0.0, 0.0),
            (0.4, 0.0, 0.8),
            (2.0, 0.0, 0.0),
            (2.0, 0.25, 0.6),
            (2.0, 1.0, 0.0),
            (1.5, 1.0, 0.9),
            (0.0, 1.0, 0.0),
        )
    )
    loop: list[OrientedEdge] = []
    for index in range(len(vertices)):
        following = (index + 1) % len(vertices)
        if index % 2:
            edge = geometry.add_line(vertices[following], vertices[index])
            loop.append(OrientedEdge(edge, False))
        else:
            edge = geometry.add_line(vertices[index], vertices[following])
            loop.append(OrientedEdge(edge, True))

    monkeypatch.setattr(
        model_module,
        "closest_uv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact construction mapping consulted generic inverse")
        ),
    )
    face = geometry.add_face_from_loop(loop, corners=(1, 3, 5, 7))
    made = geometry.face_trim_loops_uv(face)[0]
    lengths = np.asarray([geometry.edge_length(item.edge) for item in loop])
    expected = np.asarray(
        (
            (0.0, 1.0 - lengths[7] / (lengths[7] + lengths[0])),
            (0.0, 0.0),
            (lengths[1] / (lengths[1] + lengths[2]), 0.0),
            (1.0, 0.0),
            (1.0, lengths[3] / (lengths[3] + lengths[4])),
            (1.0, 1.0),
            (1.0 - lengths[5] / (lengths[5] + lengths[6]), 1.0),
            (0.0, 1.0),
        )
    )

    assert geometry.faces[face].corners == (1, 3, 5, 7)
    assert tuple(item.forward for item in geometry.faces[face].loop) == tuple(
        item.forward for item in loop
    )
    assert made == pytest.approx(expected)


def test_public_trim_preserves_explicit_parameterization() -> None:
    geometry, face = _warped_face()
    geometry.set_face_parameterization(
        face,
        Plane(
            np.zeros(3),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 2.0, 0.0)),
        ),
    )

    made = geometry.face_trim_loops_uv(face)[0]

    assert isinstance(geometry.faces[face].surface, CoonsSurface)
    assert made == pytest.approx(0.5 * _expected_warped_outer_uv())


def test_curved_outer_keeps_existing_validation_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GeometryModel()
    a, b, c, d, via = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, -0.2, 0.2),
        )
    )
    edges = (
        geometry.add_arc(a, via, b),
        geometry.add_line(b, c),
        geometry.add_line(c, d),
        geometry.add_line(d, a),
    )
    original = model_module.closest_uv
    calls = 0

    def spy(*args: object, **kwargs: object) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "closest_uv", spy)
    face = geometry.add_face(edges, corners=(0, 1, 2, 3))
    validation_calls = calls

    made = geometry.face_trim_loops_uv(face, curve_samples=7)[0]

    assert validation_calls > 0
    assert calls == validation_calls
    assert made.shape == (9, 2)


def test_holes_keep_one_existing_uv_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
        )
    )
    punch_hole(geometry, face, (2.0, 2.0, 0.0), 0.5)
    geometry.set_face_surface(face, CoonsSurface())
    original = model_module.closest_uv
    calls = 0

    def spy(*args: object, **kwargs: object) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def forbidden_exact(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("hole-bearing face mixed canonical and projected UV")

    monkeypatch.setattr(model_module, "closest_uv", spy)
    monkeypatch.setattr(
        GeometryModel, "_topology_coons_outer_loop_uv", forbidden_exact
    )

    assert geometry._validate_face_geometry(face) == []  # noqa: SLF001
    loops = geometry.face_trim_loops_uv(face, curve_samples=9)

    assert len(loops) == 2
    assert calls > len(geometry.faces[face].loop)


def _crossing_edges(
    geometry: GeometryModel, offset: float
) -> tuple[int, ...]:
    vertices = geometry.add_points(
        tuple(
            np.asarray(point, dtype=float) + offset
            for point in (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (2.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 1.0),
            )
        )
    )
    return tuple(geometry.add_polyline(vertices, close=True))


@pytest.mark.parametrize("offset", (0.0, 1.0e12))
def test_physical_crossing_rejects_atomically_at_large_translation(
    offset: float,
) -> None:
    geometry = GeometryModel()
    existing = geometry.add_plate(
        geometry.add_points(
            ((10.0, 10.0, 0.0), (11.0, 10.0, 0.0), (11.0, 11.0, 0.0), (10.0, 11.0, 0.0))
        )
    )
    crossing = _crossing_edges(geometry, offset)
    before_revision = geometry.revision
    before_change = geometry.last_change_set
    before_faces = tuple(geometry.faces)
    before_existing = geometry.faces[existing]
    before_incidence = {
        edge: geometry.faces_using_edge(edge) for edge in crossing
    }
    before_next_face = geometry.id_state()["face"]

    with pytest.raises(GeometryError, match=r"face \d+ loop 0 self-intersects"):
        geometry.add_face(crossing, corners=(0, 1, 3, 4))

    assert geometry.revision == before_revision
    assert geometry.last_change_set is before_change
    assert tuple(geometry.faces) == before_faces
    assert geometry.faces[existing] is before_existing
    assert {
        edge: geometry.faces_using_edge(edge) for edge in crossing
    } == before_incidence
    assert geometry.id_state()["face"] == before_next_face + 1


def _simple_loop_points() -> np.ndarray:
    return np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )


def _forced_result(kind: IntersectionKind) -> IntersectionResult:
    if kind is IntersectionKind.DISJOINT:
        return IntersectionResult(kind)
    if kind in (
        IntersectionKind.UNCLASSIFIED,
        IntersectionKind.UNSUPPORTED,
        IntersectionKind.CAPABILITY_MISSING,
    ):
        return IntersectionResult(kind, diagnostics=("forced test result",))
    component = IntersectionComponent(
        ((0.5, 0.5, 0.0),),
        IntersectionQuality.EXACT,
        max_residual=1.0e6,
    )
    return IntersectionResult(kind, (component,))


def test_model_handler_accepts_only_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GeometryModel()
    monkeypatch.setattr(
        model_module,
        "qualified_segment_segment",
        lambda *_args, **_kwargs: IntersectionResult(IntersectionKind.DISJOINT),
    )

    assert not geometry._straight_loop_self_intersects_3d(  # noqa: SLF001
        _simple_loop_points()
    )


@pytest.mark.parametrize(
    "kind", tuple(kind for kind in IntersectionKind if kind is not IntersectionKind.DISJOINT)
)
def test_model_handler_rejects_every_non_disjoint_result(
    monkeypatch: pytest.MonkeyPatch, kind: IntersectionKind
) -> None:
    geometry = GeometryModel()
    result = _forced_result(kind)
    monkeypatch.setattr(
        model_module,
        "qualified_segment_segment",
        lambda *_args, **_kwargs: result,
    )

    assert geometry._straight_loop_self_intersects_3d(  # noqa: SLF001
        _simple_loop_points()
    )


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("forced"),
        object(),
        SimpleNamespace(kind=IntersectionKind.DISJOINT),
    ),
)
def test_model_handler_rejects_raised_or_malformed_results(
    monkeypatch: pytest.MonkeyPatch, failure: object
) -> None:
    geometry = GeometryModel()

    def broken(*_args: object, **_kwargs: object) -> object:
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(model_module, "qualified_segment_segment", broken)

    assert geometry._straight_loop_self_intersects_3d(  # noqa: SLF001
        _simple_loop_points()
    )


def test_model_handler_rejects_a_degenerate_segment() -> None:
    geometry = GeometryModel()
    points = _simple_loop_points()
    points[1] = points[0]

    assert geometry._straight_loop_self_intersects_3d(points)  # noqa: SLF001
