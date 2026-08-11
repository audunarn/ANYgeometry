"""Complete typed measurement coverage used by the modeling workspace."""

from __future__ import annotations

import numpy as np

from anygeometry import EntityRef, GeometryModel, measure
from anygeometry.generators import plate


def test_coordinates_radius_and_edge_centroid() -> None:
    geometry = GeometryModel()
    start = geometry.add_point(1.0, 0.0, 0.0)
    via = geometry.add_point(0.0, 1.0, 0.0)
    end = geometry.add_point(-1.0, 0.0, 0.0)
    arc = EntityRef("edge", geometry.add_arc(start, via, end))

    coordinates = measure(geometry, EntityRef("vertex", via), quantity="coordinates")
    radius = measure(geometry, arc, quantity="radius")
    centroid = measure(geometry, arc, quantity="centroid")

    assert coordinates.value == (0.0, 1.0, 0.0)
    assert radius.unit == "m"
    assert np.isclose(radius.value, 1.0)
    assert np.allclose(centroid.value, (0.0, 2.0 / np.pi, 0.0), atol=2.0e-5)


def test_planar_face_centroid_includes_origin_and_dimensions() -> None:
    geometry = plate(4.0, 2.0, origin=(10.0, -3.0, 5.0))
    reference = next(iter(geometry.faces.values())).ref

    centroid = measure(geometry, reference, quantity="centroid")

    assert centroid.kind == "centroid"
    assert centroid.unit == "m"
    assert np.allclose(centroid.value, (12.0, -2.0, 5.0), atol=1.0e-10)
