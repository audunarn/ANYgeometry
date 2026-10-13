"""Workloads for the generality + speed programme (G0).

Every workload builds its input model outside the timed region and returns a
``run()`` callable that performs the measured work and returns the resulting
model. Only public ANYgeometry APIs are used, so the corpus runs unchanged
against any source tree put on ``PYTHONPATH``.
"""
from __future__ import annotations

import math

import numpy as np

from anygeometry import (ConnectionIntent, GeometryModel, apply_intersections, from_dict,
                         plan_intersections, to_dict)
from anygeometry.generators import cone, cylinder


def plan_apply(model: GeometryModel) -> GeometryModel:
    handles = [model.handle("face", face) for face in sorted(model.faces)]
    plan = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    return model


def _intersect(build):
    def make():
        model = build()
        return lambda: plan_apply(model)
    return make


def pipes(k: int = 6) -> GeometryModel:
    """k vertical eight-segment cylinders through one horizontal plate."""
    model = GeometryModel()
    side = int(math.ceil(math.sqrt(k)))
    extent = 2.0 * side + 2.0
    corners = model.add_points(((0, 0, 0), (extent, 0, 0), (extent, extent, 0), (0, extent, 0)))
    model.add_plate(tuple(corners))
    for i in range(k):
        x, y = 2.0 + 2.0 * (i % side), 2.0 + 2.0 * (i // side)
        model.insert_model(cylinder(0.4, 2.0, origin=(x + 0.013 * i, y + 0.007 * i, -1.0), axis=(0, 0, 1),
                                    radial_direction=(1, 0, 0), circumferential_segments=8))
    return model


def rings(count: int = 8) -> GeometryModel:
    """One horizontal cylinder crossed by square ring plates."""
    model = GeometryModel()
    model.insert_model(cylinder(1.0, 10.0, origin=(0, 0, 0), axis=(1, 0, 0), radial_direction=(0, 1, 0),
                                circumferential_segments=8))
    for i in range(count):
        x = 0.6 + i * 9.0 / count
        corners = model.add_points(((x, -1.5, -1.5), (x, 1.5, -1.5), (x, 1.5, 1.5), (x, -1.5, 1.5)))
        model.add_plate(tuple(corners))
    return model


def crossing_grid(n: int = 16) -> GeometryModel:
    """n/2 x n/2 crossing vertical web plates standing on a base plate."""
    model = GeometryModel()
    extent = float(n // 2 + 1)
    corners = model.add_points(((0, 0, 0), (extent, 0, 0), (extent, extent, 0), (0, extent, 0)))
    model.add_plate(tuple(corners))
    for i in range(1, n // 2 + 1):
        x = float(i) + 0.01 * i
        model.add_plate(tuple(model.add_points(((x, 0, 0), (x, extent, 0), (x, extent, 1), (x, 0, 1)))))
        y = float(i) + 0.007 * i
        model.add_plate(tuple(model.add_points(((0, y, 0), (extent, y, 0), (extent, y, 1), (0, y, 1)))))
    return model


def tilted(n: int = 8) -> GeometryModel:
    """n mutually crossing tilted plates in generic position."""
    model = GeometryModel()
    rng = np.random.default_rng(3)
    for i in range(n):
        angle = math.pi * i / n
        c, s = math.cos(angle), math.sin(angle)
        tilt = 0.3 * float(rng.standard_normal())
        points = [(-3 * c, -3 * s, -2 + tilt), (3 * c, 3 * s, -2 - tilt), (3 * c, 3 * s, 2 - tilt),
                  (-3 * c, -3 * s, 2 + tilt)]
        points = [(x + 0.05 * i, y + 0.03 * i, z) for x, y, z in points]
        model.add_plate(tuple(model.add_points(points)))
    return model


CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))


def cubic_wall_pipe() -> GeometryModel:
    """A cubic Bezier wall (extrusion) crossed by an eight-segment pipe."""
    model = GeometryModel()
    points = model.add_points(CUBIC)
    edge = model.add_spline(points[0], tuple(points[1:-1]), points[-1])
    model.extrude([edge], (.25, 0., 1.5))
    model.insert_model(cylinder(.7, 8., origin=(-2., .4, .6), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=8))
    return model


def cone_pipe() -> GeometryModel:
    """A cone hitting a cylinder ten degrees off perpendicular."""
    model = GeometryModel()
    model.insert_model(cylinder(1.0, 6.0, origin=(-3., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=8))
    tilt = math.radians(10.)
    model.insert_model(cone(.2, .6, 3., origin=(0.1, 0.05, -2.5), axis=(math.sin(tilt), 0., math.cos(tilt)),
                            radial_direction=(0., 1., 0.), circumferential_segments=8))
    return model


def plate_grid(n: int = 60) -> GeometryModel:
    model = GeometryModel()
    ids = {}
    for i in range(n + 1):
        for j in range(n + 1):
            ids[i, j] = model.add_point(float(i), float(j), 0.0)
    for i in range(n):
        for j in range(n):
            model.add_plate((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))
    return model


def document_round_trip(n: int = 60):
    def make():
        model = plate_grid(n)

        def run():
            back = from_dict(to_dict(model))
            back.validate_topology()
            return back
        return run
    return make


WORKLOADS = {
    "pipes6": _intersect(pipes),
    "rings8": _intersect(rings),
    "grid16": _intersect(crossing_grid),
    "grid32": _intersect(lambda: crossing_grid(32)),
    "tilted8": _intersect(tilted),
    "cubic_wall_pipe": _intersect(cubic_wall_pipe),
    "cone_pipe": _intersect(cone_pipe),
    "doc60": document_round_trip(60),
}

# Workloads whose quality metrics (joint residual) are meaningful and affordable.
QUALITY = {"pipes6", "rings8", "grid16", "tilted8", "cubic_wall_pipe", "cone_pipe"}
