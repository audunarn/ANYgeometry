"""Adversarial corpus for the robust-predicate programme (slice R0).

Small, fast cases from GENERALITY_ARCHITECTURE.md R.4: exact-tolerance vertex
constructions, underflow/overflow-scale coordinates, float32 inputs, the
cylinder seam case of the S1 review, the plane ``u = 0`` edge family, stations
at 0.75 and 1.5 tolerance, a planar face with a rounded (arc) opening and query
points between arc and chord, plus the G0 workloads of
``benchmarks/generality/corpus.py``.

Each case is a :class:`Case`: a ``build()`` returning ``(model, context)`` and a
tuple of :class:`Operation`. An operation is ``run(model, context, reverse)``
and returns a JSON-able, float-free outcome (the decision it made). Operations
whose result must not depend on operand order are marked ``reversible``; the
perturbation harness then also runs them with ``reverse=True``.

The corpus only builds models and calls engine operations. It never patches
the engine; perturbation lives in ``tools/perturbation_harness.py``.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_G0 = ROOT / "benchmarks" / "generality"

from anygeometry import (  # noqa: E402
    Cone, ConnectionIntent, CoonsSurface, Cylinder, GeometryModel, NativeArcStation, Plane,
    RuledSurface, TolerancePolicy, apply_intersections, plan_intersections,
    query_intersection, query_prepared_native_arc_parameter_maps,
    query_prepared_native_arc_station_incidence, query_trimmed_surface_charts, to_dict)
from anygeometry.arrangement_geometry import LinePath  # noqa: E402
from anygeometry.entities import OrientedEdge  # noqa: E402


# --------------------------------------------------------------------------- types

@dataclass(frozen=True)
class Operation:
    name: str
    run: Callable  # (model, context, reverse) -> outcome
    reversible: bool = False


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable  # () -> (model, context)
    operations: tuple[Operation, ...]
    group: str = "adversarial"
    note: str = ""
    inventory: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- outcomes

def topology_digest(model) -> str:
    """Float-free canonical document digest (as benchmarks/generality/run.py split_document)."""
    document = dict(to_dict(model))
    document.pop("model_id", None)
    document.pop("checksum", None)

    def strip(value):
        if isinstance(value, float):
            return "<f>"
        if isinstance(value, dict):
            return {key: strip(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    skeleton = strip(document)
    return hashlib.sha256(json.dumps(skeleton, sort_keys=True, default=str).encode()).hexdigest()[:16]


def shape(model) -> dict:
    return {"faces": len(model.faces), "edges": len(model.edges), "vertices": len(model.vertices)}


def _handles(model, reverse):
    handles = [model.handle("face", face) for face in sorted(model.faces)]
    return handles[::-1] if reverse else handles


def op_plan_apply(model, context, reverse):
    """plan_intersections + apply_intersections over all faces (CONNECT)."""
    handles = _handles(model, reverse)
    plan = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    return {"digest": topology_digest(model), "shape": shape(model)}


def _query_outcome(result):
    return {"kind": result.kind.value, "classified": bool(result.classified),
            "components": len(result.components),
            "dimension": None if result.dimension is None else str(result.dimension.value)}


def op_query(first_key="first", second_key="second"):
    def run(model, context, reverse):
        first, second = context[first_key], context[second_key]
        if reverse:
            first, second = second, first
        return _query_outcome(query_intersection(model, first, second))
    return Operation(f"query[{first_key},{second_key}]", run, reversible=True)


def _membership(model, face_id, point, chart):
    """Float-free membership decisions of one point against one face."""
    tolerance = float(model.tolerance.length)
    row = {}
    if isinstance(chart, Exception):
        row["chart_contains"] = f"error:{type(chart).__name__}"
    else:
        try:
            p = tuple(float(x) for x in point)
            # boundary=True: closed face (material test); boundary=False:
            # open face (the form the engine uses for hole loops).
            row["chart_contains"] = bool(chart.domain.contains(LinePath(p, p), 0., tolerance))
            row["chart_interior"] = bool(chart.domain.contains(LinePath(p, p), 0., tolerance,
                                                               boundary=False))
        except Exception as error:  # noqa: BLE001 - an error is a decision outcome
            row["chart_contains"] = f"error:{type(error).__name__}"
    try:
        # Coordinates (the clipped uv itself, the projection distance) are not
        # recorded: only the decisions made from them.
        uv = model.face_local_uv(face_id, point)
        row["face_contains_uv"] = bool(model.face_contains_uv(face_id, uv))
    except Exception as error:  # noqa: BLE001
        row["face_contains_uv"] = f"error:{type(error).__name__}"
    try:
        _, _, distance = model.project_to_face(face_id, point)
        row["projection_within_tol"] = bool(distance <= tolerance)
    except Exception as error:  # noqa: BLE001
        row["projection_within_tol"] = f"error:{type(error).__name__}"
    return row


def op_points(face_key="face", points_key="points"):
    """Point-in-face (certified chart domain, polygon route) and project_to_face."""
    def run(model, context, reverse):
        face_id = context[face_key]
        try:
            chart = query_trimmed_surface_charts(model, [model.handle("face", face_id)]).charts[0]
        except Exception as error:  # noqa: BLE001
            chart = error
        return {label: _membership(model, face_id, point, chart)
                for label, point in context[points_key].items()}
    return Operation(f"points[{face_key}]", run)


def op_charts(model, context, reverse):
    """query_trimmed_surface_charts over every face; float-free chart structure."""
    rows = {}
    for face_id in sorted(model.faces):
        try:
            charts = query_trimmed_surface_charts(model, [model.handle("face", face_id)]).charts
            rows[str(face_id)] = [[type(chart.support).__name__,
                                   [[type(path.curve).__name__ for path in loop]
                                    for loop in chart.boundaries]] for chart in charts]
        except Exception as error:  # noqa: BLE001
            rows[str(face_id)] = f"error:{type(error).__name__}:{error}"
    return rows


CHARTS = Operation("charts", op_charts)
PLAN_APPLY = Operation("plan_apply", op_plan_apply, reversible=True)


# --------------------------------------------------------------------------- builders

def surface_face(model, surface) -> int:
    """Face over a whole analytic patch (as tests/test_curved_intersection_matrix.py)."""
    if isinstance(surface, (Cylinder, Cone)):
        lower = tuple(model.add_point(*surface.evaluate(u, 0.0)) for u in (0.0, 0.5, 1.0))
        upper = tuple(model.add_point(*surface.evaluate(u, 1.0)) for u in (0.0, 0.5, 1.0))
        bottom = model.add_arc(lower[0], lower[1], lower[2])
        right = model.add_line(lower[2], upper[2])
        top = model.add_arc(upper[0], upper[1], upper[2])
        left = model.add_line(lower[0], upper[0])
        return model.add_face_from_loop(
            (OrientedEdge(bottom, True), OrientedEdge(right, True),
             OrientedEdge(top, False), OrientedEdge(left, False)), (0, 1, 2, 3), surface=surface)
    corners = tuple(model.add_point(*surface.evaluate(u, v))
                    for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    return model.add_face(model.add_polyline(corners, close=True), (0, 1, 2, 3), surface=surface)


def matrix_plane() -> Plane:
    return Plane(np.asarray((0.0, 0.0, 0.0)), np.asarray((2.0, 0.0, 0.0)), np.asarray((0.0, 2.0, 0.0)))


def matrix_supports() -> dict:
    z, axis, radial = np.zeros(3), np.asarray((0.0, 0.0, 1.0)), np.asarray((1.0, 0.0, 0.0))
    return {
        "plane": matrix_plane(),
        "cylinder": Cylinder(z, axis, radial, 2.0, 3.0, sweep_angle=1.4),
        "cone": Cone(z, axis, radial, 2.0, 1.0, 3.0, sweep_angle=1.4),
        "ruled": RuledSurface(np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.25))),
                              np.asarray(((0.0, 2.0, 0.5), (2.0, 2.0, 0.75)))),
        "coons": CoonsSurface(np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.25))),
                              np.asarray(((2.0, 0.0, 0.25), (2.0, 2.0, 0.75))),
                              np.asarray(((0.0, 2.0, 0.5), (2.0, 2.0, 0.75))),
                              np.asarray(((0.0, 0.0, 0.0), (0.0, 2.0, 0.5)))),
    }


def plane_family(second: str):
    """test_plane_support_completes_fifteen_face_pair_families: plane face x <second>."""
    def build():
        model = GeometryModel()
        first = surface_face(model, matrix_plane())
        other = surface_face(model, matrix_supports()[second])
        return model, {"first": model.handle("face", first), "second": model.handle("face", other)}
    return build


def generic_plane_edge(seed: int):
    """A generic affine plane face and a second plate standing on the rounded
    evaluation of its ``u = 0`` edge (R.0: lstsq vs Cramer disagree on sign(u))."""
    def build():
        rng = np.random.default_rng(seed)
        origin = rng.uniform(-3, 3, 3)
        u_vector = rng.uniform(-2, 2, 3)
        v_vector = rng.uniform(-2, 2, 3)
        plane = Plane(origin, u_vector, v_vector)
        model = GeometryModel()
        first = surface_face(model, plane)
        a, b = plane.evaluate(0.0, 0.2), plane.evaluate(0.0, 0.8)
        normal = plane.normal
        out = normal - 0.5 * u_vector / np.linalg.norm(u_vector)
        corners = model.add_points((a, b, b + out, a + out))
        second = model.add_plate(tuple(corners))
        lying = {f"edge_u0_{k}": plane.evaluate(0.0, float(t))
                 for k, t in enumerate(rng.uniform(0.05, 0.95, 4))}
        return model, {"first": model.handle("face", first), "second": model.handle("face", second),
                       "face": first, "points": lying}
    return build


def tolerance_scaled(scale: float) -> TolerancePolicy:
    base = TolerancePolicy()
    return TolerancePolicy(length=base.length * scale, merge_length=base.merge_length * scale,
                           area=max(base.area * scale * scale, 5e-324),
                           surface_residual=base.surface_residual * scale)


def exact_tolerance_vertex(factor: str, scale: float = 1.0):
    """Two plates whose corners are exactly ``d`` apart (``d`` relative to the
    model tolerance; the distance is an exact float, no rounding in the input).

    A: square z=0, corner at the origin. B: vertical plate in x = d whose bottom
    corner is (d, 0, 0): vertex-vertex and edge-edge distance exactly d.
    """
    def build():
        policy = tolerance_scaled(scale) if scale != 1.0 else TolerancePolicy()
        tol = float(policy.length)
        named = {"0.75tol": 0.75 * tol, "prev(tol)": math.nextafter(tol, 0.0), "tol": tol,
                 "next(tol)": math.nextafter(tol, math.inf), "1.5tol": 1.5 * tol}
        # Otherwise an absolute offset (underflow regime: its square is 0 or subnormal).
        d = named[factor] if factor in named else float(factor)
        s = scale
        model = GeometryModel(tolerance=policy)
        a = model.add_plate(tuple(model.add_points(((-s, -s, 0.), (0., -s, 0.), (0., 0., 0.), (-s, 0., 0.)))))
        b = model.add_plate(tuple(model.add_points(((d, 0., 0.), (d, s, 0.), (d, s, s), (d, 0., s)))))
        return model, {"first": model.handle("face", a), "second": model.handle("face", b), "face": a,
                       "points": {"corner_b": (d, 0., 0.), "edge_b": (d, -0.5 * s, 0.)}}
    return build


def scaled_crossing(scale: float):
    """Two generic crossing plates scaled to underflow/overflow-scale coordinates."""
    def build():
        policy = tolerance_scaled(scale)
        model = GeometryModel(tolerance=policy)
        s = scale
        a = model.add_plate(tuple(model.add_points(
            [(x * s, y * s, z * s) for x, y, z in ((0., 0., 0.), (1., .1, 0.), (1.1, 1., 0.), (.1, .9, 0.))])))
        b = model.add_plate(tuple(model.add_points(
            [(x * s, y * s, z * s) for x, y, z in ((.5, -.2, -.5), (.6, 1.2, -.5), (.6, 1.2, .5), (.5, -.2, .5))])))
        return model, {"first": model.handle("face", a), "second": model.handle("face", b)}
    return build


def float32_tilted(n: int = 3):
    """Tilted generic plates whose coordinates are float32 roundings."""
    def build():
        model = GeometryModel()
        rng = np.random.default_rng(3)
        for i in range(n):
            angle = math.pi * i / n
            c, sn = math.cos(angle), math.sin(angle)
            tilt = 0.3 * float(rng.standard_normal())
            points = [(-3 * c, -3 * sn, -2 + tilt), (3 * c, 3 * sn, -2 - tilt), (3 * c, 3 * sn, 2 - tilt),
                      (-3 * c, -3 * sn, 2 + tilt)]
            points = [tuple(float(np.float32(value)) for value in (x + .05 * i, y + .03 * i, z))
                      for x, y, z in points]
            model.add_plate(tuple(model.add_points(points)))
        faces = sorted(model.faces)
        return model, {"first": model.handle("face", faces[0]), "second": model.handle("face", faces[1])}
    return build


SEAM_CYLINDER = dict(origin=(0, 0, 0), axis=(.885075041426785, .3822621141833038, -1.1992640930491332),
                     radial=(-2.0412527590777985, 2.141861457996631, -.05345460228944404))
SEAM_POINT = (-.29766741399452923, .9148405533160117, -.5696146703292175)


def seam_cylinder():
    """S1 review s1c case 2: a point on the u=0 seam where NumPy snaps u to 0.0
    and a scalar route returns 1.42e-14."""
    def build():
        surface = Cylinder(SEAM_CYLINDER["origin"], SEAM_CYLINDER["axis"], SEAM_CYLINDER["radial"], 1, 1, 0, 1)
        model = GeometryModel()
        face = surface_face(model, surface)
        point = np.asarray(SEAM_POINT)
        # A plate in the plane spanned by the axis and the seam ruling.
        a, b = surface.evaluate(0.0, -0.25), surface.evaluate(0.0, 1.25)
        axis_a = surface.origin - 0.25 * surface.height * surface.axis
        axis_b = surface.origin + 1.25 * surface.height * surface.axis
        outward = 0.5 * surface.radius * surface.radial_direction
        plate = model.add_plate(tuple(model.add_points((axis_a, a + outward, b + outward, axis_b))))
        points = {"seam_point": point,
                  "seam_next_up": np.nextafter(point, np.inf),
                  "seam_next_down": np.nextafter(point, -np.inf),
                  "evaluate_0_0.5": surface.evaluate(0.0, 0.5),
                  "evaluate_0_0.25": surface.evaluate(0.0, 0.25)}
        return model, {"face": face, "points": points, "first": model.handle("face", face),
                       "second": model.handle("face", plate)}
    return build


def _rotation():
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def arc_opening(tilted: bool):
    """A 4 x 4 planar face with a circular opening of four arcs (centre (2, 2),
    radius 1); query points between arc and chord, on the arc, and at 0.75 /
    1.5 tolerance from it (R.4: the barge's rounded openings, rev-4 major 6b)."""
    def build():
        rotation = _rotation() if tilted else np.eye(3)
        offset = np.asarray((0.37, -1.21, 0.53)) if tilted else np.zeros(3)

        def world(x, y):
            return rotation @ np.asarray((x, y, 0.0)) + offset

        model = GeometryModel()
        tol = float(model.tolerance.length)
        outer = model.add_points([world(x, y) for x, y in ((0, 0), (4, 0), (4, 4), (0, 4))])
        face = model.add_plate(tuple(outer))
        h = math.sqrt(0.5)
        ring = model.add_points([world(2 + math.cos(k * math.pi / 4), 2 + math.sin(k * math.pi / 4))
                                 for k in range(8)])
        arcs = [model.add_arc(ring[2 * k], ring[2 * k + 1], ring[(2 * k + 2) % 8]) for k in range(4)]
        from dataclasses import replace
        with model.transaction():
            model._put_entity("face", replace(model.faces[face],
                                              holes=(tuple(OrientedEdge(e, True) for e in arcs),)))
        # Crossing plate through the opening (arc x plane junctions).
        cut = model.add_plate(tuple(model.add_points(
            [rotation @ np.asarray(p) + offset for p in ((2.6, -0.5, -1.0), (2.6, 4.5, -1.0),
                                                          (2.6, 4.5, 1.0), (2.6, -0.5, 1.0))])))

        def polar(radius, angle=math.pi / 4):
            return world(2 + radius * math.cos(angle), 2 + radius * math.sin(angle))

        points = {"between_arc_chord": polar(0.85), "on_arc": polar(1.0),
                  "arc_vertex": world(2 + 1.0, 2.0),
                  "out_0.75tol": polar(1.0 + 0.75 * tol), "in_0.75tol": polar(1.0 - 0.75 * tol),
                  "out_1.5tol": polar(1.0 + 1.5 * tol), "in_1.5tol": polar(1.0 - 1.5 * tol),
                  "material": polar(1.2),
                  # midpoint of the chord (3, 2)-(2, 3) under the first arc
                  "chord_mid": polar(h)}
        probe_a = model.add_point(*(polar(0.85) - 0.5 * (rotation @ np.asarray((0., 0., 1.)))))
        probe_b = model.add_point(*(polar(0.85) + 0.5 * (rotation @ np.asarray((0., 0., 1.)))))
        probe = model.add_line(probe_a, probe_b)
        return model, {"face": face, "points": points, "first": model.handle("face", face),
                       "second": model.handle("face", cut), "probe": model.handle("edge", probe)}
    return build


# --- native arc stations at 0.75 / 1.5 tolerance (tests/test_native_arc_station_incidence.py)

def _nice_fixture():
    model = GeometryModel()
    a, b, c = model.add_points(((8., 0., 0.), (9., 1., 0.), (10., 0., 0.)))
    arc = model.add_arc(a, b, c)
    e = model.add_point(9., -1., 0.)
    face = model.add_face_from_loop((OrientedEdge(arc, True), OrientedEdge(model.add_line(c, e), True),
                                     OrientedEdge(model.add_line(e, a), True)))
    plan = plan_intersections(model, (face,), policy="connect")
    apply_intersections(model, plan, policy="connect")
    return model, {"arc": arc}


def _station_outcome(model, stations):
    try:
        binding = query_prepared_native_arc_station_incidence(model, stations)
    except Exception as error:  # noqa: BLE001
        return {"error": type(error).__name__, "message": str(error)[:80]}
    return [{"classification": r.classification,
             "current_zero": r.current_bound == 0, "ancestor_zero": r.ancestor_bound == 0}
            for r in binding.records]


def op_station(factor):
    def run(model, context, reverse):
        tol = float(model.tolerance.length)
        point = (8. + {"0.75": 3 * tol / 4, "1.5": 3 * tol / 2}[factor], 0., 0.)
        return _station_outcome(model, [NativeArcStation(context["arc"], Fraction(0), point)])
    return Operation(f"station_{factor}tol", run)


def _harmonic_point(frame, s):
    w = float(frame.sweep) * float(s)
    return tuple(float(x + y * math.cos(w) + z * math.sin(w))
                 for x, y, z in zip(frame.center, frame.cosine, frame.sine))


def _split_cylinder_fixture():
    from anygeometry.generators.structural import cylinder
    model = cylinder(1., 1., circumferential_segments=3)
    roots = frozenset(model.edges)
    model.add_plate(model.add_points(((2., .5, -.5), (2., .5, 1.5), (-2., .5, 1.5), (-2., .5, -.5))))
    return model, {"roots": roots}


def op_station_selection(model, context, reverse):
    """Which split-arc record has current_bound == 0 (the selection the native
    arc station test makes) after plan/apply."""
    plan = plan_intersections(model, tuple(model.faces), policy="connect")
    apply_intersections(model, plan, policy="connect")
    maps = query_prepared_native_arc_parameter_maps(model)
    rows = [row for row in maps.records if row.edge_id not in context["roots"]]
    stations = [NativeArcStation(row.edge_id, Fraction(0), _harmonic_point(row.current_frame, 0))
                for row in rows]
    outcome = _station_outcome(model, stations)
    return {"digest": topology_digest(model), "records": outcome}


# --------------------------------------------------------------------------- G0

def _g0_cases():
    sys.path.insert(0, str(_G0))
    try:
        import corpus as g0  # noqa: PLC0415 - benchmarks/generality/corpus.py
    finally:
        sys.path.remove(str(_G0))
    builders = {"pipes6": g0.pipes, "rings8": g0.rings, "grid16": g0.crossing_grid,
                "grid32": lambda: g0.crossing_grid(32), "tilted8": g0.tilted,
                "cubic_wall_pipe": g0.cubic_wall_pipe, "cone_pipe": g0.cone_pipe}
    cases = []
    for name, builder in builders.items():
        cases.append(Case(f"g0/{name}", (lambda b=builder: (b(), {})), (PLAN_APPLY,), group="g0",
                          note="G0 workload (benchmarks/generality/corpus.py), plan/apply over all faces"))
    return cases


# --------------------------------------------------------------------------- registry

QUERY = op_query()
POINTS = op_points()


def _cases():
    cases = []
    for second in ("plane", "cylinder", "cone", "ruled", "coons"):
        cases.append(Case(f"plane_family/{second}", plane_family(second), (QUERY,),
                          note="test_plane_support_completes_fifteen_face_pair_families",
                          inventory=("A1", "E3", "E4", "E5")))
    for seed in (1, 2, 3):
        cases.append(Case(f"plane_u0_edge/seed{seed}", generic_plane_edge(seed), (QUERY, POINTS, PLAN_APPLY),
                          note="generic affine plane, plate on the rounded u=0 edge",
                          inventory=("A1", "B1", "E4")))
    for factor in ("0.75tol", "prev(tol)", "tol", "next(tol)", "1.5tol"):
        cases.append(Case(f"exact_tol_vertex/{factor}", exact_tolerance_vertex(factor),
                          (PLAN_APPLY, QUERY, POINTS),
                          note="corners exactly d apart", inventory=("B1", "B3", "C1", "C2")))
    # Underflow regime at unit scale: offsets whose squares underflow to zero
    # or to subnormals (review s1: 1e-170, 1e-200 vertex identity).
    for offset in ("1e-170", "1e-200", "5e-324"):
        cases.append(Case(f"exact_tol_vertex/offset{offset}", exact_tolerance_vertex(offset),
                          (PLAN_APPLY, QUERY), note="corners an underflowing distance apart",
                          inventory=("C1", "C2")))
    # Whole model (coordinates and tolerance policy) scaled. Scales that the
    # engine refuses at construction (>= 1e100 here: "invalid surface
    # geometry") are outside its accepted range and not in the corpus.
    for scale in (1e-150, 1e-100, 1e20, 1e50):
        cases.append(Case(f"exact_tol_vertex/tol@{scale:g}", exact_tolerance_vertex("tol", scale),
                          (PLAN_APPLY, QUERY), note="exact-tolerance corners at extreme scale",
                          inventory=("C1", "C2")))
    for scale in (1e-100, 1e3, 1e8):
        cases.append(Case(f"scaled_crossing/{scale:g}", scaled_crossing(scale), (PLAN_APPLY, QUERY),
                          note="generic crossing at extreme scale", inventory=("A1", "B1", "C1")))
    cases.append(Case("float32_tilted3", float32_tilted(), (PLAN_APPLY, QUERY),
                      note="float32-rounded coordinates", inventory=("A1", "B1", "C1")))
    cases.append(Case("seam_cylinder", seam_cylinder(), (POINTS, QUERY, PLAN_APPLY, CHARTS),
                      note="S1 review s1c case 2", inventory=("A2", "A3", "A4", "D1")))
    for tilted in (False, True):
        cases.append(Case(f"arc_opening/{'tilted' if tilted else 'flat'}", arc_opening(tilted),
                          (POINTS, QUERY, op_query("probe", "first"), PLAN_APPLY, CHARTS),
                          note="planar face with a circular arc opening",
                          inventory=("A1", "B2", "D1", "E4")))
    cases.append(Case("native_arc_station/nice", _nice_fixture,
                      (op_station("0.75"), op_station("1.5")),
                      note="test_direct_ancestor_versus_double_tolerance", inventory=("B2", "B3")))
    cases.append(Case("native_arc_station/split_cylinder", _split_cylinder_fixture,
                      (Operation("station_selection", op_station_selection),),
                      note="test_direct_ancestor_versus_double_tolerance (current_bound == 0 selection)",
                      inventory=("B2", "B7")))
    return cases


ADVERSARIAL = tuple(_cases())
G0 = tuple(_g0_cases())
HEAVY = {"g0/grid32"}
CASES = {case.name: case for case in (*ADVERSARIAL, *G0)}


def default_cases() -> list[str]:
    return [name for name in CASES if name not in HEAVY]


if __name__ == "__main__":
    import time
    for case in CASES.values():
        start = time.perf_counter()
        model, context = case.build()
        for operation in case.operations:
            model, context = case.build()
            try:
                outcome = operation.run(model, context, False)
            except Exception as error:  # noqa: BLE001
                outcome = f"error {type(error).__name__}: {error}"
            print(f"{case.name:36s} {operation.name:22s} {json.dumps(outcome, default=str)[:150]}")
        print(f"{'':36s} {time.perf_counter() - start:.2f}s", flush=True)
