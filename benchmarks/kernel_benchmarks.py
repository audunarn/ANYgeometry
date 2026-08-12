"""Deterministic scaling benchmarks for the strict geometry kernel.

The benchmark deliberately uses only public APIs.  It is not a microbenchmark
and it does not impose machine-specific wall-clock thresholds.  Qualification
compares entity visits and broad-/narrow-phase counts as well as elapsed time
and peak Python allocation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Callable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anygeometry import EntityRef, GeometryModel, OrientedEdge, Plane, to_dict  # noqa: E402
from anygeometry.generators import cylinder, stiffened_panel  # noqa: E402


@contextmanager
def measured(
    results: dict[str, Any], name: str, *, track_python_memory: bool = True
) -> Iterator[None]:
    if track_python_memory:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        result = {"seconds": elapsed}
        if track_python_memory:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result["peak_python_bytes"] = peak
        results[name] = result


def counts(model: GeometryModel) -> dict[str, int]:
    result = {
        "vertices": len(model.vertices),
        "edges": len(model.edges),
        "faces": len(model.faces),
    }
    for name in ("parts", "sheets", "members", "attachments", "junctions"):
        value = getattr(model, name, None)
        if value is not None:
            result[name] = len(value)
    return result


def untraced_seconds(factory: Callable[[], object]) -> float:
    """Measure normal runtime separately from allocation-instrumented runtime."""

    started = time.perf_counter()
    value = factory()
    elapsed = time.perf_counter() - started
    del value
    return elapsed


def grid(nx: int, ny: int) -> GeometryModel:
    model = GeometryModel()
    positions = ((float(i), float(j), 0.0) for j in range(ny + 1) for i in range(nx + 1))
    made = model.add_points(positions)

    def vertex(i: int, j: int) -> int:
        return made[j * (nx + 1) + i]

    horizontal: dict[tuple[int, int], int] = {}
    vertical: dict[tuple[int, int], int] = {}
    for j in range(ny + 1):
        for i in range(nx):
            horizontal[i, j] = model.add_line(vertex(i, j), vertex(i + 1, j))
    for j in range(ny):
        for i in range(nx + 1):
            vertical[i, j] = model.add_line(vertex(i, j), vertex(i, j + 1))
    for j in range(ny):
        for i in range(nx):
            loop = (
                OrientedEdge(horizontal[i, j], True),
                OrientedEdge(vertical[i + 1, j], True),
                OrientedEdge(horizontal[i, j + 1], False),
                OrientedEdge(vertical[i, j], False),
            )
            model.add_face_from_loop(
                loop,
                (0, 1, 2, 3),
                surface=Plane(
                    np.asarray((float(i), float(j), 0.0)),
                    np.asarray((1.0, 0.0, 0.0)),
                    np.asarray((0.0, 1.0, 0.0)),
                ),
            )
    return model


def member_lattice(count: int) -> GeometryModel:
    model = GeometryModel()
    positions: list[tuple[float, float, float]] = []
    for index in range(count):
        row, column = divmod(index, 100)
        positions.extend(
            (
                (float(column), float(row), 0.0),
                (float(column) + 0.75, float(row), 0.0),
            )
        )
    vertices = model.add_points(positions)
    edges = [
        model.add_line(vertices[2 * index], vertices[2 * index + 1])
        for index in range(count)
    ]
    add_member = getattr(model, "add_member", None)
    add_members = getattr(model, "add_members", None)
    if add_members is not None:
        part = model.add_part(name="benchmark lattice")
        add_members(((edge,) for edge in edges), part_id=part)
    elif add_member is not None:
        part = model.add_part(name="benchmark lattice")
        for edge in edges:
            add_member((edge,), part_id=part)
    else:
        model.add_to_group("benchmark_members", (EntityRef("edge", edge) for edge in edges))
    return model


def duplicate_crossing_model(count: int) -> GeometryModel:
    model = GeometryModel()
    member_edges: list[int] = []
    for index in range(count):
        offset = float(index) * 2.0
        a, b, c, d = model.add_points(
            (
                (offset, 0.0, 0.0),
                (offset + 1.0, 1.0, 0.0),
                (offset, 1.0, 0.0),
                (offset + 1.0, 0.0, 0.0),
            )
        )
        first = model.add_line(a, b)
        model.add_line(a, b)
        model.add_line(c, d)
        member_edges.append(first)
    add_members = getattr(model, "add_members", None)
    if add_members is not None:
        part = model.add_part(name="hostile members")
        add_members(((edge,) for edge in member_edges), part_id=part)
    else:
        add_member = getattr(model, "add_member", None)
        if add_member is not None:
            part = model.add_part(name="hostile members")
            for edge in member_edges:
                add_member((edge,), part_id=part)
    return model


def audit_diagnostics(model: GeometryModel) -> dict[str, Any]:
    audit = getattr(model, "strict_audit", None)
    if audit is None:
        return {"available": False}
    report = audit()
    metrics = getattr(report, "metrics", None)
    return {
        "available": True,
        "clean": bool(getattr(report, "clean", False)),
        "issues": len(getattr(report, "issues", ())),
        "candidate_count": int(getattr(metrics, "candidate_count", 0)),
        "classified_count": int(getattr(metrics, "classified_count", 0)),
        "unclassified_count": int(getattr(metrics, "unclassified_count", 0)),
        "narrow_phase_count": int(getattr(metrics, "narrow_phase_tests", 0)),
        "index_node_visits": int(getattr(metrics, "index_node_visits", 0)),
        "index_leaf_tests": int(getattr(metrics, "index_leaf_tests", 0)),
        "index_updates": int(getattr(metrics, "index_updates", 0)),
    }


def run(*, qualification: bool) -> dict[str, Any]:
    face_side = 100 if qualification else 20
    member_count = 10_000 if qualification else 500
    hostile_count = 2_000 if qualification else 100
    lineage_length = 1_000 if qualification else 100
    results: dict[str, Any] = {
        "profile": "qualification" if qualification else "smoke",
        "parameters": {
            "plate_grid": [face_side, face_side],
            "members": member_count,
            "hostile_sets": hostile_count,
            "lineage_length": lineage_length,
        },
        "measurements": {},
    }
    measurements: dict[str, Any] = results["measurements"]

    holder: dict[str, GeometryModel] = {}
    with measured(measurements, "plate_grid_construction"):
        holder["plate"] = grid(face_side, face_side)
    measurements["plate_grid_construction"]["untraced_seconds"] = untraced_seconds(
        lambda: grid(face_side, face_side)
    )
    plate = holder["plate"]
    measurements["plate_grid_construction"]["entities"] = counts(plate)

    # Qualify the intentionally conformal grid before the edit below deforms
    # one shared vertex out of its neighbouring planar supports.  The edit is
    # a transaction/index benchmark, not a valid final design state.
    with measured(
        measurements,
        "plate_grid_full_audit",
        # Tracemalloc multiplies the full audit runtime several-fold.  The
        # 10k qualification records deterministic work counts instead; the
        # smoke profile retains peak allocation measurement.
        track_python_memory=not qualification,
    ):
        plate_audit = audit_diagnostics(plate)
    measurements["plate_grid_full_audit"].update(plate_audit)

    center = (face_side // 2) * (face_side + 1) + face_side // 2 + 1
    with measured(measurements, "large_model_local_edit"):
        position = plate.vertex_position(center)
        plate.move_point(center, float(position[0]), float(position[1]), 0.125)
    change_set = getattr(plate, "last_change_set", None)
    if change_set is not None:
        measurements["large_model_local_edit"]["changed"] = len(
            getattr(change_set, "modified", ())
        )
        measurements["large_model_local_edit"]["index_updates"] = len(
            getattr(change_set, "spatial_updates", ())
        )

    with measured(measurements, "plate_grid_serialization"):
        document = to_dict(plate)
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    untraced_started = time.perf_counter()
    untraced_document = to_dict(plate)
    json.dumps(
        untraced_document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    measurements["plate_grid_serialization"]["untraced_seconds"] = (
        time.perf_counter() - untraced_started
    )
    measurements["plate_grid_serialization"]["document_records"] = sum(
        len(document.get(key, ())) for key in ("vertices", "edges", "faces")
    )

    with measured(measurements, "member_lattice_construction"):
        holder["members"] = member_lattice(member_count)
    measurements["member_lattice_construction"]["untraced_seconds"] = untraced_seconds(
        lambda: member_lattice(member_count)
    )
    measurements["member_lattice_construction"]["entities"] = counts(holder["members"])

    panel_size = 50.0 if qualification else 10.0
    with measured(measurements, "mixed_stiffened_panel_construction"):
        holder["panel"] = stiffened_panel(
            panel_size,
            panel_size / 2.0,
            longitudinal_spacing=0.5,
            transverse_spacing=1.0,
        )
    measurements["mixed_stiffened_panel_construction"]["untraced_seconds"] = (
        untraced_seconds(
            lambda: stiffened_panel(
                panel_size,
                panel_size / 2.0,
                longitudinal_spacing=0.5,
                transverse_spacing=1.0,
            )
        )
    )
    measurements["mixed_stiffened_panel_construction"]["entities"] = counts(holder["panel"])

    with measured(measurements, "cylinder_member_construction"):
        holder["cylinder"] = cylinder(
            8.0,
            20.0,
            circumferential_segments=128 if qualification else 16,
            ring_spacing=0.5,
        )
    measurements["cylinder_member_construction"]["untraced_seconds"] = (
        untraced_seconds(
            lambda: cylinder(
                8.0,
                20.0,
                circumferential_segments=128 if qualification else 16,
                ring_spacing=0.5,
            )
        )
    )
    measurements["cylinder_member_construction"]["entities"] = counts(holder["cylinder"])

    with measured(measurements, "duplicate_crossing_construction"):
        holder["hostile"] = duplicate_crossing_model(hostile_count)
    measurements["duplicate_crossing_construction"]["entities"] = counts(holder["hostile"])
    with measured(
        measurements,
        "duplicate_crossing_full_audit",
        track_python_memory=not qualification,
    ):
        hostile_audit = audit_diagnostics(holder["hostile"])
    measurements["duplicate_crossing_full_audit"].update(hostile_audit)

    lineage = GeometryModel()
    start, end = lineage.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    edge = lineage.add_line(start, end)
    original = EntityRef("edge", edge)
    active = edge
    with measured(measurements, "replacement_lineage_construction"):
        for _ in range(lineage_length):
            # Retain the long descendant so the benchmark grows lineage rather
            # than eventually underflowing a repeatedly halved segment.
            _vertex, children = lineage.split_edge(active, 1.0e-4)
            active = children[1]
    with measured(measurements, "replacement_lineage_resolution"):
        resolved = lineage.resolve_ref(original)
    measurements["replacement_lineage_resolution"]["descendants"] = len(resolved)

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    results = run(qualification=options.qualification)
    encoded = json.dumps(results, indent=2, sort_keys=True)
    if options.output is not None:
        options.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
