"""OPT-S: refusal-only retry of a Plane arrangement with rotated decomposition-seam directions.

Reproducers (``data/seamrot_caps_arrangement.json``, captured arrange_material inputs of the AIP-Gen2 tripod
heave-plate caps):

* ``kept_1082_449`` / ``original_1083_809``: one cylinder-wall arc trace with both ends interior. The seam
  perpendicular to the authored ``u`` through an arc end is tangent to the arc there; the two
  line/circle roots (about 2e-7 m apart) become two vertices joined by a seam piece and an arc piece that
  coincide within about 4e-15 m but are not merged (no line/arc duplicate rule), and the cyclic order at
  the two ends is not planar: "unresolved bridge".
* ``reframe_1082_121``: two plate ends lie on one seam line within 8.5e-10 m (one vertex), their shallow
  boundary hits 1.056e-9 m apart (two vertices): non-transitive tolerance identity, "unresolved bridge".
* ``original_1082_121``: the seam-carried vertex gaps of the float cell route: "failed material
  conservation" (ratio 8.34).

Each refuses on the unchanged route and is accepted after one retry with both seam directions rotated by
(3/5, 4/5), with every check and tolerance unchanged.
"""
import json
import pathlib

import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryError
from anygeometry import material_arrangement as ma
from anygeometry.arrangement_geometry import LinePath
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.surfaces import Plane

DATA = pathlib.Path(__file__).resolve().parent / "data" / "seamrot_caps_arrangement.json"
BRIDGE = "arrangement contains an unresolved bridge"
CONSERVATION = "arrangement failed material conservation"


def _curve(item):
    if item["kind"] == "line":
        return LinePath(tuple(item["start"]), tuple(item["end"]))
    return EllipticArc(tuple(item["center"]), tuple(item["u_vector"]), tuple(item["v_vector"]),
                       item["start_angle"], item["sweep_angle"])


def _case(name):
    record = json.loads(DATA.read_text())["cases"][name]
    support = Plane(*(tuple(record["support"][key]) for key in ("origin", "u_vector", "v_vector")))
    boundary = tuple(ArrangementPath(_curve(item), source_edge=item["source_edge"]) for item in record["boundary"])
    traces = tuple(ArrangementPath(_curve(item), owners=tuple(item["owners"])) for item in record["traces"])
    return MaterialDomain(record["face_id"], support, (boundary,)), traces, record["tolerance"], record["area_tolerance"]


def _arrange(name, **extra):
    domain, traces, tolerance, area_tolerance = _case(name)
    return arrange_material(domain, traces, tolerance=tolerance, area_tolerance=area_tolerance, **extra)


def _assert_conserved(name, arrangement):
    """Material conservation recomputed from the RETURNED cells (outer loops and holes rebuilt from the paths)."""
    domain, _traces, _tolerance, area_tolerance = _case(name)
    native = area_tolerance / domain.area_jacobian
    original = abs(domain.area_loop(domain.boundaries[0], native * .05))

    def loop_area(loop):
        pieces = tuple(ArrangementPath(arrangement.paths[i].curve if forward
                                       else arrangement.paths[i].curve.subcurve(1., 0.)) for i, forward in loop)
        return domain.area_loop(pieces, native * .05)
    outers = [loop_area(cell.outer) for cell in arrangement.cells]
    assert all(area > 0 for area in outers)
    recomputed = sum(outers) + sum(loop_area(hole) for cell in arrangement.cells for hole in cell.holes)
    assert abs(original - recomputed) <= native
    assert abs(original - arrangement.area) <= native
    assert arrangement.native_area_tolerance == native
    assert arrangement.face_id == domain.face_id


def _seam_directions(name, arrangement):
    """Unit directions (in the support's orthonormal authored frame) of the returned decomposition seams."""
    domain = _case(name)[0]
    u = np.asarray(domain.support.u_vector) / np.linalg.norm(domain.support.u_vector)
    n = np.cross(domain.support.u_vector, domain.support.v_vector)
    w = np.cross(n / np.linalg.norm(n), u)
    found = set()
    for path in arrangement.paths:
        if path.decomposition and not path.owners:
            d = np.asarray(path.curve.end) - np.asarray(path.curve.start)
            d = d / np.linalg.norm(d)
            c, s = float(d @ u), float(d @ w)
            if c < -1e-9 or (abs(c) <= 1e-9 and s < 0):
                c, s = -c, -s
            found.add((round(c, 9), round(s, 9)))
    return found


@pytest.mark.parametrize("name", ["kept_1082_449", "original_1083_809"])
def test_arc_end_tangent_seam_bridge_is_accepted_after_rotation(name):
    with collect_runtime_diagnostics() as counters:
        arrangement = _arrange(name)
    assert counters["arrangement_seam_rotation_retries"] == 1
    assert counters["arrangement_seam_rotation_accepted"] == 1
    assert len(arrangement.cells) == 10
    _assert_conserved(name, arrangement)
    # Every returned seam runs along the rotated frame: (3/5, 4/5) or its perpendicular (-4/5, 3/5).
    assert _seam_directions(name, arrangement) <= {(0.6, 0.8), (0.8, -0.6)}
    assert _seam_directions(name, arrangement)


def test_original_1082_subset_conservation_refusal_is_accepted_after_rotation():
    with collect_runtime_diagnostics() as counters:
        arrangement = _arrange("original_1082_121")
    assert counters["arrangement_seam_rotation_retries"] == 1
    assert counters["arrangement_seam_rotation_accepted"] == 1
    assert len(arrangement.cells) > 1000
    _assert_conserved("original_1082_121", arrangement)


def test_reframe_1082_subset_collinear_seam_bridge_is_accepted_after_rotation():
    with collect_runtime_diagnostics() as counters:
        arrangement = _arrange("reframe_1082_121")
    assert counters["arrangement_seam_rotation_retries"] == 1
    assert counters["arrangement_seam_rotation_accepted"] == 1
    assert len(arrangement.cells) > 1000
    _assert_conserved("reframe_1082_121", arrangement)


def _spy(monkeypatch, after=None):
    """Record each attempt's rotation and outcome; ``after(attempt, outcome)`` runs after each attempt."""
    calls = []
    once = ma._arrange_material_once

    def wrapped(*args, **kwargs):
        try:
            result = once(*args, **kwargs)
        except BaseException as error:
            calls.append((kwargs.get("seam_rotation"), error))
            if after:
                after(len(calls), error)
            raise
        calls.append((kwargs.get("seam_rotation"), result))
        if after:
            after(len(calls), result)
        return result
    monkeypatch.setattr(ma, "_arrange_material_once", wrapped)
    return calls


def test_failed_retry_raises_the_original_error(monkeypatch):
    # (1, 0) rebuilds the same seams, so the retry refuses as well: the FIRST attempt's error object is raised.
    monkeypatch.setattr(ma, "_SEAM_RETRY_ROTATION", (1.0, 0.0))
    calls = _spy(monkeypatch)
    with collect_runtime_diagnostics() as counters, pytest.raises(GeometryError, match=BRIDGE) as raised:
        _arrange("kept_1082_449")
    assert [rotation for rotation, _ in calls] == [None, (1.0, 0.0)]
    assert raised.value is calls[0][1] and raised.value is not calls[1][1]
    assert counters["arrangement_seam_rotation_retries"] == 1
    assert counters.get("arrangement_seam_rotation_accepted", 0) == 0


def test_retry_failing_with_another_error_raises_the_original(monkeypatch):
    def broken_seams(*args, **kwargs):
        if kwargs.get("rotation") is not None:
            raise ZeroDivisionError("retry-only failure")
        return seams(*args, **kwargs)
    seams = ma._add_exact_planar_seams
    monkeypatch.setattr(ma, "_add_exact_planar_seams", broken_seams)
    calls = _spy(monkeypatch)
    with pytest.raises(GeometryError, match=BRIDGE) as raised:
        _arrange("kept_1082_449")
    assert raised.value is calls[0][1]
    assert isinstance(calls[1][1], ZeroDivisionError)


def test_cancellation_during_the_retry_propagates(monkeypatch):
    state = {"cancel": False, "polls_after": 0}

    def cancellation():
        if state["cancel"]:
            state["polls_after"] += 1
        return state["cancel"]

    def broken_seams(*args, **kwargs):
        # Request cancellation while the retry builds its seams; the retry's own check() must see it.
        if kwargs.get("rotation") is not None:
            state["cancel"] = True
        return seams(*args, **kwargs)
    seams = ma._add_exact_planar_seams
    monkeypatch.setattr(ma, "_add_exact_planar_seams", broken_seams)
    calls = _spy(monkeypatch)
    with pytest.raises(GeometryError, match="cancelled"):
        _arrange("kept_1082_449", cancellation_check=cancellation)
    assert [rotation for rotation, _ in calls] == [None, ma._SEAM_RETRY_ROTATION]
    assert "cancelled" in str(calls[1][1])
    assert state["polls_after"] >= 1


def test_cancellation_seen_after_an_unrelated_retry_failure_is_reported_as_cancellation(monkeypatch):
    # Review OPT-S P2: the retry fails with an unrelated error while cancellation has been requested; the
    # caller gets the cancellation (not the unrelated error, not the original refusal).
    state = {"cancel": False}

    def broken_seams(*args, **kwargs):
        if kwargs.get("rotation") is not None:
            state["cancel"] = True
            raise ZeroDivisionError("retry-only failure")
        return seams(*args, **kwargs)
    seams = ma._add_exact_planar_seams
    monkeypatch.setattr(ma, "_add_exact_planar_seams", broken_seams)
    with pytest.raises(GeometryError, match="^intersection arrangement cancelled$"):
        _arrange("kept_1082_449", cancellation_check=lambda: state["cancel"])


def test_cancellation_requested_before_the_retry_is_raised(monkeypatch):
    state = {"cancel": False}
    calls = _spy(monkeypatch, after=lambda attempt, _outcome: state.update(cancel=True))
    with pytest.raises(GeometryError, match="intersection arrangement cancelled"):
        _arrange("kept_1082_449", cancellation_check=lambda: state["cancel"])
    assert len(calls) == 1


def test_no_retry_under_an_explicit_predicate_budget(monkeypatch):
    calls = _spy(monkeypatch)
    with pytest.raises(GeometryError, match=BRIDGE):
        _arrange("kept_1082_449", max_predicates=10 ** 9)
    assert len(calls) == 1


def test_no_retry_without_seams(monkeypatch):
    # The cap boundary alone has no interior trace end, so no seams: a refusal there is not retried.
    domain, _traces, tolerance, area_tolerance = _case("kept_1082_449")
    once = ma._arrange_material_once
    calls = []

    def refusing(*args, **kwargs):
        calls.append(kwargs.get("seam_rotation"))
        once(*args, **kwargs)
        assert kwargs["record"]["seams"] == 0
        raise GeometryError(CONSERVATION)
    monkeypatch.setattr(ma, "_arrange_material_once", refusing)
    with pytest.raises(GeometryError, match=CONSERVATION):
        arrange_material(domain, (), tolerance=tolerance, area_tolerance=area_tolerance)
    assert calls == [None]


def test_other_refusals_are_not_retried(monkeypatch):
    calls = []

    def refusing(*args, **kwargs):
        calls.append(kwargs.get("seam_rotation"))
        kwargs["record"]["seams"] = 4
        raise GeometryError("arrangement half-edge cycle is inconsistent")
    monkeypatch.setattr(ma, "_arrange_material_once", refusing)
    with pytest.raises(GeometryError, match="inconsistent"):
        _arrange("kept_1082_449")
    assert calls == [None]


def test_accepted_first_attempt_is_returned_without_retry(monkeypatch):
    # A Plane face with seams (one interior plate trace) that the unchanged route accepts: one attempt only.
    support = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    corners = [(0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.)]
    boundary = tuple(ArrangementPath(LinePath(a, b), source_edge=k) for k, (a, b) in
                     enumerate(zip(corners, corners[1:] + corners[:1])))
    domain = MaterialDomain(7, support, (boundary,))
    traces = (ArrangementPath(LinePath((0.5, 0.25, 0.), (1.5, 0.75, 0.)), owners=(8,)),)
    calls = _spy(monkeypatch)
    with collect_runtime_diagnostics() as counters:
        arrangement = arrange_material(domain, traces, tolerance=1e-9)
    assert [rotation for rotation, _ in calls] == [None]
    assert calls[0][1] is arrangement
    assert counters.get("arrangement_seam_rotation_retries", 0) == 0
    assert any(path.decomposition for path in arrangement.paths)
