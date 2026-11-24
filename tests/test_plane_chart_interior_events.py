"""Plane-chart route: a boundary event of one branch must not cut the other branch inside the patch.

The chart parameter ``t`` of a boundary-plane resultant root is shared by both branches of the plane/cylinder
intersection. The route retains every such root for classification (dropping one is not safe), so at a root where
branch -1 crosses a height plane, branch +1 used to be cut too although it is well inside the patch there. That cut
reached the prepared topology as a valence-2 vertex on the web root of the ANYfem floating tripod (36 webs: 6.2 mm
from the upper mitre; 60 webs: 91 mm) and made ANYmesh quad-first refuse.

The supports below are the STORED doubles of the 36-web tripod's Arm 1 inclined transition skin (face 39, a full
sweep Cylinder) and its web at 20 degrees (face 43, a Plane), taken from the prepared ANYfem 59612ee model.
"""
import json

import numpy as np
import pytest

from anygeometry import EntityRef, IntersectionBatchPolicy, apply_intersections, plan_intersections
from anygeometry.analytic_supports import plane_cylinder_support
from anygeometry.plane_ruled_support import patch_margins
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.surfaces import Cylinder, Plane

TOL = 1e-9

SKIN = Cylinder((50.94974746830583, 0.0, -1.4644660940672614), (-0.7071067811865477, 0.0, 0.7071067811865475),
                (0.7071067811865475, 0.0, 0.7071067811865477), 6.0, 28.870057685088806, -0.08726646259971647,
                6.283185307179586)
WEB = Plane((51.63815572471545, 2.0521208599540124, 5.820681942187218),
            (-0.2819077862357773, -0.10260604299770049, -0.11677002839743267),
            (-13.81747378252824, 0.0, 13.817473782528232))


def _interior_ends(result, cylinder, margin=1e-6):
    """Curve ends strictly inside the native patch (not on a height bound): chart splits that no boundary makes."""
    return [curve.evaluate(u) for curve in result.curves for u in (0.0, 1.0)
            if min(patch_margins(cylinder, curve.evaluate(u))) > margin]


def test_tripod_web_root_is_one_piece_per_branch():
    with collect_runtime_diagnostics() as counters:
        result = plane_cylinder_support(WEB, SKIN, tolerance=TOL)
    assert counters["plane_cylinder_plane_chart_routes"] == 1
    assert not _interior_ends(result, SKIN)
    assert len(result.curves) == 2 and not result.points
    for curve in result.curves:
        axial = [float((p - np.asarray(SKIN.origin)) @ np.asarray(SKIN.axis))
                 for p in curve.evaluate(np.array([0.0, 1.0]))]
        assert sorted(axial) == pytest.approx([0.0, SKIN.height], abs=TOL)
        for end in curve.evaluate(np.array([0.0, 1.0])):
            assert min(patch_margins(SKIN, end)) >= -TOL


def test_merged_branch_covers_the_split_pieces():
    """The merged chart is the same branch: every point of the old pieces' image lies on the new curve."""
    result = plane_cylinder_support(WEB, SKIN, tolerance=TOL)
    for curve in result.curves:
        samples = curve.evaluate(np.linspace(0.0, 1.0, 257))
        normal = np.cross(WEB.u_vector, WEB.v_vector)
        normal = normal / np.linalg.norm(normal)
        assert np.max(np.abs((samples - np.asarray(WEB.origin)) @ normal)) <= TOL
        rel = samples - np.asarray(SKIN.origin)
        axis = np.asarray(SKIN.axis)
        radial = rel - np.outer(rel @ axis, axis)
        assert np.max(np.abs(np.linalg.norm(radial, axis=1) - SKIN.radius)) <= TOL


def test_prepared_web_root_has_no_valence_two_vertex():
    """Plan + apply on the tripod skin and web faces alone: the shared web root is ONE edge (no stray vertex)."""
    import pathlib
    from anygeometry import serialization
    path = pathlib.Path(__file__).with_name("data") / "tripod36_skin39_web43.json"
    model = serialization.from_dict(json.loads(path.read_text()))
    policy = IntersectionBatchPolicy(seam_policy="ring_split")
    plan = plan_intersections(model, [EntityRef("face", face) for face in sorted(model.faces)], policy=policy)
    contact = [(arrangement.face_id, path) for arrangement in plan.arrangements for path in arrangement.paths
               if len(path.owners) > 1]
    assert sorted(face for face, _path in contact) == [39, 43], [
        (face, path.curve.evaluate(0.).round(5), path.curve.evaluate(1.).round(5)) for face, path in contact]
    apply_intersections(model, plan, policy=policy)
    shared = [edge for edge in model.edges if len(model.faces_using_edge(edge)) > 1]
    assert len(shared) == 1
    edge = model.edges[shared[0]]
    ends = sorted(tuple(np.round(model.vertex_position(v), 5)) for v in (edge.start, edge.end))
    assert ends == [(37.82068, 2.05212, 19.63816), (51.63816, 2.05212, 5.82068)]


def test_no_join_across_a_rounding_ambiguous_angular_bound():
    """A huge stored start angle makes the computed angular margin uncertain by about 2 eps |start| r (4e-4 m here):
    an apparent +7e-6 m margin at the angular crossing is NOT certified interior, so the crossing must stay a split
    (joining used to return one curve running 8e-7 m outside the patch). Case from Codex review round 1."""
    import math
    start = -1e12
    cylinder = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1e5, start, 0.5)
    normal = np.array([1., 0., 1e-11]) / np.linalg.norm([1., 0., 1e-11])
    plane = Plane((math.cos(start) + 5e-7, 0., 0.), (0., 1., 0.), tuple(np.cross(normal, [0., 1., 0.])))
    with collect_runtime_diagnostics() as counters:
        result = plane_cylinder_support(plane, cylinder, tolerance=1e-8)
    assert counters["plane_cylinder_plane_chart_routes"] == 1
    assert counters.get("plane_chart_interior_events_joined", 0) == 0
    assert len(result.curves) == 2
