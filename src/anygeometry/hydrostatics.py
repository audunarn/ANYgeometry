"""Hydrostatics of a closed shell at a waterplane (generality slice H1b, for ANYstability).

``hydrostatics(model, faces, point, normal)`` takes a closed, consistently oriented shell (``closed_shell``)
and a waterplane through ``point`` with the upward unit ``normal`` (heel and trim are just its direction).
Everything is a clipped hull-surface integral of ``face_moments`` below the plane, in the waterplane frame
(``xi`` along the longitudinal axis projected into the plane, ``eta = n x xi`` transverse, ``zeta`` up):

- displaced volume and centre of buoyancy (fields ``zeta, xi zeta, eta zeta, zeta^2 / 2``: ``F = phi n``
  vanishes on the waterplane, so no waterplane face is built);
- waterplane area, its centroid (centre of flotation) and second moments (divergence-free fields
  ``1, xi, eta, xi^2, xi eta, eta^2`` times ``n``: the waterplane integral is minus the hull integral);
- ``I_T`` (about the longitudinal axis through the centre of flotation), ``I_L`` (about the transverse
  axis), ``BM_T = I_T / V``, ``BM_L = I_L / V``.

Values are quadratures with an order-doubling ``estimate``; the call refuses (``GeometryError``) when a
quantity has not settled to ``rtol`` of its scale (from the whole shell's volume and extent). A shell
entirely above the plane displaces nothing (the waterplane quantities are then zero and BM undefined).
"""
from __future__ import annotations

import math
import weakref

import numpy as np

from .errors import GeometryError
from .face_moments import Frame, shell_moments, volume_fields, waterplane_fields


def waterplane_frame(point, normal=(0., 0., 1.), longitudinal=(1., 0., 0.)) -> Frame:
    """The waterplane frame: ``xi`` along ``longitudinal`` projected into the plane, ``zeta`` along ``normal``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    e1 = np.asarray(longitudinal, dtype=float)
    e1 = e1 - (e1 @ n) * n
    if not float(np.linalg.norm(e1)) > 1e-12:
        raise GeometryError("the longitudinal direction must not be normal to the waterplane")
    return Frame(point, e1, np.cross(n, e1))


_SHELLS: dict = {}            # id(model) -> (weak reference, revision, faces, order, value)


def _shell(model, faces, order):
    """``(sign, volume, vertex mean, extent)`` of the whole shell (unclipped, in a world frame at its
    vertex mean): the loops' orientation and the refusal scales. Cached per model revision and faces
    (a stability sweep evaluates many waterplanes of one shell)."""
    # keyed on the model object itself (a weak reference checked on use) and its revision: clones that
    # share an identity and a revision can differ (review H1 round 1)
    entry = _SHELLS.get(id(model))
    if entry is not None and entry[0]() is model and entry[1:4] == (model.revision, tuple(faces), order):
        return entry[4]
    positions = np.asarray([model.vertices[v].position for v in
                            {model.oriented_start_vertex(item) for face_id in faces
                             for loop in (model.faces[face_id].loop,) + tuple(model.faces[face_id].holes)
                             for item in loop}], dtype=float)
    centre = positions.mean(axis=0)
    whole = shell_moments(model, faces, Frame.world(centre), volume_fields, order)
    sign = 1.0 if whole[0] > 0 else -1.0
    total = sign * float(whole[0])
    if not total > 0:
        raise GeometryError("the shell encloses no volume")
    extent = max(float(np.max(np.linalg.norm(positions - centre, axis=1))), 1e-300)
    if len(_SHELLS) >= 32:
        _SHELLS.clear()
    value = (sign, total, centre, extent)
    _SHELLS[id(model)] = (weakref.ref(model), model.revision, tuple(faces), order, value)
    return value


def hydrostatics(model, faces, point, normal=(0., 0., 1.), longitudinal=(1., 0., 0.), order: int = 24,
                 rtol: float = 1e-8) -> dict:
    from .shell_volume import closed_shell
    faces = list(faces)
    problems = closed_shell(model, faces)
    if problems:
        raise GeometryError("not a closed shell: " + "; ".join(problems))
    frame = waterplane_frame(point, normal, longitudinal)
    sign, total, centre, extent = _shell(model, faces, order)
    # the same plane with its origin at the foot of the shell's centre (review H1 round 1: a far origin
    # cancelled the central waterplane moments)
    foot = centre - float((centre - frame.origin) @ frame.normal) * frame.normal
    frame = Frame(foot, frame.axes[0], frame.axes[1])
    # clipped moments at two orders
    results = []
    def both(local):
        return np.hstack((volume_fields(local), waterplane_fields(local)))
    for k in (order, 2 * order):
        moments = shell_moments(model, faces, frame, both, k, clip=True)
        results.append((sign * moments[:4], -sign * moments[4:]))
    (v1, p1), (v2, p2) = results
    # scales: volume ~ V, first moments ~ V L (here the frame origin may lie off the shell: use the
    # distance of the frame origin to the shell as well), area ~ L^2, area moments ~ L^3, L^4
    reach = extent + float(np.linalg.norm(frame.origin - centre))
    scales_v = np.array([total, total * reach, total * reach, total * reach])
    scales_p = np.array([extent ** 2, extent ** 2 * reach, extent ** 2 * reach] + [extent ** 2 * reach ** 2] * 3)
    dv, dp = np.abs(v2 - v1), np.abs(p2 - p1)
    if not (np.all(dv <= rtol * scales_v) and np.all(dp <= rtol * scales_p)):
        raise GeometryError(f"the hydrostatic quadrature did not settle (differences {float(np.max(dv)):.3g}, "
                            f"{float(np.max(dp)):.3g})")
    volume = float(v2[0])
    area = float(p2[0])
    result = {"volume": volume, "waterplane_area": area,
              "estimate": float(max(np.max(dv / scales_v), np.max(dp / scales_p))), "frame": frame}
    if volume > 0:
        local_b = v2[1:] / volume
        result["buoyancy_centre"] = frame.origin + frame.axes.T @ local_b
    else:
        result["buoyancy_centre"] = None
    if area > 0:
        xi_f, eta_f = float(p2[1]) / area, float(p2[2]) / area
        i_xx = float(p2[3]) - area * xi_f * xi_f        # int xi^2: about the transverse axis (longitudinal)
        i_xy = float(p2[4]) - area * xi_f * eta_f
        i_yy = float(p2[5]) - area * eta_f * eta_f      # int eta^2: about the longitudinal axis (transverse)
        result.update(flotation_centre=frame.origin + xi_f * frame.axes[0] + eta_f * frame.axes[1],
                      i_transverse=i_yy, i_longitudinal=i_xx, i_product=i_xy)
        if volume > 0:
            result.update(bm_transverse=i_yy / volume, bm_longitudinal=i_xx / volume)
    else:
        result.update(flotation_centre=None, i_transverse=0.0, i_longitudinal=0.0, i_product=0.0)
    return result


# ------------------------------------------------------------------ sweeps for stability (H1c)


def heel_normal(heel: float, trim: float = 0.0) -> np.ndarray:
    """The upward waterplane normal of a hull heeled by ``heel`` (radians, positive: the +y side goes
    down, the waterplane rises towards +y) and trimmed by ``trim`` (radians, positive: the +x end goes
    down), in the hull's (body) axes."""
    n = np.array([-math.sin(trim), -math.sin(heel) * math.cos(trim), math.cos(heel) * math.cos(trim)])
    return n / np.linalg.norm(n)


def _hull_points(model, faces) -> np.ndarray:
    from .nurbs import NurbsSurface
    from .support_curves import PlaneNurbsCurve
    points, surfaces = [], []
    for face_id in faces:
        face = model.faces[face_id]
        if isinstance(face.surface, NurbsSurface):
            surfaces.append(face.surface)
        for loop in (face.loop,) + tuple(face.holes):
            for item in loop:
                edge = model.edges[item.edge]
                points += [model.vertices[edge.start].position, model.vertices[edge.end].position]
                if isinstance(edge.curve, PlaneNurbsCurve):
                    surfaces.append(edge.curve.surface)
    for surface in surfaces:
        points += [point for row in surface.controls for point in row]
    return np.asarray(points, dtype=float)


def float_at(model, faces, volume: float, normal=(0., 0., 1.), reference=(0., 0., 0.),
             longitudinal=(1., 0., 0.), start: float | None = None, order: int = 24, rtol: float = 1e-8,
             tolerance: float = 1e-12, max_iterations: int = 40) -> dict:
    """The waterplane of the given ``normal`` that displaces ``volume``: the plane through
    ``reference + zeta n`` with ``zeta`` from Newton (``dV / dzeta`` is the waterplane area), safeguarded
    by bisection. Returns the ``hydrostatics`` result with ``zeta`` added."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    reference = np.asarray(reference, dtype=float)
    faces = list(faces)
    # a bracket from points that bound the shell: its vertices and the control nets of its NURBS
    # surfaces (rational, positive weights: the surface lies in the net's convex hull; section edges
    # lie on such surfaces) (review H1 round 1: vertices alone missed a bottom below them)
    heights = (_hull_points(model, faces) - reference) @ n
    span = float(np.max(heights) - np.min(heights)) or 1.0
    low, high = float(np.min(heights)) - 1e-9 * span, float(np.max(heights)) + 1e-9 * span
    zeta = .5 * (low + high) if start is None or not low < start < high else float(start)

    def at(z):
        return hydrostatics(model, faces, reference + z * n, n, longitudinal, order, rtol)
    result = at(zeta)
    for _ in range(max_iterations):
        error = result["volume"] - volume
        if abs(error) <= tolerance * max(volume, 1e-300):
            break
        if error > 0:
            high = min(high, zeta)
        else:
            low = max(low, zeta)
        area = result["waterplane_area"]
        step = zeta - error / area if area > 0 else None
        zeta = step if step is not None and low < step < high else .5 * (low + high)
        result = at(zeta)
    else:
        raise GeometryError("float_at: no waterplane found for the displacement")
    result["zeta"] = zeta
    return result


def hydrostatic_table(model, faces, drafts, reference=(0., 0., 0.), normal=(0., 0., 1.), **options) -> list:
    """``hydrostatics`` at the waterplanes through ``reference + draft n`` for each draft."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    rows = []
    for draft in drafts:
        row = hydrostatics(model, faces, np.asarray(reference, dtype=float) + draft * n, n, **options)
        row["draft"] = float(draft)
        rows.append(row)
    return rows


def gz_curve(model, faces, volume: float, gravity_centre, heels, trim: float = 0.0,
             reference=(0., 0., 0.), **options) -> list:
    """Righting levers at the given heels (radians) for a displacement ``volume`` and a centre of
    gravity (body axes), with free sinkage and fixed trim: at each heel the waterplane with
    ``heel_normal(heel, trim)`` that displaces ``volume`` (``float_at``), and ``GZ = (B - G) . eta``
    with ``eta`` the waterplane's transverse axis (positive: righting). Rows: heel, gz, zeta, result."""
    g = np.asarray(gravity_centre, dtype=float)
    rows, start = [], None
    for heel in heels:
        result = float_at(model, faces, volume, heel_normal(heel, trim), reference, start=start, **options)
        start = result["zeta"]
        eta = result["frame"].axes[1]
        rows.append({"heel": float(heel), "gz": float((result["buoyancy_centre"] - g) @ eta),
                     "zeta": start, "result": result})
    return rows
