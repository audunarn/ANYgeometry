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


def hydrostatics(model, faces, point, normal=(0., 0., 1.), longitudinal=(1., 0., 0.), order: int = 24,
                 rtol: float = 1e-8) -> dict:
    from .shell_volume import closed_shell
    faces = list(faces)
    problems = closed_shell(model, faces)
    if problems:
        raise GeometryError("not a closed shell: " + "; ".join(problems))
    frame = waterplane_frame(point, normal, longitudinal)
    positions = np.asarray([model.vertices[v].position for v in
                            {model.oriented_start_vertex(item) for face_id in faces
                             for loop in (model.faces[face_id].loop,) + tuple(model.faces[face_id].holes)
                             for item in loop}], dtype=float)
    # the whole shell (unclipped) in a frame at its vertex mean: orientation and scales
    centre = Frame(positions.mean(axis=0), frame.axes[0], frame.axes[1])
    whole = shell_moments(model, faces, centre, volume_fields, order)
    sign = 1.0 if whole[0] > 0 else -1.0
    total = sign * float(whole[0])
    if not total > 0:
        raise GeometryError("the shell encloses no volume")
    extent = max(float(np.max(np.linalg.norm(positions - centre.origin, axis=1))), 1e-300)
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
    reach = extent + float(np.linalg.norm(frame.origin - centre.origin))
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
