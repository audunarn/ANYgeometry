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
from fractions import Fraction

import numpy as np

from .errors import GeometryError
from .face_moments import Frame, shell_moments, volume_fields, waterplane_fields


def _scaled_normal(normal) -> tuple:
    """The waterplane normal scaled by a power of two so its largest component lies in [1/2, 1):
    the same plane direction (exactly, except components more than ~2**1074 below the largest,
    which may round to zero), with lengths and offsets free of overflow and underflow for any
    finite normal. For ordinary normals every later norm, unit vector and offset is bit-identical
    to the unscaled one (power-of-two scaling is exact). Confirmation review of H1, round 2:
    subnormal and near-limit normals gave wrong planes or overflowed."""
    values = tuple(float(value) for value in normal)
    if not values or not all(math.isfinite(value) for value in values) or not any(values):
        raise GeometryError("the waterplane normal must be a finite non-zero vector")
    exponent = math.frexp(max(abs(value) for value in values))[1]
    return tuple(math.ldexp(value, -exponent) for value in values)


def waterplane_frame(point, normal=(0., 0., 1.), longitudinal=(1., 0., 0.)) -> Frame:
    """The waterplane frame: ``xi`` along ``longitudinal`` projected into the plane, ``zeta`` along ``normal``."""
    n = np.asarray(_scaled_normal(normal))
    n = n / np.linalg.norm(n)
    given = np.asarray(longitudinal, dtype=float)
    e1 = given - (given @ n) * n
    # refuse a longitudinal direction within about 1e-6 rad of the normal: the projection cancels and
    # its direction is not meaningful (review H1 round 4); then a second projection restores
    # orthogonality to rounding
    if not float(np.linalg.norm(e1)) > 1e-6 * float(np.linalg.norm(given)):
        raise GeometryError("the longitudinal direction is (nearly) normal to the waterplane")
    e1 = e1 / np.linalg.norm(e1)
    e1 = e1 - (e1 @ n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return Frame(point, e1, e2 / np.linalg.norm(e2), normal=n)


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
                 rtol: float = 1e-8, shift: float = 0.0) -> dict:
    """``shift``: the waterplane is moved by ``shift`` along the unit normal from ``point`` (kept exact:
    ``point + shift n`` in floats rounds away the offset for a point far from the shell, review H1
    round 5)."""
    raw_normal = _scaled_normal(normal)
    from .shell_volume import closed_shell
    faces = list(faces)
    problems = closed_shell(model, faces)
    if problems:
        raise GeometryError("not a closed shell: " + "; ".join(problems))
    frame = waterplane_frame(point, normal, longitudinal)
    sign, total, centre, extent = _shell(model, faces, order)
    # the same plane with its origin at the foot of the shell's centre (review H1 round 1: a far origin
    # cancelled the central waterplane moments)
    # the offset exactly from the given point and the given (unnormalised) normal: the plane is
    # raw_n . X = raw_n . point (review H1 rounds 2-3: a rounded unit normal tilts a plane given far away)
    offset = sum(Fraction(a) * (Fraction(float(c)) - Fraction(float(p)))
                 for a, c, p in zip(raw_normal, centre, point)) / Fraction(float(np.linalg.norm(raw_normal)))
    offset -= Fraction(float(shift))
    foot = centre - float(offset) * frame.normal
    frame = Frame(foot, frame.axes[0], frame.axes[1], normal=frame.normal)
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


def _height_range(points, reference, normal):
    """Lowest and highest of ``(point - reference) . normal / |normal|`` over ``points``, computed exactly
    (with ``|normal|`` as ``hydrostatics`` rounds it) and rounded outward."""
    normal = _scaled_normal(normal)
    raw = [Fraction(value) for value in normal]
    length = Fraction(float(np.linalg.norm(normal)))
    origin = [Fraction(float(value)) for value in reference]
    heights = [sum(a * (Fraction(float(x)) - r) for a, x, r in zip(raw, point, origin)) / length
               for point in np.asarray(points, dtype=float)]
    lowest, highest = min(heights), max(heights)
    low, high = float(lowest), float(highest)
    if low > lowest:
        low = math.nextafter(low, -math.inf)
    if high < highest:
        high = math.nextafter(high, math.inf)
    return low, high


def float_at(model, faces, volume: float, normal=(0., 0., 1.), reference=(0., 0., 0.),
             longitudinal=(1., 0., 0.), start: float | None = None, order: int = 24, rtol: float = 1e-8,
             tolerance: float = 1e-12, max_iterations: int = 40) -> dict:
    """The waterplane of the given ``normal`` that displaces ``volume``: the plane through
    ``reference + zeta n`` with ``zeta`` from Newton (``dV / dzeta`` is the waterplane area), safeguarded
    by bisection. Returns the ``hydrostatics`` result with ``zeta`` added."""
    # the caller's normal goes to hydrostatics unchanged: it takes the plane's offset exactly from
    # the unnormalised normal (confirmation review of H1: a pre-rounded unit normal tilts the plane
    # through a far reference point)
    normal = tuple(float(value) for value in normal)
    reference = np.asarray(reference, dtype=float)
    faces = list(faces)
    # a bracket from points that bound the shell: its vertices and the control nets of its NURBS
    # surfaces (rational, positive weights: the surface lies in the net's convex hull; section edges
    # lie on such surfaces) (review H1 round 1: vertices alone missed a bottom below them). Heights
    # along the normal exactly, rounded outward (float dot products cancel for a far reference).
    lowest, highest = _height_range(_hull_points(model, faces), reference, normal)
    span = (highest - lowest) or 1.0
    low, high = lowest - 1e-9 * span, highest + 1e-9 * span
    zeta = .5 * (low + high) if start is None or not low < start < high else float(start)
    total = _shell(model, faces, order)[1]

    def at(z):
        return hydrostatics(model, faces, reference, normal, longitudinal, order, rtol, shift=z)
    result = at(zeta)
    for _ in range(max_iterations):
        error = result["volume"] - volume
        # never tighter than the rounding of the whole shell's volume (review H1-F round 1: a tiny
        # displacement could not reach 1e-12 of itself)
        if abs(error) <= max(tolerance * volume, 64 * np.finfo(float).eps * total):
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
    normal = tuple(float(value) for value in normal)          # unrounded: hydrostatics takes it exactly
    rows = []
    for draft in drafts:
        row = hydrostatics(model, faces, np.asarray(reference, dtype=float), normal, shift=draft, **options)
        row["draft"] = float(draft)
        rows.append(row)
    return rows


def float_free_trim(model, faces, volume: float, gravity_centre, heel: float = 0.0, reference=(0., 0., 0.),
                    longitudinal=(1., 0., 0.), start=None, order: int = 24, rtol: float = 1e-8,
                    tolerance: float = 1e-11, max_iterations: int = 40) -> dict:
    """Free floating at a heel: the sinkage ``zeta`` and trim (radians) at which the waterplane with
    ``heel_normal(heel, trim)`` through ``reference + zeta n`` displaces ``volume`` and the centre of
    buoyancy lies on the normal through the centre of gravity longitudinally, ``(B - G) . xi = 0``
    (H1-F2). Newton on (zeta, trim) with a central-difference Jacobian and step halving; ``start`` is
    ``(zeta, trim)`` (default: ``float_at`` at zero trim). Returns the ``hydrostatics`` result with
    ``zeta`` and ``trim``; refuses when the residuals do not fall below ``tolerance`` (relative to the
    volume and to the shell's extent)."""
    g = np.asarray(gravity_centre, dtype=float)
    faces = list(faces)
    _sign, _total, centre, extent = _shell(model, faces, order)

    def evaluate(z, trim):
        result = hydrostatics(model, faces, reference, heel_normal(heel, trim), longitudinal, order, rtol,
                              shift=z)
        if result["buoyancy_centre"] is None:
            raise GeometryError("float_free_trim: the hull is dry at this waterplane")
        lever = float((result["buoyancy_centre"] - g) @ result["frame"].axes[0])
        return result, np.array([(result["volume"] - volume) / volume, lever / extent])

    try:
        return _free_trim_newton(evaluate, model, faces, volume, heel, reference, longitudinal, start, order,
                                 rtol, tolerance, max_iterations, extent)
    except GeometryError:
        # the bracketed fallback (review H1-F round 1: Newton stalled where the longitudinal stiffness
        # vanishes, and finite-difference probes stepped into a dry hull)
        return _free_trim_bracketed(model, faces, volume, g, heel, reference, longitudinal, order, rtol,
                                    tolerance, extent)


def _free_trim_bracketed(model, faces, volume, g, heel, reference, longitudinal, order, rtol, tolerance, extent,
                         limit: float = 1.2, samples: int = 49):
    """For each trim the sinkage displacing the volume (``float_at``, bracketed), and the trim where the
    relative longitudinal lever ``(B - G) . xi / extent`` vanishes: a scanned trim already within the
    tolerance is returned; otherwise the sign change nearest to zero on ``[-limit, limit]`` rad is
    bisected until the lever is within the tolerance. Every return has both residuals checked; an
    exhausted interval refuses (review H1-F round 2: a zero at a bracket end was lost and the width
    stop returned a non-equilibrium)."""
    total = _shell(model, faces, order)[1]

    def lever(trim):
        try:
            result = float_at(model, faces, volume, heel_normal(heel, trim), reference, longitudinal,
                              order=order, rtol=rtol)
        except GeometryError:
            return None, None
        if result["buoyancy_centre"] is None:
            return None, None
        return float((result["buoyancy_centre"] - g) @ result["frame"].axes[0]) / extent, result

    def accepted(value, result, trim):
        volume_ok = abs(result["volume"] - volume) <= max(tolerance * volume, 64 * np.finfo(float).eps * total)
        if value is not None and abs(value) <= tolerance and volume_ok:
            result.update(trim=float(trim))
            return result
        return None
    grid = sorted(np.linspace(-limit, limit, samples), key=abs)
    # scanned from zero outwards; stops at the first sign change between neighbouring scanned trims
    # (the bracket nearest zero; CI tiers C1: the full scan was most of the fallback's cost)
    values = {}
    bracket = None
    for trim in grid:
        value, result = lever(trim)
        values[trim] = value
        if value is not None:
            done = accepted(value, result, trim)
            if done is not None:
                return done
        ordered = sorted(t for t in values if values[t] is not None)
        candidates = [(a, b) for a, b in zip(ordered, ordered[1:]) if values[a] * values[b] < 0
                      and not any(a < other < b for other in values)]
        if candidates:
            bracket = min(candidates, key=lambda pair: min(abs(pair[0]), abs(pair[1])))
            break
    if bracket is None:
        raise GeometryError("float_free_trim: no trim equilibrium within +-%.2f rad" % limit)
    lo, hi = bracket
    f_lo, f_hi = values[lo], values[hi]
    # false position with the Illinois modification, kept inside the sign bracket (bisection when a
    # step would not shrink it), until the residuals hold or the bracket is exhausted
    side = 0
    while hi - lo > 4 * np.finfo(float).eps * max(1.0, abs(lo), abs(hi)):
        middle = lo - f_lo * (hi - lo) / (f_hi - f_lo)
        if not lo < middle < hi:
            middle = .5 * (lo + hi)
        f_mid, result = lever(middle)
        if f_mid is None:
            raise GeometryError("float_free_trim: a bracketed trim left the hull")
        done = accepted(f_mid, result, middle)
        if done is not None:
            return done
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = middle, f_mid
            if side == -1:
                f_hi *= .5
            side = -1
        else:
            hi, f_hi = middle, f_mid
            if side == 1:
                f_lo *= .5
            side = 1
    raise GeometryError("float_free_trim: the trim equilibrium is not resolved to the tolerance "
                        "(the lever changes too fast with the trim)")


def _free_trim_newton(evaluate, model, faces, volume, heel, reference, longitudinal, start, order, rtol,
                      tolerance, max_iterations, extent):
    if start is None:
        z = float_at(model, faces, volume, heel_normal(heel, 0.0), reference, longitudinal, order=order,
                     rtol=rtol)["zeta"]
        trim = 0.0
    else:
        z, trim = (float(value) for value in start)
    result, residual = evaluate(z, trim)
    for _ in range(max_iterations):
        if float(np.max(np.abs(residual))) <= tolerance:
            result.update(zeta=z, trim=trim)
            return result
        hz, ht = 1e-6 * extent, 1e-6
        columns = []
        for dz, dt, h in ((hz, 0.0, hz), (0.0, ht, ht)):
            plus = evaluate(z + dz, trim + dt)[1]
            minus = evaluate(z - dz, trim - dt)[1]
            columns.append((plus - minus) / (2 * h))
        jacobian = np.stack(columns, axis=1)
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            raise GeometryError("float_free_trim: singular Jacobian (no trim equilibrium)") from None
        scale = 1.0
        for _halving in range(30):
            candidate = (z + scale * step[0], trim + scale * step[1])
            try:
                trial, trial_residual = evaluate(*candidate)
            except GeometryError:
                scale *= .5
                continue
            if float(np.linalg.norm(trial_residual)) < float(np.linalg.norm(residual)):
                break
            scale *= .5
        else:
            raise GeometryError("float_free_trim: no descent step")
        (z, trim), result, residual = candidate, trial, trial_residual
    raise GeometryError("float_free_trim: did not converge")


def gz_curve(model, faces, volume: float, gravity_centre, heels, trim: float = 0.0,
             reference=(0., 0., 0.), free_trim: bool = False, **options) -> list:
    """Righting levers at the given heels (radians) for a displacement ``volume`` and a centre of
    gravity (body axes), with free sinkage and fixed ``trim`` (``float_at``), or with free trim as well
    (``free_trim``: ``float_free_trim``, B and G on one normal longitudinally). ``GZ = (B - G) . eta``
    with ``eta`` the waterplane's transverse axis (positive: righting). Rows: heel, gz, zeta, trim, result."""
    g = np.asarray(gravity_centre, dtype=float)
    rows, start = [], None
    for heel in heels:
        if free_trim:
            result = float_free_trim(model, faces, volume, g, heel, reference, start=start, **options)
            start = (result["zeta"], result["trim"])
            zeta, row_trim = result["zeta"], result["trim"]
        else:
            result = float_at(model, faces, volume, heel_normal(heel, trim), reference,
                              start=start, **options)
            start = zeta = result["zeta"]
            row_trim = trim
        eta = result["frame"].axes[1]
        rows.append({"heel": float(heel), "gz": float((result["buoyancy_centre"] - g) @ eta),
                     "zeta": zeta, "trim": row_trim, "result": result})
    return rows
