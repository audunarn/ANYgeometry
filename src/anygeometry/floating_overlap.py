"""Conservative planar polygon overlap used to map panel pressure to structural cells (no nearest-panel search).

Polygons are lists of 3D vertices of PLANAR CONVEX faces, ordered so the right-hand normal points outward (into the
fluid).  Everything is exact polygon clipping (Sutherland-Hodgman) in the source plane; no tolerance-based snapping of
areas.  Overlap between non-near-coplanar or oppositely oriented polygons is zero by definition, so a source panel can
only feed cells of the same physical plate.  The caller checks coverage with ``coverage_report``.
"""
from __future__ import annotations

import numpy as np

from .errors import GeometryError


def area_vector(poly) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise GeometryError("polygon needs at least three 3D vertices")
    p = p - p.mean(axis=0)               # the area vector of a closed polygon is origin independent; avoid far-origin cancellation
    q = np.roll(p, -1, axis=0)
    return 0.5 * np.cross(p, q).sum(axis=0)


def polygon_area(poly) -> float:
    return float(np.linalg.norm(area_vector(poly)))


def polygon_plane(poly):
    """(unit outward normal, centroid, area); refuses a non-planar polygon (relative 1e-9) or zero area."""
    p = np.asarray(poly, dtype=np.float64)
    av = area_vector(p)
    a = float(np.linalg.norm(av))
    if a <= 0.0:
        raise GeometryError("degenerate polygon")
    n = av / a
    scale = max(1.0, float(np.abs(p).max()))
    if np.abs((p - p[0]) @ n).max() > 1e-9 * scale:
        raise GeometryError("floating overlap requires planar polygons")
    # area-weighted centroid by fan triangles
    c = np.zeros(3); tot = 0.0
    for k in range(1, len(p) - 1):
        t = p[[0, k, k + 1]]; ta = 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
        c += ta * t.mean(axis=0); tot += ta
    return n, c / tot, a


def clip_halfspace(poly, normal, offset):
    """Keep ``normal . x <= offset``; returns an (m,3) array or None when empty/degenerate."""
    p = np.asarray(poly, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    d = p @ n - float(offset)
    out = []
    for i in range(len(p)):
        j = (i + 1) % len(p)
        di, dj = d[i], d[j]
        if di <= 0.0:
            out.append(p[i])
        if (di < 0.0 < dj) or (dj < 0.0 < di):
            out.append(p[i] + (p[j] - p[i]) * (di / (di - dj)))
    if len(out) < 3:
        return None
    r = np.array(out)
    return r if np.linalg.norm(area_vector(r)) > 0.0 else None


def wet_polygon(poly, z: float = 0.0):
    """Part of a planar polygon at or below the waterplane z (mean-plane partial regions only)."""
    return clip_halfspace(poly, (0.0, 0.0, 1.0), z)


def fan_triangles(poly):
    p = np.asarray(poly, dtype=np.float64)
    return [p[[0, k, k + 1]] for k in range(1, len(p) - 1) if np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) > 0]


def _basis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def panelize(poly, size: float):
    """Split a planar convex polygon into convex pieces on a regular grid of ~``size`` (orientation preserved)."""
    n, c, a = polygon_plane(poly)
    p = np.asarray(poly, dtype=np.float64)
    if not size > 0.0 or not np.isfinite(size):
        raise GeometryError("panel size must be positive")
    origin = p.mean(axis=0)               # grid and clipping in a body-local frame: results are covariant with the hull's position
    p = p - origin
    u, v = _basis(n)
    su, sv = p @ u, p @ v
    nu = max(1, int(np.ceil((su.max() - su.min()) / size)))
    nv = max(1, int(np.ceil((sv.max() - sv.min()) / size)))
    pieces = []
    for i in range(nu):
        lo_u = su.min() + (su.max() - su.min()) * i / nu; hi_u = su.min() + (su.max() - su.min()) * (i + 1) / nu
        for j in range(nv):
            lo_v = sv.min() + (sv.max() - sv.min()) * j / nv; hi_v = sv.min() + (sv.max() - sv.min()) * (j + 1) / nv
            q = clip_halfspace(p, u, hi_u)
            for nrm, off in ((-u, -lo_u), (v, hi_v), (-v, -lo_v)):
                q = None if q is None else clip_halfspace(q, nrm, off)
            if q is not None and polygon_area(q) > 1e-12 * a:
                pieces.append(q)
    total = sum(polygon_area(q) for q in pieces)
    if abs(total - a) > 1e-9 * a:
        raise GeometryError("panelization lost area (non-convex polygon?)")
    return [q + origin for q in pieces]


def overlap_polygon(source, target, *, distance_tol: float = 1e-6, normal_cos_tol: float = 1.0 - 1e-6):
    """Physical overlap polygon (in the source plane) of two coplanar same-orientation convex polygons, or None."""
    ns, cs, _ = polygon_plane(source)
    nt, ct, _ = polygon_plane(target)
    if float(ns @ nt) < normal_cos_tol:
        return None
    t = np.asarray(target, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    if np.abs((t - cs) @ ns).max() > distance_tol * max(1.0, float(np.abs(s).max())):
        return None
    q = t - np.outer((t - cs) @ ns, ns)
    for i in range(len(s)):
        e = s[(i + 1) % len(s)] - s[i]
        m = np.cross(e, ns); ml = np.linalg.norm(m)
        if ml == 0.0:
            continue
        m = m / ml
        q = clip_halfspace(q, m, float(m @ s[i]))     # outward edge normal for a CCW (outward-normal) polygon
        if q is None:
            return None
    return q


def overlap_area(source, target, *, distance_tol: float = 1e-6, normal_cos_tol: float = 1.0 - 1e-6) -> float:
    """Area of the overlap polygon; 0 if the polygons are not the same plate."""
    q = overlap_polygon(source, target, distance_tol=distance_tol, normal_cos_tol=normal_cos_tol)
    return 0.0 if q is None else polygon_area(q)


def overlap_regions(sources, targets, *, distance_tol: float = 1e-6, normal_cos_tol: float = 1.0 - 1e-6):
    """All non-empty overlap POLYGONS: list of (source index, target index, polygon (m,3)) with bounding-box prefilter."""
    S = [np.asarray(s, dtype=np.float64) for s in sources]
    T = [np.asarray(t, dtype=np.float64) for t in targets]
    smin = np.array([s.min(axis=0) for s in S]); smax = np.array([s.max(axis=0) for s in S])
    pad = 1e-6 * max(1.0, float(np.abs(smax).max()) if len(S) else 1.0)
    out = []
    for j, t in enumerate(T):
        lo, hi = t.min(axis=0) - pad, t.max(axis=0) + pad
        for i in np.nonzero(np.all(smin <= hi, axis=1) & np.all(smax >= lo, axis=1))[0]:
            q = overlap_polygon(S[i], t, distance_tol=distance_tol, normal_cos_tol=normal_cos_tol)
            if q is not None and polygon_area(q) > 0.0:
                out.append((int(i), j, q))
    return out


def overlap_matrix(sources, targets, *, distance_tol: float = 1e-6, normal_cos_tol: float = 1.0 - 1e-6):
    """Sparse overlap triplets (source index, target index, area) with bounding-box prefilter."""
    S = [np.asarray(s, dtype=np.float64) for s in sources]
    T = [np.asarray(t, dtype=np.float64) for t in targets]
    smin = np.array([s.min(axis=0) for s in S]); smax = np.array([s.max(axis=0) for s in S])
    rows, cols, vals = [], [], []
    pad = 1e-6 * max(1.0, float(np.abs(smax).max()) if len(S) else 1.0)
    for j, t in enumerate(T):
        lo, hi = t.min(axis=0) - pad, t.max(axis=0) + pad
        cand = np.nonzero(np.all(smin <= hi, axis=1) & np.all(smax >= lo, axis=1))[0]
        for i in cand:
            a = overlap_area(S[i], t, distance_tol=distance_tol, normal_cos_tol=normal_cos_tol)
            if a > 0.0:
                rows.append(int(i)); cols.append(j); vals.append(a)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(vals, dtype=np.float64)


def coverage_report(source_areas, target_wet_areas, rows, cols, vals):
    """Relative uncovered source area / unmatched target wet area (the caller refuses above its tolerance)."""
    sa = np.asarray(source_areas, dtype=np.float64); ta = np.asarray(target_wet_areas, dtype=np.float64)
    fed_s = np.bincount(rows, weights=vals, minlength=len(sa)); fed_t = np.bincount(cols, weights=vals, minlength=len(ta))
    return {"source_uncovered_rel": float(np.abs(sa - fed_s).max() / sa.max()) if len(sa) else 0.0,
            "source_uncovered_total_rel": float(abs(sa.sum() - fed_s.sum()) / sa.sum()) if len(sa) else 0.0,
            "target_unmatched_rel": float(np.abs(ta - fed_t).max() / ta.max()) if len(ta) else 0.0}
