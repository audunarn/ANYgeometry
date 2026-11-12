"""Bounded panel assignment on the *target* surface, without force corrections.

Convex planar source/target polygons have consistently outward winding and SI
coordinates. Coplanar partitions use exact clipping. Independent facets use
closest-surface assignment: certified constant-panel triangles plus a reported
upper bound on the target area whose panel assignment remains unresolved.
That area bound is geometric; it is not a pressure/force error estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from .floating_overlap import (fan_triangles, _overlap_polygon_admitted, polygon_area, polygon_plane,
                               _polygon_plane_admitted, clip_halfspace)
from .errors import GeometryError


class PanelProjectionError(ValueError):
    """Input, coverage, ambiguity or explicit mapping budget was refused."""


class PanelProjectionCancelled(PanelProjectionError):
    """Preparation cancelled without publishing a partial map."""


@dataclass(frozen=True, kw_only=True)
class MappingLimits:
    gap_m: float
    max_angle: float
    coverage_tolerance: float
    convergence_tolerance: float
    max_depth: int = 8
    max_triangles: int = 100000

    def __post_init__(self):
        for name in ('gap_m', 'max_angle', 'coverage_tolerance', 'convergence_tolerance'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise PanelProjectionError(f'{name} must be finite and nonnegative')
        if self.max_angle >= math.pi / 2:
            raise PanelProjectionError('max_angle must be less than pi/2 radians')
        if self.coverage_tolerance >= 1 or self.convergence_tolerance >= 1:
            raise PanelProjectionError('area tolerances must be less than one')
        for name in ('max_depth', 'max_triangles'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == 'max_triangles' else 0):
                raise PanelProjectionError(f'invalid {name}')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _frozen(a, dtype=float):
    a = np.ascontiguousarray(a, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class PanelProjection:
    triangles: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    diagnostics: object
    source_sha256: str
    target_sha256: str
    content_sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'triangles', _frozen(self.triangles))
        object.__setattr__(self, 'source_indices', _frozen(self.source_indices, np.int64))
        object.__setattr__(self, 'target_indices', _frozen(self.target_indices, np.int64))
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))

    def _payload(self):
        return dict(triangles=self.triangles.tolist(), source_indices=self.source_indices.tolist(),
                    target_indices=self.target_indices.tolist(), diagnostics=dict(self.diagnostics),
                    source_sha256=self.source_sha256, target_sha256=self.target_sha256)

    def validate(self):
        if _digest(self._payload()) != self.content_sha256:
            raise PanelProjectionError('projection content identity mismatch')


def _polygons(values, check):
    out = []
    for value in values:
        check()
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) < 3:
            raise PanelProjectionError('each polygon needs at least three real 3D vertices')
        p = np.array(raw, dtype=float, copy=True)
        if not np.isfinite(p).all():
            raise PanelProjectionError('nonfinite polygon coordinates')
        # One planarity criterion (floating_overlap.planarity_tolerance), judged once here; the
        # overlap calls below receive the polygons as certified, so nothing admitted is refused later.
        try:
            n, _, area = polygon_plane(p, planarity='strict')
        except (ValueError, GeometryError) as error:
            raise PanelProjectionError(str(error)) from error
        extent = np.linalg.norm(np.ptp(p, axis=0))
        eps = 64 * np.finfo(float).eps * max(extent, np.finfo(float).tiny)
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            edge = b-a
            if np.linalg.norm(edge) <= eps or np.min(np.cross(edge, p-a) @ n) < -eps*extent:
                raise PanelProjectionError('polygon must be convex with distinct consecutive vertices')
        out.append((p, n, area))
    if not out:
        raise PanelProjectionError('source and target polygons must be nonempty')
    return out


def validate_panel_geometry(polygons):
    """Admit convex planar facets and return immutable geometry without repair.

    Returns ``(polygons, normals, centers, areas)``. Polygons retain input order
    and winding; normals follow that winding. Centers are area-weighted polygon
    centroids, not vertex averages. No hydrodynamic conventions are inferred.
    """
    admitted = _polygons(polygons, lambda: None)
    planes = [_polygon_plane_admitted(p) for p, _, _ in admitted]
    return (tuple(_frozen(p) for p, _, _ in admitted),
            _frozen([n for n, _, _ in planes]),
            _frozen([c for _, c, _ in planes]),
            _frozen([a for _, _, a in planes]))


class _Index:
    """Small dependency-free bounding-volume tree; no dense source x target table."""
    def __init__(self, polygons, ids=None, check=lambda: None):
        check()
        self.ids = np.arange(len(polygons)) if ids is None else np.asarray(ids)
        self.lo = np.min([polygons[i][0].min(axis=0) for i in self.ids], axis=0)
        self.hi = np.max([polygons[i][0].max(axis=0) for i in self.ids], axis=0)
        self.children = None
        if len(self.ids) > 8:
            axis = int(np.argmax(self.hi-self.lo))
            order = sorted(self.ids, key=lambda i: float(polygons[i][0][:, axis].mean()))
            mid = len(order)//2
            self.children = (_Index(polygons, order[:mid], check), _Index(polygons, order[mid:], check))

    def query(self, lo, hi):
        if np.any(self.lo > hi) or np.any(self.hi < lo):
            return []
        if self.children is None:
            return self.ids.tolist()
        return self.children[0].query(lo, hi)+self.children[1].query(lo, hi)


def _closest(point, polygon, normal, eps):
    """Exact closest point on a convex planar polygon, including its boundary."""
    q = point - np.dot(point-polygon[0], normal)*normal
    edges = np.roll(polygon, -1, axis=0)-polygon
    signed = np.cross(edges, q-polygon) @ normal
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(signed/lengths) >= -eps:
        return float(np.linalg.norm(point-q)), q, bool(np.min(signed/lengths) > eps)
    u = np.clip(np.einsum('ij,ij->i', point-polygon, edges)/np.einsum('ij,ij->i', edges, edges), 0, 1)
    points = polygon+u[:, None]*edges
    d = np.linalg.norm(points-point, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), points[k], False


# A nearer panel that fails the angle limit may be passed over for the nearest compatible panel
# only inside a corner: every panel nearer than the compatible one must be connected to it through
# shared vertices (one local piece of surface), and the compatible panel must be closer than this
# fraction of the smallest panel diameter in that cluster, i.e. well inside the source resolution.
# Anything else, including a T-junction or a point far from the junction, stays a typed refusal.
CORNER_LOCALITY = 0.1


def _touch(pa, pb, tol):
    return any(np.min(np.linalg.norm(pb-v, axis=1)) <= tol for v in pa)


def _diameter(p):
    return max(float(np.linalg.norm(u-v)) for u in p for v in p)


def _near_shared_boundary(point, first, second, radius, tol):
    """Test proximity to a shared vertex or to its overlapping incident edge."""
    for ia, vertex in enumerate(first):
        for ib in np.flatnonzero(np.linalg.norm(second - vertex, axis=1) <= tol):
            if np.linalg.norm(point - vertex) <= radius + tol:
                return True
            for da in (-1, 1):
                edge = first[(ia + da) % len(first)] - vertex
                length = np.linalg.norm(edge)
                direction = edge / length
                for db in (-1, 1):
                    other = second[(ib + db) % len(second)] - second[ib]
                    other_length = np.linalg.norm(other)
                    shared_length = min(length, other_length)
                    if shared_length * np.linalg.norm(direction - other / other_length) > tol:
                        continue
                    along = np.clip((point - vertex) @ direction, 0., shared_length)
                    if np.linalg.norm(point - (vertex + along * direction)) <= radius + tol:
                        return True
    return False


def _corner_row(point, values, source, compatible, tol):
    """The nearest compatible panel when every nearer panel belongs to its local corner cluster."""
    for k, row in enumerate(values):
        if compatible(row):
            break
    else:
        return None
    cluster = [row]+list(values[:k])
    polys = [source[r[0]][0] for r in cluster]
    radius = CORNER_LOCALITY * min(_diameter(p) for p in polys)
    if row[1] > radius + tol:
        return None
    reached, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for m in range(len(cluster)):
            if m not in reached and _near_shared_boundary(point, polys[i], polys[m], radius, tol):
                reached.add(m)
                frontier.append(m)
    if len(reached) != len(cluster):
        return None
    return row


def _split(t):
    a, b, c = t
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return (np.array([a, ab, ca]), np.array([ab, b, bc]), np.array([ca, bc, c]), np.array([ab, bc, ca]))


def _support_area(p, normal):
    """Generated polygon area using its known support normal, without a centroid."""
    if p is None or len(p)<3:return 0.
    if not np.isfinite(p).all():raise PanelProjectionError('nonfinite generated support fragment')
    return abs(float(sum(np.cross(p[k]-p[0],p[k+1]-p[0])@normal for k in range(1,len(p)-1)))/2)


def _support_overlap(source,target,normal):
    """Clip already coplanar generated fragments in their local support frame."""
    if _support_area(source,normal)==0 or _support_area(target,normal)==0:return None
    q=target
    for i in range(len(source)):
        edge=source[(i+1)%len(source)]-source[i]
        outward=np.cross(edge,normal);length=np.linalg.norm(outward)
        if length==0:continue
        outward/=length
        q=clip_halfspace(q,outward,float(outward@source[i]))
        if q is None:return None
    return q if _support_area(q,normal)>0 else None


def _support_precision_area(p):
    # Edge arithmetic in a local frame plus coordinate subtraction roundoff.
    # This is a precision/error bound, deducted from coverage, never gap repair.
    extent=float(np.linalg.norm(np.ptp(p,axis=0)))
    return 128*np.finfo(float).eps*extent*(extent+float(np.abs(p).max()))


def prepare_panel_projection(source_polygons, target_polygons, *, limits, cancel=None,
                             source_regions=None, target_regions=None, two_sided_regions=(),source_coverage_regions=()):
    """Return target triangles and their zero-based source/target panel indices.

    Paired region strings restrict correspondence (use a body/region composite
    string where needed). ``cancel`` is a zero-argument predicate. Unresolved
    assignment area on *each* target must be <= convergence_tolerance times its
    area; max_depth and max_triangles are hard limits. Missing coverage, opposing
    normals, equally close distinct surfaces, and exhausted budgets fail closed.
    """
    if type(limits) is not MappingLimits:
        raise PanelProjectionError('explicit MappingLimits required')
    def check():
        if cancel is not None and cancel():
            raise PanelProjectionCancelled('panel projection cancelled')
    check()
    source, target = _polygons(source_polygons, check), _polygons(target_polygons, check)
    if (source_regions is None) != (target_regions is None):
        raise PanelProjectionError('source_regions and target_regions must be supplied together')
    sr = tuple('' for _ in source) if source_regions is None else tuple(source_regions)
    tr = tuple('' for _ in target) if target_regions is None else tuple(target_regions)
    if len(sr) != len(source) or len(tr) != len(target) or any(type(s) is not str for s in sr+tr):
        raise PanelProjectionError('region inventory must contain one string per polygon')
    two_sided_regions=tuple(two_sided_regions)
    source_coverage_regions=tuple(source_coverage_regions)
    if (len(set(source_coverage_regions))!=len(source_coverage_regions) or
            any(type(r) is not str or r not in sr or r not in tr for r in source_coverage_regions)):
        raise PanelProjectionError('source coverage regions must occur in both inventories')
    if (len(set(two_sided_regions))!=len(two_sided_regions) or
            any(type(r) is not str or r not in sr or r not in tr for r in two_sided_regions)):
        raise PanelProjectionError('two-sided regions must explicitly occur in source and target inventories')
    source_hash = _digest(dict(polygons=[p.tolist() for p, _, _ in source], regions=sr))
    target_hash = _digest(dict(polygons=[p.tolist() for p, _, _ in target], regions=tr))
    if two_sided_regions:
        source_hash=_digest(dict(source=source_hash,two_sided_regions=two_sided_regions))
        target_hash=_digest(dict(target=target_hash,two_sided_regions=two_sided_regions))
    region_rows = {}
    for i, region in enumerate(sr):
        region_rows.setdefault(region, []).append(i)
    indexes = {r: _Index(source, rows, check) for r, rows in region_rows.items()}
    triangles, sources, targets = [], [], []
    source_precision_error=np.zeros(len(source))
    diag = dict(algorithm='bounded_closest_panel_v1', limits=asdict(limits), exact_targets=0,
                projected_targets=0, unresolved_area_bound_m2=0., max_gap_m=0., max_angle_rad=0.,
                max_depth_used=0, target_area_m2=sum(a for _, _, a in target),
                uncovered_target_area_m2=0.)
    def append(t, i, j):
        if _support_area(t,target[j][1])==0.:
            source_precision_error[i]+=_support_precision_area(source[i][0])
            return
        if len(triangles) >= limits.max_triangles:
            raise PanelProjectionError('max_triangles exhausted')
        triangles.append(t); sources.append(i); targets.append(j)
    for j, (p, nt, area) in enumerate(target):
        check()
        two_sided=tr[j] in two_sided_regions
        extent = float(np.linalg.norm(np.ptp(p, axis=0)))
        eps = 128*np.finfo(float).eps*max(extent, float(np.abs(p).max()), 1.)
        index = indexes.get(tr[j])
        if index is None:
            raise PanelProjectionError(f'target {j}: region has no source panels')

        def emit(t, i, upper):
            # The one publication path (exact, certified and unresolved): the selected row is validated against
            # the angle and gap limits here, and the diagnostics describe the published labels only.
            cosine = float(np.clip(source[i][1] @ nt, -1, 1))
            if two_sided:cosine=abs(cosine)
            if upper > limits.gap_m+eps:
                raise PanelProjectionError(f'target {j}: closest surface exceeds gap_m')
            if cosine+128*np.finfo(float).eps < math.cos(limits.max_angle):
                raise PanelProjectionError(f'target {j}: closest surface exceeds max_angle')
            diag['max_gap_m'] = max(diag['max_gap_m'], upper)
            diag['max_angle_rad'] = max(diag['max_angle_rad'], math.acos(cosine))
            append(t, i, j)
        candidates = sorted(index.query(p.min(axis=0)-limits.gap_m-eps, p.max(axis=0)+limits.gap_m+eps))
        if not candidates:
            raise PanelProjectionError(f'target {j}: no source within gap_m')
        exact = []
        for i in candidates:
            check()
            q, ns, _ = source[i]
            if two_sided and float(ns@nt)<0:
                q=q[::-1]; ns=-ns
            if float(ns @ nt) < 1-128*np.finfo(float).eps or np.max(np.abs((q-p[0]) @ nt)) > eps:
                continue
            # Local coordinates avoid translation-sensitive halfspace cancellation.
            clipped = _overlap_polygon_admitted(q-p[0], p-p[0], distance_tol=eps, normal_cos_tol=1-128*np.finfo(float).eps)
            if clipped is not None:
                clipped -= ((clipped @ nt)[:, None])*nt
                if _support_area(clipped,nt)>0.:exact.append((i, clipped))
                else:source_precision_error[i]+=_support_precision_area(source[i][0])
        exact_area = sum(_support_area(q,nt) for _, q in exact)
        if exact_area:
            # Positive-area duplicate coverage is ambiguity, never compensating coverage.
            clipped_index = _Index([(q, nt, _support_area(q,nt)) for _, q in exact], check=check)
            for k, (_, q) in enumerate(exact):
                for m in clipped_index.query(q.min(axis=0)-eps, q.max(axis=0)+eps):
                    if m >= k: continue
                    r = exact[m][1]
                    check()
                    hit = _support_overlap(q,r,nt)
                    if hit is not None and _support_area(hit,nt) > 128*np.finfo(float).eps*area:
                        raise PanelProjectionError(f'target {j}: ambiguous overlapping source panels')
            if abs(exact_area-area) <= limits.coverage_tolerance*area + eps*extent:
                for i, q in exact:
                    for t in fan_triangles(q):
                        t = t+p[0]
                        emit(t, i, max(_closest(x, source[i][0], source[i][1], eps)[0] for x in t))
                diag['exact_targets'] += 1
                diag['uncovered_target_area_m2'] += max(0., area-exact_area)
                continue
            # Do not fill holes in a coplanar source surface by nearest edge loads.
            if all(abs((abs(float(source[i][1] @ nt)) if two_sided else float(source[i][1] @ nt))-1) < 128*np.finfo(float).eps and
                   np.max(np.abs((source[i][0]-p[0]) @ nt)) <= eps for i in candidates):
                raise PanelProjectionError(f'target {j}: incomplete coplanar coverage')
        diag['projected_targets'] += 1

        def distances(point):
            check()
            values = [(i, *_closest(point, source[i][0], source[i][1], eps)) for i in candidates]
            values.sort(key=lambda row: (row[1], row[0]))
            return values

        def assign(point):
            values = distances(point)
            best = values[0]
            # A vertex on a sharp edge ties between source panels at the same closest point; the
            # lower index must not win when its normal is incompatible. Only a best that would
            # fail the angle limit is replaced, by the first tied panel that satisfies it, so
            # inputs that were accepted before keep their assignment.
            def compatible(row):
                c = float(np.clip(source[row[0]][1] @ nt, -1, 1))
                return (abs(c) if two_sided else c)+128*np.finfo(float).eps >= math.cos(limits.max_angle)
            if not compatible(best):
                best = next((r for r in values[1:] if r[1]-values[0][1] <= eps and compatible(r)), best)
            if not compatible(best):
                # Corner: the nearer panels are incompatible neighbours of the compatible one.
                best = _corner_row(point, values, source, compatible, eps) or best
            def checked_angle(row):
                if row[1] > limits.gap_m+eps:
                    raise PanelProjectionError(f'target {j}: closest surface exceeds gap_m')
                cosine = float(np.clip(source[row[0]][1] @ nt, -1, 1))
                if two_sided:cosine=abs(cosine)
                # Comparing cosines avoids acos amplifying roundoff at zero angle.
                if cosine+128*np.finfo(float).eps < math.cos(limits.max_angle):
                    raise PanelProjectionError(f'target {j}: closest surface exceeds max_angle')
                return math.acos(cosine)
            angle = checked_angle(best)
            def same_plane_seam(a, b):
                # Coplanar neighbours where exactly one holds the foot of the sample: their distances tie
                # only to second order in the in-plane offset, and the foot decides (it is on that panel).
                if a[3] == b[3] or float(source[a[0]][1] @ source[b[0]][1]) < 1-128*np.finfo(float).eps:
                    return False
                # The panels must also be neighbours (share a vertex): detached coplanar sheets stay ambiguous.
                return (_touch(source[a[0]][0], source[b[0]][0], eps) and
                        float(np.max(np.abs((source[b[0]][0]-source[a[0]][0][0]) @ source[a[0]][1]))) <= eps)
            if not best[3]:
                best = next((o for o in values if o[3] and abs(o[1]-best[1]) <= eps and same_plane_seam(best, o)), best)
            # Rows more than eps nearer than best were passed over as corner neighbours above.
            if any(other is not best and abs(other[1]-best[1]) <= eps and
                   (np.linalg.norm(other[2]-best[2]) > eps or (best[3] and other[3])) and
                   not same_plane_seam(best, other) for other in values):
                raise PanelProjectionError(f'target {j}: ambiguous equally close source surfaces')
            return best[0], values

        pending = list(fan_triangles(p))
        for depth in range(limits.max_depth+1):
            check()
            diag['max_depth_used'] = max(diag['max_depth_used'], depth)
            unresolved = []
            for t in pending:
                center = t.mean(axis=0)
                chosen, at_center = assign(center)
                vertex_values = [distances(x) for x in t]
                # Distance to a convex polygon is convex: vertex maximum bounds
                # the entire triangle. Other distances are 1-Lipschitz.
                upper = max(next(v[1] for v in rows if v[0] == chosen) for rows in vertex_values)
                radius = max(np.linalg.norm(t-center, axis=1))
                lower = min((v[1]-radius for v in at_center if v[0] != chosen), default=math.inf)
                certified = upper <= limits.gap_m+eps and upper+eps < lower
                if certified:
                    emit(t, chosen, upper)
                else:
                    unresolved.append((t, chosen, upper))
            uncertain_area = sum(polygon_area(t) for t, _, _ in unresolved)
            if uncertain_area <= limits.convergence_tolerance*area and all(u <= limits.gap_m+eps for _, _, u in unresolved):
                for t, i, u in unresolved: emit(t, i, u)
                diag['unresolved_area_bound_m2'] += uncertain_area
                break
            if depth == limits.max_depth:
                raise PanelProjectionError(f'target {j}: max_depth exhausted before convergence ({uncertain_area/area:g} unresolved area fraction)')
            if len(triangles)+4*len(unresolved) > limits.max_triangles:
                raise PanelProjectionError('max_triangles exhausted before convergence')
            pending = [child for t, _, _ in unresolved for child in _split(t)]
    # A dipole pressure jump acts once on one physical midsurface. Test assigned
    # footprints on each ORIGINAL source plane, including opposed target winding.
    # Resultant cancellation cannot hide duplicate structural sheet coverage.
    if two_sided_regions:
        by_source={}
        global_footprints=[];global_sources=[];global_targets=[]
        for k,i in enumerate(sources):
            if sr[i] in two_sided_regions:by_source.setdefault(i,[]).append(k)
        for i,ks in by_source.items():
            check();p,ns,area=source[i];origin=p[0]
            eps=128*np.finfo(float).eps*max(1.,float(np.abs(p).max()),float(np.ptp(p,axis=0).max()))
            footprints=[]
            for k in ks:
                check()
                q=triangles[k]-origin
                q=q-np.outer(q@ns,ns)
                if float(np.cross(q[1]-q[0],q[2]-q[0])@ns)<0:q=q[::-1]
                qa=_support_area(q,ns)
                if qa==0.:
                    source_precision_error[i]+=_support_precision_area(p)
                    continue
                footprints.append((q,ns,qa,k))
                world=q+origin
                if _support_area(world,ns)>0.:
                    global_footprints.append((world,ns,_support_area(world,ns)));global_sources.append(i);global_targets.append(targets[k])
                else:source_precision_error[i]+=_support_precision_area(p)
            footprint_index=_Index([row[:3] for row in footprints],check=check)
            for k,(q,_,qa,original_k) in enumerate(footprints):
                check()
                for m in footprint_index.query(q.min(axis=0)-eps,q.max(axis=0)+eps):
                    if m>=k:continue
                    check()
                    overlap=_support_overlap(q,footprints[m][0],ns)
                    overlap_area=_support_area(overlap,ns)
                    if overlap_area:
                        precision=_support_precision_area(p)
                        if targets[original_k]!=targets[footprints[m][3]] and overlap_area>precision:
                            raise PanelProjectionError('duplicate dipole target midsurface coverage')
                        # Same-target triangles are one partition, not two sheets.
                        # Count any apparent overlap against coverage nonetheless.
                        source_precision_error[i]+=overlap_area+precision
        # Coplanar source seams do not create independent physical sheets.
        # Compare across assigned indices on the same support plane, but never
        # collapse distinct parallel source sheets into one global plane.
        gi=_Index(global_footprints,check=check)
        for k,(q,nq,qa) in enumerate(global_footprints):
            check();i=global_sources[k]
            eps=128*np.finfo(float).eps*max(1.,float(np.abs(q).max()))
            for m in gi.query(q.min(axis=0)-eps,q.max(axis=0)+eps):
                if m>=k or global_sources[m]==i:continue
                check();r,nr,ra=global_footprints[m]
                if sr[global_sources[m]]!=sr[i] or abs(abs(float(nq@nr))-1)>128*np.finfo(float).eps:continue
                if np.max(np.abs((r-q[0])@nq))>eps:continue
                if float(nq@nr)<0:r=r[::-1]
                overlap=_support_overlap(q-q[0],r-q[0],nq)
                overlap_area=_support_area(overlap,nq)
                if overlap_area:
                    other=global_sources[m]
                    precision=max(_support_precision_area(source[i][0]),_support_precision_area(source[other][0]))
                    if global_targets[k]!=global_targets[m] and overlap_area>precision:
                        raise PanelProjectionError('duplicate dipole target coverage across source-panel seam')
                    source_precision_error[i]+=overlap_area+precision
                    source_precision_error[other]+=overlap_area+precision
        diag['two_sided_regions']=two_sided_regions
        diag['duplicate_target_coverage_checked']=True
    if source_coverage_regions:
        # Only actual interior footprints certify physical source support.
        # Nearest-edge extrapolation cannot certify an omitted sheet strip.
        # Subtract possible overlaps/precision loss below to obtain a conservative
        # interior coverage lower bound, rather than double-counting fragments.
        if not set(source_coverage_regions)<=set(two_sided_regions):
            raise PanelProjectionError('source coverage requires two-sided duplicate admission')
        covered=np.zeros(len(source))
        for i,(p,_,_) in enumerate(source):
            if sr[i] in source_coverage_regions:
                source_precision_error[i]+=_support_precision_area(p)
        for t,i,j in zip(triangles,sources,targets):
            if sr[i] not in source_coverage_regions:continue
            check();p,ns,area=source[i]
            # Every projection/clip/area computation has finite precision, even
            # when it happens to produce exact or empty coverage numerically.
            source_precision_error[i]+=max(_support_precision_area(p),_support_precision_area(target[j][0]))
            q=t-p[0];q-=np.outer(q@ns,ns)
            if _support_area(q,ns)==0.:continue
            if float(np.cross(q[1]-q[0],q[2]-q[0])@ns)<0:q=q[::-1]
            eps=128*np.finfo(float).eps*max(1.,float(np.abs(p).max()))
            overlap=_support_overlap(q,p-p[0],ns)
            covered[i]+=_support_area(overlap,ns)
        missing=0.
        for i,(p,ns,area) in enumerate(source):
            if sr[i] not in source_coverage_regions:continue
            check();gap=max(0.,area-covered[i])+source_precision_error[i];missing+=gap
            if gap>limits.coverage_tolerance*area:
                raise PanelProjectionError(f'source {i}: incomplete dipole interior support coverage')
        diag['source_coverage_regions']=source_coverage_regions
        diag['uncovered_source_area_m2']=missing
        diag['source_coverage_precision_area_bound_m2']=float(source_precision_error.sum())
    check()
    diag['triangle_count'] = len(triangles)
    diag['source_panels_used'] = len(set(sources))
    diag['source_panel_count'] = len(source)
    diag['mapped_area_m2'] = sum(polygon_area(t) for t in triangles)
    # Nested limits are stored as a canonical JSON string so diagnostics are deeply immutable.
    diag['limits'] = json.dumps(diag['limits'], sort_keys=True, separators=(',', ':'))
    result = PanelProjection(_frozen(triangles), _frozen(sources, np.int64), _frozen(targets, np.int64),
                             MappingProxyType(diag), source_hash, target_hash, '')
    object.__setattr__(result, 'content_sha256', _digest(result._payload()))
    return result
