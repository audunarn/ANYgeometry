"""Exact per-face charts for explicitly cut topology-Coons translations.

The serialized recipe is a geometric definition, never an admission receipt.
Every view rederives its tensor net and every boundary correspondence. World
edges retain their original definitions, so one shared straight connector can
have different declared charts on its incident faces.
"""
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction as Q
import math

from .curves import Straight
from .errors import GeometryError
from .nurbs import NurbsCurve, NurbsSurface, CurveOnSurface, _IDENTITY, nurbs_record, nurbs_from_record
from .support_curves import PlaneNurbsCurve
from .surfaces import CoonsSurface

RECIPE = '_nurbs_sweep_chart_v1'


def _stored(value):
    result = float(value)
    if Q(result) != value:
        raise GeometryError('exact sweep chart coefficient is not representable in stored doubles')
    return result


def _tensor(profile, vector):
    if not isinstance(profile, NurbsCurve) or profile.dimension != 3 or not any(vector):
        raise GeometryError('sweep chart requires a spatial rational profile and nonzero translation')
    controls = tuple((point, tuple(_stored(Q(x)+d) for x, d in zip(point, vector)))
                     for point in profile.controls)
    weights = None if profile.weights is None else tuple((w, w) for w in profile.weights)
    return NurbsSurface(profile.degree, 1, profile.knots, (0., 0., 1., 1.), controls,
                        weights, profile.domain, (0., 1.))


def has_sweep_chart(face):
    return (isinstance(face.surface, CoonsSurface) and not face.surface.has_boundaries
            and isinstance(face.parameterization, NurbsSurface) and RECIPE in face.metadata)


def _decode(face):
    if not has_sweep_chart(face):
        raise GeometryError('face has no declared exact sweep chart')
    record = face.metadata[RECIPE]
    if not isinstance(record, Mapping) or set(record) not in ({'profile', 'translation'}, {'profile', 'translation', 'connectors'}):
        raise GeometryError('invalid exact sweep chart definition')
    profile_record = record['profile']
    profile = nurbs_from_record(profile_record.to_dict() if hasattr(profile_record, 'to_dict') else dict(profile_record))
    try:
        pairs = record['translation']
        if len(pairs) != 3 or any(len(pair) != 2 or any(type(x) is not int for x in pair) for pair in pairs):
            raise ValueError
        vector = tuple(Q(*pair) for pair in pairs)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise GeometryError('invalid exact sweep translation') from error
    surface = _tensor(profile, vector)
    if surface != face.parameterization:
        raise GeometryError('declared exact sweep tensor differs from its parameterization')
    return profile, vector, surface


def authored_query_uv(model, face_id, support, rows):
    """Exact dihedral transport into the ORIGINAL topology-Coons chart.

    The sweep recognizer may rotate its bottom side or choose the opposite
    profile to make the translation positive. Neither choice changes the
    authored face chart. Side IDs and exact declared vectors determine this
    transport; world inverse/proximity never selects a branch.
    """
    from ._translational_pairs import _sweep
    from .curves import Straight
    face = model.faces[face_id]
    if isinstance(face.surface, NurbsSurface):
        if face.surface != support:
            raise GeometryError('query tensor does not match the authored NURBS chart')
        return tuple(tuple(map(float, row)) for row in rows)
    if has_sweep_chart(face):
        if chart_model(model, face_id).support != support:
            raise GeometryError('query tensor does not match the declared sweep chart')
        return tuple(tuple(map(float, row)) for row in rows)
    recognized = _sweep(model, face_id)
    if recognized is None or _tensor(recognized[0], recognized[1]) != support:
        raise GeometryError('authored Coons query chart transport is not certified')
    sides = tuple(side[0] for side in face.sides())
    for k in range(4):
        bottom, right, top, left = (sides[(k+j)%4] for j in range(4))
        if (isinstance(model.edges[bottom.edge].curve, NurbsCurve)
                and isinstance(model.edges[top.edge].curve, NurbsCurve)
                and isinstance(model.edges[right.edge].curve, Straight)
                and isinstance(model.edges[left.edge].curve, Straight)):
            break
    else:
        raise GeometryError('authored Coons side correspondence changed')
    start = model.vertex_position(model.oriented_start_vertex(bottom))
    tip = model.vertex_position(model.oriented_end_vertex(top))
    direction = tuple(Q(float(b))-Q(float(a)) for a, b in zip(start, tip))
    if direction == recognized[1]:
        reverse = False
    elif tuple(-d for d in direction) == recognized[1]:
        reverse = True
    else:
        raise GeometryError('authored Coons translation correspondence changed')
    result = []
    for u, v in rows:
        u, v = Q(float(u)), Q(float(v))
        if reverse:
            v = 1-v
        result.append(tuple(map(float, ((u, v), (1-v, u), (1-u, 1-v), (v, 1-u))[k])))
    return tuple(result)


def _qualified_sweep_chart_face(model, face_id):
    """Derive the existing exact recipe without modifying the source model."""
    face = model.faces[face_id]
    if has_sweep_chart(face):
        chart_model(model, face_id)
        return face
    from ._translational_pairs import _sweep
    recognized = _sweep(model, face_id)
    if recognized is None:
        raise GeometryError('face is not an exact topology-Coons rational translation')
    profile, vector, connectors = recognized
    surface = _tensor(profile, vector)
    metadata = dict(face.metadata)
    metadata[RECIPE] = {'profile': nurbs_record(profile),
                        'translation': tuple((v.numerator, v.denominator) for v in vector)}
    # The topology-Coons formula cancels its connector/corner bilinear terms:
    # its corner constants are the evaluated profile endpoints (the exact
    # stored doubles checked by _sweep), not the algebraic profile endpoints.
    # Thus physical support is exactly P(u)+v*d, while an original Straight
    # connector can have a small CONSTANT source-to-worldline offset.
    records = []
    for u, edge_id in enumerate(connectors):
        edge = model.edges[edge_id]
        points = tuple(tuple(float(x) for x in model.vertex_position(v)) for v in (edge.start, edge.end))
        delta = tuple(Q(b)-Q(a) for a, b in zip(*points))
        if delta == vector:
            v0, v1, base_id, tip_id, base, tip = 0., 1., edge.start, edge.end, *points
        elif delta == tuple(-d for d in vector):
            v0, v1, base_id, tip_id, base, tip = 1., 0., edge.end, edge.start, points[1], points[0]
        else:
            raise GeometryError('declared sweep connector is not the structural translation')
        if tuple(float(x) for x in profile.evaluate(u)) != base:
            raise GeometryError('declared sweep connector has a different evaluated profile joint')
        records.append((edge_id, u, base_id, tip_id, base, tip, ((edge_id, v0, v1),)))
    metadata[RECIPE]['connectors'] = tuple(records)
    made = replace(face, parameterization=surface, metadata=metadata)
    # Certify all original world boundaries before installing any definition.
    view = SweepChartModel(model, made)
    for loop in (face.loop, *face.holes):
        for use in loop:
            view.chart_edge(model.edges[use.edge])
    return made


def install_sweep_chart(model, face_id):
    """Declare a proven tensor chart only inside an authorized cutting stage."""
    if has_sweep_chart(model.faces[face_id]):
        chart_model(model, face_id)
        return
    model._put_entity('face', _qualified_sweep_chart_face(model, face_id))


class _Edges(Mapping):
    def __init__(self, view):
        self.view = view
    def __getitem__(self, key):
        return self.view.chart_edge(self.view.raw.edges[key])
    def __iter__(self):
        return iter(self.view.raw.edges)
    def __len__(self):
        return len(self.view.raw.edges)


class _Faces(Mapping):
    def __init__(self, view):
        self.view = view
    def __getitem__(self, key):
        return self.view.chart_face(self.view.raw.faces[key])
    def __iter__(self):
        return iter(self.view.raw.faces)
    def __len__(self):
        return len(self.view.raw.faces)


class SweepChartModel:
    """Internal chart view; world topology/mutations stay on the actual model."""
    def __init__(self, model, face):
        self.raw = model
        self.face_id = face.id
        self.profile, self.vector, self.support = _decode(face)
        self.recipe = face.metadata[RECIPE]
        self.marker = face.surface
        self.edges, self.faces = _Edges(self), _Faces(self)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def chart_face(self, face):
        if has_sweep_chart(face) and face.parameterization == self.support:
            profile, vector, _support = _decode(face)
            if profile != self.profile or vector != self.vector:
                return face
            if face.parameterization != self.support:
                raise GeometryError('sweep child changed its exact parameterization')
            return replace(face, surface=self.support)
        return face

    def chart_edge(self, edge):
        curve = edge.curve
        if isinstance(curve, PlaneNurbsCurve):
            if curve.surface != self.support:
                raise GeometryError('sweep trim section belongs to another support')
            return edge
        if isinstance(curve, CurveOnSurface):
            if curve.support != self.support or curve.transform != _IDENTITY:
                raise GeometryError('sweep trim curve belongs to another support')
            return edge
        profile, vector = self.profile, self.vector
        if isinstance(curve, NurbsCurve):
            if (curve.degree != profile.degree or curve.knots != profile.knots
                    or curve.weights != profile.weights or len(curve.controls) != len(profile.controls)):
                raise GeometryError('sweep boundary is not a restriction of the declared profile')
            offsets = {tuple(Q(y)-Q(x) for x, y in zip(a, b))
                       for a, b in zip(profile.controls, curve.controls)}
            if offsets == {(Q(0),)*3}:
                v = 0.
            elif offsets == {vector}:
                v = 1.
            else:
                raise GeometryError('sweep boundary has no exact profile translation correspondence')
            a, b = map(Q, profile.domain)
            lo, hi = sorted((a, b))
            if not all(lo <= Q(k) <= hi for k in curve.domain):
                raise GeometryError('sweep boundary exceeds the original profile chart')
            pcurve = NurbsCurve(1, (float(lo), float(lo), float(hi), float(hi)),
                ((float((lo-a)/(b-a)), v), (float((hi-a)/(b-a)), v)), domain=curve.domain)
        elif isinstance(curve, Straight):
            if 'connectors' in self.recipe:
                return self._connector_edge(edge)
            endpoints = [tuple(Q(float(x)) for x in self.raw.vertex_position(v)) for v in (edge.start, edge.end)]
            pivot = next(i for i, value in enumerate(vector) if value)
            candidates = []
            for u in (0, 1):
                base = tuple(profile.exact_point(u))
                values = [(point[pivot]-base[pivot])/vector[pivot] for point in endpoints]
                if all(0 <= t <= 1 and all(x == a+t*d for x, a, d in zip(point, base, vector))
                       for point, t in zip(endpoints, values)):
                    candidates.append((u, base, values))
            if len(candidates) != 1:
                raise GeometryError('straight sweep boundary lacks unique exact connector correspondence')
            u, base, values = candidates[0]
            a, b = base[pivot], base[pivot]+vector[pivot]
            lo, hi = sorted((a, b))
            pcurve = NurbsCurve(1, (_stored(lo), _stored(lo), _stored(hi), _stored(hi)),
                ((u, float((lo-a)/(b-a))), (u, float((hi-a)/(b-a)))),
                domain=tuple(_stored(point[pivot]) for point in endpoints))
        else:
            raise GeometryError('sweep boundary family has no exact declared chart correspondence')
        return replace(edge, curve=CurveOnSurface(self.support, pcurve))

    def _connector_edge(self, edge):
        """Declared lineage, with bounded worldline/source discrepancy."""
        candidates = []
        try:
            records = self.recipe['connectors']
            if len(records) != 2:
                raise ValueError
            for source_id, u, base_id, tip_id, base, tip, uses in records:
                if type(source_id) is not int or type(u) is not int or u not in (0, 1):
                    raise ValueError
                if tuple(float(x) for x in self.raw.vertex_position(base_id)) != tuple(base) or tuple(
                        float(x) for x in self.raw.vertex_position(tip_id)) != tuple(tip):
                    raise GeometryError('sweep connector source vertices changed')
                if tuple(Q(b)-Q(a) for a, b in zip(base, tip)) != self.vector:
                    raise GeometryError('sweep connector declared translation changed')
                if tuple(float(x) for x in self.profile.evaluate(u)) != tuple(base):
                    raise GeometryError('sweep connector no longer has the original evaluated profile joint')
                seen = set()
                ranges = []
                for child, a, b in uses:
                    if type(child) is not int or child in seen or not (0 <= a <= 1 and 0 <= b <= 1 and a != b):
                        raise ValueError
                    seen.add(child)
                    if child not in self.raw.edges or not isinstance(self.raw.edges[child].curve, Straight):
                        raise GeometryError('sweep connector descendant no longer has its original straight definition')
                    ranges.append((min(Q(a), Q(b)), max(Q(a), Q(b))))
                    if child == edge.id:
                        candidates.append((u, tuple(map(Q, base)), Q(a), Q(b)))
                reach = Q(0)
                for a, b in sorted(ranges):
                    if a != reach:
                        raise GeometryError('sweep connector descendant intervals have a gap or overlap')
                    reach = b
                if reach != 1:
                    raise GeometryError('sweep connector descendant intervals omit source material')
                if edge.id == source_id and edge.id not in seen:
                    if (edge.start, edge.end) == (base_id, tip_id):
                        candidates.append((u, tuple(map(Q, base)), Q(0), Q(1)))
                    elif (edge.start, edge.end) == (tip_id, base_id):
                        candidates.append((u, tuple(map(Q, base)), Q(1), Q(0)))
                    else:
                        raise GeometryError('historical connector does not match its declared source vertices')
        except GeometryError:
            raise
        except (TypeError, ValueError, KeyError) as cause:
            raise GeometryError('malformed sweep connector interval lineage') from cause
        if len(candidates) != 1:
            raise GeometryError('straight sweep boundary lacks declared connector interval lineage')
        u, base, a, b = candidates[0]
        exact = tuple(self.profile.exact_point(u))
        extent = math.sqrt(sum(float(x*x) for x in self.vector))
        tolerance = Q(self.raw.tolerance.effective_length(extent))
        # Both actual and source paths are affine in the SAME declared v.
        # Endpoint L1 bounds therefore bound the entire correspondence span;
        # no nearest-point test or endpoint-based identity inference occurs.
        errors = []
        for vertex, t in ((edge.start, a), (edge.end, b)):
            point = tuple(Q(float(x)) for x in self.raw.vertex_position(vertex))
            declared = tuple(x+t*d for x, d in zip(base, self.vector))
            source = tuple(x+t*d for x, d in zip(exact, self.vector))
            errors.append(sum(abs(x-y) for x, y in zip(point, source)))
            if sum(abs(x-y) for x, y in zip(point, declared)) > tolerance:
                raise GeometryError('sweep descendant differs from its declared source interval')
        if max(errors) > tolerance:
            raise GeometryError('sweep connector full-span source correspondence exceeds joint precision')
        pcurve = NurbsCurve(1, (0., 0., 1., 1.), ((u, 0.), (u, 1.)), domain=(float(a), float(b)))
        return replace(edge, curve=CurveOnSurface(self.support, pcurve))

    def source_straight_event_key(self, edge, plane, event, context):
        from .nurbs_plane_arrangement import _straight_event_key
        if not isinstance(edge.curve, Straight) or 'connectors' not in self.recipe:
            return _straight_event_key(self.raw, edge, plane, event)
        declared = self._connector_edge(edge).curve.pcurve
        u = int(declared.controls[0][0])
        point = tuple(self.profile.exact_point(u))
        joints = context.setdefault('exact_sweep_joints', {})
        if edge.id in joints and joints[edge.id] != point:
            raise GeometryError('adjacent sweep profiles have incompatible exact source joints')
        joints[edge.id] = point
        a, b = [tuple(Q(float(x)) for x in row) for row in (plane.u_vector, plane.v_vector)]
        normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
        offset = sum(n*Q(float(x)) for n, x in zip(normal, plane.origin))
        denominator = sum(n*d for n, d in zip(normal, self.vector))
        if not denominator:
            raise GeometryError('sweep source connector has no isolated plane root')
        station = (offset-sum(n*x for n, x in zip(normal, point)))/denominator
        lo, hi = map(Q, declared.domain)
        local = (station-lo)/(hi-lo)
        if not event.segment_interval[0] <= local <= event.segment_interval[1]:
            raise GeometryError('declared source connector root is outside its event isolator')
        return ('sweep_source_plane_root', edge.id, point, self.vector, tuple(normal), offset, station)

    def split_edge(self, edge_id, t=.5, **kwargs):
        """Keep explicit connector descendant intervals across shared edits."""
        result = self.raw.split_edge(edge_id, t, **kwargs)
        _vertex, children = result
        for face in tuple(self.raw.faces.values()):
            if not has_sweep_chart(face) or 'connectors' not in face.metadata[RECIPE]:
                continue
            recipe = dict(face.metadata[RECIPE])
            changed, records = False, []
            for source_id, u, base_id, tip_id, base, tip, uses in recipe['connectors']:
                new_uses = []
                for child, a, b in uses:
                    if child == edge_id:
                        middle = float(Q(a)+Q(float(t))*(Q(b)-Q(a)))
                        if not min(a, b) < middle < max(a, b):
                            raise GeometryError('sweep connector descendant station is not representable')
                        new_uses.extend(((children[0], a, middle), (children[1], middle, b)))
                        changed = True
                    else:
                        new_uses.append((child, a, b))
                records.append((source_id, u, base_id, tip_id, base, tip, tuple(new_uses)))
            if changed:
                recipe['connectors'] = tuple(records)
                metadata = dict(face.metadata)
                metadata[RECIPE] = recipe
                self.raw._put_entity('face', replace(face, metadata=metadata))
                if face.id == self.face_id:
                    self.recipe = self.raw.faces[face.id].metadata[RECIPE]
        return result

    def add_face_from_loop(self, loop, corners=None, *, surface=None):
        if surface != self.support:
            raise GeometryError('sweep chart view cannot publish another support')
        return self.raw.add_face_from_loop(loop, corners, surface=self.marker)

    def _put_entity(self, kind, entity):
        if kind == 'face' and entity.surface == self.support and entity.metadata.get(RECIPE) == self.recipe:
            entity = replace(entity, surface=self.marker, parameterization=self.support)
        self.raw._put_entity(kind, entity)


def chart_model(model, face_id):
    view = SweepChartModel(model, model.faces[face_id])
    face = model.faces[face_id]
    for loop in (face.loop, *face.holes):
        for use in loop:
            view.chart_edge(model.edges[use.edge])
    return view


class _SourceSweepChartView(SweepChartModel):
    """Read-only material-domain adapter; no metadata is installed on raw."""
    def __init__(self,model,face):
        self._qualified_source_face=face
        super().__init__(model,face)

    def chart_face(self,face):
        if face.id==self.face_id:
            return replace(self._qualified_source_face,surface=self.support)
        return super().chart_face(face)

    def _put_entity(self,*args,**kwargs):
        raise GeometryError('source sweep read view cannot mutate topology')

    def split_edge(self,*args,**kwargs):
        raise GeometryError('source sweep read view cannot mutate topology')


def source_sweep_chart_model(model,face_id):
    """Optional exact raw-Coons material view, retaining original IDs/state.

    Ordinary unsupported surfaces return None and keep their existing route.
    A recognized recipe must pass the SAME full connector correspondence as an
    installed chart. Callers qualify/bind against the original source model.
    """
    from .surfaces import CoonsSurface
    from ._translational_pairs import _sweep
    face=model.faces[face_id]
    if (type(face.surface) is not CoonsSurface or face.parameterization is not None
            or face.holes or len(face.corners)!=4 or has_sweep_chart(face)):
        return None
    if _sweep(model,face_id) is None:return None
    made=_qualified_sweep_chart_face(model,face_id)
    return _SourceSweepChartView(model,made)
