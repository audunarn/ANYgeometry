"""Versioned sketch entities and direct editing operations.

Sampling in this module is presentation only. Accepted topology uses the exact
owner curves; trimming validation lives in :mod:`sketch_cut`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Mapping, Sequence
from uuid import uuid4
import numpy as np

from .errors import GeometryError
from .entities import EntityRef
from .nurbs import NurbsCurve
from .reference import bspline as _reference
from .curves import arc_frame


@dataclass(frozen=True)
class SketchEntity:
    id: str
    kind: str
    points: tuple[str, ...]
    radius: float | None = None
    degree: int = 3
    weights: tuple[float, ...] = ()
    closed: bool = False

    def __post_init__(self):
        if not self.id or self.kind not in ('line', 'arc', 'circle', 'spline', 'nurbs', 'fillet'):
            raise GeometryError('invalid sketch entity ID or kind')
        object.__setattr__(self, 'points', tuple(self.points))
        object.__setattr__(self, 'weights', tuple(float(x) for x in self.weights))
        if self.radius is not None and (not np.isfinite(self.radius) or self.radius <= 0):
            raise GeometryError('radius must be finite and positive')

    def to_dict(self):
        value = asdict(self); value['points'] = list(self.points); value['weights'] = list(self.weights)
        return value

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class SketchWire:
    id: str
    entities: tuple[str, ...]
    closed: bool = False

    def __post_init__(self):
        if not self.id:
            raise GeometryError('a contour needs a stable ID')
        object.__setattr__(self, 'entities', tuple(self.entities))

    def to_dict(self):
        return {'id': self.id, 'entities': list(self.entities), 'closed': self.closed}

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class SketchSolveResult:
    points: Mapping[str, tuple[float, float]]
    converged: bool
    rank: int
    degrees_of_freedom: int
    conflicts: tuple[str, ...]
    residual: float


def upgrade_sketch(definition):
    """Upgrade intent without changing legacy output slot meanings."""
    if definition.version == 2:
        return definition
    pairs = list(zip(definition.path, definition.path[1:]))
    if definition.closed:
        pairs.append((definition.path[-1], definition.path[0]))
    entities = tuple(SketchEntity(f'legacy-{i}', 'line', pair) for i, pair in enumerate(pairs))
    constraints = tuple(replace(c, id=c.id or f'constraint-{i}') for i, c in enumerate(definition.constraints))
    return replace(definition, version=2, entities=entities, wires=(SketchWire('profile', tuple(e.id for e in entities), definition.closed),), constraints=constraints)


def solve_sketch_edit(definition, plane, *, fixed_points=(), drag_targets=None, tolerance=1e-9, maximum_iterations=60):
    """Project editing targets onto the constraint manifold.

    Constraints are hard. Targets are soft, including targets for constrained
    points; failure returns the unchanged input and deterministic diagnostics.
    """
    from .sketch import _constraint_system
    keys = tuple(definition.points); n = len(keys); index = {k:i for i,k in enumerate(keys)}
    original = np.asarray([definition.points[k] for k in keys], dtype=float).reshape((n, 2))
    coordinates = original.copy(); targets = dict(drag_targets or {})
    fixed = set(fixed_points) | {c.first for c in definition.constraints if c.kind == 'fixed'}
    if (set(targets) | fixed) - set(keys):
        raise GeometryError('editing target references unavailable point')
    for c in definition.constraints:
        if c.boundary_index is not None:
            bound = plane.boundary_edges if c.kind == 'on_edge' else plane.boundary_vertices
            if c.boundary_index >= len(bound):
                raise GeometryError('sketch support reference is unavailable')
    constraints = tuple(c for c in definition.constraints if c.kind != 'fixed')
    anchors={key:original[index[key]].copy() for key in fixed}
    anchors.update({c.first:np.asarray(c.position,dtype=float) for c in definition.constraints if c.kind=='fixed' and c.position is not None})
    for c in definition.constraints:
        if c.kind=='on_vertex':
            fixed.add(c.first)
            anchors[c.first]=np.asarray(plane.boundary_vertices[c.boundary_index],dtype=float)
    def system(x):
        r,j = _constraint_system(x, keys, constraints, plane)
        rows = list(j.reshape((-1,2*n))) if n else []; values = list(r)
        for key in sorted(fixed):
            for axis in (0,1):
                row = np.zeros(2*n); row[2*index[key]+axis] = 1.
                rows.append(row); values.append(x[index[key],axis]-anchors[key][axis])
        point_map={k:tuple(x[i]) for k,i in index.items()}
        for entity in definition.entities:
            if entity.radius is None or entity.kind not in ('circle','arc'): continue
            current=sketch_entity_radius(definition,entity,point_map)
            row=np.zeros(2*n)
            for key in entity.points:
                for axis in (0,1):
                    step=max(1e-7,abs(x[index[key],axis])*1e-7)
                    before=dict(point_map); after=dict(point_map)
                    lo=list(before[key]); hi=list(after[key]); lo[axis]-=step; hi[axis]+=step
                    before[key]=tuple(lo); after[key]=tuple(hi)
                    row[2*index[key]+axis]=(sketch_entity_radius(definition,entity,after)-sketch_entity_radius(definition,entity,before))/(2*step)
            rows.append(row); values.append(current-entity.radius)
        return np.asarray(values), np.asarray(rows).reshape((-1,2*n))
    if not n:
        return SketchSolveResult({}, True, 0, 0, (), 0.)
    desired = original.copy()
    for key,value in targets.items():
        value = np.asarray(value,dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise GeometryError('drag target needs two finite coordinates')
        desired[index[key]] = value
    # Incremental projection follows curved feasible paths while remaining close
    # to the original free coordinates. Constrained drag targets are never rows
    # in the hard Jacobian.
    for fraction in np.linspace(0.,1.,17)[1:]:
        candidate = coordinates + (desired-original)/16.
        for iteration in range(maximum_iterations):
            residual,jacobian = system(candidate)
            if not len(residual) or np.max(np.abs(residual)) <= tolerance:
                break
            correction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            candidate += correction.reshape((n,2))
            if not np.all(np.isfinite(candidate)):
                break
        residual,jacobian = system(candidate)
        if not np.all(np.isfinite(candidate)) or (len(residual) and np.max(np.abs(residual)) > tolerance):
            rank = int(np.linalg.matrix_rank(jacobian)) if jacobian.size else 0
            conflicts = tuple(c.id or f'constraint-{i}' for i,c in enumerate(definition.constraints))+tuple(f'radius/{e.id}' for e in definition.entities if e.radius is not None and e.kind in ('circle','arc'))
            return SketchSolveResult(dict(definition.points),False,rank,2*n-rank,conflicts,float(np.max(np.abs(residual))) if len(residual) else float('inf'))
        coordinates = candidate
        for key in fixed:
            coordinates[index[key]] = anchors[key]
    residual,jacobian = system(coordinates)
    rank = int(np.linalg.matrix_rank(jacobian)) if jacobian.size else 0
    return SketchSolveResult({k:tuple(map(float,coordinates[i])) for i,k in enumerate(keys)},True,rank,2*n-rank,(),float(np.max(np.abs(residual))) if len(residual) else 0.)


def _interpolation_controls(degree, knots, parameters, values):
    """Controls of the B-spline through ``values`` at ``parameters`` (exact collocation).

    The square collocation matrix is nonsingular exactly when every site lies
    inside the support of its basis function (Schoenberg-Whitney); that is
    checked before solving. B-spline collocation matrices are totally positive,
    so Gaussian elimination with partial pivoting is backward stable here.
    """
    knots = np.asarray(knots,dtype=float); count = len(parameters)
    interior = np.arange(1,count-1)
    if len(knots) != count+degree+1 or np.any(knots[interior] >= parameters[interior]) \
            or np.any(parameters[interior] >= knots[interior+degree+1]):
        raise GeometryError('spline fit points violate the Schoenberg-Whitney condition')
    spans = _reference.find_spans(degree,knots,count,parameters)
    basis = _reference.basis_derivatives(degree,knots,spans,parameters,0)[:,0,:]
    matrix = np.zeros((count,count))
    for row,(span,local) in enumerate(zip(spans,basis)):
        matrix[row,span-degree:span+1] = local
    try:
        controls = np.linalg.solve(matrix,values)
    except np.linalg.LinAlgError as error:
        raise GeometryError('spline fit-point interpolation system is singular') from error
    if not np.all(np.isfinite(controls)):
        raise GeometryError('spline fit-point interpolation is not representable in float64')
    return tuple(tuple(map(float,point)) for point in controls)


def _nurbs(points, degree=3, weights=(), fit=False, closed=False):
    values = np.asarray(points,dtype=float)
    if len(values) < 2:
        raise GeometryError('a spline needs at least two points')
    if closed:
        values = np.vstack((values,values[0]))
    degree = min(int(degree),len(values)-1)
    if degree < 1:
        raise GeometryError('spline degree must be positive')
    if fit:
        if not np.all(np.isfinite(values)):
            raise GeometryError('spline fit points must be finite')
        distances = np.linalg.norm(np.diff(values,axis=0),axis=1)
        if np.any(distances <= 1e-14):
            raise GeometryError('spline fit points must be distinct neighbours')
        t = np.r_[0.,np.cumsum(distances)]; t /= t[-1]
        if np.any(np.diff(t) <= 0.):
            raise GeometryError('spline fit point chord parameters must be strictly increasing')
        # Three fit points define a quadratic, not two straight spans.
        # Its explicit clamped knots avoid relying on an even-degree
        # not-a-knot convention. Retain the existing cubic default for
        # larger fit-point sets when a quadratic was requested.
        if degree == 2:
            if len(values) == 3:
                knots = np.asarray((0., 0., 0., 1., 1., 1.))
            else:
                degree = 3
        if degree != 2:
            if degree % 2 != 1:
                raise GeometryError('fit-point sketch splines need an odd degree (or a three-point quadratic)')
            # Not-a-knot end conditions for odd degree: interior knots at the
            # chord parameters, omitting (degree-1)/2 + 1 sites at each end.
            m = (degree-1)//2
            knots = np.r_[(t[0],)*(degree+1),t[m+1:len(t)-m-1],(t[-1],)*(degree+1)]
        return NurbsCurve(degree,tuple(map(float,knots)),_interpolation_controls(degree,knots,t,values))
    count = len(values); interior = count-degree-1
    knots = (0.,)*(degree+1)+tuple(i/(interior+1) for i in range(1,interior+1))+(1.,)*(degree+1)
    w = tuple(weights)
    if closed and w:
        w += (w[0],)
    return NurbsCurve(degree,knots,tuple(map(tuple,values)),w or None)


def _entity_parts(definition, entity, points=None):
    """Return exact local (kind, coordinates/curve) pieces."""
    points = definition.points if points is None else points
    values = [np.asarray(points[k],dtype=float) for k in entity.points]
    minimum = {'line':2,'arc':3,'circle':2,'spline':2,'nurbs':2,'fillet':3}[entity.kind]
    if len(values) < minimum:
        raise GeometryError(f'{entity.kind} needs {minimum} points')
    if entity.kind == 'line':
        return [('line',values[:2])]
    if entity.kind == 'arc':
        arc_frame(*[np.r_[v,0.] for v in values[:3]])
        return [('arc',values[:3])]
    if entity.kind == 'circle':
        center,rim = values[:2]; vector = rim-center; radius = float(np.linalg.norm(vector))
        if radius <= 1e-14:
            raise GeometryError('circle radius must be positive')
        perpendicular = np.asarray((-vector[1],vector[0]))
        corners=(vector,perpendicular,-vector,-perpendicular,vector)
        return [('arc',[center+corners[i],center+(corners[i]+corners[i+1])/np.sqrt(2),center+corners[i+1]]) for i in range(4)]
    if entity.kind == 'fillet':
        first,corner,last = values[:3]; a=first-corner; b=last-corner
        lengths=np.asarray((np.linalg.norm(a),np.linalg.norm(b))); a/=lengths[0]; b/=lengths[1]
        theta=float(np.arccos(np.clip(a@b,-1,1)))
        if theta <= 1e-8 or theta >= np.pi-1e-8 or entity.radius is None:
            raise GeometryError('rounded corner needs two non-collinear lines and radius')
        distance=entity.radius/np.tan(theta/2)
        if distance >= min(lengths):
            raise GeometryError('rounded corner radius exceeds adjacent lines')
        p=corner+a*distance; q=corner+b*distance; center=corner+(a+b)/np.linalg.norm(a+b)*entity.radius/np.sin(theta/2)
        middle=center+(corner-center)/np.linalg.norm(corner-center)*entity.radius
        return [('line',[first,p]),('arc',[p,middle,q]),('line',[q,last])]
    curve = _nurbs(values,entity.degree,entity.weights,entity.kind=='spline',entity.closed)
    # A closed curve is split so topology endpoints remain distinct.
    return [('nurbs',curve.subcurve(i/4,(i+1)/4)) for i in range(4)] if entity.closed else [('nurbs',curve)]


def sample_sketch_entity(definition, entity, *, count=48, points=None):
    if isinstance(entity,str):
        entity=next(e for e in definition.entities if e.id==entity)
    result=[]
    for kind,value in _entity_parts(definition,entity,points):
        if kind=='line':
            samples=np.asarray(value)
        elif kind=='arc':
            frame=arc_frame(*[np.r_[v,0.] for v in value]); t=np.linspace(0,frame.sweep,max(3,count))
            samples=(frame.center+frame.radius*(np.cos(t)[:,None]*frame.e1+np.sin(t)[:,None]*frame.e2))[:,:2]
        else:
            samples=value.evaluate(np.linspace(0,1,max(3,count)))
        result.extend(tuple(map(float,p)) for p in samples[(1 if result else 0):])
    return tuple(result)


def sketch_entity_radius(definition,entity,points=None):
    if isinstance(entity,str): entity=next(e for e in definition.entities if e.id==entity)
    p=definition.points if points is None else points
    if entity.kind=='circle': return float(np.linalg.norm(np.asarray(p[entity.points[1]])-p[entity.points[0]]))
    if entity.kind=='fillet': return entity.radius
    if entity.kind=='arc': return arc_frame(*[np.r_[p[k],0.] for k in entity.points]).radius
    raise GeometryError('select an arc, circle or rounded corner')


def set_sketch_radius(definition,entity_id,radius):
    entity=next(e for e in definition.entities if e.id==entity_id); radius=float(radius)
    if not np.isfinite(radius) or radius<=0: raise GeometryError('radius must be finite and positive')
    if entity.kind=='fillet':
        updated=replace(entity,radius=radius); _entity_parts(definition,updated)
        return replace(definition,entities=tuple(updated if e.id==entity_id else e for e in definition.entities))
    points=dict(definition.points)
    if entity.kind=='circle':
        a,b=entity.points; direction=np.asarray(points[b])-points[a]; length=np.linalg.norm(direction)
        if length<=1e-14: raise GeometryError('circle has no radius direction')
        points[b]=tuple(np.asarray(points[a])+direction*radius/length)
    elif entity.kind=='arc':
        frame=arc_frame(*[np.r_[points[k],0.] for k in entity.points])
        for key in entity.points:
            points[key]=tuple((frame.center+(np.r_[points[key],0.]-frame.center)*radius/frame.radius)[:2])
    else: raise GeometryError('select an arc, circle or rounded corner')
    updated=replace(entity,radius=radius)
    return replace(definition,points=points,entities=tuple(updated if e.id==entity_id else e for e in definition.entities))


def _order_sketch_contour(geometry, edge_ids):
    """Order sketch/inner-trim contours, including an arc and its chord.

    Outer face construction still requires three edges. A sketch or inner
    trim may have two distinct curved edges sharing its two endpoint vertices.
    """
    from .entities import OrientedEdge

    identifiers = tuple(edge_ids)
    if len(identifiers) != 2:
        return geometry.order_loop(identifiers)
    if identifiers[0] == identifiers[1]:
        raise GeometryError('a closed sketch contour needs distinct edges')
    first, second = (geometry.edges[key] for key in identifiers)
    if (first.start == first.end
            or {first.start, first.end} != {second.start, second.end}):
        raise GeometryError('sketch contour edges do not form a closed loop')
    return (OrientedEdge(first.id, True),
            OrientedEdge(second.id, second.start == first.end))


def materialize_entities(geometry,plane,definition):
    result=solve_sketch_edit(definition,plane)
    if not result.converged: raise GeometryError('inconsistent sketch constraints: '+', '.join(result.conflicts))
    if not definition.entities: raise GeometryError('a finished sketch needs geometry')
    cache={}
    aliases={k:k for k in result.points}
    for c in definition.constraints:
        if c.kind=='coincident':
            a,b=aliases[c.first],aliases[c.second]
            aliases={k:(a if v==b else v) for k,v in aliases.items()}
    def vertex(local,key):
        key=aliases.get(key,key)
        if key not in cache: cache[key]=geometry.add_point(*plane.world(local))
        return cache[key]
    # Bound reference anchors are solve-time only: they never become sketch outputs.
    # Only actual bound identities are skipped; an old user point named "ref:x" is kept.
    vertices={k:vertex(p,k) for k,p in result.points.items() if k not in definition.bound_reference_keys}
    outputs={f'point/{k}':EntityRef('vertex',v) for k,v in vertices.items()}; edges={}
    for entity in definition.entities:
        owned=[]
        parts=_entity_parts(definition,entity,result.points)
        first_key=entity.points[1] if entity.kind=='circle' else entity.points[0]
        last_key=first_key if entity.closed or entity.kind=='circle' else entity.points[-1]
        for piece,(kind,value) in enumerate(parts):
            start_key=first_key if piece==0 else (entity.id,'joint',piece)
            end_key=last_key if piece==len(parts)-1 else (entity.id,'joint',piece+1)
            if kind=='line': edge=geometry.add_line(vertex(value[0],start_key),vertex(value[1],end_key))
            elif kind=='arc':
                via_key=entity.points[1] if entity.kind=='arc' else (entity.id,'via',piece)
                edge=geometry.add_arc(vertex(value[0],start_key),vertex(value[1],via_key),vertex(value[2],end_key))
            else:
                controls=tuple(tuple(plane.world(p)) for p in value.controls)
                curve=replace(value,controls=controls); endpoints=curve.evaluate(np.asarray((0.,1.)))
                edge=geometry.add_curve(vertex(plane.local(endpoints[0]),start_key),vertex(plane.local(endpoints[1]),end_key),curve)
            owned.append(edge); outputs[f'entity/{entity.id}/edge/{piece}']=EntityRef('edge',edge)
        edges[entity.id]=owned
        if entity.id.startswith('legacy-'):
            outputs[f'profile/edge/{entity.id[7:]}']=EntityRef('edge',owned[0])
    for wire in definition.wires:
        flattened=[edge for key in wire.entities for edge in edges[key]]
        if not flattened: raise GeometryError('a finished contour cannot be empty')
        # Wire membership is intent, not an ordinal topology identity. Stable
        # entity-piece outputs are the only accepted v2 edge references.
        if wire.closed:
            _order_sketch_contour(geometry,flattened)
    if abs(definition.extrusion)>1e-15:
        before=set(geometry.faces); geometry.extrude([e for group in edges.values() for e in group],tuple(plane.normal*definition.extrusion))
        outputs.update({f'extrusion/face/{i}':EntityRef('face',v) for i,v in enumerate(sorted(set(geometry.faces)-before))})
    return outputs


def sketch_wire_edge_keys(definition,wire_ids=()):
    legacy=definition.version==1
    definition=upgrade_sketch(definition)
    available={w.id:w for w in definition.wires if w.closed}
    chosen=tuple(wire_ids) or tuple(available)
    if not chosen: raise GeometryError('select a closed sketch contour')
    if any(key not in available for key in chosen): raise GeometryError('a selected sketch contour is missing or open')
    entities={e.id:e for e in definition.entities}
    if legacy:
        return tuple(tuple(f'profile/edge/{i}' for i in range(len(available[key].entities))) for key in chosen)
    return tuple(tuple(f'entity/{entity_id}/edge/{piece}'
                       for entity_id in available[key].entities
                       for piece in range(len(_entity_parts(definition,entities[entity_id]))))
                 for key in chosen)


def round_sketch_corner(definition,first_entity_id,second_entity_id,radius):
    definition=upgrade_sketch(definition); lookup={e.id:e for e in definition.entities}
    first=lookup[first_entity_id]; second=lookup[second_entity_id]
    if first.kind!='line' or second.kind!='line': raise GeometryError('select two connected straight lines')
    common=set(first.points)&set(second.points)
    if len(common)!=1: raise GeometryError('select two lines sharing one corner')
    corner=common.pop(); a=next(k for k in first.points if k!=corner); b=next(k for k in second.points if k!=corner)
    # Both original lines are retired. Reusing either identity for the shortened
    # composite would map its old corner endpoint onto a derived tangent point,
    # conflicting with the retained virtual driving corner's point identity.
    fillet=SketchEntity(str(uuid4()),'fillet',(a,corner,b),float(radius))
    _entity_parts(definition,fillet)
    wires=[]
    for wire in definition.wires:
        members=list(wire.entities)
        if first.id in members or second.id in members:
            if not {first.id,second.id}<=set(members): raise GeometryError('both lines must belong to the same contour')
            i=members.index(first.id); j=members.index(second.id)
            if (i-j)%len(members) not in (1,len(members)-1): raise GeometryError('rounded corner lines must be consecutive')
            if (i+1)%len(members)!=j:
                fillet=replace(fillet,points=(b,corner,a))
            members=[fillet.id if key==first.id else key for key in members if key!=second.id]
        wires.append(replace(wire,entities=tuple(members)))
    return replace(definition,entities=tuple(fillet if e.id==first.id else e for e in definition.entities if e.id!=second.id),wires=tuple(wires))
