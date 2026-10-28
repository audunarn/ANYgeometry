"""Ruled lofts on canonical NURBS supports, with conservative admission."""
from fractions import Fraction as Q
from dataclasses import replace
from math import comb
import numpy as np
from .curves import Straight, Spline, Arc
from .entities import OrientedEdge
from .errors import GeometryError
from .nurbs import NurbsCurve, NurbsSurface, CurveOnSurface, MAX_DEGREE, _exact_refine
from .surfaces import CoonsSurface


class LoftNotCertified(GeometryError):
    """The conservative loft admission proof could not establish validity."""



def _check(cancel):
    if cancel is not None and cancel():
        raise GeometryError('loft cancelled')


def _product(a, b):
    p, q = len(a)-1, len(b)-1
    return [sum(a[i]*b[k-i]*Q(comb(p,i)*comb(q,k-i),comb(p+q,k))
                for i in range(max(0,k-q),min(p,k)+1)) for k in range(p+q+1)]


def _elevate(values, degree):
    result = list(values)
    while len(result)-1 < degree:
        n = len(result)
        result = [result[0]]+[Q(i,n)*result[i-1]+Q(n-i,n)*result[i]
                              for i in range(1,n)]+[result[-1]]
    return result


def _homogeneous(curve, lower, upper):
    start, end = map(Q,curve.domain)
    a,b = start+Q(lower)*(end-start),start+Q(upper)*(end-start)
    weights = curve.weights or (1.,)*len(curve.controls)
    net = [tuple(Q(x)*Q(w) for x in point)+(Q(w),) for point,w in zip(curve.controls,weights)]
    _, refined, active = _exact_refine(curve.degree,list(map(Q,curve.knots)),net,min(a,b),max(a,b))
    rows = [refined[i] for i in active]
    if len(rows) != curve.degree+1:
        raise GeometryError('loft interval crosses an unresolved NURBS knot')
    return rows if a < b else rows[::-1]


def _curve(model, edge):
    curve = edge.curve
    if isinstance(curve,NurbsCurve): return curve
    if isinstance(curve,CurveOnSurface):
        # _iso_world is an internal exact hull/evaluation record, not a public
        # NurbsCurve conversion (it includes a parameter map and error slack).
        # Retain the source support/transform contract rather than discarding
        # that information to manufacture a standalone curve.
        raise GeometryError('loft does not yet support curve-on-surface profiles; select standalone profile curves')
    if isinstance(curve,Straight):
        points = (model.vertex_position(edge.start),model.vertex_position(edge.end))
    elif isinstance(curve,Spline): points = model._spline_points(edge)
    else: raise GeometryError('loft requires lines, Bezier splines or NURBS curves; convert this profile explicitly')
    degree = len(points)-1
    return NurbsCurve(degree,(0.,)*(degree+1)+(1.,)*(degree+1),tuple(tuple(p) for p in points))


def _ordered_profile_edges(model, identifiers, reverse, start):
    ids = tuple(identifiers)
    if not ids or len(set(ids)) != len(ids):
        raise GeometryError('loft profile must be a nonempty chain without repeated edges')
    edges = [model._require_edge(i) for i in ids]
    counts = {}
    for edge in edges:
        for vertex in (edge.start,edge.end): counts[vertex] = counts.get(vertex,0)+1
    if any(count > 2 for count in counts.values()): raise GeometryError('loft profile branches')
    closed = all(count == 2 for count in counts.values())
    if not closed and start != 0: raise GeometryError('open loft profiles have no seam; start must be zero')
    if closed:
        if start >= len(edges): raise GeometryError('loft seam index is outside the profile')
        edges = edges[start:]+edges[:start]
    def follow(forward):
        result, current = [], edges[0].start if forward else edges[0].end
        for edge in edges:
            if edge.start == current: forward,current = True,edge.end
            elif edge.end == current: forward,current = False,edge.start
            else: return None
            result.append((edge,forward))
        return result
    ordered = follow(True) or follow(False)
    if ordered is None: raise GeometryError('loft edges must be ordered as one connected chain')
    if reverse: ordered = [(edge,not forward) for edge,forward in reversed(ordered)]
    return ordered,closed


def _profile(model, identifiers, reverse, start):
    ordered,closed = _ordered_profile_edges(model,identifiers,reverse,start)
    curves = [_curve(model,edge) if forward else _curve(model,edge).subcurve(1.,0.) for edge,forward in ordered]
    lengths = np.asarray([model.edge_length(edge.id) for edge,_ in ordered])
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= model.tolerance.length):
        raise GeometryError('loft profile contains a zero-length edge')
    breaks = np.r_[0.,np.cumsum(lengths)/sum(lengths)]; breaks[-1] = 1.
    anchors = set(map(float,breaks))
    for curve,a,b in zip(curves,breaks[:-1],breaks[1:]):
        anchors.update(float(a+t*(b-a)) for t in curve.breakpoints() if 0.<t<1.)
    return curves,breaks,anchors,closed


def _merge_profile_anchors(profiles,tolerance):
    tagged=sorted((value,index) for index,profile in enumerate(profiles) for value in profile[2])
    groups=[]
    for value,index in tagged:
        if not groups or value-groups[-1][0][0]>tolerance:groups.append([])
        groups[-1].append((value,index))
    for group in groups:
        if len({index for _,index in group})!=len(group):
            raise LoftNotCertified('profile stations are closer than the geometry parameter tolerance')
    return [group[0][0] for group in groups]


def _piece(profile, a, b, tolerance=0.):
    curves,breaks,_,_closed = profile
    index = min(int(np.searchsorted(breaks,(a+b)/2,side='right'))-1,len(curves)-1)
    left,right = breaks[index:index+2]
    def parameter(value):
        knots=(0.,*curves[index].breakpoints(),1.)
        nearest=min(knots,key=lambda t:abs(float(left+t*(right-left))-value))
        if abs(float(left+nearest*(right-left))-value)<=tolerance:return nearest
        return float((value-left)/(right-left))
    return _homogeneous(curves[index],parameter(a),parameter(b))


def _patch(first, second):
    # Common product denominator preserves LINEAR world interpolation in v.
    wa,wb = [p[-1] for p in first],[p[-1] for p in second]
    if len(set(wa)) == len(set(wb)) == 1:
        degree=max(len(first),len(second))-1
        return ([_elevate([p[k]/wa[0] for p in first],degree) for k in range(3)],
                [_elevate([p[k]/wb[0] for p in second],degree) for k in range(3)],
                [Q(1)]*(degree+1))
    return ([_product([p[k] for p in first],wb) for k in range(3)],
            [_product([p[k] for p in second],wa) for k in range(3)],_product(wa,wb))


def _derivative(values):
    n = len(values)-1
    return [n*(b-a) for a,b in zip(values,values[1:])]


def _subtract(a,b): return [x-y for x,y in zip(a,b)]


def _affine_curve(curve):
    """True only if stored rational controls define affine parameterization."""
    weights = curve.weights or (1.,)*len(curve.controls)
    if len(set(weights)) != 1: return False
    first,last = curve.controls[0],curve.controls[-1]
    degree = len(curve.controls)-1
    return all(Q(x) == Q(a)+Q(i,degree)*(Q(b)-Q(a))
               for i,point in enumerate(curve.controls) for x,a,b in zip(point,first,last))


def _bound_storage_error(model, a, b, w, controls, weights):
    """Bound boundary change from rounding and shared-endpoint normalization.

    Positive rational weights form convex combinations. Relative weight error r
    perturbs normalized coefficients in L1 by at most 2r/(1-r). Coordinate
    control error is bounded independently; no samples establish this claim.
    """
    normalized = [v/w[0] for v in w]
    relative = max(abs(Q(row[0])/value-1) for row,value in zip(weights,normalized))
    if relative >= 1: raise LoftNotCertified('loft stored weights cannot preserve the source boundary')
    for side,index in ((a,0),(b,1)):
        coordinate_bounds = []; scale2 = Q(0)
        for k in range(3):
            exact = [value/weight for value,weight in zip(side[k],w)]
            change = max(abs(Q(point[index][k])-value) for point,value in zip(controls,exact))
            span = max(exact)-min(exact)
            scale2 += span*span
            coordinate_bounds.append(change+2*relative/(1-relative)*span)
        error2 = sum(value*value for value in coordinate_bounds)
        tolerance = Q(model.tolerance.effective_curve_fit_residual(float(scale2)**.5))
        if error2 > tolerance*tolerance:
            raise LoftNotCertified('loft stored boundary error exceeds the geometry curve-fit tolerance')


def _regular_projection(patches, axes):
    x,y = axes
    signs = []
    for a,b,w in patches:
        dx,dy = _subtract(b[x],a[x]),_subtract(b[y],a[y])
        for row in (a,b):
            ux = _subtract(_product(_derivative(row[x]),w),_product(row[x],_derivative(w)))
            uy = _subtract(_product(_derivative(row[y]),w),_product(row[y],_derivative(w)))
            determinant = _subtract(_product(ux,dy),_product(uy,dx))
            if min(determinant)>0: signs.append(1)
            elif max(determinant)<0: signs.append(-1)
            else: return False
    return len(set(signs)) == 1


def _certify(surface, patches, cancel):
    from .nurbs_audit import certify_simple,certify_only_shared_ends
    boundaries = (surface.iso_curve('v',0.),surface.iso_curve('u',1.),
                  surface.iso_curve('v',1.).subcurve(1.,0.),surface.iso_curve('u',0.).subcurve(1.,0.))
    for axes in ((0,1),(0,2),(1,2)):
        _check(cancel)
        if not _regular_projection(patches,axes): continue
        curves = [NurbsCurve(c.degree,c.knots,tuple(tuple(p[k] for k in axes) for p in c.controls),c.weights,c.domain) for c in boundaries]
        if not all(certify_simple(c) is True for c in curves): continue
        if all(certify_only_shared_ends(a,b) is True for i,a in enumerate(curves) for b in curves[i+1:]): return
    raise LoftNotCertified('loft regularity and simple projection could not be certified; change profile correspondence')


def _separate_patch_hulls(patches, closed, cancel):
    """Exact sufficient separation, allowing only an authored common ruling."""
    hulls = [tuple(tuple(side[k][i]/w[i] for k in range(3))
                   for side in (a,b) for i in range(len(w))) for a,b,w in patches]
    for i,first in enumerate(hulls):
        for j in range(i+1,len(hulls)):
            _check(cancel)
            second = hulls[j]
            if any(max(p[k] for p in first)<min(p[k] for p in second) or
                   max(p[k] for p in second)<min(p[k] for p in first) for k in range(3)): continue
            adjacent = j == i+1 or (closed and i == 0 and j == len(hulls)-1)
            centre_a = tuple(sum(p[k] for p in first)/len(first) for k in range(3))
            centre_b = tuple(sum(p[k] for p in second)/len(second) for k in range(3))
            normal = tuple(y-x for x,y in zip(centre_a,centre_b))
            if adjacent:
                a,b,w = patches[i]
                index = 0 if closed and i == 0 and j == len(hulls)-1 else -1
                start = tuple(a[k][index]/w[index] for k in range(3))
                end = tuple(b[k][index]/w[index] for k in range(3))
                direction = tuple(y-x for x,y in zip(start,end))
                norm = sum(x*x for x in direction)
                if not norm: raise GeometryError('loft has a collapsed connector')
                projection = sum(n*d for n,d in zip(normal,direction))/norm
                normal = tuple(n-projection*d for n,d in zip(normal,direction))
                def cross(a,b):
                    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
                def on_segment(p):
                    t = sum((v-o)*d for v,o,d in zip(p,start,direction))/norm
                    return 0 <= t <= 1 and all(v == o+t*d for v,o,d in zip(p,start,direction))
                def separates(n):
                    def side(p):return sum(x*(v-o) for x,v,o in zip(n,p,start))
                    return (all(side(p)<=0 and (side(p)<0 or on_segment(p)) for p in first) and
                            all(side(p)>=0 and (side(p)>0 or on_segment(p)) for p in second))
                candidates=[normal]
                for p in (*first,*second):
                    n=cross(direction,tuple(v-o for v,o in zip(p,start)))
                    if any(n):candidates.extend((n,tuple(-x for x in n)))
                # The feasible separator cone lies in the plane perpendicular
                # to the common ruling. Boundary rays are induced by controls;
                # sums of ray pairs include an interior ray of every sector.
                separated=any(separates(n) for n in candidates)
                if not separated:
                    separated=any(separates(tuple(x+y for x,y in zip(a,b)))
                                  for k,a in enumerate(candidates) for b in candidates[k+1:])
                if separated:continue
            else:
                def cross(a,b):
                    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
                def differences(points):
                    return [tuple(y-x for x,y in zip(a,b)) for k,a in enumerate(points) for b in points[k+1:]]
                da,db=differences(first),differences(second)
                axes=[normal]
                axes.extend(cross(a,b) for k,a in enumerate(da) for b in da[k+1:])
                axes.extend(cross(a,b) for k,a in enumerate(db) for b in db[k+1:])
                axes.extend(cross(a,b) for a in da for b in db)
                def separated(n):
                    pa=[sum(x*y for x,y in zip(n,p)) for p in first]
                    pb=[sum(x*y for x,y in zip(n,p)) for p in second]
                    return max(pa)<min(pb) or max(pb)<min(pa)
                if any(separated(n) for n in axes):continue
            raise LoftNotCertified(f'loft patch separation could not be certified for patches {i} and {j}; change profile correspondence')


def _translated_arc_loft(model, first_edges, second_edges, reverse_first, reverse_second,
                         first_start, second_start, flip_normal, cancel):
    """Exact authored Arc extrusion; no rational-circle reclassification.

    A nonzero extrusion transverse to the arc plane is injective: projection
    along the extrusion onto that plane uniquely recovers the simple arc.
    """
    if len(first_edges) != 1 or len(second_edges) != 1 or first_start or second_start:
        raise GeometryError('Arc loft currently needs one translated arc in each open profile')
    edges = [model._require_edge(ids[0]) for ids in (first_edges,second_edges)]
    if not all(isinstance(edge.curve,Arc) for edge in edges):
        raise GeometryError('Arc loft currently requires two exactly translated arcs')
    triples = []
    for edge,reverse in zip(edges,(reverse_first,reverse_second)):
        ids = (edge.start,edge.curve.via_vertex,edge.end)
        if reverse: ids = ids[::-1]
        triples.append(tuple(tuple(Q(float(x)) for x in model.vertex_position(i)) for i in ids))
    delta = tuple(b-a for a,b in zip(triples[0][0],triples[1][0]))
    if any(tuple(b-a for a,b in zip(p,q)) != delta for p,q in zip(*triples)):
        raise GeometryError('Arc loft requires exactly translated endpoint and via-point definitions')
    a,b,c = triples[0]
    u,v = tuple(y-x for x,y in zip(a,b)),tuple(y-x for x,y in zip(a,c))
    normal = (u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
    if sum(n*d for n,d in zip(normal,delta)) == 0:
        raise LoftNotCertified('Arc loft extrusion must be transverse to the profile plane')
    _check(cancel)
    ids = [tuple(model.add_point(*map(float,p)) for p in row) for row in triples]
    _check(cancel)
    bottom = model.add_arc(ids[0][0],ids[0][1],ids[0][2])
    top = model.add_arc(ids[1][0],ids[1][1],ids[1][2])
    left,right = model.add_line(ids[0][0],ids[1][0]),model.add_line(ids[0][2],ids[1][2])
    loop = (OrientedEdge(bottom,True),OrientedEdge(right,True),OrientedEdge(top,False),OrientedEdge(left,False))
    if flip_normal:
        loop = (OrientedEdge(bottom,False),OrientedEdge(left,True),OrientedEdge(top,True),OrientedEdge(right,False))
    face = model._add_face_from_loop(loop,(0,1,2,3),surface=CoonsSurface())
    _check(cancel)
    return [face]


def build_loft(model, first_edges, second_edges, *, reverse_first=False, reverse_second=False,
               first_start=0, second_start=0, flip_normal=False, cancellation_check=None):
    """Construct an admitted open ruled strip inside the model transaction."""
    for value in (reverse_first,reverse_second,flip_normal):
        if type(value) is not bool: raise GeometryError('loft direction flags must be booleans')
    for value in (first_start,second_start):
        if type(value) is not int or value < 0: raise GeometryError('loft start indices must be nonnegative integers')
    _check(cancellation_check)
    first_edges,second_edges = tuple(first_edges),tuple(second_edges)
    if any(isinstance(model._require_edge(edge).curve,Arc) for edge in (*first_edges,*second_edges)):
        from .loft_circles import build_circle_loft
        return build_circle_loft(model,first_edges,second_edges,reverse_first=reverse_first,
                                 reverse_second=reverse_second,first_start=first_start,
                                 second_start=second_start,flip_normal=flip_normal,
                                 cancellation_check=cancellation_check)
    first = _profile(model,first_edges,reverse_first,first_start)
    second = _profile(model,second_edges,reverse_second,second_start)
    if first[3] != second[3]:
        raise GeometryError('loft profiles must both be open or both closed')
    closed = first[3]
    anchors = _merge_profile_anchors((first,second),model.tolerance.parameter)
    patches = []
    for a,b in zip(anchors,anchors[1:]):
        _check(cancellation_check)
        patches.append(_patch(_piece(first,a,b,model.tolerance.parameter),_piece(second,a,b,model.tolerance.parameter)))
    degree = max(len(w)-1 for _,_,w in patches)
    if degree > MAX_DEGREE: raise GeometryError('loft rational product exceeds supported NURBS degree')
    surfaces,canonical = [],[]
    for number,(a,b,w) in enumerate(patches):
        a,b = [[_elevate(values,degree) for values in side] for side in (a,b)]
        w = _elevate(w,degree)
        cp = [(tuple(float(a[k][i]/w[i]) for k in range(3)),tuple(float(b[k][i]/w[i]) for k in range(3))) for i in range(degree+1)]
        wp = [(float(v/w[0]),float(v/w[0])) for v in w]
        if surfaces:
            if not np.allclose(cp[0],surfaces[-1].controls[-1],rtol=0,atol=model.tolerance.length):
                raise GeometryError('loft profile boundaries are not continuous')
            cp[0] = surfaces[-1].controls[-1]
        if closed and number == len(patches)-1:
            if not np.allclose(cp[-1],surfaces[0].controls[0],rtol=0,atol=model.tolerance.length):
                raise GeometryError('loft closed profile endpoints disagree')
            cp[-1] = surfaces[0].controls[0]
        _bound_storage_error(model,a,b,w,cp,wp)
        # Certify the actual stored coefficients, after boundary rounding.
        canonical.append(([[Q(cp[i][0][k])*Q(wp[i][0]) for i in range(degree+1)] for k in range(3)],
                          [[Q(cp[i][1][k])*Q(wp[i][0]) for i in range(degree+1)] for k in range(3)],
                          [Q(row[0]) for row in wp]))
        knots = (0.,)*(degree+1)+(1.,)*(degree+1)
        support = NurbsSurface(degree,1,knots,(0.,0.,1.,1.),tuple(cp),tuple(wp))
        _certify(support,[canonical[-1]],cancellation_check)
        surfaces.append(support)
    _separate_patch_hulls(canonical,closed,cancellation_check)
    vertices = [[model.add_point(*surface.controls[0][j]) for j in (0,1)] for surface in surfaces]
    vertices.append(vertices[0] if closed else [model.add_point(*surfaces[-1].controls[-1][j]) for j in (0,1)])
    connectors = [model.add_line(ids[0],ids[1]) for ids in vertices[:-1]]
    connectors.append(connectors[0] if closed else model.add_line(*vertices[-1]))
    faces = []
    for i,support in enumerate(surfaces):
        _check(cancellation_check)
        first_curve,second_curve = support.iso_curve('v',0.),support.iso_curve('v',1.)
        straight_first,straight_second = _affine_curve(first_curve),_affine_curve(second_curve)
        bottom = (model.add_line(vertices[i][0],vertices[i+1][0]) if straight_first else
                  model.add_curve(vertices[i][0],vertices[i+1][0],first_curve))
        top = (model.add_line(vertices[i][1],vertices[i+1][1]) if straight_second else
               model.add_curve(vertices[i][1],vertices[i+1][1],second_curve))
        loop = (OrientedEdge(bottom,True),OrientedEdge(connectors[i+1],True),OrientedEdge(top,False),OrientedEdge(connectors[i],False))
        if flip_normal:
            loop = (OrientedEdge(bottom,False),OrientedEdge(connectors[i],True),
                    OrientedEdge(top,True),OrientedEdge(connectors[i+1],False))
        # Topology Coons is authoritative. Straight connectors cancel its corner
        # blend, leaving precisely (1-v)A(u)+vB(u). Both profile rows have the
        # same stored rational denominator, so the canonical tensor agrees.
        face_id = model._add_face_from_loop(loop,(0,1,2,3),surface=CoonsSurface())
        # The topology map already evaluates the exact ruled equation. Keep
        # it authoritative for display too; no optional inverse-map contract
        # is implied by the internal tensor used to establish admission.
        faces.append(face_id)
    _check(cancellation_check)
    return faces


def loft_profiles_from_sketch(model, feature_id):
    """Resolve authored sketch wires to materialized profile edge identities."""
    from .sketch import SketchDefinition
    from .sketch_entities import upgrade_sketch, _entity_parts
    record = model.features.get(feature_id)
    if record.kind not in ('geometry.sketch.plane','geometry.sketch.extrude') or record.suppressed:
        raise GeometryError('select an active sketch feature')
    original = SketchDefinition.from_parameters(record.parameters)
    definition = upgrade_sketch(original)
    entities = {entity.id:entity for entity in definition.entities}
    result = []
    for wire in definition.wires:
        keys = (tuple(f'profile/edge/{i}' for i in range(len(wire.entities))) if original.version == 1 else
                tuple(f'entity/{identifier}/edge/{piece}' for identifier in wire.entities
                      for piece in range(len(_entity_parts(definition,entities[identifier])))))
        try: refs = tuple(record.outputs[key] for key in keys)
        except KeyError as error: raise GeometryError('sketch profile needs successful regeneration') from error
        if any(ref.kind != 'edge' or ref.id not in model.edges for ref in refs):
            raise GeometryError('sketch profile references stale geometry')
        if refs: result.append((wire.id,tuple(ref.id for ref in refs)))
    return tuple(result)


def loft_profile_options(model, first_edges, second_edges):
    """Propose deterministic endpoint alignment; callers persist these flags.

    Equal-cost choices retain authored chain direction. This is a proposal,
    never an implicit remapping during feature regeneration.
    """
    first_edges,second_edges = tuple(first_edges),tuple(second_edges)
    if any(isinstance(model._require_edge(i).curve,Arc) for i in (*first_edges,*second_edges)):
        from .loft_circles import circle_options
        return circle_options(model,first_edges,second_edges)
    first = _profile(model,first_edges,False,0)
    second = _profile(model,second_edges,False,0)
    if first[3] != second[3]: raise GeometryError('loft profiles must both be open or both closed')
    def at(profile,t):
        curves,breaks,_anchors,_closed = profile
        i = min(int(np.searchsorted(breaks,t,side='right'))-1,len(curves)-1)
        local = (t-breaks[i])/(breaks[i+1]-breaks[i])
        return curves[i].evaluate([local])[0]
    choices = []
    for start in range(len(second_edges) if first[3] else 1):
        for reverse in (False,True):
            candidate = _profile(model,second_edges,reverse,start)
            values = np.linspace(0.,1.,9) if first[3] else (0.,1.)
            cost = sum((Q(float(x))-Q(float(y)))**2 for t in values
                       for x,y in zip(at(first,t),at(candidate,t)))
            choices.append((cost,start,reverse))
    _,start,reverse = min(choices)
    return dict(reverse_first=False,reverse_second=reverse,first_start=0,second_start=start,flip_normal=False)
