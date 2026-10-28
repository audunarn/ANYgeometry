"""Finite declared-contact exclusions, never approximation or publication authority.

Only polynomial common-affine-projection pairs with complete rectangular source
material are eligible. All boxes use SHARED NATIVE KNOT coordinates. Unknown
geometry refuses; no sampling or tolerance enlargement authorizes an exclusion.
"""
from collections import Counter
from dataclasses import dataclass, fields
from types import SimpleNamespace
from fractions import Fraction as Q
from weakref import WeakKeyDictionary, ref

from .definition_binding import definition_checksum
from .errors import GeometryError
from .nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from .material_regions import query_material_surface_regions, validate_material_surface_regions_binding
from .nurbs_surface_intersection import query_nurbs_surface_intersection, NurbsIntersectionCurve, IntersectionStatus
from .nurbs_surface_intersection._records import StructuralCertificate
from .nurbs_surface_intersection._structural import ScalarField
from .nurbs_surface_intersection._bernstein import bezier_piece, derivative_grid, grid_ranges, subbox
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve, intersection_curve_operation
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .plane_nurbs import charge_box, charge_split, depth_limit
from .tolerance import TolerancePolicy


class LocalContactNotCertified(GeometryError):
    """Missing finite contact authority, never evidence of non-contact."""


@dataclass(frozen=True, slots=True)
class ContactStar:
    component: int
    parameter_interval: tuple
    inner_box: tuple
    outer_box: tuple
    dependent_axis: int
    derivative_margin: Q
    side_margin: Q
    exterior_squared_lower: Q
    # One complement per source support, including its full native ranges.
    exterior_rectangles: tuple


@dataclass(frozen=True, eq=False)
class LocalContactStars:
    source_model_id: object
    face_ids: tuple
    source_checksum: str
    intent: str
    coordinate_system: str
    components: tuple
    stars: tuple
    work_units: int


_ISSUED = WeakKeyDictionary()


def _fail(message):
    raise LocalContactNotCertified(message)


def _runtime():
    # Pin the effective functions used by this adapter, including in-place code
    # and default mutation. Native coefficient snapshots remain a separate gate.
    functions=(NurbsSurface.exact_point,NurbsSurface.exact_net,NurbsSurface.patch_bounds,
        NurbsCurve.exact_point,NurbsSurfaceIntersectionCurve._shape,
        NurbsSurfaceIntersectionCurve.chart_box,query_nurbs_surface_intersection,
        query_material_surface_regions,validate_material_surface_regions_binding)
    return tuple((f,getattr(f,'__code__',None),definition_checksum(getattr(f,'__defaults__',None)),
                  definition_checksum(getattr(f,'__kwdefaults__',None))) for f in functions)


def _state(source):
    from .prepared_material_source_transfer import _state as native_state
    return (source.model_id, *native_state(source), _runtime())


def _rectangle(domain, support):
    """Prove the ACTUAL loop is the four full sides, not just its hull."""
    if len(domain.boundaries) != 1 or len(domain.boundaries[0]) != 4:
        _fail('contact requires an unholed four-side source material rectangle')
    sides = []
    identity = tuple(tuple(Q(int(i == j)) for j in range(4)) for i in range(4))
    for path in domain.boundaries[0]:
        curve = path.curve
        if (type(curve) is not CurveOnSurface or definition_checksum(curve.support) != definition_checksum(support)
                or tuple(tuple(map(Q, row)) for row in curve.transform) != identity or curve.also_on):
            _fail('material boundary lacks exact unchanged native pcurve authority')
        pc = curve.pcurve
        if (type(pc) is not NurbsCurve or pc.degree != 1 or len(pc.controls) != 2
                or len(pc.knots) != 4 or pc.knots[0] != pc.knots[1]
                or pc.knots[2] != pc.knots[3] or pc.knots[1] >= pc.knots[2]
                or set(map(Q, pc.domain)) != {Q(pc.knots[1]), Q(pc.knots[2])}):
            _fail('material rectangle needs complete linear pcurve sides')
        a, b = pc.exact_point(0), pc.exact_point(1)
        sides.append((a, b))
    corners = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))
    expected = {frozenset((corners[i], corners[(i+1)%4])) for i in range(4)}
    if {frozenset(s) for s in sides} != expected or any(sides[i][1] != sides[(i+1)%4][0] for i in range(4)):
        _fail('actual source pcurve loop does not cover the full rectangle exactly once')
    return tuple(tuple(sorted(map(Q, r))) for r in (support.u_range, support.v_range))


def _c1(surface):
    for p, knots, active in ((surface.degree_u, surface.knots_u, surface.u_range),
                              (surface.degree_v, surface.knots_v, surface.v_range)):
        lo, hi = sorted(map(Q, active))
        if p < 1 or any(n > p-1 for k, n in Counter(map(Q, knots)).items() if lo < k < hi):
            _fail('contact support has unresolved C1 knot regularity')


def _outside(full, outer):
    (a,b),(c,d) = full
    (u,v),(w,x) = outer
    return tuple(box for box in (((a,u),(c,d)),((v,b),(c,d)),((u,v),(c,w)),((u,v),(x,d)))
                 if all(lo < hi for lo,hi in box))


def _graph(field, outer, axis, take):
    """Exact complete knot-piece derivative and opposing-side margins."""
    cuts = [sorted({*outer[k], *(Q(t) for t in knots if outer[k][0] < Q(t) < outer[k][1])})
            for k,knots in enumerate((field.knots_u,field.knots_v))]
    derivative = []; sides = [[], []]
    net = field.exact_net()
    for i in range(len(cuts[0])-1):
        for j in range(len(cuts[1])-1):
            take()
            box = ((cuts[0][i],cuts[0][i+1]),(cuts[1][j],cuts[1][j+1]))
            grid = bezier_piece(field,net,*box)
            d = grid_ranges(derivative_grid(grid,axis))[2]
            derivative.append((d.lo/(box[axis][1]-box[axis][0]), d.hi/(box[axis][1]-box[axis][0])))
            for end in (0,1):
                if box[axis][end] == outer[axis][end]:
                    local = [(Q(0),Q(1)),(Q(0),Q(1))]
                    local[axis] = (Q(end),Q(end))
                    value = grid_ranges(subbox(grid,*local[0],*local[1]))[2]
                    sides[end].append((value.lo,value.hi))
    sign = 1 if min(v[0] for v in derivative)>0 else -1 if max(v[1] for v in derivative)<0 else 0
    if not sign: return None
    left = max(v[1] for v in sides[0]) if sign==1 else -min(v[0] for v in sides[0])
    right = min(v[0] for v in sides[1]) if sign==1 else -max(v[1] for v in sides[1])
    if left >= 0 or right <= 0: return None
    return (min(v[0] for v in derivative) if sign==1 else -max(v[1] for v in derivative), min(-left,right))


def validate_local_contact_stars(source, result, *, cancellation_check=None):
    from .local_sweep_plane_contact import SweepPlaneContactStars,validate_sweep_plane_contact_stars
    if type(result) is SweepPlaneContactStars:
        return validate_sweep_plane_contact_stars(source,result,cancellation_check=cancellation_check)
    row = _ISSUED.get(result) if type(result) is LocalContactStars else None
    if row is None or row[0]() is not source: _fail('contact receipt was not issued for this source')
    def guard():
        if _state(source)!=row[1] or definition_checksum(result)!=row[2]: _fail('contact source or receipt changed')
    guard()
    if cancellation_check is not None and cancellation_check('local contact binding'): _fail('contact cancelled')
    guard()
    validate_material_surface_regions_binding(source,row[3],cancellation_check=None)
    guard()


def query_local_contact_stars(source, face_pair, *, connect_intent=False, policy=None, cancellation_check=None, allow_disjoint=False, _numerical_boundary=None):
    """Freshly qualify a finite atlas; explicit intent permits proof, not exemption.

    A returned source receipt alone cannot be consumed as an exclusion: actual
    prepared realization is mandatory in local_contact_transport.
    """
    if connect_intent is not True: _fail('explicit selected CONNECT intent is required')
    initial = _state(source)
    from ._detached_geometry_read import detach_numeric, NumericalBoundary
    if _numerical_boundary is not None:
        if type(_numerical_boundary) is not NumericalBoundary or _numerical_boundary.source is not source:
            _fail('unknown contact numerical boundary')
        _numerical_boundary.guard()
    faces = tuple(face_pair)
    if len(faces)!=2 or len(set(faces))!=2 or any(type(f) is not int or f not in source.faces for f in faces):
        _fail('contact needs two distinct selected source face IDs')
    policy = policy or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    cap, depth = policy.max_boxes_per_pair, depth_limit(policy.max_subdivision_depth)
    if type(cap) is not int or cap<1 or type(depth) is not int or depth<1: _fail('invalid contact budget')
    effective=SimpleNamespace(**{f.name:getattr(policy,f.name) for f in fields(policy)})
    effective.max_subdivision_depth=depth
    signature = definition_checksum(policy)
    count = 0
    def take():
        nonlocal count
        if definition_checksum(policy)!=signature: _fail('contact policy changed')
        if _numerical_boundary is None and _state(source)!=initial: _fail('contact operands changed')
        count += 1
        if count>cap: _fail('contact box budget exhausted')
        charge_box()
        if cancellation_check is not None and cancellation_check('local contact qualification'): _fail('contact cancelled')
        if definition_checksum(policy)!=signature: _fail('contact policy changed during callback')
        if _numerical_boundary is None and _state(source)!=initial: _fail('contact operands changed during callback')
        return False
    take()
    supports = tuple(source.faces[f].surface for f in faces)
    if any(type(s) is not NurbsSurface for s in supports): _fail('contact requires native polynomial NURBS supports')
    regions = query_material_surface_regions(source,faces,cancellation_check=lambda *args:take())
    charts = {c.face.id:c for r in regions.regions for c in r.sources}
    if set(charts)!=set(faces): _fail('contact source material selection incomplete')
    full = tuple(_rectangle(charts[f].domain,s) for f,s in zip(faces,supports))
    supports = tuple(detach_numeric(s) for s in supports)
    tolerance = detach_numeric(source.tolerance)
    if _state(source)!=initial: _fail('contact source changed while detaching operands')
    for s in supports: _c1(s)
    result = query_nurbs_surface_intersection(*supports,tolerances=tolerance,qualification=effective,cancellation_check=take)
    if allow_disjoint is True and result.status is IntersectionStatus.DISJOINT and result.cover_complete:
        take()
        validate_material_surface_regions_binding(source,regions,cancellation_check=None)
        if _state(source)!=initial: _fail('disjoint contact source changed before return')
        return None
    if result.status is not IntersectionStatus.CERTIFIED or not result.cover_complete or type(result.structural) is not StructuralCertificate:
        _fail('contact needs complete polynomial common-projection component coverage')
    cert = result.structural
    field = ScalarField(result.first_source,cert)
    common = cert.domain
    factor = cert.determinant**2/(sum(x*x for row in cert.affine for x in row[1:])*sum(x*x for row in cert.rows for x in row))
    owners = tuple(NurbsIntersectionCurve.from_result(result,i) for i in range(len(result.components)))
    stars=[]
    with intersection_curve_operation(take):
        for ci,owner in enumerate(owners):
            curve = NurbsSurfaceIntersectionCurve(owner,
                tolerance_values=tuple((f.name,getattr(tolerance,f.name)) for f in fields(tolerance)),
                qualification_values=tuple(vars(effective).items()))
            n = len(owner.segments)
            queue = [(Q(i,n),Q(i+1,n),0) for i in reversed(range(n))]
            while queue:
                a,b,level = queue.pop(); take()
                uv = curve.chart_box(a,b)
                native = tuple(tuple(sorted(Q(r[0])+(Q(r[1])-Q(r[0]))*x for x in bounds))
                    for r,bounds in zip((supports[0].u_range,supports[0].v_range),uv))
                made = None
                # Proposals only; all sign/coverage authority is independently exact.
                for axis in (0,1):
                    for expansion in (2,4,8):
                        widths = tuple(max(hi-lo,(common[k][1]-common[k][0])*Q(1,2**(level+8))) for k,(lo,hi) in enumerate(native))
                        inner = tuple((lo-widths[k]/8,hi+widths[k]/8) for k,(lo,hi) in enumerate(native))
                        outer = tuple((lo-widths[k]*(expansion if k==axis else Q(1,4)),hi+widths[k]*(expansion if k==axis else Q(1,4))) for k,(lo,hi) in enumerate(native))
                        if any(not common[k][0]<lo<hi<common[k][1] for k,(lo,hi) in enumerate(outer)): continue
                        margins = _graph(field,outer,axis,take)
                        if margins is None: continue
                        delta = min(min(i[0]-o[0],o[1]-i[1]) for i,o in zip(inner,outer))
                        made = ContactStar(ci,(a,b),inner,outer,axis,*margins,factor*delta**2,tuple(_outside(f,outer) for f in full))
                        break
                    if made is not None: break
                if made is None:
                    if level>=depth: _fail('finite regular material contact atlas unresolved')
                    charge_split(); mid=(a+b)/2
                    queue.extend(((mid,b,level+1),(a,mid,level+1)))
                else: stars.append(made)
        # Every queued interval, including all endpoints/seams, was covered.
        take()
    validate_material_surface_regions_binding(source,regions,cancellation_check=None)
    take()
    validate_material_surface_regions_binding(source,regions,cancellation_check=None)
    receipt=LocalContactStars(source.model_id,faces,initial[2],'CONNECT','shared_native_knots',owners,tuple(stars),count)
    if _state(source)!=initial: _fail('contact source changed before issuance')
    _ISSUED[receipt]=(ref(source),initial,definition_checksum(receipt),regions)
    return receipt






@dataclass(frozen=True, slots=True)
class ConditionalContactProposal:
    """Source planning data ONLY; actual prepared realization is still owed."""
    source_face: int
    partner_face: int
    source_native_box: tuple
    excluded_partner_native_box: tuple
    retained_partner_native_rectangles: tuple
    exterior_squared_lower: Q
    component: int
    requires_realization: bool = True


def query_local_contact_proposal(source,stars,source_face,native_box,*,cancellation_check=None):
    validate_local_contact_stars(source,stars,cancellation_check=cancellation_check)
    box=tuple(tuple(map(Q,p)) for p in native_box)
    if len(box)!=2 or any(len(p)!=2 or p[0]>p[1] for p in box): _fail('invalid conditional contact footprint')
    if type(source_face) is not int or source_face not in stars.face_ids: _fail('conditional contact face outside issued scope')
    partner=1-stars.face_ids.index(source_face)
    for star in stars.stars:
        if all(b[0]<=p[0]<=p[1]<=b[1] for b,p in zip(star.inner_box,box)):
            proposal=ConditionalContactProposal(source_face,stars.face_ids[partner],box,star.outer_box,
                star.exterior_rectangles[partner],star.exterior_squared_lower,star.component)
            validate_local_contact_stars(source,stars)
            return proposal
    validate_local_contact_stars(source,stars)
    return None


def local_contact_fingerprint(source,stars):
    validate_local_contact_stars(source,stars)
    return dict(version=1,intent=stars.intent,face_ids=list(stars.face_ids),
        source_model_id=str(source.model_id),source_checksum=stars.source_checksum,
        definition_checksum=definition_checksum(stars))



@dataclass(frozen=True, slots=True)
class _ContactProposalView:
    """Detached numerical data; enclosing operation owes callback/final guards."""
    face_ids: tuple
    rows: tuple

    def query(self, face, box):
        if type(face) is not int or face not in self.face_ids:
            _fail('internal contact face is not canonical')
        if (type(box) is not tuple or len(box)!=2 or
                any(type(p) is not tuple or len(p)!=2 or
                    any(type(x) is not Q for x in p) or p[0]>p[1] for p in box)):
            _fail('internal contact box is not canonical')
        partner=1-self.face_ids.index(face)
        for inner,outer,exterior,lower,component in self.rows:
            if all(b[0]<=p[0]<=p[1]<=b[1] for b,p in zip(inner,box)):
                return ConditionalContactProposal(face,self.face_ids[partner],box,outer,
                    exterior[partner],lower,component)
        return None


def _prepare_local_contact_proposal_query(source, stars):
    _check_contact_view_context()
    validate_local_contact_stars(source,stars)
    def scalar(x):
        if type(x) is not Q: _fail('contact view requires exact canonical fractions')
        return Q(x.numerator,x.denominator)
    def box(value):return tuple(tuple(scalar(x) for x in p) for p in value)
    faces=tuple(stars.face_ids)
    if len(faces)!=2 or any(type(f) is not int for f in faces):_fail('invalid contact view faces')
    rows=[]
    for star in stars.stars:
        if type(star.component) is not int:_fail('invalid contact component')
        rows.append((box(star.inner_box),box(star.outer_box),
            tuple(tuple(box(rectangle) for rectangle in side) for side in star.exterior_rectangles),
            scalar(star.exterior_squared_lower),star.component))
    result=_ContactProposalView(faces,tuple(rows))
    _check_contact_view_context()
    validate_local_contact_stars(source,stars)
    return result


class _ContactProposalRegistry:
    """One unpublished field operation; ALL consulted contacts, including cuts."""
    __slots__=("source","receipts","_runtime_check","sweep_operations","graph_operations")
    def __init__(self, source):
        self._runtime_check=_check_contact_view_context
        self._runtime_check()
        self.source=source
        self.receipts=[]
        self.sweep_operations=[]
        self.graph_operations=[]
    def guard(self):
        self._runtime_check()
        for operation in self.sweep_operations:operation.guard()
        for operation in self.graph_operations:operation.guard()
        for receipt in self.receipts:
            validate_local_contact_stars(self.source,receipt)
    def prepare(self, receipt):
        self.guard()
        view=_prepare_local_contact_proposal_query(self.source,receipt)
        if not any(receipt is old for old in self.receipts):self.receipts.append(receipt)
        self.guard()
        return view
    def register_sweep(self,operation):
        from .local_sweep_plane_contact import _SweepCompetition
        if type(operation) is not _SweepCompetition:_fail('unknown sweep competition operation')
        operation.guard();self.sweep_operations.append(operation)
    def register_graph(self,operation):
        from ._local_graph_clearance import GraphClearance
        if type(operation) is not GraphClearance:_fail('unknown graph clearance operation')
        operation.guard();self.graph_operations.append(operation)
    def add_sweep_receipt(self,receipt):
        validate_local_contact_stars(self.source,receipt)
        self.receipts.append(receipt)
    def wrap_callback(self, callback):
        if callback is None:return None
        guard=self.guard  # Retain original methods across caller code.
        runtime_check=self._runtime_check
        def guarded(*args,**kwargs):
            runtime_check();guard()
            value=callback(*args,**kwargs)
            runtime_check();guard()
            return value
        return guarded


def _contact_view_context():
    from ._local_graph_clearance import GraphClearance,runtime_signature as graph_signature
    classes=(_ContactProposalView,_ContactProposalRegistry,ConditionalContactProposal,Q)
    functions=(graph_signature,GraphClearance.__init__,GraphClearance.guard,GraphClearance.query,
        _prepare_local_contact_proposal_query,_ContactProposalView.query,
        _ContactProposalView.__init__,ConditionalContactProposal.__init__,
        _ContactProposalRegistry.__init__,_ContactProposalRegistry.guard,
        _ContactProposalRegistry.prepare,_ContactProposalRegistry.register_sweep,_ContactProposalRegistry.register_graph,
        _ContactProposalRegistry.add_sweep_receipt,_ContactProposalRegistry.wrap_callback,
        _contact_view_context,_check_contact_view_context)
    return (classes,tuple(c.__getattribute__ for c in classes),
        tuple((f,getattr(f,'__code__',None),definition_checksum(getattr(f,'__defaults__',None)),
            definition_checksum(getattr(f,'__kwdefaults__',None))) for f in functions))


def _check_contact_view_context():
    try: current=_contact_view_context()
    except (AttributeError,TypeError):_fail('contact proposal runtime changed')
    if current!=_CONTACT_VIEW_CONTEXT:
        _fail('contact proposal runtime changed')


_CONTACT_VIEW_CONTEXT=_contact_view_context()
