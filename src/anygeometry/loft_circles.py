"""Ruled circle-profile lofts preserving authored circular Arc geometry.

Circle recognition is tolerance bounded. Admission uses the complete circular
disk homotopy, not sampled face normals or a tessellated self-intersection test.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
import math
import numpy as np

from .curves import Arc, arc_frame, sample_arc
from .entities import OrientedEdge
from .errors import GeometryError
from .surfaces import CoonsSurface


def _norm(values):
    values=tuple(Q(float(x)) if not isinstance(x,Q) else x for x in values)
    scale=max(map(abs,values),default=Q(0))
    if not scale:return Q(0)
    square=sum((x/scale)**2 for x in values)
    result=math.sqrt(float(square))
    while Q(result)*Q(result)<square: result=math.nextafter(result,math.inf)
    return Q(result)*scale


def _inverse(matrix):
    rows=[list(row)+[Q(int(i==j)) for j in range(3)] for i,row in enumerate(matrix)]
    for j in range(3):
        pivot=next((i for i in range(j,3) if rows[i][j]),None)
        if pivot is None: raise GeometryError('circle loft midpoint frame is singular; change direction or seam')
        rows[j],rows[pivot]=rows[pivot],rows[j]
        value=rows[j][j];rows[j]=[x/value for x in rows[j]]
        for i in range(3):
            if i!=j:
                value=rows[i][j];rows[i]=[x-value*y for x,y in zip(rows[i],rows[j])]
    return [row[3:] for row in rows]


@dataclass
class CircleProfile:
    points: list
    frames: list
    breaks: np.ndarray
    center: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    value_error: Q
    derivative_error: Q
    closed: bool
    span: float


def _trig(angle):
    """Rational Taylor centers and rigorous absolute Lagrange remainders."""
    x=Q(angle)
    sine=sum((-1)**k*x**(2*k+1)/math.factorial(2*k+1) for k in range(32))
    cosine=sum((-1)**k*x**(2*k)/math.factorial(2*k) for k in range(33))
    return sine,cosine,abs(x)**65/math.factorial(65),abs(x)**66/math.factorial(66)


def _errors(frames,points,breaks,center,cosine,sine,total_span=math.tau):
    """Bound complete arcs, their angular derivatives and Coons endpoint correction.

    All stored coefficients are interpreted exactly. Trigonometric coefficient
    rotations use rational Taylor enclosures; no libm accuracy assumption enters
    the proof. The arc evaluators retain their ordinary owner float semantics.
    """
    vector=lambda row: tuple(x if isinstance(x,Q) else Q(float(x)) for x in row)
    base_u,base_v,base_c=map(vector,(cosine,sine,center))
    value_error=Q(0);derivative_error=Q(0)
    for frame,ends,a,b in zip(frames,points,breaks[:-1],breaks[1:]):
        theta=Q(total_span)*Q(float(a));span=Q(total_span)*(Q(float(b))-Q(float(a)))
        radius=Q(float(frame.radius));sign=1 if frame.sweep>=0 else -1
        u=tuple(radius*x for x in vector(frame.e1))
        v=tuple(radius*sign*x for x in vector(frame.e2))
        sn,cs,se,ce=_trig(theta)
        expected_u=tuple(x*cs+y*sn for x,y in zip(base_u,base_v))
        expected_v=tuple(-x*sn+y*cs for x,y in zip(base_u,base_v))
        eu_error=_norm(base_u)*ce+_norm(base_v)*se
        ev_error=_norm(base_u)*se+_norm(base_v)*ce
        omega=abs(Q(float(frame.sweep)));phase=abs(omega-span);rate=omega/span
        coefficient=(_norm(x-y for x,y in zip(u,expected_u))+eu_error+
                     _norm(x-y for x,y in zip(v,expected_v))+ev_error)
        size=_norm(expected_u)+eu_error+_norm(expected_v)+ev_error
        fc=vector(frame.center)
        start=_norm(x-y-z for x,y,z in zip(vector(ends[0]),fc,u))
        sn,cs,se,ce=_trig(omega)
        end=(_norm(x-y-z*cs-w*sn for x,y,z,w in zip(vector(ends[2]),fc,u,v))+
             _norm(u)*ce+_norm(v)*se)
        ev=_norm(x-y for x,y in zip(fc,base_c))+coefficient+size*phase+max(start,end)
        ed=rate*coefficient+size*(abs(rate-1)+phase)+(start+end)/span
        value_error=max(value_error,ev);derivative_error=max(derivative_error,ed)
    return value_error,derivative_error


def circle_profile(model,identifiers,reverse=False,start=0):
    from .loft import _ordered_profile_edges,LoftNotCertified
    ordered,closed=_ordered_profile_edges(model,identifiers,reverse,start)
    if not all(isinstance(edge.curve,Arc) for edge,_ in ordered):
        raise GeometryError('circular loft needs two circular Arc profiles')
    points=[];frames=[]
    for edge,forward in ordered:
        ids=(edge.start,edge.curve.via_vertex,edge.end)
        if not forward: ids=ids[::-1]
        row=np.asarray([model.vertex_position(i) for i in ids]);points.append(row)
        frames.append(arc_frame(*row))
    lengths=np.asarray([frame.length for frame in frames])
    breaks=np.r_[0.,np.cumsum(lengths)/sum(lengths)];breaks[-1]=1.
    frame=frames[0];center=frame.center;cosine=frame.radius*frame.e1
    sine=frame.radius*frame.e2*np.sign(frame.sweep)
    tolerance=model.tolerance.effective_curve_fit_residual(2*frame.radius)
    total_span=sum(abs(f.sweep) for f in frames)
    if closed and abs(total_span-math.tau)>model.tolerance.angular*len(frames):
        raise GeometryError('Arc chain does not make one circular turn')
    # Recognition is a bound on whole intervals, not just vertex proximity.
    reference_span=math.tau if closed else total_span
    ev,ed=_errors(frames,points,breaks,center,cosine,sine,reference_span)
    if ev>Q(tolerance):
        raise LoftNotCertified('Arc profile cannot be recognized as one circle within its curve-fit tolerance')
    return CircleProfile(points,frames,breaks,center,cosine,sine,ev,ed,closed,reference_span)


def _at(profile,t):
    i=min(int(np.searchsorted(profile.breaks,t,side='right'))-1,len(profile.frames)-1)
    a,b=profile.breaks[i:i+2]
    if t==a:return profile.points[i][0].copy()
    if t==b:return profile.points[i][2].copy()
    return sample_arc(profile.frames[i],np.array([(t-a)/(b-a)]))[0]


def circle_options(model,first_edges,second_edges):
    first=circle_profile(model,first_edges)
    candidates=[]
    for start in range(len(second_edges) if first.closed else 1):
        for reverse in (False,True):
            second=circle_profile(model,second_edges,reverse,start)
            cost=sum((Q(float(x))-Q(float(y)))**2 for t in np.linspace(0,1,17)
                     for x,y in zip(_at(first,t),_at(second,t)))
            candidates.append((cost,start,reverse))
    _,start,reverse=min(candidates)
    return dict(reverse_first=False,reverse_second=reverse,first_start=0,second_start=start,flip_normal=False)


def _parallel_admission(first,second,uncertainty):
    """For exactly parallel authored planes, v uniquely identifies a section."""
    u0,v0,u1,v1=[tuple(Q(float(x)) for x in row) for row in
                 (first.cosine,first.sine,second.cosine,second.sine)]
    def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    normal=cross(u0,v0)
    def plane(points):return [dot(normal,tuple(Q(float(x)) for x in p)) for row in points for p in row]
    p,q=plane(first.points),plane(second.points)
    if len(set(p))!=1 or len(set(q))!=1 or p[0]==q[0]: return False
    # Authored vertices alone do not establish containment of the actual Arc
    # frame. Require every stored trigonometric coefficient in the same plane;
    # endpoint corrections then also lie in that plane. Otherwise the general
    # homotopy proof must account for vertical uncertainty and spacing.
    for profile,level in ((first,p[0]),(second,q[0])):
        for frame in profile.frames:
            center=tuple(Q(float(x)) for x in frame.center)
            u=tuple(Q(float(frame.radius))*Q(float(x)) for x in frame.e1)
            v=tuple(Q(float(frame.radius))*Q(float(x)) for x in frame.e2)
            if dot(normal,center)!=level or dot(normal,u)!=0 or dot(normal,v)!=0:
                return False
    du,dv=tuple(b-a for a,b in zip(u0,u1)),tuple(b-a for a,b in zip(v0,v1))
    c0=dot(normal,cross(u0,v0));c1=dot(normal,cross(du,v0))+dot(normal,cross(u0,dv));c2=dot(normal,cross(du,dv))
    values=[c0,c0+c1+c2]
    if c2 and 0 < -c1/(2*c2) < 1:
        t=-c1/(2*c2);values.append(c0+c1*t+c2*t*t)
    area=min(values)/_norm(normal)
    if area<=0:return False
    # sigma_min >= area / Frobenius norm of the two-column radial frame.
    radial=max(_norm((*u0,*v0)),_norm((*u1,*v1)))
    return area/radial>uncertainty


def _tau_interval():
    # Machin identity pi=16 atan(1/5)-4 atan(1/239), with alternating
    # rational series and a bound by the first omitted term.
    def atan_inverse(n):
        x=Q(1,n)
        value=sum((-1)**k*x**(2*k+1)/(2*k+1) for k in range(48))
        error=x**97/97
        return value,error
    a,ea=atan_inverse(5);b,eb=atan_inverse(239)
    center=32*a-8*b;error=32*ea+8*eb
    return center-error,center+error


def _periodic_errors(profile):
    """Extend profile error continuously around a mathematical full disk."""
    from .loft import LoftNotCertified
    low,high=_tau_interval();span=Q(profile.span)
    ev,ed=profile.value_error,profile.derivative_error
    if profile.closed:
        delta=max(abs(span-low),abs(span-high))
        ratio=span/low
        ratio_error=max(abs(span/low-1),abs(span/high-1))
        size=_norm(profile.cosine)+_norm(profile.sine)
        ev+=size*delta
        ed=ed*ratio+size*(ratio_error+delta)
    else:
        if span>=low:
            raise LoftNotCertified('open circular loft needs less than one full turn')
        # Linearly bridge the error vector across the remaining angular gap.
        # The bridge stays in the ev ball; its derivative is at most2ev/gap.
        ed=max(ed,2*ev/(low-span))
    return replace(profile,value_error=ev,derivative_error=ed)


def _admit(first,second):
    from .loft import LoftNotCertified
    first,second=_periodic_errors(first),_periodic_errors(second)
    ev=max(first.value_error,second.value_error)
    ed=max(first.derivative_error,second.derivative_error)
    spatial=ev+ed
    if _parallel_admission(first,second,2*spatial):return
    if _normalized_frustum_admission(first,second):return
    u0,v0,u1,v1,c0,c1=[tuple(Q(float(x)) for x in row) for row in
                       (first.cosine,first.sine,second.cosine,second.sine,first.center,second.center)]
    midpoint=[((u0[i]+u1[i])/2,(v0[i]+v1[i])/2,c1[i]-c0[i]) for i in range(3)]
    try:inverse=_inverse(midpoint)
    except GeometryError as error:raise LoftNotCertified(str(error)) from error
    difference=[(u1[i]-u0[i],v1[i]-v0[i]) for i in range(3)]
    mapped=[[sum(inverse[i][k]*difference[k][j] for k in range(3)) for j in range(2)] for i in range(3)]
    sym=(mapped[0][0],(mapped[0][1]+mapped[1][0])/2,(mapped[0][1]+mapped[1][0])/2,mapped[1][1])
    bnorm=_norm((*mapped[0],*mapped[1]));cnorm=_norm(mapped[2])
    alpha=1-_norm(sym)/2;beta=1-cnorm;off=(bnorm+cnorm/2)/2
    perturbation=_norm(x for row in inverse for x in row)*_norm((spatial,spatial,first.value_error+second.value_error))
    # Symmetric Jacobian block lower bound is positive on the complete convex
    # disk x [-1/2,1/2]. Strong monotonicity proves global injectivity.
    if alpha<=perturbation or beta<=perturbation or (alpha-perturbation)*(beta-perturbation)<=off*off:
        raise LoftNotCertified('circle loft can fold or its global injectivity bound is inconclusive; change seam, direction or spacing')


def _normalized_frustum_admission(first,second):
    """Injectivity on a convex radius-normalized frustum, independent of axes.

    Use X = r(v) * x / r_mean (and Y analogously). The domain radius is
    positive affine in v, hence convex. Unit-radius frames blend with
    lambda = v*r1/r(v); |lambda'|*domain_radius <= r_max/r_mean.
    A fixed midpoint inverse turns the reference map into the identity.
    A whole-domain derivative perturbation norm below one proves monotonicity.
    """
    r0,r1=Q(first.frames[0].radius),Q(second.frames[0].radius)
    mean=(r0+r1)/2;maximum=max(r0,r1)
    u0,v0,u1,v1,c0,c1=[tuple(Q(float(x)) for x in row) for row in
                       (first.cosine,first.sine,second.cosine,second.sine,first.center,second.center)]
    midpoint=[(mean*(u0[i]/r0+u1[i]/r1)/2,mean*(v0[i]/r0+v1[i]/r1)/2,c1[i]-c0[i]) for i in range(3)]
    try:inverse=_inverse(midpoint)
    except GeometryError:return False
    difference=[(mean*(u1[i]/r1-u0[i]/r0),mean*(v1[i]/r1-v0[i]/r0)) for i in range(3)]
    mapped=[[sum(inverse[i][k]*difference[k][j] for k in range(3)) for j in range(2)] for i in range(3)]
    reference_bound=_norm(x for row in mapped for x in row)*_norm((Q(1,2),maximum/mean))
    spatial=mean*max((first.value_error+first.derivative_error)/r0,
                     (second.value_error+second.derivative_error)/r1)
    vertical=maximum*(first.value_error/r0+second.value_error/r1)
    uncertainty=_norm(x for row in inverse for x in row)*_norm((spatial,spatial,vertical))
    return reference_bound+uncertainty<1


def _refine_profile(model,profile,anchors):
    from .loft import LoftNotCertified
    points=[];frames=[]
    for a,b in zip(anchors,anchors[1:]):
        i=min(int(np.searchsorted(profile.breaks,(a+b)/2,side='right'))-1,len(profile.frames)-1)
        low,high=profile.breaks[i:i+2]
        def local(t):
            if abs(t-low)<=model.tolerance.parameter:return 0.
            if abs(t-high)<=model.tolerance.parameter:return 1.
            return float((t-low)/(high-low))
        left,right=local(a),local(b)
        if left<0 or right>1 or right<=left:
            raise GeometryError('circle correspondence has an unresolved short Arc interval')
        if left==0 and right==1:
            row=profile.points[i].copy()
        else:
            row=sample_arc(profile.frames[i],np.array([left,(left+right)/2,right]))
            if left==0:row[0]=profile.points[i][0]
            if right==1:row[2]=profile.points[i][2]
        if points:row[0]=points[-1][2]
        if profile.closed and b==1.:row[2]=points[0][0]
        frame=arc_frame(*row)
        # A restriction is still an Arc. Bound its entire curve displacement
        # against the source interval, including refit and parameter rounding.
        old=profile.frames[i]
        angle=Q(float(old.sweep))*Q(left)
        sn,cs,se,ce=_trig(angle)
        radius=Q(float(old.radius));sign=1 if old.sweep>=0 else -1
        u=tuple(radius*Q(float(x)) for x in old.e1)
        v=tuple(radius*Q(float(x)) for x in old.e2)
        cosine=tuple(x*cs+y*sn for x,y in zip(u,v))
        sine=tuple((-x*sn+y*cs)*sign for x,y in zip(u,v))
        coefficient_error=(_norm(u)+_norm(v))*(se+ce)
        exact_span=abs(Q(float(old.sweep)))*(Q(right)-Q(left))
        stored_span=float(exact_span)
        ev,_=_errors([frame],[row],np.array([0.,1.]),old.center,cosine,sine,stored_span)
        ev+=coefficient_error+(_norm(u)+_norm(v))*abs(exact_span-Q(stored_span))
        if ev>Q(model.tolerance.effective_curve_fit_residual(2*old.radius)):
            raise LoftNotCertified('restricted Arc cannot preserve the source within its curve-fit tolerance')
        points.append(row);frames.append(frame)
    ev,ed=_errors(frames,points,np.array(anchors),profile.center,profile.cosine,profile.sine,profile.span)
    return CircleProfile(points,frames,np.array(anchors),profile.center,profile.cosine,profile.sine,
                         ev,ed,profile.closed,profile.span)


def build_circle_loft(model,first_edges,second_edges,*,reverse_first=False,reverse_second=False,
                      first_start=0,second_start=0,flip_normal=False,cancellation_check=None):
    from .loft import _check,_merge_profile_anchors
    _check(cancellation_check)
    first=circle_profile(model,first_edges,reverse_first,first_start)
    second=circle_profile(model,second_edges,reverse_second,second_start)
    if first.closed!=second.closed:raise GeometryError('circular loft profiles must both be open or both closed')
    if abs(first.span-second.span)>model.tolerance.angular*(len(first.frames)+len(second.frames)):
        raise GeometryError('paired Arc profiles need matching angular extent')
    # Both boundaries must use one common angular coordinate in the disk proof.
    # Retain the actual frame sweeps; _refine_profile includes their mismatch in
    # its position and derivative uncertainty against this reference span.
    second.span=first.span
    # Match close correspondence stations under the owner's parameter policy;
    # actual profile geometry is preserved by the whole-interval restriction fit.
    count=(max(1,int(math.ceil((first.span-model.tolerance.angular)/(math.pi/2)))) if first.closed else 1)
    anchors=_merge_profile_anchors(((None,None,set(first.breaks)),(None,None,set(second.breaks))),model.tolerance.parameter)
    for value in np.linspace(0,1,count+1):
        if all(abs(value-old)>model.tolerance.parameter for old in anchors):anchors.append(float(value))
    anchors.sort()
    anchors[0],anchors[-1]=0.,1.
    first=_refine_profile(model,first,anchors);second=_refine_profile(model,second,anchors)
    _check(cancellation_check);_admit(first,second)
    vertices=[[model.add_point(*a[0]),model.add_point(*b[0])] for a,b in zip(first.points,second.points)]
    vertices.append(vertices[0] if first.closed else [model.add_point(*first.points[-1][2]),model.add_point(*second.points[-1][2])])
    _check(cancellation_check)
    connectors=[model.add_line(*pair) for pair in vertices[:-1]]
    connectors.append(connectors[0] if first.closed else model.add_line(*vertices[-1]))
    faces=[]
    for i,(a,b) in enumerate(zip(first.points,second.points)):
        _check(cancellation_check)
        av,bv=model.add_point(*a[1]),model.add_point(*b[1])
        bottom=model.add_arc(vertices[i][0],av,vertices[i+1][0]);top=model.add_arc(vertices[i][1],bv,vertices[i+1][1])
        loop=(OrientedEdge(bottom,True),OrientedEdge(connectors[i+1],True),OrientedEdge(top,False),OrientedEdge(connectors[i],False))
        if flip_normal:
            loop=(OrientedEdge(bottom,False),OrientedEdge(connectors[i],True),OrientedEdge(top,True),OrientedEdge(connectors[i+1],False))
        # Admission proves this topology-corrected Arc-boundary map. A fitted
        # analytic surface is not an interchangeable authoritative map.
        face=model._add_face_from_loop(loop,(0,1,2,3),surface=CoonsSurface())
        faces.append(face)
    _check(cancellation_check)
    return faces
