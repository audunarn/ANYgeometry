"""Independent exact arithmetic for CH3 proof verification.

No producer arithmetic is imported. Coefficients are ascending Fractions. Root
isolation uses a Sturm chain; integration uses left endpoint Taylor polynomials
and an absolute Bernstein bound for the exact rational residual.
"""
from fractions import Fraction as F
from math import comb, factorial, gcd, lcm,isqrt
from contextvars import ContextVar
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
_active=ContextVar('ch3_verify_arithmetic',default=None)
def audit(value=None):
    state=_active.get()
    if state is not None:
        meter,bits=state;meter.unit('verify_fraction')
        if value is not None:
            meter.fit(value)
            if max(abs(value.numerator).bit_length(),value.denominator.bit_length())>bits:fail('verifier arithmetic bit cap exceeded','BIT_LENGTH_EXCEEDED')
    return value

def trim(p):
    p = list(map(F,p))
    for x in p:audit(x)
    while p and not p[-1]: p.pop()
    return tuple(p)
def add(a,b): return trim([(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0) for i in range(max(len(a),len(b)))])
def neg(a): return tuple(-x for x in a)
def sub(a,b): return add(a,neg(b))
def scale(a,b): return trim([x*b for x in a])
def mul(a,b):
    c=[F(0)]*max(0,len(a)+len(b)-1)
    for i,x in enumerate(a):
        audit()
        for j,y in enumerate(b): c[i+j]=audit(c[i+j]+audit(x*y))
    return trim(c)
def power(a,n):
    r=(F(1),)
    while n:
        if n&1:r=mul(r,a)
        a=mul(a,a); n//=2
    return r
def derivative(a):return trim([i*a[i] for i in range(1,len(a))])
def evaluate(a,x):
    y=F(0)
    for c in reversed(a):y=audit(audit(y*x)+c)
    return y
def divide(a,b):
    if not b:raise ZeroDivisionError
    r=list(a); q=[F(0)]*max(0,len(a)-len(b)+1)
    while len(r)>=len(b):
        audit()
        i=len(r)-len(b);t=r[-1]/b[-1];q[i]=t
        for j,c in enumerate(b):r[i+j]-=t*c
        while r and not r[-1]:r.pop()
    return trim(q),tuple(r)
def pgcd(a,b):
    # Primitive pseudo-remainders prevent rational Euclidean coefficient
    # swell. Every scaling is nonzero, so the gcd over Q is preserved.
    a=primitive(a);b=primitive(b)
    while b:
        r=a
        while r and len(r)>=len(b):
            audit();shift=len(r)-len(b);lead=r[-1];base=b[-1]
            rr=[audit(base*x) for x in r]
            for i,x in enumerate(b):rr[i+shift]=audit(rr[i+shift]-audit(lead*x))
            r=primitive(trim(rr))
        a,b=b,r
    return scale(a,F(1,a[-1])) if a else ()
def primitive(a):
    if not a:return ()
    den=lcm(*(x.denominator for x in a));v=[int(x*den) for x in a];g=gcd(*v)
    if v[-1]<0:g=-g
    return tuple(x//g for x in v)
def squarefree(a):return divide(a,pgcd(a,derivative(a)))[0] if len(a)>1 else a
def bernstein(a,lo,hi):
    n=len(a)-1
    if n<0:return (F(0),)
    c=[audit(sum((audit(audit(audit(a[j]*comb(j,i))*audit(lo**(j-i)))*audit((hi-lo)**i)) for j in range(i,n+1)),F(0))) for i in range(n+1)]
    return tuple(audit(sum((audit(c[i]*F(comb(k,i),comb(n,i))) for i in range(k+1)),F(0))) for k in range(n+1))
def bounds(a,lo,hi):
    b=bernstein(a,lo,hi);return min(b),max(b)
def sign(x):return (x>0)-(x<0)
def fail(message,category='VERIFICATION_FAILED',sub_reason='enclosure'):
    raise CertifiedHydrostaticsRefusal(getattr(RefusalCategory,category),message,phase='verify',sub_reason=sub_reason if category=='VERIFICATION_FAILED' else None)

class Arithmetic:
    def __init__(self,meter,policy):
        self.meter=meter;self.policy=policy;self.root_cache={};self.integrals=0;self.pieces=0
        _active.set((meter,policy['max_bits']))
    def tick(self,kind='verify_polynomial',n=1):self.meter.unit(kind,n)
    def fit(self,p):
        for x in p:audit(F(x))
        return p
    def chain(self,p):
        p=trim(squarefree(p));s=[p,derivative(p)]
        while s[-1]:
            self.tick();r=neg(divide(s[-2],s[-1])[1])
            if not r:break
            s.append(scale(r,1/abs(r[-1])))
        return tuple(s)
    def variations(self,s,x,side):
        values=[]
        for p in s:
            k=0;v=evaluate(p,x)
            while not v and p:p=derivative(p);k+=1;v=evaluate(p,x)
            if v:values.append(sign(v)*((-1)**k if side<0 else 1))
        return sum(a!=b for a,b in zip(values,values[1:]))
    def roots(self,p,lo,hi):
        p=trim(squarefree(p));key=(p,F(lo),F(hi))
        if key in self.root_cache:return self.root_cache[key]
        if len(p)<2:return []
        s=self.chain(p);out=[]
        rational_roots=None
        if len(p)==2:rational_roots=[-p[0]/p[1]]
        elif len(p)==3:
            disc=p[1]*p[1]-4*p[0]*p[2]
            if disc>=0:
                an,ad=isqrt(disc.numerator),isqrt(disc.denominator)
                if an*an==disc.numerator and ad*ad==disc.denominator:
                    sd=F(an,ad);rational_roots=sorted(set(((-p[1]-sd)/(2*p[2]),(-p[1]+sd)/(2*p[2]))))
        if rational_roots is not None:
            out=[Root(self,p,r,r,s) for r in rational_roots if lo<=r<=hi];self.root_cache[key]=out;return out
        def walk(a,b,depth=0):
            self.tick('verify_isolation')
            count=self.variations(s,a,1)-self.variations(s,b,-1)
            if count==0:return
            if depth>self.policy['max_isolation_bits']:fail('root isolation budget exhausted','BUDGET_EXHAUSTED')
            if count==1:out.append(Root(self,p,a,b,s));return
            c=(a+b)/2
            walk(a,c,depth+1)
            if evaluate(p,c)==0:out.append(Root(self,p,c,c,s))
            walk(c,b,depth+1)
        if evaluate(p,lo)==0:out.append(Root(self,p,F(lo),F(lo),s))
        walk(F(lo),F(hi))
        if evaluate(p,hi)==0 and hi!=lo:out.append(Root(self,p,F(hi),F(hi),s))
        # Sturm counts above concern OPEN intervals. Consumers use CLOSED
        # isolating bounds, so remove any other root retained at an endpoint
        # before equality, sign and all-real-ordinal predicates can see it.
        for root in out:
            while root.lo!=root.hi and (evaluate(p,root.lo)==0 or evaluate(p,root.hi)==0):root.bisect()
        self.root_cache[key]=out
        return out
    def integrate(self,r,a,b,tolerance):
        self.integrals+=1
        if compare(a,b)==0:return Interval(0)
        if compare(a,b)>0:return -self.integrate(r,b,a,tolerance)
        tolerance=F(tolerance)
        if tolerance<=0:fail('positive verifier enclosure allowance required','BUDGET_EXHAUSTED')
        for endpoint in (a,b):
            if isinstance(endpoint,Root):endpoint.refine(100)
        def ends():return (a.lo,a.hi) if isinstance(a,Root) else (F(a),F(a)),(b.lo,b.hi) if isinstance(b,Root) else (F(b),F(b))
        for k in range(self.policy['max_isolation_bits']):
            (al,ah),(bl,bh)=ends()
            limit=self.range(r,al,bh)
            if limit is not None and ah<=bl:
                error=(ah-al+bh-bl)*max(abs(limit.lo),abs(limit.hi))
                if 2*error<=tolerance/2:break
            for endpoint in (a,b):
                if isinstance(endpoint,Root):endpoint.bisect()
        else:fail('endpoint refinement exhausted','BUDGET_EXHAUSTED')
        answer=self._piece(r,ah,bl,tolerance-2*error,0)
        return answer+Interval(-error,error)
    def range(self,r,a,b):
        nl,nh=bounds(r.n,a,b);dl,dh=bounds(r.d,a,b)
        if dl<=0<=dh:return None
        return Interval(nl,nh)/Interval(dl,dh)
    def _piece(self,r,a,b,tol,depth):
        self.tick('verify_taylor_piece');self.pieces+=1
        if self.pieces>self.policy['max_pieces_per_integral']*max(1,self.integrals):fail('verifier Taylor subdivision budget exhausted','BUDGET_EXHAUSTED')
        if a==b:return Interval(0)
        q,rem=divide(r.n,r.d)
        if not rem:
            return Interval(sum((x*(b**(i+1)-a**(i+1))/(i+1) for i,x in enumerate(q)),F(0)))
        dl,dh=bounds(r.d,a,b);minimum=min(abs(dl),abs(dh)) if dl*dh>0 else 0
        if minimum:
            # In t=(u-a)/(b-a), construct the actual left Taylor quotient.
            # The exact residual identity f-T=(N-D*T)/D supplies the
            # remainder, bounded with Bernstein coefficients on [0,1].
            # This avoids the producer's derivative/midpoint recurrence.
            h=b-a
            def local(p):return trim([sum((p[j]*comb(j,i)*a**(j-i)*h**i for j in range(i,len(p))),F(0)) for i in range(len(p))])
            nn,dd=local(r.n),local(r.d);coeff=[]
            for j in range(self.policy['max_taylor_order']+1):
                self.tick('verify_taylor_coefficient')
                v=((nn[j] if j<len(nn) else 0)-sum((dd[k]*coeff[j-k] for k in range(1,min(j,len(dd)-1)+1)),F(0)))/dd[0]
                self.meter.fit(v);coeff.append(v)
                if j in (3,7,self.policy['max_taylor_order']):
                    residual=sub(nn,mul(dd,tuple(coeff)));lo,hi=bounds(residual,F(0),F(1))
                    rho=h*max(abs(lo),abs(hi))/minimum
                    if 2*rho<=tol:
                        main=h*sum((c/F(i+1) for i,c in enumerate(coeff)),F(0))
                        return Interval(main-rho,main+rho)
        if depth>256:fail('verifier refinement exhausted','BUDGET_EXHAUSTED')
        c=(a+b)/2
        return self._piece(r,a,c,tol/2,depth+1)+self._piece(r,c,b,tol/2,depth+1)

class Root:
    def __init__(self,ar,p,lo,hi,chain):self.ar=ar;self.p=p;self.lo=lo;self.hi=hi;self.chain=chain;self.bits=0
    def bisect(self):
        self.ar.tick('verify_root_refine');self.bits+=1
        if self.bits>self.ar.policy['max_isolation_bits']:fail('root refinement exhausted','BUDGET_EXHAUSTED')
        if self.lo==self.hi:return
        c=(self.lo+self.hi)/2
        if evaluate(self.p,c)==0:self.lo=self.hi=c;return
        count=self.ar.variations(self.chain,self.lo,1)-self.ar.variations(self.chain,c,-1)
        if count:self.hi=c
        else:self.lo=c
    def refine(self,bits):
        while self.hi-self.lo>F(1,2**bits):self.bisect()
    def identity(self):
        p=primitive(self.p);bound=1+max((abs(x/p[-1]) for x in p[:-1]),default=0);bound=F(int(bound)+1)
        ordinal=self.ar.variations(self.chain,-bound,1)-self.ar.variations(self.chain,self.lo,-1)
        return {'poly':list(p),'ordinal':ordinal}

def compare(a,b):
    if a is b:return 0
    if not isinstance(a,Root) and not isinstance(b,Root):return sign(a-b)
    for _ in range(4096):
        al,ah=(a.lo,a.hi) if isinstance(a,Root) else (a,a);bl,bh=(b.lo,b.hi) if isinstance(b,Root) else (b,b)
        if ah<bl:return -1
        if bh<al:return 1
        if al==ah==bl==bh:return 0
        if isinstance(a,Root) and isinstance(b,Root):
            g=pgcd(a.p,b.p)
            if len(g)>1 and a.ar.roots(g,max(al,bl),min(ah,bh)):return 0
        elif isinstance(a,Root) and evaluate(a.p,F(b))==0:return 0
        elif isinstance(b,Root) and evaluate(b.p,F(a))==0:return 0
        if isinstance(a,Root):a.bisect()
        if isinstance(b,Root):b.bisect()
    fail('algebraic comparison exhausted','BUDGET_EXHAUSTED')

class Rational:
    def __init__(self,n=0,d=(F(1),)):
        if isinstance(n,Rational):self.n,self.d=n.n,n.d;return
        n=trim(n if isinstance(n,(tuple,list)) else (n,));d=trim(d)
        if not d:raise ZeroDivisionError
        g=pgcd(n,d);n=divide(n,g)[0] if g else ();d=divide(d,g)[0] if g else (F(1),)
        self.n,self.d=scale(n,1/d[-1]),scale(d,1/d[-1])
    def __add__(self,b):
        b=Rational(b);return Rational(add(mul(self.n,b.d),mul(b.n,self.d)),mul(self.d,b.d))
    __radd__=__add__
    def __neg__(self):return Rational(neg(self.n),self.d)
    def __sub__(self,b):return self+-Rational(b)
    def __rsub__(self,b):return Rational(b)+-self
    def __mul__(self,b):
        b=Rational(b);return Rational(mul(self.n,b.n),mul(self.d,b.d))
    __rmul__=__mul__
    def __truediv__(self,b):
        b=Rational(b);return Rational(mul(self.n,b.d),mul(self.d,b.n))
    def __pow__(self,n):return Rational(power(self.n,n),power(self.d,n))
    def derivative(self):return Rational(sub(mul(derivative(self.n),self.d),mul(self.n,derivative(self.d))),mul(self.d,self.d))
    def at(self,u):return evaluate(self.n,u)/evaluate(self.d,u)

class Interval:
    def __init__(self,lo,hi=None):self.lo=audit(F(lo));self.hi=audit(F(lo if hi is None else hi))
    def __add__(self,b):
        b=b if isinstance(b,Interval) else Interval(b);return Interval(self.lo+b.lo,self.hi+b.hi)
    __radd__=__add__
    def __neg__(self):return Interval(-self.hi,-self.lo)
    def __sub__(self,b):return self+-b
    def __rsub__(self,b):return Interval(b)+-self
    def __mul__(self,b):
        b=b if isinstance(b,Interval) else Interval(b);v=[a*c for a in (self.lo,self.hi) for c in (b.lo,b.hi)];return Interval(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,b):
        b=b if isinstance(b,Interval) else Interval(b)
        if b.lo<=0<=b.hi:raise ZeroDivisionError
        return self*Interval(1/b.hi,1/b.lo)
    @property
    def width(self):return self.hi-self.lo

def dot(a,b):return sum((x*y for x,y in zip(a,b)),0)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def vsub(a,b):return tuple(x-y for x,y in zip(a,b))
