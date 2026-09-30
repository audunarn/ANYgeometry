"""Rational polynomial elimination for Bezier arrangement curves."""
from fractions import Fraction
from itertools import combinations
import math
import numpy as np
from .analytic_roots import isolate_real_roots, _division
from .cylinder_curve_events import _trim, _add, _scale, _multiply
from .errors import GeometryError


def _powers(curve):
    degree=len(curve.controls)-1
    return tuple(_trim(tuple(sum(Fraction(float(curve.controls[i][coordinate]))
        *((-1)**(k-i)*math.comb(degree,i)*math.comb(degree-i,k-i)) for i in range(k+1))
        for k in range(degree+1))) for coordinate in range(3))


def _equal_image(a,b,tolerance):
    first,second=_powers(a),_powers(b)
    # On [0,1], the sum of absolute power coefficients bounds the complete
    # coordinate difference. No display interpolation certifies coincidence.
    return sum(sum(abs(float(value)) for value in _add(x,_scale(y,-1)))
               for x,y in zip(first,second)) <= tolerance


def _resultant(f,g,check):
    while len(f)>1 and f[-1]==(0,): f=f[:-1]
    while len(g)>1 and g[-1]==(0,): g=g[:-1]
    m,n=len(f)-1,len(g)-1
    if not m or not n:
        polynomial=f[0] if not m else g[0]
        count=n if not m else m
        result=(Fraction(1),)
        for _ in range(count): result=_multiply(result,polynomial)
        return result if count else (Fraction(0),)
    size=m+n
    rows=[]
    for coefficients,count in ((f,n),(g,m)):
        for shift in range(count):
            rows.append([(Fraction(0),)]*shift+list(reversed(coefficients))+
                        [(Fraction(0),)]*(size-shift-len(coefficients)))
    previous=(Fraction(1),); sign=1
    for k in range(size-1):
        check()
        pivot=next((row for row in range(k,size) if rows[row][k]!=(0,)),None)
        if pivot is None: return (Fraction(0),)
        if pivot!=k:
            rows[k],rows[pivot]=rows[pivot],rows[k]; sign=-sign
        value=rows[k][k]
        for i in range(k+1,size):
            for j in range(k+1,size):
                numerator=_add(_multiply(value,rows[i][j]),_scale(_multiply(rows[i][k],rows[k][j]),-1))
                quotient,remainder=_division(numerator,previous)
                if remainder!=(0,):
                    raise GeometryError('Bezier resultant exact division failed')
                rows[i][j]=quotient
            rows[i][k]=(Fraction(0),)
        previous=value
    return _scale(rows[-1][-1],sign)


def bezier_junctions(first,second,*,tolerance=1e-10,cancellation_check=None):
    from .arrangement_geometry import point_parameters
    def check():
        if cancellation_check is not None and cancellation_check():
            raise GeometryError('Bezier intersection predicate cancelled')
    check()
    endpoints=[(t,s) for s in (0.,1.) for t in point_parameters(first,second.evaluate(s),tolerance=tolerance)]
    endpoints.extend((t,s) for t in (0.,1.) for s in point_parameters(second,first.evaluate(t),tolerance=tolerance))
    endpoints=sorted(set(endpoints))
    if len(endpoints)>1:
        low,high=endpoints[0],endpoints[-1]
        if high[0]>low[0] and high[1]!=low[1] and _equal_image(
                first.subcurve(low[0],high[0]),second.subcurve(low[1],high[1]),tolerance):
            return (low,high)
    a,b=_powers(first),_powers(second)
    candidates=[]
    for x,y in combinations(range(3),2):
        check()
        f=(_add(a[x],(-b[x][0],)),*((-value,) for value in b[x][1:]))
        g=(_add(a[y],(-b[y][0],)),*((-value,) for value in b[y][1:]))
        polynomial=_resultant(f,g,check)
        if polynomial==(0,): continue
        roots=isolate_real_roots(polynomial,tolerance=4*np.finfo(float).eps,
                                cancellation_check=cancellation_check)
        candidates.extend(min(1.,max(0.,root.witness)) for root in roots
                          if -32*np.finfo(float).eps<=root.witness<=1+32*np.finfo(float).eps)
        # One nonzero projected resultant contains every spatial intersection.
        break
    else:
        # A common projected factor can represent a partial common curve.
        # Qualify its spatial image over the entire candidate interval.
        pairs=endpoints
        if not pairs: return ()
        if len(pairs)==1: return tuple(pairs)
        low,high=pairs[0],pairs[-1]
        if _equal_image(first.subcurve(low[0],high[0]),second.subcurve(low[1],high[1]),tolerance):
            return (low,high)
        raise GeometryError('common Bezier factor has no qualified spatial interval')
    result=[]
    for t in (0.,1.,*candidates):
        check()
        for s in point_parameters(second,first.evaluate(t),tolerance=tolerance):
            if not any(max(abs(t-u),abs(s-v))<=128*np.finfo(float).eps for u,v in result):
                result.append((t,s))
    return tuple(result)


def bezier_branch_junctions(bezier,branch,*,tolerance=1e-10,cancellation_check=None):
    from .member_arrangements import _support_roots
    from .arrangement_geometry import point_parameters
    def check():
        if cancellation_check is not None and cancellation_check():
            raise GeometryError('Bezier/branch predicate cancelled')
    original=bezier.transformed(np.linalg.inv(np.asarray(branch.transform)))
    constraints=[_support_roots(original,support.surface(),tolerance,check)
                 for support in (branch.first,branch.second)]
    if all(roots is None for roots in constraints):
        raise GeometryError('degenerate polynomial cylinder-intersection interval')
    return tuple((t,s) for t in sorted(set(parameter for roots in constraints if roots is not None for parameter in roots))
                 for s in point_parameters(branch,bezier.evaluate(t),tolerance=tolerance))
