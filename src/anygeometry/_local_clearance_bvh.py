"""Exact AABB index of an explicitly retained competitor ledger.

No geometry, contact, topology or exclusion authority is created. Equal keys
are refused; same-feature and adjacent domains are never filtered. A zero
lower bound is inconclusive. Empty means no supplied records, not infinity.
"""
from collections import namedtuple
from fractions import Fraction as Q
from .errors import GeometryError
from .plane_nurbs import charge_box, depth_limit


class ClearanceBVHRefused(GeometryError):
    pass


CompetitorPiece = namedtuple('CompetitorPiece', 'key box')
_Node = namedtuple('_Node', 'box left right piece index')
ClearanceCounters = namedtuple('ClearanceCounters',
    'operations build_records build_nodes build_items query_nodes query_bounds leaf_tests pruned_nodes')
ClearanceQuery = namedtuple('ClearanceQuery',
    'minimum_squared limiting_keys status counters cumulative')
_FIELDS = ClearanceCounters._fields


class ClearanceBudget:
    """One caller-owned operation ledger, reused across builds AND queries.

    Configuration cannot be reassigned. Failed/cancelled operations retain
    their charges; the adapter never installs/replaces the active owner Work.
    Counters describe charged records/nodes, not CPU or wall-clock savings.
    """
    __slots__ = ('_configuration', '_counts')

    def __init__(self, max_operations, *, max_depth=64, cancellation_check=None):
        if type(max_operations) is not int or max_operations <= 0:
            raise ClearanceBVHRefused('clearance budget needs a positive integer cap')
        if type(max_depth) is not int or max_depth < 0:
            raise ClearanceBVHRefused('clearance depth must be a nonnegative integer')
        if cancellation_check is not None and not callable(cancellation_check):
            raise ClearanceBVHRefused('clearance cancellation must be callable or None')
        object.__setattr__(self,'_configuration',(max_operations,max_depth,cancellation_check))
        object.__setattr__(self,'_counts',[0]*len(_FIELDS))

    def __setattr__(self, name, value):
        raise ClearanceBVHRefused('clearance budget configuration/counters cannot be reassigned')

    @property
    def max_operations(self): return self._configuration[0]
    @property
    def max_depth(self): return self._configuration[1]
    @property
    def counters(self): return ClearanceCounters(*self._counts)

    def _take(self, category, pinned):
        if self._configuration is not pinned:
            raise ClearanceBVHRefused('clearance budget configuration changed')
        before=tuple(self._counts)
        if (len(before)!=len(_FIELDS) or any(type(n) is not int or n<0 for n in before)
                or before[0]!=sum(before[1:])):
            raise ClearanceBVHRefused('clearance budget counters malformed')
        if before[0]>=pinned[0]: raise ClearanceBVHRefused('clearance budget exhausted')
        # Account attempted work before the shared Work gate/callback. Nothing
        # resets this ledger if either one raises.
        self._counts[0]+=1
        self._counts[_FIELDS.index(category)]+=1
        expected=tuple(self._counts)
        charge_box()
        callback=pinned[2]
        cancelled=callback('local clearance BVH') if callback is not None else False
        if self._configuration is not pinned or tuple(self._counts)!=expected:
            raise ClearanceBVHRefused('clearance budget changed during callback')
        if cancelled: raise ClearanceBVHRefused('local clearance BVH cancelled')


class ClearanceBVH(tuple):
    """Immutable mathematical index. Construct only with build_clearance_bvh."""
    __slots__=()
    def __new__(cls, *args):
        raise ClearanceBVHRefused('use build_clearance_bvh for a validated index')
    @property
    def record_count(self): return self[1]


def _key(value):
    if type(value) in (str,int,Q): return value
    if type(value) is tuple: return tuple(_key(item) for item in value)
    raise ClearanceBVHRefused('competitor key must contain only exact immutable str/int/Fraction/tuple values')


def _box(value):
    if type(value) is not tuple or len(value)!=3:
        raise ClearanceBVHRefused('AABB requires an immutable tuple of three exact intervals')
    rows=[]
    for row in value:
        if type(row) is not tuple or len(row)!=2 or any(type(x) not in (int,Q) for x in row):
            raise ClearanceBVHRefused('AABB endpoints must be exact int/Fraction pairs')
        a,b=map(Q,row)
        if a>b: raise ClearanceBVHRefused('AABB interval is reversed')
        rows.append((a,b))
    return tuple(rows)


def _distance(first, second):
    return sum((max(Q(0),a-c1,c-b1)**2 for (a,b1),(c,c1) in zip(first,second)),Q(0))


def _budget(value):
    if type(value) is not ClearanceBudget:
        raise ClearanceBVHRefused('an explicit shared ClearanceBudget is required')
    return value._configuration


def build_clearance_bvh(records, *, budget):
    """Retain every record; deterministic median split of rational centroids.

    Input order is the tie-breaker, never comparison of heterogeneous keys.
    All records and hierarchy nodes are copied into recursively immutable tuples.
    """
    pinned=_budget(budget)
    if type(records) is not tuple:
        raise ClearanceBVHRefused('competitor ledger must be a finite immutable tuple')
    copied=[];seen=set()
    for index,record in enumerate(records):
        budget._take('build_records',pinned)
        if type(record) is not CompetitorPiece:
            raise ClearanceBVHRefused('competitor ledger needs exact CompetitorPiece records')
        key=_key(record.key);box=_box(record.box)
        if key in seen: raise ClearanceBVHRefused('duplicate competitor key')
        seen.add(key);copied.append((index,CompetitorPiece(key,box)))
    def build(items,depth):
        budget._take('build_nodes',pinned)
        if depth>depth_limit(pinned[1]): raise ClearanceBVHRefused('clearance hierarchy depth exhausted')
        if not items: return None
        if len(items)==1:
            index,piece=items[0]
            return _Node(piece.box,None,None,piece,index)
        for _ in items: budget._take('build_items',pinned)
        bounds=tuple((min(p.box[k][0] for _,p in items),max(p.box[k][1] for _,p in items)) for k in range(3))
        axis=max(range(3),key=lambda k:(bounds[k][1]-bounds[k][0],-k))
        ordered=sorted(items,key=lambda row:(sum(row[1].box[axis]),row[0]))
        middle=len(ordered)//2
        return _Node(bounds,build(ordered[:middle],depth+1),build(ordered[middle:],depth+1),None,None)
    root=build(copied,0)
    if budget._configuration is not pinned: raise ClearanceBVHRefused('clearance budget changed before return')
    return tuple.__new__(ClearanceBVH,(root,len(copied)))


def query_clearance_bvh(tree, inner_box, *, budget):
    """Exact minimum AABB squared-distance lower bound and ALL minimizing keys.

    A subtree is pruned only for lower > current minimum: equality stays in
    order to retain every limiting competitor (including every zero leaf).
    The owner must decide what to subdivide or exclude; this query does neither.
    """
    pinned=_budget(budget)
    if type(tree) is not ClearanceBVH:
        raise ClearanceBVHRefused('clearance query needs an index built by this module')
    inner=_box(inner_box);before=budget.counters
    root=tree[0];best=None;limiting=[]
    stack=[] if root is None else [(root,0)]
    if root is None: budget._take('query_nodes',pinned)
    while stack:
        node,depth=stack.pop()
        budget._take('query_nodes',pinned)
        if depth>depth_limit(pinned[1]): raise ClearanceBVHRefused('clearance query depth exhausted')
        budget._take('query_bounds',pinned)
        lower=_distance(inner,node.box)
        if best is not None and lower>best:
            budget._take('pruned_nodes',pinned)
            continue
        if node.piece is not None:
            budget._take('leaf_tests',pinned)
            if best is None or lower<best: best=lower;limiting=[(node.index,node.piece.key)]
            elif lower==best: limiting.append((node.index,node.piece.key))
        else:
            children=[node.left,node.right]
            # LIFO near-first, left-before-right on equal exact distance.
            ranked=[]
            for index,child in enumerate(children):
                budget._take('query_bounds',pinned)
                ranked.append((_distance(inner,child.box),index,child))
            ranked.sort(key=lambda row:(row[0],row[1]),reverse=True)
            stack.extend((child,depth+1) for _,_,child in ranked)
    cumulative=budget.counters
    delta=ClearanceCounters(*(a-b for a,b in zip(cumulative,before)))
    return ClearanceQuery(best,tuple(key for _,key in sorted(limiting)),
        'empty' if best is None else ('zero' if best==0 else 'positive'),delta,cumulative)
