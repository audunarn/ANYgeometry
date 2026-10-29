"""Exact indexed skeleton versus the retained exhaustive leaf oracle."""
from fractions import Fraction as Q
from types import CodeType, FunctionType
import inspect
import pytest
from anygeometry import CurvilinearMaterialCell, GeometryError
from anygeometry.regular_source_reference import SourceNode, SourceSide
from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified
from anygeometry.section_audit import Work
import anygeometry.source_cell_complex as module
from test_source_cell_certification import skeleton_operands


def grid(n):
    nodes = {j*(n+1)+i: SourceNode(('node', j*(n+1)+i),
             ((Q(i),Q(i)),(Q(j),Q(j))))
             for j in range(n+1) for i in range(n+1)}
    cells, boundary, sides = [], {}, {}
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i; b=a+1; d=a+n+1; c=d+1
            for ids in ((a,b,c),(a,c,d)):
                row=len(cells); cells.append(CurvilinearMaterialCell(None,ids,(),(),()))
                for k,(x,y) in enumerate(zip(ids,ids[1:]+ids[:1])):
                    p,q=nodes[x].box,nodes[y].box
                    if any(p[axis][0]==q[axis][0] and p[axis][0] in (0,n)
                           for axis in (0,1)):
                        boundary[row,k]=len(boundary)
                        sides[row,k]=SourceSide(nodes[x],nodes[y])
    return tuple(cells),boundary,sides,nodes


def outcome(proof, operands):
    try:
        proof(*operands,lambda:None)
    except GeometryError as error:
        return type(error),str(error)
    return None


def test_exact_tree_is_a_complete_disjoint_occurrence_partition():
    # Inspect the actual nested calculators without exposing another production
    # service. Both have no closure; their only proof inputs are explicit.
    calculators={code.co_name:FunctionType(code,vars(module))
                 for code in module._skeleton.__code__.co_consts
                 if isinstance(code,CodeType) and code.co_name.startswith('_boundary_')}
    _,boundary,sides,_=grid(4)
    occurrences=[(0,1,module._SideEnclosures(side)) for side in sides.values()]
    occurrences.append(occurrences[0])  # identical side, distinct occurrence
    charges=[]
    tree=calculators['_boundary_index'](occurrences,lambda:charges.append(1))
    assert len(charges)==2*len(occurrences)-1
    def verify(node):
        box,left,right,occurrence=node
        if occurrence is not None:
            assert left is right is None
            assert box==occurrences[occurrence][2].box(Q(0),Q(1))
            return [occurrence]
        a,b=verify(left),verify(right)
        assert not set(a)&set(b)
        assert box==tuple((min(x[0],y[0]),max(x[1],y[1]))
                          for x,y in zip(left[0],right[0]))
        return a+b
    assert sorted(verify(tree))==list(range(len(occurrences)))
    for box in (((Q(1),Q(3)),(Q(1),Q(3))),  # fully separated interior
                ((Q(0),Q(0)),(Q(0),Q(4))),  # touching is not pruned
                ((Q(-1),Q(5)),(Q(2),Q(2)))):
        charges.clear()
        got=set(calculators['_boundary_candidates'](tree,box,lambda:charges.append(1)))
        want={i for i,(_,_,side) in enumerate(occurrences)
              if not module._gap(box,side.box(Q(0),Q(1)))}
        assert got==want and charges


@pytest.mark.parametrize('n',(1,2,4))
def test_index_matches_exhaustive_grid_shared_endpoint_oracle(n):
    operands=grid(n)
    assert outcome(module._skeleton,operands)==outcome(exhaustive_skeleton,operands)==None


@pytest.mark.parametrize('kind',('crossing','touching','collinear'))
@pytest.mark.parametrize('reverse',(False,True))
def test_undeclared_contacts_match_exhaustive_leaf_refusal(kind,reverse):
    cells,boundary,sides,nodes=skeleton_operands()
    nodes=dict(nodes);boundary=dict(boundary);sides=dict(sides)
    # Replace one remote boundary occurrence by a genuine exact SourceSide.
    # Its IDs stay nonincident to the first square's internal diagonal.
    coords={'crossing':((0,1),(1,0)), 'touching':((Q(1,2),Q(1,2)),(2,0)),
            'collinear':((Q(1,4),Q(1,4)),(Q(3,4),Q(3,4)))}[kind]
    key=next(k for k in boundary if min(cells[k[0]].node_ids)>=4)
    ids=cells[key[0]].node_ids;c,d=ids[key[1]],ids[(key[1]+1)%3]
    if reverse:coords=coords[::-1]
    for identifier,point in zip((c,d),coords):
        nodes[identifier]=SourceNode(('node',identifier),tuple((Q(x),Q(x)) for x in point))
    sides[key]=SourceSide(nodes[c],nodes[d])
    operands=cells,boundary,sides,nodes
    indexed=outcome(module._skeleton,operands)
    assert indexed==outcome(exhaustive_skeleton,operands)
    assert indexed is not None


@pytest.mark.parametrize('reverse',(False,True))
def test_curved_rational_leaf_and_reversal_match_exhaustive(reverse):
    from test_rational_source_side import side,parabola
    cells,boundary,sides,nodes=skeleton_operands()
    sides=dict(sides)
    key=next(iter(boundary))
    sides[key]=side(parabola(),Q(1) if reverse else Q(0),Q(0) if reverse else Q(1))
    operands=cells,boundary,sides,nodes
    assert outcome(module._skeleton,operands)==outcome(exhaustive_skeleton,operands)


def test_large_cartesian_grid_has_complete_boundary_cover_with_fixed_budget():
    operands=grid(64)
    cells,boundary,sides,nodes=operands
    internal={tuple(sorted((a,b))) for i,cell in enumerate(cells)
              for k,(a,b) in enumerate(zip(cell.node_ids,cell.node_ids[1:]+cell.node_ids[:1]))
              if (i,k) not in boundary}
    assert len(boundary)==256 and len(internal)*len(boundary)>2_000_000
    work=Work(2_000_000,40); token=CURRENT_WORK.set(work)
    checks=0
    def check():
        nonlocal checks
        checks+=1
    try:module._skeleton(*operands,check)
    finally:CURRENT_WORK.reset(token)
    assert 2*len(boundary)-1<work.boxes<2_000_000
    # The unchanged internal/internal sweep checks once per outer row and
    # charges each overlap pair. Only the new index charges/checks every node.
    assert checks>len(internal)


def test_new_index_construction_and_comparison_keep_work_and_cancellation():
    operands=grid(2)
    token=CURRENT_WORK.set(Work(2,40))
    try:
        with pytest.raises(PlaneNurbsNotCertified,match='budget'):
            module._skeleton(*operands,lambda:None)
    finally:CURRENT_WORK.reset(token)
    error=RuntimeError('original cancellation');calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==5:raise error
    with pytest.raises(RuntimeError) as caught:module._skeleton(*operands,check)
    assert caught.value is error and calls==5


def test_callback_replacement_of_gap_refuses_before_index_certificate(monkeypatch):
    operands=grid(2); calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==4:monkeypatch.setattr(module,'_gap',lambda a,b:True)
    with pytest.raises(GeometryError,match='calculator changed'):
        module._skeleton(*operands,check)


@pytest.mark.parametrize('mutation',('builder_code','builder_binding','charge_code'))
def test_callback_cannot_replace_active_index_calculator_or_charge(mutation):
    """Actual callback attacks the producer's frame at the first parent union."""
    attacked=[]
    def forged(boundary,take):
        # A fake remote leaf would omit almost every real boundary occurrence.
        return (((Q(1000),Q(1001)),(Q(1000),Q(1001))),None,None,0)
    def check():
        charge=inspect.currentframe().f_back
        frame=charge.f_back
        if frame.f_code.co_name!='_boundary_index' or attacked:
            return
        if not frame.f_locals.get('combine',False):
            return
        assert 'build' not in frame.f_locals  # recursive mutable service removed
        take=charge.f_locals['take']
        if mutation=='builder_code':
            charge.f_locals['calculators'][0].__code__=forged.__code__
        elif mutation=='builder_binding':
            index=take.__code__.co_freevars.index('_boundary_index')
            take.__closure__[index].cell_contents=forged
        else:
            take.__code__=take.__code__.replace(co_name='untrusted_charge')
        attacked.append(mutation)
    with pytest.raises(GeometryError,match='calculator changed'):
        module._skeleton(*grid(2),check)
    assert attacked==[mutation]


# The original exhaustive skeleton leaf oracle is appended below. It retains
# the old contact/subdivision/endpoint/derivative arithmetic and no index.

from anygeometry.source_cell_complex import (_SideEnclosures,_gap,_difference,
    _cross,_dot,_same_strict_sign,_orientation)
from anygeometry.plane_nurbs import charge_box,charge_split
from anygeometry.nurbs_metric import scale

def exhaustive_skeleton(cells, boundary_sides, sides, nodes, check):
    """Certify noncollapsed source skeleton and absence of undeclared contact."""
    internal, boundary = {}, []
    for i, cell in enumerate(cells):
        for side, (a, b) in enumerate(zip(cell.node_ids, cell.node_ids[1:]+cell.node_ids[:1])):
            if (i, side) in boundary_sides:
                boundary.append((a, b, _SideEnclosures(sides[(i, side)])))
            else:
                key = tuple(sorted((a, b)))
                internal[key] = key
    rows = []
    for a, b in internal:
        first, last = nodes[a].box, nodes[b].box
        delta = _difference(last, first)
        if not any(lo > 0 or hi < 0 for lo, hi in delta):
            raise GeometryError('source internal edge is not certified noncollapsed')
        box = tuple((min(x[0], y[0]), max(x[1], y[1])) for x, y in zip(first, last))
        rows.append((a, b, first, last, box))
    rows.sort(key=lambda row: row[4][0][0])
    examined = 0
    for index, (a, b, first, last, box) in enumerate(rows):
        check()
        for c, d, p, q, other in rows[index+1:]:
            if other[0][0] > box[0][1]:
                break
            if _gap(box, other):
                continue
            charge_box()
            examined += 1
            if examined > 2000000:
                raise GeometryError('source skeleton proof pair budget exhausted')
            shared = {a, b}&{c, d}
            if shared:
                joint = next(iter(shared))
                va = _difference(nodes[b if a == joint else a].box, nodes[joint].box)
                vb = _difference(nodes[d if c == joint else c].box, nodes[joint].box)
                cross = _cross(va, vb)
                if cross[0] > 0 or cross[1] < 0 or _dot(va, vb)[1] < 0:
                    continue
            elif (_same_strict_sign(_orientation(first, last, p), _orientation(first, last, q)) or
                  _same_strict_sign(_orientation(p, q, first), _orientation(p, q, last))):
                continue
            raise GeometryError('source internal edges have an undeclared or unresolved contact')
        for c, d, curve in boundary:
            # every internal-edge x boundary-side examination is a logical proof step: it is checked,
            # charged and counted against the pair cap BEFORE the (possibly memoized) full-side test,
            # so a cache hit can never bypass cancellation or the existing budgets
            check()
            charge_box()
            examined += 1
            if examined > 2000000:
                raise GeometryError('source skeleton proof pair budget exhausted')
            if _gap(box, curve.box(Q(0), Q(1))):
                continue
            pending = [(Q(0), Q(1), 0)]
            while pending:
                check()
                lo, hi, depth = pending.pop()
                charge_box()
                examined += 1
                if examined > 2000000:
                    raise GeometryError('source skeleton proof pair budget exhausted')
                curved_box = curve.box(lo, hi)
                side = _orientation(first, last, curved_box)
                if _gap(box, curved_box) or side[0] > 0 or side[1] < 0:
                    continue
                joint = a if a in (c, d) else b if b in (c, d) else None
                if joint is not None and ((joint == c and lo == 0) or (joint == d and hi == 1)):
                    line = _difference(nodes[b if a == joint else a].box, nodes[joint].box)
                    derivative = curve.derivative(lo, hi)
                    if joint == d:
                        derivative = tuple(scale(x, Q(-1)) for x in derivative)
                    crossing = _cross(line, derivative)
                    if crossing[0] > 0 or crossing[1] < 0 or _dot(line, derivative)[1] < 0:
                        continue
                if depth >= 20:
                    raise GeometryError('source skeleton boundary contact is not certified')
                charge_split()
                middle = (lo+hi)/2
                pending.extend(((lo, middle, depth+1), (middle, hi, depth+1)))

