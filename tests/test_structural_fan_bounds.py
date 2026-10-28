"""True implicit curved-fan residual proof; portable actual failure operands."""
from dataclasses import replace
from fractions import Fraction as Q
import json
from pathlib import Path

import pytest

from anygeometry import GeometryError, Plane
from anygeometry.nurbs import NurbsSurface, nurbs_from_record
from anygeometry.support_curves import PlaneNurbsCurve, plane_nurbs_from_record
from anygeometry.regular_source_reference import SourceNode, SourceSide, source_node, source_side
from anygeometry.nurbs_runtime_binding import _section_literal
from anygeometry._structural_fan_bounds import structural_fan_bound, _controls, _split, _compose
from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work


def unpack(value):
    if type(value) is list: return [unpack(x) for x in value]
    if type(value) is not dict: return value
    if set(value) == {'q'}: return Q(int(value['q'][0]), int(value['q'][1]))
    if set(value) == {'f'}: return float.fromhex(value['f'])
    if set(value) == {'b'}: return bytes.fromhex(value['b'])
    if set(value) == {'t'}: return tuple(unpack(x) for x in value['t'])
    return {key: unpack(item) for key, item in value.items()}


def captured():
    """Native loaders regenerate truth; captured records are only operands."""
    data = json.loads((Path(__file__).parent/'data'/'principal_curved_fan82.json').read_text())
    support = nurbs_from_record(unpack(data['support']['record']))
    row = data['side']
    curve = plane_nurbs_from_record(unpack(row['curve']['record']))
    nodes = tuple(SourceNode(*unpack(value)) for value in data['nodes'])
    pieces = []
    for item in row['pieces']:
        arc = curve._arcs()[item['arc']]
        assert _section_literal(arc) == unpack(item['literal'])
        pieces.append((unpack(item['a']), unpack(item['b']), arc, nodes[item['first']], nodes[item['last']]))
    side = SourceSide(nodes[row['start']], nodes[row['end']], curve, tuple(pieces),
                      unpack(row['parameters']), unpack(row['transport']))
    args = support, side, unpack(data['centre_uv']), unpack(data['xyz_side']), unpack(data['centre_xyz'])
    kwargs = {key: unpack(value) for key, value in data['kwargs'].items()}
    return args, kwargs


def bound(args, kwargs, callback=None):
    return structural_fan_bound(*args, orientation=kwargs['orientation'],
                               surface_error=kwargs['surface_error'], cancellation_check=callback)


def test_actual_failed_principal_curved_fan82_certifies_through_genuine_caller(monkeypatch):
    args, kwargs = captured()
    def general_backend_forbidden(*args, **kwargs):
        raise AssertionError('actual failed cell fell through to the old general proof')
    monkeypatch.setattr('anygeometry.curvilinear_reference_maps._source_box', general_backend_forbidden)
    result, pieces = certify_fan_subtriangle(*args, **kwargs)
    assert 0 <= result <= kwargs['surface_error']
    assert 0 < pieces <= 32768


def test_captured_reverse_side_preserves_exact_fan_and_orientation():
    args, kwargs = captured()
    support, side, centre, xyz, target = args
    lo, hi, arc, first, last = side.pieces[0]
    reverse = replace(side, start=last, end=first, pieces=((lo, hi, arc, last, first),),
                      parameters=tuple(reversed(side.parameters)))
    result = bound((support, reverse, centre, tuple(reversed(xyz)), target),
                   dict(kwargs, orientation=-kwargs['orientation']))
    assert result is not None and result[0] <= kwargs['surface_error']


def analytic():
    # S(U,V)=(V,U,U²/8); exact plane V+U/4=3/4.
    support = NurbsSurface(2, 1, (0.,)*3+(1.,)*3, (0., 0., 1., 1.),
        (((0., 0., 0.), (1., 0., 0.)), ((0., .5, 0.), (1., .5, 0.)),
         ((0., 1., .125), (1., 1., .125))))
    curve = PlaneNurbsCurve.from_plane(Plane((.75, 0., 0.), (-.25, 1., 0.), (0., 0., 1.)), support)
    first, last = source_node(curve, Q(3, 4)), source_node(curve, Q(1, 4))
    side = source_side(curve, (Q(3, 4), Q(1, 4)), first, last)
    # Off-plane centre is inside the same support; literal FE targets fixed.
    centre = Q(1, 2), Q(1, 4)
    targets = tuple(tuple(support.exact_point(node.box[0][0], Q(3, 4)-node.box[0][0]/4)) for node in (first, last))
    args = support, side, centre, targets, tuple(support.exact_point(*centre))
    return args, {'orientation': 1, 'surface_error': Q(1, 50)}


def test_analytic_cropped_graph_has_independent_quadratic_residual_bound():
    args, kwargs = analytic()
    result = bound(args, kwargs)
    assert result is not None and result[0] <= Q(1, 50)
    # All U lie in[1/4,3/4]. Residual in the sole quadratic world
    # coordinate is variance(U)/8 <= (1/2)²/32=1/128.
    assert result[0] >= Q(1, 256)  # conservative upper bound must cover an actual mid-edge residual


def test_uncertain_endpoint_boxes_include_whole_returned_isolators():
    args, kwargs = analytic()
    support, side, centre, xyz, target = args
    def wider(node):
        return replace(node, box=tuple((a-Q(1, 10**8), b+Q(1, 10**8)) for a, b in node.box))
    first, last = wider(side.start), wider(side.end)
    lo, hi, arc, _, _ = side.pieces[0]
    widened = replace(side, start=first, end=last, pieces=((lo, hi, arc, first, last),))
    result = bound((support, widened, centre, xyz, target), kwargs)
    assert result is not None and result[0] <= kwargs['surface_error']


@pytest.mark.parametrize('kind', ('orientation', 'outside', 'zero_error'))
def test_strict_orientation_chart_and_error_never_waived(kind):
    args, kwargs = analytic()
    if kind == 'orientation': kwargs['orientation'] = -1
    if kind == 'outside': args = (*args[:2], (Q(-1), Q(1, 4)), *args[3:])
    if kind == 'zero_error': kwargs['surface_error'] = Q(0)
    token = CURRENT_WORK.set(Work(1000, 0))
    try: assert bound(args, kwargs) is None
    finally: CURRENT_WORK.reset(token)


def test_unsupported_multi_piece_and_mismatched_support_return_none():
    args, kwargs = analytic()
    support, side, *rest = args
    assert bound((support, replace(side, pieces=side.pieces*2), *rest), kwargs) is None
    different = replace(support, weights=((1., 2.),)*3)
    assert bound((different, side, *rest), kwargs) is None


def test_actual_matching_nonuniform_rational_support_keeps_general_backend():
    args, kwargs = analytic()
    support = replace(args[0], weights=((1., 1.), (2., 2.), (1., 1.)))
    curve = PlaneNurbsCurve(args[1].curve.plane_record, support, args[1].curve.path)
    first, last = source_node(curve, Q(3, 4)), source_node(curve, Q(1, 4))
    side = source_side(curve, (Q(3, 4), Q(1, 4)), first, last)
    assert bound((support, side, *args[2:]), kwargs) is None


def test_zero_work_and_cancellation_remain_typed_refusals():
    args, kwargs = analytic()
    token = CURRENT_WORK.set(Work(0, 12))
    try:
        with pytest.raises(GeometryError): bound(args, kwargs)
    finally: CURRENT_WORK.reset(token)
    with pytest.raises(GeometryError, match='cancelled'): bound(args, kwargs, lambda: True)


@pytest.mark.parametrize('kind', ('domain', 'node', 'native_arc', 'root_law'))
def test_callback_mutations_cannot_redirect_proof(kind, monkeypatch):
    args, kwargs = analytic()
    side = args[1]
    curve = side.curve
    arc = side.pieces[0][2]
    target, name, old, new = {
        'domain': (curve, 'domain', curve.domain, (Q(1, 4), Q(3, 4))),
        'node': (side.start, 'box', side.start.box, ((Q(0), Q(0)), (Q(0), Q(0)))),
        'native_arc': (arc, 's0', arc.s0, arc.s0+Q(1, 100)),
        'root_law': (curve, '_INITIAL_ROOT_BISECTIONS', curve._INITIAL_ROOT_BISECTIONS, 0),
    }[kind]
    def mutate(): object.__setattr__(target, name, new)
    try:
        with pytest.raises(GeometryError): bound(args, kwargs, mutate)
    finally: object.__setattr__(target, name, old)


def test_exact_bernstein_subdivision_preserves_polynomial_values():
    net = _controls({(0, 0): (Q(2), Q(2)), (1, 1): (Q(3), Q(3)), (3, 2): (Q(-1), Q(-1))})
    left, right = _split(net, 0)
    assert left[0] == net[0] and right[-1] == net[-1]
    assert left[-1] == right[0]


def test_genuine_diagonal_section_endpoint_exact_zero_certifies():
    support = analytic()[0][0]
    # Regular true section V=U; the endpoint at(U,V)=(0,0) makes
    # f(U0) exactly zero and sparse interval algebra prunes its constant.
    plane = Plane((0., 0., 0.), (1., 1., 0.), (0., 0., 1.))
    curve = PlaneNurbsCurve.from_plane(plane, support)
    first, last = source_node(curve, Q(0)), source_node(curve, Q(1))
    assert first.box[0] == (Q(0), Q(0))
    side = source_side(curve, (Q(0), Q(1)), first, last)
    centre = Q(1, 2), Q(1, 4)
    xyz = tuple(tuple(support.exact_point(u, u)) for u in (Q(0), Q(1)))
    assert structural_fan_bound(support, side, centre, xyz, tuple(support.exact_point(*centre)),
                                orientation=-1, surface_error=Q(1, 10)) is not None
    result = certify_fan_subtriangle(support, side, centre, xyz, tuple(support.exact_point(*centre)),
                                    orientation=-1, surface_error=Q(1, 10))
    assert result[0] <= Q(1, 10)


def test_pure_zero_polynomial_has_zero_bernstein_controls():
    zero = _compose((Q(0),)*4, {(0, 0): (Q(1, 4), Q(1, 4))})
    assert zero == {}
    assert zero.get((0, 0), (Q(0), Q(0))) == (Q(0), Q(0))
    assert all(value == (Q(0), Q(0)) for row in _controls(zero) for value in row)


@pytest.mark.parametrize('axis', (0, 1))
def test_split_interior_values_equal_independent_cubic_restriction(axis):
    from math import comb
    # Independent analytic expression, including cubic/mixed terms on both
    # axes. Child-net evaluation uses Bernstein basis, not split recursion.
    def polynomial(s, t):
        return Q(2)+3*s*t-s**3*t**2+2*t**3-s**2
    net = _controls({(0, 0): (Q(2),)*2, (1, 1): (Q(3),)*2,
                     (3, 2): (Q(-1),)*2, (0, 3): (Q(2),)*2, (2, 0): (Q(-1),)*2})
    for side, child in enumerate(_split(net, axis)):
        for s, t in ((Q(1, 3), Q(2, 5)), (Q(3, 7), Q(4, 9))):
            value = tuple(sum(comb(3, i)*s**i*(1-s)**(3-i)*comb(3, j)*t**j*(1-t)**(3-j)*child[i][j][end]
                              for i in range(4) for j in range(4)) for end in (0, 1))
            u, v = ((s+side)/2, t) if axis == 0 else (s, (t+side)/2)
            assert value == (polynomial(u, v),)*2
