"""One-operation pure Q4 reuse preserves literal bindings and fan frontiers."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest

from anygeometry import GeometryError
from anygeometry.nurbs import NurbsSurface
from anygeometry.polynomial_cell_bounds import quad_residual_bound
from anygeometry._quad_bound_reuse import quad_bound_operation, quadratic_quad_bound, _ACTIVE
from anygeometry.regular_source_reference import SourceNode, SourceSide
from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle, _polynomial_fast_path
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work


@pytest.mark.parametrize('first', ('_quad_bound_reuse', 'curvilinear_reference_maps'))
def test_cold_module_import_orders_have_no_partial_initialization(first):
    import os
    from pathlib import Path
    import subprocess
    import sys
    source = str(Path(__file__).resolve().parents[1] / 'src')
    code = (
        'import importlib, pathlib; '
        f'first=importlib.import_module("anygeometry.{first}"); '
        'memo=importlib.import_module("anygeometry._quad_bound_reuse"); '
        'maps=importlib.import_module("anygeometry.curvilinear_reference_maps"); '
        f'assert pathlib.Path(memo.__file__).resolve().is_relative_to(pathlib.Path({source!r})); '
        'assert memo.maps is maps; assert callable(memo.quadratic_quad_bound); '
        'assert any(row[1]=="_norm_upper" and row[2] is maps._norm_upper for row in memo._FUNCTIONS)'
    )
    env = dict(os.environ, PYTHONPATH=source + os.pathsep + os.environ.get('PYTHONPATH', ''))
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def operands():
    knots = (0., 0., 0., 1., 1., 1.)
    z = (0., 0., 1.)
    support = NurbsSurface(2, 2, knots, knots,
        tuple(tuple((i/2, j/2, z[i]+z[j]) for j in range(3)) for i in range(3)))
    uv = ((Q(1,4), Q(1,4)), (Q(1,2), Q(1,4)), (Q(1,2), Q(1,2)), (Q(1,4), Q(1,2)))
    nodes = tuple(SourceNode(('interior', i), tuple((x, x) for x in point)) for i, point in enumerate(uv))
    boxes = tuple(node.box for node in nodes)
    xyz = tuple(support.exact_point(*point) for point in uv)
    return support, nodes, boxes, xyz


def fan(support, nodes, boxes, xyz, side, check=None):
    return certify_fan_subtriangle(support, SourceSide(nodes[side], nodes[(side+1)%4]),
        tuple(sum(point[i][0] for point in boxes)/4 for i in (0, 1)),
        (xyz[side], xyz[(side+1)%4]), tuple(sum(point[i] for point in xyz)/4 for i in range(3)),
        orientation=1, surface_error=Q(1), quad_xyz=xyz, quad_uv=boxes, quad_side=side,
        quad_nodes=nodes, cancellation_check=check)


def test_four_actual_fans_reuse_bound_but_keep_callbacks_and_box_charges():
    support, nodes, boxes, xyz = operands()
    expected = quad_residual_bound(support, boxes, xyz, 1)
    work, callbacks = Work(4, 32), []
    token = CURRENT_WORK.set(work)
    try:
        with quad_bound_operation() as state:
            for side in range(4):
                assert fan(support, nodes, boxes, xyz, side, lambda: callbacks.append(1)) == (expected, 1)
            assert (state['calculations'], state['hits']) == (1, 3)
        assert len(callbacks) == 4 and work.boxes == 4
    finally:
        CURRENT_WORK.reset(token)


@pytest.mark.parametrize('mutation', ('surface', 'uv_interval', 'xyz', 'orientation'))
def test_complete_literals_recompute_and_match_uncached(mutation):
    support, nodes, boxes, xyz = operands()
    with quad_bound_operation() as state:
        quadratic_quad_bound(support, boxes, xyz, 1)
        orientation = 1
        if mutation == 'surface':
            support = replace(support, u_range=(.1, .9))
        elif mutation == 'uv_interval':
            boxes = (((boxes[0][0][0], boxes[0][0][1]+Q(1,10000)), boxes[0][1]), *boxes[1:])
        elif mutation == 'xyz':
            xyz = ((xyz[0][0], xyz[0][1], xyz[0][2]+Q(1,1000)), *xyz[1:])
        else:
            orientation = -1
        assert quadratic_quad_bound(support, boxes, xyz, orientation) == quad_residual_bound(support, boxes, xyz, orientation)
        assert state['calculations'] == 2 and state['hits'] == 0


@pytest.mark.parametrize('method', ('exact_point', 'exact_net'))
def test_mutated_exact_implementation_disables_reuse(monkeypatch, method):
    support, nodes, boxes, xyz = operands()
    with quad_bound_operation() as state:
        quadratic_quad_bound(support, boxes, xyz, 1)
        original = getattr(NurbsSurface, method)
        calls = []
        def forward(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)
        monkeypatch.setattr(NurbsSurface, method, forward)
        for _ in range(2):
            assert quadratic_quad_bound(support, boxes, xyz, 1) == quad_residual_bound(support, boxes, xyz, 1)
        assert calls and state['hits'] == 0


def test_cancel_and_work_exhaustion_still_apply_to_warm_hit():
    support, nodes, boxes, xyz = operands()
    class Cancel(BaseException):
        pass
    def cancel():
        raise Cancel()
    with quad_bound_operation() as state:
        fan(support, nodes, boxes, xyz, 0)
        with pytest.raises(Cancel):
            fan(support, nodes, boxes, xyz, 1, cancel)
        assert state['hits'] == 0
        token = CURRENT_WORK.set(Work(0, 32))
        try:
            with pytest.raises(GeometryError, match='budget exhausted'):
                fan(support, nodes, boxes, xyz, 1)
        finally:
            CURRENT_WORK.reset(token)


def test_canonical_side_identity_guard_precedes_warm_reuse():
    support, nodes, boxes, xyz = operands()
    with quad_bound_operation() as state:
        fan(support, nodes, boxes, xyz, 0)
        forged = (replace(nodes[0], identity=('foreign',)), *nodes[1:])
        with pytest.raises(GeometryError, match='canonical corner'):
            certify_fan_subtriangle(support, SourceSide(nodes[0], nodes[1]), (.375,.375),
                xyz[:2], (0,0,0), orientation=1, surface_error=1, quad_uv=boxes,
                quad_xyz=xyz, quad_side=0, quad_nodes=forged)
        assert state['hits'] == 0


def test_no_scope_and_nested_exception_cleanup_preserve_fresh_path():
    support, nodes, boxes, xyz = operands()
    assert _ACTIVE.get() is None
    assert quadratic_quad_bound(support, boxes, xyz, 1) == quad_residual_bound(support, boxes, xyz, 1)
    with quad_bound_operation() as outer:
        quadratic_quad_bound(support, boxes, xyz, 1)
        with pytest.raises(RuntimeError), quad_bound_operation() as inner:
            quadratic_quad_bound(support, boxes, xyz, 1)
            assert inner['hits'] == 0
            raise RuntimeError('abort inner operation')
        assert quadratic_quad_bound(support, boxes, xyz, 1) is outer['entry'][1]
        assert outer['hits'] == 1
    assert _ACTIVE.get() is None


def test_inconclusive_orientation_is_not_cached_and_curved_side_is_ineligible():
    support, nodes, boxes, xyz = operands()
    with quad_bound_operation() as state:
        assert quadratic_quad_bound(support, boxes, xyz, -1) is None
        assert quadratic_quad_bound(support, boxes, xyz, -1) is None
        assert state['hits'] == 0 and state['entry'] is None
        side = SourceSide(nodes[0], nodes[1], curve=object())
        assert _polynomial_fast_path(support, side, (0,0), xyz[:2], (0,0,0), 1, xyz, boxes, 0) is None


@pytest.mark.parametrize('attack', ('ceil', 'keyword_default', 'surface_alias'))
def test_reviewed_effective_context_changes_refuse_reuse(monkeypatch, attack):
    import math
    from anygeometry import nurbs
    from anygeometry._quad_bound_reuse import _context
    support, nodes, boxes, xyz = operands()
    with quad_bound_operation() as state:
        quadratic_quad_bound(support, boxes, xyz, 1)
        if attack == 'ceil':
            original = math.ceil
            monkeypatch.setattr(math, 'ceil', lambda value: original(value))
        elif attack == 'keyword_default':
            monkeypatch.setattr(NurbsSurface.exact_point, '__kwdefaults__', {'net': object()})
        else:
            class OtherSurface(NurbsSurface):
                pass
            monkeypatch.setattr(nurbs, 'NurbsSurface', OtherSurface)
        assert not _context(support)
        assert quadratic_quad_bound(support, boxes, xyz, 1) == quad_residual_bound(support, boxes, xyz, 1)
        assert state['hits'] == 0


def test_raw_same_object_literal_mutation_recomputes():
    support, nodes, boxes, xyz = operands()
    with quad_bound_operation() as state:
        quadratic_quad_bound(support, boxes, xyz, 1)
        object.__setattr__(support, 'u_range', (.1, .9))
        assert quadratic_quad_bound(support, boxes, xyz, 1) == quad_residual_bound(support, boxes, xyz, 1)
        assert state['calculations'] == 2 and state['hits'] == 0


def test_cubic_fallback_retains_each_calls_work_charges():
    from test_cubic_cell_bounds import cubic
    support = cubic()
    _, nodes, boxes, xyz = operands()
    work = Work(200000, 32)
    token = CURRENT_WORK.set(work)
    try:
        expected = quad_residual_bound(support, boxes, xyz, 1)
        one_call = work.boxes
        with quad_bound_operation() as state:
            assert quadratic_quad_bound(support, boxes, xyz, 1) == expected
            assert quadratic_quad_bound(support, boxes, xyz, 1) == expected
            assert state['hits'] == 0 and state['calculations'] == 0
        assert one_call > 0 and work.boxes == 3*one_call
    finally:
        CURRENT_WORK.reset(token)
