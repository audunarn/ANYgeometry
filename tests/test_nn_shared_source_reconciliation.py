"""NN dispatch retention across the frozen common true-map proof handoff.

These local contracts supplement, not replace, whole-cell/transfer gates.
"""
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
from anygeometry.errors import SourceCellRefinementRequired
from anygeometry.nurbs_intersection_curve import intersection_curve_operation


@pytest.mark.parametrize('support_index',(0,1))
@pytest.mark.parametrize('reverse',(False,True))
def test_nn_nodes_sides_native_derivatives_and_boundary_map_retain_owner_chart(support_index,reverse):
    from test_nurbs_boundary_certification import graph_curve
    from anygeometry.regular_source_reference import source_node,source_side,IntersectionSourceSide,parameter_transport_bound
    from anygeometry.curvilinear_reference_maps import native_derivatives,boundary_map
    from anygeometry._source_cell_point_error import fan_point_error_lower_squared
    curve=graph_curve()
    support=(curve.owner.first_source,curve.owner.second_source)[support_index].rebuild()
    curve=curve.for_support(support).subcurve(Q(3,4),Q(1,4)) if reverse else curve.for_support(support).subcurve(Q(1,4),Q(3,4))
    with intersection_curve_operation():
        first,last=(source_node(curve,t) for t in (Q(0),Q(1)))
        assert first.identity==curve.point_identity(0) and last.identity==curve.point_identity(1)
        assert first.box==curve.chart_box(0,0) and last.box==curve.chart_box(1,1)
        side=source_side(curve,(Q(0),Q(1)),first,last)
        assert type(side) is IntersectionSourceSide
        assert parameter_transport_bound(side)==0
        assert side.box(0,0)==first.box and side.box(1,1)==last.box
        assert side.box(Q(1,4),Q(3,4))==curve.chart_box(Q(1,4),Q(3,4))
        assert side.derivative(Q(1,4),Q(3,4))==curve.chart_derivative_bounds(Q(1,4),Q(3,4))
        assert native_derivatives(curve,Q(1,4),Q(3,4))==curve.chart_derivative_bounds(Q(1,4),Q(3,4))
        targets=tuple(tuple((lo+hi)/2 for lo,hi in node.box) for node in (first,last))
        mapped=boundary_map(support,curve,(Q(0),Q(1)),targets)
        assert mapped.curve.chart_support==support_index
        assert mapped.endpoint_boxes==(first.box,last.box)
        # The imported lower witness has no NN true-side proof. It MUST NOT
        # sample/guess, claim success or issue a false refinement signal.
        def no_work():
            raise AssertionError('unknown NN lower witness did numerical work')
        assert fan_point_error_lower_squared(support,side,(Q(1,2),Q(1,2)),
            ((0,0,0),(0,0,0)),(0,0,0),check=no_work) is None


def test_refinement_error_is_strict_control_signal_with_original_full_error():
    error=SourceCellRefinementRequired(3,(10,11,12),Q(1,100),Q(2,10000),requested_error=Q(1,50))
    assert isinstance(error,GeometryError)
    assert (error.face_id,error.node_ids,error.error_limit,error.lower_error_squared,error.requested_error)==(
        3,(10,11,12),Q(1,100),Q(2,10000),Q(1,50))
    for lower,full in ((Q(1,10000),Q(1,50)),(Q(2,10000),Q(1,200))):
        with pytest.raises(GeometryError):
            SourceCellRefinementRequired(3,(10,11,12),Q(1,100),lower,requested_error=full)


def test_adopted_error_and_fresh_cos_pins_and_other_primitives_still_match():
    from anygeometry.nurbs_surface_intersection._dependencies import PINS,verify_dependencies
    assert dict(PINS)['errors.py']=='8dd49f04145b15f5d6bde5a91458b0df13d1c889a7dd760f04092dfcd064b2bd'
    assert dict(PINS)['nurbs.py']=='53310819017985a2bd6153a3449f27b24d087ecee713eb9c379f1eb021f21b77'
    assert dict(PINS)['_cert_exact_runtime.py']=='5b600f1be3d641fe1f6f4feee05542ef72b665f946c160aa745bd296e38e7725'
    assert dict(PINS)['plane_nurbs.py']=='f60c8b6d80ab2c652af2db32df3adfd0817af467b2aac20d775281991aa21db2'
    assert dict(PINS)['tolerance.py']=='f7709e89d2294dab1b71aecdb6a6a92f212a42905531e21ace78e89f1e0ab7f5'
    verify_dependencies()


def test_new_exact_runtime_dependency_mismatch_is_refused(monkeypatch):
    from pathlib import Path
    from anygeometry.nurbs_surface_intersection import _dependencies
    from anygeometry.nurbs_surface_intersection._source import NurbsSurfaceIntersectionIntegrityError
    target = Path(_dependencies.__file__).resolve().parent.parent / '_cert_exact_runtime.py'
    original = Path.read_bytes
    observed = []
    def changed_read(path):
        data = original(path)
        if path == target:
            observed.append(path)
            return data + b'\n# simulated incompatible source\n'
        return data
    monkeypatch.setattr(Path, 'read_bytes', changed_read)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError, match=r'dependency changed: _cert_exact_runtime\.py'):
        _dependencies.verify_dependencies()
    assert observed == [target]


def test_frozen_direct_dependency_alias_and_initial_root_contract():
    from anygeometry.curves import LineRange as CarrierAlias,CurveShape
    from anygeometry.line_ranges import LineRange
    from anygeometry.support_curves import PlaneNurbsCurve
    from typing import get_args
    assert CarrierAlias is LineRange
    assert LineRange in get_args(CurveShape)
    assert type(PlaneNurbsCurve._INITIAL_ROOT_BISECTIONS) is int
    assert PlaneNurbsCurve._INITIAL_ROOT_BISECTIONS==60
    assert PlaneNurbsCurve._MAX_BISECTIONS==1100
    # Importing the frozen internal carrier closes the direct dependency only;
    # this test supplies no model/serialization/runtime admission authority.
