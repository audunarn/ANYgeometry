"""Registered exact Green/structural section measure contracts (owner UNRUN)."""
from fractions import Fraction as Q
from dataclasses import replace
import math
import pytest

from anygeometry import Plane,GeometryError,LineRange
from anygeometry.nurbs import NurbsCurve,NurbsSurface,CurveOnSurface
from anygeometry.support_curves import PlaneNurbsCurve
from anygeometry.arrangement_geometry import LinePath
from anygeometry.material_arrangement import ArrangementPath,MaterialDomain
from anygeometry.planar_exact import ExactPlane
from anygeometry.polynomial_green import polynomial_planar_loop_area,section_green


def polynomial_fixture():
    support = NurbsSurface(2,1,(0.,0.,0.,1.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(4.,0.,0.)),((0.,1.,1.),(4.,1.,1.)),((0.,2.,0.),(4.,2.,0.))))
    plane = Plane((2.,0.,0.),(0.,2.,0.),(0.,0.,3.))
    return support,plane


def test_polynomial_nurbs_green_exact_range_reversal_and_repeated_split():
    _,plane = polynomial_fixture()
    curve = NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((2.,0.,0.),(2.,1.,1.),(2.,2.,0.)),domain=(.25,.75))
    first,last = curve.exact_point(0),curve.exact_point(1)
    close = LinePath(tuple(map(float,last)),tuple(map(float,first)))
    loop = (ArrangementPath(curve),ArrangementPath(close))
    value = polynomial_planar_loop_area(plane,loop,1.e-15)
    assert value == float(-Q(1,72))
    pieces = (ArrangementPath(curve.subcurve(0,.5)),ArrangementPath(curve.subcurve(.5,1)),ArrangementPath(close))
    assert polynomial_planar_loop_area(plane,pieces,1.e-15) == value
    reverse = (ArrangementPath(curve.subcurve(1,0)),ArrangementPath(LinePath(close.end,close.start)))
    assert polynomial_planar_loop_area(plane,reverse,1.e-15) == -value


def test_curve_on_surface_iso_and_retained_line_range_integrate_actual_polynomial_image():
    support,plane = polynomial_fixture()
    pcurve = NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5)))
    curve = CurveOnSurface(support,pcurve)
    close = LineRange((2.,0.,0.),(2.,2.,0.),(Q(1),Q(0)))
    loop = (ArrangementPath(curve),ArrangementPath(close))
    assert polynomial_planar_loop_area(plane,loop,1.e-14) == float(-Q(1,9))


def test_uniform_extrusion_section_green_encloses_actual_midpoint_tangent_law_and_requested_precision():
    support,plane = polynomial_fixture()
    curve = PlaneNurbsCurve.from_plane(plane,support)
    close = LinePath((2.,2.,0.),(2.,0.,0.))
    loop = (ArrangementPath(curve),ArrangementPath(close))
    value = polynomial_planar_loop_area(plane,loop,1.e-13)
    assert value == float(-Q(1,9))
    direct = section_green(ExactPlane.from_plane(plane),curve)
    assert direct is not None and direct[1] >= 0
    assert polynomial_planar_loop_area(plane,loop,1.e-40) is None
    reversed_curve = replace(curve,domain=(1.,0.))
    reverse = (ArrangementPath(reversed_curve),ArrangementPath(LinePath(close.end,close.start)))
    assert polynomial_planar_loop_area(plane,reverse,1.e-13) == -value


def test_rational_nontranslation_and_non_profile_slicing_retain_existing_backend():
    support,plane = polynomial_fixture()
    rational = NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((2.,0.,0.),(2.,1.,1.),(2.,2.,0.)),weights=(1.,2.,1.))
    assert polynomial_planar_loop_area(plane,(ArrangementPath(rational),),1.e-9) is None
    rows = list(support.controls)
    rows[1] = (rows[1][0],(4.1,1.,1.))
    nontranslation = replace(support,controls=tuple(rows))
    curve = PlaneNurbsCurve.from_plane(plane,nontranslation)
    assert section_green(ExactPlane.from_plane(plane),curve) is None
    other_plane = Plane((0.,1.,0.),(1.,0.,0.),(0.,0.,1.))
    other = PlaneNurbsCurve.from_plane(other_plane,support)
    assert any(arc.slicing != 0 for arc in other._arcs())
    assert section_green(ExactPlane.from_plane(other_plane),other) is None


def test_polynomial_measure_honours_zero_box_work_and_current_native_cache_validation():
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    support,plane = polynomial_fixture()
    curve = PlaneNurbsCurve.from_plane(plane,support)
    token = CURRENT_WORK.set(Work(0,12))
    try:
        with pytest.raises(GeometryError):
            polynomial_planar_loop_area(plane,(ArrangementPath(curve),),1.e-12)
    finally:
        CURRENT_WORK.reset(token)
    arc = curve._arcs()[0]
    original = arc.s0
    try:
        object.__setattr__(arc,'s0',(arc.s0+arc.s1)/2)
        with pytest.raises(GeometryError,match='runtime cache'):
            section_green(ExactPlane.from_plane(plane),curve)
    finally:
        object.__setattr__(arc,'s0',original)


def captured_principal_sections():
    """Actual captured literal principal profile/plane, without consumer imports."""
    f = float.fromhex
    knots = tuple(map(f,('0x0.0p+0',)*4+('0x1.4149e6efc8513p-2','0x1.1e2774825e09ep-1',
        '0x1.7b9a22a385f1dp-1')+('0x1.0000000000000p+0',)*4))
    yz = (('0x1.d73f5b2414d8bp+0','-0x1.6ed2c57d8a68ep+1'),
          ('0x1.f5ecfaa84eba4p+2','-0x1.45beaf6f98957p+1'),
          ('0x1.5c1539c4150f2p+2','0x1.3ecf65fff14e5p+2'),
          ('-0x1.1ba9ee1ecfcefp+2','0x1.8884b49455b5dp+2'),
          ('-0x1.ca320c2a6cb37p+2','-0x1.8d2ed307e45b2p+1'),
          ('-0x1.1df11492dd5d0p-3','-0x1.1e26e509bdb17p+2'),
          ('0x1.d73f5b2414d8bp+0','-0x1.6ed2c57d8a68ep+1'))
    controls = tuple(((0.,f(y),f(z)),(5.,f(y),f(z))) for y,z in yz)
    plane = Plane(tuple(map(f,('0x1.39399b2fc653bp+2','-0x1.a4fb77a07e778p+2','-0x1.8000000000000p+2'))),
        tuple(map(f,('-0x1.3273365f8ca76p+2','0x1.a4fb77a07e778p+3','0x0.0p+0'))),(0.,0.,12.))
    curves = tuple(PlaneNurbsCurve.from_plane(plane,NurbsSurface(3,1,knots,(0.,0.,1.,1.),controls,
        u_range=(i/4,(i+1)/4))) for i in range(4))
    return plane,curves


def test_actual_captured_principal_sections_use_registered_owner_area_at_actual_tolerance():
    from anygeometry import TolerancePolicy
    import numpy as np
    plane,curves = captured_principal_sections()
    loop = tuple(ArrangementPath(curve) for curve in curves)
    bounds = [curve.bounds() for curve in curves]
    length = float(np.linalg.norm(np.max([b[1] for b in bounds],axis=0)-np.min([b[0] for b in bounds],axis=0)))
    jacobian = float(np.linalg.norm(np.cross(plane.u_vector,plane.v_vector)))
    tolerance = TolerancePolicy().effective_area(length)/jacobian*.05
    value = polynomial_planar_loop_area(plane,loop,tolerance)
    assert value is not None and math.isfinite(value) and value != 0
    domain = MaterialDomain(1,plane,(loop,))
    assert domain.area_loop(loop,tolerance) == value
    assert all(section_green(ExactPlane.from_plane(plane),curve) is not None for curve in curves)


@pytest.mark.parametrize('knots',[(0.,1.),(2.,6.)])
@pytest.mark.parametrize('domain',[(.25,.75),(.75,.25)])
def test_independent_analytic_uniform_section_primitive_with_crop_reverse_and_shifted_knots(knots,domain):
    """S=(4v,u,u²), exact Plane UV=(u/nX,u²), Green primitive=u³/(6nX)."""
    lo,hi = knots
    support = NurbsSurface(2,1,(lo,lo,lo,hi,hi,hi),(0.,0.,1.,1.),
        (((0.,0.,0.),(4.,0.,0.)),((0.,.5,0.),(4.,.5,0.)),((0.,1.,1.),(4.,1.,1.))),u_range=knots)
    plane = Plane((2.,0.,0.),(-.8,.6,0.),(0.,0.,1.))
    full = PlaneNurbsCurve.from_plane(plane,support)
    assert len(full._arcs()) == 1 and full._arcs()[0].slicing == 0
    curve = replace(full,domain=domain)
    frame = ExactPlane.from_plane(plane)
    result = section_green(frame,curve)
    assert result is not None
    a,b = map(Q,domain)
    expected = (b**3-a**3)/(6*Q(.6))
    assert result[0] == expected and result[1] >= 0
    # Independent point checks of the existing rounded evaluator/implicit
    # tangent law, through exact Plane projection. The certificate remains
    # the analytic bound; these stations do not establish an area proof.
    for t in (.125,.5,.875):
        u = a+Q(t)*(b-a)
        point = frame.uv(curve.evaluate(t))
        tangent = frame.uv_direction(curve.derivative(t))
        actual = (point[0]*tangent[1]-point[1]*tangent[0])/2
        ideal = u*u*(b-a)/(2*Q(.6))
        assert abs(actual-ideal) <= result[1]


def test_large_world_origin_requires_fallback_when_actual_rounding_exceeds_requested_tolerance():
    shift = 2.**40
    support = NurbsSurface(2,1,(0.,0.,0.,1.,1.,1.),(0.,0.,1.,1.),
        (((shift,shift,shift),(shift+4,shift,shift)),
         ((shift,shift+.5,shift),(shift+4,shift+.5,shift)),
         ((shift,shift+1,shift+1),(shift+4,shift+1,shift+1))))
    plane = Plane((shift+2,shift,shift),(-.8,.6,0.),(0.,0.,1.))
    curve = PlaneNurbsCurve.from_plane(plane,support)
    direct = section_green(ExactPlane.from_plane(plane),curve)
    assert direct is not None and direct[1] > Q(1.e-12)
    assert polynomial_planar_loop_area(plane,(ArrangementPath(curve),),1.e-12) is None


def test_unregistered_authored_endpoint_correction_subtype_keeps_existing_backend():
    class CorrectedPolynomial(NurbsCurve):
        authored_endpoint_correction = (0.,0.,.01)
        def evaluate(self,parameters):
            return super().evaluate(parameters)+self.authored_endpoint_correction
    curve = CorrectedPolynomial(2,(0.,0.,0.,1.,1.,1.),((2.,0.,0.),(2.,1.,1.),(2.,2.,0.)))
    _,plane = polynomial_fixture()
    assert polynomial_planar_loop_area(plane,(ArrangementPath(curve),),1.e-9) is None


@pytest.mark.parametrize('family',['nurbs','cos'])
def test_pure_polynomial_extraction_zero_work_and_cancel_refuse_before_knot_insertion(family,monkeypatch):
    from anygeometry import nurbs
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    support,plane = polynomial_fixture()
    curve = (NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((2.,0.,0.),(2.,1.,1.),(2.,2.,0.)),domain=(.25,.75))
             if family == 'nurbs' else CurveOnSurface(support,
                 NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5)),domain=(.25,.75))))
    calls = []
    def unexpected(*args,**kwargs):
        calls.append(True)
        raise AssertionError('extraction must not run after entry refusal')
    monkeypatch.setattr(nurbs,'_exact_insert',unexpected)
    loop = (ArrangementPath(curve),)
    token = CURRENT_WORK.set(Work(0,12))
    try:
        with pytest.raises(GeometryError):
            polynomial_planar_loop_area(plane,loop,1.e-12)
    finally:
        CURRENT_WORK.reset(token)
    with pytest.raises(GeometryError,match='cancelled'):
        polynomial_planar_loop_area(plane,loop,1.e-12,cancellation_check=lambda: True)
    assert not calls


def test_bernstein_product_work_and_cancellation_have_inner_checkpoints():
    from anygeometry.polynomial_green import bernstein_green
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    controls = tuple((Q(i),Q(i*i)) for i in range(8))
    token = CURRENT_WORK.set(Work(2,12))
    try:
        with pytest.raises(GeometryError):
            bernstein_green(controls)
    finally:
        CURRENT_WORK.reset(token)
    calls = []
    def cancel():
        calls.append(True)
        return len(calls) == 3
    with pytest.raises(GeometryError,match='cancelled'):
        bernstein_green(controls,cancellation_check=cancel)
    assert len(calls) == 3


def test_callback_equal_definition_path_swap_and_detached_original_support_mutation_refuse():
    support,plane = polynomial_fixture()
    curve = CurveOnSurface(support,NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5))))
    path = ArrangementPath(curve)
    clean = replace(curve,support=replace(support),pcurve=replace(curve.pcurve))
    moved = replace(support,controls=tuple(tuple((x,y,z+.125) for x,y,z in row) for row in support.controls))
    calls = []
    def attack():
        calls.append(True)
        if len(calls) == 2:  # extractor has captured the original path.curve
            object.__setattr__(path,'curve',clean)
            object.__setattr__(curve,'support',moved)
        return False
    try:
        with pytest.raises(GeometryError,match='operand objects changed'):
            polynomial_planar_loop_area(plane,(path,),1.e-12,cancellation_check=attack)
        assert len(calls) == 2
    finally:
        object.__setattr__(path,'curve',curve)
        object.__setattr__(curve,'support',support)


@pytest.mark.parametrize('field',['support','pcurve'])
def test_callback_equal_definition_nested_swap_and_detached_original_mutation_refuse(field):
    support,plane = polynomial_fixture()
    pcurve = NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5)))
    curve = CurveOnSurface(support,pcurve)
    original = getattr(curve,field)
    clean = replace(original)
    old_controls = original.controls
    bad_controls = (tuple(tuple((x,y,z+.125) for x,y,z in row) for row in support.controls)
                    if field == 'support' else ((0.,.25),(1.,.25)))
    calls = []
    def attack():
        calls.append(True)
        if len(calls) == 3:  # pcurve/support references are already captured
            object.__setattr__(curve,field,clean)
            object.__setattr__(original,'controls',bad_controls)
        return False
    try:
        with pytest.raises(GeometryError,match='operand objects changed'):
            polynomial_planar_loop_area(plane,(ArrangementPath(curve),),1.e-12,cancellation_check=attack)
        assert len(calls) == 3
    finally:
        object.__setattr__(original,'controls',old_controls)
        object.__setattr__(curve,field,original)


def test_callback_equal_definition_section_surface_swap_refuses_captured_old_surface():
    support,plane = polynomial_fixture()
    curve = PlaneNurbsCurve.from_plane(plane,support)
    clean = replace(support)
    old_controls = support.controls
    calls = []
    def attack():
        calls.append(True)
        if len(calls) == 3:
            object.__setattr__(curve,'surface',clean)
            object.__setattr__(support,'controls',tuple(tuple((x,y,z+.125) for x,y,z in row) for row in old_controls))
        return False
    try:
        with pytest.raises(GeometryError,match='operand objects changed'):
            polynomial_planar_loop_area(plane,(ArrangementPath(curve),),1.e-12,cancellation_check=attack)
        assert len(calls) == 3
    finally:
        object.__setattr__(support,'controls',old_controls)
        object.__setattr__(curve,'surface',support)


@pytest.mark.parametrize('mutation',['append','truncate'])
def test_callback_mutable_loop_length_change_refuses_even_with_original_prefix(mutation):
    _,plane = polynomial_fixture()
    first = ArrangementPath(LinePath((2.,0.,0.),(2.,1.,1.)))
    second = ArrangementPath(LinePath((2.,1.,1.),(2.,0.,0.)))
    loop = [first,second]
    original = list(loop)
    calls = []
    def attack():
        calls.append(True)
        if len(calls) == 2:
            loop.append(second) if mutation == 'append' else loop.pop()
        return False
    try:
        with pytest.raises(GeometryError,match='operand objects changed'):
            polynomial_planar_loop_area(plane,loop,1.e-12,cancellation_check=attack)
        assert len(calls) == 2
    finally:
        loop[:] = original


def test_callback_equal_value_section_path_object_swap_refuses_before_runtime_consumption():
    support,plane = polynomial_fixture()
    curve = PlaneNurbsCurve.from_plane(plane,support)
    original = curve.path
    class EqualPath(int):
        pass
    replacement = EqualPath(original)
    assert replacement == original and replacement is not original
    calls = []
    def attack():
        calls.append(True)
        if len(calls) == 2:
            object.__setattr__(curve,'path',replacement)
        return False
    try:
        with pytest.raises(GeometryError,match='operand objects changed'):
            polynomial_planar_loop_area(plane,(ArrangementPath(curve),),1.e-12,cancellation_check=attack)
        assert len(calls) == 2
    finally:
        object.__setattr__(curve,'path',original)


def test_mutable_scalar_tolerance_inflation_cannot_relax_captured_requested_precision():
    import numpy as np
    support,plane = polynomial_fixture()
    curve = PlaneNurbsCurve.from_plane(plane,support)
    loop = (ArrangementPath(curve),)
    tolerance = np.array(1.e-40)
    assert polynomial_planar_loop_area(plane,loop,tolerance) is None
    def inflate():
        tolerance[...] = 1.
        return False
    assert polynomial_planar_loop_area(plane,loop,tolerance,cancellation_check=inflate) is None
    assert float(tolerance) == 1.
