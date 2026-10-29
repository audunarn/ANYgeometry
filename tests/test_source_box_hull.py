"""Exact old enclosure equivalence; the specialization grants no admission."""
from dataclasses import replace
from fractions import Fraction as Q
import sys

import pytest

from anygeometry import GeometryError
from anygeometry import nurbs, surface_capabilities as caps
from anygeometry.nurbs import NurbsSurface
from anygeometry import _source_box_hull as hull
from anygeometry.curvilinear_reference_maps import _source_box


def surface(degree=2, *, rational=False, multispan=False, cropped=False, reversed=False, extreme=False):
    count = degree+2 if multispan else degree+1
    knots = (0.,)*(degree+1) + ((.375,) if multispan else ()) + (1.,)*(degree+1)
    controls = tuple(tuple((i/(count-1), j/(count-1), (i*i+j*j)/16.)
                           for j in range(count)) for i in range(count))
    weights = tuple(tuple((1e6 if (i+j)%2 else 1.) if extreme else (1.+(i+j)/8.)
                          for j in range(count)) for i in range(count)) if rational else None
    interval = (.125,.875) if cropped else (0.,1.)
    if reversed:
        interval = interval[::-1]
    return NurbsSurface(degree,degree,knots,knots,controls,weights,interval,interval)


BOXES = (((0.,1.),(0.,1.)), ((0.,0.),(1.,1.)), ((.25,.25),(.5,.5)),
         ((.125,.375),(.375,.875)), ((.9,.1),(.7,.2)),
         ((0.,sys.float_info.min),(1.-sys.float_info.epsilon,1.)))


@pytest.mark.parametrize('options', ({}, {'degree':3}, {'rational':True},
    {'multispan':True}, {'degree':3,'multispan':True,'rational':True},
    {'cropped':True}, {'reversed':True}, {'rational':True,'extreme':True}))
@pytest.mark.parametrize('box', BOXES)
def test_exact_enclosure_matches_original_capability(options, box):
    item = surface(**options)
    assert hull.source_enclosure(item,box) == caps.capabilities(item).enclosure(box)


def test_shifted_nonunit_knots_with_cropped_reversed_chart():
    item = surface(multispan=True,rational=True)
    item = replace(item,knots_u=tuple(3.+5.*k for k in item.knots_u),
        knots_v=tuple(-7.+3.*k for k in item.knots_v),u_range=(7.5,3.5),v_range=(-6.75,-4.25))
    box = ((.2,.8),(.375,.375))
    assert hull.source_enclosure(item,box) == caps.capabilities(item).enclosure(box)


def test_asymmetric_linear_cubic_support():
    item = NurbsSurface(1,3,(0.,0.,1.,1.),(0.,0.,0.,0.,1.,1.,1.,1.),
        tuple(tuple((float(i),j/3.,(i+j*j)/8.) for j in range(4)) for i in range(2)))
    box = ((.125,.875),(.5,.5))
    assert hull.source_enclosure(item,box) == caps.capabilities(item).enclosure(box)


def test_higher_degree_and_repeated_interior_knots_keep_original_route():
    high = surface(degree=4)
    knots = (0.,0.,0.,.5,.5,1.,1.,1.)
    repeated = NurbsSurface(2,2,knots,knots,
        tuple(tuple((i/4.,j/4.,(i+j)/8.) for j in range(5)) for i in range(5)))
    for item in (high,repeated):
        assert not hull._canonical_literals(item)
        assert hull.source_enclosure(item,BOXES[3]) == caps.capabilities(item).enclosure(BOXES[3])


def test_native_path_omits_only_unused_jacobian_work():
    item, box = surface(), ((.2,.6),(.3,.8))
    expected = caps.capabilities(item).enclosure(box)
    calls = []
    watched = {NurbsSurface.patch_bounds.__code__, nurbs._derivative_bound.__code__,
               NurbsSurface.knot_patch_hull.__code__}
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code in watched:
            calls.append(frame.f_code)
    old = sys.getprofile()
    try:
        sys.setprofile(profile)
        actual = hull.source_enclosure(item,box)
    finally:
        sys.setprofile(old)
    assert actual == expected
    assert calls == [NurbsSurface.knot_patch_hull.__code__]


@pytest.mark.parametrize('box', (((-.1,.5),(0.,1.)), ((0.,1.),(.2,1.1))))
def test_outside_chart_preserves_original_refusal(box):
    item = surface()
    with pytest.raises(GeometryError) as original:
        caps.capabilities(item).enclosure(box)
    with pytest.raises(type(original.value), match='chart box must lie inside'):
        hull.source_enclosure(item,box)


def test_source_box_retains_fraction_endpoint_outward_rounding():
    item = surface(rational=True,cropped=True)
    box = ((Q(1,3),Q(1,3)),(Q(2,7),Q(4,7)))
    rounded = tuple((nurbs._float_down(a),nurbs._float_up(b)) for a,b in box)
    expected = caps.capabilities(item).enclosure(rounded)
    assert _source_box(item,box) == tuple((Q(float(a)),Q(float(b))) for a,b in zip(expected.lower,expected.upper))


@pytest.mark.parametrize('name', ('patch_bounds','knot_patch_hull','exact_net','support_key'))
def test_effective_method_wrapper_takes_original_route(monkeypatch,name):
    item = surface()
    original = getattr(NurbsSurface,name)
    def wrapper(*args,**kwargs):
        return original(*args,**kwargs)
    monkeypatch.setattr(NurbsSurface,name,wrapper)
    assert not hull._context(item)
    assert hull.source_enclosure(item,BOXES[2]) == caps.capabilities(item).enclosure(BOXES[2])


def test_unknown_subclass_and_instance_override_use_original_enclosure(monkeypatch):
    class Custom(NurbsSurface):
        def patch_bounds(self,box,net=None):
            return (Q(-7),)*3,(Q(9),)*3,Q(0)
    original = surface()
    custom = Custom(original.degree_u,original.degree_v,original.knots_u,original.knots_v,original.controls)
    assert hull.source_enclosure(custom,BOXES[0]) == caps.capabilities(custom).enclosure(BOXES[0])
    with pytest.raises(AttributeError):
        object.__setattr__(original,'patch_bounds',lambda box,net=None: ((Q(-3),)*3,(Q(4),)*3,Q(0)))
    # The native slotted class rejects instance method replacement. A subclass
    # has an instance dictionary and must preserve its effective override.
    object.__setattr__(custom,'patch_bounds',lambda box,net=None: ((Q(-3),)*3,(Q(4),)*3,Q(0)))
    assert not hull._context(custom)
    assert hull.source_enclosure(custom,BOXES[0]) == caps.capabilities(custom).enclosure(BOXES[0])


def test_raw_valid_control_change_is_read_fresh():
    item = surface()
    before = hull.source_enclosure(item,BOXES[0])
    object.__setattr__(item,'controls',tuple(tuple((x,y,z+1.) for x,y,z in row) for row in item.controls))
    actual = hull.source_enclosure(item,BOXES[0])
    assert actual == caps.capabilities(item).enclosure(BOXES[0])
    assert actual != before


def test_raw_malformed_control_shape_falls_back_with_same_exception():
    item = surface()
    object.__setattr__(item,'controls',())
    assert not hull._canonical_literals(item)
    with pytest.raises(Exception) as original:
        caps.capabilities(item).enclosure(BOXES[0])
    with pytest.raises(type(original.value)):
        hull.source_enclosure(item,BOXES[0])


def test_omitted_derivative_override_falls_back(monkeypatch):
    item = surface()
    def refuse(*args,**kwargs):
        raise GeometryError('effective derivative override')
    monkeypatch.setattr(nurbs,'_derivative_bound',refuse)
    with pytest.raises(GeometryError,match='effective derivative override'):
        hull.source_enclosure(item,BOXES[0])


def test_math_and_capability_overrides_fall_back(monkeypatch):
    item = surface()
    original = caps.capabilities
    calls = []
    def observer(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(caps,'capabilities',observer)
    hull.source_enclosure(item,BOXES[0])
    assert calls == [item]


def test_in_place_method_defaults_change_disables_specialization(monkeypatch):
    item = surface()
    function = NurbsSurface.knot_patch_hull
    monkeypatch.setattr(function,'__defaults__',(object(),))
    assert not hull._context(item)


def test_in_place_keyword_default_change_disables_specialization(monkeypatch):
    item = surface()
    function = NurbsSurface.knot_patch_hull
    monkeypatch.setattr(function,'__kwdefaults__',{'unknown': []})
    assert not hull._context(item)


def test_math_isqrt_override_is_not_skipped(monkeypatch):
    item = surface()
    def refuse(value):
        raise GeometryError('effective sqrt override')
    monkeypatch.setattr(hull.math,'isqrt',refuse)
    with pytest.raises(GeometryError,match='effective sqrt override'):
        hull.source_enclosure(item,BOXES[0])


def test_replaced_surface_class_alias_disables_specialization(monkeypatch):
    item = surface()
    class Unknown(NurbsSurface):
        pass
    monkeypatch.setattr(nurbs,'NurbsSurface',Unknown)
    assert not hull._context(item)


def test_noncanonical_cached_net_uses_original_path(monkeypatch):
    item = surface()
    net = item.exact_net()
    altered = tuple(tuple((x,y,z+w,w) for x,y,z,w in row) for row in net)
    monkeypatch.setitem(caps._NETS,item,altered)
    expected = caps.capabilities(item).enclosure(BOXES[0])
    calls = []
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code is NurbsSurface.patch_bounds.__code__:
            calls.append(1)
    old = sys.getprofile()
    try:
        sys.setprofile(profile)
        actual = hull.source_enclosure(item,BOXES[0])
    finally:
        sys.setprofile(old)
    assert actual == expected and calls == [1]


def test_in_place_method_code_change_disables_specialization(monkeypatch):
    item = surface()
    def changed(self,box,net=None):
        return (-3,)*3,(4,)*3,0
    monkeypatch.setattr(NurbsSurface.patch_bounds,'__code__',changed.__code__)
    assert not hull._context(item)
    assert hull.source_enclosure(item,BOXES[0]) == caps.capabilities(item).enclosure(BOXES[0])
