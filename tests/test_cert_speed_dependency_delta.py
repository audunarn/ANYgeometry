"""New arithmetic must not gain mutable Fraction dispatch dependencies."""
from fractions import Fraction as Q
import pytest

from anygeometry import support_curves
from anygeometry.nurbs import NurbsSurface
from anygeometry.nurbs_metric import mul, scale
from test_cert_speed_exact_arithmetic import old_casteljau, old_surface


def slots(value):
    return value._numerator, value._denominator


@pytest.mark.parametrize('method', ['__ge__', '__le__'])
@pytest.mark.parametrize('answer', [False, True])
def test_interval_no_new_fraction_comparison(monkeypatch, method, answer):
    endpoints = [Q(-3), Q(-1), Q(0), Q(2), Q(5)]
    intervals = [(a, b) for a in endpoints for b in endpoints]
    monkeypatch.setattr(Q, method, lambda self, other: answer)
    for a in intervals:
        for b in intervals:
            products = [x*y for x in a for y in b]
            assert mul(a, b) == (min(products), max(products))
        for factor in endpoints:
            products = [x*factor for x in a]
            assert scale(a, factor) == (min(products), max(products))


@pytest.mark.parametrize('mutation', ['numerator', 'denominator', 'constructor'])
def test_integer_paths_no_new_fraction_property_or_constructor(monkeypatch, mutation):
    values, t = [Q(1, 3), Q(2, 3)], Q(1, 2)
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                          (((1., 2., 3.), (2., 3., 4.)), ((3., 4., 5.), (4., 5., 6.))))
    net = surface.exact_net()
    selected = (values[0], net[0][0][0])
    if mutation == 'constructor':
        original = Q.__new__
        def altered(cls, numerator=0, denominator=None):
            if denominator is not None:
                return original(cls, 0)
            return original(cls, numerator)
        monkeypatch.setattr(Q, '__new__', staticmethod(altered))
    else:
        slot = '_' + mutation
        monkeypatch.setattr(Q, mutation, property(lambda self:
            getattr(self, slot) + (1 if any(self is item for item in selected) else 0)))
    assert slots(support_curves._de_casteljau(values, t)) == slots(old_casteljau(values, t))
    assert tuple(map(slots, surface.exact_point(t, t, net))) == tuple(map(slots, old_surface(surface, t, t, net)))


def test_missing_factory_preserves_legacy_recurrence(monkeypatch):
    class LegacyMeta(type(Q)):
        def __getattribute__(cls, name):
            if name == '_from_coprime_ints':
                raise AttributeError(name)
            return super().__getattribute__(name)
    class LegacyFraction(Q, metaclass=LegacyMeta):
        pass
    values, t = [LegacyFraction(1, 3), LegacyFraction(2, 3)], LegacyFraction(1, 2)
    monkeypatch.setattr(support_curves, 'Fraction', LegacyFraction)
    assert slots(support_curves._de_casteljau(values, t)) == slots(old_casteljau(values, t))


def test_surface_negative_and_zero_homogeneous_denominator():
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                          (((1., 2., 3.), (2., 3., 4.)), ((3., 4., 5.), (4., 5., 6.))))
    for weight in (Q(-2, 3), Q(0)):
        net = tuple(tuple(entry[:3] + (weight,) for entry in row) for row in surface.exact_net())
        if weight:
            assert surface.exact_point(Q(1, 2), Q(1, 2), net) == old_surface(surface, Q(1, 2), Q(1, 2), net)
        else:
            with pytest.raises(ZeroDivisionError) as expected:
                old_surface(surface, Q(1, 2), Q(1, 2), net)
            with pytest.raises(ZeroDivisionError) as actual:
                surface.exact_point(Q(1, 2), Q(1, 2), net)
            assert str(actual.value) == str(expected.value)


def test_hostile_fraction_alias_is_not_inspected_on_generic_paths(monkeypatch):
    from anygeometry import nurbs
    reads = []
    class Alias:
        def __call__(self, *args, **kwargs):
            return Q(*args, **kwargs)
        def __getattribute__(self, name):
            if name == '_from_coprime_ints':
                reads.append(name)
                raise RuntimeError('new alias attribute dispatch')
            return object.__getattribute__(self, name)
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                          (((1., 2., 3.), (2., 3., 4.)), ((3., 4., 5.), (4., 5., 6.))))
    net = surface.exact_net()
    monkeypatch.setattr(support_curves, 'Fraction', Alias())
    monkeypatch.setattr(nurbs, 'Fraction', Alias())
    sentinel = object()
    assert support_curves._de_casteljau([sentinel], sentinel) is sentinel
    with pytest.raises(IndexError):
        support_curves._de_casteljau([], sentinel)
    for values, t in (([1, 2], .25), ([Q(1), Q(2)], Q(1, 3)),
                      ([1, Q(2)], Q(1, 3))):
        assert support_curves._de_casteljau(values, t) == old_casteljau(values, t)
    for supplied_net in (net, list(net)):
        assert surface.exact_point(Q(1, 3), Q(2, 7), supplied_net) == old_surface(
            surface, Q(1, 3), Q(2, 7), supplied_net)
    assert reads == []
