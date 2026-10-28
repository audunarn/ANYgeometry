"""Added fast-path operations must not call mutable builtin shadows."""
from fractions import Fraction as Q
import pytest

from anygeometry import nurbs, nurbs_metric, support_curves
from test_cert_speed_exact_arithmetic import old_casteljau, old_surface


@pytest.mark.parametrize('name', ['len', 'tuple', 'type'])
@pytest.mark.parametrize('generic', [False, True])
def test_interval_builtin_shadow_is_never_called(monkeypatch, name, generic):
    calls = []
    a, b = (Q(1), Q(2)), (Q(3), Q(4))
    if generic:
        a, b = list(a), list(b)
    def shadow(*args, **kwargs):
        calls.append(name)
        raise AssertionError('new builtin shadow executed')
    monkeypatch.setattr(nurbs_metric, name, shadow, raising=False)
    assert nurbs_metric.mul(a, b) == (Q(3), Q(8))
    assert nurbs_metric.scale(a, Q(-2)) == (Q(-4), Q(-2))
    assert calls == []


@pytest.mark.parametrize('name', ['all', 'getattr', 'type'])
def test_casteljau_builtin_shadow_is_never_called(monkeypatch, name):
    calls = []
    def shadow(*args, **kwargs):
        calls.append(name)
        raise AssertionError('new builtin shadow executed')
    monkeypatch.setattr(support_curves, name, shadow, raising=False)
    with pytest.raises(IndexError):
        support_curves._de_casteljau([], Q(1, 2))
    sentinel = object()
    assert support_curves._de_casteljau([sentinel], sentinel) is sentinel
    for values, t in (([Q(1, 3), Q(2, 3)], Q(1, 2)), ([1, 2], .5)):
        assert support_curves._de_casteljau(values, t) == old_casteljau(values, t)
    assert calls == []


@pytest.mark.parametrize('name', ['abs', 'all', 'getattr', 'list', 'type'])
@pytest.mark.parametrize('generic', [False, True])
def test_surface_builtin_shadow_is_never_called(monkeypatch, name, generic):
    surface = nurbs.NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((1., 2., 3.), (2., 3., 4.)), ((3., 4., 5.), (4., 5., 6.))))
    net = tuple(tuple(entry[:3]+(Q(-1),) for entry in row) for row in surface.exact_net())
    if generic:
        net = list(net)
    expected = old_surface(surface, Q(1, 2), Q(1, 2), net)
    calls = []
    def shadow(*args, **kwargs):
        calls.append(name)
        raise AssertionError('new builtin shadow executed')
    monkeypatch.setattr(nurbs, name, shadow, raising=False)
    assert surface.exact_point(Q(1, 2), Q(1, 2), net) == expected
    assert calls == []


@pytest.mark.parametrize('own_factory', [False, True])
def test_substituted_fraction_subclass_is_generic_even_with_factory(monkeypatch, own_factory):
    calls = []
    armed = False
    class HostileMeta(type(Q)):
        def __getattribute__(cls, name):
            if armed and name in ('__dict__', '_from_coprime_ints'):
                calls.append(name)
                raise AssertionError('subclass alias metadata inspected')
            return super().__getattribute__(name)
    class Other(Q, metaclass=HostileMeta):
        pass
    if own_factory:
        type.__setattr__(Other, '_from_coprime_ints', classmethod(lambda cls, n, d: Q(99)))
    values, t = [Other(1, 3), Other(2, 3)], Other(1, 2)
    a, b = (Other(1), Other(2)), (Other(3), Other(4))
    products = [x*y for x in a for y in b]
    expected_point = old_casteljau(values, t)
    surface = nurbs.NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((1., 2., 3.), (2., 3., 4.)), ((3., 4., 5.), (4., 5., 6.))))
    net = surface.exact_net()
    expected_surface = old_surface(surface, Q(1, 2), Q(1, 2), net)
    monkeypatch.setattr(support_curves, 'Fraction', Other)
    monkeypatch.setattr(nurbs_metric, 'Q', Other)
    monkeypatch.setattr(nurbs, 'Fraction', Other)
    armed = True
    assert support_curves._de_casteljau(values, t) == expected_point
    assert nurbs_metric.mul(a, b) == (min(products), max(products))
    assert nurbs_metric.scale(a, Other(2)) == (Q(2), Q(4))
    assert surface.exact_point(Q(1, 2), Q(1, 2), net) == expected_surface
    assert calls == []


@pytest.mark.parametrize('change', ['defaults', 'code', 'declaration', 'import_alias', 'binding_defaults'])
def test_guarded_context_refuses_preentry_helper_mutation(monkeypatch, change):
    from anygeometry import _cert_exact_runtime as runtime
    from anygeometry import prepared_material_curvilinear as prepared
    from anygeometry.errors import GeometryError
    if change == 'defaults':
        monkeypatch.setattr(runtime.fraction_kind, '__defaults__', (object(),))
    elif change == 'code':
        def altered(alias, _canonical=None):
            return None
        monkeypatch.setattr(runtime.fraction_kind, '__code__', altered.__code__)
    elif change == 'declaration':
        monkeypatch.setattr(runtime, '_DECLARATION', tuple(list(runtime._DECLARATION)))
    elif change == 'import_alias':
        monkeypatch.setattr(nurbs, '_fraction_kind', lambda alias: alias)
    else:
        monkeypatch.setattr(runtime.binding, '__defaults__', (tuple(list(runtime._DECLARATION)),))
    with pytest.raises(GeometryError, match='exact arithmetic'):
        prepared._cell_reference_calculator_context()


def test_public_chart_callback_shadow_keeps_exact_arithmetic(monkeypatch):
    surface = nurbs.NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., .1))))
    expected = nurbs_metric.certify_projected_chart(surface)
    invoked = []
    def shadow(*args):
        raise AssertionError('callback-installed type shadow executed')
    def callback():
        invoked.append(True)
        monkeypatch.setattr(nurbs_metric, 'type', shadow, raising=False)
    assert nurbs_metric.certify_projected_chart(surface, cancellation_check=callback) == expected
    assert invoked


@pytest.mark.parametrize('change', ['defaults', 'declaration', 'import_alias'])
def test_public_sampling_callback_cannot_renew_admission(monkeypatch, change):
    from anygeometry import _cert_exact_runtime as runtime
    from anygeometry import prepared_material_curvilinear as prepared
    from anygeometry.errors import GeometryError
    from test_adaptive_rational_sampling import rounded
    model, regions, seeds = rounded()
    invoked = []
    def callback(phase):
        if not invoked:
            invoked.append(phase)
            if change == 'defaults':
                monkeypatch.setattr(runtime.fraction_kind, '__defaults__', (object(),))
            elif change == 'declaration':
                monkeypatch.setattr(runtime, '_DECLARATION', tuple(list(runtime._DECLARATION)))
            else:
                monkeypatch.setattr(support_curves, '_fraction_kind', lambda alias: alias)
    with pytest.raises(GeometryError, match='exact arithmetic|calculator|changed'):
        prepared.query_prepared_material_boundary_sampling(model, regions, seeds,
            chord_error=.1, error_measure='cell_reference', cancellation_check=callback)
    assert invoked


@pytest.mark.parametrize('target', ['fraction_kind', 'binding', 'context'])
@pytest.mark.parametrize('attribute', ['__defaults__', '__kwdefaults__'])
def test_new_metadata_refuses_without_hostile_comparison(monkeypatch, target, attribute):
    from anygeometry import _cert_exact_runtime as runtime
    from anygeometry import prepared_material_curvilinear as prepared
    from anygeometry.errors import GeometryError
    function = prepared._exact_arithmetic_context if target == 'context' else getattr(runtime, target)
    original = getattr(function, attribute)
    calls = []
    class Evil:
        def __eq__(self, other):
            calls.append(True)
            setattr(function, attribute, original)
            return True
        __hash__ = object.__hash__
    replacement = (Evil(),) if attribute == '__defaults__' else {Evil(): Evil()}
    monkeypatch.setattr(function, attribute, replacement)
    with pytest.raises(GeometryError, match='exact arithmetic'):
        prepared._cell_reference_calculator_context()
    assert calls == []
    assert getattr(function, attribute) is replacement


def test_unknown_new_helper_refused_before_attribute_dispatch(monkeypatch):
    from anygeometry import _cert_exact_runtime as runtime
    from anygeometry import prepared_material_curvilinear as prepared
    from anygeometry.errors import GeometryError
    calls = []
    class Unknown:
        def __getattribute__(self, name):
            if name in ('__code__', '__defaults__', '__kwdefaults__'):
                calls.append(name)
                raise AssertionError('unknown helper metadata dispatched')
            return object.__getattribute__(self, name)
    monkeypatch.setattr(runtime, 'fraction_kind', Unknown())
    with pytest.raises(GeometryError, match='exact arithmetic'):
        prepared._cell_reference_calculator_context()
    assert calls == []


def test_new_fraction_alias_context_edge_uses_identity(monkeypatch):
    from anygeometry import prepared_material_curvilinear as prepared
    calls = []
    class Alias:
        def __eq__(self, other):
            calls.append(True)
            return True
    before = prepared._cell_reference_calculator_context()
    monkeypatch.setattr(support_curves, 'Fraction', Alias())
    after = prepared._cell_reference_calculator_context()
    assert after != before
    assert calls == []
