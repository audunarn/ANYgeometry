"""A parameter choice may be reused; source qualification and Work never are."""
from fractions import Fraction as Q
from dataclasses import replace
import json
import sys

import pytest
import anygeometry as owner
import anygeometry.prepared_material_curvilinear as module
from anygeometry import nurbs_domain
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work
from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as POLICY
from test_regular_source_reference import closed_quarters
from test_sampling_chart_arithmetic_reuse import _geometry


def _fixture():
    model, face, _, edges = closed_quarters()
    regions = owner.query_material_surface_regions(model, (face,))
    seeds = tuple((model.handle('edge', edge), (Q(0), Q(1))) for edge in edges)
    return model, regions, seeds, face


@pytest.mark.slow
def test_actual_one_face_reuses_selection_with_fresh_equal_sampling(tmp_path):
    model, regions, seeds, face = _fixture()
    original = owner.to_dict(model)
    selected = []
    code = nurbs_domain.contains_loop.__code__
    prior = sys.getprofile()

    def observe(frame, event, arg):
        if prior is not None:
            prior(frame, event, arg)
        if event == 'call' and frame.f_code is code:
            selected.append(1)

    try:
        sys.setprofile(observe)
        with module.prepared_material_boundary_sampling_operation():
            first = module.query_prepared_material_boundary_sampling(
                model, regions, seeds, chord_error=.1,
                error_measure='cell_reference', proposal_fraction=.25)
            used = len(selected)
            spent = CURRENT_WORK.get().boxes
            second = module.query_prepared_material_boundary_sampling(
                model, regions, first.edge_parameters, chord_error=.1,
                error_measure='cell_reference', proposal_fraction=.25)
            after = CURRENT_WORK.get().boxes
            (tmp_path/'work-counts.json').write_text(json.dumps(dict(
                cold_boxes=spent,equal_boxes=after-spent,
                cold_selections=used,equal_selections=len(selected)-used),indent=2))
            assert used == 1
            assert len(selected) == used
            assert after > spent
            assert _geometry(second) == _geometry(first)
            assert first.chords.reference_uv_by_face == ((model.handle('face', face), (.5, .5)),)
            for result in (first, second):
                assert all(c.world_error <= .1*.25 for c in result.chords.intervals)
                module.validate_prepared_material_boundary_sampling_binding(model, regions, result)
    finally:
        sys.setprofile(prior)
    assert sys.getprofile() is prior
    assert owner.to_dict(model) == original


def _getter(model, regions, check=lambda: None):
    return module._sampling_anchor_choice(model, regions, check)


@pytest.mark.parametrize('fault', ('source', 'holes', 'tolerance', 'domain'))
def test_retained_choice_refuses_changed_source_inputs(fault):
    model, regions, _, face = _fixture()
    chart = regions.regions[0].sources[0]
    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        assert get(face) == ((Q(1, 2),)*2,)*2
        if fault == 'source':
            model.add_point(4., 4., 4.)
            with pytest.raises(owner.GeometryError):
                get(face)
        else:
            target, name = ((chart.domain, 'boundaries') if fault == 'holes' else
                            (chart, 'domain') if fault == 'domain' else (chart, 'world_tolerance'))
            old = getattr(target, name)
            value = (old + (old[0],) if fault == 'holes' else
                     replace(old, face_id=old.face_id+1) if fault == 'domain' else old*2)
            object.__setattr__(target, name, value)
            try:
                with pytest.raises(owner.GeometryError):
                    get(face)
            finally:
                object.__setattr__(target, name, old)


@pytest.mark.parametrize('target', ('method', 'callee', 'global', 'default', 'search', 'patch_bounds'))
def test_callback_calculator_changes_refuse_before_poison_invocation(monkeypatch, target):
    from anygeometry.material_arrangement import MaterialDomain
    from anygeometry import nurbs_section_faces as faces
    model, regions, _, face = _fixture()
    invoked = []
    armed = False

    def poison(*args, **kwargs):
        invoked.append(True)
        raise AssertionError('changed calculator must not run')

    def check():
        if armed:
            if target == 'default':
                monkeypatch.setattr(faces.chart_boundary_contact, '__defaults__', (3,))
            else:
                changed = {'method': (MaterialDomain, 'contains_loop'),
                           'callee': (faces, '_piece_rect'),
                           'patch_bounds': (module.NurbsSurface, 'patch_bounds'),
                           'global': (nurbs_domain, 'Fraction'),
                           'search': (module, '_sampling_anchor_arithmetic')}
                obj, name = changed[target]
                monkeypatch.setattr(obj, name, poison)
    # Undo while still in the scope, so its own final context check also runs.
    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions, check)
        get(face)
        armed = True
        try:
            with pytest.raises(owner.GeometryError):
                get(face)
        finally:
            armed = False
            monkeypatch.undo()
        assert not invoked


@pytest.mark.parametrize('kind', ('cancel', 'exception'))
def test_hit_preserves_callback_terminal_identity(kind):
    from anygeometry.cancellation import CancellationSignal
    model, regions, _, face = _fixture()
    marker = RuntimeError('original callback')
    signal = CancellationSignal()
    armed = False

    def check():
        if armed:
            if kind == 'cancel':
                module._check(signal, 'anchor selection')
            raise marker

    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions, check)
        get(face)
        armed = True
        signal.cancel()
        expected = owner.GeometryError if kind == 'cancel' else RuntimeError
        with pytest.raises(expected) as caught:
            get(face)
        if kind == 'exception':
            assert caught.value is marker


@pytest.mark.slow
def test_effective_depth_separates_choices_under_the_same_operation(monkeypatch):
    model, regions, _, face = _fixture()
    calls = []
    original = sys.getprofile()
    code = module._sampling_anchor_arithmetic.__code__

    def observe(frame, event, arg):
        if original is not None:
            original(frame, event, arg)
        if event == 'call' and frame.f_code is code:
            calls.append(1)
    try:
        sys.setprofile(observe)
        with module.prepared_material_boundary_sampling_operation():
            first = _getter(model, regions)
            value = first(face)
            assert first(face) == value and len(calls) == 1
            child = module._sampling_worker(replace(POLICY, max_subdivision_depth=31))
            token = CURRENT_WORK.set(child)
            try:
                second = _getter(model, regions)
                assert second(face) == value and len(calls) == 2
            finally:
                CURRENT_WORK.reset(token)
            assert first(face) == value and len(calls) == 2
    finally:
        sys.setprofile(original)


@pytest.mark.parametrize('fault', ('raw_reset', 'local_reset', 'exhausted'))
def test_hit_cannot_conceal_consumed_or_exhausted_work(fault):
    model, regions, _, face = _fixture()
    parent = Work(200_000, 32)
    token = CURRENT_WORK.set(parent)
    consumed = None
    try:
        with pytest.raises(owner.GeometryError):
            with module.prepared_material_boundary_sampling_operation():
                get = _getter(model, regions)
                get(face)
                consumed = parent.boxes
                assert consumed > 0
                if fault == 'raw_reset':
                    parent.boxes = 0
                elif fault == 'local_reset':
                    CURRENT_WORK.get().local.boxes = 0
                else:
                    parent.boxes = parent.max_boxes + 1
                get(face)
        if fault == 'raw_reset':
            assert parent.boxes >= consumed
    finally:
        CURRENT_WORK.reset(token)


def test_at_capacity_hit_is_free_and_unknown_parent_recomputes():
    model, regions, _, face = _fixture()
    parent = Work(200_000, 32)
    token = CURRENT_WORK.set(parent)
    try:
        with module.prepared_material_boundary_sampling_operation():
            get = _getter(model, regions)
            value = get(face)
            parent.boxes = parent.max_boxes
            assert get(face) == value and parent.boxes == parent.max_boxes
    finally:
        CURRENT_WORK.reset(token)
    class UnknownBudget(Work):
        pass
    parent = UnknownBudget(200_000, 32)
    token = CURRENT_WORK.set(parent)
    try:
        with module.prepared_material_boundary_sampling_operation():
            get = _getter(model, regions)
            value = get(face)
            spent = parent.boxes
            assert get(face) == value and parent.boxes > spent > 0
    finally:
        CURRENT_WORK.reset(token)


def test_face_only_interface_and_selector_cell_cannot_seed_a_choice():
    model, regions, _, face = _fixture()
    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        for arguments in ((True,), (model.handle('face', face),), (face, ((Q(0),)*2,)*2)):
            with pytest.raises((owner.GeometryError, TypeError)):
                get(*arguments)
        cell = dict(zip(module._sampling_anchor_choice.__code__.co_freevars,
                        module._sampling_anchor_choice.__closure__))['anchor_choice']
        old = cell.cell_contents
        cell.cell_contents = False
        try:
            with pytest.raises(owner.GeometryError):
                get(face)
        finally:
            cell.cell_contents = old
        assert get(face) == ((Q(1, 2),)*2,)*2


def test_selected_parameter_uses_the_shared_bounded_bank_after_eviction(request):
    model, regions, _, face = _fixture()
    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        value = get(face)
        spent = CURRENT_WORK.get().boxes
        variable, _ = module._sampling_arithmetic_variable()
        state = variable.get()
        other = object()
        for index in range(4096):
            state[0](other, model, (), index, None, lambda: ((Q(0), Q(1)),)*2)
        assert get(face) == value and CURRENT_WORK.get().boxes > spent
    request.node.user_properties.append(('execution_scope', 'shared-bank-eviction-component'))


@pytest.mark.parametrize('reason', ('exception', 'no_anchor'))
def test_failed_selection_is_never_a_cached_success(reason, request):
    model, regions, _, face = _fixture()
    calls = []
    marker = (RuntimeError('original failure') if reason == 'exception' else
              owner.GeometryError('curvilinear source has no certified interior regular reference anchor'))

    def selection(chart):
        calls.append(1)
        if len(calls) == 1:
            raise marker
        return module._sampling_anchor_arithmetic(chart)

    # Fault-injected private component namespace; no public receipt or native
    # qualification is issued, and the registered producer is not replaced.
    factory = module._sampling_arithmetic_producer(selection, anchor_choice=True)
    with module.prepared_material_boundary_sampling_operation():
        get = factory(model, regions, lambda: None)
        with pytest.raises(type(marker)) as caught:
            get(face)
        assert caught.value is marker
        value = get(face)
        assert value == ((Q(1, 2),)*2,)*2 and len(calls) == 2
        assert get(face) == value and len(calls) == 2
    request.node.user_properties.append(('execution_scope', 'fault-injected-arithmetic-component'))


def test_domain_subclass_keeps_original_uncached_search_component(request):
    from anygeometry.material_arrangement import MaterialDomain
    model, regions, _, _ = _fixture()
    chart = regions.regions[0].sources[0]
    class UnregisteredDomain(MaterialDomain):
        pass
    domain = chart.domain
    other = replace(chart, domain=UnregisteredDomain(domain.face_id, domain.support, domain.boundaries))
    assert module._sampling_anchor_eligible(chart)
    assert not module._sampling_anchor_eligible(other)
    assert module._sampling_anchor_arithmetic(other) == module._sampling_anchor_arithmetic(chart)
    request.node.user_properties.append(('execution_scope', 'uncached-selector-component'))


@pytest.mark.slow
def test_final_callback_drift_cannot_publish_a_retained_choice(monkeypatch):
    model, regions, seeds, _ = _fixture()
    original = owner.to_dict(model)
    invoked = []
    changed = []

    def poison(*args, **kwargs):
        invoked.append(True)
        raise AssertionError('late changed calculator must not run')

    def callback(phase=None):
        if phase == 'curvilinear boundary qualification complete':
            changed.append(phase)
            monkeypatch.setattr(nurbs_domain, '_chart_tolerance', poison)
        return False

    with module.prepared_material_boundary_sampling_operation():
        _getter(model, regions)(regions.regions[0].sources[0].face.id)
        try:
            with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
                module.query_prepared_material_boundary_sampling(
                    model, regions, seeds, chord_error=.1, error_measure='cell_reference',
                    proposal_fraction=.25, cancellation_check=callback)
        finally:
            monkeypatch.undo()
    assert not invoked
    assert changed == ['curvilinear boundary qualification complete']
    assert owner.to_dict(model) == original


@pytest.mark.parametrize('field', ('support', 'boundaries'))
def test_callback_chart_property_drift_refuses_before_getter(field, monkeypatch):
    from anygeometry.trimmed_charts import TrimmedSurfaceChart
    model, regions, _, face = _fixture()
    invoked = []
    armed = False

    def poison(self):
        invoked.append(True)
        raise AssertionError('changed chart property must not execute')

    def check():
        if armed:
            monkeypatch.setattr(TrimmedSurfaceChart, field, property(poison))

    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions, check)
        get(face)
        armed = True
        try:
            with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
                get(face)
        finally:
            armed = False
            monkeypatch.undo()
        assert not invoked


@pytest.mark.parametrize('fresh', (False, True))
def test_new_descriptor_binding_refuses_without_invocation(fresh, monkeypatch, tmp_path):
    from anygeometry.trimmed_charts import TrimmedSurfaceChart
    model, regions, _, face = _fixture()
    actual_face = regions.regions[0].sources[0].face
    original = owner.to_dict(model)
    invoked = []

    class ExternalDescriptor(property):
        pass

    def replacement(self):
        invoked.append(True)
        return actual_face

    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        get(face)
        monkeypatch.setattr(TrimmedSurfaceChart, 'face', ExternalDescriptor(replacement))
        try:
            with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
                selected = _getter(model, regions) if fresh else get
                selected(face)
        finally:
            monkeypatch.undo()
            (tmp_path/'descriptor-audit.json').write_text(json.dumps(dict(
                fresh=fresh,replacement_invocations=len(invoked))))
        assert not invoked
    assert owner.to_dict(model) == original


def test_namespace_identity_does_not_compare_payload_and_holds_old_binding():
    import weakref
    from anygeometry.trimmed_charts import TrimmedSurfaceChart
    invoked = []
    class Payload:
        def __eq__(self, other):
            invoked.append(True)
            raise AssertionError('namespace payload equality must not execute')
    name = '_anchor_binding_probe'
    assert name not in vars(TrimmedSurfaceChart)
    first = Payload()
    weak = weakref.ref(first)
    setattr(TrimmedSurfaceChart, name, first)
    try:
        old = module._sampling_anchor_context()
        assert module._sampling_anchor_context() == old
        setattr(TrimmedSurfaceChart, name, Payload())
        del first
        assert weak() is not None  # the snapshot holds a strong old binding
        assert module._sampling_anchor_context() != old
        assert not invoked
    finally:
        delattr(TrimmedSurfaceChart, name)


@pytest.mark.parametrize('kind', ('signed_zero', 'foreign_equality'))
def test_anchor_default_tuple_rebinding_refuses_without_equality(kind, tmp_path):
    model, regions, _, face = _fixture()
    function = nurbs_domain.world_delta
    original = function.__defaults__
    assert original == (0.0, 0.0)
    invoked = []

    class ForeignEqual:
        def __eq__(self, other):
            invoked.append(True)
            return True

    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        get(face)
        function.__defaults__ = (-0.0, 0.0) if kind == 'signed_zero' else (ForeignEqual(), 0.0)
        try:
            with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
                get(face)
        finally:
            function.__defaults__ = original
            (tmp_path/'default-audit.json').write_text(json.dumps(dict(
                kind=kind,foreign_equality_calls=len(invoked))))
        assert not invoked


def test_anchor_code_object_rebinding_refuses_identical_code_values():
    model, regions, _, face = _fixture()
    function = nurbs_domain.world_delta
    original = function.__code__
    replacement = original.replace()
    assert replacement is not original
    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        get(face)
        function.__code__ = replacement
        try:
            with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
                get(face)
        finally:
            function.__code__ = original


def test_function_ownership_ignores_cosmetic_module_equality():
    model, regions, _, face = _fixture()
    function = nurbs_domain.world_delta
    original = function.__module__
    invoked = []

    class ForeignEqual:
        def __eq__(self, other):
            invoked.append(True)
            return True

    with module.prepared_material_boundary_sampling_operation():
        get = _getter(model, regions)
        value = get(face)
        function.__module__ = ForeignEqual()
        try:
            assert get(face) == value
            assert not invoked
        finally:
            function.__module__ = original


@pytest.mark.parametrize('kind', ('default_subclass', 'keyword_subclass', 'keyword_name'))
def test_unrecognized_function_metadata_is_rejected_without_hooks(kind, request):
    function = nurbs_domain.world_delta
    defaults, keywords = function.__defaults__, function.__kwdefaults__
    invoked = []

    class TupleSubclass(tuple):
        def __eq__(self, other):
            invoked.append(True)
            return True

    class DictSubclass(dict):
        def items(self):
            invoked.append(True)
            raise AssertionError('unknown mapping hooks must not execute')

    try:
        if kind == 'default_subclass':
            function.__defaults__ = TupleSubclass(defaults)
        elif kind == 'keyword_subclass':
            function.__kwdefaults__ = DictSubclass(unused=1)
        else:
            function.__kwdefaults__ = {1: 2}
        with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
            module._sampling_anchor_context()
        assert not invoked
    finally:
        function.__defaults__, function.__kwdefaults__ = defaults, keywords
    request.node.user_properties.append(('execution_scope', 'passive-function-metadata-component'))


@pytest.mark.parametrize('kind', ('metaclass_namespace', 'module_namespace', 'foreign_static_callable'))
def test_unknown_metadata_provider_refuses_without_hooks(kind, monkeypatch, request):
    import types
    from anygeometry import material_arrangement
    module._sampling_anchor_context()  # actual canonical metadata baseline
    invoked = []
    if kind == 'metaclass_namespace':
        class Meta(type):
            @property
            def __dict__(cls):
                invoked.append(True)
                raise AssertionError('unknown metaclass namespace hook')
        class Foreign(metaclass=Meta):
            pass
        monkeypatch.setattr(material_arrangement, 'MaterialDomain', Foreign)
    elif kind == 'module_namespace':
        class ForeignModule(types.ModuleType):
            @property
            def __dict__(self):
                invoked.append(True)
                raise AssertionError('unknown module namespace hook')
        monkeypatch.setattr(nurbs_domain, '__class__', ForeignModule)
    else:
        class ForeignCallable:
            def __call__(self):
                raise AssertionError('unknown callable body')
            def __getattribute__(self, name):
                if name == '__func__':
                    invoked.append(True)
                    raise AssertionError('unknown callable metadata hook')
                return object.__getattribute__(self, name)
        monkeypatch.setattr(material_arrangement.MaterialDomain, '_anchor_static_probe',
                            staticmethod(ForeignCallable()), raising=False)
    try:
        with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
            module._sampling_anchor_context()
        assert not invoked
    finally:
        monkeypatch.undo()
    request.node.user_properties.append(('execution_scope', 'passive-metadata-provider-component'))


@pytest.mark.slow
def test_public_complete_callback_alias_drift_is_typed_before_generic_context(monkeypatch):
    model, regions, seeds, face = _fixture()
    original = owner.to_dict(model)
    reached, invoked = [], []

    class Foreign:
        def __init__(self, *args, **kwargs):
            invoked.append(True)
            raise AssertionError('foreign alias must not perform numerical work')

    def callback(phase=None):
        if phase == 'curvilinear boundary qualification complete':
            reached.append(phase)
            monkeypatch.setattr(module, 'PlaneNurbsCurve', Foreign)
        return False

    with module.prepared_material_boundary_sampling_operation():
        _getter(model, regions)(face)  # prime the actual owner anchor producer
        try:
            with pytest.raises(owner.GeometryError, match='^boundary sampling anchor calculator context changed$'):
                module.query_prepared_material_boundary_sampling(
                    model, regions, seeds, chord_error=.1, error_measure='cell_reference',
                    proposal_fraction=.25, cancellation_check=callback)
        finally:
            monkeypatch.undo()
    assert reached == ['curvilinear boundary qualification complete']
    assert not invoked
    assert owner.to_dict(model) == original
