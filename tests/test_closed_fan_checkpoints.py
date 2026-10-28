from fractions import Fraction as Q
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import sys
import pytest
import anygeometry.curvilinear_reference_maps as maps
import anygeometry.source_cell_complex as complex
from anygeometry import GeometryError,validate_prepared_material_curvilinear_cells
from anygeometry.plane_nurbs import CURRENT_WORK,PlaneNurbsNotCertified
from anygeometry.section_audit import Work
from test_rational_source_side import side,parabola
from test_prepared_material_curvilinear import square_cells


def fan(boundary,check):
    centre=(Q(1,2),Q(3,4))
    return maps.certify_fan_subtriangle(boundary.curve.support,boundary,centre,
        ((0.,0.,0.),(1.,1.,0.)),(*centre,Q(0)),orientation=1,
        surface_error=Q(3,10),cancellation_check=check)


@contextmanager
def counts(code):
    tool=next(i for i in range(5,-1,-1) if sys.monitoring.get_tool(i) is None)
    count=[0]
    def started(*args):count[0]+=1
    sys.monitoring.use_tool_id(tool,'closed-fan-test')
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,started)
    sys.monitoring.set_local_events(tool,code,sys.monitoring.events.PY_START)
    try:yield count
    finally:
        sys.monitoring.set_local_events(tool,code,0);sys.monitoring.free_tool_id(tool)


def test_genuine_rational_numeric_bound_identical_with_fewer_checkpoint_validations():
    boundary=side(parabola())
    with counts(type(boundary)._validate.__code__) as strict:
        expected=fan(boundary,lambda:False)
    with maps._closed_fan_steps(None) as step:
        assert step is not None and not maps._closed_fan_step(step)
        with maps._closed_fan_window(step):
            with counts(type(boundary)._validate.__code__) as closed:
                actual=fan(boundary,step)
    assert actual==expected and 0<closed[0]<strict[0]


def test_public_cell_proof_bounds_and_correspondence_equal_strict_callback():
    model,regions,chords,cells=square_cells()
    normal=validate_prepared_material_curvilinear_cells(model,regions,chords,cells,surface_error=.01)
    strict=validate_prepared_material_curvilinear_cells(model,regions,chords,cells,
        surface_error=.01,cancellation_check=lambda phase:False)
    assert (normal.source_map_bounds,normal.reference_nodes,normal.boundary_correspondence)==(
        strict.source_map_bounds,strict.reference_nodes,strict.boundary_correspondence)


@pytest.mark.parametrize('kind',('work','custom','parent'))
def test_every_non_none_work_is_conservative_fallback(kind):
    class Custom:
        trusted=True
        parent=Work(50,20)
        def take(self):raise AssertionError('guard must not invoke Work')
    work=Work(50,20) if kind=='work' else Custom()
    if kind=='parent':work.parent=Custom()
    token=CURRENT_WORK.set(work)
    try:
        with maps._closed_fan_steps(None) as step:assert step is None
    finally:CURRENT_WORK.reset(token)


def test_public_none_with_callback_work_retains_actual_work_exception():
    model,regions,chords,cells=square_cells();error=RuntimeError('work callback')
    class CallbackWork:
        max_depth=32;subdivisions=0
        trusted=True
        def take(self):raise error
    token=CURRENT_WORK.set(CallbackWork())
    try:
        with pytest.raises(RuntimeError) as caught:
            validate_prepared_material_curvilinear_cells(model,regions,chords,cells,surface_error=.01)
    finally:CURRENT_WORK.reset(token)
    assert caught.value is error


def test_plain_marker_is_not_authority_and_external_exception_is_original():
    error=RuntimeError('external')
    def check():raise error
    check.closed=True
    assert not maps._closed_fan_step(check)
    with maps._closed_fan_steps(check) as step:assert step is None
    with pytest.raises(RuntimeError) as caught:fan(side(parabola()),check)
    assert caught.value is error


def test_scope_nested_thread_expiry_and_work_transition():
    with maps._closed_fan_steps(None) as outer:
        with maps._closed_fan_window(outer):assert maps._closed_fan_step(outer)
        with maps._closed_fan_steps(None) as inner:
            with pytest.raises(GeometryError):outer()
            assert inner() is False
        assert outer() is False
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(GeometryError):pool.submit(outer).result()
        token=CURRENT_WORK.set(Work(1,1))
        try:
            assert outer() is False
            with maps._closed_fan_window(outer) as selected:assert selected is None
            assert not maps._closed_fan_step(outer)
        finally:CURRENT_WORK.reset(token)
    with pytest.raises(GeometryError):outer()
    assert not maps._CLOSED_FAN_STEPS


@pytest.mark.parametrize('mutation',('code','defaults','binding','helper'))
def test_issued_checkpoint_mutation_refuses(mutation,monkeypatch):
    with maps._closed_fan_steps(None) as step:
        if mutation=='code':step.__code__=step.__code__.replace(co_name='changed')
        elif mutation=='defaults':step.__defaults__=(None,)
        elif mutation=='binding':step.__closure__[0].cell_contents=lambda:False
        else:monkeypatch.setattr(maps,'_closed_fan_live',lambda step:True)
        with pytest.raises(GeometryError):maps._closed_fan_step(step) if mutation!='helper' else step()
    assert not maps._CLOSED_FAN_STEPS


def test_opaque_local_limit_query_never_has_active_fan_lease():
    model,regions,chords,cells=square_cells();seen=[]
    with maps._closed_fan_steps(None) as step:
        def opaque(face,footprints):
            assert maps._CLOSED_FAN_ACTIVE.get() is None
            assert not maps._closed_fan_step(step)
            seen.append(1)
            return Q(1,100),('test_footprint',)
        complex.certify_source_cells(model,regions,chords,cells,Q(1,100),step,
            local_limit_query=opaque)
    assert seen


def test_opaque_query_work_transition_preserves_original_work_exception():
    model,regions,chords,cells=square_cells();error=RuntimeError('late Work callback');tokens=[]
    class OpaqueWork:
        max_depth=32;subdivisions=0
        def take(self):raise error
    with maps._closed_fan_steps(None) as step:
        def opaque(face,footprints):
            if not tokens:tokens.append(CURRENT_WORK.set(OpaqueWork()))
            return Q(1,100),('late_work',)
        try:
            with pytest.raises(RuntimeError) as caught:
                complex.certify_source_cells(model,regions,chords,cells,Q(1,100),step,
                    local_limit_query=opaque)
        finally:
            for token in reversed(tokens):CURRENT_WORK.reset(token)
    assert caught.value is error


def test_opaque_query_work_and_service_transition_keeps_contextualized_result(monkeypatch):
    from anygeometry import SourceCellApproximationInconclusive
    from test_source_cell_inconclusive import fan as inconclusive_fan
    model,regions,chords,cells=square_cells();tokens=[];actual=maps.certify_fan_subtriangle
    def terminal(*args,**kwargs):
        limit=Q(kwargs['surface_error'])
        return actual(**inconclusive_fan(offset=2*limit),surface_error=limit,
                      cancellation_check=kwargs['cancellation_check'])
    with maps._closed_fan_steps(None) as step:
        def opaque(face,footprints):
            if not tokens:tokens.append(CURRENT_WORK.set(Work(100,0)))
            monkeypatch.setattr(complex,'certify_fan_subtriangle',terminal)
            return Q(1,400),('late_work_service',)
        try:
            with pytest.raises(SourceCellApproximationInconclusive):
                complex.certify_source_cells(model,regions,chords,cells,Q(1,400),step,
                    local_limit_query=opaque)
        finally:
            for token in reversed(tokens):CURRENT_WORK.reset(token)


def test_mutated_live_side_inventory_cannot_admit_override(monkeypatch):
    import anygeometry.regular_source_reference as reference
    boundary=side(parabola());actual=reference.RationalSourceSide.box;observed=[]
    def opaque_box(self,lo,hi):
        observed.append((lo,hi))
        return actual(self,lo,hi)
    monkeypatch.setattr(reference.RationalSourceSide,'box',opaque_box)
    monkeypatch.setattr(reference,'_RATIONAL_SIDE_METHODS',())
    with counts(reference.RationalSourceSide._validate.__code__) as strict:
        expected=fan(boundary,lambda:False)
    strict_calls=len(observed)
    observed.clear()
    with maps._closed_fan_steps(None) as step:
        with maps._closed_fan_window(step):
            with counts(reference.RationalSourceSide._validate.__code__) as guarded:
                result=fan(boundary,step)
    assert result==expected and guarded[0]==strict[0]
    assert strict_calls>0 and len(observed)==strict_calls  # neither route may hide the override


def test_work_transition_at_actual_fan_entry_reselects_full_guards():
    boundary=side(parabola())
    token=CURRENT_WORK.set(Work(100000,32))
    try:
        with counts(type(boundary)._validate.__code__) as strict:
            expected=fan(boundary,lambda:False)
    finally:CURRENT_WORK.reset(token)
    tokens=[];tool=next(i for i in range(5,-1,-1) if sys.monitoring.get_tool(i) is None)
    def enter(*args):
        if not tokens:tokens.append(CURRENT_WORK.set(Work(100000,32)))
    sys.monitoring.use_tool_id(tool,'late-work-test')
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,enter)
    sys.monitoring.set_local_events(tool,maps._certify_fan_subtriangle.__code__,sys.monitoring.events.PY_START)
    try:
        with maps._closed_fan_steps(None) as step:
            with maps._closed_fan_window(step):
                with counts(type(boundary)._validate.__code__) as guarded:
                    result=fan(boundary,step)
    finally:
        sys.monitoring.set_local_events(tool,maps._certify_fan_subtriangle.__code__,0)
        sys.monitoring.free_tool_id(tool)
        for token in reversed(tokens):CURRENT_WORK.reset(token)
    assert tokens and result==expected and guarded[0]==strict[0]


@pytest.mark.parametrize('mutation',('binding','thread','work_alias'))
def test_checkpoint_guards_do_not_execute_unknown_comparison_or_callout(mutation,monkeypatch):
    class Poison:
        def __eq__(self,other):raise AssertionError('user comparison executed')
        def __call__(self,*args):raise AssertionError('user callable executed')
        def get(self):raise AssertionError('unknown context get executed')
    with maps._closed_fan_steps(None) as step:
        if mutation=='binding':step.__closure__[0].cell_contents=Poison()
        elif mutation=='thread':monkeypatch.setattr(maps,'_closed_thread',Poison())
        else:
            import anygeometry.plane_nurbs as work
            monkeypatch.setattr(work,'CURRENT_WORK',Poison())
        with pytest.raises(GeometryError):step()
    assert not maps._CLOSED_FAN_STEPS


def test_work_exhaustion_and_native_poison_are_not_hidden_by_closed_path():
    boundary=side(parabola());token=CURRENT_WORK.set(Work(0,20))
    try:
        with maps._closed_fan_steps(None) as step:
            assert step is None
            with pytest.raises(PlaneNurbsNotCertified):fan(boundary,lambda:False)
    finally:CURRENT_WORK.reset(token)
    p=boundary.curve.pcurve;original=p._prepared
    from anygeometry.nurbs import NurbsCurve
    other=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.),(.1,.9),(1.,1.)))
    object.__setattr__(p,'_prepared',other._prepared)
    try:
        with maps._closed_fan_steps(None) as step:
            with maps._closed_fan_window(step):
                with pytest.raises(GeometryError):fan(boundary,step)
    finally:object.__setattr__(p,'_prepared',original)


@pytest.mark.parametrize('attribute',('__defaults__','__kwdefaults__'))
@pytest.mark.parametrize('when',('initial','callback','exception'))
def test_calculator_metadata_never_calls_hostile_repr_or_equality(monkeypatch,attribute,when):
    from anygeometry import _local_curve_geometry as jets
    boundary=side(parabola());entered=[];error=RuntimeError('original callback after metadata mutation')
    class Hostile:
        def __repr__(self):entered.append('repr');raise AssertionError('repr entered')
        def __eq__(self,other):entered.append('eq');raise AssertionError('eq entered')
    def mutate():monkeypatch.setattr(jets._curve_jet,attribute,(Hostile(),) if attribute=='__defaults__' else {'unknown':Hostile()})
    if when=='initial':
        mutate()
        with pytest.raises(GeometryError):fan(boundary,lambda:False)
    else:
        def callback():
            mutate()
            if when=='exception':raise error
            return False
        with pytest.raises(RuntimeError if when=='exception' else GeometryError) as caught:fan(boundary,callback)
        if when=='exception':assert caught.value is error
    assert not entered


@pytest.mark.parametrize('mutation',('source','native'))
def test_genuine_mid_fan_mutation_refuses_strict_and_active_closed(mutation):
    outcomes=[];validation_counts=[]
    for closed in (False,True):
        boundary=side(parabola());pcurve=boundary.curve.pcurve
        attribute='controls' if mutation=='source' else '_prepared'
        original=getattr(pcurve,attribute)
        changed=parabola();object.__setattr__(changed,'controls',((0.,0.),(.2,.8),(1.,1.)))
        if mutation=='native':
            from anygeometry.nurbs import NurbsCurve
            changed=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.),(.2,.8),(1.,1.)))
        value=changed.controls if mutation=='source' else changed._prepared
        fired=[];tool=next(i for i in range(5,-1,-1) if sys.monitoring.get_tool(i) is None)
        def returned(*args):
            if not fired:
                fired.append(maps._CLOSED_FAN_ACTIVE.get() is not None)
                object.__setattr__(pcurve,attribute,value)
        sys.monitoring.use_tool_id(tool,'mid-fan-source-mutation')
        sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,returned)
        sys.monitoring.set_local_events(tool,maps._source_box.__code__,sys.monitoring.events.PY_RETURN)
        try:
            with counts(type(boundary)._validate.__code__) as validations:
                with pytest.raises(GeometryError) as caught:
                    if closed:
                        with maps._closed_fan_steps(None) as step:
                            assert step is not None
                            with maps._closed_fan_window(step):
                                assert maps._closed_fan_step(step)
                                fan(boundary,step)
                    else:fan(boundary,lambda:False)
                outcomes.append(type(caught.value));validation_counts.append(validations[0])
        finally:
            sys.monitoring.set_local_events(tool,maps._source_box.__code__,0);sys.monitoring.free_tool_id(tool)
            object.__setattr__(pcurve,attribute,original)
        assert fired==[closed]
    assert outcomes==[GeometryError,GeometryError]
    assert 0<validation_counts[1]<validation_counts[0]



def test_declared_calculator_function_and_original_code_live_through_callback():
    import gc,types,weakref
    from anygeometry import _local_curve_geometry as jets
    boundary=side(parabola());original=jets._divide;observed=[]
    def install_unique_service():
        # No monkeypatch undo stack or test local keeps the clone/code alive.
        clone=types.FunctionType(original.__code__.replace(co_name='unique_lifetime_divide'),
            original.__globals__,original.__name__,original.__defaults__,original.__closure__)
        clone.__kwdefaults__=original.__kwdefaults__
        jets._divide=clone
        return weakref.ref(clone),weakref.ref(clone.__code__)
    function_ref,code_ref=install_unique_service()
    def callback():
        # Mutating __code__ also proves retaining only the function is insufficient.
        jets._divide.__code__=jets._divide.__code__.replace(co_name='changed_lifetime_divide')
        jets._divide=original
        gc.collect()
        observed.append((function_ref() is not None,code_ref() is not None))
        assert observed[-1]==(True,True)
        return False
    try:
        with pytest.raises(GeometryError,match='context or endpoints changed'):
            fan(boundary,callback)
    finally:jets._divide=original
    assert observed==[(True,True)]
