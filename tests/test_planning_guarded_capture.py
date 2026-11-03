"""Current owned selectors and manifest/settings contracts, without Plane exclusions."""
import numpy as np
import pytest
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline, pin_module


@pytest.mark.parametrize('which', ['chart', 'extruded', 'plane'])
def test_actual_selector_results_preserve_the_model(which):
    from anygeometry._planning_canaries import _fixture
    from anygeometry import nurbs_sweep_chart, extruded_supports, intersections
    model, _ = _fixture()
    identifier = next(iter(model.faces))
    face = model.faces[identifier]
    before = dict(vars(model))
    function, args, expected = {
        'chart': (nurbs_sweep_chart.has_sweep_chart, (face,), False),
        'extruded': (extruded_supports.extruded_support, (model, identifier), None),
        'plane': (intersections._qualified_face_plane, (model, identifier), face.surface),
    }[which]
    assert function(*args) is expected
    assert all(vars(model)[key] is value for key, value in before.items())


def test_opaque_callable_state_is_refused_without_execution():
    calls = []
    class Foreign:
        def __call__(self):
            calls.append(True)
    with pytest.raises(QualificationDecline):
        manifest.capture_callable_manifest((Foreign(),))
    assert calls == []


def test_loaded_capture_preserves_context_or_refuses_the_exact_unsupported_profile():
    import sys
    from anygeometry import _planning_numpy_context as context
    flags, callback, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    source = sys.modules.get('numpy._core._ufunc_config')
    native = sys.modules.get('numpy._core._multiarray_umath')
    if source is None or native is None:
        with pytest.raises(QualificationDecline, match='NumPy context owner modules unavailable'):
            context.capture_numpy_context(package_only=True)
    elif (pin_module(source)['sha256'] != context._SOURCE_SHA256 or
          pin_module(native)['sha256'] != context._NATIVE_SHA256):
        with pytest.raises(QualificationDecline, match='NumPy context source/native image outside profile'):
            context.capture_numpy_context(package_only=True)
    else:
        # Only an admitted image reaches the real package capture and mutation check.
        captured = manifest.capture_loaded_manifest()
        assert captured.wire['workload']['kind'] == 'package-numerical-v1'
        captured.verify()
        try:
            np.seterr(over='raise' if flags['over'] != 'raise' else 'ignore')
            with pytest.raises(PlannerSourceChanged, match='NumPy context changed'):
                captured.verify()
        finally:
            np.seterr(**flags)
    assert np.geterr() == flags and np.geterrcall() is callback and np.getbufsize() == buffer
