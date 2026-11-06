"""Observed NumPy context binding; never changes H flags or callbacks.

Capturing warn/raise flags does not qualify their reachable Python warning/error
dispatch. Operation contracts must retain that state or decline the operation.
"""
import hashlib
from pathlib import Path
import sys
import types
import numpy as np
from ._planning_code import code_digest
from ._planning_qualification import QualificationDecline

_SOURCE_SHA256='c5e272dcaeca3d5e4fa496e45bd9dc598fd49e72c794cf4ceab5f80bd78db369'
_NATIVE_SHA256='ecafa466c06217851e7e0e8ca9fdb6b039811b32f021e1b9833334c09ada70bf'


def _validate_getters(*,package_only=False):
    """Compare initial loaded getters to pinned source before invoking any one."""
    from ._planning_qualification import pin_module,is_package_module
    source=sys.modules.get('numpy._core._ufunc_config')
    native=sys.modules.get('numpy._core._multiarray_umath')
    umath=sys.modules.get('numpy._core.umath')
    if any(not is_package_module(module) if package_only else type(module) is not types.ModuleType for module in (source,native,umath)):
        raise QualificationDecline('NumPy context owner modules unavailable')
    source_pin=pin_module(source,package_only=package_only);native_pin=pin_module(native,package_only=package_only)
    if (source_pin.get('kind')!='python' or source_pin.get('sha256')!=_SOURCE_SHA256 or
            native_pin.get('kind')!='native' or native_pin.get('sha256')!=_NATIVE_SHA256):
        raise QualificationDecline('NumPy context source/native image outside profile')
    getter=vars(native).get('_get_extobj_dict')
    if (type(getter) is not types.BuiltinFunctionType or getter.__name__!='_get_extobj_dict' or
            vars(source).get('_get_extobj_dict') is not getter or vars(umath).get('_get_extobj_dict') is not getter):
        raise QualificationDecline('NumPy native context getter owner mismatch')
    filename=source_pin['path'];raw=Path(filename).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=_SOURCE_SHA256:raise QualificationDecline('NumPy getter source changed while reading')
    compiled=compile(raw,filename,'exec',dont_inherit=True)
    expected={item.co_name:item for item in compiled.co_consts if type(item) is types.CodeType}
    for name in ('geterr','geterrcall','getbufsize'):
        function=vars(np).get(name)
        if (type(function) is not types.FunctionType or vars(source).get(name) is not function or
                function.__globals__ is not vars(source) or function.__closure__ or
                function.__defaults__ is not None or function.__kwdefaults__ is not None or
                code_digest(function.__code__)!=code_digest(expected[name])):
            raise QualificationDecline('NumPy context getter differs from pinned source: '+name)
    return (source,native,umath,getter,*(vars(np)[name] for name in ('geterr','geterrcall','getbufsize')))


def _snapshot(*,package_only=False,bindings=None):
    # The owner-load anchor binds these exact getters and their live edges before
    # invoking them. A late getter replacement is never called to self-certify.
    if not package_only:
        from ._planning_ledger import runtime_infrastructure
        handle,pin=runtime_infrastructure()
        anchored={row[1]:row[3] for row in vars(handle[0])['ANCHOR'] if row[0]=='numpy'}
        if any(vars(np).get(name) is not anchored.get(name) for name in ('geterr','geterrcall','getbufsize')):
            raise QualificationDecline('NumPy context getter has no original owner binding')
    current=_validate_getters(package_only=package_only)
    if bindings is not None and any(a is not b for a,b in zip(current,bindings)):
        raise QualificationDecline('NumPy context interface binding changed')
    errors=np.geterr();callback=np.geterrcall();buffer=np.getbufsize()
    if callback is not None:raise QualificationDecline('active NumPy error callback is unsupported')
    if (type(errors) is not dict or set(errors)!= {'divide','over','under','invalid'} or
            any(type(value) is not str or value not in ('ignore','warn','raise') for value in errors.values())):
        raise QualificationDecline('unsupported NumPy error flag context')
    if type(buffer) is not int or buffer<16 or buffer%16:
        raise QualificationDecline('unsupported NumPy buffer context')
    return tuple(sorted(errors.items())),buffer


class NumpyContext:
    def __init__(self,*,package_only=False):
        self.package_only=package_only
        self._bindings=_validate_getters(package_only=package_only)
        from ._planning_qualification import ModuleObserver
        self._modules=tuple((module,type(module),None if type(module) is types.ModuleType else ModuleObserver(module)) for module in (np,*self._bindings[:3])) if package_only else ()
        self._record=_snapshot(package_only=package_only,bindings=self._bindings)

    @property
    def wire(self):
        return dict(errors=dict(self._record[0]),callback=None,buffer_size=self._record[1])

    @property
    def binding(self):
        return dict(source_sha256=_SOURCE_SHA256,native_sha256=_NATIVE_SHA256,
                    getters=['geterr','geterrcall','getbufsize'])

    def verify(self):
        from ._planning_executor import PlannerSourceChanged
        try:
            for module,kind,observer in self._modules:
                if type(module) is not kind:raise QualificationDecline('NumPy module class changed')
                if observer is not None:observer.verify()
            if _snapshot(package_only=self.package_only,bindings=self._bindings)!=self._record:raise QualificationDecline('NumPy effective context changed')
        except Exception as error:raise PlannerSourceChanged('NumPy context changed: '+str(error)) from error


def capture_numpy_context(*,package_only=False):return NumpyContext(package_only=package_only)
