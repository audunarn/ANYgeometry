"""Stable code-field fingerprints, independent of marshal reference encoding."""
import hashlib
import struct
import types
from ._planning_qualification import QualificationDecline


_FIELDS=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals',
    'co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames',
    'co_freevars','co_cellvars','co_filename','co_name','co_qualname',
    'co_firstlineno','co_linetable','co_exceptiontable')


def _encode(value):
    """Closed record encoding; does not call the JSON implementation being bound."""
    typ=type(value)
    if typ is str:tag=b's';payload=value.encode('utf-8','surrogatepass')
    elif typ is bool:tag=b'b';payload=b'1' if value else b'0'
    elif typ is int:
        tag=b'i';payload=(b'-' if value<0 else b'+')+abs(value).to_bytes(max(1,(abs(value).bit_length()+7)//8),'big')
    elif typ in (tuple,list):tag=b'l';payload=b''.join(_encode(item) for item in value)
    else:raise QualificationDecline('unsupported code record')
    return tag+str(len(payload)).encode('ascii')+b':'+payload


def code_digest(code):
    if type(code) is not types.CodeType:
        raise QualificationDecline('code fingerprint requires exact CodeType')
    count=0
    def record(value,depth=0):
        nonlocal count
        count+=1
        if count>80000 or depth>64:
            raise QualificationDecline('code fingerprint exceeds finite bound')
        typ=type(value)
        if value is None:return ['none']
        if value is Ellipsis:return ['ellipsis']
        if typ in (str,int,bool):return [typ.__name__,value]
        if typ is bytes:return ['bytes',value.hex()]
        if typ is float:return ['float',struct.pack('>d',value).hex()]
        if typ is complex:return ['complex',struct.pack('>dd',value.real,value.imag).hex()]
        if typ is slice:
            return ['slice',record(value.start,depth+1),record(value.stop,depth+1),record(value.step,depth+1)]
        if typ in (tuple,frozenset):
            rows=[record(item,depth+1) for item in value]
            if typ is frozenset:rows.sort(key=_encode)
            return [typ.__name__,rows]
        if typ is types.CodeType:
            return ['code',[(name,record(object.__getattribute__(value,name),depth+1)) for name in _FIELDS]]
        raise QualificationDecline('unsupported code constant type')
    return hashlib.sha256(_encode(record(code))).hexdigest()


def passive_data(value,*,key=False):
    """Encode bounded exact data without invoking user repr, comparison or hooks."""
    count=0
    def record(item,depth=0):
        nonlocal count
        count+=1
        if count>80000 or depth>64:raise QualificationDecline('passive data bound')
        typ=type(item)
        if typ in (str,bool,int):return [typ.__name__,item]
        if typ is float:return ['float',struct.pack('>d',item).hex()]
        if typ is tuple:return ['tuple',[record(part,depth+1) for part in item]]
        if not key:
            if item is None:return ['none']
            if typ is bytes:return ['bytes',item.hex()]
            if typ is complex:return ['complex',struct.pack('>dd',item.real,item.imag).hex()]
            if typ is list:return ['list',[record(part,depth+1) for part in item]]
            if typ is dict and all(type(name) is str for name in item):
                return ['dict',[(name,record(part,depth+1)) for name,part in sorted(item.items())]]
        raise QualificationDecline('unsupported passive data type')
    return _encode(record(value))

def exact_numerical_record(value):
    """Qualification-only typed numerical values, including exact floating bits.

    This is not a transport codec. It includes every declared dataclass field;
    no metadata flag, cache name or approximate equality silently omits values.
    Unknown types, object arrays and cycles refuse. This compares declared fields;
    model/codec admission separately qualifies object extras and derived caches.
    Array layout is represented
    by dtype/shape and canonical C-order element bytes, not incidental strides.
    """
    import dataclasses,enum,sys,uuid
    import numpy as np
    active=set();count=0
    def record(item):
        nonlocal count
        count+=1
        if count>100000:raise QualificationDecline('exact numerical evidence bound')
        typ=type(item)
        if item is None:return ['none']
        if typ in (bool,int,str):return [typ.__name__,item]
        if typ is bytes:return ['bytes',item.hex()]
        if typ is float:return ['float64',struct.pack('>d',item).hex()]
        if typ is complex:return ['complex128',struct.pack('>dd',item.real,item.imag).hex()]
        if typ is uuid.UUID:return ['uuid',item.hex]
        if typ is np.ndarray:
            if item.dtype.hasobject:raise QualificationDecline('object array in numerical evidence')
            return ['ndarray',item.dtype.str,repr(item.dtype.descr),list(item.shape),item.tobytes(order='C').hex()]
        if np.generic in type.__getattribute__(typ,'__mro__'):
            if item.dtype.hasobject:raise QualificationDecline('object scalar in numerical evidence')
            return ['numpy-scalar',item.dtype.str,item.tobytes().hex()]
        if enum.Enum in type.__getattribute__(typ,'__mro__'):
            return ['enum',typ.__module__,typ.__qualname__,object.__getattribute__(item,'_name_'),record(object.__getattribute__(item,'_value_'))]
        identity=id(item)
        if identity in active:raise QualificationDecline('cyclic numerical evidence')
        active.add(identity)
        try:
            if typ in (tuple,list):return [typ.__name__,[record(part) for part in item]]
            if typ in (set,frozenset):return [typ.__name__,sorted((record(part) for part in item),key=_encode)]
            if typ is dict:return ['dict',sorted(([record(key),record(part)] for key,part in item.items()),key=_encode)]
            if typ.__module__.startswith('anygeometry.') and '__dataclass_fields__' in vars(typ):
                owner=sys.modules.get(typ.__module__);bound=owner
                for name in typ.__qualname__.split('.'):
                    if bound is None:break
                    bound=vars(bound).get(name)
                if bound is not typ:raise QualificationDecline('unregistered numerical record type')
                return ['record',typ.__module__,typ.__qualname__,[[field.name,record(object.__getattribute__(item,field.name))] for field in dataclasses.fields(typ)]]
            raise QualificationDecline('unsupported exact numerical evidence type: '+typ.__module__+'.'+typ.__qualname__)
        finally:active.remove(identity)
    return record(value)


def stage_cost_vector(kind,tasks):
    """Bind existing ordered task IDs/cost bits; never compute numerical costs."""
    import json,math
    if type(kind) is not str or kind not in ('pair','face') or type(tasks) not in (tuple,list) or len(tasks)>100000:
        raise QualificationDecline('stage cost vector schema/bound')
    rows=[];seen=set();total=0.
    for task in tasks:
        if type(task) is not dict or task.get('kind')!=kind:
            raise QualificationDecline('stage cost vector task kind')
        names=('first','second') if kind=='pair' else ('face',)
        ids=tuple(task.get(name) for name in names);cost=task.get('cost')
        if (any(type(value) is not int or value<1 for value in ids) or ids in seen or
                type(cost) not in (float,int)):
            raise QualificationDecline('stage cost vector identity/type')
        try:cost=float(cost)
        except (OverflowError,ValueError) as error:raise QualificationDecline('stage cost vector cost') from error
        if not math.isfinite(cost) or cost<=0:raise QualificationDecline('stage cost vector cost')
        seen.add(ids);rows.append([*ids,struct.pack('!d',cost).hex()]);total+=cost
    if not math.isfinite(total):raise QualificationDecline('stage cost vector total')
    payload=dict(schema='stage-cost-vector-v1',stage=kind,rows=rows)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(payload,sha256=digest,count=len(rows),total_weight=total)

class InfrastructurePin:
    """Finite H/P resource-code identity, separate from numerical execution roots.

    All owned definitions and module bindings in the selected infrastructure are
    retained. Python bodies are checked against their current source before
    capture; generated dataclass members require their registered owning-class
    binding. External provider internals stop at pinned source/native interfaces.
    """
    def __init__(self,role):
        import importlib,sys,dataclasses,dis,ast,ctypes
        from pathlib import Path
        from ._planning_manifest import _RetainedChecks,_ImmutableEvidence,_is_type,_dataclass_slot_values,_prepare_slot_metadata
        from ._planning_qualification import pin_module,is_package_module,ModuleObserver
        from . import _planning_native
        if role not in ('host','planner'):raise QualificationDecline('infrastructure role')
        names=('_planning_code','_planning_manifest','_planning_ledger','_planning_qualification',
            '_planning_executor','_planning_session','_planning_process','_planning_windows_launch',
            '_planning_native','_planning_numpy_context','_planning_batch','_planning_values','_planning_model_state')
        if role=='host':names+=('_planning_host','_planning_controller','_planning_memory')
        self.modules={};self.checks=_RetainedChecks();self.source=_ImmutableEvidence()
        self.references=[];self.nodes=0;self.rows=[];seen=set();pending=[];generated={}
        modules=[importlib.import_module('anygeometry.'+name) for name in names]
        if role=='planner':
            main=sys.modules.get('__main__')
            if type(main) is not types.ModuleType or Path(vars(main).get('__file__','')).resolve()!=Path(__file__).with_name('_planning_child.py').resolve():
                raise QualificationDecline('planner executable source owner')
            modules.append(main)
        owned={module.__name__:module for module in modules}
        def pin(module):
            if module.__name__ not in self.modules:
                observer=None if type(module) is types.ModuleType else ModuleObserver(module)
                self.modules[module.__name__]=(module,pin_module(module,package_only=True,observer=observer),vars(module).get('__file__'),vars(module).get('__spec__'),type(module),observer)
            return self.modules[module.__name__][1]
        def bind(path,namespace,name):
            self.checks.binding(path,namespace,name)
            value=namespace.get(name)
            if type(value) is types.FunctionType:
                self.checks.add(path+'.interface',lambda value=value:(id(value.__code__),id(value.__globals__),id(value.__defaults__),id(value.__kwdefaults__)),[value.__code__],early=True)
            elif _is_type(value):
                for key in ('__new__','__init__','__call__'):
                    for base in type.__getattribute__(value,'__mro__'):
                        if key not in vars(base):continue
                        function=vars(base)[key]
                        self.checks.binding(path+'.'+key,vars(base),key)
                        if type(function) is types.FunctionType:
                            self.checks.add(path+'.'+key+'.code',lambda function=function:id(function.__code__),[function.__code__],early=True)
                        break
        def generated_method(cls,function):
            from ._planning_memory import MemoryBudget
            if cls is not MemoryBudget or type(cls) is not type or cls.__mro__!=(cls,object):
                raise QualificationDecline('unregistered infrastructure dataclass producer')
            fields=dataclasses.fields(cls)
            names=('host_peak_estimate_bytes','tree_limit_bytes','total_limit_bytes','free_fraction')
            annotations=(int,int,int,tuple)
            if tuple(field.name for field in fields)!=names:
                raise QualificationDecline('resource dataclass field schema changed')
            for field,annotation in zip(fields,annotations):
                if (type(field) is not dataclasses.Field or field.type is not annotation or
                        field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING or
                        field.init is not True or field.repr is not True or field.compare is not True or
                        field.hash is not None or field.kw_only is not False or dict(field.metadata) or field.doc is not None):
                    raise QualificationDecline('resource dataclass declaration changed')
            flags={name:item for name,_,item in _dataclass_slot_values(vars(cls)['__dataclass_params__'])}
            if flags!=dict(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,
                    match_args=True,kw_only=False,slots=False,weakref_slot=False):
                raise QualificationDecline('resource dataclass producer flags changed')
            if cls not in generated:
                pin(dataclasses)
                factory=dataclasses.make_dataclass
                expected=self.source.source(dataclasses,factory.__code__.co_filename,syntax=False)[3].get(('make_dataclass',factory.__code__.co_firstlineno))
                if factory.__globals__ is not vars(dataclasses) or expected is None or code_digest(factory.__code__)!=code_digest(expected):
                    raise QualificationDecline('dataclass generator interface changed')
                generated[cls]=dataclasses.make_dataclass(cls.__name__,list(zip(names,annotations)),
                    namespace={'__module__':cls.__module__,'__post_init__':vars(cls)['__post_init__']},frozen=True)
            shadow=generated[cls];expected=vars(shadow).get(function.__name__)
            if type(expected) is types.FunctionType and expected.__code__.co_filename!=function.__code__.co_filename:
                expected=vars(expected).get('__wrapped__')
            if type(expected) is not types.FunctionType or code_digest(function.__code__)!=code_digest(expected.__code__):
                raise QualificationDecline('generated resource method differs from declared producer')
            if function.__defaults__!=expected.__defaults__ or function.__kwdefaults__!=expected.__kwdefaults__:
                raise QualificationDecline('generated resource method defaults changed')
            cells=tuple(cell.cell_contents for cell in function.__closure__ or ())
            wanted=tuple(cls if cell.cell_contents is shadow else cell.cell_contents for cell in expected.__closure__ or ())
            if len(cells)!=len(wanted) or any(a is not b for a,b in zip(cells,wanted)):
                raise QualificationDecline('generated resource method closure changed')
        def put(path,value,cls=None):
            self.nodes+=1
            if self.nodes>80000:raise QualificationDecline('infrastructure binding bound exceeded')
            self.references.append(value);pending.append((path,value,cls))
        for module in modules:
            pin(module);namespace=vars(module)
            _,tree,_,codes=self.source.source(module,str(self.source.resolve(module.__file__)),syntax=True)
            declared={op.argval for op in dis.get_instructions(codes['<module>']) if op.opname in ('STORE_NAME','STORE_GLOBAL')}
            metadata={'__name__','__file__','__package__','__doc__','__builtins__','__loader__','__spec__','__cached__','__annotations__','__annotate__'}
            if set(namespace)-declared-metadata:raise QualificationDecline('undeclared infrastructure module state: '+module.__name__)
            for node in tree.body:
                if type(node) is not ast.Assign or len(node.targets)!=1 or type(node.targets[0]) is not ast.Name:continue
                try:literal=ast.literal_eval(node.value);expected_literal=passive_data(literal)
                except (ValueError,TypeError,QualificationDecline):continue
                if node.targets[0].id not in namespace or passive_data(namespace[node.targets[0].id])!=expected_literal:
                    raise QualificationDecline('infrastructure static declaration changed: '+node.targets[0].id)
            self.checks.add(module.__name__+'.keys',lambda namespace=namespace:tuple(sorted(namespace)))
            for name,value in sorted(namespace.items()):
                if name in ('__builtins__','__loader__','__spec__'):continue
                bind(module.__name__+'.'+name,namespace,name)
                if name=='_OWNER_ANCHOR' and module is _planning_native:continue # native.identity owns this exact adapter
                if is_package_module(value):pin(value);continue
                # Imported numerical interfaces are already deeply attested by
                # common/constructor/H graphs; retain this infrastructure edge.
                owner=getattr(value,'__module__',None) if type(value) is types.FunctionType or _is_type(value) else None
                if owner is not None and owner not in owned:continue
                if name.startswith('__') and name not in ('__annotations__','__annotate__'):continue
                put(module.__name__+'.'+name,value)
        while pending:
            path,value,cls=pending.pop();typ=type(value)
            if id(value) in seen:continue
            seen.add(id(value))
            if value is None or typ in (str,bytes,bool,int,float,complex):continue
            if typ is types.FunctionType:
                actual=sys.modules.get(value.__globals__.get('__name__'))
                if not is_package_module(actual) or vars(actual) is not value.__globals__:
                    raise QualificationDecline('infrastructure function global owner')
                pin(actual)
                filename=value.__code__.co_filename
                member=None if cls is None else vars(cls).get(value.__name__)
                owned_repr=(actual.__name__=='reprlib' and member is value and
                    type(vars(value).get('__wrapped__')) is types.FunctionType and
                    vars(value)['__wrapped__'].__globals__ is vars(owned[cls.__module__]))
                external=actual.__name__ not in owned and not owned_repr
                if external:
                    # Nested reset/interface tables can hold external Python
                    # functions too. Their implementation stops at the trusted
                    # provider interface, exactly as module-level imports do.
                    pass
                elif filename.startswith('<'):
                    # Only a method of a registered dataclass (or its exact
                    # repr wrapper body) confers generated-body ownership.
                    member=None if cls is None else vars(cls).get(value.__name__)
                    if (cls is None or '__dataclass_fields__' not in vars(cls) or
                            not (member is value or getattr(member,'__wrapped__',None) is value) or
                            value.__globals__ is not vars(owned[cls.__module__])):
                        raise QualificationDecline('unregistered infrastructure generated method: '+path+' / '+value.__qualname__+' / '+str(None if cls is None else cls.__qualname__))
                    generated_method(cls,value)
                else:
                    _,_,_,codes=self.source.source(actual,filename,syntax=False)
                    expected=codes.get((value.__code__.co_qualname,value.__code__.co_firstlineno))
                    if expected is None or self.source.digest(expected)!=self.source.digest(value.__code__):
                        raise QualificationDecline('infrastructure live body differs from source: '+path)
                self.rows.append((path,'function',code_digest(value.__code__)))
                self.checks.add(path,lambda value=value:(id(value.__code__),id(value.__globals__),id(value.__defaults__),
                    id(value.__kwdefaults__),id(value.__closure__),tuple(self.checks.atom(cell.cell_contents) for cell in value.__closure__ or ())),[value.__code__],early=True)
                self.checks.mapping(path+'.attributes',vars(value))
                if not external:
                    put(path+'.defaults',value.__defaults__,cls);put(path+'.kwdefaults',value.__kwdefaults__,cls)
                    for index,cell in enumerate(value.__closure__ or ()):put(path+'.cell'+str(index),cell.cell_contents,cls)
                if actual.__name__ in owned:
                    todo=[value.__code__]
                    while todo:
                        code=todo.pop();todo.extend(item for item in code.co_consts if type(item) is types.CodeType)
                        ops=list(dis.get_instructions(code))
                        for index,op in enumerate(ops):
                            if op.opname not in ('LOAD_GLOBAL','LOAD_NAME'):continue
                            namespace=value.__globals__ if op.argval in value.__globals__ else value.__builtins__
                            if op.argval in namespace:
                                bind(path+'.global:'+op.argval,namespace,op.argval)
                                current=namespace[op.argval]
                                for following in ops[index+1:]:
                                    if following.opname not in ('LOAD_ATTR','LOAD_METHOD') or not is_package_module(current):break
                                    if following.argval not in vars(current):raise QualificationDecline('infrastructure imported interface absent')
                                    pin(current);bind(path+'.interface:'+current.__name__+'.'+following.argval,vars(current),following.argval)
                                    current=vars(current)[following.argval]

            elif _is_type(value):
                if value.__module__ not in owned:continue
                if vars(owned[value.__module__]).get(value.__name__) is not value:
                    raise QualificationDecline('infrastructure class owner')
                if type(value) is not type(ctypes.Structure):
                    for base in type.__getattribute__(value,'__mro__'):_prepare_slot_metadata(base,self.source,initial=True)
                self.checks.add(path,lambda value=value:(tuple(map(id,type.__getattribute__(value,'__mro__'))),
                    tuple(sorted((name,self.checks.atom(item)) for name,item in vars(value).items()))),list(vars(value).values()),early=True)
                for name,item in vars(value).items():
                    if name not in ('__dict__','__weakref__'):put(path+'.'+name,item,value)
            elif typ in (dict,types.MappingProxyType):
                self.checks.mapping(path,value)
                for name,item in value.items():put(path+'.key',name,cls);put(path+'.value',item,cls)
            elif typ in (tuple,list,set,frozenset):
                if typ in (tuple,list):self.checks.sequence(path,lambda value=value:tuple(value))
                else:self.checks.add(path,lambda value=value:tuple(sorted(self.checks.atom(item) for item in value)),tuple(value))
                for item in value:put(path+'.item',item,cls)
            elif typ in (staticmethod,classmethod):put(path+'.function',value.__func__,cls)
            elif typ is property:
                for name in ('fget','fset','fdel'):put(path+'.'+name,getattr(value,name),cls)
            elif typ in (dataclasses.Field,dataclasses._DataclassParams):
                self.checks.add(path,lambda value=value:tuple((name,id(descriptor),self.checks.atom(item)) for name,descriptor,item in _dataclass_slot_values(value)))
                for name,descriptor,item in _dataclass_slot_values(value):put(path+'.'+name,item,cls)
            elif is_package_module(value):pin(value)
            elif typ is ctypes.CField and ctypes.CField is vars(sys.modules['_ctypes']).get('CField'):
                pin(sys.modules['ctypes']);pin(sys.modules['_ctypes'])
            else:
                # Finite imported native/provider objects (ctypes field/type,
                # dataclass sentinels and typing metadata) are interface values.
                provider=sys.modules.get(typ.__module__)
                if provider is None or typ.__module__ not in ('builtins','_ctypes','_thread','types','typing','dataclasses','__future__','contextvars','_contextvars'):
                    raise QualificationDecline('unknown infrastructure state: '+path+' / '+type.__getattribute__(typ,'__module__')+'.'+type.__getattribute__(typ,'__qualname__'))
                pin(provider)
        self.native=_planning_native.identity()
        self.available_memory=None
        if role=='host':
            from . import _planning_memory as memory
            self.available_memory=memory.available_memory_identity()
            # DLL attribute resolution is the imported API boundary, not an
            # invitation to traverse ctypes implementation state.
            bind('available-memory.provider-getattr',vars(ctypes.CDLL),'__getattr__')
            bind('available-memory.loader-getattr',vars(ctypes.LibraryLoader),'__getattr__')
            bind('available-memory.library',vars(ctypes.windll),'kernel32')
            kernel=memory._memory_kernel()
            for name in ('_handle','_name'):self.checks.binding('available-memory.library.'+name,vars(kernel),name)
            for name in ('_AvailablePrototype','_ResolvePrototype','_ContainingPrototype','_FilenamePrototype','_DirectoryPrototype'):
                prototype=getattr(memory,name)
                self.checks.add('available-memory.signature.'+name,
                    lambda prototype=prototype:tuple(sorted((key,self.checks.atom(value)) for key,value in vars(prototype).items())),
                    list(vars(prototype).values()),early=True)
        self.wire=dict(role=role,nodes=self.nodes,files={name:row[1] for name,row in sorted(self.modules.items())},
            definitions=self.rows,native=self.native,available_memory=self.available_memory)
        self.digest=hashlib.sha256(passive_data(self.wire)).hexdigest()
        self.source.verify_paths()
        self.parsing_release=self.source.release_parsing()

    def verify(self):
        import sys
        from ._planning_qualification import pin_module,runtime_files
        from ._planning_executor import PlannerSourceChanged
        from . import _planning_native
        try:
            # Refuse interface replacements before runtime enumeration can call
            # any of them. These checks only read retained original bindings.
            self.checks.verify()
            runtime=runtime_files()
            for name,(module,expected,filename,spec,kind,observer) in self.modules.items():
                if type(module) is not kind:raise QualificationDecline('infrastructure module class changed: '+name)
                if observer is not None:observer.verify()
                if (sys.modules.get(name) is not module or vars(module).get('__file__')!=filename or vars(module).get('__spec__') is not spec):
                    raise QualificationDecline('infrastructure module binding changed: '+name)
                if pin_module(module,runtime=runtime,package_only=True,observer=observer)!=expected:raise QualificationDecline('infrastructure source/native image changed: '+name)
            self.checks.verify()
            if _planning_native.identity()!=self.native:raise QualificationDecline('infrastructure resource helper changed')
            if self.available_memory is not None:
                from ._planning_memory import available_memory_identity
                if available_memory_identity()!=self.available_memory:
                    raise QualificationDecline('available memory provider image changed')
        except Exception as error:raise PlannerSourceChanged('infrastructure changed: '+str(error)) from error
