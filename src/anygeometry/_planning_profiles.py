"""Guarded finite branch profiles; no profile alone admits a whole workload.

New source pins qualify these precise branch arguments. They do not refresh any
legacy guarded pin, mutate kernels, or exempt reachable external mutable stores.
"""
import ast
import builtins
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import types
import numpy as np
from ._planning_code import code_digest
from ._planning_qualification import QualificationDecline

_SOURCE_PINS = {
  "anygeometry.batch_intersections": "fe108aba93c77a6a2cd1d3851181417447216c6a7f9e7b7726e6c611c38b5079",
  "anygeometry.material_arrangement": "d9ad70f498dec1c23e483a83e0769faacc0fe5ef8b2924aa0ec660c878f4a8f4",
  "anygeometry.nurbs_sweep_chart": "b89ff25198bab5cab66a3b0b9b8cc91fa7544f548d52a47cf9da06ef770c6f8a",
  "anygeometry.extruded_supports": "8fa07e24e0a87c60b93e40ab3186b6f49686624a276de6f189586846b15f249c",
  "anygeometry.intersections": "2588de65b3741ca6332259112bb0bd383183dd6b3aebe3cee6d5d2a7ebca58df",
  "anygeometry.serialization": "641c0ebbc2c27d4e22ad99eec51617e3d9c2e85040b54d6f883a8e38ceeb3154",
  "anygeometry._planning_stages": "022ad5a3ac0cd2b42e3db093f184b4d035b909789703667291597c6d5c7a4495",
  "json": "0439e6546415d4147396afac324beadb7a8dc2c1e2391dcb541fb34c83986d8f",
  "json.encoder": "498ba55bf9907208e58426f98dd72e2756417ab5478b141284abd373863e621c"
}


def _plain(value):
    typ=type(value)
    if value is None or typ in (str,int,bool):return (typ.__name__,value)
    if typ is float:return ('float',struct.pack('>d',value).hex())
    if typ in (tuple,list):return (typ.__name__,tuple(_plain(item) for item in value))
    if typ is dict and all(type(key) is str for key in value):
        return ('dict',tuple((key,_plain(item)) for key,item in sorted(value.items())))
    raise QualificationDecline('branch profile default is not closed passive data')


def _codes(code):
    yield code
    for part in code.co_consts:
        if type(part) is types.CodeType:yield from _codes(part)


def _resolve(module,path):
    value=module
    for name in path.split('.'):
        namespace=vars(value) if type(value) in (types.ModuleType,type) else None
        if namespace is None or name not in namespace:raise QualificationDecline('branch symbol missing')
        value=namespace[name]
        if type(value) in (classmethod,staticmethod):value=value.__func__
    return value


class BranchProfiles:
    """Bind source and actual loaded selectors before checking declarative data."""
    def __init__(self):
        from . import batch_intersections as batch,material_arrangement as material
        from . import nurbs_sweep_chart as charts,extruded_supports as extruded,intersections
        from . import serialization,_planning_stages,entities,model,surfaces,nurbs
        from ._planning_ledger import runtime_infrastructure
        self.infrastructure,self.infrastructure_pin=runtime_infrastructure()
        self.model_type=model.GeometryModel;self.face_type=entities.Face;self.domain_type=material.MaterialDomain
        self.supports=(surfaces.Plane,surfaces.Cylinder)
        predicate_bindings={name:vars(surfaces)[name] for name in
            ('Plane','Cylinder','Cone','ExtrudedSurface','CoonsSurface')}
        predicate_bindings.update(NurbsSurface=nurbs.NurbsSurface,GeometryModel=model.GeometryModel,
                                  MaterialDomain=material.MaterialDomain,Face=entities.Face,JSONEncoder=json.encoder.JSONEncoder)
        predicate_bindings.update({name:vars(builtins)[name] for name in
                                   ('isinstance','type','len','object','tuple','dict','float','str')})
        self.sources={};self.functions=[];self.namespaces=[]
        selected=((batch,('_pair_paths','_pair_paths_exact')),
            (material,('MaterialDomain.from_model',)),
            (charts,('has_sweep_chart','source_sweep_chart_model')),
            (extruded,('extruded_support',)),(intersections,('_qualified_face_plane',)),
            (serialization,('_canonical_payload',)),(_planning_stages,('stage_task',)),
            (json,('dumps',)),(json.encoder,('JSONEncoder.__init__','JSONEncoder.encode','JSONEncoder.iterencode','JSONEncoder.default')))
        for module,paths in selected:
            filename=str(Path(module.__file__).resolve());source=Path(filename).read_bytes()
            sha=hashlib.sha256(source).hexdigest()
            if _SOURCE_PINS.get(module.__name__)!=sha:raise QualificationDecline('branch profile source not registered: '+module.__name__)
            self.sources[module.__name__]=(module,filename,sha)
            compiled=compile(source,filename,'exec',dont_inherit=True)
            expected={code.co_qualname:code for code in _codes(compiled)}
            tree=ast.parse(source,filename)
            definitions={node.lineno:node for node in ast.walk(tree) if type(node) in (ast.FunctionDef,ast.AsyncFunctionDef)}
            for path in paths:
                function=_resolve(module,path)
                if type(function) is not types.FunctionType or function.__globals__ is not vars(module):
                    raise QualificationDecline('branch function is not exact owner Python code')
                original=expected.get(path)
                if original is None or code_digest(original)!=code_digest(function.__code__):
                    raise QualificationDecline('branch loaded code differs from pinned source: '+path)
                node=definitions.get(original.co_firstlineno)
                if node is None:
                    node=next((candidate for candidate in definitions.values() if
                        candidate.decorator_list and candidate.decorator_list[0].lineno==original.co_firstlineno),None)
                if node is None:raise QualificationDecline('branch definition source mapping unavailable')
                try:
                    positional=tuple(ast.literal_eval(item) for item in node.args.defaults) or None
                    keyword={arg.arg:ast.literal_eval(item) for arg,item in zip(node.args.kwonlyargs,node.args.kw_defaults) if item is not None} or None
                except (ValueError,TypeError):raise QualificationDecline('branch default expression not registered')
                if _plain(function.__defaults__)!=_plain(positional) or _plain(function.__kwdefaults__)!=_plain(keyword):
                    raise QualificationDecline('branch defaults differ from pinned source')
                if function.__closure__:raise QualificationDecline('unexpected branch closure')
                bindings=tuple((name,vars(module)[name]) for name in sorted(set(function.__code__.co_names)) if name in vars(module))
                if any(name in predicate_bindings and bound is not predicate_bindings[name] for name,bound in bindings):
                    raise QualificationDecline('initial branch predicate alias/builtin shadow')
                self.functions.append((module,path,function,function.__code__,_plain(positional),_plain(keyword),bindings))
        # Exact selected type/metaclass/namespace bindings close the predicates.
        for module,names in ((model,('GeometryModel',)),(entities,('Face',)),(material,('MaterialDomain',)),
                (surfaces,('Plane','Cylinder','Cone','ExtrudedSurface','CoonsSurface')),(nurbs,('NurbsSurface',)),
                (json.encoder,('JSONEncoder',)),(np,('ndarray','float64'))):
            for name in names:
                cls=vars(module)[name]
                if type(cls) is not type or cls.__module__!=module.__name__ or cls.__name__!=name:
                    raise QualificationDecline('custom branch type metaclass/owner')
                if module is surfaces and cls.__bases__!=(object,):raise QualificationDecline('unsupported surface predicate bases')
                self.namespaces.append((module,name,cls,tuple(vars(cls).items()),cls.__mro__))
        for cls,names in ((self.face_type,('surface','loop','holes')),(self.domain_type,('support',))):
            for name in names:
                descriptor=vars(cls).get(name)
                if type(descriptor) is not types.MemberDescriptorType or descriptor.__objclass__ is not cls:
                    raise QualificationDecline('branch field is not an exact native slot')
        self.native=(builtins.type,builtins.isinstance,builtins.len,json.encoder.c_make_encoder,
                     json.encoder.encode_basestring,json.encoder.encode_basestring_ascii)
        if self.native[3] is None:raise QualificationDecline('branch JSON native encoder absent')
        if (json.JSONEncoder.__new__ is not object.__new__ or json.JSONEncoder.__getattribute__ is not object.__getattribute__ or
                json.JSONEncoder.__setattr__ is not object.__setattr__ or '__getattr__' in vars(json.JSONEncoder)):
            raise QualificationDecline('JSON encoder construction/access override')
        for name in ('skipkeys','ensure_ascii','check_circular','allow_nan','sort_keys','indent'):
            if name in vars(json.JSONEncoder):raise QualificationDecline('JSON encoder option descriptor override')
        for name,expected in (('item_separator',', '),('key_separator',': ')):
            if type(vars(json.JSONEncoder).get(name)) is not str or vars(json.JSONEncoder)[name]!=expected:
                raise QualificationDecline('JSON encoder separator descriptor override')
        descriptor=vars(self.model_type).get('__dict__')
        if (type(descriptor) is not types.GetSetDescriptorType or descriptor.__objclass__ is not self.model_type or
                self.model_type.__getattribute__ is not object.__getattribute__ or
                any(name in vars(self.model_type) for name in ('faces','_faces'))):
            raise QualificationDecline('branch model instance namespace override')
        for name in ('dtype','size'):
            descriptor=vars(np.ndarray).get(name)
            if type(descriptor) is not types.GetSetDescriptorType or descriptor.__objclass__ is not np.ndarray:
                raise QualificationDecline('branch ndarray native descriptor mismatch')

    def verify(self):
        from ._planning_executor import PlannerSourceChanged
        from ._planning_ledger import runtime_infrastructure
        try:
            handle,pin=runtime_infrastructure()
            if handle is not self.infrastructure or pin!=self.infrastructure_pin:raise QualificationDecline('branch anchor drift')
            for name,(module,filename,sha) in self.sources.items():
                if sys.modules.get(name) is not module or hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=sha:
                    raise QualificationDecline('branch source/module drift')
            for module,path,function,code,defaults,kwargs,bindings in self.functions:
                if (_resolve(module,path) is not function or function.__code__ is not code or
                        _plain(function.__defaults__)!=defaults or _plain(function.__kwdefaults__)!=kwargs or
                        function.__closure__ or any(vars(module).get(name) is not value for name,value in bindings)):
                    raise QualificationDecline('branch loaded function/default/binding drift')
            for module,name,cls,namespace,mro in self.namespaces:
                current=vars(cls)
                if vars(module).get(name) is not cls or cls.__mro__ is not mro or len(current)!=len(namespace) or any(current.get(key) is not value for key,value in namespace):
                    raise QualificationDecline('branch type/descriptor drift')
            observed=(builtins.type,builtins.isinstance,builtins.len,json.encoder.c_make_encoder,
                      json.encoder.encode_basestring,json.encoder.encode_basestring_ascii)
            if any(left is not right for left,right in zip(self.native,observed)):raise QualificationDecline('branch native binding drift')
        except Exception as error:raise PlannerSourceChanged('branch profile changed: '+str(error)) from error

    def face(self,model,face_id):
        self.verify()
        if type(model) is not self.model_type or type(face_id) is not int:raise QualificationDecline('branch model/id type')
        namespace=types.GetSetDescriptorType.__get__(vars(self.model_type)['__dict__'],model,self.model_type)
        faces=namespace.get('faces')
        if type(faces) is not types.MappingProxyType:raise QualificationDecline('branch face namespace type')
        backing=gc.get_referents(faces)
        if len(backing)!=1 or type(backing[0]) is not dict or namespace.get('_faces') is not backing[0]:
            raise QualificationDecline('branch face mapping backing is not owner plain dict')
        face=backing[0][face_id]
        if type(face) is not self.face_type:raise QualificationDecline('branch face record type')
        support=object.__getattribute__(face,'surface')
        if type(support) not in self.supports:raise QualificationDecline('unsupported declarative face support')
        if type(object.__getattribute__(face,'holes')) is not tuple or type(object.__getattribute__(face,'loop')) is not tuple:
            raise QualificationDecline('branch boundary collection type')
        return 'unchanged-plane-cylinder-support'

    def capture(self,function,*args,**kwargs):
        """Connect only completed selector proofs to the callable manifest.

        Closed JSON encoding is admitted here; norm/pair predicates still need
        their additional executable dependency contracts before exclusion.
        """
        return _GuardedSelector(self,function,args,kwargs)

    def pair(self,first,second,*,route,shadow):
        self.verify()
        if route is not None or shadow is not None:raise QualificationDecline('unsupported pair route/shadow')
        for domain in (first,second):
            if type(domain) is not self.domain_type or type(object.__getattribute__(domain,'support')) not in self.supports:
                raise QualificationDecline('unsupported actual material support')
        return 'plane-cylinder-pair'

    def norm(self,value,kwargs):
        self.verify()
        if type(kwargs) is not dict or any(type(key) is not str or key not in ('ord','axis','keepdims') for key in kwargs):raise QualificationDecline('norm kwargs')
        if kwargs.get('ord') is not None or kwargs.get('axis') is not None or kwargs.get('keepdims',False) is not False:
            raise QualificationDecline('unsupported norm branch')
        if type(value) is not np.ndarray or value.dtype.type is not np.float64 or not value.dtype.isnative or value.size>100000:
            raise QualificationDecline('norm exact real array domain')
        if any(not math.isfinite(part[0]) for part in struct.iter_unpack('=d',value.tobytes())):
            raise QualificationDecline('norm nonfinite input')
        return 'norm-default-real'

    def json(self,value,kwargs):
        self.verify()
        if type(kwargs) is not dict or any(type(key) is not str for key in kwargs):raise QualificationDecline('JSON kwargs schema')
        effective=dict(skipkeys=False,ensure_ascii=True,check_circular=True,allow_nan=True,
            cls=None,indent=None,separators=None,default=None,sort_keys=False)
        if any(key not in effective for key in kwargs):raise QualificationDecline('JSON unknown keyword')
        effective.update(kwargs)
        expected=dict(skipkeys=False,ensure_ascii=effective['ensure_ascii'],check_circular=True,allow_nan=False,
            cls=None,indent=None,separators=(',',':'),default=None,sort_keys=True)
        if type(effective['ensure_ascii']) is not bool or _plain(effective)!=_plain(expected):raise QualificationDecline('JSON unsupported options')
        def string_bound(text):
            total=2
            for character in text:
                point=ord(character)
                if point<32:total+=6
                elif character in ('"','\\'):total+=2
                elif effective['ensure_ascii']:
                    total+=12 if point>0xffff else 6 if point>127 else 1
                else:
                    if 0xd800<=point<=0xdfff:raise QualificationDecline('JSON UTF8 surrogate outside profile')
                    total+=1 if point<128 else 2 if point<2048 else 3 if point<65536 else 4
            return total
        queue=[(value,0)];count=0;byte_bound=0
        while queue:
            item,depth=queue.pop();count+=1
            if count>100000 or depth>64:raise QualificationDecline('JSON tree bound/cycle')
            typ=type(item)
            if item is None or typ is bool:byte_bound+=5
            elif typ is str:byte_bound+=string_bound(item)
            elif typ is int:
                if item.bit_length()>2048:raise QualificationDecline('JSON integer conversion context outside profile')
                byte_bound+=item.bit_length()+2
            elif typ is float and math.isfinite(item):byte_bound+=32
            elif typ in (list,tuple):
                byte_bound+=2+len(item);queue.extend((part,depth+1) for part in item)
            elif typ is dict and all(type(key) is str for key in item):
                byte_bound+=2+2*len(item)+sum(string_bound(key) for key in item)
                queue.extend((part,depth+1) for part in item.values())
            else:raise QualificationDecline('JSON tree is not closed passive data')
            if byte_bound>4_000_000:raise QualificationDecline('JSON guarded payload bound')
        return 'json-stage' if effective['ensure_ascii'] else 'json-canonical'


class _GuardedSelector:
    def __init__(self,profile,function,args,kwargs):
        from ._planning_manifest import CallableManifest
        self.profile=profile;self.function=function;self.args=args;self.kwargs=kwargs
        self.validate()
        self.manifest=CallableManifest((function,),branch_profile=self)

    def validate(self):
        self.profile.verify()
        if self.function is json.dumps:
            if len(self.args)!=1:raise QualificationDecline('JSON operation requires one exact tree')
            self.profile.json(self.args[0],self.kwargs)
            return
        allowed={('anygeometry.nurbs_sweep_chart','has_sweep_chart'),
                 ('anygeometry.extruded_supports','extruded_support'),
                 ('anygeometry.intersections','_qualified_face_plane')}
        key=(self.function.__module__,self.function.__qualname__)
        if key not in allowed or self.kwargs:raise QualificationDecline('selector branch has no completed call contract')
        if not any(module.__name__==key[0] and path==key[1] and function is self.function
                   for module,path,function,*rest in self.profile.functions):
            raise QualificationDecline('selector is not the registered bound function')
        if key[1]=='has_sweep_chart':
            if len(self.args)!=1 or type(self.args[0]) is not self.profile.face_type:
                raise QualificationDecline('selector exact face required')
            support=object.__getattribute__(self.args[0],'surface')
            if type(support) not in self.profile.supports:raise QualificationDecline('selector unsupported actual support')
        else:
            if len(self.args)!=2:raise QualificationDecline('selector exact model/id required')
            self.profile.face(*self.args)
            if key[1]=='_qualified_face_plane':
                face=object.__getattribute__(self.args[0],'faces')[self.args[1]]
                if type(object.__getattribute__(face,'surface')) is not self.profile.supports[0]:
                    raise QualificationDecline('plane selector requires exact Plane')

    def dependencies(self,function):
        # Full selector code/source/defaults and all live names are bound by the
        # profile. For these exact types the body returns before any kernel call.
        return () if function is self.function else None

    def wire(self):
        return dict(kind='guarded-selector-v1',function=[self.function.__module__,self.function.__qualname__],
                    sources={name:row[2] for name,row in sorted(self.profile.sources.items())},
                    infrastructure=self.profile.infrastructure_pin)

    def __call__(self):
        self.manifest.verify()
        try:result=self.function(*self.args,**self.kwargs)
        finally:self.manifest.verify()
        return result
