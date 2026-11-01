"""Closed PP5 stage value codec. Constructors and checked derived values only.

Registry is the sector60 real-stage census plus point/task input records. Unknown
records, nonempty instance caches, cycles, dtype adapters and extra state refuse.
No pickle, import-by-name, arbitrary callable, or lazy cache values cross the wire.
"""
import hashlib
import math
import struct
import sys
from dataclasses import fields,is_dataclass
import numpy as np
from ._planning_qualification import QualificationDecline
from ._planning_constructor import audit
from .surfaces import Plane,Cylinder
from .arrangement_geometry import LinePath
from .material_arrangement import ArrangementPath,ArrangementPoint,ArrangementCell,MaterialArrangement
from .exact_curves import EllipticArc,CylinderIntersectionCurve,_CylinderSupport
from .branch_curves import BezierQuadricCurve
from .quadric_algebra import QuadricSupport
from .plane_ruled import PlaneRuledSupport


def registry():
    return (Plane,Cylinder,LinePath,ArrangementPath,ArrangementPoint,ArrangementCell,
            MaterialArrangement,EllipticArc,CylinderIntersectionCurve,_CylinderSupport,
            BezierQuadricCurve,QuadricSupport,PlaneRuledSupport)


def same(a,b):
    if type(a) is not type(b):return False
    if type(a) in (np.float64,np.int64,np.bool_):return a.tobytes()==b.tobytes()
    if type(a) is float:return struct.pack('!d',a)==struct.pack('!d',b)
    if type(a) is np.ndarray:
        return a.dtype==b.dtype and a.shape==b.shape and a.flags.c_contiguous==b.flags.c_contiguous and a.tobytes()==b.tobytes()
    if type(a) in (tuple,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if type(a) in (str,int,bool,type(None)):return a==b
    if type(a) in registry():return all(same(getattr(a,f.name),getattr(b,f.name)) for f in fields(a) if f.init)
    return False


def cache_count(record,name):
    value=getattr(record,name)
    return sys.getrefcount(value)


def array_manifest(value):
    if type(value) is not np.ndarray or value.dtype!=np.dtype(float) or not value.flags.c_contiguous or not np.all(np.isfinite(value)):
        raise QualificationDecline('stage derived array schema')
    return [value.dtype.str,list(value.shape),hashlib.sha256(value.tobytes()).hexdigest(),bool(value.flags.writeable)]


def record_manifest(value,fresh):
    t=type(value);declared=fields(value);extra=set(getattr(value,'__dict__',{}))-{f.name for f in declared}
    allowed={'_circumferential'} if t is Cylinder else ({'normal'} if t is Plane and 'normal' in extra else set())
    if extra!=allowed:raise QualificationDecline('stage unknown record extra')
    for f in declared:
        if not f.init:
            old,new=getattr(value,f.name),getattr(fresh,f.name)
            if not (old is new is None or type(old) is type(new) is dict and len(old)==len(new)==0):
                raise QualificationDecline('stage nonempty/unknown instance cache')
            if type(old) is dict and cache_count(value,f.name)!=cache_count(fresh,f.name):
                raise QualificationDecline('stage aliased instance cache')
    if t is Cylinder:
        audit({'record':value})
    manifest=[]
    for name in sorted(extra):
        if not same(getattr(value,name),getattr(fresh,name)):
            raise QualificationDecline('stage derived recomputation mismatch')
        manifest.append([name,array_manifest(getattr(value,name))])
    return manifest


def encode_value(value,*,limit=100000):
    seen={};active=set();nodes=0;arrays=[]
    def encode(v):
        nonlocal nodes
        nodes+=1
        if nodes>limit:raise QualificationDecline('stage value node limit='+str(limit))
        t=type(v)
        if v is None:return ['none']
        if t in (str,int,bool):return [t.__name__,v]
        if t in (np.float64,np.int64,np.bool_):
            if t is np.float64 and not np.isfinite(v):raise QualificationDecline('stage nonfinite scalar')
            return ['scalar',v.dtype.str,v.tobytes().hex()]
        if t is float:
            if not math.isfinite(v):raise QualificationDecline('stage nonfinite float')
            return ['float',struct.pack('!d',v).hex()]
        if t not in (tuple,list,dict,np.ndarray) and t not in registry():raise QualificationDecline('stage unsupported type: '+t.__name__)
        if id(v) in active:raise QualificationDecline('stage cyclic value')
        if id(v) in seen:return ['ref',seen[id(v)]]
        ordinal=len(seen);seen[id(v)]=ordinal;active.add(id(v))
        try:
            if t in (tuple,list):return [t.__name__,ordinal,[encode(x) for x in v]]
            if t is dict:return ['dict',ordinal,[[encode(k),encode(x)] for k,x in v.items()]]
            if t is np.ndarray:
                if v.dtype!=np.dtype(float) or not v.flags.c_contiguous or not np.all(np.isfinite(v)):
                    raise QualificationDecline('stage array schema')
                if any(np.shares_memory(v,old,max_work=1000) for old in arrays):
                    raise QualificationDecline('stage distinct-array alias')
                arrays.append(v)
                return ['array',ordinal,v.dtype.str,list(v.shape),v.tobytes().hex(),bool(v.flags.writeable)]
            declaration=fields(v)
            args=[[f.name,encode(getattr(v,f.name))] for f in declaration if f.init]
            # All nested values qualified before executing the exact constructor.
            fresh=t(**{f.name:getattr(v,f.name) for f in declaration if f.init})
            if not same(v,fresh):raise QualificationDecline('stage constructor changed fields')
            return ['record',ordinal,t.__name__,args,record_manifest(v,fresh)]
        finally:active.remove(id(v))
    try:return encode(value)
    except QualificationDecline:raise
    except Exception as exc:raise QualificationDecline('stage encode refused: '+type(exc).__name__) from exc


def decode_value(packet,*,limit=100000):
    memo=[];nodes=0;classes={t.__name__:t for t in registry()}
    def decode(p):
        nonlocal nodes
        nodes+=1
        if nodes>limit:raise QualificationDecline('stage value node limit='+str(limit))
        if type(p) is not list or not p or type(p[0]) is not str:raise QualificationDecline('stage wire schema')
        tag=p[0]
        if tag=='none' and len(p)==1:return None
        if tag in ('str','int','bool') and len(p)==2:
            typ={'str':str,'int':int,'bool':bool}[tag]
            if type(p[1]) is not typ:raise QualificationDecline('stage primitive type')
            return p[1]
        if tag=='float' and len(p)==2:
            value=struct.unpack('!d',bytes.fromhex(p[1]))[0]
            if not math.isfinite(value):raise QualificationDecline('stage nonfinite float')
            return value
        if tag=='scalar' and len(p)==3:
            dtypes={np.dtype(t).str:np.dtype(t) for t in (np.float64,np.int64,np.bool_)}
            if type(p[1]) is not str or p[1] not in dtypes:raise QualificationDecline('stage scalar dtype')
            raw=bytes.fromhex(p[2]);dtype=dtypes[p[1]]
            if len(raw)!=dtype.itemsize:raise QualificationDecline('stage scalar size')
            value=np.frombuffer(raw,dtype=dtype)[0]
            if type(value) is np.float64 and not np.isfinite(value):raise QualificationDecline('stage nonfinite scalar')
            return value
        if tag=='ref' and len(p)==2:
            if type(p[1]) is not int or not 0<=p[1]<len(memo) or memo[p[1]] is None:raise QualificationDecline('stage invalid reference')
            return memo[p[1]]
        if len(p)<3 or type(p[1]) is not int or p[1]!=len(memo):raise QualificationDecline('stage object ordinal')
        ordinal=len(memo);memo.append(None)
        if tag in ('tuple','list') and len(p)==3 and type(p[2]) is list:
            items=[decode(x) for x in p[2]];value=tuple(items) if tag=='tuple' else items
        elif tag=='dict' and len(p)==3 and type(p[2]) is list:
            value={}
            for key,item in p[2]:
                key=decode(key)
                if key in value:raise QualificationDecline('stage duplicate dict key')
                value[key]=decode(item)
        elif tag=='array' and len(p)==6:
            if p[2]!=np.dtype(float).str or type(p[3]) is not list or any(type(n) is not int or n<0 for n in p[3]) or type(p[5]) is not bool:raise QualificationDecline('stage array schema')
            value=np.frombuffer(bytes.fromhex(p[4]),dtype=float).copy().reshape(p[3])
            if not np.all(np.isfinite(value)):raise QualificationDecline('stage nonfinite array')
            value.flags.writeable=p[5]
        elif tag=='record' and len(p)==5 and type(p[2]) is str and p[2] in classes:
            t=classes[p[2]];names=[f.name for f in fields(t) if f.init]
            if type(p[3]) is not list or [row[0] for row in p[3]]!=names:raise QualificationDecline('stage record fields')
            value=t(**{key:decode(item) for key,item in p[3]})
            if type(p[4]) is not list:raise QualificationDecline('stage derived manifest')
            allowed=['_circumferential'] if t is Cylinder else ['normal'] if t is Plane and p[4] else []
            if [row[0] for row in p[4]]!=allowed:raise QualificationDecline('stage derived coverage')
            for name,manifest in p[4]:
                current=array_manifest(getattr(value,name))
                if (type(manifest) is not list or len(manifest)!=4 or type(manifest[3]) is not bool or
                    type(manifest[1]) is not list or any(type(n) is not int for n in manifest[1]) or current[:3]!=manifest[:3]):raise QualificationDecline('stage derived binding')
                getattr(value,name).flags.writeable=manifest[3]
        else:raise QualificationDecline('stage unknown wire tag')
        memo[ordinal]=value;return value
    try:
        value=decode(packet)
        if encode_value(value,limit=limit)!=packet:raise QualificationDecline('stage noncanonical reconstruction')
        return value
    except QualificationDecline:raise
    except Exception as exc:raise QualificationDecline('stage decode refused: '+type(exc).__name__) from exc
