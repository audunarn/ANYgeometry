"""Closed PP5 stage value codec. Constructors and checked derived values only.

Registry is the sector60 real-stage census plus point/task input records. Unknown
records, nonempty instance caches, cycles, dtype adapters and extra state refuse.
No pickle, import-by-name, arbitrary callable, or lazy cache values cross the wire.
"""
import base64
import hashlib
import json
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


def record_manifest(value,fresh,*,_audits=None):
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
        if _audits is None:audit({'record':value})
        else:_audits[id(value)]=value
    manifest=[]
    for name in sorted(extra):
        if not same(getattr(value,name),getattr(fresh,name)):
            raise QualificationDecline('stage derived recomputation mismatch')
        manifest.append([name,array_manifest(getattr(value,name))])
    return manifest


def encode_value(value,*,limit=100000,codec=None,payload_limit=4_000_000):
    _value_codec(codec)
    packet=_encode_value(value,limit=limit)
    return packet if codec is None else compact_value_v2(packet,limit=limit,payload_limit=payload_limit)


def _encode_value(value,*,limit,_audits=None):
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
            return ['record',ordinal,t.__name__,args,record_manifest(v,fresh,_audits=_audits)]
        finally:active.remove(id(v))
    try:return encode(value)
    except QualificationDecline:raise
    except Exception as exc:raise QualificationDecline('stage encode refused: '+type(exc).__name__) from exc


def comparison_packet(packet,*,limit=100000,codec=None,payload_limit=4_000_000):
    """Compare canonical wire values independently of readonly Plane cache presence.

    Decode first validates the complete original packet. Only private decoded
    copies gain absent normal caches; existing flags/bytes and every declared
    field/alias remain in the re-encoded packet. Caller values are untouched.
    """
    value=decode_value(packet,limit=limit,codec=codec,payload_limit=payload_limit);pending=[value];seen=set()
    while pending:
        current=pending.pop();typ=type(current)
        if id(current) in seen:continue
        seen.add(id(current))
        if typ is Plane and 'normal' not in vars(current):_=current.normal
        if typ in (tuple,list):pending.extend(current)
        elif typ is dict:
            for key,item in current.items():pending.extend((key,item))
        elif typ in registry():
            pending.extend(getattr(current,field.name) for field in fields(current) if field.init)
    return encode_value(value,limit=limit)


def decode_value(packet,*,limit=100000,codec=None,payload_limit=4_000_000):
    _value_codec(codec)
    return _decode_versioned(packet,limit=limit,codec=codec,payload_limit=payload_limit)


def decode_stage_values(packets,*,limit=100000,payload_limit=4_000_000,codec=None):
    """Closed H production transaction; no decoded values escape before audit.

    Every wire string is retained independently of the caller's list before any
    constructor runs. Per-value canonical checks are unchanged. Only Cylinder
    owner audits are deferred, to one fresh census after all constructors. The
    aggregate constructor walk retains its 100000-node refusal bound. No state
    or census survives the call, including a failed reconstruction.
    """
    _value_codec(codec)
    if type(packets) not in (tuple,list) or len(packets)>100000:
        raise QualificationDecline('stage result sequence schema')
    snapshot=tuple(packets)
    try:
        size=0
        for packet in snapshot:
            if type(packet) is not str:raise QualificationDecline('stage result string schema')
            size+=len(packet.encode('utf-8'))
            if size>payload_limit:raise QualificationDecline('stage result payload limit')
        if len(json.dumps(snapshot,separators=(',',':'),ensure_ascii=False).encode('utf-8'))>payload_limit:
            raise QualificationDecline('stage result payload limit')
    except UnicodeError as exc:raise QualificationDecline('stage result UTF-8 schema') from exc
    audits={};values=[]
    try:
        for packet in snapshot:
            tree=json.loads(packet)
            if codec is not None and _wire_json(tree)!=packet:raise QualificationDecline('stage noncanonical v2 string')
            values.append(_decode_versioned(tree,limit=limit,codec=codec,payload_limit=payload_limit,_audits=audits))
        if audits:audit(tuple(audits.values()),limit=100000)
        return tuple(values)
    except QualificationDecline:raise
    except Exception as exc:raise QualificationDecline('stage decode transaction refused: '+type(exc).__name__) from exc


# Field order is copied from the owned source declarations, not a wire table.
# Its runtime agreement and loaded binding are checked by the package manifest.
_V2_RECORD_FIELDS=(
    ('Plane',('origin','u_vector','v_vector')),
    ('Cylinder',('origin','axis','radial_direction','radius','height','start_angle','sweep_angle')),
    ('LinePath',('start','end')),
    ('ArrangementPath',('curve','source_edge','owners','decomposition','member_ids')),
    ('ArrangementPoint',('position','member_ids')),
    ('ArrangementCell',('outer','holes')),
    ('MaterialArrangement',('face_id','paths','cells','area','world_tolerance','native_area_tolerance','support','orientation','certified')),
    ('EllipticArc',('center','u_vector','v_vector','start_angle','sweep_angle')),
    ('CylinderIntersectionCurve',('first','second','start_angle','sweep_angle','branch','parameterization','transform')),
    ('_CylinderSupport',('origin','axis','radial_direction','radius','height','start_angle','sweep_angle')),
    ('BezierQuadricCurve',('first','second','start','sweep','branch','parameterization','transform')),
    ('QuadricSupport',('kind','origin','axis','radius','slope','matrix','linear','constant','u_vector','v_vector')),
    ('PlaneRuledSupport',('base','directrix_vector','ruling','t0','length')))
_V2_SCALAR_DTYPES=('<f8','<i8','|b1')


def _wire_json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _value_codec(codec):
    if codec is not None and (type(codec) is not str or codec!='v2'):
        raise QualificationDecline('stage value codec version')
    if codec is not None and (np.dtype(float).str!='<f8' or
            tuple(np.dtype(t).str for t in (np.float64,np.int64,np.bool_))!=_V2_SCALAR_DTYPES):
        raise QualificationDecline('stage v2 runtime dtype/ABI')


def value_codec_contract():
    _value_codec('v2')
    if tuple((cls.__name__,tuple(f.name for f in fields(cls) if f.init)) for cls in registry())!=_V2_RECORD_FIELDS:
        raise QualificationDecline('stage v2 owned record schema changed')
    return dict(version='v2',schema_sha256=hashlib.sha256(_wire_json((_V2_RECORD_FIELDS,_V2_SCALAR_DTYPES,
        'v2-refined-primitive-tags-1-5-6-7-8-9-10-11')).encode()).hexdigest())


def verify_value_codec_contract(value):
    if (type(value) is not dict or set(value)!={'version','schema_sha256'} or
            any(type(item) is not str for item in value.values()) or value!=value_codec_contract()):
        raise QualificationDecline('stage value codec request binding')


def _transport_tasks_v2(tasks,*,limit,payload_limit):
    if type(tasks) not in (list,tuple) or len(tasks)>100000:raise QualificationDecline('stage transport task sequence')
    result=[]
    for task in tasks:
        if type(task) is not dict:raise QualificationDecline('stage transport task type')
        kind=task.get('kind')
        required=({'kind','first','second','tolerance','face_tolerances','cost'} if kind=='pair' else
            {'kind','face','inputs','tolerance','area_tolerance','seam','cost'} if kind=='face' else set())
        if not required or not required<=set(task) or set(task)-required-({'support'} if kind=='face' else set()):
            raise QualificationDecline('stage transport task fields')
        row={}
        for key,item in task.items():
            if key in ('inputs','support'):
                row[key]=compact_value_v2(item,limit=limit,payload_limit=payload_limit)
            elif key=='face_tolerances':
                if type(item) is not list or len(item)!=2 or any(type(x) not in (int,float) for x in item):
                    raise QualificationDecline('stage transport face tolerances')
                row[key]=list(item)
            else:
                if type(item) not in (int,float,str):raise QualificationDecline('stage transport metadata type')
                row[key]=item
        result.append(row)
    try:
        if len(_wire_json(result).encode('utf-8'))>payload_limit:raise QualificationDecline('stage task transport payload limit')
    except (ValueError,UnicodeError) as error:raise QualificationDecline('stage task transport JSON') from error
    return tuple(result)


def transport_tasks_v2(tasks,*,limit=100000,payload_limit=4_000_000):
    return _transport_tasks_v2(tasks,limit=limit,payload_limit=payload_limit)


def verify_transport_tasks_v2(original,transported,*,limit=100000,payload_limit=4_000_000):
    # Rebuild from retained H tasks; compare exact order/IDs/cost/float bits as
    # canonical JSON, independently of the P/W request digest and dispatch.
    expected=_transport_tasks_v2(original,limit=limit,payload_limit=payload_limit)
    if type(transported) not in (tuple,list) or _wire_json(expected)!=_wire_json(transported):
        raise QualificationDecline('stage original/transport task binding')


def _v2_bytes(text,*,expanding,payload_limit):
    if type(text) is not str or len(text)>2*payload_limit:raise QualificationDecline('stage v2 byte string bound/type')
    if expanding:
        raw=base64.b64decode(text,validate=True)
        if base64.b64encode(raw).decode('ascii')!=text:raise QualificationDecline('stage v2 noncanonical base64')
        return raw.hex()
    raw=bytes.fromhex(text)
    if raw.hex()!=text:raise QualificationDecline('stage noncanonical expanded bytes')
    return base64.b64encode(raw).decode('ascii')


def _v2_derived(value):
    # Exact finite metadata schema; actual owner/hash/shape is still recomputed
    # by the original decoder. Copy all mutable lists before construction.
    if type(value) is not list or len(value)>1:raise QualificationDecline('stage v2 derived schema')
    result=[]
    for row in value:
        if type(row) is not list or len(row)!=2 or type(row[0]) is not str or row[0] not in ('normal','_circumferential'):
            raise QualificationDecline('stage v2 derived row')
        m=row[1]
        if (type(m) is not list or len(m)!=4 or m[0]!='<f8' or type(m[0]) is not str or
                type(m[1]) is not list or any(type(n) is not int for n in m[1]) or
                type(m[2]) is not str or len(m[2])!=64 or any(c not in '0123456789abcdef' for c in m[2]) or type(m[3]) is not bool):
            raise QualificationDecline('stage v2 derived manifest')
        result.append([row[0],[m[0],list(m[1]),m[2],m[3]]])
    return result


def _v2_transform(packet,*,expanding,limit,payload_limit):
    _value_codec('v2');nodes=0
    def visit(p):
        nonlocal nodes
        nodes+=1
        if nodes>limit:raise QualificationDecline('stage value node limit='+str(limit))
        if expanding:
            if p is None:return ['none']
            if type(p) in (bool,int):return [type(p).__name__,p]
            if type(p) is str:
                raw=_v2_bytes(p,expanding=True,payload_limit=payload_limit)
                if len(raw)!=16:raise QualificationDecline('stage v2 float byte length')
                return ['float',raw]
            if type(p) is not list or not p or type(p[0]) is not int:raise QualificationDecline('stage v2 wire tag')
            tag=p[0]
            if tag==1 and len(p)==2 and type(p[1]) is str:return ['str',p[1]]
            if tag==5 and len(p)==3 and type(p[1]) is int and 0<=p[1]<3:
                return ['scalar',_V2_SCALAR_DTYPES[p[1]],_v2_bytes(p[2],expanding=True,payload_limit=payload_limit)]
            if tag==11 and len(p)==2 and type(p[1]) is int:return ['ref',p[1]]
            if len(p)<3 or type(p[1]) is not int:raise QualificationDecline('stage v2 ordinal type')
            if tag in (6,7) and len(p)==3 and type(p[2]) is list:
                return ['tuple' if tag==6 else 'list',p[1],[visit(x) for x in p[2]]]
            if tag==8 and len(p)==3 and type(p[2]) is list:
                rows=[]
                for row in p[2]:
                    if type(row) is not list or len(row)!=2:raise QualificationDecline('stage v2 dict row')
                    rows.append([visit(row[0]),visit(row[1])])
                return ['dict',p[1],rows]
            if tag==9 and len(p)==5 and type(p[2]) is list and all(type(n) is int and n>=0 for n in p[2]) and type(p[4]) is bool:
                return ['array',p[1],'<f8',list(p[2]),_v2_bytes(p[3],expanding=True,payload_limit=payload_limit),p[4]]
            if tag==10 and len(p) in (4,5) and type(p[2]) is int and 0<=p[2]<len(_V2_RECORD_FIELDS) and type(p[3]) is list:
                name,names=_V2_RECORD_FIELDS[p[2]]
                if len(p[3])!=len(names):raise QualificationDecline('stage v2 record arity')
                derived=[] if len(p)==4 else _v2_derived(p[4])
                if len(p)==5 and not derived:raise QualificationDecline('stage v2 explicit empty manifest')
                return ['record',p[1],name,[[key,visit(item)] for key,item in zip(names,p[3])],derived]
        else:
            if type(p) is not list or not p or type(p[0]) is not str:raise QualificationDecline('stage expanded wire tag')
            tag=p[0]
            if tag=='none' and len(p)==1:return None
            if tag in ('bool','int') and len(p)==2 and type(p[1]) is {'bool':bool,'int':int}[tag]:return p[1]
            if tag=='str' and len(p)==2 and type(p[1]) is str:return [1,p[1]]
            if tag=='float' and len(p)==2:return _v2_bytes(p[1],expanding=False,payload_limit=payload_limit)
            if tag=='scalar' and len(p)==3 and type(p[1]) is str and p[1] in _V2_SCALAR_DTYPES:
                return [5,_V2_SCALAR_DTYPES.index(p[1]),_v2_bytes(p[2],expanding=False,payload_limit=payload_limit)]
            if tag=='ref' and len(p)==2 and type(p[1]) is int:return [11,p[1]]
            if len(p)<3 or type(p[1]) is not int:raise QualificationDecline('stage expanded ordinal type')
            if tag in ('tuple','list') and len(p)==3 and type(p[2]) is list:
                return [6 if tag=='tuple' else 7,p[1],[visit(x) for x in p[2]]]
            if tag=='dict' and len(p)==3 and type(p[2]) is list:
                rows=[]
                for row in p[2]:
                    if type(row) is not list or len(row)!=2:raise QualificationDecline('stage expanded dict row')
                    rows.append([visit(row[0]),visit(row[1])])
                return [8,p[1],rows]
            if tag=='array' and len(p)==6 and p[2]=='<f8' and type(p[2]) is str and type(p[3]) is list and all(type(n) is int and n>=0 for n in p[3]) and type(p[5]) is bool:
                return [9,p[1],list(p[3]),_v2_bytes(p[4],expanding=False,payload_limit=payload_limit),p[5]]
            if tag=='record' and len(p)==5 and type(p[2]) is str and type(p[3]) is list:
                index=next((i for i,(name,names) in enumerate(_V2_RECORD_FIELDS) if name==p[2]),None)
                if index is None:raise QualificationDecline('stage expanded registry type')
                names=_V2_RECORD_FIELDS[index][1]
                if any(type(row) is not list or len(row)!=2 or type(row[0]) is not str for row in p[3]) or tuple(row[0] for row in p[3])!=names:
                    raise QualificationDecline('stage expanded record fields')
                result=[10,p[1],index,[visit(row[1]) for row in p[3]]];derived=_v2_derived(p[4])
                if derived:result.append(derived)
                return result
        raise QualificationDecline('stage unknown value grammar')
    try:
        if expanding:
            if type(packet) is not list or len(packet)!=2 or type(packet[0]) is not str or packet[0]!='v2':
                raise QualificationDecline('stage v2 wrapper')
            result=visit(packet[1])
        else:result=['v2',visit(packet)]
        # Both the transported value and its finite expansion remain bounded.
        if len(_wire_json(result).encode('utf-8'))>payload_limit or len(_wire_json(packet).encode('utf-8'))>payload_limit:
            raise QualificationDecline('stage value expanded/compact payload limit')
        return result
    except QualificationDecline:raise
    except Exception as error:raise QualificationDecline('stage v2 syntax refused: '+type(error).__name__) from error


def compact_value_v2(packet,*,limit=100000,payload_limit=4_000_000):
    return _v2_transform(packet,expanding=False,limit=limit,payload_limit=payload_limit)


def expand_value_v2(packet,*,limit=100000,payload_limit=4_000_000):
    return _v2_transform(packet,expanding=True,limit=limit,payload_limit=payload_limit)


def _decode_versioned(packet,*,limit,codec,payload_limit,_audits=None):
    if codec is None:return _decode_value(packet,limit=limit,_audits=_audits)
    expanded=expand_value_v2(packet,limit=limit,payload_limit=payload_limit)
    canonical=compact_value_v2(expanded,limit=limit,payload_limit=payload_limit)
    if canonical!=packet:raise QualificationDecline('stage noncanonical v2 reconstruction')
    # The original decoder performs constructor/recomputation/alias validation
    # and re-encodes its independent reconstruction before any object escapes.
    return _decode_value(expanded,limit=limit,_audits=_audits)


def _decode_value(packet,*,limit,_audits=None):
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
        if _encode_value(value,limit=limit,_audits=_audits)!=packet:raise QualificationDecline('stage noncanonical reconstruction')
        return value
    except QualificationDecline:raise
    except Exception as exc:raise QualificationDecline('stage decode refused: '+type(exc).__name__) from exc
