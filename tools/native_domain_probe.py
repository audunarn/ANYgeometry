"""DIAGNOSTIC ONLY: bounded passive return-event capture for one original test."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import pytest

TARGET='tests/test_prepared_material_source_transfer.py::test_source_edge_transport_retains_weighted_cropped_and_reversed_native_domains[False]'
BASE='68098b042e2e5d421dc5d62891b32c47b2e44e80'
MAX_ROWS=512
MAX_BYTES=2_000_000
EXPECTED={}  # Filled from the unchanged source checkout when preparing the spec.
_STATE=None


def _copy(value,depth=0):
    if depth>12:raise ValueError('diagnostic depth')
    kind=type(value)
    if value is None or kind in (bool,str):
        if kind is str and len(value)>512:raise ValueError('diagnostic string')
        return value
    if kind is int:
        if value.bit_length()>8192:raise ValueError('diagnostic integer')
        return value
    if kind is float:return {'float_hex':value.hex()}
    fractions=sys.modules.get('fractions')
    if fractions is not None and kind is vars(fractions).get('Fraction'):
        return {'q':[_copy(value.numerator,depth+1),_copy(value.denominator,depth+1)]}
    numpy=sys.modules.get('numpy')
    scalar_names=('float32','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool_','longlong','ulonglong','intp','uintp')
    if numpy is not None and any(kind is vars(numpy).get(name) for name in scalar_names):
        return _copy(kind.item(value),depth+1)
    if numpy is not None and kind is vars(numpy).get('ndarray'):
        if value.size>1024:raise ValueError('diagnostic array')
        return _copy(kind.tolist(value),depth+1)
    if kind in (tuple,list):
        if len(value)>128:raise ValueError('diagnostic sequence')
        return [_copy(x,depth+1) for x in value]
    if kind is dict:
        if len(value)>128:raise ValueError('diagnostic mapping')
        if all(type(k) is str for k in value):return {k:_copy(v,depth+1) for k,v in value.items()}
        return {'items':[[_copy(k,depth+1),_copy(v,depth+1)] for k,v in value.items()]}
    raise TypeError('unsupported diagnostic value: '+kind.__name__)


def _raw(value,name):return object.__getattribute__(value,name)


def _curve(curve):
    row={'type':type(curve).__name__}
    for key in ('domain','path'):
        try:value=_raw(curve,key)
        except AttributeError:continue
        row[key]=_copy(value)
    try:pcurve=_raw(curve,'pcurve')
    except AttributeError:return row
    row['pcurve']={k:_copy(_raw(pcurve,k)) for k in ('degree','controls','weights','domain','knots')}
    return row


def _capture(frame,result):
    d=frame.f_locals;name=frame.f_code.co_name
    row={'function':name,'filename':frame.f_code.co_filename,'returned_none':result is None}
    if name=='_affine_parent_joint_bound':
        row['bound']=_copy(result)
        for key in ('net','du','dv','chart_starts','low','high','extent','bbox','tolerance','points'):
            if key in d:row[key]=_copy(d[key])
    elif name=='_refused_trim_station':
        row['returned']=_copy(result)
        for key in ('ordinary','bound','rectangle','source_end','fixed','ranges','level','station','low','high'):
            if key in d:row[key]=_copy(d[key])
        row['joint_present']=d.get('joint') is not None
        if d.get('joint') is not None:
            curve,point=d['joint'];row['joint_point']=_copy(point);row['joint_curve']=_curve(curve)
        row['section_domain']=_copy(_raw(d['section'],'domain'))
        event=d['event'];row['event']={k:_copy(_raw(event,k)) for k in ('section_interval','segment_interval','chart_enclosure')}
        row['arcs_observed']='arcs' in d
        if 'arcs' in d:
            if len(d['arcs'])>64:raise ValueError('diagnostic arcs')
            row['arcs']=[{k:_copy(_raw(arc,k)) for k in ('slicing','s0','s1')} for arc in d['arcs']]
    elif name=='fragment_nurbs_plane':
        for key in ('parent','joint_bound','bbox','vertices'):
            if key in d:row[key]=_copy(d[key])
        model=d['model'];table=_raw(model,'vertices')
        if len(table)>128:raise ValueError('diagnostic vertices')
        row['actual_vertices']=[{'id':key,'position':_copy(_raw(value,'position'))} for key,value in table.items()]
        row['returned']=_copy(result)
    else:
        for key in ('face_id','ends','tolerance','tol2'):
            if key in d:row[key]=_copy(d[key])
        row['returned']=_copy(result)
        if 'loops' in d:
            row['loops']=[[{'edge':_raw(use,'edge'),'forward':_raw(use,'forward')} for use in loop] for loop in d['loops']]
            edges=_raw(d['model'],'edges')
            row['curves']={str(use.edge):_curve(edges[use.edge].curve) for loop in d['loops'] for use in loop}
    return row


def _observe(state,frame,event,result):
    if event!='return':return
    module=frame.f_globals.get('__name__')
    name=frame.f_code.co_name
    allowed={'anygeometry.nurbs_plane_arrangement':('_affine_parent_joint_bound','fragment_nurbs_plane','_refused_trim_station'),
             'anygeometry.nurbs_section_faces':('validate_nurbs_section_face',)}
    if module not in allowed or name not in allowed[module]:return
    try:
        if len(state['rows'])>=MAX_ROWS:raise ValueError('diagnostic row count')
        state['rows'].append(_capture(frame,result))
    except Exception as error:
        state['incomplete']=True
        if len(state['errors'])<16:state['errors'].append(type(error).__name__)


def source_report(root):
    baseline=Path(os.environ['NATIVE_DOMAIN_SOURCE_BASELINE'])
    expected=json.loads(baseline.read_text(encoding='utf-8'))
    rows={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected['files']}
    return {'base':BASE,'source_head':(root/'.git/HEAD').read_text().strip(),'expected':expected['files'],'actual':rows,'match':rows==expected['files'],'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest()}


def _derived(rows):
    # Only completed copies are used AFTER the original decision/profile restoration.
    from fractions import Fraction as Q
    def decode(x):
        if type(x) is dict and set(x)=={'q'}:return Q(*x['q'])
        if type(x) is dict and set(x)=={'float_hex'}:return Q(float.fromhex(x['float_hex']))
        if type(x) is list:return [decode(v) for v in x]
        return x
    result=[]
    for row in rows:
        if row['function']!='validate_nurbs_section_face' or 'ends' not in row or 'loops' not in row:continue
        ends={tuple(key):decode(value) for key,value in row['ends']['items']}
        for index,loop in enumerate(row['loops']):
            for position,use in enumerate(loop):
                following=(position+1)%len(loop);a=ends[(index,position)][1];b=ends[(index,following)][0]
                spread=sum(max(abs(x[0]-y[1]),abs(x[1]-y[0]))**2 for x,y in zip(a,b))
                exact=all(row['curves'][str(loop[k]['edge'])]['type']!='PlaneNurbsCurve' for k in (position,following))
                tol=decode(row['tol2'])
                result.append(dict(face=row.get('face_id'),edge=use['edge'],next_edge=loop[following]['edge'],branch='exact' if exact else 'spread',spread2=[spread.numerator,spread.denominator],tolerance2=[tol.numerator,tol.denominator],refused=(a!=b) if exact else spread>tol))
    return result


def _write(state,path):
    state['derived']=_derived(state['rows'])
    encoded=json.dumps(state,indent=2,allow_nan=False).encode('utf-8')
    while len(encoded)>MAX_BYTES and state['rows']:
        state['incomplete']=True;state['rows'].pop();state['derived']=[]
        encoded=json.dumps(state,indent=2,allow_nan=False).encode('utf-8')
    if len(encoded)>MAX_BYTES:raise ValueError('diagnostic byte bound')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(encoded)


@pytest.hookimpl(hookwrapper=True,tryfirst=True)
def pytest_runtest_protocol(item,nextitem):
    global _STATE
    if item.nodeid!=TARGET:
        yield;return
    prior=sys.getprofile()
    state={'diagnostic_only':True,'qualification':False,'target':TARGET,'nodeid':item.nodeid,'rows':[],
           'errors':[],'incomplete':False,'reports':[],'source_head_base':BASE,'workflow_sha':os.environ.get('GITHUB_SHA'),
           'history':os.environ.get('NATIVE_DOMAIN_HISTORY'),'runtime':{'python':sys.version,'platform':sys.platform}}
    try:
        state['runtime'].update(numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'))
        state['sources']=source_report(Path(item.config.rootpath))
    except Exception as error:
        state['incomplete']=True;state['errors'].append(type(error).__name__)
    _STATE=state
    def observe(frame,event,arg):
        if prior is not None:prior(frame,event,arg)
        _observe(state,frame,event,arg)
    started=time.monotonic()
    try:
        sys.setprofile(observe)
        yield
    finally:
        sys.setprofile(prior);state['profile_restored']=sys.getprofile() is prior
        state['wall_seconds']=time.monotonic()-started;_STATE=None
        try:_write(state,os.environ.get('NATIVE_DOMAIN_PROBE_OUT','native-domain-probe.json'))
        except Exception as error:sys.stderr.write('DIAGNOSTIC CAPTURE INCOMPLETE: '+type(error).__name__+'\n')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome=yield
    if _STATE is not None and item.nodeid==TARGET:
        report=outcome.get_result();_STATE['reports'].append({'when':report.when,'outcome':report.outcome})


def _verify_owned_standalone(pid,created,command):
    import psutil
    owned=psutil.Process(pid)
    if owned.create_time()!=created:
        raise RuntimeError('owned standalone PID identity changed')
    actual=owned.cmdline()
    normalize=lambda value:os.path.normcase(os.path.realpath(value))
    if (not actual or not command or normalize(actual[0])!=normalize(command[0])
            or actual[1:]!=command[1:]):
        raise RuntimeError('owned standalone PID command changed')


def standalone():
    """One trusted in-process fixture, with a110s owned pytest PID cap; no group kills."""
    import subprocess
    import psutil
    folder=Path(os.environ.get('NATIVE_DOMAIN_PROBE_OUT','diagnostic/native-domain-probe.json')).parent
    folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pytest',TARGET,'-q','--junitxml=kernel-results.xml']
    env=os.environ.copy();env.pop('ANYGEOMETRY_TEST_SHARD',None);env.pop('ANYGEOMETRY_SHARD_REPORT',None)
    started=time.monotonic();process=None;status='failed';code=2;clean=False
    with (folder/'standalone-stdout.txt').open('xb') as log:
        try:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            created=psutil.Process(process.pid).create_time()
            try:code=process.wait(timeout=110);status='completed'
            except subprocess.TimeoutExpired:
                _verify_owned_standalone(process.pid,created,command)
                process.kill();process.wait(timeout=5);code=124;status='timeout'
        finally:
            if process is not None and process.poll() is not None:
                try:os.killpg(process.pid,0)
                except ProcessLookupError:clean=True
                except PermissionError:pass
            (folder/'standalone-result.json').write_text(json.dumps(dict(diagnostic_only=True,command=command,
                status=status,exit_code=code,process_wall_seconds=time.monotonic()-started,
                cleanup_verified=clean,scope='trusted oneface inprocess; no whole-session/process enumeration'),indent=2))
    if not clean:raise RuntimeError('owned standalone process-group absence unverified')
    return code


if __name__=='__main__':
    if sys.argv[1:]!=['--standalone']:raise SystemExit('only --standalone diagnostic supported')
    raise SystemExit(standalone())
