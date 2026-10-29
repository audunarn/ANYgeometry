"""Isolated persistent installed-artifact worker for ch3e_pilot.py.

One JSON command per line; all scientific artifacts are written before replies.
Never invoke this directly as a qualification claim.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from fractions import Fraction
if hasattr(sys, 'set_int_max_str_digits'): sys.set_int_max_str_digits(100000)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ch3e_schema import write_json,NAMES


def fraction(pair): return Fraction(*pair)

def query_from_dict(data):
    from anygeometry.certified_hydrostatics import HalfSpaceQuery
    return HalfSpaceQuery(tuple(map(fraction,data['normal'])),fraction(data['offset']),
        tuple(map(fraction,data['waterplane_origin'])),
        tuple(tuple(map(fraction,row)) for row in data['waterplane_axes']),
        {k:fraction(v) for k,v in data['absolute_accuracy'].items()},tuple(data['forms']))


def main():
    import anygeometry
    from anygeometry.b1_example import build_b1_hull,build_b1_half_hull
    from anygeometry import serialization
    from anygeometry.certified_hydrostatics import (CertifiedHydrostaticsPolicy,
        prepare_certified_hydrostatics,query_certified_hydrostatics,verify_certified_hydrostatics)
    root=Path(sys.argv[1]).resolve(); site=Path(sys.argv[2]).resolve()
    imported=Path(anygeometry.__file__).resolve()
    if not imported.is_relative_to(site): raise RuntimeError(f'import outside pinned install: {imported}')
    if importlib.metadata.version('ANYgeometry') != '0.4.7': raise RuntimeError('expected version 0.4.7')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise RuntimeError('thread settings must be explicitly pinned to one')
    wheel=Path(sys.argv[3]).resolve()
    with zipfile.ZipFile(wheel) as archive:
        wheel_files={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                     if name.startswith('anygeometry/') and not name.endswith('/')}
    actual_package_files={str(p.relative_to(site)).replace('\\','/') for p in imported.parent.rglob('*')
                          if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual_package_files != set(wheel_files):
        raise RuntimeError(f'installed package file set differs from wheel: {sorted(actual_package_files ^ set(wheel_files))}')
    for name,expected_hash in wheel_files.items():
        installed=site/name
        if not installed.is_file() or hashlib.sha256(installed.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'installed bytes differ from pinned wheel: {name}')
    hashes={str(p.relative_to(imported.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(imported.parent.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    write_json(root/'installed-runtime.json',dict(module=str(imported),python=sys.executable,
        version=importlib.metadata.version('ANYgeometry'),files=hashes,sys_path=sys.path,
        effective_threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    models={}; results={}
    for line in sys.stdin:
        cmd=json.loads(line); started=time.perf_counter(); timings={}
        def timed(name,call,*args,**kwargs):
            call_start=time.perf_counter()
            try: return call(*args,**kwargs)
            finally: timings[name]=time.perf_counter()-call_start
        try:
            op=cmd['op']; kind=cmd.get('kind','full')
            if op == 'prepare':
                model,faces=timed('fixture_build_s',build_b1_half_hull if kind == 'half' else build_b1_hull)
                policy=CertifiedHydrostaticsPolicy(convention_version='2')
                prepared=timed('cold_prepare_call_s',prepare_certified_hydrostatics,model,faces.values(),policy=policy)
                document=timed('serialization_s',serialization.to_dict,model)
                write_json(root/f'{kind}-model.json',document)
                write_json(root/f'{kind}-boundary.json',dict(faces=faces,model=prepared.model_identity(),
                    artifact=prepared.artifact_identity(),policy=policy.as_dict(),units='m'))
                models[kind]=(model,faces,prepared)
                value=dict(model=prepared.model_identity(),artifact=prepared.artifact_identity())
            elif op == 'query':
                model,faces,prepared=models[kind]; q=query_from_dict(cmd['query'])
                name=cmd['id']; write_json(root/f'{name}-request.json',q.as_dict())
                result=timed('warm_query_call_s',query_certified_hydrostatics,prepared,q)
                write_json(root/f'{name}-response.json',result.claim())
                (root/f'{name}-certificate.json').write_bytes(result.proof)
                results[name]=(q,result,kind)
                value=dict(occupancy=result.occupancy,proof_sha256=hashlib.sha256(result.proof).hexdigest())
            elif op == 'verify':
                name=cmd['id']; q,result,kind=results[name]
                model,faces,prepared=models[kind]
                # Expected comes from the independently pinned preparation/runtime,
                # not from the untrusted certificate being checked.
                expected=dict(artifact=prepared.implementation,convention_version='2',units='m',policy=prepared.policy)
                checked=timed('independent_verify_call_s',verify_certified_hydrostatics,model,faces.values(),q,result,expected)
                value=dict(proof_sha256=checked.proof_sha256,occupancy=checked.occupancy,
                           work=checked.work,claim=checked.result.claim())
                write_json(root/f'{name}-verified.json',value)
                value={key:val for key,val in value.items() if key != 'claim'}
            elif op == 'reference':
                from ch3e_reference import reference,analytic_stations
                q=cmd['query']
                value=reference(tuple(map(fraction,q['normal'])),fraction(q['offset']),
                    tuple(map(fraction,q['waterplane_origin'])),
                    tuple(tuple(map(fraction,row)) for row in q['waterplane_axes']),half=kind=='half')
                write_json(root/f"{cmd['id']}-reference.json",value)
                write_json(root/'analytic-stations.json',analytic_stations())
            elif op == 'stop': break
            else: raise ValueError('unknown operation')
            reply=dict(status='ok',result=value)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            reply=dict(status='error',type=type(exc).__name__,message=str(exc),
                       category=getattr(getattr(exc,'category',None),'value',None),
                       phase=getattr(exc,'phase',None),sub_reason=getattr(exc,'sub_reason',None))
            write_json(root/f"failure-{cmd.get('id',cmd['op'])}.json",dict(command=cmd,**reply))
        reply['worker_phase_wall_s']=time.perf_counter()-started
        reply['actual_call_timings_s']=timings
        print(json.dumps(reply,allow_nan=False),flush=True)

if __name__ == '__main__': main()
