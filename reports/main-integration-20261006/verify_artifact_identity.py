"""Verify the existing installed candidate without rebuilding passed artifacts."""
from pathlib import Path
import json, os, subprocess, sys, time

evidence = Path(__file__).resolve().parent
artifact_dir = evidence / (sys.argv[1] if len(sys.argv)>1 else 'installed-01')
artifact = json.loads((artifact_dir / 'artifact.json').read_text())
record = json.loads((artifact_dir / 'run.json').read_text())
outside = Path(artifact['outside_checkout'])
script = outside / 'verify_identity.py'
script.write_text('''import hashlib, json, sys, zipfile
from pathlib import Path
import numpy, anygeometry
from threadpoolctl import threadpool_info
record=json.loads(Path(sys.argv[1]).read_text())
wheel=json.loads(Path(sys.argv[2]).read_text())['wheel']
origin=Path(anygeometry.__file__).resolve()
assert origin.is_relative_to(Path(sys.prefix).resolve())
with zipfile.ZipFile(wheel) as archive:
    for name,digest in record['source_sha256'].items():
        member=name.replace('\\\\','/').removeprefix('src/')
        assert hashlib.sha256(archive.read(member)).hexdigest()==digest, member
for name in ('plan_independent_components','validate_component_partition_binding',
             'query_trimmed_surface_charts_by_face','clone_prepared_geometry',
             'query_prepared_native_material_reference_scope'):
    assert callable(getattr(anygeometry,name))
from anygeometry.polynomial_extrusion_support import prove_polynomial_extrusion_support
assert callable(prove_polynomial_extrusion_support)
pools=threadpool_info()
assert pools and all(row['num_threads']==1 for row in pools)
print(json.dumps(dict(origin=str(origin),numpy=numpy.__version__,threadpools=pools,
                     packaged_sources=len(record['source_sha256']),verified=True)))
''')
command=[str(outside/'venv/Scripts/python.exe'),str(script),str(artifact_dir/'run.json'),str(artifact_dir/'artifact.json')]
env=dict(os.environ);env.pop('PYTHONPATH',None)
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): env[name]='1'
remaining=60-record.get('prior_process_seconds',record.get('unit_process_seconds',0))-record['aggregate_process_seconds']
start=time.perf_counter()
with (artifact_dir/'identity.stdout.log').open('wb') as out, (artifact_dir/'identity.stderr.log').open('wb') as err:
    process=subprocess.Popen(command,cwd=outside,env=env,stdout=out,stderr=err)
    try: code=process.wait(timeout=min(10,remaining-.5));status='completed'
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=err,stderr=err)
        code=process.wait();status='timeout'
elapsed=time.perf_counter()-start
result=dict(command=command,cwd=str(outside),status=status,exit_code=code,
            process_wall_seconds=elapsed,cumulative_process_seconds=60-remaining+elapsed,
            remaining_seconds=remaining-elapsed)
(artifact_dir/'identity-run.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
raise SystemExit(code if status=='completed' else 124)
