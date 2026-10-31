"""Bounded PP1 evidence runner; only the child and its Windows Job are owned."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
label, *arguments = sys.argv[1:]
folder = ROOT / '.pp1-evidence' / label
folder.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TEMP=str(folder), TMP=str(folder),
           TASK_EVIDENCE_DIRECTORY=str(folder))
if arguments[0] == 'pytest':
    command = [sys.executable, '-B', '-m', *arguments, '-p', 'no:cacheprovider',
               '--basetemp', str(folder / 'pytest-temp')]
else:
    command = [sys.executable, '-B', '-u', *arguments]
spec = importlib.util.spec_from_file_location('pp1_owned_job', Path.home() / 'bin' / 'claude-review.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
job = helper.WindowsJob()
record = dict(command=command, cwd=str(ROOT), cap_seconds=110,
              environment_overrides={k: env[k] for k in ('PYTHONDONTWRITEBYTECODE', 'TEMP', 'TMP')})
identity_paths = set((ROOT/'src/anygeometry').rglob('*.py'))
identity_paths.add(Path(__file__).resolve())
identity_paths.update(ROOT/a for a in arguments if (ROOT/a).is_file())
record['input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(identity_paths)}
start = time.perf_counter()
code = None
status = 'startup_failed'
try:
    with (folder / 'stdout.txt').open('wb') as out, (folder / 'stderr.txt').open('wb') as err:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW | 4)
        record['owned_pid'] = child.pid
        job.assign_and_resume(child)
        try:
            code = child.wait(timeout=110)
            status = 'passed' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            status = 'timeout'
finally:
    try:
        buf = ctypes.create_string_buffer(8 + 64 * ctypes.sizeof(ctypes.c_void_p))
        if not job.api.QueryInformationJobObject(job.handle, 3, buf, len(buf), None):
            raise ctypes.WinError()
        count = ctypes.c_uint32.from_buffer(buf, 4).value
        if count > 64:
            raise RuntimeError('owned PID list overflow')
        checked = []
        for i in range(count):
            pid = ctypes.c_size_t.from_buffer(buf, 8 + i * ctypes.sizeof(ctypes.c_void_p)).value
            try:
                cmd = psutil.Process(pid).cmdline()
            except psutil.NoSuchProcess:
                continue
            if not cmd or Path(cmd[0]).resolve() != Path(sys.executable).resolve():
                raise RuntimeError('unexpected owned executable')
            if cmd != command:
                raise RuntimeError('unexpected owned command')
            checked.append(dict(pid=pid, command=cmd))
        record['checked_before_cleanup'] = checked
        cleanup = job.cleanup()
    except Exception as exc:
        cleanup = dict(verified=False, error=str(exc))
        status = 'cleanup_failed'
    record.update(status=status, exit_code=code, process_wall_seconds=time.perf_counter()-start,
                  cleanup=cleanup)
    record['inputs_unchanged'] = all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() ==
        record['input_sha256'][str(p.relative_to(ROOT))] for p in identity_paths)
    if not record['inputs_unchanged']:
        record['status'] = status = 'inputs_changed'
    (folder / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=status, exit_code=code, cleanup=cleanup, evidence=str(folder))))
    print((folder / 'stdout.txt').read_text(errors='replace')[-1600:])
    print((folder / 'stderr.txt').read_text(errors='replace')[-600:])
