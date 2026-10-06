"""Bound focused development tests; retain exact process-wall evidence."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "main-integration-20261006"
EVIDENCE.mkdir(parents=True, exist_ok=True)
previous = [json.loads(p.read_text()) for p in EVIDENCE.glob("check-*/result.json")]
used = sum(r["process_wall_seconds"] for r in previous)
used += sum(json.loads(p.read_text())["aggregate_process_seconds"] for p in EVIDENCE.glob("installed-*/run.json"))
used += sum(json.loads(p.read_text())["process_wall_seconds"] for p in EVIDENCE.glob("installed-*/identity-run.json"))
remaining = 60.0 - used
if remaining <= 0:
    raise SystemExit("focused process-wall budget exhausted")
attempt = EVIDENCE / ("check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
attempt.mkdir()
environment = os.environ.copy()
environment["PYTHONPATH"] = str(ROOT / "src")
environment["EPOCH_RUNTIME_EVIDENCE"] = str(attempt / "epoch-runtime.json")
environment["NATIVE_SCOPE_RUNTIME_EVIDENCE"] = str(attempt / "native-runtime.json")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    environment[name] = "1"
command = [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--junitxml=" + str(attempt / "junit.xml"), *sys.argv[1:]]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for folder in (ROOT / "src" / "anygeometry", ROOT / "tests") for p in sorted(folder.glob("*.py"))}
runtime = {"executable": sys.executable, "python": sys.version,
           "environment": {k:environment[k] for k in ("PYTHONPATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
timeout = min(40.0, max(0.0, remaining - 5.0))
if timeout <= 0:
    raise SystemExit("verification reserve reached")
(attempt / "launch.json").write_text(json.dumps({"command":command,"cwd":str(ROOT),"runtime":runtime,
    "candidate_sha256":hashes,"timeout_seconds":timeout,"budget_used_before":used}, indent=2))
start = time.perf_counter()
with (attempt / "stdout.txt").open("wb") as stdout, (attempt / "stderr.txt").open("wb") as stderr:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    status = "exit"
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = "timeout"
        # pytest cases in this slice launch no subprocesses; retain tree safety
        # if that contract changes in a later invocation.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=stderr, stderr=stderr)
        process.wait()
        code = process.returncode
elapsed = time.perf_counter() - start
result = {"command":command,"status":status,"exit_code":code,"process_wall_seconds":elapsed,
          "cumulative_process_wall_seconds":used+elapsed,"budget_remaining_seconds":60-used-elapsed,
          "runtime":runtime,"candidate_sha256":hashes}
if (attempt / "junit.xml").exists():
    import xml.etree.ElementTree as ET
    suite = ET.parse(attempt / "junit.xml").getroot().find("testsuite")
    result["counts"] = {k:int(suite.attrib.get(k,0)) for k in ("tests","failures","errors","skipped")}
(attempt / "result.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ("candidate_sha256","runtime","command")}))
print(str(attempt))
raise SystemExit(code if status == "exit" else 124)
