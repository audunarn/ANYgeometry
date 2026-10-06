"""Bounded offline wheel build/install/spawn check, with command evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    root = Path(__file__).resolve().parent
    evidence = root / (sys.argv[1] if len(sys.argv) > 1 else "installed-01")
    evidence.mkdir(exist_ok=False)
    wheels = evidence / "wheels"
    wheels.mkdir()
    outside = Path(tempfile.mkdtemp(prefix="anygeometry-parallel-wheel-"))
    records = []
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"

    def run(command, cwd):
        ordinal = len(records)
        started = time.perf_counter()
        with (evidence / f"{ordinal}.stdout.log").open("wb") as out, (evidence / f"{ordinal}.stderr.log").open("wb") as err:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err)
            try:
                code = process.wait(timeout=30.)
                disposition = "completed"
            except subprocess.TimeoutExpired:
                # Only build/install commands here; spawn helper bounds its children.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                code = process.wait()
                disposition = "timeout"
        records.append({"command": command, "cwd": str(cwd), "status": disposition,
            "exit_code": code, "process_wall_seconds": time.perf_counter()-started})
        (evidence / "run.json").write_text(json.dumps({"outside_checkout": str(outside), "records": records,
            "dependency_environment": "venv uses existing system NumPy; candidate owner package is installed locally",
            "aggregate_process_seconds": sum(row["process_wall_seconds"] for row in records)}, indent=2))
        if code:
            raise RuntimeError(f"artifact step {ordinal} failed; preserve {evidence}")

    run([sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-index", "--no-build-isolation", "--no-deps",
         "--wheel-dir", str(wheels), "."], root.parents[1])
    wheel, = wheels.glob("*.whl")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(outside / "venv")], outside)
    interpreter = outside / "venv/Scripts/python.exe"
    run([str(interpreter), "-m", "pip", "install", "--no-cache-dir", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], outside)
    script = outside / "installed_spawn.py"
    shutil.copyfile(root / "installed_spawn.py", script)
    run([str(interpreter), str(script)], outside)
    result = {"wheel": str(wheel), "sha256": wheel_hash, "outside_checkout": str(outside),
              "spawn_results": [json.loads(p.read_text()) for p in sorted(outside.glob("spawn-*.json"))]}
    (evidence / "artifact.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
