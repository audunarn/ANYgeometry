"""Build isolated installed-wheel evidence, never using ambient owner packages.

Creates a new external directory and keeps complete logs, artifact hashes,
pip install reports and per-case results. Never publishes or deletes evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email import message_from_bytes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback
import zipfile

MCP_SOURCE = "4253c001b07263fa2e9015082cf4950db65cd2d3"
BASE_MODULES = {"ANYgeometry": "anygeometry", "numpy": "numpy"}
CONSUMERS = {"ANYmesher": "0.5.0"}
FULL = {"ANYfem": "0.4.1", "ANYsolver": "0.4.6", "ANYmaterial": "0.2.0", "ANYfileio": "0.3.2"}
MODULES = {"ANYmesher": "anymesher", "ANYfem": "anyfem", "ANYsolver": "anysolver",
           "ANYmaterial": "anymaterial", "ANYfileio": "anyfileio", "ANYgeometry-mcp": "anygeometry_mcp", "mcp": "mcp"}


def identity(path):
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        member, = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        info = message_from_bytes(archive.read(member))
    return info["Name"].lower().replace("_", "-"), info["Version"]


def classification(candidate, baseline):
    if candidate == "passed" and baseline == "passed":
        return "passed"
    if candidate != "passed" and baseline == "passed":
        return "candidate_regression"
    if candidate != "passed" and baseline != "passed":
        return "existing_consumer_or_environment_failure"
    return "baseline_failure_requires_review"


class Runner:
    def __init__(self, root):
        self.root = root
        self.index = 0
        self.env = {k: v for k, v in os.environ.items()
                    if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not k.startswith("PIP_")}
        self.env.update(PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        self.commands = []

    def run(self, args, *, check=True):
        self.index += 1
        log = self.root / f"command-{self.index:02d}.log"
        record = {"args": [str(a) for a in args], "log": log.name}
        self.commands.append(record)
        with log.open("w", encoding="utf-8") as output:
            try:
                result = subprocess.run(record["args"], cwd=self.root, env=self.env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=900, check=False)
                record["exit_code"] = result.returncode
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                raise
        if check and result.returncode:
            raise RuntimeError(f"command {self.index} failed ({result.returncode}); see {log}")
        return result.returncode

    def environment(self, name):
        destination = self.root / name
        self.run([sys.executable, "-I", "-m", "venv", "--copies", destination])
        return destination, destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("wheel", "consumers"))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory outside source checkouts")
    parser.add_argument("--profile", choices=("full", "macos"), default="full")
    parser.add_argument("--mcp-source", type=Path, help="checkout containing the frozen MCP source commit")
    parser.add_argument("--wheelhouse", type=Path, help="optional offline dependency wheel directory")
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--expected-machine", default="", help="require the explicitly selected architecture")
    args = parser.parse_args()
    if args.expected_machine and platform.machine() != args.expected_machine:
        parser.error(f"expected architecture {args.expected_machine}, got {platform.machine()}")
    workspace = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    if root.exists() or root.is_relative_to(workspace) or any((p / ".git").exists() for p in [root, *root.parents]):
        parser.error("output must be a new directory outside all source checkouts")
    candidate = args.candidate.resolve(strict=True)
    name, version = wheel_metadata(candidate)
    if name != "anygeometry":
        parser.error("candidate must be an ANYgeometry wheel")
    if args.mode == "consumers" and args.mcp_source is None:
        parser.error("consumer checks require a checkout containing the frozen MCP source")
    root.mkdir(parents=True)
    print(f"REPORT_ROOT={root}", flush=True)
    runner = Runner(root)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "profile": args.profile, "python": sys.version, "platform": platform.platform(),
              "source_commit": os.environ.get("GITHUB_SHA"), "workflow_run": os.environ.get("GITHUB_RUN_ID"),
              "candidate": identity(candidate), "status": "failed", "runs": {}, "commands": runner.commands}
    shutil.copyfile(Path(__file__).with_name("compatibility_probe.py"), root / "probe.py")
    probe_hash = identity(root / "probe.py")["sha256"]
    report["probe_sha256"] = probe_hash
    download_python = sys.executable
    pip = [download_python, "-I", "-m", "pip", "--isolated"]
    download = [*pip, "download", "--only-binary=:all:"]
    if args.wheelhouse:
        download += ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
    wheels = root / "dependencies"
    wheels.mkdir()
    try:
        owners = dict(CONSUMERS, **FULL) if args.profile == "full" else dict(CONSUMERS)
        requirements = [str(candidate), "numpy>=1.26", "shapely>=2.0"]
        if args.mode == "consumers":
            archive = root / "mcp-source.tar"
            runner.run(["git", "-C", args.mcp_source.resolve(strict=True), "archive",
                        "--format=tar", "--output", archive, MCP_SOURCE])
            source = root / "mcp-source"
            source.mkdir()
            with tarfile.open(archive) as archive_file:
                archive_file.extractall(source, filter="data")
            mcp_dist = root / "mcp-dist"
            runner.run([*pip, "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", mcp_dist, source])
            mcp_wheel, = mcp_dist.glob("*.whl")
            if wheel_metadata(mcp_wheel) != ("anygeometry-mcp", "0.1.0"):
                raise ValueError("MCP artifact does not match frozen adapter metadata")
            report["mcp"] = dict(identity(mcp_wheel), source=MCP_SOURCE, archive=identity(archive))
            requirements += [str(mcp_wheel), *(f"{n}=={v}" for n, v in owners.items())]
        runner.run([*download, "--dest", wheels, *requirements])
        artifacts = sorted(wheels.glob("*.whl"))
        report["resolved_artifacts"] = [dict(identity(p), distribution=wheel_metadata(p)[0],
                                               version=wheel_metadata(p)[1]) for p in artifacts]
        dependencies = [p for p in artifacts if wheel_metadata(p)[0] != "anygeometry"]

        def probe(label, wheel, *, present=True, consumer=False):
            lane = {"status": "failed"}
            report["runs"][label] = lane
            try:
                environment, executable = runner.environment(label + "-venv")
                work = root / label
                work.mkdir()
                selected = dependencies if present else [p for p in dependencies if wheel_metadata(p)[0] == "numpy"]
                runner.run([executable, "-I", "-m", "pip", "--isolated", "install", "--no-index",
                            "--report", work / "install.json", str(wheel), *selected])
                modules = dict(BASE_MODULES)
                versions = {"ANYgeometry": wheel_metadata(wheel)[1]}
                checks = ["core", "planar"]
                if present:
                    modules["shapely"] = "shapely"
                if consumer:
                    modules.update({n: MODULES[n] for n in owners})
                    modules.update({"ANYgeometry-mcp": "anygeometry_mcp", "mcp": "mcp"})
                    versions.update(owners, **{"ANYgeometry-mcp": "0.1.0"})
                    checks += ["mesher", "mcp"]
                    if args.profile == "full":
                        checks += ["fem", "fileio"]
                config = {"environment": str(environment), "wheel": str(wheel),
                          "wheel_sha256": identity(wheel)["sha256"], "modules": modules,
                          "versions": versions, "planar": present, "checks": checks,
                          "coordinates": args.mode == "wheel"}
                config_path = work / "config.json"
                config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                code = runner.run([executable, "-I", root / "probe.py", config_path], check=False)
                details = json.loads((work / "probe.json").read_text(encoding="utf-8"))
                lane.update(status="passed" if code == 0 and details["status"] == "passed" else "failed",
                            exit_code=code, report=str(work / "probe.json"), cases=details["cases"])
            except Exception:
                lane["error"] = traceback.format_exc()

        if args.mode == "wheel":
            probe("without-planar", candidate, present=False)
            probe("with-planar", candidate)
            report["status"] = "passed" if all(r["status"] == "passed" for r in report["runs"].values()) else "failed"
        else:
            if args.baseline_wheel:
                baseline = args.baseline_wheel.resolve(strict=True)
            else:
                control = root / "released"
                control.mkdir()
                runner.run([*download, "--no-deps", "--dest", control, "ANYgeometry==0.4.3"])
                baseline, = control.glob("*.whl")
            if wheel_metadata(baseline) != ("anygeometry", "0.4.3"):
                raise ValueError("baseline must be the released ANYgeometry 0.4.3 wheel")
            # The released ledger identifies the public wheel, not merely its version.
            expected = "6ab8f398de8ad1cc19b5c0d70a9787e1a15d3f5253c1125134eeb5050ec4b8b5"
            if identity(baseline)["sha256"] != expected:
                raise ValueError("baseline hash differs from the published 0.4.3 release ledger")
            report["baseline"] = identity(baseline)
            probe("candidate", candidate, consumer=True)
            probe("baseline", baseline, consumer=True)
            report["classification"] = classification(report["runs"]["candidate"]["status"], report["runs"]["baseline"]["status"])
            cases = {label: {case["name"]: case["status"] for case in lane.get("cases", [])}
                     for label, lane in report["runs"].items()}
            report["case_comparisons"] = {
                name: classification(cases["candidate"].get(name, "unavailable"), cases["baseline"].get(name, "unavailable"))
                for name in sorted(set(cases["candidate"]) | set(cases["baseline"]))}
            if "candidate_regression" in report["case_comparisons"].values():
                report["classification"] = "candidate_regression"
            report["status"] = "passed" if report["classification"] == "passed" else "failed"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "classification": report.get("classification"),
                      "report": str(root / "report.json"), "runs": {n: r["status"] for n, r in report["runs"].items()}}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
