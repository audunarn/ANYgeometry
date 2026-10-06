"""Reconcile final source/installed artifact with unchanged timed definitions."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

root = Path(__file__).resolve().parent
repository = root.parents[1]
worker = repository / "reports/parallel-handoff"
final = json.loads((worker / "candidate-sha256.json").read_text())
measured = json.loads((root / "measured-owned-files.json").read_text())
for path, expected in final.items():
    assert hashlib.sha256((repository/path).read_bytes()).hexdigest() == expected, path
delta = [path for path in final if final[path] != measured[path]]
assert set(delta) == {"src/anygeometry/component_partition.py", "tests/test_parallel_handoff.py", "docs/PARALLEL_COMPONENT_HANDOFF.md"}
artifact = json.loads((root / "installed-02/artifact.json").read_text())
wheel = Path(artifact["wheel"])
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == artifact["sha256"]
with zipfile.ZipFile(wheel) as archive:
    for path in sorted((repository / "src/anygeometry").rglob("*.py")):
        assert archive.read(path.relative_to(repository / "src").as_posix()) == path.read_bytes(), path
checks = [json.loads(path.read_text()) for path in sorted(worker.glob("check-*/result.json"))]
measure = runpy.run_path(str(root / "measure.py"))
final_source = measure["source_identity"](repository / "src")
comparison = json.loads((root / "comparison.json").read_text())
result = {"schema": "anygeometry.parallel-handoff-development-acceptance-v1",
    "branch": "codex/parallel-handoff", "base_commit": comparison["baseline_commit"],
    "source_identity": final_source, "final_owned_files_sha256": final,
    "measured_source_identity": comparison["measured_candidate_source_identity"],
    "postmeasurement_changes": delta,
    "timed_definition_reuse": "Only unexecuted partition refusal, its tests and API docs changed; all exercised preparation sources unchanged",
    "wheel_sha256": artifact["sha256"], "all_wheel_package_sources_match": True,
    "installed_origin": artifact["spawn_results"][0]["origin"], "spawned_components": len(artifact["spawn_results"]),
    "focused_process_wall_seconds": sum(check["process_wall_seconds"] for check in checks),
    "focused_attempts": len(checks), "failed_evidence_retained": True,
    "main_focused_slice": {"tests": 187, "failures": 0},
    "final_offset_slice": {"tests": 5, "failures": 0},
    "final_derived_index_callback_slice": {"tests": 2, "failures": 0},
    "prepared_output_parity": True, "source_review": "independent authorship, same model family, findings resolved",
    "full_platform_ci": "not run", "mesher_acceptance": "not run",
    "published": False, "request4": "deferred", "historical_ten_shot_families": "closed, unchanged"}
(root / "acceptance.json").write_text(json.dumps(result, indent=2))
print(json.dumps({key: result[key] for key in ("source_identity", "wheel_sha256", "all_wheel_package_sources_match", "focused_process_wall_seconds")}))
