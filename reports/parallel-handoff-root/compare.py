"""Reconcile retained geometry-only evidence; never executes geometry."""
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
rows = []
identities = set()
for size in (16, 32, 48):
    baseline = [json.loads((root / f"baseline-01/grid-{size}-{i}.json").read_text()) for i in range(3)]
    frozen = baseline[0]
    for entry in baseline:
        assert entry["source_unchanged"]
        for field in ("document_digest", "chart_digest", "topology_counts", "joint_edges", "fixture_sha256"):
            assert entry[field] == frozen[field], (size, "baseline", field)
    for route in ("individual", "by_face"):
        candidate = [json.loads((root / f"candidate-01/{route}-grid-{size}-{i}.json").read_text()) for i in range(3)]
        for entry in candidate:
            assert entry["source_unchanged"]
            identities.add(entry["source_identity"])
            for field in ("document_digest", "chart_digest", "topology_counts", "joint_edges", "fixture_sha256"):
                assert entry[field] == frozen[field], (size, route, field)
        before = statistics.mean(entry["pipeline_seconds"] for entry in baseline[:2])
        after = statistics.mean(entry["pipeline_seconds"] for entry in candidate[:2])
        rows.append({"web_count": size, "route": route, "baseline_mean_seconds": before,
            "candidate_mean_seconds": after, "measured_reduction_percent": 100*(1-after/before),
            "baseline_repeats_seconds": [entry["pipeline_seconds"] for entry in baseline[:2]],
            "candidate_repeats_seconds": [entry["pipeline_seconds"] for entry in candidate[:2]],
            "baseline_profile_counts": baseline[2]["counts"], "candidate_profile_counts": candidate[2]["counts"],
            "document_digest": frozen["document_digest"], "chart_digest": frozen["chart_digest"],
            "fixture_sha256": frozen["fixture_sha256"], "topology_counts": frozen["topology_counts"]})
assert len(identities) == 1
evidence = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("baseline-01", "candidate-01") for p in sorted((root/directory).iterdir()) if p.is_file()}
result = {"schema": "anygeometry.parallel-preparation-comparison-v1", "rows": rows,
    "baseline_commit": "c6dc4261995a1e0c1ac9c19ba8ed96131d24b880",
    "measured_candidate_source_identity": identities.pop(), "prepared_outputs_identical": True,
    "candidate_checks": 18, "baseline_checks": 9, "evidence_sha256": evidence,
    "scope": "geometry-only queries, planning and application; no mesher, native or scientific mesh acceptance"}
(root / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps([{key: row[key] for key in ("web_count", "route", "baseline_mean_seconds", "candidate_mean_seconds", "measured_reduction_percent")} for row in rows]))
