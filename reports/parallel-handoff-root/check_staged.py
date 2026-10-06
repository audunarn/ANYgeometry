"""Inspect immutable evidence staged bytes, without rewriting captured logs."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
raw = subprocess.check_output(["git", "ls-files", "--stage", "-z", "--", "reports/parallel-handoff", "reports/parallel-handoff-root"], cwd=root)
entries = []
for row in raw.split(b"\0"):
    if not row:
        continue
    header, name = row.split(b"\t", 1)
    entries.append((name.decode(), header.split()[1].decode()))
paths = "\n".join(name for name, signature in entries) + "\n"
actual = subprocess.check_output(["git", "hash-object", "--no-filters", "--stdin-paths"],
    input=paths.encode(), cwd=root).decode().splitlines()
assert len(actual) == len(entries)
errors = [name for (name, signature), blob in zip(entries, actual) if blob != signature]
result = {"files": len(entries), "byte_identical_to_index": not errors, "errors": errors}
(Path(__file__).resolve().parent / "index-evidence-bytes.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
assert not errors
