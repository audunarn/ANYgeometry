"""Separate, post-download identity/payload review for the 0.4.5 release."""
import email
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "release-bundle"
OLD, = (ROOT.parent / "general_intersections/hosted-run-36829763205/general-intersections-windows-latest").rglob("anygeometry-0.4.4-py3-none-any.whl")

def identity(path):
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

wheel = BUNDLE / "anygeometry-0.4.5-py3-none-any.whl"
sdist = BUNDLE / "anygeometry-0.4.5.tar.gz"
assert {p.name for p in BUNDLE.iterdir()} == {wheel.name, sdist.name, "SHA256SUMS"}
rows = [identity(wheel), identity(sdist)]
assert identity(OLD)["sha256"] == "2f154358ec2b873db48994d14be1fd4659411a128751407772d60becb667c5f6"
assert (BUNDLE / "SHA256SUMS").read_text().splitlines() == [
    row["sha256"] + "  " + row["filename"] for row in rows]
with zipfile.ZipFile(wheel) as new, zipfile.ZipFile(OLD) as old:
    package = {name for name in new.namelist() if name.startswith("anygeometry/")}
    assert package == {name for name in old.namelist() if name.startswith("anygeometry/")}
    for name in package:
        expected = old.read(name)
        if name == "anygeometry/__init__.py":
            assert expected.count(b'__version__ = "0.4.4"') == 1
            expected = expected.replace(b'__version__ = "0.4.4"', b'__version__ = "0.4.5"')
        assert new.read(name) == expected, name
    assert "anygeometry/py.typed" in package
    metadata_name, = [name for name in new.namelist() if name.endswith(".dist-info/METADATA")]
    metadata = email.message_from_bytes(new.read(metadata_name))
    assert metadata["Name"] == "ANYgeometry" and metadata["Version"] == "0.4.5"
    assert metadata["License-Expression"] == "MPL-2.0"
    wheel_sources = {name: new.read(name) for name in package}
with tarfile.open(sdist, "r:gz") as archive:
    for name, content in wheel_sources.items():
        item = archive.extractfile("anygeometry-0.4.5/src/" + name)
        assert item is not None and item.read() == content, name
    metadata = email.message_from_binary_file(archive.extractfile("anygeometry-0.4.5/PKG-INFO"))
    assert metadata["Version"] == "0.4.5" and metadata["License-Expression"] == "MPL-2.0"
candidate_wheel = ROOT / "ci-candidate" / wheel.name
candidate_sdist = ROOT / "ci-candidate" / sdist.name
with zipfile.ZipFile(wheel) as release, zipfile.ZipFile(candidate_wheel) as candidate:
    assert set(release.namelist()) == set(candidate.namelist())
    for name in release.namelist():
        assert release.read(name) == candidate.read(name), name
    wheel_members = len(release.namelist())
with tarfile.open(sdist, "r:gz") as release, tarfile.open(candidate_sdist, "r:gz") as candidate:
    release_files = {item.name: release.extractfile(item).read() for item in release if item.isfile()}
    candidate_files = {item.name: candidate.extractfile(item).read() for item in candidate if item.isfile()}
    assert release_files == candidate_files
    sdist_members = len(release_files)
report = {"schema": "anygeometry.release-payload-review-v1", "status": "passed",
          "artifacts": rows, "qualified_development_wheel": identity(OLD),
          "package_members": len(package), "only_runtime_change": "__version__: 0.4.4 to 0.4.5",
          "sdist_wheel_source_equality": True, "checksum_closed_set": True,
          "ci_candidate_artifacts": [identity(candidate_wheel), identity(candidate_sdist)],
          "ci_release_wheel_members_identical": wheel_members,
          "ci_release_sdist_members_identical": sdist_members,
          "scientific_evidence": "../general_intersections/installed-qualification-36829763205.json",
          "review_scope": "artifact identity/payload; scientific acceptance uses independent analytic oracles in the bound qualification, not checksums"}
(ROOT / "payload-review.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "package_members": len(package), "artifacts": rows}))
