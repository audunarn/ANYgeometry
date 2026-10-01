"""Standalone installed-package checks; copied outside the checkout by the runner.

No pytest or source-tree imports are required. Every failure is retained in the
JSON report, and a failed required case gives the process a nonzero exit code.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile


def verify_environment(config):
    prefix = Path(sys.prefix).resolve()
    assert prefix == Path(config["environment"]).resolve(), (prefix, config["environment"])
    assert Path(sys.executable).resolve().is_relative_to(prefix)
    assert sys.prefix != sys.base_prefix
    origins = {}
    for distribution, module_name in config["modules"].items():
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(prefix), (module_name, origin)
        dist = metadata.distribution(distribution)
        assert Path(dist.locate_file("")).resolve().is_relative_to(prefix)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        assert not direct.get("dir_info", {}).get("editable"), distribution
        origins[module_name] = str(origin)
    for name, version in config["versions"].items():
        assert metadata.version(name) == version, (name, metadata.version(name), version)
    if "mcp" in config["modules"]:
        assert metadata.version("mcp").split(".", 1)[0] == "2", metadata.version("mcp")

    wheel = Path(config["wheel"])
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert digest == config["wheel_sha256"], "input wheel changed"
    dist = metadata.distribution("ANYgeometry")
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            installed = Path(dist.locate_file(name)).resolve()
            assert installed.is_relative_to(prefix), name
            assert installed.is_file(), name
            assert hashlib.sha256(installed.read_bytes()).digest() == hashlib.sha256(archive.read(name)).digest(), name
            checked += 1
    assert checked > 0
    return {"origins": origins, "wheel_sha256": digest, "verified_files": checked}


def core(work, config):
    import anygeometry as ag
    from anygeometry.serialization import VERSION
    from anygeometry.generators import plate
    assert VERSION in (4, 5)  # Released control is schema 4; development is 5.
    assert Path(ag.__file__).with_name("py.typed").is_file()
    assert subprocess.check_output([sys.executable, "-I", "-m", "anygeometry", "--version"],
                                   cwd=work, text=True).strip() == ag.__version__
    model = plate(2., 1.)
    document = ag.to_dict(model)
    path = work / "roundtrip.json.gz"
    ag.write_geometry(path, model)
    assert ag.to_dict(ag.read_geometry(path)) == document
    if config.get("coordinates"):
        import numpy as np
        transform = np.eye(4)
        transform[:3, 3] = (10000., 20000., 30000.)
        model.set_document_settings(units="mm", coordinate_transform=transform, local_origin=(11., 22., 33.))
        before = ag.to_dict(model)
        np.testing.assert_array_equal(ag.model_to_world_points(model, [1000., 0., 0.]), [11000., 20000., 30000.])
        np.testing.assert_array_equal(ag.world_to_model_points(model, [11000., 20000., 30000.]), [1000., 0., 0.])
        np.testing.assert_array_equal(ag.model_to_world_vectors(model, [1., 2., 3.]), [1., 2., 3.])
        np.testing.assert_array_equal(ag.world_to_model_vectors(model, [1., 2., 3.]), [1., 2., 3.])
        assert ag.to_dict(model) == before


def planar(work, config):
    import anygeometry as ag
    available = importlib.util.find_spec("shapely") is not None
    assert available == config["planar"], "planar dependency presence differs from requested mode"
    model = ag.GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    edge = model.add_line(*model.add_points(((-1, .5, 0), (2, .5, 0))))
    result = ag.query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    if not available:
        assert result.kind is ag.IntersectionKind.CAPABILITY_MISSING
        assert not result.classified
        assert result.diagnostics == ("planar_backend_unavailable",)
    else:
        assert result.classified
        assert result.kind is not ag.IntersectionKind.CAPABILITY_MISSING
        overlap = ag.GeometryModel()
        first = overlap.add_plate(overlap.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))))
        second = overlap.add_plate(overlap.add_points(((1, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0))))
        plan = ag.plan_coplanar_fragmentation(overlap, [first, second],
                                            ownership_policy=ag.OverlapOwnershipPolicy.FIRST_SELECTED)
        ag.apply_coplanar_fragmentation(overlap, plan)
        assert not overlap.validate_topology()


def mesher(work, config):
    import anygeometry as ag
    import anymesher as am
    from anygeometry.generators import plate

    def mesh_and_check(model):
        before = ag.to_dict(model)
        mesh = am.generate_mesh(model, backend="mapped", target_size=.5)
        assert mesh.num_nodes > 0 and mesh.num_elements > 0
        assert str(mesh.geometry_model_id) == str(model.model_id)
        assert mesh.geometry_revision == model.revision
        assert ag.to_dict(model) == before
        for face in model.faces:
            assert mesh.elements_on(model.handle("face", face))
        return mesh

    model = plate(2., 1.)
    old_face = model.group("shell")[0]
    mesh_and_check(model)
    ag.split_face_at(model, old_face.id, axis=0, fraction=.5)
    descendants = model.resolve_ref(old_face)
    assert len(descendants) == 2
    changed = mesh_and_check(model)
    assert {element for face in descendants for element in changed.elements_on(face)} == set(changed.shells)

    # Exact quarter-cylinder carrier, edited by a public translation before mesh.
    curved = ag.GeometryModel()
    a, b, c = curved.add_points(((1., 0., 0.), (2**-.5, 2**-.5, 0.), (0., 1., 0.)))
    arc = curved.add_arc(a, b, c)
    face = curved.extrude((arc,), (0., 0., 2.))[0]
    ag.translate_entities(curved, [curved.entity_ref("face", face)], (3., 4., 5.))
    mesh_and_check(curved)

    feature_model = ag.GeometryModel()
    feature_model.features.capture_baseline(feature_model)
    feature = feature_model.features.append("generator.plate", parameters={"length": 2., "width": 1.})
    assert feature_model.regenerate_features().success
    downstream_refs = tuple(feature_model.features.get(feature.feature_id).outputs.values())
    source_id, source_revision = feature_model.model_id, feature_model.revision
    mesh_and_check(feature_model)
    feature_model.features.update(feature.feature_id, parameters={"length": 3., "width": 1.})
    assert feature_model.regenerate_features().success
    assert feature_model.model_id == source_id and feature_model.revision > source_revision
    assert downstream_refs and all(feature_model.resolve_ref(ref) for ref in downstream_refs)
    remesh = mesh_and_check(feature_model)
    outputs = feature_model.features.get(feature.feature_id).outputs.values()
    faces = [ref for output in outputs for ref in feature_model.resolve_ref(output) if ref.kind == "face"]
    assert faces and all(remesh.elements_on(ref) for ref in faces)
    owners = ag.feature_entity_owners(feature_model)
    for previous in downstream_refs:
        for current in feature_model.resolve_ref(previous):
            assert owners[current] == feature.feature_id
            if current.kind == "face":
                assert remesh.elements_on(current)
    assert feature_model.features.get(feature.feature_id).feature_id == feature.feature_id


def fem(work, config):
    from anyfem.document import DocumentSession
    from anyfem.io import project_from_dict, project_to_dict
    from anyfem.model.project import Project
    from anygeometry.generators import plate
    project = Project("installed geometry compatibility")
    project.geometry = plate(2., 1.)
    before = project_to_dict(project)
    restored = project_from_dict(json.loads(json.dumps(before)))
    assert project_to_dict(restored) == before
    session = DocumentSession(restored)
    try:
        with session.transaction("deliberate rollback"):
            restored.geometry.add_point(9., 8., 7.)
            raise RuntimeError("rollback sentinel")
    except RuntimeError as error:
        assert str(error) == "rollback sentinel"
    assert project_to_dict(restored) == before
    assert session.commands.project is restored


def fileio(work, config):
    import anymaterial
    import anymesher
    from anyfileio import read_sesam_semantics
    # Minimal owner-neutral SESAM records, no native CAD provider involved.
    path = work / "triangle.FEM"
    path.write_text("\n".join((
        "IDENT          100               1", "UNITS            1               1               1",
        "MISOSEL          1  2.100000D+11  3.000000D-01  7.850000D+03",
        "GELTH           10  2.000000D-02",
        "GCOORD           1               0               0               0",
        "GCOORD           2               1               0               0",
        "GCOORD           3               0               1               0",
        "GELMNT1        100               0              25               0               1               2               3",
        "GELREF1        100               1              10", "IEND", "")), encoding="ascii")
    result = read_sesam_semantics(path)
    assert isinstance(result.mesh, anymesher.Mesh)
    assert result.mesh.num_nodes == 3 and result.mesh.num_elements == 1
    assert result.mesh.tris[100] == (1, 2, 3)
    assert result.thickness_of_element[100] == .02
    assert result.material_of_element[100] == 1
    assert isinstance(result.materials[1], anymaterial.MaterialSpec)
    assert result.materials[1].constants["elastic_modulus"] == 2.1e11


def mcp(work, config):
    import anygeometry as ag
    from anygeometry_mcp import GeometryAutomationRuntime
    from anygeometry_mcp.server import build_server
    path = work / "automation.json"
    ag.write_geometry(path, ag.GeometryModel())
    runtime = GeometryAutomationRuntime(path)
    server = build_server(runtime)
    # Use SDK discovery, not the SDK 1.x private tool-manager implementation.
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"kernel_capabilities", "model_summary", "select_entities",
                                           "describe_entities", "query_geometry", "plan_edit", "apply_edit"}
    assert runtime.kernel_capabilities()["result"]["protocol_version"] == 1
    def batch():
        return {"protocol_version": 1, "request_id": "create", "model_id": str(runtime.model.model_id),
                "expected_revision": runtime.model.revision,
                "commands": [{"name": "p", "operation": "create_point", "arguments": {
                    "position": {"value": [1, 2, 3], "unit": "m", "frame": "model_local"}}}]}
    plan = runtime.plan_edit(batch(), session_id="first")["result"]
    request = {"protocol_version": 1, "request_id": "apply", "model_id": str(runtime.model.model_id),
               "expected_revision": runtime.model.revision, "plan": plan, "approved": False}
    assert runtime.apply_edit(request, session_id="first")["error"]["code"] == "APPROVAL_REQUIRED"
    request["approved"] = True
    assert runtime.apply_edit(dict(request, write_back=True), session_id="first")["error"]["code"] == "WRITE_BACK_DISABLED"
    first = runtime.apply_edit(request, session_id="first")
    assert first["ok"]
    assert runtime.apply_edit(request, session_id="first") == first
    assert runtime.apply_edit(request, session_id="other")["error"]["code"] == "STALE_REVISION"
    assert ag.read_geometry(path).revision == 0

    runtime = GeometryAutomationRuntime(path, allow_write_back=True)
    plan = runtime.plan_edit(batch())["result"]
    request = dict(request, request_id="save", model_id=str(runtime.model.model_id),
                   expected_revision=runtime.model.revision, plan=plan, write_back=True)
    assert runtime.apply_edit(request)["ok"]
    restored = GeometryAutomationRuntime(path, allow_write_back=True)
    assert restored.model.revision == 1 and len(restored.model.vertices) == 1
    request["expected_revision"] = restored.model.revision
    assert restored.apply_edit(request)["error"]["code"] == "STALE_PLAN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    work = args.config.resolve().parent
    report = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
              "executable": sys.executable, "distributions": sorted(
                  ({"name": d.metadata["Name"], "version": d.version} for d in metadata.distributions()),
                  key=lambda d: d["name"].lower()), "cases": []}
    checks = {"core": core, "planar": planar, "mesher": mesher, "fem": fem, "fileio": fileio, "mcp": mcp}
    for name in ["environment_before", *config["checks"], "environment_after"]:
        try:
            detail = verify_environment(config) if name.startswith("environment_") else checks[name](work, config)
            report["cases"].append({"name": name, "status": "passed", "detail": detail})
        except Exception:
            report["cases"].append({"name": name, "status": "failed", "traceback": traceback.format_exc()})
            # Never run consumer code if the initial environment is contaminated.
            if name == "environment_before":
                break
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    (work / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "failed": [c["name"] for c in report["cases"] if c["status"] != "passed"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
