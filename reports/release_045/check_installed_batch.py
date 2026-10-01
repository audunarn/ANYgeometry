"""Run with isolated Python from the external release-review environment."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import zipfile

import anygeometry as geometry

prefix = Path(sys.prefix).resolve()
origin = Path(geometry.__file__).resolve()
assert origin.is_relative_to(prefix) and prefix != Path(sys.base_prefix).resolve()
assert not any((parent / ".git").exists() for parent in (prefix, *prefix.parents))
wheel = Path(sys.argv[1]).resolve(strict=True)
with zipfile.ZipFile(wheel) as archive:
    distribution = metadata.distribution("ANYgeometry")
    for name in archive.namelist():
        if name.startswith("anygeometry/"):
            assert Path(distribution.locate_file(name)).read_bytes() == archive.read(name), name
assert geometry.__version__ == "0.4.5"
assert geometry.EllipticArc and geometry.CylinderIntersectionCurve
model = geometry.GeometryModel()
first = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
second = model.add_plate(model.add_points(((2,1,-1),(2,3,-1),(2,3,1),(2,1,1))))
before = geometry.to_dict(model)
plan = geometry.plan_intersections(model, (first, second), policy=geometry.ConnectionIntent.CONNECT)
assert geometry.to_dict(model) == before
geometry.apply_intersections(model, plan, policy=geometry.ConnectionIntent.CONNECT)
assert model.validate_topology() == ()
charts = geometry.query_trimmed_surface_charts(model)
assert abs(sum(chart.material_area for chart in charts.charts) - 20.0) < 1e-9
after = geometry.to_dict(model)
assert geometry.apply_intersections(model, plan, policy=geometry.ConnectionIntent.CONNECT).reused
assert geometry.to_dict(model) == after
loaded = geometry.from_dict(after)
assert loaded.validate_topology() == () and geometry.to_dict(loaded) == after
result = {"status": "passed", "version": geometry.__version__, "origin": str(origin),
          "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
          "batch_read_only_plan": True, "topology_valid": True,
          "interior_ended_joint": True, "material_area": 20.0,
          "idempotence": True, "schema_roundtrip": True}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
