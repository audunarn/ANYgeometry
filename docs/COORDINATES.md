# Model-local and world coordinates

The public helpers `model_to_world_points`, `world_to_model_points`,
`model_to_world_vectors` and `world_to_model_vectors` are available from
`anygeometry` and `anygeometry.coordinates`. They are read-only queries.

## Convention

Let T be `model.coordinate_transform`, or the identity when absent. For a
model-local point p, compute h = T @ [p.x, p.y, p.z, 1] and return h[:3]/h[3].
The inverse uses the complete inverse of T with the same homogeneous division.
This retains existing automation behavior even for near-affine bottom rows
admitted by the existing document setter. A zero/near-zero homogeneous weight,
unusable inverse or nonfinite result fails with `GeometryError`.

Vectors use A = T[:3, :3], or its inverse, without translation or normalization.
They are ordinary vectors/directions, **not surface normals**. Normal mapping
and CRS reprojection are outside this API.

Both frames use `model.units`; translations in T use those same units. The
helpers perform no unit conversion. `local_origin` and `crs_metadata` remain
metadata and are neither added/subtracted nor interpreted as transformations.
World means this stored affine frame, not a promise of a geographic CRS.

Inputs are finite real NumPy-compatible arrays shaped (..., 3), including a
single (3,) point and empty batches. Outputs are new float arrays of the same
shape. Inputs, model revision, IDs, geometry stores and caches are unchanged.
Malformed inputs raise `GeometryError`; no partial batch is returned.

## Example in millimetres

```python
import numpy as np
from anygeometry import GeometryModel, model_to_world_points, world_to_model_points

model = GeometryModel()
transform = np.eye(4)
transform[:3, 3] = (10000, 20000, 30000)
model.set_document_settings(units="mm", coordinate_transform=transform,
                            local_origin=(500, 600, 700))
world_mm = model_to_world_points(model, (1000, 0, 0))
assert np.array_equal(world_mm, (11000, 20000, 30000))
assert np.array_equal(world_to_model_points(model, world_mm), (1000, 0, 0))
world_metres = world_mm / 1000  # Explicit output-unit conversion, exactly once.
```

Automation converts unit-bearing inputs into model units before using this
mapping. Its `model_local`/`world` names, unit rules and protocol version 1 stay
unchanged. Geometry schema 4 gains no fields or migration requirement.

These APIs are maintenance additions under Unreleased. The published 0.4.3
wheel does not acquire them retroactively. CAD owners must accept an exact
updated source/artifact before adopting the [owner handoff](OWNER_HANDOFFS.md).
