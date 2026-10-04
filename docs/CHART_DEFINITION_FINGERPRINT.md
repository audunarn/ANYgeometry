# Pinning a chart definition across callbacks

`anygeometry.chart_definition_fingerprint(result)` returns an opaque content
token for `TrimmedSurfaceCharts`, `MaterialSurfaceRegions`, or one selected
`MaterialSurfaceRegion`. It covers every definition field, including nested
supports, trim paths, ownership/reference handles, constraints and attachments.
Collection tokens also include the stored model identity, revision and source
checksum. Owner-marked implementation caches are excluded.

The query is read-only, has no model argument and does not serialize or inspect
the live model. It does not validate the result against that model. An unchanged
token alone does not establish source currency, completeness, meshing permission
or quality. A selected region token does not include its enclosing collection's
source checksum; retain and validate the enclosing collection too.

For a consumer-selected reference metric, capture the collection token before
calling an owner derivative evaluator. Keep that original token across all
callbacks and compare it after the operation, alongside public owner binding
validation and the consumer's selected face/region and metric checks. For a
selected region, retain its token as well. Recomputing and replacing the original
token after a callback would admit a refreshed support under an old metric.

```python
from anygeometry import (
    chart_definition_fingerprint,
    validate_trimmed_surface_charts_binding,
)

entry = chart_definition_fingerprint(charts)
validate_trimmed_surface_charts_binding(model, charts)
# Use public owner evaluation and retain the original entry across callbacks.
validate_trimmed_surface_charts_binding(model, charts)
if chart_definition_fingerprint(charts) != entry:
    raise RuntimeError("selected chart definition changed")
```

Compare tokens only for equality within the same ANYgeometry implementation.
Do not interpret the digest, persist it as a cross-version compatibility promise,
or substitute it for document checksums and artifact identities. Unsupported
input types raise `TypeError`; unrepresentable definition content (including
nonfinite values) raises `GeometryError`. This API changes no document schema,
geometry classification, chart selection, numerical tolerance or release version.
