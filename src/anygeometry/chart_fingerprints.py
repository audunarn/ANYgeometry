"""Read-only content tokens for public chart and material-region results."""

from .definition_binding import definition_checksum
from .errors import GeometryError
from .material_regions import MaterialSurfaceRegion, MaterialSurfaceRegions
from .trimmed_charts import TrimmedSurfaceCharts


def chart_definition_fingerprint(result):
    """Return an opaque snapshot token for a public chart/region definition.

    Accepts ``TrimmedSurfaceCharts``, ``MaterialSurfaceRegions`` and a selected
    ``MaterialSurfaceRegion``. All definition fields, including nested supports,
    trim paths and reference handles, participate. Collection tokens also bind
    their stored owner identity, revision and source checksum. Implementation
    caches explicitly excluded by the owner do not participate.

    This function inspects only the supplied result, never its live model. A
    token is NOT a source validation or meshing certificate. Validate the result
    with its public owner binding API as well; retain the original token across
    callbacks to detect a refreshed definition. Compare tokens only for equality
    within the same ANYgeometry implementation, not as document/release hashes.
    """
    if not isinstance(result, (TrimmedSurfaceCharts, MaterialSurfaceRegions,
                               MaterialSurfaceRegion)):
        raise TypeError("chart fingerprint requires a public chart or material-region result")
    try:
        return "chart-definition-v1:" + definition_checksum(result)
    except (TypeError, ValueError, RecursionError) as error:
        raise GeometryError("chart definition cannot be fingerprinted") from error
