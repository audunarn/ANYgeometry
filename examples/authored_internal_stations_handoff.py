"""Portable paired-line station evidence; no mesh or subdivision permission."""
from fractions import Fraction

import numpy as np

from examples.authored_child_coverage_handoff import build
from anygeometry import (
    query_prepared_authored_internal_stations,
    validate_prepared_authored_internal_station_coordinates,
    query_prepared_vertex_preimages, validate_prepared_vertex_preimages_binding,
)


def verify():
    model, correspondence, left, right = build()
    edge, incidence = correspondence.interior_incidence[0]
    receipt = query_prepared_authored_internal_stations(model, correspondence, edge,
                                                       (0, Fraction(1, 3), 1))
    xyz = np.array([[float(Fraction(*value)) for value in point] for point in receipt.current_points])
    validate_prepared_authored_internal_station_coordinates(model, receipt, xyz)
    assert receipt.child_incidence == incidence
    assert set(face for face, _ in incidence) == {left, right}
    assert all(Fraction(*point[0]) == 3 for point in receipt.authored_points)
    vertices = query_prepared_vertex_preimages(model)
    validate_prepared_vertex_preimages_binding(model, vertices)
    assert set(receipt.endpoint_ids) <= set(vertices.without_authored_vertex)
    return {'internal_edge': edge, 'children': (left, right), 'parameters': receipt.parameters,
            'decomposition_seam_tag': receipt.decomposition_seam_tag,
            'endpoint_ids': receipt.endpoint_ids,
            'endpoints_without_original_vertex': True,
            'source_edge_references_not_excluded': True,
            'accepted_mesh': False, 'subdivision_permission': False}


if __name__ == '__main__':
    print(verify())
