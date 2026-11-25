"""The display-tessellation acceptance corpus loads and every face lies on its support (plan S0)."""
import numpy as np
import pytest

import display_tessellation_corpus as corpus


@pytest.mark.parametrize("name", sorted(corpus.ALL))
def test_corpus_case_builds_valid_faces_on_their_supports(name):
    case = corpus.ALL[name]()
    g = case.model
    assert g.faces or "edge" in case.info
    for face_id in case.faces.values():
        face = g.faces[face_id]
        support = face.surface
        assert support is not None
        t = np.linspace(0., 1., 33)
        for loop in (face.loop, *face.holes):
            for use in loop:
                points = g.evaluate_edge_many(use.edge, t)
                back = np.array([support.evaluate(*support.local_uv(p)) for p in points])
                scale = max(1.0, float(np.abs(points).max()))
                assert np.abs(back - points).max() <= 1e-9 * scale, (name, face_id, use.edge)


def test_tripod_fixture_is_the_menu_example_skins():
    case = corpus.tripod_skins()
    assert len(case.faces) == 18
    shared = [e for e in case.model.edges if len(case.model.faces_using_edge(e)) > 1]
    assert len(shared) == 33
