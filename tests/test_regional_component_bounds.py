from anygeometry import GeometryModel, plan_independent_components


def test_opt_in_bounds_for_untrimmed_bilinear_faces():
    model = GeometryModel()
    faces = []
    for x in (0.0, 5.0):
        vertices = model.add_points([(x, 0, 0), (x+1, 0, 0), (x+1, 1, 0.1), (x, 1, 0)])
        faces.append(model.add_face(model.add_polyline(vertices, close=True)))
    source = model.revision
    assert not plan_independent_components(model, face_ids=faces).certified
    plan = plan_independent_components(model, face_ids=faces, allow_untrimmed_bilinear=True)
    assert plan.certified
    assert [tuple(group.face_ids) for group in plan.components] == [(faces[0],), (faces[1],)]
    assert model.revision == source
