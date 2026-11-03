"""Small real kernel tasks for PP1b startup qualification; no global model."""
from .model import GeometryModel
from .batch_intersections import _pair_paths
from .material_arrangement import MaterialDomain, ArrangementPath, arrange_material
from .definition_binding import definition_checksum


def _fixture():
    model = GeometryModel(model_id='132c674e-c5a8-41a9-973c-6e87a8b71092')
    faces = tuple(model.add_plate(model.add_points(points)) for points in (
        ((0.,0.,0.), (5.,0.,0.), (5.,5.,0.), (0.,5.,0.)),
        ((0.,2.5,-2.5), (5.,2.5,-2.5), (5.,2.5,2.5), (0.,2.5,2.5)),
        ((2.5,0.,-2.5), (2.5,5.,-2.5), (2.5,5.,2.5), (2.5,0.,2.5))))
    return model, tuple(MaterialDomain.from_model(model, face) for face in faces)


def pair_canary():
    model, domains = _fixture()
    paths = tuple(_pair_paths(domains[0], other, model,
                  model.tolerance.effective_length(5.), lambda: None) for other in domains[1:])
    if tuple(map(len, paths)) != (1, 1):
        raise ValueError('canary pair task count changed')
    return definition_checksum(paths)


def face_canary():
    model, domains = _fixture()
    paths = tuple(ArrangementPath(path, owners=(domains[0].face_id, other.face_id))
                  for other in domains[1:]
                  for path in _pair_paths(domains[0], other, model,
                      model.tolerance.effective_length(5.), lambda: None))
    result = arrange_material(domains[0], paths,
                              tolerance=model.tolerance.effective_length(5.),
                              area_tolerance=model.tolerance.effective_area(5.),
                              cancellation_check=lambda: False)
    if len(result.cells) != 4:
        raise ValueError('canary face cell count changed')
    return definition_checksum(result)


def startup_canaries():
    return {'pair': pair_canary, 'face': face_canary}
