"""PP2 explicit small-model task schema; no implicit plan/stage discovery."""
from ._planning_executor import thaw_model
from ._planning_qualification import QualificationDecline
from .material_arrangement import MaterialDomain, ArrangementPath, arrange_material
from .batch_intersections import _pair_paths
from .definition_binding import definition_checksum


def model_task(packet, name):
    model = thaw_model(packet)
    faces = tuple(model.faces)
    if len(faces) != 3 or name not in ('pair', 'face'):
        raise QualificationDecline('PP2 diagnostic task requires exactly three faces')
    domains = tuple(MaterialDomain.from_model(model, face) for face in faces)
    # This is an explicit diagnostic workload, not the production task options.
    # Scale and all tolerances follow the PP1 canary fixture contract.
    tolerance = model.tolerance.effective_length(5.)
    paths = tuple(tuple(_pair_paths(domains[0], other, model, tolerance, lambda: None))
                  for other in domains[1:])
    if name == 'pair':
        return definition_checksum(paths)
    traces = tuple(ArrangementPath(path, owners=(domains[0].face_id, other.face_id))
                   for other, group in zip(domains[1:], paths) for path in group)
    result = arrange_material(domains[0], traces, tolerance=tolerance,
                              area_tolerance=model.tolerance.effective_area(5.),
                              cancellation_check=lambda: False)
    return definition_checksum(result)
