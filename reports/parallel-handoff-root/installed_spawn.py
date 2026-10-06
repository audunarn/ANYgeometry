"""Installed-wheel smoke; plain closure spawn, no meshing or source imports."""
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys


def worker(envelope, output):
    import anygeometry as g
    import numpy as np
    assert Path(g.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    closure = g.model_closure_from_dict(envelope)
    model = closure.working_model
    encoded = g.to_dict(model)
    decoded = g.from_dict(encoded)
    assert g.to_dict(decoded) == encoded
    assert g.model_closure_to_dict(closure) == envelope
    charts = g.query_trimmed_surface_charts_by_face(model)
    assert charts.results and all(row.error is None for row in charts.results)
    for row in charts.results:
        g.validate_trimmed_surface_charts_binding(model, row.charts)
    Path(output).write_text(json.dumps({"origin": g.__file__, "numpy": np.__version__,
        "mapping_count": len(closure.work_to_source), "source_model_id": str(closure.source_model_id),
        "working_model_id": str(model.model_id), "faces": len(charts.results),
        "envelope_sha256": hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest()}, indent=2))


def main():
    import anygeometry as g
    from anygeometry.structural import (AttachmentEvidence, AttachmentKind,
        AttachmentTargetKind, ConnectionIntent, JunctionKind, JunctionMemberUse, ParameterRange)
    assert Path(g.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    model = g.GeometryModel()
    for offset in (0., 20.):
        points = model.add_points(((offset, 0, 0), (offset+2, 0, 0), (offset+2, 1, 0), (offset, 1, 0)))
        face = model.add_plate(points)
        part = model.add_part(name=f"component-{offset}")
        sheet = model.add_sheet((face,), part_id=part)
        edge = model.faces[face].loop[0].edge
        member = model.add_member((edge,), part_id=part, orientation_reference=("vertex", points[0]))
        attachment = model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY,
            AttachmentTargetKind.EDGE, edge, ParameterRange(0., 1.), (ParameterRange(0., 1.),),
            connection_intent=ConnectionIntent.CONNECT, evidence=AttachmentEvidence.EXACT,
            max_residual=0., tolerance_used=model.tolerance.effective_coincidence(2.),
            part_id=part, sheet_id=sheet, lineage=(("edge", edge),))
        model.add_junction(JunctionKind.OVERLAP, (JunctionMemberUse(member, ParameterRange(0., 1.)),),
            sheet_ids=(sheet,), attachment_ids=(attachment,), connection_intent=ConnectionIntent.CONNECT)
    before = g.to_dict(model)
    partition = g.plan_independent_components(model, separation=.01)
    assert partition.certified and len(partition.components) == 2
    g.validate_component_partition_binding(model, partition)
    envelopes = [g.model_closure_to_dict(g.extract_model_closure(model, component.handles))
                 for component in partition.components]
    context = multiprocessing.get_context("spawn")
    for index, envelope in enumerate(envelopes):
        output = Path(f"spawn-{index}.json")
        process = context.Process(target=worker, args=(envelope, str(output)))
        process.start()
        process.join(15.)
        if process.is_alive():
            process.kill(); process.join()
            raise RuntimeError("installed spawn exceeded 15 seconds")
        assert process.exitcode == 0
        assert json.loads(output.read_text())["source_model_id"] == str(model.model_id)
    assert before == g.to_dict(model)
    print(json.dumps({"origin": g.__file__, "components": len(partition.components),
        "spawned": len(envelopes), "authored_unchanged": True, "prefix": sys.prefix}))


if __name__ == "__main__":
    main()
