"""Real default-H loaded workload acceptance; no toy callable-root substitution."""
import numpy as np
import pytest
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model
from anygeometry._planning_stages import WorkerModel,stage_task
from anygeometry._planning_values import encode_value
from anygeometry.material_arrangement import ArrangementPath,MaterialDomain,arrange_material
from anygeometry.batch_intersections import _pair_paths
from anygeometry import query_trimmed_surface_charts


@pytest.mark.parametrize('warm',[False,True])
def test_actual_plane_pair_face_workload_under_unchanged_default_host(warm):
    # R5 trusts the pinned external runtime interfaces; pytest is not a root.
    # The former Plane-only branch-profile assertion is explicitly superseded
    # by package numerical closure, while actual pair/face checks stay intact.
    model,domains=_fixture();faces=tuple(model.faces)
    if warm:query_trimmed_surface_charts(model,faces)
    errors=np.geterr();callback=np.geterrcall();buffer=np.getbufsize()
    assert errors==dict(divide='warn',over='warn',under='ignore',invalid='warn')
    assert callback is None
    try:
        captured=manifest.capture_loaded_manifest(model=model,face_ids=faces)
    except Exception as error:
        # Preserve complete refusal and traversal accounting for each attempt.
        import json,os
        from pathlib import Path
        evidence=os.environ.get('TASK_EVIDENCE_DIRECTORY')
        if evidence:
            Path(evidence,'manifest-refusals-'+str(warm)+'.json').write_text(json.dumps(
                dict(error=str(error),nodes=getattr(error,'nodes',None),
                     graph=getattr(error,'graph',None),
                     completed_roles=getattr(error,'completed_roles',{}),
                     refusals=getattr(error,'refusals',[])),indent=2),encoding='utf-8')
        raise
    import json,os
    from pathlib import Path
    evidence=os.environ.get('TASK_EVIDENCE_DIRECTORY')
    assert captured.parsing_release['source_trees']>0
    compiled=captured.callables._immutable_evidence.compiled
    code_ids={key:tuple(map(id,row[3].values())) for key,row in compiled.items()}
    assert all(row[1] is None and row[2]=={} for row in compiled.values())
    if evidence:
        Path(evidence,'manifest-accepted-'+str(warm)+'.json').write_text(json.dumps(
            dict(nodes=captured.nodes,common=captured.callables.graph,
                 roles={name:item.graph for name,item in captured.roles.items()},
                 common_sha256=captured.callables.digest,parsing_release=captured.parsing_release),indent=2),encoding='utf-8')
    tolerance=model.tolerance.effective_length(5.)
    paths=tuple(ArrangementPath(path,owners=(domains[0].face_id,other.face_id))
        for other in domains[1:] for path in _pair_paths(domains[0],other,model,tolerance,lambda:None))
    result=arrange_material(domains[0],paths,tolerance=tolerance,
        area_tolerance=model.tolerance.effective_area(5.),cancellation_check=lambda:False)
    assert len(paths)==2 and len(result.cells)==4
    # Constructor/codec/stage boundaries are real, not monkeypatched away.
    context=WorkerModel(freeze_model(model))
    pair=dict(kind='pair',first=faces[0],second=faces[1],tolerance=tolerance,
        face_tolerances=[tolerance,tolerance],cost=1.)
    face=dict(kind='face',face=faces[0],inputs=encode_value((paths,())),
        tolerance=tolerance,area_tolerance=model.tolerance.effective_area(5.),seam='baseline',cost=1.)
    assert type(stage_task(context,pair)) is str
    assert type(stage_task(context,face)) is str
    try:
        captured.verify()
        captured.verify(full=True) # independent traversal oracle agrees after actual kernel warming
    except Exception as error:
        if evidence:
            Path(evidence,'manifest-recheck-'+str(warm)+'.json').write_text(json.dumps(
                dict(error=str(error),changes=getattr(error,'manifest_changes',None)),indent=2),encoding='utf-8')
        raise
    assert np.geterr()==errors and np.geterrcall() is callback and np.getbufsize()==buffer
    assert captured.wire['workload']['kind']=='package-numerical-v1'
    assert {key:tuple(map(id,compiled[key][3].values())) for key in code_ids}==code_ids
    assert all(row[1] is None and row[2]=={} for row in compiled.values())
