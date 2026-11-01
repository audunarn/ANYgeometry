"""Real partial pair-sample coverage: cold supports must stay cold where legacy does."""
from anygeometry import batch_intersections as bi
from anygeometry.model import GeometryModel
from anygeometry._planning_host import StageControl
from anygeometry._planning_executor import read_packet
from anygeometry.serialization import to_dict,from_dict


def test_disjoint_pairs_replay_sample_dependent_support_state(tmp_path,monkeypatch):
    model=GeometryModel(model_id='132c674e-c5a8-41a9-973c-6e87a8b71092')
    for i in range(8):
        x=float(i*10)
        for points in (((x,0.,0.),(x+5.,0.,0.),(x+5.,5.,0.),(x,5.,0.)),
                       ((x,2.5,-2.5),(x+5.,2.5,-2.5),(x+5.,2.5,2.5),(x,2.5,2.5))):
            model.add_plate(model.add_points(points))
    document=to_dict(model);reference=from_dict(document)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    serial=bi.plan_intersections(reference,tuple(reference.faces),policy='connect')
    bi.apply_intersections(reference,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(reference)['checksum']['value'])
    def observed(plan):
        target=from_dict(document);bi.apply_intersections(target,plan,policy='connect')
        return plan.content_checksum,to_dict(target)['checksum']['value']
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    controls=[]
    for reuse in (False,True):
        source=from_dict(document)
        control=StageControl(tmp_path/('session' if reuse else 'legacy'),workers=2,
            expected_ci=expected,observed_ci=observed,reuse_workers=reuse)
        plan=bi.plan_intersections(source,tuple(source.faces),policy='connect',_parallel_control=control)
        assert control.mode=='parallel',control.decline_reason
        assert control.ci_verified and plan.content_checksum==serial.content_checksum
        assert to_dict(source)==document
        controls.append(control)
    old,new=controls
    assert len(new.reports[0]['sampling']['indices'])<len(new.reports[0]['results'])==8
    for a,b in zip(old.reports,new.reports):
        assert a['request_sha256']==b['sampling_sha256']
        assert a['sampling']['indices']==b['sampling']['indices']
    legacy=read_packet(old.directory/'face/request.json','request')['tasks']
    reused=read_packet(new.directory/'session/command-1.json','session-command')['tasks']
    assert legacy==reused
    lengths=[len(task['support'][4]) for task in reused]
    assert set(lengths)=={0,1},'fixture must exercise both sampled warm and unsampled cold supports'
