"""Receipt vectors bind ordered task identities and exact existing costs."""
import pytest


def test_ordered_cost_vector_matches_exact_known_tasks():
    from anygeometry._planning_code import stage_cost_vector
    rows=[dict(kind='pair',first=1,second=2,cost=17.),dict(kind='pair',first=2,second=3,cost=5.)]
    vector=stage_cost_vector('pair',rows)
    assert vector['count']==2 and vector['total_weight']==22.
    assert vector['rows']==[[1,2,'4031000000000000'],[2,3,'4014000000000000']]
    assert stage_cost_vector('pair',list(reversed(rows)))['sha256']!=vector['sha256']


@pytest.mark.parametrize('mutation',['cost','id','duplicate','kind','nonfinite'])
def test_cost_vector_refuses_untyped_or_duplicate_tasks(mutation):
    from anygeometry._planning_code import stage_cost_vector
    from anygeometry._planning_qualification import QualificationDecline
    tasks=[dict(kind='face',face=3,cost=5.)]
    if mutation=='cost':tasks[0]['cost']=True
    elif mutation=='id':tasks[0]['face']=True
    elif mutation=='duplicate':tasks*=2
    elif mutation=='kind':tasks[0]['kind']='pair'
    else:tasks[0]['cost']=float('nan')
    with pytest.raises(QualificationDecline):stage_cost_vector('face',tasks)
