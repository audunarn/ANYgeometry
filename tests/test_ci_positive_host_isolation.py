"""Fresh-H scheduler units; synthetic bootstrap never qualifies native planning."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import ci_isolated_tests as ci
from ci_planning_profile import REQUIRED_PROFILE_CASES,CONTRACT

def node(key):
    module,rest=key.split('::',1)
    return 'tests/'+module+'.py::'+rest

def test_q50_is_exact_authoritative_case_and_partitions_once():
    q=node('test_planning_r8_controller::test_real_public_callback_after_capture_is_terminal[exception]')
    ordinary='tests/test_plain.py::test_a';oracle='tests/test_oracle.py::test_a'
    groups={'core':[ordinary,q],'oracle':[oracle]}
    assert ci.split_required_positives(groups,required=False)==groups
    result=ci.split_required_positives(groups,required=True)
    assert result==dict(core=[ordinary],oracle=[oracle],planning_positive=[q])
    assert [n for names in result.values() for n in names]==[ordinary,oracle,q]
    assert groups=={'core':[ordinary,q],'oracle':[oracle]}
    assert ci.required_positive_key(q) in REQUIRED_PROFILE_CASES

def test_all22_exact_ids_have_one_fresh_phase():
    nodes=[node(k) for k in sorted(REQUIRED_PROFILE_CASES)]
    groups=ci.split_required_positives({'core':nodes,'oracle':[]},required=True)
    phases=ci.execution_phases(groups)
    assert len(phases)==22 and all(p['mode']=='core' and len(p['nodes'])==1 for p in phases)
    assert [p['nodes'][0] for p in phases]==nodes
    assert len({p['name'] for p in phases})==22

@pytest.mark.parametrize('groups',[{'core':['tests/a.py::a'],'oracle':['tests/a.py::a']},
    {'core':[],'oracle':[node(next(iter(REQUIRED_PROFILE_CASES)))]}])
def test_duplicate_or_oracle_positive_cannot_be_scheduled(groups):
    with pytest.raises(ValueError):ci.split_required_positives(groups,required=True)

def positive_xml(path,key,*,fault=None):
    module,name=key.split('::',1);suite=ET.Element('testsuite')
    case=ET.SubElement(suite,'testcase',classname='tests.'+module,name=name)
    props=ET.SubElement(case,'properties')
    for k,v in [('numpy_profile_contract',CONTRACT),('execution_scope','original-profile-assertions')]:
        ET.SubElement(props,'property',name=k,value=v)
    if fault in ('failure','error','skipped'):ET.SubElement(case,fault)
    elif fault=='refusal':ET.SubElement(props,'property',name='postcapture_qualification',value='False')
    elif fault=='property':props.clear()
    elif fault=='duplicate':suite.append(ET.fromstring(ET.tostring(case)))
    elif fault=='missing':suite.remove(case)
    elif fault=='wrong':case.set('classname','tests.other.test_'+module)
    ET.ElementTree(suite).write(path)

@pytest.mark.parametrize('fault',[None,'failure','error','skipped','refusal','property','duplicate','missing','wrong'])
def test_single_positive_requires_original_passed_case(tmp_path,fault):
    key=sorted(REQUIRED_PROFILE_CASES)[0];path=tmp_path/'one.xml'
    positive_xml(path,key,fault=fault)
    if fault is None:ci.verify_positive_junit(path,node(key))
    else:
        with pytest.raises(ValueError):ci.verify_positive_junit(path,node(key))

def test_one_driver_budget_is_not_renewed_between_phases(monkeypatch):
    clock=[100.]
    monkeypatch.setattr(ci.time,'monotonic',lambda:clock[0])
    budget=ci.PhaseBudget(1800)
    clock[0]+=100
    assert budget.remaining()==1700
    clock[0]+=1700
    with pytest.raises(TimeoutError):budget.remaining()


def mini_root(root,fault=None):
    def put(name,text):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path
    put('tools/ci_isolated_tests.py',Path(ci.__file__).read_text())
    put('tools/ci_durations.py',(TOOLS/'ci_durations.py').read_text())
    put('tools/ci_planning_profile.py',
        'import os\nREQUIRED_PROFILE_CASES='+repr(REQUIRED_PROFILE_CASES)+'\nCONTRACT='+repr(CONTRACT)+
        '\ndef preflight():\n assert os.environ.get("ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS")=="1"\n return {"scope":"MOCK-PROFILE-PROTOCOL-UNIT"}\n')
    put('tests/conftest.py',
        'import os,json,pathlib\ndef pytest_collection_finish(session):\n'
        ' p=os.environ.get("ANYGEOMETRY_SHARD_REPORT")\n if p:\n'
        '  nodes=[x.nodeid for x in session.items]\n'
        '  pathlib.Path(p).write_text(json.dumps(dict(index=1,count=1,selected=nodes,ran=nodes,slow=[])))\n')
    header='import os,json,pathlib,pytest\nLOG='+repr(str(root/'executions.jsonl'))+'\nHISTORY=[]\n'
    def body(name,label,extra=''):
        return ('def '+name+'(request):\n assert not HISTORY\n HISTORY.append(1)\n'
            ' with open(LOG,"a") as stream:stream.write(json.dumps({"label":'+repr(label)+',"pid":os.getpid()})+"\\n")\n'
            ' request.node.user_properties.extend([("numpy_profile_contract",'+repr(CONTRACT)+'),("execution_scope","original-profile-assertions")])\n'+extra)
    extra=' pytest.skip("intentional protocol unit")\n' if fault=='skipped' else (
        ' request.node.user_properties.append(("postcapture_qualification","False"))\n' if fault=='refusal' else '')
    put('tests/test_planning_numpy_context.py',header+
        body('test_capture_preserves_exact_effective_numpy_context','positive-a',extra)+
        body('test_loaded_manifest_binds_numpy_context_with_small_owned_root','positive-b'))
    put('tests/test_plain.py',header+body('test_a','core'))
    put('tests/test_oracle.py','import mpmath\n'+header+body('test_a','oracle'))
    return root/'tools/ci_isolated_tests.py'


@pytest.mark.parametrize('fault',[None,'skipped','refusal'])
def test_real_mini_driver_fresh_pids_union_and_failed_positive(tmp_path,fault):
    script=mini_root(tmp_path,fault)
    evidence=tmp_path/'evidence';report=tmp_path/'shard.json';merged=tmp_path/'all.xml'
    env=dict(os.environ,ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS='1',
        ANYGEOMETRY_TEST_SHARD='1/1',ANYGEOMETRY_SHARD_REPORT=str(report))
    result=subprocess.run([sys.executable,str(script),'--root',str(tmp_path),
        '--evidence-dir',str(evidence),'--junitxml',str(merged)],
        env=env,cwd=tmp_path,capture_output=True,text=True,timeout=35)
    (tmp_path/'driver-stdout.txt').write_text(result.stdout);(tmp_path/'driver-stderr.txt').write_text(result.stderr)
    assert (result.returncode==0)==(fault is None),result.stdout+result.stderr
    plan=json.loads((evidence/'plan.json').read_text())
    assert len(plan['assigned'])==4
    rows=[json.loads(line) for line in (tmp_path/'executions.jsonl').read_text().splitlines()]
    assert len(rows)==4 and len({row['pid'] for row in rows})==4
    assert len(plan['groups']['planning_positive'])==2
    cases=list(ET.parse(merged).iter('testcase'))
    assert len(cases)==4
    positive=sorted(evidence.glob('positive-*/positive-profile.json'))
    assert len(positive)==2
    assert all(json.loads(p.read_text())['scope']=='MOCK-PROFILE-PROTOCOL-UNIT' for p in positive)
    manifests=[json.loads(p.read_text()) for p in evidence.glob('positive-*/core-collection.json')]
    assert all(m['mpmath_loaded'] is False and m['positive_host_scope']=='fresh-H-single-authoritative-case' for m in manifests)
    if fault=='skipped':assert sum(c.find('skipped') is not None for c in cases)==1


def test_flag1_without_selected_positive_keeps_original_two_cohorts():
    groups={'core':['tests/test_plain.py::test_a'],'oracle':[]}
    assert ci.split_required_positives(groups,required=True)==groups
