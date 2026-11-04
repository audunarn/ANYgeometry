"""Inventory and failure-evidence contracts of interpreter-isolated CI cohorts."""
import json
import os
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import ci_isolated_tests as ci


def put(root,name,text):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    return path


def test_partition_follows_transitive_relative_and_dynamic_literal_imports(tmp_path):
    put(tmp_path,'tests/pkg/oracle.py','\ufefffrom importlib import import_module as load\nvalue=load("mpmath")\n')
    put(tmp_path,'tests/pkg/helper.py','from . import oracle\n')
    put(tmp_path,'tests/test_a.py','from pkg.helper import value\n')
    put(tmp_path,'tests/test_b.py','def test_b(): pass\n')
    put(tmp_path,'examples/oracle.py','import mpmath\n')
    put(tmp_path,'tests/test_c.py','from examples.oracle import value\n')
    nodes=['tests/test_a.py::test_a','tests/test_b.py::test_b','tests/test_c.py::test_c']
    groups,closure=ci.classify_nodes(tmp_path,nodes)
    assert groups=={'core':[nodes[1]],'oracle':[nodes[0],nodes[2]]}
    assert 'tests/pkg/oracle.py' in closure['oracle_modules']
    assert set(sum(groups.values(),[]))==set(nodes)


def test_named_reference_importer_and_conftest_closure_are_not_missed(tmp_path):
    put(tmp_path,'tests/ch3_reference.py','x=1\n')
    put(tmp_path,'tests/conftest.py','from ch3_reference import x\n')
    put(tmp_path,'tests/test_a.py','def test_a(): pass\n')
    groups,_=ci.classify_nodes(tmp_path,['tests/test_a.py::test_a'])
    assert groups=={'core':[],'oracle':['tests/test_a.py::test_a']}


@pytest.mark.parametrize('nodes',[['tests/a.py::x']*2,['../a.py::x'],['/a.py::x'],[1],[]])
def test_tampered_node_ids_are_rejected(nodes):
    with pytest.raises(ValueError):ci.validate_nodes(nodes)


def test_inventory_bound_is_checked_before_processing(monkeypatch):
    monkeypatch.setattr(ci,'MAX_NODES',1)
    with pytest.raises(ValueError):ci.validate_nodes(['tests/a.py::a','tests/b.py::b'])


def test_existing_windowed_decimal_selector_ids_fit_finite_metadata_bound():
    # Same long parameters as test_certified_hydrostatics_verify.py:1495-1496.
    prefix='tests/test_certified_hydrostatics_verify.py::test_windowed_decimal_grammar_matches_the_plain_grammar['
    nodes=[prefix+text+']' for text in ('9'*4096+'.'+'5'*4096+'.5','9'*8192+'.','.'+'9'*8192)]
    assert max(map(len,nodes))==8297
    assert ci.validate_nodes(nodes)==nodes
    assert ci.MAX_NODE_CHARS==16384 and ci.MAX_NODES==50000 and ci.MAX_JSON_BYTES==32*1024*1024
    with pytest.raises(ValueError):ci.validate_nodes(['tests/a.py::'+'x'*16384])


def xml(path,names,*,failure=False):
    root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite',name='pytest')
    for name in names:
        case=ET.SubElement(suite,'testcase',classname='tests.test_a',name=name,time='.1')
        props=ET.SubElement(case,'properties');ET.SubElement(props,'property',name='positive',value='original')
        if failure:ET.SubElement(case,'failure',message='actual failure').text='original trace'
    ET.ElementTree(root).write(path,encoding='utf-8')
    return path


def test_merge_preserves_real_failure_trace_and_positive_properties(tmp_path):
    a=xml(tmp_path/'a.xml',['test_a']);b=xml(tmp_path/'b.xml',['test_b'],failure=True)
    target=tmp_path/'all.xml'
    ci.merge_junit([a,b],['tests/test_a.py::test_a','tests/test_a.py::test_b'],target)
    cases=ET.parse(target).getroot().findall('.//testcase')
    assert len(cases)==2 and cases[1].find('failure').text=='original trace'
    assert cases[0].find('./properties/property').get('value')=='original'


@pytest.mark.parametrize('names',[['test_a'],['test_a','test_a'],['test_a','test_other']])
def test_merge_rejects_missing_duplicate_unknown_but_retains_actual_xml(tmp_path,names):
    a=xml(tmp_path/'a.xml',names);target=tmp_path/'all.xml'
    with pytest.raises(ValueError):
        ci.merge_junit([a],['tests/test_a.py::test_a','tests/test_a.py::test_b'],target)
    assert len(ET.parse(target).getroot().findall('.//testcase'))==len(names)


@pytest.mark.parametrize('statement',[
    'import mpmath',
    'import importlib; importlib.import_module("mpmath")',
    'import pytest; pytest.importorskip("mpmath")',
])
def test_core_import_attempt_is_runtime_error_never_skip(statement):
    code='import sys;sys.path.insert(0,'+repr(str(Path(ci.__file__).parent))+');import ci_isolated_tests as ci;ci.install_core_guard();'+statement
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'RuntimeError: mpmath import attempted' in result.stderr
    assert 'Skipped' not in result.stdout+result.stderr


def test_real_selected_cases_keep_shard_assignment_and_complete_actual_union(tmp_path):
    import ci_shards
    root=Path(ci.__file__).resolve().parents[1]
    nodes=['tests/test_ci_tools.py::test_node_ids_follow_pytest',
           'tests/test_ch3_reference_root_solver.py::test_actual_ci_nearly_double_coefficients_do_not_need_iterative_convergence']
    report=tmp_path/'shard-1.json';output=tmp_path/'kernel-results.xml';evidence=tmp_path/'phases'
    env=dict(os.environ,ANYGEOMETRY_TEST_SHARD='1/1',ANYGEOMETRY_SHARD_REPORT=str(report))
    command=[sys.executable,'-B',str(ci.__file__),'--root',str(root),'--evidence-dir',str(evidence),
             '--junitxml',str(output)]
    for node in nodes:command+=['--select',node]
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=40)
    (tmp_path/'stdout.txt').write_text(result.stdout);(tmp_path/'stderr.txt').write_text(result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    original=json.loads(report.read_text());plan=json.loads((evidence/'plan.json').read_text())
    assert set(original['ran'])==set(nodes) and plan['assigned']==original['ran']
    assert plan['groups']=={'core':[nodes[0]],'oracle':[nodes[1]]}
    core=json.loads((evidence/'core-collection.json').read_text())
    assert core['mpmath_loaded'] is False and core['collection_mode']=='directory-walk'
    assert json.loads((evidence/'inventory.json').read_text())['collection_mode']=='directory-walk'
    assert json.loads((evidence/'oracle-collection.json').read_text())['mpmath_loaded'] is True
    assert ci_shards.verify([report])==[]


def test_actual_collection_cannot_silently_deselect_an_assigned_id(tmp_path):
    root=tmp_path;manifest=tmp_path/'actual.json'
    put(root,'tests/test_tiny.py','def test_a(): pass\ndef test_b(): pass\n')
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(root=str(root),mode='core',
        nodes=['tests/test_tiny.py::test_a'],inventory=['tests/test_tiny.py::test_a','tests/test_tiny.py::test_b'],
        manifest=str(manifest),args=['--rootdir='+str(root),'--collect-only','-k','definitely_not_a_selected_name'])))
    env=dict(os.environ);env.pop('ANYGEOMETRY_TEST_SHARD',None);env.pop('ANYGEOMETRY_SHARD_REPORT',None)
    result=subprocess.run([sys.executable,'-B',str(ci.__file__),'--child',str(config)],
                          cwd=root,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0
    assert 'actual collected set/count differs' in result.stdout+result.stderr
    assert json.loads(manifest.read_text())['nodes']==[]


def test_module_inventory_filter_preserves_assignment_and_order():
    a,b,c=['tests/a.py::'+name for name in ('a','b','c')]
    assert ci.select_manifest_ids([a,b,c],[c,a],[a,b,c])==[c,a]


@pytest.mark.parametrize('actual',[
    ['tests/a.py::a'],['tests/a.py::a','tests/a.py::a'],['tests/a.py::a','tests/a.py::other'],
])
def test_module_inventory_drift_is_rejected_before_filtering(actual):
    with pytest.raises(RuntimeError,match='module inventory changed'):
        ci.select_manifest_ids(actual,['tests/a.py::a'],['tests/a.py::a','tests/a.py::b'])


def test_bootstrap_preserves_checkout_namespace_imports_at_real_collection(tmp_path):
    root=Path(ci.__file__).resolve().parents[1]
    env=dict(os.environ,ANYGEOMETRY_TEST_SHARD='1/1',ANYGEOMETRY_SHARD_REPORT=str(tmp_path/'shard.json'))
    command=[sys.executable,'-B',str(ci.__file__),'--collect-only','--root',str(root),
        '--evidence-dir',str(tmp_path/'phases'),'--select','tests/test_authored_constraint_scope.py']
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=40)
    (tmp_path/'stdout.txt').write_text(result.stdout);(tmp_path/'stderr.txt').write_text(result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    actual=json.loads((tmp_path/'phases/core-collection.json').read_text())
    assert actual['nodes'] and actual['mpmath_loaded'] is False


@pytest.mark.parametrize('flag',['-k','-m'])
def test_real_driver_selector_retains_unfiltered_inventory_and_exact_assignment(tmp_path,flag):
    root=tmp_path;tools=str(Path(ci.__file__).parent)
    put(root,'pytest.ini','[pytest]\nmarkers = chosen: selector regression\n')
    put(root,'tests/test_filter.py','import pytest\n@pytest.mark.chosen\ndef test_chosen(): pass\ndef test_other(): assert False\n')
    # Use the unchanged real shard algorithm after pytest's original selection.
    put(root,'tests/conftest.py',f'''
import json,os,sys
from pathlib import Path
sys.path.insert(0,{tools!r})
import ci_shards
def pytest_collection_finish(session):
    report=os.environ.get('ANYGEOMETRY_SHARD_REPORT')
    if not report:return
    index,count=ci_shards.parse_shard(os.environ.get('ANYGEOMETRY_TEST_SHARD','1/1'))
    selected=[item.nodeid for item in session.items]
    owner=ci_shards.assign(selected,count,ci_shards.load_table())
    session.items[:]=[item for item in session.items if owner[item.nodeid]==index]
    Path(report).write_text(json.dumps(dict(index=index,count=count,selected=sorted(selected),
        ran=[item.nodeid for item in session.items],slow=[])))
''')
    env=dict(os.environ,ANYGEOMETRY_TEST_SHARD='1/1',ANYGEOMETRY_SHARD_REPORT=str(root/'shard.json'))
    result=subprocess.run([sys.executable,'-B',str(ci.__file__),'--root',str(root),
        '--evidence-dir','phases','--junitxml','results.xml',flag,'chosen'],
        cwd=root,env=env,capture_output=True,text=True,timeout=30)
    (root/'stdout.txt').write_text(result.stdout);(root/'stderr.txt').write_text(result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    plan=json.loads((root/'phases/plan.json').read_text())
    assert plan['assigned']==['tests/test_filter.py::test_chosen']
    assert set(plan['module_inventory'])=={'tests/test_filter.py::test_chosen','tests/test_filter.py::test_other'}
    actual=json.loads((root/'phases/core-collection.json').read_text())
    assert actual['nodes']==plan['assigned'] and actual['mpmath_loaded'] is False
    assert len(ET.parse(root/'results.xml').getroot().findall('.//testcase'))==1
