"""Preserve original shard assignment; execute oracle and core in fresh interpreters.

Collection may import mpmath but executes no tests. Core import attempts fail
closed with RuntimeError, never ImportError/pytest.importorskip. No ABC resets.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
MAX_NODES=50000
MAX_JSON_BYTES=32*1024*1024
MAX_NODE_CHARS=16384


def validate_nodes(nodes,root=None):
    if type(nodes) is not list or not 0<len(nodes)<=MAX_NODES:
        raise ValueError('node inventory count/type')
    seen=set()
    for node in nodes:
        if type(node) is not str or len(node)>MAX_NODE_CHARS or node in seen or '::' not in node:
            raise ValueError('duplicate/malformed node ID')
        file=node.split('::',1)[0];path=Path(file)
        if not file.startswith('tests/') or '\\' in file or '..' in path.parts or not file.endswith('.py'):
            raise ValueError('node outside tests: '+file)
        if root is not None and (not (root/path).is_file() or not (root/path).resolve().is_relative_to(root.resolve())):
            raise ValueError('node file missing/outside root: '+file)
        seen.add(node)
    return nodes


def classify_nodes(root,nodes):
    root=Path(root);validate_nodes(nodes,root)
    files=sorted({p for folder in ('tests','tools','src','examples') for p in (root/folder).rglob('*.py')})
    if len(files)>10000:raise ValueError('module inventory bound')
    aliases={};names={};trees={}
    for path in files:
        rel=path.relative_to(root);parts=list(rel.with_suffix('').parts)
        if parts[0]=='src':parts=parts[1:]
        if parts[-1]=='__init__':parts=parts[:-1]
        name='.'.join(parts);names[path]=name
        variants={name}
        if rel.parts[0] in ('tests','tools'):
            variants.add('.'.join(parts[1:]));variants.add(parts[-1])
        for variant in variants:aliases.setdefault(variant,set()).add(path)
        if path.stat().st_size>2*1024*1024:raise ValueError('module source bound: '+str(rel))
        trees[path]=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(rel))
    edges={p:set() for p in files};seeds=set();unknown=[]
    for path,tree in trees.items():
        package=names[path].split('.') if path.name=='__init__.py' else names[path].split('.')[:-1]
        dynamic_names={'import_module','__import__','importorskip'}
        for declaration in ast.walk(tree):
            if isinstance(declaration,ast.ImportFrom) and declaration.module in ('importlib','pytest'):
                dynamic_names.update(alias.asname or alias.name for alias in declaration.names
                                     if alias.name in ('import_module','importorskip'))
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:base='.'.join(package[:len(package)-node.level+1]+([base] if base else []))
                imports.append(base);imports.extend(base+'.'+a.name for a in node.names)
            elif isinstance(node,ast.Call):
                method=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                if method in dynamic_names:
                    if node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
                        imports.append(node.args[0].value)
                    else:unknown.append([path.relative_to(root).as_posix(),node.lineno,method])
        for name in imports:
            if name=='mpmath' or name.startswith('mpmath.') or name.split('.')[-1]=='ch3_reference':seeds.add(path)
            edges[path].update(aliases.get(name,()))
        if path.name=='ch3_reference.py':seeds.add(path)
        if path.is_relative_to(root/'tests'):
            for parent in path.parents:
                if not parent.is_relative_to(root/'tests'):break
                fixture=parent/'conftest.py'
                if fixture in edges and fixture!=path:edges[path].add(fixture)
    oracle=set(seeds)
    while True:
        added={p for p,dependencies in edges.items() if dependencies&oracle}-oracle
        if not added:break
        oracle.update(added)
    groups={'core':[],'oracle':[]}
    for node in nodes:groups['oracle' if root/node.split('::',1)[0] in oracle else 'core'].append(node)
    # Qualified planning assertions must retain their clean original cohort.
    from ci_planning_profile import REQUIRED_PROFILE_CASES
    for node in groups['oracle']:
        file,*rest=node.split('::');key=Path(file).stem+'::'+'::'.join(rest)
        if key in REQUIRED_PROFILE_CASES:raise ValueError('required planning positive classified as oracle: '+node)
    return groups,dict(oracle_modules=sorted(p.relative_to(root).as_posix() for p in oracle),
        unresolved_dynamic_imports=unknown,module_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def required_positive_key(node):
    file,*rest=node.split('::')
    module=Path(file).stem
    return module+'::'+'::'.join(rest) if file=='tests/'+module+'.py' else None


def split_required_positives(groups,*,required):
    from ci_planning_profile import REQUIRED_PROFILE_CASES
    if type(required) is not bool or set(groups)!={'core','oracle'}:
        raise ValueError('invalid positive phase partition')
    flat=[node for names in groups.values() for node in names]
    validate_nodes(flat)
    if any(required_positive_key(node) in REQUIRED_PROFILE_CASES for node in groups['oracle']):
        raise ValueError('required planning positive classified as oracle')
    result={name:list(nodes) for name,nodes in groups.items()}
    if required:
        positive=[node for node in result['core'] if required_positive_key(node) in REQUIRED_PROFILE_CASES]
        chosen=set(positive)
        result['core']=[node for node in result['core'] if node not in chosen]
        if positive:result['planning_positive']=positive
    union=[node for names in result.values() for node in names]
    if len(union)!=len(set(union)) or set(union)!=set(flat):
        raise ValueError('positive partition changed assigned node union')
    return result


def execution_phases(groups):
    phases=[dict(name=name,mode=name,nodes=groups[name],positive=None)
            for name in ('core','oracle') if groups[name]]
    phases.extend(dict(name='positive-'+str(index).zfill(3),mode='core',nodes=[node],positive=node)
                  for index,node in enumerate(groups.get('planning_positive',[])))
    return phases


def verify_positive_junit(path,node):
    from ci_planning_profile import REQUIRED_PROFILE_CASES,CONTRACT
    from ci_durations import node_id
    key=required_positive_key(node)
    cases=list(ET.parse(path).iter('testcase'))
    if key not in REQUIRED_PROFILE_CASES or len(cases)!=1:
        raise ValueError('positive phase must contain exactly its authoritative case')
    case=cases[0]
    actual=case.get('classname','').rsplit('.',1)[-1]+'::'+case.get('name','')
    properties=[(p.get('name'),p.get('value')) for p in case.findall('./properties/property')]
    if (actual!=key or node_id(case.get('classname',''),case.get('name',''))!=node
            or any(case.find(kind) is not None for kind in ('failure','error','skipped'))
            or ('numpy_profile_contract',CONTRACT) not in properties
            or ('execution_scope','original-profile-assertions') not in properties
            or any(name=='postcapture_qualification' and value.lower()=='false' for name,value in properties)
            or any(name=='execution_scope' and value.endswith('refusal') for name,value in properties)):
        raise ValueError('required positive phase did not execute original passed assertions')


class PhaseBudget:
    """Secondary DRIVER budget; earlier GitHub30min whole-job cap stays stronger."""
    def __init__(self,seconds=1800):
        if type(seconds) is not int or not 0<seconds<=1800:raise ValueError('invalid CI driver budget')
        self.seconds=seconds;self.started=time.monotonic();self.deadline=self.started+seconds
    def remaining(self):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('shared CI driver budget exhausted')
        return remaining
    def report(self):
        now=time.monotonic()
        return dict(scope='driver-only; GitHub whole-job deadline starts earlier',limit_seconds=self.seconds,
            elapsed_seconds=now-self.started,remaining_driver_seconds=max(0.,self.deadline-now))


def merge_junit(paths,expected,output):
    from ci_durations import node_id
    validate_nodes(expected);combined=ET.Element('testsuites');seen=[]
    for path in paths:
        root=ET.parse(path).getroot()
        suites=[root] if root.tag=='testsuite' else list(root)
        for suite in suites:
            if suite.tag!='testsuite':raise ValueError('unexpected JUnit root child')
            combined.append(copy.deepcopy(suite))
            seen.extend(node_id(c.get('classname',''),c.get('name','')) for c in suite.iter('testcase'))
    ET.ElementTree(combined).write(output,encoding='utf-8',xml_declaration=True)
    if len(seen)!=len(set(seen)) or set(seen)!=set(expected):
        raise ValueError('JUnit union differs: missing='+str(sorted(set(expected)-set(seen))[:5])+
                         ' unknown='+str(sorted(set(seen)-set(expected))[:5])+' duplicate='+str(len(seen)-len(set(seen))))


def install_core_guard():
    import importlib.abc
    def forbidden(name):return type(name) is str and (name=='mpmath' or name.startswith('mpmath.'))
    def check():
        if any(forbidden(name) for name in sys.modules):raise RuntimeError('mpmath loaded in core interpreter')
    check()
    def audit(event,args):
        if event=='import' and args and forbidden(args[0]):raise RuntimeError('mpmath import attempted in core interpreter')
    class RejectOracle(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if forbidden(fullname):raise RuntimeError('mpmath import attempted in core interpreter')
    sys.addaudithook(audit);sys.meta_path.insert(0,RejectOracle())
    return check


def read_json(path):
    path=Path(path)
    if path.stat().st_size>MAX_JSON_BYTES:raise ValueError('JSON input bound')
    return json.loads(path.read_text(encoding='utf-8'))


def select_manifest_ids(collected,assigned,inventory):
    """Selection only, never shard assignment; reject source/param drift first."""
    validate_nodes(assigned);validate_nodes(inventory)
    if len(collected)!=len(set(collected)) or set(collected)!=set(inventory):
        raise RuntimeError('module inventory changed: missing='+str(sorted(set(inventory)-set(collected))[:3])+
                           ' unknown='+str(sorted(set(collected)-set(inventory))[:3]))
    if not set(assigned)<=set(inventory):raise RuntimeError('assigned IDs missing from module inventory')
    return list(assigned)


class PhaseProgress:
    """Flushed observations only; never a replacement for actual JUnit results."""
    def __init__(self,path,mode,nodes,binding,*,grouping='node',group_modules=None):
        from ci_shards import grouping_policy,group_modules_policy
        self.grouping=grouping_policy(grouping)
        self.group_modules=group_modules_policy(grouping,group_modules)
        validate_nodes(nodes)
        if type(binding) is not str or len(binding)!=64 or any(c not in '0123456789abcdef' for c in binding):
            raise ValueError('progress configuration binding')
        self.mode=mode;self.binding=binding;self.indices={node:i for i,node in enumerate(nodes)}
        self.completed=0;self.finished=set();self.active=None;self.reports={}
        # Tests may monkeypatch the time/os/path modules; retain real observer
        # services before fixtures, without changing any calculator or source.
        self.clock=time.monotonic;self.pid=os.getpid()
        self.encode=json.JSONEncoder(sort_keys=True,allow_nan=False).iterencode
        self.stream=Path(path).open('w',encoding='utf-8')
        try:self._write('phase_start',None)
        except BaseException as error:
            try:self.stream.close()
            except BaseException as cleanup:error.add_note('progress constructor cleanup failed: '+str(cleanup))
            raise

    def _write(self,event,node,**details):
        row=dict(schema='ci-phase-progress-v1',cohort=self.mode,event=event,nodeid=node,grouping=self.grouping,group_modules=self.group_modules,
            config_sha256=self.binding,pid=self.pid,monotonic_seconds=self.clock(),
            assigned_total=len(self.indices),assigned_index=self.indices.get(node),
            completed_count=self.completed,**details)
        # Encoder-anchor tests deliberately poison c_make_encoder. Evidence
        # uses the prebound stdlib Python path, not the service under test.
        self.stream.write(''.join(self.encode(row,_one_shot=False))+'\n')
        self.stream.flush()

    def start(self,node):
        if node not in self.indices or self.active is not None or node in self.finished:
            raise RuntimeError('progress start outside exact assignment/protocol')
        self.active=node;self.reports={};self._write('start',node)

    def report(self,node,phase,outcome):
        if node!=self.active or phase not in ('setup','call','teardown') or outcome not in ('passed','failed','skipped'):
            raise RuntimeError('progress report outside active protocol')
        prior=tuple(self.reports)
        required=() if phase=='setup' else ('setup',) if phase=='call' or self.reports.get('setup')!='passed' else ('setup','call')
        if phase in self.reports or prior!=required or (phase=='call' and self.reports.get('setup')!='passed'):
            raise RuntimeError('progress report outside legal phase ordering')
        self.reports[phase]=outcome;self._write('report',node,phase=phase,outcome=outcome)

    def finish(self,node,*,interrupted=None):
        if node!=self.active:raise RuntimeError('progress end outside active protocol')
        complete=interrupted is None
        required=('setup','call','teardown') if self.reports.get('setup')=='passed' else ('setup','teardown')
        if complete and tuple(self.reports)!=required:raise RuntimeError('protocol ended without complete actual phase reports')
        if not complete:outcome='interrupted'
        elif any(self.reports.get(p)=='failed' for p in ('setup','teardown')):outcome='error'
        elif self.reports.get('call')=='failed':outcome='failed'
        elif 'skipped' in self.reports.values():outcome='skipped'
        else:outcome='passed'
        if complete:self.completed+=1
        self._write('end',node,outcome=outcome,protocol_complete=complete,error_type=interrupted)
        self.finished.add(node);self.active=None;self.reports={}

    def close(self):self.stream.close()


def child(config):
    row=read_json(config);root=Path(row['root']);os.chdir(root)
    from ci_shards import grouping_policy,group_modules_policy,fixture_group_modules
    grouping=grouping_policy(row.get('grouping','node'))
    if grouping=='module' and os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS')=='1':
        raise ValueError('qualified positive hosts require node grouping')
    group_modules=group_modules_policy(grouping,row.get('group_modules'))
    # Match python -m pytest: the checkout root remains importable (examples,
    # tools and namespace packages) as well as pyproject's candidate src path.
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))
    check=install_core_guard() if row['mode']=='core' else lambda:None
    import pytest
    expected=row.get('nodes');manifest=Path(row['manifest']);raw_inventory=[]
    if expected is not None:validate_nodes(expected,root)
    positive=row.get('positive')
    if positive is not None:
        from ci_planning_profile import REQUIRED_PROFILE_CASES,preflight
        if (row['mode']!='core' or expected!=[positive]
                or required_positive_key(positive) not in REQUIRED_PROFILE_CASES
                or os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS')!='1'):
            raise ValueError('invalid required positive child configuration')
        profile=preflight()
        profile.update(host_scope='fresh-H-single-authoritative-case',nodeid=positive,pid=os.getpid())
        (manifest.parent/'positive-profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    module_names=(list(dict.fromkeys(node.split('::',1)[0] for node in expected))
                  if expected is not None else row.get('allowed_modules'))
    allowed=None if module_names is None else {(root/name).resolve() for name in module_names}
    progress=(PhaseProgress(manifest.with_name(row['mode']+'-progress.jsonl'),row['mode'],expected,hashfile(config),grouping=grouping,group_modules=group_modules)
              if expected is not None and row['mode'] in ('core','oracle') else None)
    class ExactCollection:
        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_protocol(self,item):
            if progress is not None:progress.start(item.nodeid)
            try:result=yield
            except BaseException as error:
                if progress is not None:progress.finish(item.nodeid,interrupted=type(error).__name__)
                raise
            else:
                if progress is not None:progress.finish(item.nodeid)
                return result
        def pytest_runtest_logreport(self,report):
            if progress is not None:progress.report(report.nodeid,report.when,report.outcome)
        @pytest.hookimpl(tryfirst=True)
        def pytest_ignore_collect(self,collection_path):
            if allowed is None:return None
            path=collection_path.resolve()
            if path.is_dir() and not any(module.is_relative_to(path) for module in allowed):return True
            if path.suffix=='.py' and path.name!='conftest.py' and path not in allowed:return True
            return None
        @pytest.hookimpl(tryfirst=True)
        def pytest_runtestloop(self,session):
            actual=[item.nodeid for item in session.items]
            check()
            actual_modules=fixture_group_modules(session.items) if grouping=='module' else []
            if expected is not None and grouping=='module':
                admitted=sorted(set(group_modules or []) & {node.split('::',1)[0] for node in actual})
                if group_modules is None or actual_modules!=admitted:
                    raise RuntimeError('actual scoped fixture modules differ from admitted phase policy')
            recorded_modules=(getattr(session.config,'_anygeometry_shard_group_modules',actual_modules)
                              if row['mode']=='collect' else group_modules)
            manifest.write_text(json.dumps(dict(nodes=raw_inventory if row['mode']=='inventory' else actual,
                mode=row['mode'],mpmath_loaded='mpmath' in sys.modules,grouping=grouping,group_modules=recorded_modules,
                present_group_modules=actual_modules,
                collection_mode='original' if allowed is None else 'directory-walk',pid=os.getpid(),
                positive_host_scope='fresh-H-single-authoritative-case' if positive is not None else None),indent=2),encoding='utf-8')
            if expected is not None and (len(actual)!=len(expected) or set(actual)!=set(expected)):
                raise RuntimeError('cohort actual collected set/count differs from assigned JSON IDs')
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self,config,items):
            raw_inventory[:]=[item.nodeid for item in items]
            if expected is not None:
                selected=select_manifest_ids(raw_inventory,expected,row['inventory'])
                wanted=set(selected);lookup={item.nodeid:item for item in items}
                dropped=[item for item in items if item.nodeid not in wanted]
                if dropped:config.hook.pytest_deselected(items=dropped)
                items[:]=[lookup[node] for node in selected]
            if row['mode']!='collect':
                for item in items:
                    item.user_properties.append(('ci_interpreter_cohort',row['mode']))
                    if positive is not None:item.user_properties.append(('ci_positive_host_scope','fresh-H-single-authoritative-case'))
        def pytest_runtest_teardown(self):check()
        def pytest_sessionfinish(self):check()
    args=list(row['args'])
    if expected is not None:args=['tests']+args
    try:return int(pytest.main(args,plugins=[ExactCollection()]))
    finally:
        if progress is not None:progress.close()


def run_phase(root,folder,name,config,env,*,budget=None):
    remaining=None if budget is None else budget.remaining()
    path=folder/(name+'-config.json');path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    stamp=time.monotonic()
    with (folder/(name+'-stdout.txt')).open('w',encoding='utf-8') as stdout,(folder/(name+'-stderr.txt')).open('w',encoding='utf-8') as stderr:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',str(path)],cwd=root,env=env,stdout=stdout,stderr=stderr)
    record=dict(returncode=result.returncode,process_wall_seconds=time.monotonic()-stamp,
        remaining_driver_seconds_at_start=remaining,driver_budget=None if budget is None else budget.report())
    (folder/(name+'-result.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    if budget is not None:budget.remaining()
    return result.returncode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',type=Path)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--evidence-dir',type=Path,default=Path('ci-isolation'))
    parser.add_argument('--junitxml',type=Path,default=Path('kernel-results.xml'))
    parser.add_argument('--select',action='append',default=[])
    parser.add_argument('-k',default='');parser.add_argument('-m',default='')
    parser.add_argument('--maxfail',type=int,default=1);parser.add_argument('--durations',type=int,default=20)
    parser.add_argument('--collect-only',action='store_true')
    from ci_test_profiles import PROFILES,module_manifest,node_receipt,validate_selection
    parser.add_argument('--profile',choices=PROFILES,default=None,
                        help='full preserves the release inventory; other profiles are partial development selections')
    args=parser.parse_args(argv)
    if args.child:return child(args.child)
    from ci_shards import grouping_policy,group_modules_policy,load_table,scheduling_receipt
    grouping=grouping_policy(os.environ.get('ANYGEOMETRY_SHARD_GROUPING','node'))
    if grouping=='module' and os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS')=='1':
        raise ValueError('qualified positive hosts require node grouping')
    focused=args.profile is None and bool(args.select or args.k or args.m or os.environ.get('PYTEST_ADDOPTS'))
    args.profile=args.profile or 'full'
    validate_selection('unit' if focused else args.profile,args.select,args.k,args.m,os.environ)
    budget=PhaseBudget()
    root=args.root.resolve();folder=(root/args.evidence_dir).resolve()
    if folder.exists() and any(folder.iterdir()):raise ValueError('evidence directory must be fresh')
    folder.mkdir(parents=True,exist_ok=True)
    profile=module_manifest(root,args.profile)
    if focused:profile.update(profile='legacy-focused',scope='partial-development-only')
    if args.profile!='full' and any(selector.split('::',1)[0] not in profile['selected_modules'] for selector in args.select):
        raise ValueError('explicit selector outside requested profile modules')
    (folder/'profile-modules.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    report=Path(os.environ.get('ANYGEOMETRY_SHARD_REPORT',str(folder/'original-shard.json')))
    if not report.is_absolute():report=root/report
    env=dict(os.environ);env['ANYGEOMETRY_SHARD_REPORT']=str(report)
    common=['--rootdir='+str(root),'--maxfail='+str(args.maxfail),'--durations='+str(args.durations)]
    if args.k:common+=['-k',args.k]
    if args.m:common+=['-m',args.m]
    collection=dict(root=str(root),mode='collect',grouping=grouping,manifest=str(folder/'collection.json'),
                    args=(args.select or ['tests'])+common+['--collect-only','-q'])
    if args.profile!='full':collection['allowed_modules']=profile['selected_modules']
    code=run_phase(root,folder,'collection',collection,env,budget=budget)
    if code:raise RuntimeError('original collection failed; retained phase evidence')
    shard=read_json(report)
    if grouping_policy(shard.get('grouping','node'))!=grouping:raise ValueError('original shard grouping policy differs')
    collected=read_json(folder/'collection.json')
    if grouping_policy(collected.get('grouping','node'))!=grouping:raise ValueError('collection grouping policy differs')
    reported_modules=shard.get('group_modules',[] if grouping=='node' else None)
    if grouping=='module' and type(reported_modules) is not list:raise ValueError('shard lacks admitted scoped fixture modules')
    if grouping=='module' and type(collected.get('group_modules')) is not list:raise ValueError('collection lacks admitted scoped fixture modules')
    group_modules=group_modules_policy(grouping,reported_modules)
    if group_modules_policy(grouping,collected.get('group_modules'))!=group_modules:raise ValueError('collection scoped fixture modules differ')
    assigned=validate_nodes(shard['ran'],root)
    selected=validate_nodes(shard['selected'],root)
    schedule=scheduling_receipt(selected,shard['count'],load_table(root/'reports/ci/durations.json'),grouping=grouping,group_modules=group_modules)
    (folder/'schedule-receipt.json').write_text(json.dumps(schedule,indent=2),encoding='utf-8')
    receipt=node_receipt(selected,args.profile)
    receipt['grouping']=grouping
    receipt['group_modules']=group_modules
    if focused:receipt.update(profile='legacy-focused',scope='partial-development-only')
    if receipt['selected']!=selected:raise ValueError('collected nodes escaped requested test profile')
    receipt['module_manifest_sha256']=hashfile(folder/'profile-modules.json')
    (folder/'profile-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    if not set(assigned)<=set(selected) or not set(shard['slow'])<=set(selected):raise ValueError('invalid original shard report')
    if read_json(folder/'collection.json')['nodes']!=assigned:raise ValueError('original report differs from actual shard collection')
    env.pop('ANYGEOMETRY_TEST_SHARD',None);env.pop('ANYGEOMETRY_SHARD_REPORT',None)
    env.pop('ANYGEOMETRY_SHARD_GROUPING',None)
    # Default CI already collected the complete global inventory. A partial
    # local selector needs its modules' full inventory, via the same directory
    # traversal/allowlist mechanism, never hundreds of positional selectors.
    if args.select or args.k or args.m or os.environ.get('PYTEST_ADDOPTS'):
        modules=list(dict.fromkeys(node.split('::',1)[0] for node in assigned))
        inventory_config=dict(root=str(root),mode='inventory',grouping=grouping,group_modules=group_modules,manifest=str(folder/'inventory.json'),
            allowed_modules=modules,args=['tests','--rootdir='+str(root),'--collect-only','-q'])
        if run_phase(root,folder,'inventory',inventory_config,env,budget=budget):raise RuntimeError('module inventory collection failed')
        inventory=validate_nodes(read_json(folder/'inventory.json')['nodes'],root)
    else:inventory=selected
    if not set(assigned)<=set(inventory):raise ValueError('assigned IDs changed before module inventory')
    groups,closure=classify_nodes(root,assigned)
    groups=split_required_positives(groups,required=env.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS')=='1')
    plan=dict(shard_report=str(report),shard_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
              index=shard['index'],count=shard['count'],assigned=assigned,groups=groups,grouping=grouping,group_modules=group_modules,
              module_inventory=inventory,closure=closure,profile_receipt_sha256=hashfile(folder/'profile-receipt.json'),
              schedule_receipt_sha256=hashfile(folder/'schedule-receipt.json'))
    (folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[];codes=[]
    for phase in execution_phases(groups):
        group,nodes=phase['mode'],phase['nodes']
        location=folder/phase['name'] if phase['positive'] is not None else folder
        location.mkdir(exist_ok=True)
        junit=location/(group+'.xml')
        options=common+(['--collect-only','-q'] if args.collect_only else ['--junitxml='+str(junit)])
        files={node.split('::',1)[0] for node in nodes}
        config=dict(root=str(root),mode=group,nodes=nodes,positive=phase['positive'],grouping=grouping,group_modules=group_modules,
            inventory=[node for node in inventory if node.split('::',1)[0] in files],
            manifest=str(location/(group+'-collection.json')),args=options)
        code=run_phase(root,location,group,config,env,budget=budget)
        phase_manifest=Path(config['manifest'])
        if phase_manifest.exists():
            actual_phase=read_json(phase_manifest)
            if (grouping_policy(actual_phase.get('grouping','node'))!=grouping
                    or group_modules_policy(grouping,actual_phase.get('group_modules'))!=group_modules):
                raise ValueError('returned phase grouping/admitted-module policy differs')
        if not args.collect_only:
            if junit.exists():results.append(junit)
            if phase['positive'] is not None:
                try:verify_positive_junit(junit,phase['positive'])
                except (ValueError,OSError,ET.ParseError) as error:
                    (location/'positive-validation-error.txt').write_text(str(error),encoding='utf-8')
                    code=code or 1
        codes.append(code)
    if hashfile(report)!=plan['shard_report_sha256']:raise ValueError('original shard report changed')
    if args.collect_only:
        if any(codes):raise RuntimeError('cohort collection guard failed')
    else:
        output=args.junitxml if args.junitxml.is_absolute() else root/args.junitxml
        merge_junit(results,assigned,output)
        if any(codes):raise RuntimeError('cohort failed; retained real combined JUnit')
    (folder/'complete.json').write_text(json.dumps(dict(status='collection-verified' if args.collect_only else 'complete',
        assigned=len(assigned),core=len(groups['core']),oracle=len(groups['oracle']),grouping=grouping,group_modules=group_modules,
        planning_positive=len(groups.get('planning_positive',[])),driver_budget=budget.report()),indent=2),encoding='utf-8')
    print(json.dumps(dict(assigned=len(assigned),core=len(groups['core']),oracle=len(groups['oracle']),planning_positive=len(groups.get('planning_positive',[])),collect_only=args.collect_only)))
    return 0


def hashfile(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':raise SystemExit(main())
