"""Installed candidate geometry/mesher/FEM replay checks outside source checkouts."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import platform
from check_compatibility import Runner,identity,wheel_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','mesher','fem','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--cases',nargs='+',choices=('skew','parallel','perpendicular','tangent','replay-050','replay-025','oblique','concave','growing','multiple-cuts','cylinder-holes'),
                        default=['replay-050','replay-025','parallel','perpendicular','skew','tangent','oblique','concave','growing','multiple-cuts','cylinder-holes'])
    parser.add_argument('--wheelhouse',type=Path,help='use public dependency wheels without network access')
    parser.add_argument('--expected-machine',default='')
    parser.add_argument('--frontal-diagnostic',action='store_true',help='explicit frontal triangle controls; not default-policy acceptance')
    parser.add_argument('--baseline',action='store_true',help='record released replay outcome, including capabilities absent in older owners')
    args=parser.parse_args()
    if args.expected_machine and platform.machine().lower()!=args.expected_machine.lower():
        parser.error('runner architecture does not match requested coverage')
    if args.baseline and (args.frontal_diagnostic or any(not case.startswith('replay-') for case in args.cases)):
        parser.error('baseline comparison supports the original replay with released automatic defaults')
    root=args.output.resolve()
    if root.exists() or any((parent/'.git').exists() for parent in (root,*root.parents)):
        parser.error('output must be new and outside all source checkouts')
    artifacts={name:getattr(args,name).resolve(strict=True) for name in ('geometry','mesher','fem')}
    for name,expected in (('geometry','anygeometry'),('mesher','anymesher'),('fem','anyfem')):
        if wheel_metadata(artifacts[name])[0]!=expected:
            parser.error('incorrect '+name+' candidate artifact')
        version=wheel_metadata(artifacts[name])[1]
        versions={'geometry':'0.4.3' if args.baseline else '0.4.4',
                  'mesher':'0.5.0' if args.baseline else '0.5.1','fem':'0.4.1'}
        if version!=versions[name]:
            parser.error('unexpected '+name+' artifact version: '+version)
    if args.baseline and identity(artifacts['geometry'])['sha256'].lower()!='6ab8f398de8ad1cc19b5c0d70a9787e1a15d3f5253c1125134eeb5050ec4b8b5':
        parser.error('released geometry baseline does not match frozen artifact evidence')
    root.mkdir(parents=True)
    runner=Runner(root)
    report={'python':sys.version,'machine':platform.machine(),'artifacts':{key:identity(value) for key,value in artifacts.items()},
            'commands':runner.commands,'status':'failed','cases':args.cases,
            'frontal_diagnostic':args.frontal_diagnostic,'baseline':args.baseline}
    fixture_dir=Path(__file__).resolve().parent/'general_intersections'
    fixture_names=('replay.py','verify_replay.py','verify_baseline_replay.py','cylinder_cases.py','mesh_cylinder_cases.py','verify_cylinder_mesh.py','mesh_material_cases.py')
    report['fixtures']={}
    for name in fixture_names:
        shutil.copyfile(fixture_dir/name,root/name)
        report['fixtures'][name]=identity(root/name)
    wrapper='''import sys,runpy,json,hashlib,base64,importlib,importlib.metadata as metadata,zipfile
from pathlib import Path
root=Path(__file__).parent.resolve()
prefix=Path(sys.prefix).resolve()
assert prefix.is_relative_to(root)
assert not any((parent/'.git').exists() for parent in (root,*root.parents))
origins={}
for module in ('anygeometry','anymesher','anyfem','anysolver','anymaterial','anyfileio'):
    path=Path(importlib.import_module(module).__file__).resolve()
    assert path.is_relative_to(prefix),(module,str(path))
    origins[module]=str(path)
for distribution,wheel in json.loads((root/'candidate-artifacts.json').read_text()).items():
    installed=metadata.distribution(distribution)
    assert not installed.read_text('direct_url.json') or not json.loads(installed.read_text('direct_url.json')).get('dir_info',{}).get('editable')
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or '.data/' in name or '.dist-info/' in name:continue
            path=Path(installed.locate_file(name)).resolve()
            assert path.is_relative_to(prefix)
            assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(archive.read(name)).digest(),name
from anymesher.native_cpp import NATIVE_CPP_AVAILABLE,COMPILED_TRIANGULATION_AVAILABLE
assert NATIVE_CPP_AVAILABLE and COMPILED_TRIANGULATION_AVAILABLE,'candidate requires its compiled triangulation artifact'
(root/'installed-origins.json').write_text(json.dumps({'origins':origins,'native_cpp_available':NATIVE_CPP_AVAILABLE,
 'compiled_triangulation_available':COMPILED_TRIANGULATION_AVAILABLE,
 'resolved':{item.metadata['Name']:item.version for item in metadata.distributions()}},indent=2))
sys.path.insert(0,str(root))
sys.argv=sys.argv[1:]
runpy.run_path(str(root/sys.argv[0]),run_name='__main__')
'''
    (root/'run_case.py').write_text(wrapper)
    (root/'candidate-artifacts.json').write_text(json.dumps({wheel_metadata(path)[0]:str(path) for path in artifacts.values()}))
    try:
        environment,python=runner.environment('candidate-environment')
        runner.run([python,'-I','-m','pip','install','--report',root/'candidate-install.json',
                    *([] if args.wheelhouse is None else ['--no-index','--find-links',args.wheelhouse.resolve(strict=True)]),
                    *artifacts.values(),'ANYsolver==0.4.6','ANYmaterial==0.2.0','ANYfileio==0.3.2'])
        runner.run([python,'-I','-m','pip','check'])
        for case in args.cases:
            if case.startswith('replay-'):
                runner.run([python,'-I',root/'run_case.py',
                            'verify_baseline_replay.py' if args.baseline else 'verify_replay.py',
                            '.5' if case=='replay-050' else '.25'])
            elif case in ('oblique','concave','growing','multiple-cuts','cylinder-holes'):
                if args.frontal_diagnostic:parser.error('material cases exercise application automatic defaults')
                runner.run([python,'-I',root/'run_case.py','mesh_material_cases.py',case])
            else:
                runner.run([python,'-I',root/'run_case.py','mesh_cylinder_cases.py',case,
                            *(['frontal'] if args.frontal_diagnostic else [])])
                runner.run([python,'-I',root/'run_case.py','verify_cylinder_mesh.py',case])
        report['status']='baseline_recorded' if args.baseline else 'passed'
    finally:
        (root/'general-intersections-report.json').write_text(json.dumps(report,indent=2))
        print('REPORT_ROOT='+str(root),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
