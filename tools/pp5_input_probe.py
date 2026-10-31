"""PP5 input-size/admission preflight only; never invokes planning or meshing."""
import dataclasses, hashlib, json, os, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPO.parents[1]
sys.path.insert(0, str(ROOT/'resume'))
import env_setup
original = env_setup.repo_root
env_setup.repo_root = lambda name: REPO if name == 'ANYgeometry' else (ROOT/'int/ANYmesh' if name == 'ANYmesh' else original(name))
env_setup.setup()
from anyfem.io import load_project
from anygeometry.serialization import to_dict
from anygeometry._planning_executor import freeze_model
from anygeometry import _planning_qualification as q
from anygeometry.closure import extract_model_closure
source = pathlib.Path('C:/ANYworkspaceWork/AIP Gen2/ANYfem-local-assessment/hydrodynamics/AIP-Gen2-full-tripod-no-tower.anyfem')
p = load_project(source)
g = p.geometry
subsets = {}
for path in sorted((ROOT/'redesign/runs').glob('R11*-ladder-*/ladder_subsets.json')):
    subsets.update(json.loads(path.read_text(encoding='utf-8')))
def measure(model):
    document = to_dict(model)
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    pending, seen = [(vars(model), 'model')], set()
    extras = []
    while pending:
        value, location = pending.pop()
        if id(value) in seen: continue
        seen.add(id(value))
        if type(value) is dict:
            pending.extend((v, location + '[' + repr(k) + ']') for k,v in value.items())
            pending.extend((k, location + '.key') for k in value)
        elif type(value) in (tuple, list, set, frozenset): pending.extend((v, location+'[]') for v in value)
        elif not isinstance(value, type) and dataclasses.is_dataclass(value):
            declared = dataclasses.fields(value)
            extra = set(getattr(value, '__dict__', {})) - {f.name for f in declared}
            if extra: extras.append(dict(path=location, type=type(value).__module__+'.'+type(value).__qualname__, extra={k:type(vars(value)[k]).__name__ for k in sorted(extra)}))
            pending.extend((getattr(value, f.name), location+'.'+f.name) for f in declared)
    row = dict(faces=len(model.faces), document_bytes=len(encoded), document_sha256=hashlib.sha256(encoded).hexdigest(), record_walk_objects=len(seen), limits=dict(payload_bytes=4000000, record_walk_objects=100000, planner_seconds=100, diagnostic_faces=3))
    json_pending, json_seen = [document], set()
    while json_pending:
        item = json_pending.pop()
        if type(item) in (type(None),bool,int,float,complex,str,bytes): continue
        if id(item) in json_seen: continue
        json_seen.add(id(item))
        if type(item) is dict: json_pending.extend(item.keys()); json_pending.extend(item.values())
        elif type(item) in (tuple,list,set,frozenset): json_pending.extend(item)
    row['serialized_type_walk_objects'] = len(json_seen)
    q.check_types(document, {'types':q.input_type_names()})
    row['serialized_type_walk_passed'] = True
    wrapper = dict(document=document, extension_record=q.model_extension_record(model))
    row['prospective_model_payload_bytes'] = len(json.dumps(wrapper,sort_keys=True,separators=(',', ':'),allow_nan=False).encode())
    row['extra_record_count'] = len(extras)
    row['extra_record_examples'] = extras[:5]
    row['instance_caches'] = {key: len(vars(model)[key]) for key in ('_arc_cache','_edge_length_cache','_trim_loops_cache','_topology_corner_cache','_chain_arrays','_change_hooks')}
    try:
        packet = freeze_model(model)
        row.update(admitted=True, model_payload_bytes=len(packet['payload'].encode()))
    except Exception as exc:
        row.update(admitted=False, refusal_type=type(exc).__name__, refusal=str(exc))
    return row
record = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source=measure(g))
# The reference scripts plan subsets against this SAME full project geometry.
record['subset_routes'] = {name: dict(operands=len(subsets[name]), model='source') for name in ('elbow_sector60','elbow_r7')}
n2c_path = ROOT/'model-n2-20261030/AIP-Gen2-full-tripod-no-tower.MODEL-N2c-rotated-arms.SIMPLIFIED-EXERCISE.anyfem'
assert hashlib.sha256(n2c_path.read_bytes()).hexdigest() == '969cc5f9c8efe5b628686c9a89b1de02ea2dea8c1618f4295cb049c2621f468a'
p = load_project(n2c_path)
g = p.geometry
record['N2c_source'] = measure(g)
record['N2c_source_sha256'] = hashlib.sha256(n2c_path.read_bytes()).hexdigest()
members = {m for e in p.beam_edges for m in g.members_using_edge(e)}
handles = tuple([g.handle('face',i) for i in sorted(g.faces)] + [g.handle('edge',i) for i in p.beam_edges] + [g.handle('sheet',i) for i in sorted(g.sheets)] + [g.handle('member',i) for i in sorted(members)])
closure = extract_model_closure(g, handles, include_structural_closure=True, include_features=False)
assert len(closure.working_model.faces) == 561
record['N2c_closure'] = measure(closure.working_model)
assert not env_setup.foreign_modules()
path = pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY'])/'sizes.json'
path.write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))