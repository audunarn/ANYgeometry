"""Diagnostic replay of retained results; NOT a new parallel acceptance run."""
import json,os,pathlib,sys
W=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(W/'src'),str(W/'tests')]
from anygeometry import _planning_executor as a,_planning_host as host
from test_planning_host import test_public_real_pair_and_face_stages_and_apply_ci
from pytest import MonkeyPatch
base=W/'.pp1-evidence/pp5c-host-green1/pytest-temp/test_public_real_pair_and_face0/parallel'
class Replay:
    def __init__(self,directory,**kwargs):self.kind=pathlib.Path(directory).name
    def submit_stage(self,*args):
        report=a.read_packet(base/self.kind/'result.json','result');report['results']=a.reduce_results(report.pop('rows'),sum(r['task_count'] for r in report['worker_reports']));return report
    def close(self):pass
out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY']);monkey=MonkeyPatch();monkey.setattr(host,'OwnedPlanner',Replay)
try:test_public_real_pair_and_face_stages_and_apply_ci(monkey,out)
except AssertionError as exc:print(type(exc).__name__,str(exc))
finally:monkey.undo()
p=out/'parallel/sample-mismatch-face.json';data=json.loads(p.read_text());diffs=[]
def compare(a,b,path='root'):
    if type(a) is list and type(b) is list:
        if len(a)!=len(b):diffs.append([path,'length',len(a),len(b)])
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'['+str(i)+']')
    elif type(a) is not type(b) or a!=b:diffs.append([path,a,b])
compare(data['reference'],data['candidate']);(out/'differences.json').write_text(json.dumps(diffs,indent=2));print(json.dumps(dict(differences=len(diffs),first=diffs[:15])))
