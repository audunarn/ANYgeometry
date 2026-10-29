"""One-use installed CH3e pilot with a 600 s process-wall ceiling.

Registration capture is separate, non-numerical preparation. The run subcommand
creates one new evidence directory and never retries or resumes a failed pilot.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
if hasattr(sys, 'set_int_max_str_digits'): sys.set_int_max_str_digits(100000)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ch3e_schema import canonical,digest,write_json,capture,validate_registration,NAMES


def consume_authority(marker,candidate,root,registration_identity):
    with Path(marker).open('x',encoding='utf8') as authority:
        authority.write(json.dumps(dict(consumed=True,candidate_sha256=digest(candidate),
            evidence=str(root),registration_identity=registration_identity)))


def quantity_intervals(claim):
    items={'V':claim['volume'],**claim['moments']}
    sections=claim['sections']
    if set(sections) != {'plane'}: raise ValueError('pilot requires one generic plane section')
    section=sections['plane']
    if section['section'] == 'PRESENT': items.update(section['quantities'])
    else: items.update({k:None for k in NAMES[4:]})
    result={}
    for name in NAMES:
        item=items[name]
        if item is None: result[name]=None; continue
        if item.get('accuracy_met') is not True: raise ValueError(f'{name} accuracy not met')
        bounds=item['binary64']; lo=float.fromhex(bounds['lo']); hi=float.fromhex(bounds['hi'])
        if not all(map(math.isfinite,(lo,hi))) or hi < lo: raise ValueError('invalid binary64 enclosure')
        if Fraction(hi)-Fraction(lo) > Fraction(1,10**9): raise ValueError(f'{name} binary64 width exceeds 1e-9')
        result[name]=(lo,hi)
    return result


def compare_reference(claim,reference):
    intervals=quantity_intervals(claim); refs=reference['values']
    uncertainty=reference['estimated_absolute_uncertainty']
    if not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1e-9:
        raise ValueError('reference uncertainty unsuitable for 1e-9 corroboration')
    comparisons={}
    for key,bounds in intervals.items():
        expected=refs[key]
        if bounds is None or expected is None:
            ok=bounds is None and expected is None; discrepancy=None
        else:
            lo,hi=bounds; discrepancy=abs((lo+hi)/2-expected)
            ok=lo <= expected+uncertainty and hi >= expected-uncertainty
        comparisons[key]=dict(agrees=ok,midpoint_difference=discrepancy)
    return dict(agrees=all(x['agrees'] for x in comparisons.values()),
                reference_uncertainty=uncertainty,reference_uncertainty_is_rigorous=False,
                quantities=comparisons)


def relation(a,b,scale=1):
    if a is None or b is None: return False
    lo,hi=(scale*b[0],scale*b[1]) if scale>0 else (scale*b[1],scale*b[0])
    return max(a[0],lo) <= min(a[1],hi)


def retained_relations(root):
    def intervals(name): return quantity_intervals(json.loads((root/f'{name}-response.json').read_text()))
    out={}
    try:
        a=intervals('full-b1-upright-draft-1.75')['V']; b=intervals('half-b1-upright-draft-1.75')['V']
        out['full_volume_equals_twice_matching_half']=relation(a,b,2)
    except (OSError,ValueError,KeyError): out['full_volume_equals_twice_matching_half']=None
    try:
        plus=intervals('full-b1-heel-plus-0.1'); minus=intervals('full-b1-heel-minus-0.1')
        out['heel_reflection']={k:relation(plus[k],minus[k],-1 if k in ('Jy','Mv','Iuv') else 1) for k in NAMES}
    except (OSError,ValueError,KeyError): out['heel_reflection']=None
    return out


class Pilot:
    def __init__(self,args):
        self.started=time.perf_counter(); self.root=Path(args.out).resolve()
        self.root.mkdir(parents=True,exist_ok=False)
        self.ledger=[]; self.proc=None; self.args=args; self.commands=[]
        self.stdout=None; self.stderr=None
    def elapsed(self): return time.perf_counter()-self.started
    def save(self):
        write_json(self.root/'ledger.json',dict(total_limit_s=600.,work_boundary_s=480.,
            reserve_s=120.,elapsed_process_wall_s=self.elapsed(),stages=self.ledger,
            launch_command=self.commands,accounting='continuous parent perf_counter including startup, I/O and finalization'))
    def start(self):
        worker=Path(__file__).with_name('ch3e_worker.py').resolve()
        env=os.environ.copy()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): env[key]='1'
        env.pop('PYTHONPATH',None)
        self.commands=[str(Path(self.args.python).resolve()),'-I',str(worker),str(self.root),str(Path(self.args.site).resolve()),str(Path(self.args.wheel).resolve())]
        self.stdout=(self.root/'worker-stdout.jsonl').open('w',encoding='utf8')
        self.stderr=(self.root/'worker-stderr.log').open('w',encoding='utf8')
        self.proc=subprocess.Popen(self.commands,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.stderr,text=True,encoding='utf8',env=env,cwd=self.root)
        self.responses=queue.Queue()
        def read():
            for line in self.proc.stdout:
                self.stdout.write(line); self.stdout.flush(); self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=read,daemon=True).start()
    def command(self,cmd,*,final=False):
        start=self.elapsed(); boundary=595. if final else 480.
        cap=120. if cmd['op']=='prepare' else 30. if cmd['op']=='query' else boundary-start
        allowance=min(cap,boundary-start)
        if allowance <= 0: raise TimeoutError('no remaining authorised process-wall allowance')
        record=dict(command=cmd,start_process_wall_s=start,timeout_s=allowance,
                    used_before_s=start,remaining_before_s=600-start)
        self.ledger.append(record); self.save()
        write_json(self.root/f"command-{len(self.ledger):03}.json",cmd)
        try:
            self.proc.stdin.write(json.dumps(cmd)+'\n');self.proc.stdin.flush()
            remaining=allowance-(self.elapsed()-start)
            if remaining <= 0: raise queue.Empty
            line=self.responses.get(timeout=remaining)
            if line is None: raise RuntimeError('worker exited without a response')
            response=json.loads(line)
            record['response']=response
            record['status']=response['status']
            if response['status'] != 'ok': raise RuntimeError(f"{cmd['op']}: {response}")
            return response['result']
        except queue.Empty:
            record['status']='timeout'; self.stop(); raise TimeoutError('owned worker process-tree timeout')
        except Exception:
            record.setdefault('status','error');raise
        finally:
            record['end_process_wall_s']=self.elapsed()
            record['process_wall_s']=record['end_process_wall_s']-start
            record['remaining_after_s']=max(0.,600-self.elapsed());self.save()
    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            if os.name=='nt':
                try:
                    killed=subprocess.run(['taskkill','/PID',str(self.proc.pid),'/T','/F'],capture_output=True,timeout=2)
                    write_json(self.root/'termination.json',dict(command=killed.args,returncode=killed.returncode,
                        stdout=killed.stdout.decode(errors='replace'),stderr=killed.stderr.decode(errors='replace')))
                except (subprocess.TimeoutExpired,OSError) as error:
                    write_json(self.root/'termination.json',dict(error=type(error).__name__,message=str(error)))
                    self.proc.kill()
            else: self.proc.kill()
            try: self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
        if self.stderr: self.stderr.flush()
    def run(self):
        registration_path=Path(self.args.registration)
        if hashlib.sha256(registration_path.read_bytes()).hexdigest() != self.args.registration_sha256:
            raise ValueError('registration bytes differ from parent frozen hash')
        registration=validate_registration(json.loads(registration_path.read_text(encoding='utf-8-sig')))
        wheel=Path(self.args.wheel).resolve()
        write_json(self.root/'frozen-registration.json',registration)
        write_json(self.root/'candidate.json',dict(wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            registration_identity=registration['identity'],registration_sha256=self.args.registration_sha256,python=str(Path(self.args.python).resolve()),
            installed_site=str(Path(self.args.site).resolve()),
            harness_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('ch3e_*.py')}))
        completed=[];verified=[];comparisons={};failures=[]
        try:
            consume_authority(Path(self.args.authority_marker).resolve(),
                json.loads((self.root/'candidate.json').read_text()),self.root,registration['identity'])
            self.start();self.command(dict(op='prepare',kind='full'))
            for p in registration['planes']:
                name='full-'+p['registration']['registration_id']
                query_result=self.command(dict(op='query',kind='full',id=name,query=p['query']))
                expected_occupancy={'dry_limit':'EMPTY','full_limit':'FULL'}.get(p['registration']['purpose'],'PARTIAL')
                if query_result['occupancy'] != expected_occupancy:
                    raise ValueError(f'occupancy mismatch for {name}')
                completed.append(name)
                reference=self.command(dict(op='reference',kind='full',id=name,query=p['query']))
                claim=json.loads((self.root/f'{name}-response.json').read_text())
                comparison=compare_reference(claim,reference);comparisons[name]=comparison
                write_json(self.root/f'{name}-comparison.json',comparison)
                if not comparison['agrees']: raise ValueError(f'independent reference mismatch: {name}')
            matched=next(p for p in registration['planes'] if p['registration']['registration_id']=='b1-upright-draft-1.75')
            self.command(dict(op='prepare',kind='half'))
            half='half-b1-upright-draft-1.75'
            self.command(dict(op='query',kind='half',id=half,query=matched['query']));completed.append(half)
            # The independent full-volume reference and matched half/full relation
            # avoid duplicating an attitude campaign or numerical reference work.
            for name in completed:
                self.command(dict(op='verify',id=name),final=True);verified.append(name)
        except Exception as exc:
            failures.append(dict(type=type(exc).__name__,message=str(exc)))
        finally:
            self.stop()
            relations=retained_relations(self.root)
            all_relations=(relations['full_volume_equals_twice_matching_half'] is True and
                isinstance(relations['heel_reflection'],dict) and all(relations['heel_reflection'].values()))
            okay=(len(completed)==13 and len(verified)==13 and len(comparisons)==12 and
                  all(v['agrees'] for v in comparisons.values()) and all_relations and not failures)
            manifest={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                      for p in self.root.iterdir() if p.is_file() and p.name not in
                      ('ledger.json','REPORT.json','REPORT.md','artifact-manifest.json','worker-stdout.jsonl','worker-stderr.log')}
            write_json(self.root/'artifact-manifest.json',manifest)
            report=dict(schema='anygeometry-ch3e-pilot-report-v1',status='installed_verified' if okay else 'incomplete',
                completed_queries=completed,independently_verified=verified,reference_comparisons=comparisons,
                relations=relations,failures=failures,worker_exit_code=None if self.proc is None else self.proc.poll(),elapsed_process_wall_s=self.elapsed(),
                consumer_S2_admission=False,review_status='pending_independent_review',
                reference_uncertainty='estimated separately; not certificate validity')
            write_json(self.root/'REPORT.json',report)
            (self.root/'REPORT.md').write_text(f"# CH3e installed pilot\n\nStatus: {report['status']}.\n\n"
                f"Queries: {len(completed)}/13; public independent verifier: {len(verified)}/13; references: {len(comparisons)}/12.\n\n"
                f"Process wall before final evidence flush: {self.elapsed():.9f} s / 600 s.\n\n"
                f"Failures: {json.dumps(failures)}\n\nConsumer S2 admission remains separate. Independent review pending.\n",encoding='utf8')
            self.save()
        return 0 if okay else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    reg=sub.add_parser('register');reg.add_argument('--consumer-src',required=True);reg.add_argument('--out',required=True)
    run=sub.add_parser('run')
    for name in ('python','site','wheel','registration','out','authority-marker','registration-sha256'):run.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.command=='register':
        path=Path(args.out)
        if path.exists(): raise FileExistsError(path)
        write_json(path,capture(args.consumer_src)); return 0
    return Pilot(args).run()

if __name__=='__main__': raise SystemExit(main())
