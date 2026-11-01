"""PP5 host stage bridge; private controls, mandatory real plan/apply CI.

Without explicit controls the opt-in request declines to the original serial
caller. PP4 can supply admission here; PP3 owns future budgets/callback adapters.
"""
import json
import time
from pathlib import Path
import psutil
from ._planning_executor import OwnedPlanner,PlannerDeadline,PlannerSourceChanged,freeze_model
from ._planning_qualification import QualificationDecline,check_ci_equality
from ._planning_admission import stratified_samples
from ._planning_values import encode_value,decode_value
from .serialization import to_dict
from .runtime_diagnostics import _increment


class StageControl:
    def __init__(self,directory,*,expected_ci,observed_ci,workers=8,timeout=100,admission=None,payload_limit=4_000_000,walk_limit=100000):
        self.directory=Path(directory);self.expected_ci=expected_ci;self.observed_ci=observed_ci
        self.workers=workers;self.timeout=timeout;self.admission=admission
        self.limits=dict(payload_limit=payload_limit,walk_limit=walk_limit)
        self.reports=[];self.mode='not-run';self.decline_reason=None;self.ci_verified=False


class Bridge:
    def __init__(self,packet,control):
        self.packet=packet;self.control=control;self.deadline=time.monotonic()+control.timeout

    def check_deadline(self):
        if time.monotonic()>=self.deadline:raise PlannerDeadline('stage planning deadline consumed')

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        self.check_deadline();planner=None
        try:
            try:
                planner=OwnedPlanner(self.control.directory/kind,workers=self.control.workers,
                    timeout=min(110,self.deadline-time.monotonic()),**self.control.limits)
                candidate=planner.submit_stage(self.packet,tasks)
            except (OSError,psutil.AccessDenied) as exc:
                raise QualificationDecline('stage infrastructure '+type(exc).__name__+': '+str(exc)) from exc
        finally:
            if planner is not None:planner.close()
        self.check_deadline()
        sample=stratified_samples(candidate['task_layout'],candidate['request_sha256'])
        for index in sample['indices']:
            reference=json.dumps(encode_value(serial_task(tasks[index]),limit=self.control.limits['walk_limit']),sort_keys=True,separators=(',',':'),allow_nan=False)
            if reference!=candidate['results'][index]:
                self.control.directory.mkdir(parents=True,exist_ok=True)
                (self.control.directory/('sample-mismatch-'+kind+'.json')).write_text(json.dumps(dict(index=index,reference=json.loads(reference),candidate=json.loads(candidate['results'][index])),indent=2),encoding='utf-8')
                raise QualificationDecline('stage sample mismatch '+kind+':'+str(index))
            self.check_deadline()
        candidate.update(stage=kind,sampling=sample)
        self.control.reports.append(candidate)
        return tuple(decode_value(json.loads(value),limit=self.control.limits['walk_limit']) for value in candidate['results'])


def qualified_plan(model,control,serial,parallel,*,policy,mode,eligible=True):
    if type(control) is not StageControl:
        _increment('parallel_planning_decline:mandatory_ci');return serial()
    bridge=None;before=None
    def source_bytes():return json.dumps(to_dict(model),sort_keys=True,separators=(',',':'),allow_nan=False)
    try:
        if control.mode!='not-run':raise QualificationDecline('stage control already used')
        if not eligible or mode!='exact' or policy.max_predicates is not None or policy.max_candidate_pairs is not None or policy.cancellation_check is not None:
            raise QualificationDecline('PP3/special-route adapter unavailable')
        check_ci_equality(control.expected_ci,control.expected_ci)
        if not callable(control.observed_ci):raise QualificationDecline('mandatory applied-model CI callback missing')
        if type(control.timeout) not in (int,float) or not 0<control.timeout<=110:raise QualificationDecline('stage timeout must be in (0,110]')
        packet=freeze_model(model,**control.limits);before=source_bytes()
        if control.admission is not None and control.admission(model,policy) is not True:
            raise QualificationDecline('PP4 host/memory admission refused')
        if source_bytes()!=before:raise PlannerSourceChanged('source changed during admission')
        bridge=Bridge(packet,control)
        candidate=parallel(bridge)
        bridge.check_deadline()
        check_ci_equality(control.expected_ci,control.observed_ci(candidate))
        bridge.check_deadline()
        if source_bytes()!=before:raise PlannerSourceChanged('source changed during stage planning/CI')
        control.mode='parallel';control.ci_verified=True
        return candidate
    except QualificationDecline as exc:
        if before is not None and source_bytes()!=before:
            raise PlannerSourceChanged('source changed before serial fallback') from exc
        if bridge is not None:bridge.check_deadline()
        control.mode='serial';control.decline_reason=type(exc).__name__+': '+str(exc)
        _increment('parallel_planning_decline:'+str(exc))
        return serial()
