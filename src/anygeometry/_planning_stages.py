"""PP5 finite stage assignments over one constructor-decoded owned model.

Each kernel receives a disposable view with fresh selected surface constructors
and fresh model caches. The retained cold model is checked at assignment boundaries.
No mutation/clearing of the source model or its instance caches is authorized here.
"""
from types import MappingProxyType
from .model import GeometryModel
import json
import math
import os
import time
from dataclasses import replace
from ._planning_executor import freeze_model,thaw_model,pack,unpack
from ._planning_qualification import QualificationDecline
from ._planning_values import encode_value,decode_value,same
from .material_arrangement import MaterialDomain,arrange_material
from .batch_intersections import _pair_paths,_domain_bounds


class WorkerModel:
    def __init__(self,packet,*,payload_limit=4_000_000,walk_limit=100000,value_codec=None):
        from ._planning_values import _value_codec
        _value_codec(value_codec)
        self.value_codec=value_codec;self._value_codec_binding=value_codec
        self.limits=dict(payload_limit=payload_limit,walk_limit=walk_limit)
        self.model=thaw_model(packet,**self.limits)
        self.packet=packet
        self.pid=os.getpid()
        self.verify()

    def verify(self):
        if self.pid!=os.getpid():raise QualificationDecline('worker model ownership changed')
        if self.value_codec is not self._value_codec_binding:raise QualificationDecline('worker value codec changed')
        actual=unpack(freeze_model(self.model,**self.limits),'model',limit=self.limits['payload_limit']);recorded=unpack(self.packet,'model',limit=self.limits['payload_limit'])
        if recorded['extension_record']==dict(present=False,sha256=None):
            if self.model._serialization_extensions!={}:raise QualificationDecline('worker extension changed')
            actual['extension_record']=recorded['extension_record']
        digest=pack(actual,'model',limit=self.limits['payload_limit'])['sha256']
        if digest!=self.packet['sha256']:raise QualificationDecline('worker model changed')
        return digest

    def view(self,faces):
        if self.pid!=os.getpid():raise QualificationDecline('worker model ownership changed')
        view=GeometryModel(model_id=self.model.model_id,tolerance=self.model.tolerance)
        view.__dict__.update(vars(self.model))
        # These are constructor-defined empty caches already verified at admission.
        for name in ('_arc_cache','_edge_length_cache','_trim_loops_cache','_topology_corner_cache','_chain_arrays'):
            setattr(view,name,{})
        view._change_hooks=[]
        view._faces=dict(self.model.faces)
        view.faces=MappingProxyType(view._faces)
        for face_id in faces:
            face=self.model.faces[face_id]
            surface=replace(face.surface)
            if hasattr(surface,'_circumferential') and not face.surface._circumferential.flags.writeable:
                surface._circumferential.flags.writeable=False
            view._faces[face_id]=replace(face,surface=surface)
        return view


def stage_task(context,task):
    if type(context) is not WorkerModel or type(task) is not dict:
        raise QualificationDecline('stage context/task schema')
    kind=task.get('kind')
    started=(time.perf_counter(),time.process_time())
    if kind=='pair':
        model=context.view((task['first'],task['second']))
        first=MaterialDomain.from_model(model,task['first']);second=MaterialDomain.from_model(model,task['second'])
        _domain_bounds(first);_domain_bounds(second)
        points=[]
        kernel_start=(time.perf_counter(),time.process_time())
        value=_pair_paths(first,second,model,task['tolerance'],lambda:None,
                          point_contacts=points,face_tolerances=tuple(task['face_tolerances']),route=None)
        kernel_end=(time.perf_counter(),time.process_time())
        result=(value,points)
    elif kind=='face':
        model=context.view((task['face'],))
        domain=MaterialDomain.from_model(model,task['face'])
        _domain_bounds(domain)
        if 'support' in task:
            support=decode_value(task['support'],limit=context.limits['walk_limit'],codec=context.value_codec,payload_limit=context.limits['payload_limit'])
            if not same(support,domain.support):raise QualificationDecline('face support definition mismatch')
            domain=replace(domain,support=support)
        traces,points=decode_value(task['inputs'],limit=context.limits['walk_limit'],codec=context.value_codec,payload_limit=context.limits['payload_limit'])
        seam={} if task['seam']=='baseline' else {'seam_policy':task['seam']}
        kernel_start=(time.perf_counter(),time.process_time())
        result=arrange_material(domain,traces,points=points,tolerance=task['tolerance'],
            area_tolerance=task['area_tolerance'],cancellation_check=lambda:False,**seam)
        kernel_end=(time.perf_counter(),time.process_time())
    else:raise QualificationDecline('unknown stage task')
    encoded=json.dumps(encode_value(result,limit=context.limits['walk_limit'],codec=context.value_codec,payload_limit=context.limits['payload_limit']),sort_keys=True,separators=(',',':'),allow_nan=False)
    end=(time.perf_counter(),time.process_time())
    # Receipt-only comparable split: wall and process CPU seconds, same clocks for every task.
    # setup = worker view + domain construction + bounds + input/support decode (host serial
    # precomputes these outside the matched kernel); kernel = only _pair_paths / arrange_material,
    # the same calls host _timed_phase intersection_support_and_clipping / _arrangements wrap.
    context.last_timing=dict(setup_wall=kernel_start[0]-started[0],setup_cpu=kernel_start[1]-started[1],
        kernel_wall=kernel_end[0]-kernel_start[0],kernel_cpu=kernel_end[1]-kernel_start[1],
        encode_wall=end[0]-kernel_end[0],encode_cpu=end[1]-kernel_end[1],encoded_utf8_bytes=len(encoded.encode('utf-8')))
    return encoded


def dispatch(tasks,workers):
    if type(workers) is not int or workers not in (1,2,8,16):raise QualificationDecline('stage worker count')
    if not tasks:return [],[]
    count=min(workers,len(tasks));parts=[[] for _ in range(count)];loads=[0.]*count;remaining=set(range(len(tasks)))
    for task in tasks:
        if type(task) is not dict or task.get('kind') not in ('pair','face') or type(task.get('cost')) not in (int,float) or not math.isfinite(task['cost']) or task['cost']<=0:
            raise QualificationDecline('stage cost/task schema')
    # Seed each available class across workers; pure stage assignments have one class.
    for kind in ('pair','face'):
        ordered=sorted((i for i in remaining if tasks[i]['kind']==kind),key=lambda i:(-tasks[i]['cost'],i))
        available=sorted(range(count),key=lambda w:(loads[w],w))
        for i,w in zip(ordered,available):
            parts[w].append(i);loads[w]+=tasks[i]['cost'];remaining.remove(i)
    for i in sorted(remaining,key=lambda i:(-tasks[i]['cost'],i)):
        w=min(range(count),key=lambda w:(loads[w],w));parts[w].append(i);loads[w]+=tasks[i]['cost']
    layout=[None]*len(tasks)
    for w,part in enumerate(parts):
        part.sort(key=lambda i:(-tasks[i]['cost'],i))
        for j,i in enumerate(part):layout[i]=[w,j,tasks[i]['kind']]
    return parts,layout
