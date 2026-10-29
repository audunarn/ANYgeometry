"""CH3e evidence schema and exact consumer registration capture; no provider imports."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import sys

NAMES=('V','Jx','Jy','Jz','A','Mu','Mv','Iuu','Ivv','Iuv')

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def write_json(path,value):
    Path(path).write_bytes(canonical(value)+b'\n')

def pair(value):
    q=Fraction(value)
    return [q.numerator,q.denominator]

def capture(consumer_src):
    sys.path.insert(0,str(Path(consumer_src).resolve()))
    try:
        module=importlib.import_module('anystability.providers.registration')
        regs=module.b1_plane_registrations()
        if len(regs) != 12: raise ValueError('exactly twelve registrations required')
        planes=[]
        for reg in regs:
            raw=reg.to_dict()
            n=reg.plane.normal; d=reg.plane.offset_m
            upright=reg.construction == 'horizontal_z'
            origin=(2.5,0.,d if upright else 1.75)
            axes=reg.pose.rotation[:2]
            query=dict(normal=[pair(x) for x in n],offset=pair(d),
                       waterplane_origin=[pair(x) for x in origin],
                       waterplane_axes=[[pair(x) for x in row] for row in axes],
                       absolute_accuracy={k:[1,1000000000] for k in NAMES}, forms=['binary64'])
            planes.append(dict(registration=raw,registration_identity=reg.identity,query=query))
        files={}
        for name in ('registration','integrals','protocol'):
            path=Path(module.__file__).with_name(name+'.py')
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('conventions','validation'):
            path=Path(module.__file__).parent.parent/(name+'.py')
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        body=dict(schema='anygeometry-ch3e-registration-v1',planes=planes,
                  consumer_source_files=files,basis='u=e1 dot (r-o), v=e2 dot (r-o)',
                  budget=dict(total_process_wall_s=600.,work_boundary_s=480.,
                              final_verification_reserve_s=120.,cold_prep_s=120.,per_query_s=30.))
        return dict(**body,identity=digest(body))
    finally:
        sys.path.pop(0)

def validate_registration(data):
    if data.get('schema') != 'anygeometry-ch3e-registration-v1':
        raise ValueError('unknown registration schema')
    body={k:v for k,v in data.items() if k != 'identity'}
    if digest(body) != data.get('identity'): raise ValueError('registration identity mismatch')
    planes=data['planes']
    if len(planes) != 12 or len({p['registration']['registration_id'] for p in planes}) != 12:
        raise ValueError('twelve unique requests required')
    expected_ids={f'b1-upright-draft-{h}' for h in (1.0,1.75,2.5)}
    expected_ids.update(f'b1-{kind}-{sign}-0.1' for kind in ('heel','trim') for sign in ('plus','minus'))
    expected_ids.update(('b1-combined-plus-0.1-plus-0.1','b1-combined-minus-0.1-minus-0.1',
        'b1-dry-horizontal-0.25','b1-full-horizontal-3.25','b1-bottom-adjacent-0.500001'))
    if {p['registration']['registration_id'] for p in planes} != expected_ids:
        raise ValueError('registered pilot identities changed')
    for p in planes:
        reg=p['registration']; plane=reg['plane']; q=p['query']
        if q['normal'] != [pair(x) for x in plane['normal']] or q['offset'] != pair(plane['offset_m']):
            raise ValueError('query differs from exact registered coefficients')
        if reg['request']['plane'] != plane: raise ValueError('consumer request plane mismatch')
        origin=(2.5,0.,plane['offset_m'] if reg['construction']=='horizontal_z' else 1.75)
        if q['waterplane_origin'] != [pair(x) for x in origin]: raise ValueError('origin changed')
        if q['waterplane_axes'] != [[pair(x) for x in row] for row in reg['pose']['rotation_body_to_world'][:2]]:
            raise ValueError('registered dot basis changed')
        if p['query']['forms'] != ['binary64'] or p['query']['absolute_accuracy'] != {n:[1,1000000000] for n in NAMES}:
            raise ValueError('frozen quantity/accuracy policy changed')
    return data
