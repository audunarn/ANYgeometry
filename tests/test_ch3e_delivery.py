"""Cheap CH3e math/schema checks: no hull construction or certified computation."""
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from ch3e_reference import station_width,integrate
from ch3e_schema import pair,digest,validate_registration,NAMES
from ch3e_pilot import compare_reference,quantity_intervals,relation


def test_station_quadratic_matches_public_control_net():
    # Evaluate the independently stated public rational Bernstein control net.
    from anygeometry.b1_example import b1_bow_surface
    surface=b1_bow_surface()
    w=math.cos(math.pi/4)
    for u in (.12,.25,.4):
        for z in (.5,1.75,3.):
            v=z/4; b=((1-u)**2,2*w*u*(1-u),u*u); den=sum(b)
            x=(5.4*(1-v)+6*v)*(b[1]+b[2])/den
            y=(6*(1-v)+7.5*v)*(b[0]+b[1])/den
            # Public evaluate coordinates are normalized over u_range=(0,.9).
            chart_u=(u-surface.u_range[0])/(surface.u_range[1]-surface.u_range[0])
            point=surface.evaluate(chart_u,v)
            assert point==pytest.approx((x,y,z),abs=3e-15,rel=0)
            assert station_width(point[0],point[2])==pytest.approx(point[1],abs=3e-15,rel=0)


def test_slice_quadrature_has_polynomial_oracle():
    actual,error=integrate(lambda x:np.array([1.,x,x*x,x**7]),-1.,2.)
    assert actual==pytest.approx([3.,1.5,3.,255/8],abs=4e-13,rel=0)
    assert error < 1e-11


def test_plane_coefficients_are_float_exact_not_decimal_roundtrips():
    x=math.sin(.1)
    p=pair(x)
    assert Fraction(*p)==Fraction.from_float(x)
    assert Fraction(*p)!=Fraction(str(x))


def claim(absent=False):
    def quantity(value):
        return dict(accuracy_met=True,binary64=dict(lo=float(value).hex(),hi=float(value).hex()))
    return dict(volume=quantity(1),moments={k:quantity(0) for k in NAMES[1:4]},
        sections=dict(plane=dict(section='ABSENT' if absent else 'PRESENT',
            quantities=None if absent else {k:quantity(0) for k in NAMES[4:]})))


def test_reference_comparison_checks_all10_and_absence():
    c=claim(); values={k:0. for k in NAMES};values['V']=1.
    ref=dict(values=values,estimated_absolute_uncertainty=2e-11)
    assert compare_reference(c,ref)['agrees']
    values['Iuv']=.01
    assert not compare_reference(c,ref)['agrees']
    assert quantity_intervals(claim(True))['A'] is None


def test_width_and_accuracy_fail_closed():
    c=claim();c['volume']['binary64']['hi']=(1.+2e-9).hex()
    with pytest.raises(ValueError,match='width'): quantity_intervals(c)
    c=claim();c['moments']['Jy']['accuracy_met']=False
    with pytest.raises(ValueError,match='accuracy'): quantity_intervals(c)


def test_reference_uncertainty_is_not_a_certificate_waiver():
    ref=dict(values={k:0. for k in NAMES},estimated_absolute_uncertainty=1e-5)
    with pytest.raises(ValueError,match='uncertainty'):compare_reference(claim(),ref)


def test_matched_relations_preserve_reflection_sign():
    assert relation((2.,2.),(1.,1.),2)
    assert relation((-1.,-1.),(1.,1.),-1)
    assert not relation((-1.,-1.),(1.,1.),1)


def test_registration_hash_and_exact_count_fail_closed():
    body=dict(schema='anygeometry-ch3e-registration-v1',planes=[])
    data=dict(**body,identity=digest(body))
    with pytest.raises(ValueError,match='twelve'):validate_registration(data)
    data['schema']='changed'
    with pytest.raises(ValueError):validate_registration(data)

def test_one_shot_authority_is_not_reusable(tmp_path):
    from ch3e_pilot import consume_authority
    marker=tmp_path/'once.json'
    consume_authority(marker,{'wheel':'frozen'},tmp_path,'registration')
    assert json.loads(marker.read_text())['consumed'] is True
    with pytest.raises(FileExistsError):
        consume_authority(marker,{'wheel':'frozen'},tmp_path,'registration')


def test_mocked_timeout_retains_command_and_ends_owned_worker(tmp_path,monkeypatch):
    import queue
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ch3e_pilot import Pilot
    pilot=Pilot(SimpleNamespace(out=str(tmp_path/'pilot')))
    pilot.proc=SimpleNamespace(stdin=Mock())
    pilot.responses=Mock();pilot.responses.get.side_effect=queue.Empty
    stopped=[];monkeypatch.setattr(pilot,'stop',lambda:stopped.append(True))
    with pytest.raises(TimeoutError):pilot.command(dict(op='query',id='fixed'))
    ledger=json.loads((pilot.root/'ledger.json').read_text())
    assert stopped==[True]
    assert ledger['stages'][0]['status']=='timeout'
    assert ledger['stages'][0]['timeout_s']==30.
    assert json.loads((pilot.root/'command-001.json').read_text())['id']=='fixed'


def test_isolated_launch_adds_only_tool_path_not_checkout_source(tmp_path,monkeypatch):
    import io
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ch3e_pilot import Pilot
    args=SimpleNamespace(out=str(tmp_path/'evidence'),python=sys.executable,
        site=str(tmp_path/'site-packages'),wheel=str(tmp_path/'fixed.whl'))
    pilot=Pilot(args);calls=[]
    fake=SimpleNamespace(stdout=io.StringIO(''),stdin=Mock())
    def popen(command,**kwargs):calls.append((command,kwargs));return fake
    monkeypatch.setattr('ch3e_pilot.subprocess.Popen',popen)
    monkeypatch.setenv('PYTHONPATH','some/checkout/src')
    pilot.start()
    command,kwargs=calls[0]
    assert command[1]=='-I'
    assert 'PYTHONPATH' not in kwargs['env']
    assert Path(command[2]).name=='ch3e_worker.py'
    assert kwargs['cwd']==pilot.root
    pilot.stderr.close()


def test_worker_and_harness_sources_parse_without_provider_import():
    import ast
    for path in TOOLS.glob('ch3e_*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
