"""Bounded whole-domain cylinder arrangement experiments."""
import json
import sys
import traceback
import pickle
import faulthandler
from pathlib import Path
from time import perf_counter
import numpy as np
from anygeometry import GeometryModel, ConnectionIntent, IntersectionBatchPolicy, plan_intersections, apply_intersections, query_trimmed_surface_charts, to_dict
from anygeometry.generators import cylinder

cases={
 'parallel':dict(radius=.9,origin=(1.,0.,-1.),axis=(0.,0.,1.),radial_direction=(1.,0.,0.)),
 'perpendicular':dict(radius=1.,origin=(0.,-1.5,0.),axis=(0.,1.,0.),radial_direction=(1.,0.,0.)),
 'skew':dict(radius=.8,origin=(-.5,-1.,-.5),axis=(1.,2.,1.),radial_direction=(1.,0.,0.)),
 'tangent':dict(radius=1.,origin=(2.,0.,-1.5),axis=(0.,0.,1.),radial_direction=(1.,0.,0.)),
}
if __name__=='__main__':
    name=sys.argv[1]
    m=cylinder(1.,3.,origin=(0.,0.,-1.5),circumferential_segments=8)
    m.insert_model(cylinder(height=3.,circumferential_segments=8,**cases[name]))
    start=perf_counter()
    budget=float(sys.argv[2]) if len(sys.argv)>2 else None
    policy=IntersectionBatchPolicy(cancellation_check=(
        None if budget is None else lambda: perf_counter()-start>budget))
    faulthandler.dump_traceback_later(45,repeat=False)
    try:
        before=to_dict(m)
        plan=plan_intersections(m,tuple(m.faces),policy=policy)
        assert before==to_dict(m)
        print('PLANNED',len(plan.arrangements),perf_counter()-start,flush=True)
        result=apply_intersections(m,plan,policy=policy)
        print('APPLIED',len(m.faces),len(result.joint_edges),m.validate_topology(),flush=True)
        charts=query_trimmed_surface_charts(m)
        area=sum(chart.material_area for chart in charts.charts)
        expected=2*np.pi*3*(1+cases[name]['radius'])
        print('AREA',area,'EXPECTED',expected,'SECONDS',perf_counter()-start,flush=True)
        assert abs(area-expected)<1e-8
        Path(__file__).with_name(f'cylinder-{name}.json').write_text(json.dumps(to_dict(m)))
    except Exception:
        traceback.print_exc()
        error=sys.exc_info()[1]
        while error.__cause__ is not None: error=error.__cause__
        tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name=='area_loop':
                values=tb.tb_frame.f_locals
                with Path(__file__).with_name(f'cylinder-{name}-area.pickle').open('wb') as stream:
                    pickle.dump((values['self'],values['loop'],values['tolerance']),stream)
            if tb.tb_frame.f_code.co_name=='arrange_material':
                selected={key:tb.tb_frame.f_locals[key] for key in (
                    'domain','tolerance','edges','vertices','endpoints_by_edge','outgoing','chain')
                    if key in tb.tb_frame.f_locals}
                with Path(__file__).with_name(f'cylinder-{name}-arrangement.pickle').open('wb') as stream:
                    pickle.dump(selected,stream)
            tb=tb.tb_next
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
