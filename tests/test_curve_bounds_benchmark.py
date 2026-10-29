"""Root-run bounded bounds replay; no execution at authoring time.

Example: python tests/test_curve_bounds_benchmark.py --output <evidence.json>
The external root harness owns process-wall accounting and its timeout/ledger.
"""


def main():
    import argparse
    import dataclasses
    from fractions import Fraction as Q
    import json
    from pathlib import Path
    import time
    from anygeometry.tolerance import TolerancePolicy
    from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection,NurbsIntersectionCurve,encode_curve
    import anygeometry.nurbs_surface_intersection._curve_bounds as bounds
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from test_nurbs_pair_public import model_pair
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--requests',type=int,default=32)
    args = parser.parse_args()
    if not 1 <= args.requests <= 256:
        raise ValueError('bounded replay requests must be in [1,256]')
    model,faces = model_pair()
    surfaces = tuple(model.faces[face].surface for face in faces)
    result = query_nurbs_surface_intersection(*surfaces,tolerances=TolerancePolicy())
    owner = NurbsIntersectionCurve.from_result(result)
    ci = 0; count = len(result.components[ci].arcs)
    requests = tuple((i%count,Q((i//count)%8,16),Q((i//count)%8+1,16)) for i in range(args.requests))
    rows = []
    def run(label,callback):
        started = time.perf_counter()
        values = tuple(callback(ai,a,b) for ai,a,b in requests)
        rows.append({'mode':label,'requests':len(values),'elapsed':time.perf_counter()-started})
        return values
    reference = run('uncached-full',lambda ai,a,b:bounds.bound_segment(result,ci,ai,(Q(ai)+a,Q(ai)+b),(a,b)))
    evaluate = bounds.prepare_bound_evaluator(result,ci)
    prepared = run('prepared-full',lambda ai,a,b:evaluate(ai,(Q(ai)+a,Q(ai)+b),(a,b),lambda:None))
    assert prepared == reference
    world = run('prepared-world',lambda ai,a,b:evaluate(ai,(Q(ai)+a,Q(ai)+b),(a,b),lambda:None,world_only=True))
    assert world == tuple((x.world_lo,x.world_hi) for x in reference)
    curve = NurbsSurfaceIntersectionCurve(owner)
    with intersection_curve_operation():
        # Include public owner/context preparation in the first real query.
        public = run('issued-public-world',lambda ai,a,b:curve.bounds((Q(ai)+a)/count,(Q(ai)+b)/count))
        started = time.perf_counter()
        for ai,a,b in requests:
            curve.bounds((Q(ai)+a)/count,(Q(ai)+a)/count)
        rows.append({'mode':'issued-public-certified-singletons','requests':len(requests),
                     'elapsed':time.perf_counter()-started})
    for actual,expected in zip(public,world):
        assert all(tuple(a) == tuple(b) for a,b in zip(actual,expected))
    payload = {'classification':'focused-proof-parity-replay-not-mesh-acceptance','algorithm':result.algorithm,
               'literal_owner':encode_curve(owner),'rows':rows,
               'requests':[(ai,str(a),str(b)) for ai,a,b in requests],
               'full_and_world_reference_parity':True}
    Path(args.output).write_text(json.dumps(payload,indent=2))
    print(json.dumps({'rows':rows,'reference_parity':True}))


if __name__ == '__main__':
    main()
