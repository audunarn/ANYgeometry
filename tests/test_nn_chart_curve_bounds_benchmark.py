"""Root-run paired cold/warm chart-only/full-reference timing; not acceptance."""
def main():
    import argparse,json,time
    from fractions import Fraction as Q
    from pathlib import Path
    from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection,NurbsIntersectionCurve,IntersectionStatus
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from anygeometry.tolerance import TolerancePolicy
    from test_nurbs_pair_public import model_pair
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--requests',type=int,default=16)
    args=parser.parse_args()
    if not 1<=args.requests<=128:
        raise ValueError('request count must be in [1,128]')
    model,faces=model_pair()
    surfaces=tuple(model.faces[face].surface for face in faces)
    result=query_nurbs_surface_intersection(*surfaces,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED
    curve=NurbsSurfaceIntersectionCurve(NurbsIntersectionCurve.from_result(result))
    requests=tuple((Q(i,args.requests),Q(i+1,args.requests)) for i in range(args.requests))
    def reference(a,b):
        with intersection_curve_operation():
            boxes=curve._shape().enclose(a,b)
            name=('first_chart','second_chart')[curve.chart_support]
            return tuple((Q(min(getattr(row,name)[k][0] for row in boxes)),
                          Q(max(getattr(row,name)[k][1] for row in boxes))) for k in range(2))
    rows=[]; expected=None; contexts=[]
    for warm in (False,True):
        for order in (('reference','candidate'),('candidate','reference')):
            for method in order:
                # Separate fresh operations prevent one method's coefficient,
                # hull or enclosure caches priming the other's cold timing.
                with intersection_curve_operation():
                    context=curve._shape()._context
                    assert all(context is not previous for previous in contexts), 'cold operation reused a context'
                    contexts.append(context)  # keep identities alive for this bounded diagnostic
                    call=reference if method=='reference' else curve.chart_box
                    if warm:
                        tuple(call(a,b) for a,b in requests)
                    tick=time.perf_counter()
                    values=tuple(call(a,b) for a,b in requests)
                    elapsed=time.perf_counter()-tick
                    if expected is None:
                        expected=values
                    assert values==expected
                    rows.append(dict(method=method,warm=warm,order=list(order),seconds=elapsed,
                                     requests=len(requests)))
    args.output.write_text(json.dumps(dict(classification='paired-chart-proof-parity-only',
        exact_parity=True,rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(exact_parity=True,rows=rows)))


if __name__=='__main__':
    main()
