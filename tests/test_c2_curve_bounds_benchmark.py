"""Root-run C2 proposal + unchanged owner-sampler replay (author unrun)."""


def main():
    import argparse
    from fractions import Fraction as Q
    import json
    from pathlib import Path
    import time
    import anygeometry as owner
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from anygeometry.nurbs_surface_intersection._mesh_proposals import propose_c2_boundary_parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--geometry')
    parser.add_argument('--divisions',type=int,default=8)
    parser.add_argument('--edges',type=int,default=1)
    parser.add_argument('--chord-error',type=float,default=.000625)
    args = parser.parse_args()
    if not 1 <= args.divisions <= 32 or not 1 <= args.edges <= 6:
        raise ValueError('bounded replay requires divisions1..32 and edges1..6')
    with intersection_curve_operation():
        if args.geometry:
            model = owner.from_dict(json.loads(Path(args.geometry).read_text()))
        else:
            from test_nurbs_pair_public import model_pair
            model,faces = model_pair()
            owner.apply_intersections(model,owner.plan_intersections(model,faces,policy='imprint'),policy='imprint')
        regions = owner.query_material_surface_regions(model,tuple(model.faces))
        selected = [edge for edge in model.edges.values() if type(edge.curve) is NurbsSurfaceIntersectionCurve][:args.edges]
        original = tuple(Q(float(i/args.divisions)) for i in range(args.divisions+1))
        rows = []; requests = []
        for edge in selected:
            started = time.perf_counter()
            parameters = propose_c2_boundary_parameters(edge.curve,original,chord_error=args.chord_error)
            elapsed = time.perf_counter()-started
            assert set(original) <= set(parameters) and all(Q(float(t)) == t for t in parameters)
            rows.append({'edge':edge.id,'input_stations':len(original),'proposed_stations':len(parameters),
                         'proposal_seconds':elapsed,'parameters':[str(t) for t in parameters]})
            requests.append((model.handle('edge',edge.id),parameters))
        started = time.perf_counter()
        sampling = owner.query_prepared_material_boundary_sampling(model,regions,tuple(requests),chord_error=args.chord_error)
        elapsed = time.perf_counter()-started
        owner.validate_prepared_material_boundary_sampling_binding(model,regions,sampling)
        payload = {'classification':'proposal-plus-unchanged-sampler-diagnostic-not-cell-acceptance',
                   'input_geometry':args.geometry,'chord_error':args.chord_error,'rows':rows,
                   'sampler_seconds':elapsed,'verified_stations':[(edge.id,len(values)) for edge,values in sampling.edge_parameters]}
        Path(args.output).write_text(json.dumps(payload,indent=2))
        print(json.dumps(payload))


if __name__ == '__main__':
    main()
