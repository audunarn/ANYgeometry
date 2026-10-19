import cProfile, pstats, sys, os
from collections import defaultdict
sys.argv = [sys.argv[0]] + sys.argv[1:]
import prof
GEOM = {"surfaces","curves","exact_curves","arrangement_geometry","analytic_roots","quadric_algebra","quadric_curves",
 "quadric_supports","quadric_events","quadric_curve_events","branch_algebra","branch_curves","branch_supports","branch_events",
 "predicates","evaluation","cylinder_charts","cylinder_patch","trim_domains","_cross3","extruded_chart","analytic_supports",
 "curved_intersections","bezier_intersections","cylinder_curve_events","polynomial_extrusion_support","planar_area","trimmed_charts","spatial","edge_subcurve_preimages","cylinder_angular_lifts"}
for name in sys.argv[1:]:
    run = prof.W[name](); pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
    st = pstats.Stats(pr); T = 0; bym = defaultdict(float); bucket = defaultdict(float)
    # attribute numpy/builtin self time to the calling anygeometry module (first caller)
    for key, (cc, nc, tt, ct, callers) in st.stats.items():
        T += tt; fn = key[0].replace("\\", "/")
        if "anygeometry" in fn:
            mod = os.path.basename(fn)[:-3]
        else:
            best = None
            for ck, cv in callers.items():
                if "anygeometry" in ck[0].replace("\\","/"):
                    if best is None or cv[2] > best[1]: best = (os.path.basename(ck[0])[:-3], cv[2])
            mod = best[0] if best else "(lib)"
        bym[mod] += tt
        bucket["geometry kernels" if mod in GEOM else ("(lib)" if mod=="(lib)" else "model/topology/bookkeeping")] += tt
    print(f"== {name}")
    for k, v in sorted(bucket.items(), key=lambda kv:-kv[1]): print(f"  {k:30s} {100*v/T:5.1f}%")
    print("   top modules:", ", ".join(f"{k} {100*v/T:.0f}%" for k, v in sorted(bym.items(), key=lambda kv:-kv[1])[:10]))
