# CH3e public delivery and installed pilot

This harness prepares the minimum Q08 owner delivery. A successful run establishes installed-artifact point-query evidence for the frozen B1 registration. It does not admit the ANYstability provider, qualify S2/P0/P1, establish equilibrium or GZ curves, or cover tanks/CH4. Independent review and parent acceptance remain separate.

## Public API and supported domain

Use `anygeometry.b1_example.build_b1_hull()` (full hull, primary) and `build_b1_half_hull()` with defaults: x frames 1.5/3.5 m, bottom z=0.5 m, deck z=3 m. Closed outward boundary face IDs are the returned dictionary values. Model and boundary serialization are delivered together, with content hashes and the immutable prepared model/revision/document/snapshot identities. Model IDs may differ between independently constructed fixtures; the serialized delivered fixture and its identity are authoritative for its own proofs.

The public `anygeometry.certified_hydrostatics` route is:

```python
policy = CertifiedHydrostaticsPolicy(convention_version="2")
prepared = prepare_certified_hydrostatics(model, tuple(faces.values()), policy=policy)
query = HalfSpaceQuery(normal, offset, origin, axes,
                      {name: Fraction(1, 10**9) for name in names}, ("binary64",))
result = query_certified_hydrostatics(prepared, query)
expected = {"artifact": prepared.implementation, "convention_version": "2",
            "units": "m", "policy": policy}
checked = verify_certified_hydrostatics(model, tuple(faces.values()), query, result, expected)
```

Names in order are V, Jx, Jy, Jz, A, Mu, Mv, Iuu, Ivv, Iuv. J is the volume first moment about the body origin. Waterplane raw first moments are integrals of u,v; raw second moments are integrals of u²,v²,uv in that exact order. Units are SI (V m³; J m⁴; A m²; waterplane first m³; waterplane second m⁴). The public binary64 output is an interval (`float.hex()` endpoints in JSON), never an unbounded nominal value. All ten requested absolute widths are at most 1e-9 including conversion.

The frozen near-upright domain comprises the exact twelve consumer registrations, including inclined `R=Ry(-trim)Rx(heel)`, n=Rᵀe_z and the exact rounded registered offset. No coefficient is normalized or nudged. Each numeric input becomes `Fraction.from_float` through `Fraction(value)`. The basis axes are registered rotation rows 0 and 1. For horizontal cuts origin=(2.5,0,h); for inclined cuts origin=(2.5,0,1.75). Coordinates mean dot products u=e1·(r-o), v=e2·(r-o), not coefficients in a nonorthogonal basis. Physical area is measured in the actual plane. Binary64 rotation rows are not exactly orthonormal/tangent as rationals; the registered offset and declared origin are not forced into exact rational incidence. Those supplied values are retained. Handedness is the exact sign of `(e1 × e2)·n`: all twelve registered pose-row bases are positive (right handed). The general public query requires this scalar to be nonzero and permits either sign; handedness does not replace the physical area element or change the defined dot coordinates.

Dry z=.25 returns EMPTY with zero V/J and absent waterplane quantities. Full z=3.25 returns FULL with bounded full-solid V/J and absent waterplane quantities. Absent area/moments are `None`; they are not invented zeros. Required regular partial cuts must answer successfully. Exact-boundary/tangent/coplanar/ruling-containing requests outside this pilot may return public typed `CertifiedHydrostaticsRefusal`; retain category, phase, sub_reason, exact request and structured cause. A refusal is not a successful regular pilot query. No claim of unrestricted planes is made.

## Proof, domain and topology contract

The governing references are [the base certified-hydrostatics design](agent_plans/CERTIFIED_HYDROSTATICS_DESIGN.md) sections3,5,6 and6a (frame, components/holes, public proof and canonical formats), and [the CH3 design](agent_plans/CERTIFIED_HYDROSTATICS_CH3_DESIGN.md) sections2 (admitted curved class),3.4 (convention-version2 snapshot),4.4 (quantity forms/bounds),5 (section topology/limits) and6 (independent verification). CH3 extends the base contract; the historical design run matrix does not replace this delivery's exact consumer twelve-plane registration. These documents define the governing obligations, not evidence that a pending installed pilot or review has passed.

A proof is canonical UTF-8 JSON binding `artifact`, `model`, `convention`, `preparation_policy`, the complete exact `query` and the complete `result` claim. Convention2 uses the extended exact curved snapshot; it is not the planar convention1 snapshot schema. Rational fields use reduced numerator/positive-denominator pairs, binary64 bounds use hexadecimal strings, and the model binding includes selected face IDs, model identity/revision, document checksum and exact snapshot digest. The artifact binding includes actual package-file hashes and runtime identity. The wheel SHA256 is additionally pinned by this delivery's candidate/installation evidence; it is not a substitute for the public verifier's expected runtime descriptor.

`verify_certified_hydrostatics` receives the supplied model, selected boundary, exact caller query, complete result and separately pinned expected artifact/convention/units/policy. Its acceptance concerns the proof/result binding, exact query identity, current model and artifact identity, independently reconstructed integration domain and topology, and every returned quantity/form/accuracy bound. A syntactically valid proof, matching checksum, reference agreement or returned work metadata alone does not establish acceptance. The harness retains the result/certificate and verifier outcome separately; a refusal or exhausted verification budget leaves verification incomplete. CH3 quantities which are enclosed rather than exact remain enclosures, including their binary64 conversion bounds.

Each present section reports identified loops with `outer` or `hole` roles and `components` associating each outer loop with its nested holes. A component is an outer region together with its holes; it is not a count of every loop. A hole removes material from that region. Curved cut-point identities include edge ID and ordinal, so intersections on the same edge remain distinguishable. These section holes must not be confused with unsupported trim holes: CH3 refuses holes on curved faces and on planar faces with rational-arc boundaries, as specified in its admitted domain. The regular partial B1 pilot sections are expected to have one component, one outer loop and no holes; this delivery does not claim a new multiple-component/hole campaign. Absent dry/full waterplane quantities do not create an invented zero-area component. For supported degenerate queries the public `below`/`above` sections represent one-sided limits and may have different topology; the twelve-plane pilot does not claim such a degeneracy campaign.
## Independent reference

`tools/ch3e_reference.py` does not import certified engines, kernels or verifiers. It derives each station from the public rational B1 net: at v=z/4, L=5.4(1-v)+6v and B=6(1-v)+7.5v, retaining binary64 control values and w=cos(pi/4). Write t=u/(1-u), q=x/L. Then t²+2wt=q/(1-q), and positive breadth is B(1+2wt)/(1+2wt+t²). `analytic-stations.json` records fifteen station evaluations.

A separate horizontal-slice implementation clips analytic y intervals against n·r<=d at each (x,z). Exact-in-y polynomial slice integrals yield V,J; adaptive paired 12/24 Gauss integration supplies the x,z integrals, with explicit breaks where the plane meets either station side. Waterplane integrals use z=(d-nx*x-ny*y)/nz, solve station endpoints, integrate raw dot-coordinate polynomials in y and integrate in x. Physical area factor is sqrt(1+(nx/nz)²+(ny/nz)²). It shares neither production boundary decomposition nor certified integration code. Station algebra checks and polynomial quadrature oracles are cheap development checks, separate from the B1 pilot.

Reference uncertainty is a numerical estimate (quadrature discrepancies plus an explicit 2e-11 root/float allowance), **not a rigorous enclosure**. A reference with estimated uncertainty above1e-9 fails corroboration. Reference comparisons and certificate validity are reported separately. Matching numerical answers cannot substitute for public independent verification of all claims and the complete domain.

The matched half-hull check uses only upright z=1.75: Vfull=2Vhalf. It does not assert that doubling one half works for inclined cuts. Full-hull ±0.1 heel checks reflection of all quantities: Jy, Mv, Iuv change sign; the other seven are invariant. Relations use overlapping returned binary64 enclosures.

## Preparation and installation

The parent freezes source/package/version, builds the 0.4.7 wheel from Git archive bytes, installs in an isolated environment and records build/install evidence. This harness does not mutate Git or install packages. Example commands (replace paths with the parent's frozen artifact locations):

```powershell
& C:/Python/Python314/python.exe tools/ch3e_pilot.py register `
  --consumer-src C:/Github/ANYstability/src --out C:/evidence/registration.json
& C:/Python/Python314/python.exe -m venv C:/evidence/installed-env
& C:/evidence/installed-env/Scripts/python.exe -m pip install C:/delivery/ANYgeometry-0.4.7-py3-none-any.whl
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& C:/evidence/installed-env/Scripts/python.exe -I C:/delivery/tools/ch3e_pilot.py run `
  --python C:/evidence/installed-env/Scripts/python.exe `
  --site C:/evidence/installed-env/Lib/site-packages `
  --wheel C:/delivery/ANYgeometry-0.4.7-py3-none-any.whl `
  --registration C:/evidence/registration.json --registration-sha256 <parent-frozen-SHA256> `
  --authority-marker C:/evidence/frozen-pilot-authority.consumed.json --out C:/evidence/pilot-once
```

The registration is captured directly from the consumer's `b1_plane_registrations().to_dict()` objects, with source-file hashes and exact rational requests. Registration capture does not construct the geometry/provider. The parent records the registration byte hash before launch and passes it explicitly. The run requires a fresh output directory and exclusively creates the parent-designated one-shot authority marker before launching the worker; an existing marker refuses execution. It has no automatic retry/resume. Python `-I`, a cleared PYTHONPATH in the child, pinned installed module path, version check and byte comparisons of installed package files against the wheel prevent accidental checkout testing. `installed-runtime.json` records actual module paths, file hashes, effective threads and Python. Numpy is the runtime dependency used by both the wheel and reference. mpmath is a development test extra, not an installed pilot prerequisite.

## Evidence and budgets

`candidate.json` binds wheel SHA256, registration identity, harness file hashes and runtime paths. Model/boundary JSONs bind fixture, face set, preparation policy, snapshot and implementation. Every request, full response claim, canonical certificate, independent verified claim, reference and comparison is retained. Worker stdout/stderr and exact command JSONs retain failure/partial output; errors have public typed fields where available. Certificates are never replaced by cached accepted values or bypassed verifier decisions.

Continuous parent `perf_counter` measures process wall, including launch, imports, fixture construction/serialization, preparation, queries, reference work, verification, evidence I/O and handoff. This accounting is not pytest duration and not a sum of rounded shell times. Each stage records start/end, cap, command identity, status, used/remaining and worker-specific wall separately. Cold preparation cap120s, warm query cap30s, total600s with a480s work boundary and120s final verification/handoff reserve. Public verifier operations may use remaining final reserve; verification stops at595s to leave5s for final evidence. There is one persistent owned process, no simultaneous numerical jobs, no retries, no extension. On timeout its entire owned Windows process tree is terminated by PID and partial evidence is retained. The parent must not run other numerical work concurrently.

`REPORT.json`, `REPORT.md` and `ledger.json` distinguish incomplete runs from installed_verified. A checked-out test pass is only development evidence. An installed_verified pilot still awaits the required independent review and parent acceptance; it confers no consumer admission or release/publication authority.
