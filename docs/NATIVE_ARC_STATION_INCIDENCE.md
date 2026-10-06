# Prepared native Arc station incidence

`query_prepared_native_arc_station_incidence(model, stations, *, maps=None,
tolerance=None, policy=None, preserve_ancestor=True, cancellation_check=None)`
certifies one immutable, read-only batch of explicit finite stations against
captured native Arc harmonics after actual `plan_intersections` /
`apply_intersections` preparation. Each `NativeArcStation(edge_id, s, point)`
names one current edge ID, one finite parameter `s` in [0,1] and one supplied
finite XYZ point. `s` and the point are normalized to exact rationals at
construction; nonfinite or invalid input is refused there. Coordinates and
tolerances are geometry-document coordinates and units; no transform, unit or
CRS conversion is performed. An optional `ancestor_edge_id` selects one
original or alias ancestor map when an edge carries several; without it,
exactly one candidate map must exist or the request is refused as ambiguous.

The certified quantity is the direct outward Euclidean distance bound from
the supplied point to the captured harmonic `H(s) = C + U cos(w s) +
V sin(w s)` of the named current edge, computed with the existing owner
`_Proof` interval sincos/sqrt arithmetic on the exact binary-rational frame
coefficients authenticated by `query_prepared_native_arc_parameter_maps`.
Every bound is compared with the full unchanged model positional tolerance on
its own. The caller may request a stricter tolerance, never a looser one.

## Ancestor scope (default)

With `preserve_ancestor=True` each station needs one unambiguous qualified
ancestor map (classification `exact` or `bounded`); unavailable, ambiguous or
unqualified maps refuse. When the selected map claims ancestor preservation,
the direct bound to the ancestor harmonic `H_ancestor(a + (b-a) s)` is
certified separately. The two bounds each receive the full tolerance and are
never added: a point whose current and ancestor bounds are both within
tolerance is admitted even when their sum exceeds it, and a single bound at
one and a half tolerances is refused even though the sum of two full
tolerances would cover it. Records carry the selected map's ancestor
identity (edge, model, revision, source checksum), interval and frames, and
`scope='ancestor'`.

## Current-only scope

With `preserve_ancestor=False` the batch certifies only the supplied XYZ
against the actual captured current harmonic of each named current edge. No
ancestor is invented from endpoints, IDs or closest curves, and no
ancestor-preservation claim is made, so no map receipt is required: the
binding is the authenticated prepared current Arc definition, captured
(detached) before the first caller callback and re-checked after the final
query callback and after the validator callback. A supplied map receipt is
owner-guarded historical evidence without re-execution cost; its frames never
replace actual current definitions captured from active edges and vertices.
Newly generated joint Arcs need no authored Arc ancestry. Its classification, qualified or not,
does not gate this scope, and the record honestly reports the observed
classification. Current-only records carry `scope='current_only'` with every
ancestor field `None`. Current-only stations cannot select an ancestor.
Unqualified source maps remain refused whenever ancestor preservation is
requested.

## Frozen request and aggregate budget

The whole request is frozen into a detached, revalidated snapshot before the
first caller callback can run; only that snapshot is certified and stored in
the receipt. After the final callback the caller's original objects are
and container are re-compared (object identity, selection, order, edge, `s`, XYZ, ancestor) and any change
refuses atomically, so final-check and mid-arithmetic
`object.__setattr__` adversaries cannot desynchronize certified records from
the stored request. Later mutation of the caller's station objects cannot
alter an issued receipt.

One `NativeArcStationIncidencePolicy` bounds the aggregate NEW arithmetic of
the whole batch. When no map receipt is supplied in the ancestor scope, the
map query receives the remaining allowance and its actual work is charged
into the same proof; there is no second default 200000 budget. A previously
issued map receipt is historical: it is not re-executed and its proof cost is
not charged again. Cancellation, exhausted budget, stale or forged inputs
issue no partial receipt.

## Authentication and narrow semantics

Receipts are owner-issued with pinned digests; the binding validator refuses
forged, copied, mutated, wrong-model and stale receipts, and re-checks the
prepared ancestry (and, for current-only receipts, the captured current
definitions) after its callback. The public query is read-only. Equality of
saved and mesh XYZ coordinates is never accepted as proof: an off-curve
supplied point refuses even if it equals some stored coordinate.

This contract certifies nothing else. It is not a global libm or float
implementation theorem, not exact three-point document-circle incidence, not
surface/chart material or reference closure, and not high-order or mesh
admission. Flags and claims stay narrow to the certified outward bounds
above.
