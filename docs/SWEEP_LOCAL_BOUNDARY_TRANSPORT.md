Local boundary allocation queries use a freshly derived source sweep chart for
canonical raw Coons faces without an installed chart recipe. The public source
view retains original face/edge identity and performs the existing full connector
correspondence checks. An unsupported view retains the previous frozen-edge path
and its conservative chart-enclosure refusal.

Only the UV footprint lookup uses the view's CurveOnSurface descriptor. It does
not replace the actual edge, install chart metadata, change world sampling, or
declare raw connector vertices exactly on the ideal rational support. Existing
nonzero boundary-to-support transport/error records, interval allocations,
incident-face minima, source binding, cancellation and final guards remain in
force. No new tolerance, inverse mapping, fitted geometry or receipt is added.
