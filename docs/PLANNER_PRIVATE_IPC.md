# Private planner IPC on Windows

Private packet reads and atomic packet/initialization publication share a bounded
retry policy. It applies only to Windows `PermissionError` values with WinError
5, 32 or 33, or errno 13 when no WinError is present. Other errors propagate.
Each logical stat/read or encoded temporary-write/replace has at most eight
attempts and a 50 ms monotonic retry window. Requested sleeps are bounded by
that window; OS scheduling and filesystem calls are not claimed preemptible.

The same encoded body is reused. Packet schema, JSON, checksum, payload and node
checks are unchanged and never retried. Source hashes and model checks do not
use the retry helper. The session adapter supplies cancellation, deadline and
episode hooks to the inner read operation, avoiding nested retry budgets.
Terminal callback exceptions stay outside JSON error normalization. Failed
planner status is interpreted before success-only infrastructure bindings;
all success bindings remain required.

Focused verification includes an actual Windows reader holding the destination,
transient and permanent failures, non-Windows/nonmatching errors, corrupted
packets, deadline/cancellation precedence and public cold/warm eight-worker
canaries. One successful canary in each cache state is regression evidence,
not a statistical guarantee against all filesystem failures. Persistent errors
still refuse or terminate according to the existing transaction contract.
