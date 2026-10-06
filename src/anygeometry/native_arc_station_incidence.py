"""Owner-bound finite-station incidence evidence for captured native Arc maps.

Each station is one explicit finite point supplied by the caller against one
explicitly named captured harmonic H(s) = C + U cos(w s) + V sin(w s) whose
exact binary-rational coefficients were authenticated by
``query_prepared_native_arc_parameter_maps``.  The certified quantity is the
direct outward Euclidean distance bound from the supplied point to that
captured function at that finite station.  Stations are certified in one of
two explicit scopes.  The default ``preserve_ancestor=True`` scope certifies
the captured current harmonic and, when the selected map claims ancestor
preservation, the direct bound to the ancestor harmonic H_ancestor(a + (b-a)s)
separately; unqualified maps are refused.  The explicit current-only scope
(``preserve_ancestor=False``) certifies only the supplied XYZ against the
actual captured current harmonic of the named current edge, bound to the
authenticated prepared current Arc definition; no ancestor is invented, no
ancestor-preservation claim is made, and no map receipt is required.  In that
scope the binding Arc definitions are captured live from the actual current
model edges and vertices through the owner ``_arc_edge_definition`` helper
before the first caller callback, never from ancestry records, so a newly
generated joint Arc without authored ancestry certifies exactly like an
inherited split child; the immutable captured definitions are pinned in the
receipt and re-checked against the actual model after the final query
callback and around the validator callback, and a supplied map receipt is
historical evidence only whose frames never replace the actual captured
current definition.  Records
distinguish the two scopes; ancestor fields are present only for
ancestor-scope evidence.

Each bound is compared with the full unchanged model positional tolerance on
its own; the two bounds are never added and no looser tolerance is accepted.
The whole request is frozen into a detached, revalidated snapshot before the
first caller callback can run; only that snapshot is certified, and the
caller's original container and objects are re-compared after the final
callback: the container must still hold the same station objects in the same
order with unchanged field values.  When no
map receipt is supplied in the ancestor scope, the NEW map-query arithmetic
shares the caller's single aggregate policy: the map query receives the
remaining allowance and its actual work is charged into the same proof; a
previously issued map receipt is historical and is not re-executed or
re-charged.

This contract certifies nothing else: it is not a global libm/float
implementation theorem, not three-point document-circle incidence, not
surface/chart material or reference closure, and not high-order or mesh
admission.  Equality of saved and mesh XYZ coordinates is never accepted as
proof; only the certified outward bounds above admit a station.  Coordinates
and tolerances are geometry-document coordinates and units; no transform,
unit or CRS conversion is performed.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from fractions import Fraction as F
from numbers import Integral
from weakref import WeakKeyDictionary, ref

from .cylinder_charts import _Proof, _Refusal
from .definition_binding import definition_checksum
from .edge_subcurve_preimages import (_arc_edge_definition,
    query_prepared_edge_subcurve_preimages)
from .errors import GeometryError
from .native_arc_parameter_maps import (NativeArcFrame,
    NativeArcParameterMapError, NativeArcParameterMapPolicy,
    PreparedNativeArcParameterMaps, query_prepared_native_arc_parameter_maps,
    validate_prepared_native_arc_parameter_maps_binding,
    _digest, _frame, _issued_binding_digest)


class NativeArcStationIncidenceError(GeometryError):
    """Invalid station request, refusal, or aggregate proof exhaustion; no partial receipt."""


def _finite_rational(value, description):
    if isinstance(value, bool) or isinstance(value, Integral):
        return F(int(value))
    if isinstance(value, F):
        return value
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            raise NativeArcStationIncidenceError(description)
        return F(value)
    raise NativeArcStationIncidenceError(description)


@dataclass(frozen=True, slots=True)
class NativeArcStationIncidencePolicy:
    """Aggregate arithmetic budget, independent of model or station counts."""
    max_interval_operations: int = 200000

    def __post_init__(self):
        if type(self.max_interval_operations) is not int or self.max_interval_operations <= 0:
            raise NativeArcStationIncidenceError('invalid aggregate work budget')


@dataclass(frozen=True, slots=True)
class NativeArcStation:
    """One explicit finite station: current edge ID, s in [0,1], finite XYZ.

    ``s`` and ``point`` are normalized to exact rationals of the supplied
    values; nonfinite or invalid input is refused at construction.  An
    optional ``ancestor_edge_id`` selects one original or alias ancestor map
    when an edge carries several; without it, exactly one candidate map must
    exist or the request is refused as ambiguous.  Current-only batches
    (``preserve_ancestor=False``) cannot select an ancestor.
    """
    edge_id: int
    s: F
    point: tuple
    ancestor_edge_id: int | None = None

    def __post_init__(self):
        if isinstance(self.edge_id, bool) or type(self.edge_id) is not int or self.edge_id <= 0:
            raise NativeArcStationIncidenceError('native Arc station edge must be a positive integer')
        if self.ancestor_edge_id is not None and (isinstance(self.ancestor_edge_id, bool)
                or type(self.ancestor_edge_id) is not int or self.ancestor_edge_id <= 0):
            raise NativeArcStationIncidenceError('native Arc station ancestor must be a positive integer')
        s = _finite_rational(self.s, 'native Arc station parameter must be a finite value in [0,1]')
        if not (0 <= s <= 1):
            raise NativeArcStationIncidenceError('native Arc station parameter must be a finite value in [0,1]')
        try:
            coordinates = tuple(self.point)
        except TypeError as error:
            raise NativeArcStationIncidenceError(
                'native Arc station point must be three finite coordinates') from error
        if len(coordinates) != 3:
            raise NativeArcStationIncidenceError('native Arc station point must be three finite coordinates')
        point = tuple(_finite_rational(x, 'native Arc station point must be three finite coordinates')
            for x in coordinates)
        object.__setattr__(self, 's', s)
        object.__setattr__(self, 'point', point)


@dataclass(frozen=True, slots=True)
class NativeArcStationIncidence:
    """Certified direct distance bounds for one station against named captures.

    ``scope`` is ``ancestor`` or ``current_only``.  ``current_bound`` is the
    certified outward Euclidean distance from the supplied point to the
    captured current harmonic H(s).  In the ancestor scope, ``ancestor_bound``
    is the certified outward distance to the ancestor harmonic at the ancestor
    parameter a + (b-a) s, present exactly when the selected map claims
    ancestor preservation, and the ancestor identity/interval/frame fields
    carry the selected map's captured ancestry.  In the current-only scope no
    ancestor is claimed: every ancestor field is ``None`` and only the current
    fields are populated.  ``classification`` is ``exact`` (every certified
    bound is zero) or ``bounded`` (every bound within tolerance); each bound is
    compared with the full tolerance separately.
    """
    edge_id: int
    s: F
    point: tuple
    ancestor_edge_id: int
    ancestor_model_id: object
    ancestor_revision: int
    ancestor_source_checksum: str
    interval: tuple
    current_frame: NativeArcFrame
    ancestor_frame: NativeArcFrame
    current_bound: F
    ancestor_bound: F | None
    classification: str
    map_classification: str
    scope: str = 'ancestor'


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PreparedNativeArcStationIncidence:
    model_id: object
    revision: int
    source_checksum: str
    ancestry_digest: str
    map_receipt_digest: str | None
    maps: object
    policy: NativeArcStationIncidencePolicy
    tolerance: F
    stations: tuple
    records: tuple
    work_counts: tuple
    preserve_ancestor: bool = True
    # Current-only scope only: the immutable captured actual current Arc
    # definitions pinned at query time, one per selected current edge ID,
    # re-checked against the actual model by the query and the validator.
    # Ancestor-scope receipts leave it empty; their binding is the map receipt.
    current_definitions: tuple = ()


_issued = WeakKeyDictionary()


def _harmonic_bound(proof, frame, parameter, point):
    """Certified outward distance from an exact point to H(parameter)."""
    sine, cosine = proof.sincos(frame.sweep * parameter)
    offsets = []
    for center, u, v, x in zip(frame.center, frame.cosine, frame.sine, point):
        captured = proof.add(proof.add(center, proof.mul(u, cosine)), proof.mul(v, sine))
        offsets.append(proof.sub(proof.i(x), captured))
    return proof.sqrt(proof.add(proof.add(proof.square(offsets[0]),
        proof.square(offsets[1])), proof.square(offsets[2]))).hi


def _station_record(proof, station, row, tolerance):
    proof.charge()
    current_bound = _harmonic_bound(proof, row.current_frame, station.s, station.point)
    ancestor_bound = None
    if row.classification in ('exact', 'bounded'):
        lower, upper = row.interval
        ancestor_bound = _harmonic_bound(proof, row.source_frame,
            lower + (upper-lower)*station.s, station.point)
    bounds = (current_bound,) if ancestor_bound is None else (current_bound, ancestor_bound)
    if current_bound > tolerance:
        raise NativeArcStationIncidenceError(
            'native Arc station distance exceeds the unchanged model positional tolerance')
    if ancestor_bound is not None and ancestor_bound > tolerance:
        raise NativeArcStationIncidenceError(
            'native Arc station ancestor distance exceeds the unchanged model positional tolerance')
    classification = 'exact' if all(bound == 0 for bound in bounds) else 'bounded'
    return NativeArcStationIncidence(station.edge_id, station.s, station.point,
        row.ancestor_edge_id, row.ancestor_model_id, row.ancestor_revision,
        row.ancestor_source_checksum, row.interval, row.current_frame,
        row.source_frame, current_bound, ancestor_bound, classification,
        row.classification)


def _frozen_request(stations):
    """Detached, revalidated copy of the caller request; never the caller objects."""
    return tuple(NativeArcStation(station.edge_id, station.s, station.point,
        station.ancestor_edge_id) for station in stations)


def _verify_original_request(originals, frozen, original_objects):
    """After the final callback the caller's request must be exactly unchanged."""
    if len(originals) != len(frozen):
        raise NativeArcStationIncidenceError('native Arc station request changed during proof')
    for original, selected in zip(originals, original_objects):
        if original is not selected or type(original) is not NativeArcStation:
            raise NativeArcStationIncidenceError('native Arc station request changed during proof')
    try:
        unchanged = _frozen_request(originals) == frozen
    except (GeometryError, TypeError, ValueError, OverflowError) as error:
        raise NativeArcStationIncidenceError(
            'native Arc station request changed during proof') from error
    if not unchanged:
        raise NativeArcStationIncidenceError('native Arc station request changed during proof')


def _capture_current_definitions(model, edge_ids):
    """Live actual current Arc definitions straight from the current model.

    The capture reads the actual current ``model.edges``/``model.vertices``
    through the owner ``_arc_edge_definition`` helper, never ancestry records:
    a newly generated joint Arc without authored ancestry captures exactly
    like an inherited split child.  Runs before the first caller callback and
    returns one immutable definition per edge ID in ascending edge order.
    """
    captured = {}
    for edge_id in edge_ids:
        if edge_id in captured:
            continue
        try:
            definition = _arc_edge_definition(model, edge_id)
        except (GeometryError, TypeError, ValueError) as error:
            raise NativeArcStationIncidenceError(
                'requested native Arc station edge is unavailable') from error
        if definition is None:
            raise NativeArcStationIncidenceError(
                'requested native Arc station edge is not a native Arc')
        captured[edge_id] = definition
    return tuple(captured[edge_id] for edge_id in sorted(captured))


def _recheck_current_definitions(model, pinned, phase):
    """The actual current model definitions must still equal the pinned ones."""
    for definition in pinned:
        try:
            current = _arc_edge_definition(model, definition.edge_id)
        except (GeometryError, TypeError, ValueError) as error:
            raise NativeArcStationIncidenceError(
                'native Arc station current definitions changed ' + phase) from error
        if current is None or current != definition:
            raise NativeArcStationIncidenceError(
                'native Arc station current definitions changed ' + phase)


def _issue(model, result):
    issued = _issued.setdefault(model, {})
    key = id(result)
    issued[key] = (ref(result, lambda _: issued.pop(key, None)), _digest(result))
    return result


def _cancelled(cancellation_check, phase):
    if cancellation_check is not None and cancellation_check(phase):
        raise NativeArcStationIncidenceError('native Arc station incidence cancelled')


def _query_current_only(model, originals, original_objects, frozen, maps, proof, tolerance,
        policy, cancellation_check):
    """Certify supplied XYZ against captured current harmonics only.

    No ancestor is invented and no ancestor-preservation claim is made, so no
    map receipt is required; the binding is the authenticated prepared current
    Arc definition of each named current edge, captured before the first
    callback and re-checked after the final callback. A supplied map receipt
    is guarded historical evidence; its frames never replace the actual
    captured current definition. Its classification, qualified or not,
    does not affect this scope.
    """
    if any(station.ancestor_edge_id is not None for station in frozen):
        raise NativeArcStationIncidenceError(
            'current-only native Arc station cannot select an ancestor')
    original = query_prepared_edge_subcurve_preimages(model)
    ancestry_digest = definition_checksum(original)
    captured = _capture_current_definitions(model, (station.edge_id for station in frozen))
    by_edge = {definition.edge_id: definition for definition in captured}
    map_receipt_digest = None
    map_rows = ()
    if maps is not None:
        try:
            map_receipt_digest = _issued_binding_digest(model, maps)
            if (maps.model_id != model.model_id or maps.revision != model.revision
                    or maps.source_checksum != original.source_checksum
                    or maps.ancestry_digest != ancestry_digest):
                raise NativeArcStationIncidenceError('native Arc station maps are stale')
        except NativeArcParameterMapError as error:
            raise NativeArcStationIncidenceError(str(error)) from error
        map_snapshot = deepcopy(maps)
        map_rows = map_snapshot.records + map_snapshot.alias_records
    selected = []
    for station in frozen:
        if maps is not None:
            rows = tuple(row for row in map_rows if row.edge_id == station.edge_id)
            classes = {row.classification for row in rows}
            selected.append((station, by_edge[station.edge_id], None,
                next(iter(classes)) if len(classes) == 1 else None))
        else:
            selected.append((station, by_edge[station.edge_id], None, None))
    proof.callback = lambda phase: _cancelled(cancellation_check, phase)
    frames = {}
    try:
        proof.cancel('native Arc station incidence')
        records = []
        for station, definition, frame, map_classification in selected:
            if frame is None:
                if definition not in frames:
                    proof.charge()
                    try:
                        frames[definition] = _frame(definition)
                    except (GeometryError, ValueError, TypeError, OverflowError) as error:
                        raise NativeArcStationIncidenceError(
                            'native Arc frame unavailable') from error
                frame = frames[definition]
            proof.charge()
            current_bound = _harmonic_bound(proof, frame, station.s, station.point)
            if current_bound > tolerance:
                raise NativeArcStationIncidenceError(
                    'native Arc station distance exceeds the unchanged model positional tolerance')
            classification = 'exact' if current_bound == 0 else 'bounded'
            records.append(NativeArcStationIncidence(station.edge_id, station.s,
                station.point, None, None, None, None, None, frame, None,
                current_bound, None, classification, map_classification,
                'current_only'))
        proof.cancel('native Arc station incidence final check')
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        raise NativeArcStationIncidenceError(str(error)) from error
    _verify_original_request(originals, frozen, original_objects)
    # Callback-free final guard: nothing below invokes a caller callback.
    current = query_prepared_edge_subcurve_preimages(model)
    if current is not original or definition_checksum(current) != ancestry_digest:
        raise NativeArcStationIncidenceError('ancestry changed during native Arc station proof')
    _recheck_current_definitions(model, captured, 'during proof')
    if maps is not None:
        try:
            if _issued_binding_digest(model, maps) != map_receipt_digest:
                raise NativeArcStationIncidenceError('native Arc station maps changed during proof')
        except NativeArcParameterMapError as error:
            raise NativeArcStationIncidenceError(str(error)) from error
    return _issue(model, PreparedNativeArcStationIncidence(model.model_id,
        model.revision, original.source_checksum, ancestry_digest,
        map_receipt_digest, maps, policy, tolerance, frozen, tuple(records),
        tuple(sorted(proof.counts.items())), preserve_ancestor=False,
        current_definitions=captured))


def query_prepared_native_arc_station_incidence(model, stations, *, maps=None,
        tolerance=None, policy=None, preserve_ancestor=True, cancellation_check=None):
    """Certify finite supplied stations against captured native Arc maps.

    One atomic batch: every station is certified or nothing is issued.  The
    request is frozen into a detached snapshot before the first caller
    callback; only the frozen data is certified and stored, and the caller's
    original objects must be unchanged after the final callback.  With
    ``preserve_ancestor=True`` (default) each station needs one unambiguous
    qualified ancestor map and both current and ancestor bounds are
    certified; with ``preserve_ancestor=False`` only the captured current
    harmonic of the named current edge is certified, without any
    ancestor-preservation claim and without requiring a map receipt.  The
    query is read-only and confers no material, reference, mesh or
    implementation-theorem admission.
    """
    if type(stations) not in (tuple, list):
        raise NativeArcStationIncidenceError('station request must be a finite list or tuple')
    originals = stations
    original_objects = tuple(stations)
    if any(type(station) is not NativeArcStation for station in originals):
        raise NativeArcStationIncidenceError('station request must contain NativeArcStation instances')
    if type(preserve_ancestor) is not bool:
        raise NativeArcStationIncidenceError('invalid ancestor preservation mode')
    if policy is not None and type(policy) is not NativeArcStationIncidencePolicy:
        raise NativeArcStationIncidenceError('invalid proof policy')
    policy = NativeArcStationIncidencePolicy() if policy is None else NativeArcStationIncidencePolicy(**{
        f.name: getattr(policy, f.name) for f in fields(policy)})
    model_tolerance = _finite_rational(float(model.tolerance.length),
        'model positional tolerance must be a positive finite number')
    if model_tolerance <= 0:
        raise NativeArcStationIncidenceError('model positional tolerance must be a positive finite number')
    if tolerance is None:
        tolerance = model_tolerance
    else:
        tolerance = _finite_rational(tolerance,
            'station tolerance must be positive and no looser than the unchanged model positional tolerance')
        if tolerance <= 0 or tolerance > model_tolerance:
            raise NativeArcStationIncidenceError(
                'station tolerance must be positive and no looser than the unchanged model positional tolerance')
    if maps is not None and type(maps) is not PreparedNativeArcParameterMaps:
        raise NativeArcStationIncidenceError('native Arc maps must be a prepared native Arc parameter map receipt')
    # Frozen detached request, built before any caller callback can run (the
    # maps=None map query invokes the caller callback first).  Only this
    # snapshot is certified and stored; the caller's objects are re-compared
    # after the final callback.
    frozen = _frozen_request(originals)
    proof = _Proof(policy, None)
    if not preserve_ancestor:
        return _query_current_only(model, originals, original_objects, frozen, maps, proof,
            tolerance, policy, cancellation_check)
    try:
        if maps is None:
            # The NEW map-query arithmetic shares this one caller policy: the
            # map query receives the remaining allowance and its actual work
            # is charged into this proof.  There is no second default budget.
            remaining = policy.max_interval_operations - proof.counts['interval_operations']
            maps = query_prepared_native_arc_parameter_maps(model,
                policy=NativeArcParameterMapPolicy(max_interval_operations=remaining),
                cancellation_check=cancellation_check)
            for key, count in maps.work_counts:
                proof.counts[key] += count
            for key in ('interval_operations', 'series_terms'):
                if proof.counts[key] > policy.max_interval_operations:
                    raise NativeArcStationIncidenceError('qualification_budget_exhausted:' + key)
        map_receipt_digest = _issued_binding_digest(model, maps)
        original = query_prepared_edge_subcurve_preimages(model)
        ancestry_digest = definition_checksum(original)
        if (maps.model_id != model.model_id or maps.revision != model.revision
                or maps.source_checksum != original.source_checksum
                or maps.ancestry_digest != ancestry_digest):
            raise NativeArcStationIncidenceError('native Arc station maps are stale')
    except NativeArcParameterMapError as error:
        raise NativeArcStationIncidenceError(str(error)) from error
    snapshot = deepcopy(maps)
    candidates = {}
    for row in snapshot.records + snapshot.alias_records:
        candidates.setdefault(row.edge_id, []).append(row)
    selected = []
    for station in frozen:
        rows = candidates.get(station.edge_id, ())
        if station.ancestor_edge_id is not None:
            rows = tuple(row for row in rows
                if row.ancestor_edge_id == station.ancestor_edge_id)
        if not rows:
            raise NativeArcStationIncidenceError('requested native Arc ancestry unavailable')
        if len(rows) > 1:
            raise NativeArcStationIncidenceError('ambiguous native Arc ancestry for station edge')
        if rows[0].classification not in ('exact', 'bounded'):
            raise NativeArcStationIncidenceError('unqualified native Arc parameter map for station edge')
        selected.append((station, rows[0]))
    revision, source_checksum = model.revision, original.source_checksum
    proof.callback = lambda phase: _cancelled(cancellation_check, phase)
    try:
        proof.cancel('native Arc station incidence')
        records = tuple(_station_record(proof, station, row, tolerance)
            for station, row in selected)
        proof.cancel('native Arc station incidence final check')
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        raise NativeArcStationIncidenceError(str(error)) from error
    _verify_original_request(originals, frozen, original_objects)
    # Callback-free final guard: nothing below invokes a caller callback.
    current = query_prepared_edge_subcurve_preimages(model)
    if (current is not original or definition_checksum(current) != ancestry_digest
            or model.model_id != maps.model_id or model.revision != revision
            or model.revision != maps.revision):
        raise NativeArcStationIncidenceError('ancestry changed during native Arc station proof')
    try:
        if _issued_binding_digest(model, maps) != map_receipt_digest:
            raise NativeArcStationIncidenceError('native Arc station maps changed during proof')
    except NativeArcParameterMapError as error:
        raise NativeArcStationIncidenceError(str(error)) from error
    return _issue(model, PreparedNativeArcStationIncidence(model.model_id, revision,
        source_checksum, ancestry_digest, map_receipt_digest, maps, policy,
        tolerance, frozen, records, tuple(sorted(proof.counts.items())),
        preserve_ancestor=True))


def validate_prepared_native_arc_station_incidence_binding(model, binding, *,
        cancellation_check=None):
    """Validate a producer-issued station receipt without re-proving or accepting forgery."""
    issued = _issued.get(model, {}).get(id(binding))
    if issued is None or issued[0]() is not binding:
        raise NativeArcStationIncidenceError('native Arc station receipt was not issued to this owner')
    pinned = issued[1]
    if _digest(binding) != pinned:
        raise NativeArcStationIncidenceError('native Arc station receipt changed')
    snapshot = deepcopy(binding)
    if snapshot.maps is not None:
        try:
            validate_prepared_native_arc_parameter_maps_binding(model, binding.maps)
        except NativeArcParameterMapError as error:
            raise NativeArcStationIncidenceError(str(error)) from error
    original = query_prepared_edge_subcurve_preimages(model)
    ancestry = definition_checksum(original)
    if (snapshot.model_id != model.model_id or snapshot.revision != model.revision
            or snapshot.source_checksum != original.source_checksum
            or snapshot.ancestry_digest != ancestry):
        raise NativeArcStationIncidenceError('native Arc station receipt is stale')
    # Current-only receipts additionally pin the actual captured current Arc
    # definitions before the validator callback and re-check them after it.
    captured = None
    if not snapshot.preserve_ancestor:
        captured = snapshot.current_definitions
        _recheck_current_definitions(model, captured, 'before validation')
    if cancellation_check is not None:
        if cancellation_check('native Arc station incidence binding'):
            raise NativeArcStationIncidenceError('native Arc station incidence cancelled')
    # Callback-free final guard after the validator callback.
    current = query_prepared_edge_subcurve_preimages(model)
    if (current is not original or definition_checksum(current) != ancestry
            or _digest(binding) != pinned):
        raise NativeArcStationIncidenceError('native Arc station binding changed during validation')
    if captured is not None:
        _recheck_current_definitions(model, captured, 'during validation')
