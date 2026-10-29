"""Live certificates for nominal binary64 lookup aliases of exact source stations.

Exact ancestry and literal world targets are retained. These receipts confer no
material coverage, approximation, mesh node identity or S3 authority.
"""
from dataclasses import dataclass, fields
from fractions import Fraction as Q
from weakref import WeakKeyDictionary, ref
import math

from .errors import GeometryError
from .prepared_material_curvilinear import _literal
from .prepared_material_source_transfer import (_source_station_transport_inputs, _state,
                                               _signature as _transfer_signature)
from .definition_binding import definition_checksum


@dataclass(frozen=True)
class SourceStationTransportRow:
    prepared_edge: object
    prepared_parameter: Q
    prepared_station_key: tuple
    source_edge: object
    source_parameter: Q
    nominal_parameter: float
    source_station_identity: tuple
    xyz: tuple
    combined_error_squared_bound: Q
    combined_error_bound: float
    limits: tuple


@dataclass(frozen=True, eq=False)
class PreparedSourceStationTransport:
    source_model_id: object
    prepared_model_id: object
    source_revision: int
    prepared_revision: int
    source_checksum: str
    prepared_checksum: str
    rows: tuple


_ISSUED = WeakKeyDictionary()


def _signature(result):
    if type(result.rows) is not tuple or any(type(row) is not SourceStationTransportRow for row in result.rows):
        raise GeometryError('source station transport rows are malformed')
    try:
        return tuple((f.name, (tuple(tuple((rf.name,_literal(getattr(row,rf.name))) for rf in fields(SourceStationTransportRow))
                                    for row in result.rows) if f.name == 'rows' else _literal(getattr(result,f.name))))
                     for f in fields(result))
    except (AttributeError,TypeError,ValueError,OverflowError) as error:
        raise GeometryError('source station transport literal result is malformed') from error


def _context(source, prepared, transfer):
    receipts, operands = _source_station_transport_inputs(source, prepared, transfer)
    return ((_state(source), _state(prepared)), receipts,
            _transfer_signature(transfer), tuple((_literal(cell), definition_checksum(regions), _literal(chords))
                                      for cell, regions, chords in operands)), operands


def _frontier(source, prepared, transfer, callback):
    context, operands = _context(source, prepared, transfer)
    from .cancellation import is_cancellation_signal
    passive = is_cancellation_signal(callback)
    def check(phase):
        cancelled = callback(phase) if callback is not None else False
        if (callback is not None and not passive) or phase.endswith('complete'):
            current, _ = _context(source, prepared, transfer)
            if current != context:
                raise GeometryError('source station transport operands changed during callback')
        if cancelled:
            raise GeometryError('source station transport cancelled')
    return context, operands, check


def _collision_check(stations):
    """All exact/binary stations on each affected edge participate, including endpoints."""
    aliases = {}
    for source_edge, exact_parameter in stations:
        s = Q(exact_parameter)
        if not 0 <= s <= 1:
            raise GeometryError('source station parameter is outside its exact source edge')
        r = float(s)
        if not math.isfinite(r) or (0 < s < 1 and r in (0., 1.)):
            raise GeometryError('interior source station rounds to a source endpoint')
        key = source_edge, r
        old = aliases.setdefault(key, s)
        if old != s:
            raise GeometryError('distinct exact source stations have a nominal alias collision')


def _world_enclosure(model, edge_id, lo, hi):
    """Registered true world maps only; affine routes retain exact endpoints."""
    from .curves import Straight, LineRange
    from .nurbs import NurbsCurve, CurveOnSurface, _float_down, _float_up
    from .support_curves import PlaneNurbsCurve
    from .plane_nurbs import charge_box
    edge = model.edges[edge_id]
    curve = edge.curve
    if type(curve) not in (Straight, LineRange, NurbsCurve, CurveOnSurface, PlaneNurbsCurve):
        raise GeometryError('source station curve has no registered certified world enclosure')
    charge_box()
    if type(curve) is Straight:
        a, b = (tuple(Q(float(x)) for x in model.vertex_position(v)) for v in (edge.start, edge.end))
        return 'affine', tuple(tuple(x+t*(y-x) for x,y in zip(a,b)) for t in (lo,hi))
    if type(curve) is LineRange:
        return 'affine', (curve.exact_point(lo), curve.exact_point(hi))
    low, high = curve.bounds(max(0., _float_down(lo)), min(1., _float_up(hi)))
    if (len(low) != 3 or len(high) != 3 or any(not math.isfinite(float(x)) for p in (low,high) for x in p)
            or any(x>y for x,y in zip(low,high))):
        raise GeometryError('source station kernel returned an invalid certified world enclosure')
    return 'box', (tuple(Q(float(x)) for x in low), tuple(Q(float(x)) for x in high))


def _combined_bound(source, prepared, source_edge, prepared_edge, s, r, xyz, chord_error):
    kind, enclosure = _world_enclosure(source, source_edge, min(s,Q(r)), max(s,Q(r)))
    target = tuple(Q(float(x)) for x in xyz)
    if kind == 'affine':
        bound = max(sum((x-y)**2 for x,y in zip(target,p)) for p in enclosure)
    else:
        low, high = enclosure
        bound = sum(max((x-a)**2,(x-b)**2) for x,a,b in zip(target,low,high))
    limits = []
    for label, model, edge in (('source',source,source_edge), ('prepared',prepared,prepared_edge)):
        # Existing source edge transport uses its independent joint context at
        # extent 1, even when an authored carrier's whole-edge extent is huge.
        for name in ('length','coincidence'):
            limit = getattr(model.tolerance,'effective_'+name)(1.)
            if bound > Q(limit)**2:
                raise GeometryError('combined source station alias discrepancy exceeds existing joint '+name+' tolerance')
            limits.append((label,'joint_'+name+'_at_1',limit))
        family, points = _world_enclosure(model, edge, Q(0), Q(1))
        if family == 'affine':
            extent = math.hypot(*(float(a-b) for a,b in zip(*points)))
        else:
            extent = math.hypot(*(float(b-a) for a,b in zip(*points)))
        for name in ('length', 'coincidence', 'curve_fit_residual', 'surface_residual'):
            limit = getattr(model.tolerance, 'effective_'+name)(extent)
            if bound > Q(limit)**2:
                raise GeometryError('combined source station alias discrepancy exceeds existing '+name+' tolerance')
            limits.append((label, name, limit))
    # Existing native support cap is independent of every model-owned limit.
    if bound > Q(1.e-8)**2:
        raise GeometryError('combined source station alias discrepancy exceeds native residual')
    limits.append(('native', 'residual', 1.e-8))
    if bound > Q(chord_error)**2:
        raise GeometryError('combined source station alias discrepancy exceeds registered chord error')
    limits.append(('registered_chord', 'error', chord_error))
    if bound:
        exponent = (bound.numerator.bit_length()-bound.denominator.bit_length())//2
        scaled = bound/(Q(2)**(2*exponent))
        error = math.ldexp(math.sqrt(float(scaled)), exponent)
    else:
        error = 0.
    if Q(error)**2 < bound:
        error = math.nextafter(error, math.inf)
    return bound, error, tuple(limits)


def query_prepared_source_station_transport(source, prepared, transfer, *, cancellation_check=None):
    context, operands, check = _frontier(source, prepared, transfer, cancellation_check)
    check('source station transport qualification')
    maps = {edge.id: (edge, parent, Q(a), Q(b)) for edge,parent,a,b in transfer.edge_parameter_maps}
    declared = {(edge.id,Q(t)): (parent,s) for edge,t,parent,s,_ in transfer.boundary_station_correspondence
                if parent is not None}
    stations, chord_limits = {}, {}
    for _cell, _regions, chords in operands:
        for chord in chords.intervals:
            if chord.edge.id not in maps:
                continue  # virtual traces have no source-edge registry station
            edge, parent, a, b = maps[chord.edge.id]
            for t, key, xyz in zip(chord.parameter_range, chord.station_keys, chord.xyz):
                check('source station transport station')
                t = Q(t)
                s = a+(b-a)*t
                if (edge.id,t) in declared and declared[(edge.id,t)] != (parent,s):
                    raise GeometryError('source station transport exact ancestry changed')
                value = (edge,t,key,parent,s,tuple(xyz))
                old = stations.setdefault((edge.id,t),value)
                if old != value:
                    raise GeometryError('source station has conflicting literal owner keys or XYZ')
                station = edge.id,t
                chord_limits[station] = min(chord_limits.get(station,chords.chord_error),chords.chord_error)
    affected = {row[3].id for row in stations.values() if Q(float(row[4])) != row[4]}
    selected = tuple(row for _,row in sorted(stations.items()) if row[3].id in affected)
    collisions = [(row[3].id,row[4]) for row in selected]
    collisions += [(edge,t) for edge in affected for t in (Q(0),Q(1))]
    _collision_check(collisions)
    rows, exact_targets = [], {}
    for edge,t,key,parent,s,xyz in selected:
        check('source station transport bound')
        if len(xyz) != 3 or not all(type(x) is float and math.isfinite(x) for x in xyz):
            raise GeometryError('source station target must be literal finite owner XYZ')
        identity = ('source_edge_station',source.model_id,parent.id,s)
        old = exact_targets.setdefault(identity,xyz)
        if old != xyz:
            raise GeometryError('duplicate exact source station has conflicting literal owner XYZ')
        nominal = float(s)
        bound,error,limits = _combined_bound(source,prepared,parent.id,edge.id,s,nominal,xyz,chord_limits[(edge.id,t)])
        rows.append(SourceStationTransportRow(edge,t,key,parent,s,nominal,identity,xyz,bound,error,limits))
    check('source station transport qualification complete')
    states = context[0]
    result = PreparedSourceStationTransport(source.model_id,prepared.model_id,states[0][0],states[1][0],
        states[0][1],states[1][1],tuple(rows))
    _ISSUED[result] = (ref(source),ref(prepared),ref(transfer),context,_signature(result))
    return result


def validate_prepared_source_station_transport_binding(source, prepared, transfer, result, *, cancellation_check=None):
    if type(result) is not PreparedSourceStationTransport or result not in _ISSUED:
        raise GeometryError('source station transport result was not issued by this owner')
    first,second,issued_transfer,context,signature = _ISSUED[result]
    if first() is not source or second() is not prepared or issued_transfer() is not transfer:
        raise GeometryError('source station transport belongs to other live operands')
    if _signature(result) != signature:
        raise GeometryError('source station transport literal result changed')
    current,_,check = _frontier(source,prepared,transfer,cancellation_check)
    if current != context:
        raise GeometryError('source station transport context is stale or changed')
    for phase in ('source station transport binding','source station transport binding complete'):
        check(phase)
        if _signature(result) != signature:
            raise GeometryError('source station transport result changed during callback')
