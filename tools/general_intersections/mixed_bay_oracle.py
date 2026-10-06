"""Independent first-bay expectations; deliberately NOT a completeness gate.

This development tool does not call intersection planning/discovery or sample a
display curve. It derives polynomial support-event supersets from the authored
definitions and checks nine independently expected source intervals against
prepared ownership/ancestry. The conic/BQC trim, branch and event correspondence
proof is missing: even a successful interval comparison returns PARTIAL.

Cylinder equations use the exact rational inverse of the ACTUAL stored basis
[radius*radial, radius*stored_circumferential, height*axis]. They do not assume
orthonormal binary64 vectors and do not substitute the ideal direction cylinder.
Angular support windows and rounded Straight/Arc trims are retained as an
explicit obstruction, never replaced by nominal angles or endpoint polygons.

The Sturm isolator is shared arithmetic, not the production intersection/event
classifier. Brackets remain brackets, including coincident/unresolved events;
there is no float-root merge, branch midpoint acceptance, or new tolerance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
import json
from math import comb
from pathlib import Path

from anygeometry import (
    Cylinder, GeometryError, query_prepared_edge_subcurve_preimages,
    query_prepared_face_preimages, to_dict,
    validate_prepared_edge_subcurve_preimages_binding,
    validate_prepared_face_preimages_binding,
)
from anygeometry.analytic_roots import isolate_real_roots


BUILDER_SHA256 = 'b14055460e36ef5c16e86357f7206d95b67a928fa2cd059e6a92810f243c1434'
WALL_CONTROLS = ((F(0), F(0), F(0)), (F(1), F(2), F(0)),
                 (F(2), F(-1), F(0)), (F(3), F(1), F(0)))
WALL_VECTOR = (F(1, 4), F(0), F(3, 2))
PENDING = (
    'floor/cylinder: angular and literal-trim clipping and prepared conic correspondence',
    'wall/cylinder: clipped branch multiplicity, folds and prepared BQC correspondence',
    'all families: canonical event-vertex ownership and exclusion of spurious joints',
    'material: nonoverlap and full original/current set coverage, beyond scalar area sums',
)


class OracleRefusal(GeometryError):
    """Unsupported input or unavailable proof; never an acceptance."""


class OracleMismatch(GeometryError):
    """An independently required interval has missing/wrong/duplicate coverage."""


def _check(callback):
    if callback is not None and callback():
        raise OracleRefusal('mixed-bay oracle cancelled')


def _poly(values):
    values = list(map(F, values)) or [F(0)]
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _add(a, b):
    return _poly((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
                 for i in range(max(len(a), len(b))))


def _scale(a, s):
    return _poly(x*s for x in a)


def _mul(a, b):
    out = [F(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return _poly(out)


def _value(p, t):
    result = F(0)
    for x in reversed(p):
        result = result*t+x
    return result


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return tuple(a[(i+1) % 3]*b[(i+2) % 3]-a[(i+2) % 3]*b[(i+1) % 3]
                 for i in range(3))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _inverse_columns(columns):
    a, b, c = columns
    determinant = _dot(a, _cross(b, c))
    if determinant == 0:
        raise OracleRefusal('singular actual cylinder basis')
    return tuple(tuple(x/determinant for x in row)
                 for row in (_cross(b, c), _cross(c, a), _cross(a, b)))


def _bernstein_power(controls):
    """Independent coefficient conversion; ascending powers, three axes."""
    degree = len(controls)-1
    rows = []
    for axis in range(3):
        out = [F(0)]*(degree+1)
        for i, point in enumerate(controls):
            for j in range(degree-i+1):
                out[i+j] += point[axis]*comb(degree, i)*comb(degree-i, j)*(-1)**j
        rows.append(_poly(out))
    return tuple(rows)


@dataclass(frozen=True)
class Carrier:
    face_id: int
    origin: tuple
    columns: tuple
    inverse_rows: tuple
    start_angle: F
    sweep_angle: F


@dataclass(frozen=True)
class EventPolynomial:
    family: str
    owner_ids: tuple
    kind: str
    polynomial: tuple
    domain: tuple
    # Each bracket contains one distinct real root of this polynomial. A
    # bracket shared/overlapping another event is NOT an identified common root.
    roots: tuple
    coincidence_unresolved: bool


@dataclass(frozen=True)
class CarrierSection:
    family: str
    owners: tuple
    variables: tuple
    first_range: tuple
    second_range: tuple
    # A(t) s^2 + b(t) s + c(t) = 0; axial=q(t)+k*s in [0,1].
    coefficients: tuple
    axial: tuple
    events: tuple
    qualified_material_intervals: None = None


@dataclass(frozen=True)
class RequiredInterval:
    family: str
    source_edge_id: int
    owners: tuple
    controls: tuple
    start_vertex: int
    end_vertex: int
    interval: tuple = (F(0), F(1))


@dataclass(frozen=True)
class MixedBayInventory:
    source_model_id: str
    source_revision: int
    source_checksum: str
    source_document_json: str
    raw_carriers: tuple
    builder_sha256: str
    required_intervals: tuple
    carrier_sections: tuple
    literal_seam_events: tuple
    pending: tuple = PENDING
    whole_joint_completeness: bool = False


@dataclass(frozen=True)
class IntervalObservation:
    source_edge_id: int
    prepared_edge_id: int
    owners: tuple
    interval: tuple
    # Explicitly inherited owner polynomial rounding enclosure, not exact
    # equality of current geometric curves to original restrictions.
    squared_world_error_bound: F


@dataclass(frozen=True)
class PartialComparison:
    status: str
    covered_source_edges: tuple
    observations: tuple
    pending: tuple = PENDING
    whole_joint_completeness: bool = False
    material_coverage: bool = False

    def require_complete(self):
        raise OracleRefusal('partial oracle cannot qualify joint/material completeness: '
                            + '; '.join(self.pending))


def _carrier_snapshot(model):
    out = []
    for face_id, face in sorted(model.faces.items()):
        surface = face.surface
        if type(surface) is not Cylinder:
            continue
        origin = tuple(F(float(x)) for x in surface.origin)
        axes = (surface.radial_direction, surface.circumferential_direction, surface.axis)
        scales = (F(surface.radius), F(surface.radius), F(surface.height))
        columns = tuple(tuple(F(float(x))*s for x in axis) for axis, s in zip(axes, scales))
        out.append(Carrier(face_id, origin, columns, _inverse_columns(columns),
                           F(surface.start_angle), F(surface.sweep_angle)))
    return tuple(out)


def _json(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _coordinates(carrier, base, direction):
    """Coordinates of P(t,s)=base(t)+s*direction in the exact raw basis."""
    shifted = tuple(_add(base[i], (-carrier.origin[i],)) for i in range(3))
    offsets = tuple(_add(_add(_scale(shifted[0], row[0]), _scale(shifted[1], row[1])),
                          _scale(shifted[2], row[2])) for row in carrier.inverse_rows)
    slopes = tuple(_dot(row, direction) for row in carrier.inverse_rows)
    return offsets, slopes


def _section_coefficients(carrier, base, direction):
    q, k = _coordinates(carrier, base, direction)
    a = (k[0]*k[0]+k[1]*k[1],)
    b = _scale(_add(_scale(q[0], k[0]), _scale(q[1], k[1])), 2)
    c = _add(_add(_mul(q[0], q[0]), _mul(q[1], q[1])), (-F(1),))
    return (a, b, c), (q[2], k[2])


def _linear_boundary_resultant(coefficients, offset, slope):
    """Res_s(A*s²+b*s+c, offset+slope*s); conservative at slope=0.

    A*offset²-b*offset*slope+c*slope² vanishes at every shared root.
    No branch selection or removal of extraneous roots is performed.
    """
    a, b, c = coefficients
    return _add(_add(_mul(a, _mul(offset, offset)),
                     _scale(_mul(b, offset), -slope)), _scale(c, slope*slope))


def _event(family, owners, kind, polynomial, domain, callback):
    polynomial = _poly(polynomial)
    _check(callback)
    if polynomial == (0,):
        return EventPolynomial(family, owners, kind, polynomial, domain, (), True)
    # This arithmetic uses exact coefficients and preserves certified brackets.
    # The isolator's default width is its existing numeric work target, not a
    # geometric acceptance tolerance. Root cells are never skipped/merged.
    roots = isolate_real_roots(polynomial, interval=domain, cancellation_check=callback)
    return EventPolynomial(family, owners, kind, polynomial, domain,
                           tuple((r.lower, r.upper) for r in roots), False)


def _section(carrier, family, owners, base, direction, variables, first_range,
             second_range, callback):
    coefficients, axial = _section_coefficients(carrier, base, direction)
    a, b, c = coefficients
    candidates = [('discriminant', _add(_mul(b, b), _scale(_mul(a, c), -4))),
                  ('quadratic-leading-coefficient', a)]
    for end in second_range:
        candidates.append((f'{variables[1]}={end}',
                           _add(_add(_scale(a, end*end), _scale(b, end)), c)))
    for end in (F(0), F(1)):
        candidates.append((f'axial={end}', _linear_boundary_resultant(
            coefficients, _add(axial[0], (-end,)), axial[1])))
    events = tuple(_event(family, owners, kind, p, first_range, callback)
                   for kind, p in candidates)
    return CarrierSection(family, owners, variables, first_range, second_range,
                          coefficients, axial, events)


def _controls(edge, vertices):
    kind = edge['curve']['type']
    if kind not in ('straight', 'spline'):
        raise OracleRefusal('required source interval is not a polynomial edge')
    ids = (edge['start'], *edge['curve'].get('control_vertices', ()), edge['end'])
    return tuple(tuple(F(float(x)) for x in vertices[i]['position']) for i in ids)


def _authored_expectations(document, carriers, wall):
    """Re-derive nine requirements from source definitions, without discovery."""
    if (len(document['faces']) != 10 or len(carriers) != 8
            or document['structural']['members']):
        raise OracleRefusal('only the complete bare first mixed bay is supported')
    panels = tuple(c.face_id for c in carriers)
    floor_ids = {f['id'] for f in document['faces']} - set(panels) - {wall}
    if len(floor_ids) != 1:
        raise OracleRefusal('ambiguous first-bay floor/wall ownership')
    floor, = floor_ids
    faces = {f['id']: f for f in document['faces']}
    vertices = {v['id']: v for v in document['vertices']}
    edges = {e['id']: e for e in document['edges']}
    if any(f['holes'] or f['parameterization'] is not None for f in document['faces']):
        raise OracleRefusal('first-bay oracle does not admit holes or explicit parameterizations')
    floor_points = {tuple(F(float(x)) for x in vertices[edges[e]['start']]['position'])
                    for e, _ in faces[floor]['loop']}
    if floor_points != {(F(x), F(y), F(0)) for x in (-3, 7) for y in (-2, 4)}:
        raise OracleRefusal('not the untranslated first-bay floor')
    if (faces[floor]['surface'] != {'type': 'plane', 'origin': [-3., -2., 0.],
            'u_vector': [10., 0., 0.], 'v_vector': [0., 6., 0.]}
            or len(faces[floor]['loop']) != 4
            or any(edges[e]['curve']['type'] != 'straight' for e, _ in faces[floor]['loop'])):
        raise OracleRefusal('floor support/trim differs from the authored rectangle')
    wall_loop = faces[wall]['loop']
    if len(wall_loop) != 4:
        raise OracleRefusal('wall must retain its authored four boundary definitions')
    # Verify the actual original Coons boundary polynomial reduces to B(u)+vD;
    # no ideal wall is substituted for a rounded/different authored boundary.
    oriented = []
    for edge, forward in wall_loop:
        pts = _controls(edges[edge], vertices)
        oriented.append(pts if forward else pts[::-1])
    top = tuple(tuple(x+d for x, d in zip(p, WALL_VECTOR)) for p in WALL_CONTROLS)
    if (oriented != [WALL_CONTROLS, (WALL_CONTROLS[-1], top[-1]),
                     top[::-1], (top[0], WALL_CONTROLS[0])]
            or faces[wall]['surface'] != {'type': 'coons'}
            or faces[wall]['corners'] != [0, 1, 2, 3]):
        raise OracleRefusal('actual authored wall does not equal the first-bay Coons extrusion')
    if any(c.origin != tuple(map(F, (-2., .4, .6)))
           or F(faces[c.face_id]['surface']['radius']) != F(.7)
           or F(faces[c.face_id]['surface']['height']) != 8 for c in carriers):
        raise OracleRefusal('cylinder source origin/radius/height differs from first bay')
    # All support coefficients and windows are retained. Their exact common
    # frame is checked; no assumption that rounded vectors are orthonormal.
    if any((c.origin, c.columns) != (carriers[0].origin, carriers[0].columns) for c in carriers):
        raise OracleRefusal('pipe panels do not share one actual raw carrier')
    expected = []
    bottom = wall_loop[0][0]
    def required(family, edge_id, owners):
        edge = edges[edge_id]
        return RequiredInterval(family, edge_id, tuple(sorted(owners)),
                                _controls(edge, vertices), edge['start'], edge['end'])
    expected.append(required('floor/bottom-cubic', bottom, (floor, wall)))
    panel_incidence = {}
    for panel in panels:
        for edge, _ in faces[panel]['loop']:
            panel_incidence.setdefault(edge, []).append(panel)
    seams = []
    for edge, owners in sorted(panel_incidence.items()):
        if len(owners) == 2:
            if edges[edge]['curve']['type'] != 'straight' or owners[0] == owners[1]:
                raise OracleRefusal('unexpected shared pipe-panel boundary')
            row = required('authored-pipe-seam', edge, owners)
            expected.append(row)
            seams.append(row)
        elif len(owners) != 1:
            raise OracleRefusal('nonmanifold authored panel incidence')
    if len(seams) != 8 or any(sum(p in row.owners for row in seams) != 2 for p in panels):
        raise OracleRefusal('expected eight authored axial seam intervals are unavailable')
    reached = {panels[0]}
    for _ in panels:
        reached.update(owner for seam in seams if reached.intersection(seam.owners)
                       for owner in seam.owners)
    if reached != set(panels):
        raise OracleRefusal('authored pipe adjacency is not one connected cycle')
    return floor, tuple(expected), tuple(seams)


def build_authored_inventory(fixture, *, cancellation_check=None):
    """Derive support-event supersets and nine source interval expectations.

    No preparation/discovered-joint result constructs expectations. This is not
    clipped material branches, event multiplicity, or a completeness receipt.
    """
    builder = Path(__file__).with_name('large_connected_fixtures.py')
    if sha256(builder.read_bytes()).hexdigest() != BUILDER_SHA256:
        raise OracleRefusal('independent oracle requires the frozen b1405546 builder')
    model = fixture.model
    identity = (str(model.model_id), model.revision)
    document = to_dict(model)
    serialized = _json(document)
    carriers = _carrier_snapshot(model)
    wall_specs = deepcopy(fixture.curve_joints)
    selected = tuple((h.kind, h.id) for h in fixture.operands)
    if (fixture.name != 'connected-mixed-10' or len(selected) != 10
            or set(selected) != {('face', f['id']) for f in document['faces']}
            or len(wall_specs) != 1):
        raise OracleRefusal('only the complete bare first mixed bay is supported')
    wall = wall_specs[0]['wall'][1]
    floor, expected, seams = _authored_expectations(document, carriers, wall)
    _check(cancellation_check)
    base = _bernstein_power(WALL_CONTROLS)
    sections = []
    for carrier in carriers:
        sections.append(_section(carrier, 'wall/cylinder-support', (wall, carrier.face_id),
            base, WALL_VECTOR, ('u', 'v'), (F(0), F(1)), (F(0), F(1)), cancellation_check))
        sections.append(_section(carrier, 'floor/cylinder-support', (floor, carrier.face_id),
            ((F(0), F(1)), (F(0),), (F(0),)), (F(0), F(1), F(0)),
            ('x', 'y'), (F(-3), F(7)), (F(-2), F(4)), cancellation_check))
    seam_events = []
    for seam in seams:
        p, end = seam.controls
        direction = _sub(end, p)
        # Necessary coplanarity for the wall generator B(u)+vD and the ACTUAL
        # seam line p+wE. Extraneous/parallel roots are intentionally retained.
        normal = _cross(WALL_VECTOR, direction)
        polynomial = (F(0),)
        for axis in range(3):
            polynomial = _add(polynomial, _scale(_add(base[axis], (-p[axis],)), normal[axis]))
        seam_events.append(_event('wall/authored-seam', (wall, *seam.owners),
            f'edge-{seam.source_edge_id}-line-coplanarity', polynomial, (F(0), F(1)), cancellation_check))
        seam_events.append(_event('floor/authored-seam', (floor, *seam.owners),
            f'edge-{seam.source_edge_id}-z=0', (p[2], direction[2]), (F(0), F(1)), cancellation_check))
    _check(cancellation_check)
    if (_json(to_dict(model)) != serialized or _carrier_snapshot(model) != carriers
            or (str(model.model_id), model.revision) != identity):
        raise OracleRefusal('authored source changed during independent inventory')
    return MixedBayInventory(*identity, document['checksum']['value'],
        serialized, carriers, BUILDER_SHA256, tuple(expected), tuple(sections), tuple(seam_events))


def compare_interval_observations(required, observations):
    """Pure exact interval/ownership check, independent of discovery counts.

    This helper expects authenticated observations. Only compare_prepared uses
    it as an owner-bound check. Bounds describe owner-certified rounding, not
    literal equality or conic/BQC completeness.
    """
    required, observations = tuple(required), tuple(observations)
    if not required:
        raise OracleRefusal('empty expectation set cannot establish interval coverage')
    by_source = {row.source_edge_id: row for row in required}
    if len(by_source) != len(required):
        raise OracleRefusal('duplicate expected source interval')
    grouped = {source: [] for source in by_source}
    for row in observations:
        expected = by_source.get(row.source_edge_id)
        if expected is None or tuple(sorted(row.owners)) != expected.owners:
            raise OracleMismatch('wrong source/owner association')
        if (type(row.prepared_edge_id) is not int or row.prepared_edge_id <= 0
                or len(row.interval) != 2 or any(type(x) is not F for x in row.interval)
                or type(row.squared_world_error_bound) is not F or row.squared_world_error_bound < 0):
            raise OracleRefusal('malformed authenticated interval observation')
        a, b = sorted(row.interval)
        if not expected.interval[0] <= a < b <= expected.interval[1]:
            raise OracleMismatch('empty/out-of-range source interval')
        grouped[row.source_edge_id].append((a, b, row.prepared_edge_id))
    for source, expected in by_source.items():
        rows = sorted(grouped[source])
        if len({edge for _, _, edge in rows}) != len(rows):
            raise OracleMismatch(f'duplicate prepared edge for source {source}')
        cursor = expected.interval[0]
        for a, b, _ in rows:
            if a != cursor:
                raise OracleMismatch(f'gap/overlap in required source {source}')
            cursor = b
        if cursor != expected.interval[1]:
            raise OracleMismatch(f'missing required source interval {source}')
    return PartialComparison('partial', tuple(sorted(by_source)), observations)


def compare_prepared(inventory, authored, prepared, *, cancellation_check=None):
    """Compare all nine expected source intervals with prepared common owners.

    Independent expectations precede discovery. Every matching prepared edge
    must carry a sealed polynomial source interval from this authored document.
    Missing ancestry is an explicit refusal, never coordinate inversion.
    """
    if type(inventory) is not MixedBayInventory or inventory.builder_sha256 != BUILDER_SHA256:
        raise OracleRefusal('bound first-bay inventory required')
    pinned = deepcopy(inventory)
    if (_json(to_dict(authored)) != pinned.source_document_json
            or _carrier_snapshot(authored) != pinned.raw_carriers
            or str(authored.model_id) != pinned.source_model_id
            or str(prepared.model_id) != pinned.source_model_id):
        raise OracleRefusal('wrong/stale authored or prepared model')
    source_doc = json.loads(pinned.source_document_json)
    wall_ids = [f['id'] for f in source_doc['faces'] if f['surface'] == {'type': 'coons'}]
    if len(wall_ids) != 1:
        raise OracleRefusal('ambiguous original wall definition')
    _, expected, _ = _authored_expectations(source_doc, pinned.raw_carriers, wall_ids[0])
    if (pinned.required_intervals != expected
            or pinned.source_revision != authored.revision
            or pinned.source_checksum != source_doc['checksum']['value']):
        raise OracleRefusal('inventory requirements/source binding were altered')
    # Callback-free entry acquisition and detachment prevent registry edits
    # restored later from contaminating observations.
    binding = deepcopy(query_prepared_edge_subcurve_preimages(prepared))
    face_binding = deepcopy(query_prepared_face_preimages(prepared))
    if (str(face_binding.authored_model_id) != pinned.source_model_id
            or face_binding.authored_revision != pinned.source_revision
            or face_binding.authored_checksum != pinned.source_checksum):
        raise OracleRefusal('prepared face ancestry belongs to another original source')
    face_descendants = dict(face_binding.face_descendants)
    prepared_json = _json(to_dict(prepared))
    doc = json.loads(prepared_json)
    prepared_faces = {row['id']: row for row in doc['faces']}
    owners = {}
    for source in {owner for row in pinned.required_intervals for owner in row.owners}:
        # Replacement history is a mutable convenience index, not authenticated
        # source ownership. Only the owner-sealed application-delta map counts.
        descendants = face_descendants.get(source, ())
        if not descendants:
            raise OracleRefusal('source face has no active descendants')
        owners[source] = {edge for child in descendants
                          for loop in (prepared_faces[child]['loop'], *prepared_faces[child]['holes'])
                          for edge, _ in loop}
    observations = []
    for expected in pinned.required_intervals:
        _check(cancellation_check)
        common = set.intersection(*(owners[owner] for owner in expected.owners))
        found = {}
        for row in (*binding.records, *binding.alias_records):
            ancestor = row.ancestor
            if (row.edge_id not in common or ancestor.definition.edge_id != expected.source_edge_id
                    or str(ancestor.model_id) != pinned.source_model_id
                    or ancestor.revision != pinned.source_revision
                    or ancestor.source_checksum != pinned.source_checksum):
                continue
            controls = tuple(tuple(F(*x) for x in point) for point in ancestor.definition.controls)
            if (controls != expected.controls or ancestor.definition.start != expected.start_vertex
                    or ancestor.definition.end != expected.end_vertex):
                raise OracleMismatch('sealed ancestor differs from independently expected source')
            observation = IntervalObservation(expected.source_edge_id, row.edge_id,
                expected.owners, tuple(F(*x) for x in row.interval), F(*row.squared_distance_bound))
            # A repeated identical registry alias is the same observation, not
            # extra coverage. Conflicting intervals for one child must refuse.
            if row.edge_id in found and found[row.edge_id] != observation:
                raise OracleMismatch('conflicting source intervals for one prepared edge')
            found[row.edge_id] = observation
        observations.extend(found.values())
    result = compare_interval_observations(pinned.required_intervals, observations)
    _check(cancellation_check)
    validate_prepared_edge_subcurve_preimages_binding(prepared, binding,
                                                     cancellation_check=cancellation_check)
    # No callbacks after this validation: edge validation above can invoke the
    # last callback, so the detached face ownership proof must be rechecked now.
    validate_prepared_face_preimages_binding(prepared, face_binding)
    if (inventory != pinned or _json(to_dict(prepared)) != prepared_json
            or _json(to_dict(authored)) != pinned.source_document_json
            or _carrier_snapshot(authored) != pinned.raw_carriers
            or (str(authored.model_id), authored.revision) !=
                (pinned.source_model_id, pinned.source_revision)):
        raise OracleRefusal('oracle inputs changed during prepared comparison')
    return result
