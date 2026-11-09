"""The index may omit only boxes rejected by the unchanged scan predicate."""
import math
import random
import time
import warnings

import numpy as np
import pytest

from anygeometry import material_arrangement as material
from test_persisted_certification import PLANE, POINTS, EDGES, build, cells, loop


class Scan:
    """Original enumerate(cell_boxes), with no spatial filtering."""
    def __init__(self, boxes):
        self.count = len(boxes)

    def candidates(self, x, y):
        return range(self.count)


def selected(boxes, indices, x, y):
    return [i for i in indices if boxes[i][0][0] <= x <= boxes[i][1][0]
            and boxes[i][0][1] <= y <= boxes[i][1][1]]


@pytest.mark.parametrize('size', [0, 1, 31, 32, 33, 257])
def test_candidates_preserve_closed_box_membership_and_original_order(size):
    rng = random.Random(1123)
    boxes = []
    for _ in range(size):
        x, y = rng.uniform(-30, 30), rng.uniform(-30, 30)
        boxes.append(((x, y), (x + rng.random() * 7, y + rng.random() * 7)))
    index = material._CellBoxIndex(boxes)
    probes = [(rng.uniform(-40, 40), rng.uniform(-40, 40)) for _ in range(100)]
    probes += [point for box in boxes for point in box]
    probes += [(np.nextafter(x, direction), y) for box in boxes for x, y in box
               for direction in (-math.inf, math.inf)]
    for x, y in probes:
        actual = list(index.candidates(x, y))
        assert actual == sorted(set(actual))
        assert selected(boxes, actual, x, y) == selected(boxes, range(size), x, y)


def test_degenerate_extreme_signed_zero_and_identical_boxes():
    boxes = [((-1e308, -1e308), (1e308, 1e308)), ((-0., -0.), (0., 0.)),
             ((1., 1.), (1., 1.)), ((-math.ulp(0.), 0.), (math.ulp(0.), 0.))] * 16
    index = material._CellBoxIndex(boxes)
    assert index.root is not None
    for x, y in [(0., -0.), (1., 1.), (1e308, -1e308), (math.ulp(0.), 0.)]:
        assert selected(boxes, index.candidates(x, y), x, y) == selected(boxes, range(len(boxes)), x, y)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_boxes_and_queries_fall_back_to_original_scan(bad):
    boxes = [((0., 0.), (1., 1.))] * 40
    good = material._CellBoxIndex(boxes)
    assert list(good.candidates(bad, .5)) == list(range(40))
    boxes[17] = ((bad, 0.), (1., 1.))
    index = material._CellBoxIndex(boxes)
    assert index.root is None
    assert list(index.candidates(.5, .5)) == list(range(40))


def test_unsupported_collection_size_retains_scan():
    index = material._CellBoxIndex([((0., 0.), (1., 1.))] * 100001)
    assert index.root is None


def test_optional_construction_allocation_failure_retains_scan(monkeypatch):
    calls = []
    def fail(*args):
        calls.append(True)
        raise MemoryError('optional index')
    monkeypatch.setattr(material, 'min', fail, raising=False)
    index = material._CellBoxIndex([((0., 0.), (1., 1.))] * 40)
    assert index.root is None
    assert calls == [True]


def test_optional_query_allocation_failure_retains_scan():
    index = material._CellBoxIndex([((0., 0.), (1., 1.))] * 40)
    class Exhausted:
        def __iter__(self):
            raise MemoryError('optional query')
    node = index.root
    index.root = (*node[:6], Exhausted())
    assert list(index.candidates(.5, .5)) == list(range(40))


def test_nonfinite_numpy_box_fallback_preserves_invalid_comparison_error():
    boxes = [(np.array([0., 0.]), np.array([1., 1.])) for _ in range(40)]
    boxes[3][0][0] = float('nan')
    index = material._CellBoxIndex(boxes)
    assert index.root is None
    with np.errstate(invalid='raise'):
        # NumPy versions differ in whether scalar comparisons raise here;
        # both paths must expose the same result or error, with no filtering.
        def attempt(indices):
            try:
                return selected(boxes, indices, .5, .5)
            except FloatingPointError as error:
                return type(error), str(error)
        assert attempt(index.candidates(.5, .5)) == attempt(range(40))


def fixture(kind):
    # The existing touching/shared-edge/hole fixture plus enough far-away
    # ordinary cells to exercise the production index threshold.
    points, edges, faces = dict(POINTS), list(EDGES), cells()
    for i in range(120):
        names = [f'{i}-{j}' for j in range(4)]
        x = 20 + 3 * i
        points.update(zip(names, [(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]))
        edges.extend(zip(names, names[1:] + names[:1]))
        faces.append((loop(edges, names), []))
    if kind == 'overlap':
        names = ['inner-a', 'inner-b', 'inner-c', 'inner-d']
        points.update(zip(names, [(20.2, .2), (20.8, .2), (20.8, .8), (20.2, .8)]))
        edges.extend(zip(names, names[1:] + names[:1]))
        faces.append((loop(edges, names), []))
    elif kind == 'crossing':
        points['q'] = (2., .5)
    elif kind == 'hole':
        points.update(p=(2.5, .5), q=(3.5, .5), r=(3.5, 1.5), s=(2.5, 1.5))
    elif kind == 'nearby':
        # Disjoint triangles with overlapping boxes: their diagonal edge
        # midpoints enter the other box but remain outside the exact region.
        for group in range(4):
            x = 1000 + 5 * group
            for side, coords in enumerate(([(x,0),(x+1.8,0),(x,1.8)],
                                           [(x+2,2),(x+.2,2),(x+2,.2)])):
                names = [f'near-{group}-{side}-{i}' for i in range(3)]
                points.update(zip(names, coords))
                edges.extend(zip(names, names[1:] + names[:1]))
                faces.append((loop(edges, names), []))
    return points, edges, faces


def outcome(factory, monkeypatch, kind='valid', stop=None, polygon_stop=None):
    from anygeometry import exact_predicates
    points, edges, faces = fixture(kind)
    edge_ends, position = build(points, edges)
    events = []
    original_polygon = exact_predicates.point_in_polygon
    polygon_calls = 0
    def polygon(probe, ring):
        nonlocal polygon_calls
        polygon_calls += 1
        events.append(('polygon', tuple(probe), tuple(ring)))
        if polygon_calls == polygon_stop:
            raise LookupError('narrow-phase stop')
        return original_polygon(probe, ring)
    def checked_position(vertex):
        events.append(('position', vertex))
        return position(vertex)
    def check():
        events.append(('check',))
        if stop is not None and sum(event == ('check',) for event in events) == stop:
            raise RuntimeError('caller stop')
    with monkeypatch.context() as patch:
        patch.setattr(material, '_CellBoxIndex', factory)
        patch.setattr(exact_predicates, 'point_in_polygon', polygon)
        try:
            result = material.certify_persisted_cells(PLANE, faces, edge_ends, checked_position, check=check)
            status = ('returned', result)
        except Exception as error:
            status = (type(error).__name__, str(error))
    return status, events


@pytest.mark.parametrize('kind', ['valid', 'nearby', 'overlap', 'crossing', 'hole'])
def test_complete_certification_matches_scan_results_failures_and_callbacks(monkeypatch, kind):
    indexed = outcome(material._CellBoxIndex, monkeypatch, kind)
    assert indexed == outcome(Scan, monkeypatch, kind)
    if kind in ('valid', 'nearby'):
        assert indexed[0] == ('returned', None)
    else:
        assert indexed[0][0] == 'GeometryError'


@pytest.mark.parametrize('stop', [1, 10, 150, 550, 650])
def test_callback_exception_sequence_is_identical_including_indexed_loop(monkeypatch, stop):
    indexed = outcome(material._CellBoxIndex, monkeypatch, stop=stop)
    assert indexed == outcome(Scan, monkeypatch, stop=stop)
    assert indexed[0] == ('RuntimeError', 'caller stop')


def test_sparse_boxes_prune_the_python_all_cell_scan():
    boxes = [((float(i * 3), 0.), (float(i * 3 + 1), 1.)) for i in range(4096)]
    index = material._CellBoxIndex(boxes)
    candidates = list(index.candidates(1500.5, .5))
    assert 500 in candidates
    assert len(candidates) <= 8


@pytest.mark.parametrize('kind', ['valid', 'overlap'])
def test_integrated_index_is_active_and_prunes_with_identical_narrow_phase(monkeypatch, kind):
    activity = []
    original = material._CellBoxIndex
    class Observed(original):
        def __init__(self, boxes):
            super().__init__(boxes)
            activity.append(('build', self.root is not None))
        def candidates(self, x, y):
            values = super().candidates(x, y)
            activity.append(('query', len(values) < self.count))
            return values
    indexed = outcome(Observed, monkeypatch, kind)
    assert indexed == outcome(Scan, monkeypatch, kind)
    assert ('build', True) in activity
    assert ('query', True) in activity
    assert any(event[0] == 'polygon' for event in indexed[1])


@pytest.mark.parametrize('offset', [1, 2, 3])
def test_narrow_phase_exception_call_order_matches_original_scan(monkeypatch, offset):
    original = material._CellBoxIndex
    baseline = outcome(original, monkeypatch, kind='nearby')
    count = sum(event[0] == 'polygon' for event in baseline[1])
    assert count > offset
    stop = count - offset
    indexed = outcome(original, monkeypatch, kind='nearby', polygon_stop=stop)
    assert indexed == outcome(Scan, monkeypatch, kind='nearby', polygon_stop=stop)
    assert indexed[0] == ('LookupError', 'narrow-phase stop')


def test_shared_axis_nested_and_long_boxes_preserve_order_and_bound_depth():
    boxes = [((-float(i), -1.), (float(i), 1.)) for i in range(1, 1025)]
    boxes += [((-1e308, 0.), (1e308, 0.)), ((0., -1e308), (0., 1e308))]
    index = material._CellBoxIndex(boxes)
    def depth(node):
        return 1 if node[6] is not None else 1 + max(depth(node[4]), depth(node[5]))
    assert depth(index.root) <= math.ceil(math.log2(len(boxes))) + 1
    for x, y in [(0.,0.), (.5,.5), (-1024.,0.), (1e308,0.), (0.,-1e308)]:
        actual = list(index.candidates(x, y))
        assert actual == sorted(set(actual))
        assert selected(boxes, actual, x, y) == selected(boxes, range(len(boxes)), x, y)


def test_maximum_indexed_size_is_supported_with_recorded_build_cost(record_property):
    boxes = [((0.,0.), (1.,1.))] * 100000
    started = time.perf_counter()
    index = material._CellBoxIndex(boxes)
    record_property('index_build_wall_seconds', time.perf_counter() - started)
    assert index.root is not None
    assert index.candidates(2.,2.) == []
    assert list(index.candidates(.5,.5)) == list(range(100000))


def test_index_build_and_query_add_no_numpy_warnings():
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error', RuntimeWarning)
        boxes = [(np.array([-1e308, -1e308]), np.array([1e308, 1e308]))] * 40
        index = material._CellBoxIndex(boxes)
        assert list(index.candidates(1e308, -1e308)) == list(range(40))
        boxes[0] = (np.array([float('nan'),0.]), np.array([1.,1.]))
        assert material._CellBoxIndex(boxes).root is None
