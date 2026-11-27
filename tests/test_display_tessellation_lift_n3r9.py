"""N3 Codex round 9: raw exceptions from hostile subclasses on the option-A checks (owner decision:
identity type checks; the incident-face preflight calls the model like N1 does, a failing model lookup
is a typed refusal; hostile GeometryModel subclasses are out of contract like hostile supports)."""

import numpy as np
import pytest

from anygeometry import display_tessellation as dt
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Plane

TOL = dt.DisplayTolerance(.25, sag=.001)


class _UnhashableClass(type):
    __hash__ = None


class _PlainSupport(Plane, metaclass=_UnhashableClass):
    pass


class _DefaultErrorStateModel(GeometryModel):
    def faces_using_edge(self, edge_id):
        if np.geterr()["invalid"] != "warn":
            raise RuntimeError("model lookup requires NumPy default error state")
        return super().faces_using_edge(edge_id)


class _FailingLookupModel(GeometryModel):
    armed = False

    def faces_using_edge(self, edge_id):
        if self.armed:
            raise KeyError("lookup failed")
        return super().faces_using_edge(edge_id)


def _triangle(model_cls=GeometryModel, support_cls=Plane, neighbour_cls=None):
    g = model_cls()
    a, b, c = [g.add_point(*p) for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    edges = [g.add_line(a, b), g.add_line(b, c), g.add_line(c, a)]
    f = g.add_face(edges, surface=support_cls((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    if neighbour_cls is not None:
        d = g.add_point(1, 1, 0)
        g.add_face([edges[1], g.add_line(c, d), g.add_line(d, b)],
                   surface=neighbour_cls((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    return g, f, edges


def test_unhashable_metaclass_subclass_is_a_typed_refusal():
    g, f, _ = _triangle(support_cls=_PlainSupport)
    with pytest.raises(dt.DisplayTrimRefused) as info:
        dt.face_display_mesh(g, f, TOL)
    assert info.value.code == "SUPPORT_UNSUPPORTED"


def test_unhashable_metaclass_subclass_on_a_neighbour_is_a_typed_refusal():
    g, f, _ = _triangle(neighbour_cls=_PlainSupport)
    with pytest.raises(dt.DisplayTrimRefused) as info:
        dt.face_display_mesh(g, f, TOL)
    assert info.value.code == "SUPPORT_UNSUPPORTED"


def test_preflight_calls_the_model_like_n1():
    g, f, edges = _triangle(model_cls=_DefaultErrorStateModel)
    assert len(dt.edge_display_samples(g, edges[0], TOL).points) == 2
    plain, pf, _ = _triangle()
    m, ref = dt.face_display_mesh(g, f, TOL), dt.face_display_mesh(plain, pf, TOL)
    assert np.array_equal(m.points, ref.points) and np.array_equal(m.triangles, ref.triangles)


def test_failing_model_lookup_in_the_preflight_is_a_typed_refusal():
    g, f, _ = _triangle(model_cls=_FailingLookupModel)
    g.armed = True
    with pytest.raises(dt.DisplayTrimRefused) as info:
        dt.face_display_mesh(g, f, TOL)
    assert info.value.code == "SUPPORT_UNSUPPORTED"


class _LazyDefaultErrorStateModel(GeometryModel):
    """Codex N3 r10: the lookup is a generator that checks N1's error state while it is iterated."""

    def faces_using_edge(self, edge_id):
        if np.geterr()["invalid"] != "warn":
            raise RuntimeError("lookup iteration requires N1 error state")
        yield from super().faces_using_edge(edge_id)


class _LazyFailingModel(GeometryModel):
    armed = False

    def faces_using_edge(self, edge_id):
        yield from super().faces_using_edge(edge_id)
        if self.armed:
            raise KeyError("lookup failed during iteration")


def test_lazy_lookup_is_consumed_like_n1():
    g, f, edges = _triangle(model_cls=_LazyDefaultErrorStateModel)
    assert len(dt.edge_display_samples(g, edges[0], TOL).points) == 2
    plain, pf, _ = _triangle()
    m, ref = dt.face_display_mesh(g, f, TOL), dt.face_display_mesh(plain, pf, TOL)
    assert np.array_equal(m.points, ref.points) and np.array_equal(m.triangles, ref.triangles)


def test_lazy_lookup_failing_during_iteration_is_a_typed_refusal():
    g, f, _ = _triangle(model_cls=_LazyFailingModel)
    g.armed = True
    with pytest.raises(dt.DisplayTrimRefused) as info:
        dt.face_display_mesh(g, f, TOL)
    assert info.value.code == "SUPPORT_UNSUPPORTED"
