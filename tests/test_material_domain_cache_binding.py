"""Derived exact arithmetic caches are not part of owner definitions."""
from dataclasses import fields
from fractions import Fraction
from anygeometry import GeometryModel
from anygeometry.material_arrangement import MaterialDomain
from anygeometry.definition_binding import definition_checksum


def test_material_domain_cache_warming_preserves_binding():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    domain = MaterialDomain.from_model(model, face)
    before = definition_checksum(domain)
    for name in ("_exact_chart_cache", "_exact_boundary_cache"):
        assert next(f for f in fields(domain) if f.name == name).metadata["definition"] is False
        object.__setattr__(domain, name, (Fraction(1, 3),))
        assert definition_checksum(domain) == before
    object.__setattr__(domain, "face_id", domain.face_id + 1)
    assert definition_checksum(domain) != before
