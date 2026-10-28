"""Exact structural admission: shared injective projection, one scalar numerator.

Admitted pairs: polynomial-equivalent surfaces (uniform weights, possibly different huge/tiny constants per
surface; they cancel exactly) with equal degrees and knot vectors whose control differences B_ij - A_ij are all
parallel to one constant direction D, and whose D-orthogonal projection is an affine injective map of the native
knot parameters (verified on EVERY control against Greville abscissae; fitted coefficients are only proposals).
Then A(u,v) = B(s,t) <=> (u,v) = (s,t) and G(u,v) = 0 with G = (B - A).D / (D.D).  Everything else returns a
reason and the caller retains the general 4D atlas. Nonuniform weights additionally
admit the separately proved separable rational projection subclass with positive
rank-one proportional weights, full-domain monotonicity and C1 original supports.
"""
from __future__ import annotations

from fractions import Fraction

from ._records import StructuralCertificate
from ._source import SourceDefinition

__all__ = ["derive_structural", "ScalarField"]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _uniform(weights) -> bool:
    if weights is None:
        return True
    return len({Fraction(w) for row in weights for w in row}) == 1


def _greville(knots, degree, count):
    return tuple(sum((Fraction(k) for k in knots[i + 1: i + degree + 1]), Fraction(0)) / degree
                 for i in range(count))


def derive_structural(first: SourceDefinition, second: SourceDefinition, tick=None):
    """``(certificate, reason)``; the certificate is ``None`` with a reason when the pair is not admitted."""
    pu, pv, ku, kv, ca, wa, ur_a, vr_a = first.values
    qu, qv, lu, lv, cb, wb, ur_b, vr_b = second.values
    if (pu, pv, ku, kv) != (qu, qv, lu, lv):
        return None, "degrees or knot vectors differ"
    if not (_uniform(wa) and _uniform(wb)):
        from ._rational_projection import derive_rational_projection
        return derive_rational_projection(first,second,tick=tick)
    na, nb = len(ca), len(ca[0])
    if len(cb) != na or len(cb[0]) != nb:
        return None, "control grids differ"
    a = [[tuple(Fraction(x) for x in point) for point in row] for row in ca]
    b = [[tuple(Fraction(x) for x in point) for point in row] for row in cb]
    direction = None
    differences = [[None] * nb for _ in range(na)]
    for i in range(na):
        for j in range(nb):
            d = tuple(y - x for x, y in zip(a[i][j], b[i][j]))
            differences[i][j] = d
            if any(d):
                if direction is None:
                    direction = d
                elif any(_cross(direction, d)):
                    return None, "control differences are not parallel to one direction"
    if direction is None:
        return None, "coincident control nets (identical surfaces)"
    candidates = [c for c in (_cross(direction, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))) if any(c)]
    row1 = candidates[0]
    row2 = next((c for c in candidates[1:] if any(_cross(row1, c))), None)
    if row2 is None:
        return None, "cannot build two independent projection rows"
    gu, gv = _greville(ku, pu, na), _greville(kv, pv, nb)
    projected = [[(_dot(row1, a[i][j]), _dot(row2, a[i][j])) for j in range(nb)] for i in range(na)]
    affine = []
    for r in range(2):
        b_r = (projected[na - 1][0][r] - projected[0][0][r]) / (gu[-1] - gu[0])
        c_r = (projected[0][nb - 1][r] - projected[0][0][r]) / (gv[-1] - gv[0])
        a_r = projected[0][0][r] - b_r * gu[0] - c_r * gv[0]
        for i in range(na):
            for j in range(nb):
                if projected[i][j][r] != a_r + b_r * gu[i] + c_r * gv[j]:
                    return None, "projection is not an affine map of the native parameters"
        affine.append((a_r, b_r, c_r))
    determinant = affine[0][1] * affine[1][2] - affine[1][1] * affine[0][2]
    if determinant == 0:
        return None, "projection is not injective"
    norm = _dot(direction, direction)
    field = tuple(tuple(_dot(differences[i][j], direction) / norm for j in range(nb)) for i in range(na))
    low_u, high_u = max(min(ur_a), min(ur_b)), min(max(ur_a), max(ur_b))
    low_v, high_v = max(min(vr_a), min(vr_b)), min(max(vr_a), max(vr_b))
    if low_u > high_u or low_v > high_v:
        domain = None                                   # the selected rectangles do not meet
    elif low_u == high_u or low_v == high_v:
        return None, "selected rectangles touch only along a support boundary"
    else:
        domain = ((Fraction(low_u), Fraction(high_u)), (Fraction(low_v), Fraction(high_v)))
    certificate = StructuralCertificate(direction, (row1, row2), tuple(affine), determinant, gu, gv, field, domain)
    return certificate, ""


class ScalarField:
    """Private exact equation oracle with the field names the scalar engine consumes (NOT a published surface)."""

    __slots__ = ("degree_u", "degree_v", "knots_u", "knots_v", "controls", "u_range", "v_range", "_certificate")

    def __init__(self, source: SourceDefinition, certificate: StructuralCertificate):
        degree_u, degree_v, knots_u, knots_v = source.values[:4]
        (u0, u1), (v0, v1) = certificate.domain
        self.degree_u, self.degree_v, self.knots_u, self.knots_v = degree_u, degree_v, knots_u, knots_v
        self.controls = certificate.field
        self.u_range, self.v_range = (float(u0), float(u1)), (float(v0), float(v1))
        self._certificate = certificate

    def exact_net(self):
        c = self._certificate
        one = Fraction(1)
        return tuple(tuple((c.greville_u[i], c.greville_v[j], c.field[i][j], one)
                           for j in range(len(c.greville_v))) for i in range(len(c.greville_u)))
