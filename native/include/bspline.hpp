// B-spline / NURBS curve evaluation with derivatives (Piegl & Tiller A2.1, A2.3, A3.2, A4.2).
//
// Mirrors anygeometry/reference/bspline.py. Inputs are validated by the caller
// (anygeometry.backend): clamped non-decreasing knots of length n + p + 2,
// positive weights, parameters inside [knots[p], knots[n+1]].
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace anygeometry_core {

constexpr int kMaxDegree = 15;
constexpr int kMaxOrder = 2;
constexpr int kMaxDimension = 4;

// Index i with knots[i] <= t < knots[i+1]; the end of the domain belongs to the
// last non-empty span. Same rule as reference.find_spans.
inline Py_ssize_t find_span(int degree, const double* knots, Py_ssize_t count, double t) {
    const Py_ssize_t n = count - 1;
    if (t >= knots[n + 1]) {
        Py_ssize_t last = n;
        while (last > degree && knots[last] == knots[n + 1]) {
            --last;
        }
        return last;
    }
    // upper_bound over the whole vector, then clip to [degree, n].
    const double* end = knots + n + degree + 2;
    const Py_ssize_t span = (std::upper_bound(knots, end, t) - knots) - 1;
    return std::min(std::max(span, static_cast<Py_ssize_t>(degree)), n);
}

// ders[k * (p+1) + j]: k-th derivative of the j-th non-zero basis function (A2.3).
inline void basis_derivatives(int p, const double* knots, Py_ssize_t span, double t, int order, double* ders) {
    double ndu[kMaxDegree + 1][kMaxDegree + 1];
    double left[kMaxDegree + 1], right[kMaxDegree + 1];
    ndu[0][0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            const double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    const int width = p + 1;
    for (int k = 0; k <= order; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders[k * width + j] = 0.0;
        }
    }
    for (int j = 0; j <= p; ++j) {
        ders[j] = ndu[j][p];
    }
    const int top = std::min(order, p);
    double a[2][kMaxDegree + 1];
    for (int r = 0; r <= p; ++r) {
        for (int j = 0; j <= p; ++j) {
            a[0][j] = a[1][j] = 0.0;
        }
        a[0][0] = 1.0;
        int s1 = 0, s2 = 1;
        for (int k = 1; k <= top; ++k) {
            double d = 0.0;
            const int rk = r - k, pk = p - k;
            if (r >= k) {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            const int j1 = rk >= -1 ? 1 : -rk;
            const int j2 = (r - 1 <= pk) ? k - 1 : p - r;
            for (int j = j1; j <= j2; ++j) {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d = d + a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk) {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d = d + a[s2][k] * ndu[r][pk];
            }
            ders[k * width + r] = d;
            std::swap(s1, s2);
        }
    }
    double factor = static_cast<double>(p);
    for (int k = 1; k <= top; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders[k * width + j] *= factor;
        }
        factor *= static_cast<double>(p - k);
    }
}

// Point and derivatives up to ``order`` at one parameter; out[k * dim + d].
// ``weights`` may be null for a polynomial curve.
inline void evaluate_point(int p, const double* knots, Py_ssize_t count, const double* controls, int dim,
                           const double* weights, double t, int order, double* out) {
    double ders[(kMaxOrder + 1) * (kMaxDegree + 1)];
    const Py_ssize_t span = find_span(p, knots, count, t);
    basis_derivatives(p, knots, span, t, order, ders);
    const int width = p + 1;
    const Py_ssize_t first = span - p;
    if (weights == nullptr) {
        for (int k = 0; k <= order; ++k) {
            for (int d = 0; d < dim; ++d) {
                double sum = 0.0;
                for (int j = 0; j <= p; ++j) {
                    sum += ders[k * width + j] * controls[(first + j) * dim + d];
                }
                out[k * dim + d] = sum;
            }
        }
        return;
    }
    // Rational curve, evaluated in a local frame anchored at the first active
    // control point P_r: with Delta_j = P_j - P_r,
    //   E = sum_j N_j w_j Delta_j / w,  C = P_r + E,
    //   D_k = sum_j N_j^(k) w_j (Delta_j - E),  C' = D_1 / w,  C'' = (D_2 - 2 w' C') / w.
    // Algebraically A4.2, but without its cancellation: errors scale with the
    // local control spread, and a constant curve gives exact zero derivatives.
    double w[kMaxOrder + 1], offset[kMaxDimension];
    const double* anchor = controls + first * dim;
    for (int k = 0; k <= order; ++k) {
        double wk = 0.0;
        for (int j = 0; j <= p; ++j) {
            wk += ders[k * width + j] * weights[first + j];
        }
        w[k] = wk;
    }
    for (int d = 0; d < dim; ++d) {
        double sum = 0.0;
        for (int j = 0; j <= p; ++j) {
            sum += ders[j] * (weights[first + j] * (controls[(first + j) * dim + d] - anchor[d]));
        }
        offset[d] = sum / w[0];
        out[d] = anchor[d] + offset[d];
    }
    for (int k = 1; k <= order; ++k) {
        for (int d = 0; d < dim; ++d) {
            double sum = 0.0;
            for (int j = 0; j <= p; ++j) {
                sum += ders[k * width + j]
                       * (weights[first + j] * ((controls[(first + j) * dim + d] - anchor[d]) - offset[d]));
            }
            if (k == 2) {
                sum -= 2.0 * w[1] * out[dim + d];
            }
            out[k * dim + d] = sum / w[0];
        }
    }
}

}  // namespace anygeometry_core
