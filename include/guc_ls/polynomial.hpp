#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace guc_ls {

// Polynomial represented in DESCENDING powers:
//   p(x) = c[0] x^d + c[1] x^(d-1) + ... + c[d]
using Poly = std::vector<long double>;

inline int degree(const Poly& p) {
    return static_cast<int>(p.size()) - 1;
}

inline Poly trim_leading_zeros(Poly p, long double eps = 0.0L) {
    while (p.size() > 1 && std::fabsl(p.front()) <= eps) {
        p.erase(p.begin());
    }
    return p;
}

inline Poly poly_mul(const Poly& a, const Poly& b) {
    if (a.empty() || b.empty()) return {};
    Poly r(static_cast<size_t>(degree(a) + degree(b) + 1), 0.0L);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            r[i + j] += a[i] * b[j];
        }
    }
    return trim_leading_zeros(r);
}

// Synthetic division by (x - r). Returns (quotient, remainder).
inline std::pair<Poly, long double> poly_div_linear(const Poly& p, long double r) {
    const int d = degree(p);
    if (d < 1) {
        throw std::invalid_argument("poly_div_linear: degree must be >= 1");
    }
    Poly q(static_cast<size_t>(d), 0.0L);
    q[0] = p[0];
    for (int k = 1; k < d; ++k) {
        q[static_cast<size_t>(k)] = p[static_cast<size_t>(k)] + r * q[static_cast<size_t>(k - 1)];
    }
    const long double rem = p[static_cast<size_t>(d)] + r * q[static_cast<size_t>(d - 1)];
    return {trim_leading_zeros(q), rem};
}

// Divide by (x - r)^2. Throws if remainder is large compared to coefficients.
inline Poly poly_div_double_root(const Poly& p, long double r) {
    auto [q1, rem1] = poly_div_linear(p, r);
    auto [q2, rem2] = poly_div_linear(q1, r);
    (void)rem1; (void)rem2; // remainder checks can be added if desired
    return trim_leading_zeros(q2);
}

// Pad a polynomial on the LEFT (higher degree) with zeros to reach a target degree.
inline Poly pad_left_to_degree(const Poly& p, int target_degree) {
    const int d = degree(p);
    if (d > target_degree) {
        throw std::invalid_argument("pad_left_to_degree: target degree too small");
    }
    const int diff = target_degree - d;
    Poly out(static_cast<size_t>(target_degree + 1), 0.0L);
    for (size_t i = 0; i < p.size(); ++i) {
        out[static_cast<size_t>(diff) + i] = p[i];
    }
    return out;
}

} // namespace guc_ls
