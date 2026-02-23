#include "guc_ls/roots.hpp"

#include <Eigen/Dense>
#include <stdexcept>

namespace guc_ls {

std::vector<std::complex<long double>> roots_companion(const Poly& p_in) {
    Poly p = trim_leading_zeros(p_in);
    const int d = degree(p);
    if (d < 1) return {};
    const long double a0 = p[0];
    if (a0 == 0.0L) {
        throw std::invalid_argument("roots_companion: leading coefficient is zero");
    }
    // Normalize to monic.
    for (auto& c : p) c /= a0;

    // Build companion matrix in long double and compute eigenvalues in complex<long double>.
    Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> C(d, d);
    C.setZero();
    // First row: -a1 .. -ad
    for (int j = 0; j < d; ++j) {
        C(0, j) = -p[static_cast<size_t>(j + 1)];
    }
    // Subdiagonal ones
    for (int i = 1; i < d; ++i) {
        C(i, i - 1) = 1.0L;
    }

    Eigen::EigenSolver<Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>> es(C, /* computeEigenvectors = */ false);
    const auto evals = es.eigenvalues();

    std::vector<std::complex<long double>> roots;
    roots.reserve(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) {
        const auto& z = evals(i);
        roots.emplace_back(static_cast<long double>(z.real()), static_cast<long double>(z.imag()));
    }
    return roots;
}

std::vector<long double> real_roots_companion(const Poly& p, long double imag_tol) {
    std::vector<long double> reals;
    for (const auto& z : roots_companion(p)) {
        const long double r = z.real();
        const long double im = z.imag();
        const long double tol = imag_tol * (1.0L + std::fabsl(r));
        if (std::fabsl(im) <= tol) {
            reals.push_back(r);
        }
    }
    return reals;
}

} // namespace guc_ls
