#pragma once
#include "guc_ls/polynomial.hpp"
#include <complex>
#include <vector>

namespace guc_ls {

// Compute all (complex) roots using a companion matrix eigen-decomposition.
// Requires Eigen; implemented in src/roots.cpp.
std::vector<std::complex<long double>> roots_companion(const Poly& p);

// Convenience: return real roots filtered by imag tolerance.
std::vector<long double> real_roots_companion(const Poly& p, long double imag_tol = 1e-10L);

} // namespace guc_ls
