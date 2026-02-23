#pragma once
#include "guc_ls/polynomial.hpp"
#include "guc_ls/roots.hpp"
#include <Eigen/Dense>
#include <vector>

namespace guc_ls {

struct UnitNormOptions {
    // (Kept for backwards compatibility; not used by the current PolyRoot implementation.)
    long double imag_tol = 1e-10L;

    // Tolerance for detecting |p_i - lambda| ≈ 0 in the eigen-basis.
    // Used to safely handle the singular KKT "hard case".
    long double pole_tol = 1e-12L;
};

struct UnitNormResult {
    Eigen::VectorXd u;          // unit-norm minimizer
    long double lambda = 0.0L;  // selected Lagrange multiplier
    long double cost = 0.0L;    // objective value u^T G u - 2 v^T u (constant term omitted)
    std::vector<long double> candidate_lambdas;
};

// Solve min ||B u - g||^2  s.t. ||u||=1.
UnitNormResult solve_unit_norm_ls(const Eigen::MatrixXd& B,
                                  const Eigen::VectorXd& g,
                                  const UnitNormOptions& opt = {});

// Solve using the normal matrix (G = B^T B, v = B^T g). Objective returned is u^T G u - 2 v^T u.
UnitNormResult solve_unit_norm_ls_normal(const Eigen::MatrixXd& G,
                                         const Eigen::VectorXd& v,
                                         const UnitNormOptions& opt = {});

// Jacobian du/dv from the implicit KKT linearization at a solution (u,lambda).
Eigen::MatrixXd jacobian_u_wrt_v(const Eigen::MatrixXd& G,
                                 const Eigen::VectorXd& u,
                                 long double lambda);

// First-order covariance propagation: Sigma_u ≈ J Sigma_v J^T.
Eigen::MatrixXd covariance_u(const Eigen::MatrixXd& G,
                             const Eigen::VectorXd& u,
                             long double lambda,
                             const Eigen::MatrixXd& Sigma_v);

} // namespace guc_ls
