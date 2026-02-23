#pragma once

#include <Eigen/Dense>

namespace guc_ls {

// Project an arbitrary square matrix M to the nearest orthogonal matrix (O(n))
// in Frobenius norm, i.e. argmin_{X^T X = I} ||X - M||_F.
// If enforce_det_pos==true, the projection is onto SO(n).
Eigen::MatrixXd project_orthogonal(const Eigen::MatrixXd& M, bool enforce_det_pos = false);

// Square unit-constraint LS (orthogonal Procrustes) closed-form:
//   maximize tr(X^T V)  s.t. X^T X = I.
// If enforce_det_pos==true, returns the SO(n) solution (Kabsch sign correction).
Eigen::MatrixXd procrustes_svd(const Eigen::MatrixXd& V, bool enforce_det_pos = false);

// Same Procrustes solution via polar factor:
//   X = V (V^T V)^{-1/2}.
// If enforce_det_pos==true, applies an SO(n) sign correction using the *smallest*
// singular direction (identified from eigenvalues of V^T V).
Eigen::MatrixXd procrustes_polar_sqrt(const Eigen::MatrixXd& V,
                                      bool enforce_det_pos = false,
                                      double eps = 1e-12);

// Baseline: unconstrained least squares X_ls = argmin ||A X - B||_F^2, then project to O(n)/SO(n).
Eigen::MatrixXd ls_then_project(const Eigen::MatrixXd& A,
                                const Eigen::MatrixXd& B,
                                bool enforce_det_pos = false);

// "GB representative": enumerate the 2^n symmetric square roots of S = V^T V,
// produce X = V Lambda^{-1}, keep feasible (orthogonal) candidates, and pick the best
// by maximizing tr(X^T V).
//
// For generic full-rank V, this returns the same global optimum as procrustes_svd,
// but exposes the polynomial multi-solution structure (16 solutions for n=4).
Eigen::MatrixXd procrustes_gb_enum(const Eigen::MatrixXd& V,
                                   bool enforce_det_pos = false,
                                   double eps = 1e-12);

// Geodesic distance on SO(n) between X_true and X_est (in degrees), computed from
// eigenvalue arguments of R = X_true^T X_est.
//
// For SO(3) this reduces to the usual rotation angle. For SO(4) it corresponds to
// sqrt(theta1^2 + theta2^2) where eigenvalues are e^{±i theta1}, e^{±i theta2}.
double orthogonal_geodesic_distance_deg(const Eigen::MatrixXd& X_true,
                                        const Eigen::MatrixXd& X_est);

} // namespace guc_ls
