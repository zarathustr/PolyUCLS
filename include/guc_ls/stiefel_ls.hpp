#pragma once
#include <Eigen/Dense>

namespace guc_ls {

struct StiefelOptions {
    int max_iters = 50;
    double tol = 1e-12;
    double step_damping = 1.0; // 1.0 means full Newton step; <1 enables damping.
};

struct StiefelResult {
    Eigen::MatrixXd X;        // n x l
    Eigen::MatrixXd Lambda;   // l x l (symmetric)
    double cost = 0.0;
    int iters = 0;
    bool converged = false;
};

// Solve min ||A X - B||_F^2  s.t. X^T X = I_l  (real case).
StiefelResult solve_stiefel_ls(const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& B,
                               const StiefelOptions& opt = {});

// Orthogonal Procrustes (square case): argmin ||X - V||_F^2 s.t. X^T X = I.
Eigen::MatrixXd solve_procrustes(const Eigen::MatrixXd& V);

} // namespace guc_ls
