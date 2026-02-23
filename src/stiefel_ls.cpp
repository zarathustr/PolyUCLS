#include "guc_ls/stiefel_ls.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <stdexcept>
#include <vector>

namespace guc_ls {

static Eigen::VectorXd vec_col_major(const Eigen::MatrixXd& M) {
    Eigen::VectorXd v(M.size());
    Eigen::Map<const Eigen::VectorXd> map(M.data(), M.size());
    v = map;
    return v;
}

static Eigen::MatrixXd unvec_col_major(const Eigen::VectorXd& v, int rows, int cols) {
    if (v.size() != rows * cols) throw std::invalid_argument("unvec size mismatch");
    Eigen::MatrixXd M(rows, cols);
    Eigen::Map<const Eigen::MatrixXd> map(v.data(), rows, cols);
    M = map;
    return M;
}

// Compute F(Lambda) = Σ S_i H_i S_i - I, where S_i = (Lambda + p_i I)^{-1}, H_i = h_i h_i^T.
static Eigen::MatrixXd compute_F(const Eigen::VectorXd& p,
                                 const Eigen::MatrixXd& H,
                                 const Eigen::MatrixXd& Lambda) {
    const int n = static_cast<int>(p.size());
    const int l = static_cast<int>(Lambda.rows());
    Eigen::MatrixXd F = -Eigen::MatrixXd::Identity(l, l);
    for (int i = 0; i < n; ++i) {
        const double pi = p(i);
        Eigen::MatrixXd Si = (Lambda + pi * Eigen::MatrixXd::Identity(l, l)).inverse();
        const Eigen::VectorXd hi = H.row(i).transpose();
        const Eigen::MatrixXd Hi = hi * hi.transpose();
        F += Si * Hi * Si;
    }
    // Symmetrize (numerical)
    return 0.5 * (F + F.transpose());
}

static Eigen::MatrixXd compute_dF(const Eigen::VectorXd& p,
                                  const Eigen::MatrixXd& H,
                                  const Eigen::MatrixXd& Lambda,
                                  const Eigen::MatrixXd& E) {
    const int n = static_cast<int>(p.size());
    const int l = static_cast<int>(Lambda.rows());
    Eigen::MatrixXd dF = Eigen::MatrixXd::Zero(l, l);
    for (int i = 0; i < n; ++i) {
        const double pi = p(i);
        Eigen::MatrixXd Si = (Lambda + pi * Eigen::MatrixXd::Identity(l, l)).inverse();
        const Eigen::VectorXd hi = H.row(i).transpose();
        const Eigen::MatrixXd Hi = hi * hi.transpose();
        const Eigen::MatrixXd SiE = Si * E * Si;
        dF -= (SiE * Hi * Si + Si * Hi * SiE);
    }
    return 0.5 * (dF + dF.transpose());
}

StiefelResult solve_stiefel_ls(const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& B,
                               const StiefelOptions& opt) {
    if (A.rows() != B.rows()) throw std::invalid_argument("A and B must have same number of rows");
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int l = static_cast<int>(B.cols());
    if (l > n) throw std::invalid_argument("B.cols() must be <= A.cols() (l <= n)");

    const Eigen::MatrixXd G = A.transpose() * A;
    const Eigen::MatrixXd V = A.transpose() * B;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition of G failed");
    const Eigen::VectorXd p = es.eigenvalues();
    const Eigen::MatrixXd Q = es.eigenvectors();
    const Eigen::MatrixXd H = Q.transpose() * V; // n x l

    // Newton on Lambda.
    Eigen::MatrixXd Lambda = Eigen::MatrixXd::Identity(l, l); // safe start
    bool converged = false;
    int it = 0;

    for (; it < opt.max_iters; ++it) {
        const Eigen::MatrixXd F = compute_F(p, H, Lambda);
        const double fn = F.norm();
        if (fn < opt.tol) {
            converged = true;
            break;
        }
        // Assemble Jacobian J for vec(F) w.r.t vec(Lambda).
        Eigen::MatrixXd J(l * l, l * l);
        for (int a = 0; a < l; ++a) {
            for (int b = 0; b < l; ++b) {
                Eigen::MatrixXd E = Eigen::MatrixXd::Zero(l, l);
                E(a, b) = 1.0;
                Eigen::MatrixXd dF = compute_dF(p, H, Lambda, E);
                J.col(a + b * l) = vec_col_major(dF);
            }
        }
        const Eigen::VectorXd rhs = -vec_col_major(F);
        Eigen::VectorXd delta = J.fullPivLu().solve(rhs);
        if (delta.size() != l * l) throw std::runtime_error("Newton solve failed (size)");
        Eigen::MatrixXd E = unvec_col_major(delta, l, l);

        // Damped update and symmetrize.
        Lambda = Lambda + opt.step_damping * E;
        Lambda = 0.5 * (Lambda + Lambda.transpose());
    }

    // Recover Y and then X.
    Eigen::MatrixXd Y(n, l);
    for (int i = 0; i < n; ++i) {
        const double pi = p(i);
        Eigen::MatrixXd Si = (Lambda + pi * Eigen::MatrixXd::Identity(l, l)).inverse();
        Y.row(i) = H.row(i) * Si;
    }
    Eigen::MatrixXd X = Q * Y;

    // Project back to the Stiefel manifold (polar decomposition) for numerical safety.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    X = svd.matrixU() * svd.matrixV().transpose();

    StiefelResult out;
    out.X = X;
    out.Lambda = Lambda;
    out.iters = it;
    out.converged = converged;

    const Eigen::MatrixXd R = A * X - B;
    out.cost = R.squaredNorm();
    return out;
}

Eigen::MatrixXd solve_procrustes(const Eigen::MatrixXd& V) {
    if (V.rows() != V.cols()) throw std::invalid_argument("Procrustes requires square V");
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd W = svd.matrixV();
    Eigen::MatrixXd X = U * W.transpose();

    // Ensure det(X)=+1 if desired (SO(n) vs O(n)). Here we keep O(n).
    return X;
}

} // namespace guc_ls
