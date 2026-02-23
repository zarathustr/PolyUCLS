#include "guc_ls/procrustes.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace guc_ls {

Eigen::MatrixXd project_orthogonal(const Eigen::MatrixXd& M, bool enforce_det_pos) {
    if (M.rows() != M.cols()) {
        throw std::invalid_argument("project_orthogonal: M must be square");
    }
    const int n = static_cast<int>(M.rows());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::MatrixXd X = U * V.transpose();
    if (enforce_det_pos) {
        const double detX = X.determinant();
        if (detX < 0.0) {
            Eigen::MatrixXd D = Eigen::MatrixXd::Identity(n, n);
            D(n - 1, n - 1) = -1.0;
            X = U * D * V.transpose();
        }
    }
    return X;
}

Eigen::MatrixXd procrustes_svd(const Eigen::MatrixXd& V, bool enforce_det_pos) {
    return project_orthogonal(V, enforce_det_pos);
}

Eigen::MatrixXd procrustes_polar_sqrt(const Eigen::MatrixXd& V,
                                      bool enforce_det_pos,
                                      double eps) {
    if (V.rows() != V.cols()) {
        throw std::invalid_argument("procrustes_polar_sqrt: V must be square");
    }
    const int n = static_cast<int>(V.rows());
    const Eigen::MatrixXd S = V.transpose() * V;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("procrustes_polar_sqrt: eigen decomposition failed");
    }
    const Eigen::VectorXd evals = es.eigenvalues();
    const Eigen::MatrixXd W = es.eigenvectors(); // right singular vectors

    Eigen::VectorXd inv_sqrt(n);
    Eigen::VectorXd sigma(n);
    for (int i = 0; i < n; ++i) {
        const double lam = std::max(0.0, evals(i));
        sigma(i) = std::sqrt(lam);
        inv_sqrt(i) = (lam > eps) ? (1.0 / std::sqrt(lam)) : 0.0;
    }

    const Eigen::MatrixXd SinvSqrt = W * inv_sqrt.asDiagonal() * W.transpose();
    Eigen::MatrixXd X = V * SinvSqrt;

    // SO(n) correction without using SVD:
    // If det(X)<0, flip the direction associated with the smallest singular value.
    if (enforce_det_pos && X.determinant() < 0.0) {
        int idx = 0;
        double smin = sigma(0);
        for (int i = 1; i < n; ++i) {
            if (sigma(i) < smin) { smin = sigma(i); idx = i; }
        }
        Eigen::MatrixXd D = Eigen::MatrixXd::Identity(n, n);
        D(idx, idx) = -1.0;
        X = X * (W * D * W.transpose());
    }

    return X;
}

Eigen::MatrixXd ls_then_project(const Eigen::MatrixXd& A,
                                const Eigen::MatrixXd& B,
                                bool enforce_det_pos) {
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("ls_then_project: A and B must have same number of rows");
    }
    Eigen::MatrixXd X_ls = A.colPivHouseholderQr().solve(B);
    return project_orthogonal(X_ls, enforce_det_pos);
}

Eigen::MatrixXd procrustes_gb_enum(const Eigen::MatrixXd& V,
                                   bool enforce_det_pos,
                                   double eps) {
    if (V.rows() != V.cols()) {
        throw std::invalid_argument("procrustes_gb_enum: V must be square");
    }
    const int n = static_cast<int>(V.rows());
    const Eigen::MatrixXd S = V.transpose() * V;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("procrustes_gb_enum: eigen decomposition failed");
    }
    const Eigen::VectorXd evals = es.eigenvalues();
    const Eigen::MatrixXd W = es.eigenvectors();

    Eigen::VectorXd sigma(n);
    for (int i = 0; i < n; ++i) {
        sigma(i) = std::sqrt(std::max(0.0, evals(i)));
    }

    const int num = 1 << n;
    double bestScore = -std::numeric_limits<double>::infinity();
    Eigen::MatrixXd bestX;

    for (int mask = 0; mask < num; ++mask) {
        Eigen::VectorXd d(n);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            const double sgn = ((mask >> i) & 1) ? -1.0 : 1.0;
            const double si = sigma(i);
            const double di = sgn * si;
            if (std::abs(di) <= eps) {
                ok = false;
                break;
            }
            d(i) = 1.0 / di;
        }
        if (!ok) continue;

        const Eigen::MatrixXd LambdaInv = W * d.asDiagonal() * W.transpose();
        Eigen::MatrixXd X = V * LambdaInv;

        if (enforce_det_pos && X.determinant() < 0.0) {
            continue;
        }

        const double score = (X.transpose() * V).trace();
        if (score > bestScore) {
            bestScore = score;
            bestX = X;
        }
    }

    if (bestX.size() == 0) {
        return procrustes_svd(V, enforce_det_pos);
    }
    return project_orthogonal(bestX, enforce_det_pos);
}

double orthogonal_geodesic_distance_deg(const Eigen::MatrixXd& X_true,
                                        const Eigen::MatrixXd& X_est) {
    if (X_true.rows() != X_true.cols() || X_est.rows() != X_est.cols()) {
        throw std::invalid_argument("orthogonal_geodesic_distance_deg: inputs must be square");
    }
    if (X_true.rows() != X_est.rows()) {
        throw std::invalid_argument("orthogonal_geodesic_distance_deg: dimension mismatch");
    }
    const int n = static_cast<int>(X_true.rows());
    if (n < 2) return 0.0;

    const Eigen::MatrixXd R = X_true.transpose() * X_est;
    Eigen::ComplexEigenSolver<Eigen::MatrixXd> ces(R, /* computeEigenvectors = */ false);
    if (ces.info() != Eigen::Success) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto evals = ces.eigenvalues();

    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        const std::complex<double> z(evals(i).real(), evals(i).imag());
        const double theta = std::atan2(z.imag(), z.real());
        sumsq += theta * theta;
    }
    const double d_rad = std::sqrt(0.5 * sumsq);
    const double kPi = std::acos(-1.0);
    return d_rad * (180.0 / kPi);
}

} // namespace guc_ls
