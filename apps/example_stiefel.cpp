#include <guc_ls/stiefel_ls.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <random>

static Eigen::MatrixXd random_stiefel(int n, int l, std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd A(n, l);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < l; ++j)
            A(i, j) = nd(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, l);
    return Q;
}

int main() {
    std::mt19937 rng(123);
    std::normal_distribution<double> nd(0.0, 1.0);

    const int m = 10;
    const int n = 5;
    const int l = 2;

    Eigen::MatrixXd A(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = nd(rng);

    Eigen::MatrixXd X_true = random_stiefel(n, l, rng);

    const double sigma = 1e-2;
    Eigen::MatrixXd noise(m, l);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < l; ++j)
            noise(i, j) = sigma * nd(rng);

    Eigen::MatrixXd B = A * X_true + noise;

    guc_ls::StiefelResult r = guc_ls::solve_stiefel_ls(A, B);

    std::cout << "=== Example: Stiefel-constrained LS (matrix) ===\n";
    std::cout << "converged: " << (r.converged ? "yes" : "no") << "  iters: " << r.iters << "\n";
    std::cout << "cost: " << r.cost << "\n";
    std::cout << "||X_hat - X_true||_F: " << (r.X - X_true).norm() << "\n";
    std::cout << "||X_hat^T X_hat - I||_F: " << (r.X.transpose()*r.X - Eigen::MatrixXd::Identity(l,l)).norm() << "\n";
    return 0;
}
