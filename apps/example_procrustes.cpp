#include <guc_ls/stiefel_ls.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <random>

static Eigen::MatrixXd random_orthogonal(int n, std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = nd(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
    return Q;
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);

    const int n = 3;
    Eigen::MatrixXd X_true = random_orthogonal(n, rng);

    const double sigma = 1e-2;
    Eigen::MatrixXd V = X_true;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            V(i, j) += sigma * nd(rng);

    Eigen::MatrixXd X_hat = guc_ls::solve_procrustes(V);

    std::cout << "=== Example: orthogonal Procrustes ===\n";
    std::cout << "||X_hat - X_true||_F: " << (X_hat - X_true).norm() << "\n";
    std::cout << "||X_hat^T X_hat - I||_F: " << (X_hat.transpose()*X_hat - Eigen::MatrixXd::Identity(n,n)).norm() << "\n";
    return 0;
}
