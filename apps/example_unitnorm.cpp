#include <guc_ls/unit_norm_ls.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <random>

static Eigen::VectorXd random_unit_vector(int n, std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = nd(rng);
    v.normalize();
    return v;
}

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);

    const int n = 3;
    Eigen::MatrixXd B(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B(i, j) = nd(rng);

    Eigen::VectorXd u_true = random_unit_vector(n, rng);

    const double sigma = 1e-2;
    Eigen::VectorXd noise(n);
    for (int i = 0; i < n; ++i) noise(i) = sigma * nd(rng);

    Eigen::VectorXd g = B * u_true + noise;

    guc_ls::UnitNormResult r = guc_ls::solve_unit_norm_ls(B, g);

    const double cosang = std::max(-1.0, std::min(1.0, r.u.dot(u_true)));
    const double ang_deg = std::acos(cosang) * 180.0 / M_PI;

    std::cout << "=== Example: unit-norm constrained LS (vector) ===\n";
    std::cout << "lambda*: " << static_cast<double>(r.lambda) << "\n";
    std::cout << "cost   : " << static_cast<double>(r.cost) << "\n";
    std::cout << "angle error [deg]: " << ang_deg << "\n";
    return 0;
}
