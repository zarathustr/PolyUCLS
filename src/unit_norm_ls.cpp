#include "guc_ls/unit_norm_ls.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace guc_ls {

// ---- PolyRoot (robust) implementation notes ----
//
// We solve the KKT conditions
//   (G - λ I) u = v,   ||u|| = 1
// by diagonalizing G = Q diag(p) Q^T and working in the eigen-basis:
//   (p_i - λ) y_i = h_i,  where y = Q^T u and h = Q^T v.
//
// If λ is not an eigenvalue of G, then y_i = h_i/(p_i - λ) and the
// constraint becomes the secular equation
//   φ(λ) := Σ_i h_i^2/(p_i - λ)^2 - 1 = 0.
//
// The older implementation enumerated polynomial roots and skipped values
// near poles, which breaks the classical "hard case": λ equals an eigenvalue
// and the corresponding h-components are (near) zero, making (G-λI) singular
// but still solvable with a free nullspace component.
//
// The implementation below handles BOTH:
//   (a) regular case: λ < p_min (unique root found by bracketing + bisection)
//   (b) hard/singular case: λ == p_min with h in the min-eigenspace ≈ 0
//       and the remaining fixed components having norm <= 1.

static long double secular_phi(const Eigen::VectorXd& p,
                               const Eigen::VectorXd& h,
                               long double lambda) {
    const int n = static_cast<int>(p.size());
    long double s = -1.0L;
    for (int i = 0; i < n; ++i) {
        const long double pi = static_cast<long double>(p(i));
        const long double hi = static_cast<long double>(h(i));
        const long double d = pi - lambda;
        s += (hi * hi) / (d * d);
    }
    return s;
}

static long double find_lambda_regular(const Eigen::VectorXd& p,
                                       const Eigen::VectorXd& h,
                                       long double p_min,
                                       long double pole_tol,
                                       int max_iters = 200,
                                       long double rel_tol = 1e-14L) {
    // Find the unique root of φ(λ)=0 on (-∞, p_min) in the regular case.
    // Bracket [lo, hi] where φ(lo) < 0 and φ(hi) > 0.

    // Choose hi just below p_min.
    long double step = std::max(1e-9L * (1.0L + std::fabsl(p_min)), pole_tol);
    long double hi = p_min - step;

    // Ensure φ(hi) > 0 (shrink step towards the pole if needed).
    for (int k = 0; k < 60; ++k) {
        const long double fhi = secular_phi(p, h, hi);
        if (fhi > 0.0L) break;
        step *= 0.1L;
        hi = p_min - step;
    }

    // Move lo left until φ(lo) < 0.
    long double lo = hi - (1.0L + std::fabsl(hi));
    long double flo = secular_phi(p, h, lo);
    for (int k = 0; k < 80 && flo > 0.0L; ++k) {
        const long double span = (hi - lo);
        lo -= 2.0L * (1.0L + std::fabsl(span));
        flo = secular_phi(p, h, lo);
    }
    if (!(flo < 0.0L)) {
        // As a last resort, take an extremely negative lo.
        lo = p_min - 1e6L * (1.0L + std::fabsl(p_min));
        flo = secular_phi(p, h, lo);
        if (!(flo < 0.0L)) {
            throw std::runtime_error("Failed to bracket secular equation root");
        }
    }

    // Bisection (monotone on (-∞, p_min) for the regular case).
    long double lam = 0.5L * (lo + hi);
    for (int it = 0; it < max_iters; ++it) {
        lam = 0.5L * (lo + hi);
        const long double f = secular_phi(p, h, lam);
        if (std::fabsl(f) <= 1e-16L) break;
        if (f > 0.0L) {
            hi = lam;
        } else {
            lo = lam;
        }
        const long double width = hi - lo;
        const long double scale = 1.0L + std::fabsl(lam);
        if (std::fabsl(width) <= rel_tol * scale) break;
    }
    return lam;
}

static Eigen::VectorXd recover_u_from_lambda(const Eigen::MatrixXd& Q,
                                             const Eigen::VectorXd& p,
                                             const Eigen::VectorXd& h,
                                             long double lambda,
                                             long double pole_tol,
                                             bool* used_nullspace_fill) {
    // Recover u = Q y from (p_i - λ) y_i = h_i.
    // If |p_i-λ| is tiny, we require h_i ≈ 0 and then fill a nullspace
    // component to satisfy ||y||=1.
    const int n = static_cast<int>(p.size());
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);

    std::vector<int> pole_idx;
    pole_idx.reserve(n);
    long double norm2_fixed = 0.0L;

    for (int i = 0; i < n; ++i) {
        const long double pi = static_cast<long double>(p(i));
        const long double hi = static_cast<long double>(h(i));
        const long double d = pi - lambda;
        const long double tol = pole_tol * (1.0L + std::fabsl(lambda));
        if (std::fabsl(d) <= tol) {
            pole_idx.push_back(i);
            // y_i left as 0 for now; must have h_i ≈ 0.
        } else {
            const long double yi = hi / d;
            y(i) = static_cast<double>(yi);
            norm2_fixed += yi * yi;
        }
    }

    // If there is a singular block, fill it if possible.
    if (!pole_idx.empty()) {
        if (used_nullspace_fill) *used_nullspace_fill = true;
        // Check feasibility: fixed part must not exceed 1.
        if (norm2_fixed > 1.0L + 1e-12L) {
            // Infeasible for this λ.
            return Eigen::VectorXd();
        }
        const long double rem2 = std::max(0.0L, 1.0L - norm2_fixed);
        const long double rem = std::sqrt(rem2);
        // Put the remaining norm into the first pole dimension.
        y(pole_idx.front()) = static_cast<double>(rem);
    } else {
        if (used_nullspace_fill) *used_nullspace_fill = false;
    }

    Eigen::VectorXd u = Q * y;
    const double nrm = u.norm();
    if (nrm > 0.0) u /= nrm;
    return u;
}

UnitNormResult solve_unit_norm_ls_normal(const Eigen::MatrixXd& G,
                                         const Eigen::VectorXd& v,
                                         const UnitNormOptions& opt) {
    if (G.rows() != G.cols()) throw std::invalid_argument("G must be square");
    if (G.rows() != v.size()) throw std::invalid_argument("G and v dimensions mismatch");

    const int n = static_cast<int>(v.size());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed");
    }
    const Eigen::VectorXd p = es.eigenvalues();
    const Eigen::MatrixXd Q = es.eigenvectors();
    const Eigen::VectorXd h = Q.transpose() * v;

    const long double p_min = static_cast<long double>(p.minCoeff());

    UnitNormResult out;
    out.cost = std::numeric_limits<long double>::infinity();

    // --- 1) Handle the (rare) trivial case v≈0: choose smallest-eigenvector direction.
    if (v.norm() < 1e-14) {
        Eigen::VectorXd u = Q.col(0);
        if (u.norm() > 0.0) u.normalize();
        out.u = u;
        out.lambda = p_min;
        out.cost = static_cast<long double>(u.dot(G * u) - 2.0 * v.dot(u));
        out.candidate_lambdas = {out.lambda};
        return out;
    }

    // --- 2) Check the "hard case": λ = p_min and h in the min-eigenspace is (near) zero.
    const long double eig_tol = opt.pole_tol * (1.0L + std::fabsl(p_min));
    bool hard_case = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabsl(static_cast<long double>(p(i)) - p_min) <= eig_tol) {
            if (std::fabsl(static_cast<long double>(h(i))) > 1e-12L) {
                hard_case = false;
                break;
            }
        }
    }

    if (hard_case) {
        // Compute the fixed components y_i = h_i/(p_i - p_min) for p_i > p_min.
        long double norm2_fixed = 0.0L;
        for (int i = 0; i < n; ++i) {
            const long double pi = static_cast<long double>(p(i));
            if (std::fabsl(pi - p_min) <= eig_tol) continue; // nullspace
            const long double hi = static_cast<long double>(h(i));
            const long double d = (pi - p_min);
            const long double yi = hi / d;
            norm2_fixed += yi * yi;
        }
        if (norm2_fixed <= 1.0L + 1e-12L) {
            // Feasible hard-case solution at λ=p_min.
            bool used_fill = false;
            Eigen::VectorXd u = recover_u_from_lambda(Q, p, h, p_min, opt.pole_tol, &used_fill);
            if (u.size() != 0) {
                const long double cost = static_cast<long double>(u.dot(G * u) - 2.0 * v.dot(u));
                out.u = u;
                out.lambda = p_min;
                out.cost = cost;
                out.candidate_lambdas = {out.lambda};
                return out;
            }
            // If recovery failed (numerical), fall through to regular root search.
        }
    }

    // --- 3) Regular case: solve secular equation in (-∞, p_min).
    const long double lam = find_lambda_regular(p, h, p_min, opt.pole_tol);
    bool used_fill = false;
    Eigen::VectorXd u = recover_u_from_lambda(Q, p, h, lam, opt.pole_tol, &used_fill);
    if (u.size() == 0) {
        // Fallback: normalize v (should almost never happen in regular cases).
        u = v;
        if (u.norm() > 0.0) u.normalize();
    }
    out.u = u;
    out.lambda = lam;
    out.cost = static_cast<long double>(u.dot(G * u) - 2.0 * v.dot(u));
    out.candidate_lambdas = {out.lambda};
    return out;
}

UnitNormResult solve_unit_norm_ls(const Eigen::MatrixXd& B,
                                  const Eigen::VectorXd& g,
                                  const UnitNormOptions& opt) {
    if (B.rows() != g.size()) throw std::invalid_argument("B and g dimensions mismatch");
    const Eigen::MatrixXd G = B.transpose() * B;
    const Eigen::VectorXd v = B.transpose() * g;
    UnitNormResult r = solve_unit_norm_ls_normal(G, v, opt);
    // Replace the cost with the true residual cost if desired (still comparable).
    const Eigen::VectorXd res = B * r.u - g;
    r.cost = static_cast<long double>(res.squaredNorm());
    return r;
}

Eigen::MatrixXd jacobian_u_wrt_v(const Eigen::MatrixXd& G,
                                 const Eigen::VectorXd& u,
                                 long double lambda) {
    const int n = static_cast<int>(u.size());
    Eigen::MatrixXd M = G - static_cast<double>(lambda) * Eigen::MatrixXd::Identity(n, n);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(M);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT factorization failed in jacobian_u_wrt_v");
    }
    const Eigen::MatrixXd Minv = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    const Eigen::VectorXd Mu = Minv * u;
    const double denom = u.dot(Mu);
    if (std::abs(denom) < 1e-16) {
        throw std::runtime_error("Denominator too small in jacobian_u_wrt_v");
    }
    const Eigen::RowVectorXd dldv = -(u.transpose() * Minv) / denom; // 1 x n
    const Eigen::MatrixXd dudv = Minv + Mu * dldv; // n x n
    return dudv;
}

Eigen::MatrixXd covariance_u(const Eigen::MatrixXd& G,
                             const Eigen::VectorXd& u,
                             long double lambda,
                             const Eigen::MatrixXd& Sigma_v) {
    const Eigen::MatrixXd J = jacobian_u_wrt_v(G, u, lambda);
    return J * Sigma_v * J.transpose();
}

} // namespace guc_ls
