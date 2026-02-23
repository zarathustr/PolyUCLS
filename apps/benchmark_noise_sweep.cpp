#include "guc_ls/procrustes.hpp"
#include "guc_ls/stiefel_ls.hpp"
#include "guc_ls/unit_norm_ls.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Args {
    int mc = 200;

    // Vector benchmark sizes: B is N x n, u is n x 1.
    // We set N = max(m_min, m_factor * n).
    int m_min = 80;
    int m_factor = 4;

    // Matrix benchmark sizes: A is mA x n, X is n x n.
    // We set mA = max(mA_min, mA_factor * n).
    int mA_min = 30;
    int mA_factor = 1;

    // Default requested dimensions.
    std::vector<int> unit_dims = {3, 50, 200};
    std::vector<int> proc_dims = {4, 30, 100};

    std::vector<double> noise = {0.0, 0.002, 0.005, 0.01, 0.02};
    std::string out = "guc_ls_noise_sweep.csv";

    // Problem families to run. Tokens:
    //   unitnorm, procrustes
    // and legacy tokens:
    //   unitnorm3, unitnorm50, unitnorm200, procrustes4, ...
    std::vector<std::string> problems = {"unitnorm", "procrustes"};

    unsigned seed = 1;

    // For matrix case: if true, enforce det(X)=+1 (SO(n)).
    bool enforce_det_pos = true;

    // Include Stiefel-Newton only for the small n=4 matrix case.
    bool include_stiefel_newton_small = true;
};

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> out;
    for (const auto& tok : split_csv(s)) {
        out.push_back(std::stod(tok));
    }
    return out;
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    for (const auto& tok : split_csv(s)) {
        out.push_back(std::stoi(tok));
    }
    return out;
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

static bool parse_dim_suffix(const std::string& token, const std::string& prefix, int* dim_out) {
    if (!starts_with(token, prefix)) return false;
    const std::string suf = token.substr(prefix.size());
    if (suf.empty()) return false;
    for (char c : suf) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    *dim_out = std::stoi(suf);
    return true;
}

static void sort_unique(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_val = [&](const std::string& k) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + k);
            }
            return std::string(argv[++i]);
        };

        if (key == "--mc") {
            a.mc = std::stoi(need_val(key));
        } else if (key == "--noise") {
            a.noise = parse_double_list(need_val(key));
        } else if (key == "--out") {
            a.out = need_val(key);
        } else if (key == "--problem") {
            a.problems = split_csv(need_val(key));
        } else if (key == "--seed") {
            a.seed = static_cast<unsigned>(std::stoul(need_val(key)));
        } else if (key == "--det") {
            const std::string v = need_val(key);
            a.enforce_det_pos = (v == "1" || v == "true" || v == "on");
        } else if (key == "--unit_n") {
            a.unit_dims = parse_int_list(need_val(key));
            sort_unique(a.unit_dims);
        } else if (key == "--proc_n") {
            a.proc_dims = parse_int_list(need_val(key));
            sort_unique(a.proc_dims);
        } else if (key == "--m") {
            // Backwards compatible: minimum rows for vector benchmark.
            a.m_min = std::stoi(need_val(key));
        } else if (key == "--m_factor") {
            a.m_factor = std::stoi(need_val(key));
        } else if (key == "--mA") {
            // Backwards compatible: minimum rows for matrix benchmark.
            a.mA_min = std::stoi(need_val(key));
        } else if (key == "--mA_factor") {
            a.mA_factor = std::stoi(need_val(key));
        } else if (key == "--no_stiefel") {
            const std::string v = need_val(key);
            bool off = (v == "1" || v == "true" || v == "on");
            a.include_stiefel_newton_small = !off;
        } else if (key == "-h" || key == "--help") {
            std::cout
                << "benchmark_noise_sweep options:\n"
                << "  --problem unitnorm,procrustes (default: both)\n"
                << "           legacy: unitnorm3,procrustes4,... also accepted\n"
                << "  --unit_n 3,50,200               vector dimensions (default: 3,50,200)\n"
                << "  --proc_n 4,30,100               matrix dimensions (default: 4,30,100)\n"
                << "  --mc 200                        Monte Carlo trials\n"
                << "  --noise 0,0.002,0.005,0.01      noise levels (comma-separated)\n"
                << "  --out results.csv               output CSV\n"
                << "  --m 80                          min rows for vector benchmark\n"
                << "  --m_factor 4                    rows N = max(m, m_factor*n)\n"
                << "  --mA 30                         min rows for matrix benchmark\n"
                << "  --mA_factor 1                   rows mA = max(mA, mA_factor*n)\n"
                << "  --det true|false                enforce det(X)=+1 for matrix case (default true)\n"
                << "  --no_stiefel true|false         disable StiefelNewton (only used for n=4)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (a.mc <= 0) throw std::runtime_error("--mc must be positive");
    if (a.m_min <= 0) throw std::runtime_error("--m must be positive");
    if (a.m_factor <= 0) throw std::runtime_error("--m_factor must be positive");
    if (a.mA_min <= 0) throw std::runtime_error("--mA must be positive");
    if (a.mA_factor <= 0) throw std::runtime_error("--mA_factor must be positive");
    if (a.unit_dims.empty()) throw std::runtime_error("--unit_n must contain at least one dimension");
    if (a.proc_dims.empty()) throw std::runtime_error("--proc_n must contain at least one dimension");
    return a;
}

static double clamp01(double x) {
    if (x < -1.0) return -1.0;
    if (x > 1.0) return 1.0;
    return x;
}

static Eigen::MatrixXd randn_mat(int r, int c, std::mt19937& rng, double sigma = 1.0) {
    std::normal_distribution<double> nd(0.0, sigma);
    Eigen::MatrixXd M(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            M(i, j) = nd(rng);
        }
    }
    return M;
}

static Eigen::VectorXd randn_vec(int n, std::mt19937& rng, double sigma = 1.0) {
    std::normal_distribution<double> nd(0.0, sigma);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = nd(rng);
    return v;
}

static Eigen::MatrixXd random_orthogonal(int n, std::mt19937& rng, bool det_pos) {
    Eigen::MatrixXd M = randn_mat(n, n, rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
    if (det_pos && Q.determinant() < 0.0) {
        Q.col(0) *= -1.0;
    }
    return Q;
}

// Simple Newton solver on the (u,lambda) KKT system (vector-case baseline).
static Eigen::VectorXd unitnorm_kkt_newton(const Eigen::MatrixXd& G,
                                          const Eigen::VectorXd& v,
                                          int max_iters = 50,
                                          double tol = 1e-12) {
    const int n = static_cast<int>(v.size());
    Eigen::VectorXd u = v;
    if (u.norm() > 0) u.normalize();
    double lambda = 0.0;

    for (int it = 0; it < max_iters; ++it) {
        const double f1 = u.squaredNorm() - 1.0;
        const Eigen::VectorXd f2 = (G - lambda * Eigen::MatrixXd::Identity(n, n)) * u - v;
        const double fn = std::sqrt(f1 * f1 + f2.squaredNorm());
        if (fn < tol) break;

        Eigen::MatrixXd J(n + 1, n + 1);
        J.setZero();
        J(0, 0) = 0.0;
        J.block(0, 1, 1, n) = 2.0 * u.transpose();
        J.block(1, 0, n, 1) = -u;
        J.block(1, 1, n, n) = (G - lambda * Eigen::MatrixXd::Identity(n, n));

        Eigen::VectorXd F(n + 1);
        F(0) = f1;
        F.segment(1, n) = f2;

        Eigen::VectorXd dx = J.fullPivLu().solve(F);
        if (dx.size() != n + 1) break;

        double mu = 1.0;
        if (dx.norm() > 1.0) {
            mu = 1.0 / dx.norm();
        }
        lambda -= mu * dx(0);
        u -= mu * dx.segment(1, n);
    }
    if (u.norm() > 0) u.normalize();
    return u;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        std::cerr << "Run with --help for usage.\n";
        return 2;
    }

    // Resolve which families and which dimensions to run.
    bool do_unit = false;
    bool do_proc = false;

    std::vector<int> unit_dims = args.unit_dims;
    std::vector<int> proc_dims = args.proc_dims;

    std::vector<int> unit_dims_from_problem;
    std::vector<int> proc_dims_from_problem;

    for (const auto& tok : args.problems) {
        if (tok == "unitnorm") {
            do_unit = true;
        } else if (tok == "procrustes") {
            do_proc = true;
        } else {
            int dim = 0;
            if (parse_dim_suffix(tok, "unitnorm", &dim)) {
                do_unit = true;
                unit_dims_from_problem.push_back(dim);
            } else if (parse_dim_suffix(tok, "procrustes", &dim)) {
                do_proc = true;
                proc_dims_from_problem.push_back(dim);
            } else {
                throw std::runtime_error("Unknown problem token: " + tok);
            }
        }
    }

    if (!unit_dims_from_problem.empty()) {
        unit_dims = unit_dims_from_problem;
        sort_unique(unit_dims);
    }
    if (!proc_dims_from_problem.empty()) {
        proc_dims = proc_dims_from_problem;
        sort_unique(proc_dims);
    }

    if (!do_unit && !do_proc) {
        std::cerr << "No valid problems selected. Use --problem unitnorm,procrustes\n";
        return 2;
    }

    std::mt19937 rng(args.seed);
    const double kPi = std::acos(-1.0);

    std::ofstream ofs(args.out);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << args.out << "\n";
        return 1;
    }
    ofs << "problem,solver,noise,trial,rot_err_deg,trans_err,orth_err\n";
    ofs << std::setprecision(17);

    for (double sigma : args.noise) {
        for (int t = 0; t < args.mc; ++t) {
            if (do_unit) {
                for (int n : unit_dims) {
                    const int N = std::max(args.m_min, args.m_factor * n);
                    Eigen::MatrixXd Bm = randn_mat(N, n, rng);
                    Eigen::VectorXd u_true = randn_vec(n, rng);
                    u_true.normalize();
                    Eigen::VectorXd g = Bm * u_true + sigma * randn_vec(N, rng);

                    const std::string prob_name = std::string("unitnorm") + std::to_string(n);

                    // Method 1: normalized LS baseline
                    Eigen::VectorXd u_ls = Bm.colPivHouseholderQr().solve(g);
                    if (u_ls.norm() > 0) u_ls.normalize();
                    const double ang_ls = std::acos(clamp01(u_true.dot(u_ls))) * (180.0 / kPi);
                    const double cost_ls = (Bm * u_ls - g).squaredNorm();
                    const double orth_ls = std::abs(u_ls.squaredNorm() - 1.0);
                    ofs << prob_name << ",NormLS," << sigma << ',' << t << ',' << ang_ls << ',' << cost_ls << ','
                        << orth_ls << "\n";

                    // Method 2: KKT Newton
                    const Eigen::MatrixXd G = Bm.transpose() * Bm;
                    const Eigen::VectorXd v = Bm.transpose() * g;
                    Eigen::VectorXd u_newton = unitnorm_kkt_newton(G, v);
                    const double ang_newton = std::acos(clamp01(u_true.dot(u_newton))) * (180.0 / kPi);
                    const double cost_newton = (Bm * u_newton - g).squaredNorm();
                    const double orth_newton = std::abs(u_newton.squaredNorm() - 1.0);
                    ofs << prob_name << ",KKTNewton," << sigma << ',' << t << ',' << ang_newton << ','
                        << cost_newton << ',' << orth_newton << "\n";

                    // Method 3: polynomial elimination (proposed)
                    guc_ls::UnitNormResult r = guc_ls::solve_unit_norm_ls(Bm, g);
                    const Eigen::VectorXd u_poly = r.u;
                    const double ang_poly = std::acos(clamp01(u_true.dot(u_poly))) * (180.0 / kPi);
                    const double cost_poly = static_cast<double>(r.cost);
                    const double orth_poly = std::abs(u_poly.squaredNorm() - 1.0);
                    ofs << prob_name << ",PolyRoot," << sigma << ',' << t << ',' << ang_poly << ',' << cost_poly
                        << ',' << orth_poly << "\n";
                }
            }

            if (do_proc) {
                for (int n : proc_dims) {
                    const int mA = std::max(args.mA_min, args.mA_factor * n);
                    Eigen::MatrixXd A = randn_mat(mA, n, rng);
                    Eigen::MatrixXd X_true = random_orthogonal(n, rng, args.enforce_det_pos);
                    Eigen::MatrixXd B = A * X_true + sigma * randn_mat(mA, n, rng);
                    const Eigen::MatrixXd V = A.transpose() * B;

                    const std::string prob_name = std::string("procrustes") + std::to_string(n);

                    auto eval_and_write = [&](const std::string& solver, const Eigen::MatrixXd& Xhat) {
                        const double rot_err = guc_ls::orthogonal_geodesic_distance_deg(X_true, Xhat);
                        const double res = (A * Xhat - B).squaredNorm();
                        const double orth = (Xhat.transpose() * Xhat - Eigen::MatrixXd::Identity(n, n)).norm();
                        ofs << prob_name << ',' << solver << ',' << sigma << ',' << t << ',' << rot_err << ',' << res
                            << ',' << orth << "\n";
                    };

                    // Scalable closed-form representatives
                    const Eigen::MatrixXd X_svd = guc_ls::procrustes_svd(V, args.enforce_det_pos);
                    eval_and_write("ProcrustesSVD", X_svd);

                    const Eigen::MatrixXd X_polar = guc_ls::procrustes_polar_sqrt(V, args.enforce_det_pos);
                    eval_and_write("PolarSqrt", X_polar);

                    const Eigen::MatrixXd X_lsp = guc_ls::ls_then_project(A, B, args.enforce_det_pos);
                    eval_and_write("LSProject", X_lsp);

                    // Polynomial multi-solution representative (only feasible for n=4).
                    if (n == 4) {
                        const Eigen::MatrixXd X_gb = guc_ls::procrustes_gb_enum(V, args.enforce_det_pos);
                        eval_and_write("GBEnum", X_gb);

                        if (args.include_stiefel_newton_small) {
                            try {
                                guc_ls::StiefelOptions opt;
                                opt.max_iters = 50;
                                opt.tol = 1e-12;
                                opt.step_damping = 1.0;
                                guc_ls::StiefelResult st = guc_ls::solve_stiefel_ls(A, B, opt);
                                Eigen::MatrixXd X_st = guc_ls::project_orthogonal(st.X, args.enforce_det_pos);
                                eval_and_write("StiefelNewton", X_st);
                            } catch (...) {
                            }
                        }
                    }
                }
            }
        }
        std::cout << "Done sigma=" << sigma << " (" << args.mc << " trials)\n";
    }

    std::cout << "Wrote: " << args.out << "\n";
    return 0;
}
