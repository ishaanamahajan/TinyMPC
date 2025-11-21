#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>

namespace {

using Vec2 = Eigen::Matrix<tinytype, 2, 1>;
using Mat22 = Eigen::Matrix<tinytype, 2, 2>;
using Row2 = Eigen::Matrix<tinytype, 1, 2>;

struct QPConstraint {
    Row2 a;
    tinytype b;
};

struct QPResult {
    Vec2 u;
    bool feasible = false;
};

bool satisfies(const Vec2& u,
               const std::vector<QPConstraint>& constraints,
               tinytype tol = tinytype(1e-6)) {
    for (const auto& c : constraints) {
        if (c.a.dot(u) > c.b + tol) {
            return false;
        }
    }
    return true;
}

bool solve_active_set(const Mat22& H,
                      const Vec2& f,
                      const std::vector<QPConstraint>& constraints,
                      const std::vector<int>& active,
                      Vec2* u_out,
                      tinytype tol = tinytype(1e-6)) {
    const int m = static_cast<int>(active.size());
    if (m == 0) {
        return false;
    }
    Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic> KKT(2 + m, 2 + m);
    KKT.setZero();
    KKT.block<2, 2>(0, 0) = H;
    for (int i = 0; i < m; ++i) {
        const auto& c = constraints[active[i]];
        KKT.block(0, 2 + i, 2, 1) = c.a.transpose();
        KKT.block(2 + i, 0, 1, 2) = c.a;
    }

    Eigen::Matrix<tinytype, Eigen::Dynamic, 1> rhs(2 + m);
    rhs.setZero();
    rhs.head<2>() = -f;
    for (int i = 0; i < m; ++i) {
        rhs(2 + i) = constraints[active[i]].b;
    }

    Eigen::FullPivLU<Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>> lu(KKT);
    if (!lu.isInvertible()) {
        return false;
    }
    Eigen::Matrix<tinytype, Eigen::Dynamic, 1> sol = lu.solve(rhs);
    Vec2 u = sol.head<2>();
    auto lambda = sol.tail(m);
    for (int i = 0; i < m; ++i) {
        if (lambda(i) < -tol) {
            return false;
        }
    }
    if (u_out) {
        *u_out = u;
    }
    return true;
}

QPResult solve_small_qp(const Mat22& H_in,
                        const Vec2& f_in,
                        const std::vector<QPConstraint>& constraints) {
    const tinytype tol = tinytype(1e-6);
    Mat22 H = (H_in + H_in.transpose()) * tinytype(0.5);
    H += tinytype(1e-6) * Mat22::Identity();
    Vec2 f = f_in;

    auto cost = [&](const Vec2& u) -> tinytype {
        return tinytype(0.5) * u.dot(H * u) + f.dot(u);
    };

    QPResult best;
    tinytype best_cost = std::numeric_limits<tinytype>::infinity();

    auto consider = [&](const Vec2& cand) {
        if (!std::isfinite(cand(0)) || !std::isfinite(cand(1))) {
            return;
        }
        if (!satisfies(cand, constraints, tol)) {
            return;
        }
        tinytype c = cost(cand);
        if (c < best_cost) {
            best.u = cand;
            best.feasible = true;
            best_cost = c;
        }
    };

    // Unconstrained optimum.
    Eigen::LDLT<Mat22> ldlt(H);
    if (ldlt.isPositive()) {
        Vec2 u_free = ldlt.solve(-f);
        consider(u_free);
    }

    // Single active constraint.
    for (int i = 0; i < static_cast<int>(constraints.size()); ++i) {
        Vec2 cand;
        if (solve_active_set(H, f, constraints, {i}, &cand, tol)) {
            consider(cand);
        }
    }

    // Two active constraints.
    for (int i = 0; i < static_cast<int>(constraints.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(constraints.size()); ++j) {
            Vec2 cand;
            if (solve_active_set(H, f, constraints, {i, j}, &cand, tol)) {
                consider(cand);
            }
        }
    }

    return best;
}

Vec2 clamp_to_box(const Vec2& u,
                  const Vec2& u_min,
                  const Vec2& u_max) {
    Vec2 clamped = u;
    for (int i = 0; i < 2; ++i) {
        clamped(i) = std::min(u_max(i), std::max(u_min(i), clamped(i)));
    }
    return clamped;
}

double signed_distance_point_disks(double x,
                                   double y,
                                   const std::vector<std::array<tinytype, 3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& d : disks) {
        const double dx = x - static_cast<double>(d[0]);
        const double dy = y - static_cast<double>(d[1]);
        const double rr = static_cast<double>(d[2]);
        const double sd = std::sqrt(dx * dx + dy * dy) - rr;
        if (sd < best) {
            best = sd;
        }
    }
    return best;
}

}  // namespace

extern "C" int main() {
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

    static constexpr int NX0 = 4;
    static constexpr int NU0 = 2;
    static constexpr int N = 45;

    Mat Ad(NX0, NX0);
    Ad << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;
    Mat Bd(NX0, NU0);
    Bd << 0.5, 0,
          0,   0.5,
          1,   0,
          0,   1;

    Vec x0(NX0);
    x0 << 6.0, -1.0, 0.0, 0.0;

    Vec x = x0;
    std::vector<Vec> Xdyn(N);
    Xdyn[0] = x;
    std::vector<Vec2> U(N - 1, Vec2::Zero());
    std::vector<tinytype> relax_hist(N - 1, tinytype(0));
    std::vector<tinytype> margin_hist(N - 1, tinytype(0));

    const Vec2 u_min = (Vec2() << -3.0, -3.0).finished();
    const Vec2 u_max = (Vec2() << 3.0, 3.0).finished();

    const Mat22 R = (Mat22() << 1.5, 0,
                                 0, 1.5).finished();
    const Mat22 Qgoal = tinytype(4.0) * Mat22::Identity();
    const Vec2 goal = (Vec2() << 0.0, 0.0).finished();

    const tinytype alpha0 = 2.0;
    const tinytype alpha1 = 3.0;
    const tinytype base_relax = tinytype(0.0);
    const int relax_trials = 1;
    const tinytype relax_growth = tinytype(1.0);

    const tinytype r_wall = tinytype(0.8);
    std::vector<std::array<tinytype, 3>> disks = {
        { tinytype(2.5), tinytype( 0.0), r_wall },
        { tinytype(2.5), tinytype( 1.2), r_wall },
        { tinytype(2.5), tinytype(-1.2), r_wall },
        { tinytype(3.8), tinytype( 1.2), r_wall },
        { tinytype(3.8), tinytype(-1.2), r_wall },
        { tinytype(5.0), tinytype( 1.2), r_wall },
        { tinytype(5.0), tinytype(-1.2), r_wall }
    };

    std::cout << "[CBF-U] Running nominal CBF baseline with "
              << disks.size() << " disks\n";

    for (int k = 0; k < N - 1; ++k) {
        Vec2 z = x.topRows(2);
        Vec2 v = x.bottomRows(2);

        Mat22 H = R;
        Vec2 f = Vec2::Zero();
        Vec2 r = (z + v) - goal;
        H += tinytype(0.125) * Qgoal;
        f += tinytype(0.5) * Qgoal * r;

        bool solved = false;
        Vec2 u = Vec2::Zero();
        tinytype used_relax = -1;
        tinytype min_margin = std::numeric_limits<tinytype>::infinity();
        std::vector<QPConstraint> constraints_used;

        tinytype relax = base_relax;
        for (int trial = 0; trial < relax_trials && !solved; ++trial) {
            std::vector<QPConstraint> constraints;
            constraints.reserve(disks.size() + 4);

            for (const auto& d : disks) {
                Vec2 diff;
                diff << x(0) - d[0], x(1) - d[1];
                tinytype h = diff.squaredNorm() - d[2] * d[2];
                tinytype rhs = tinytype(2.0) * v.squaredNorm()
                             + tinytype(2.0) * alpha1 * diff.dot(v)
                             + alpha0 * h;
                Row2 a;
                a << -tinytype(2.0) * diff(0), -tinytype(2.0) * diff(1);
                QPConstraint c;
                c.a = a;
                c.b = rhs + relax;
                constraints.push_back(c);
            }

            // Input bounds as linear constraints.
            {
                Row2 a;
                a << 1.0, 0.0;
                constraints.push_back({a, u_max(0)});
                a << -1.0, 0.0;
                constraints.push_back({a, -u_min(0)});
                a << 0.0, 1.0;
                constraints.push_back({a, u_max(1)});
                a << 0.0, -1.0;
                constraints.push_back({a, -u_min(1)});
            }

            QPResult res = solve_small_qp(H, f, constraints);
            if (res.feasible) {
                u = res.u;
                solved = true;
                used_relax = relax;
                constraints_used = constraints;
                for (const auto& c : constraints_used) {
                    tinytype margin = c.b - c.a.dot(u);
                    if (margin < min_margin) {
                        min_margin = margin;
                    }
                }
                break;
            }

            relax *= relax_growth;
        }

        if (!solved) {
            Eigen::LDLT<Mat22> ldlt(H);
            Vec2 u_free = Vec2::Zero();
            if (ldlt.isPositive()) {
                u_free = ldlt.solve(-f);
            }
            u = clamp_to_box(u_free, u_min, u_max);
            min_margin = tinytype(0);
        }

        U[k] = u;
        relax_hist[k] = used_relax;
        margin_hist[k] = min_margin;

        x = Ad * x + Bd * u;
        Xdyn[k + 1] = x;
    }

    const std::string csv_path = "../cbf_ushape_trajectory.csv";
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cout << "[CBF-U] Failed to open " << csv_path << "\n";
        return 1;
    }

    csv << "k,x1,x2,x3,x4,u1,u2,signed_dist,cbf_relax,cbf_margin\n";
    double min_sd = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        const Vec& xk = Xdyn[k];
        double sd = signed_distance_point_disks(xk(0), xk(1), disks);
        if (sd < min_sd) {
            min_sd = sd;
        }
        csv << k << "," << xk(0) << "," << xk(1) << "," << xk(2) << "," << xk(3);
        if (k < N - 1) {
            const Vec2& uk = U[k];
            csv << "," << uk(0) << "," << uk(1) << "," << sd << "," << relax_hist[k] << "," << margin_hist[k] << "\n";
        } else {
            csv << ",0,0," << sd << ",0,0\n";
        }
    }
    csv.close();

    std::cout << "[CBF-U] Exported cbf_ushape_trajectory.csv\n";
    std::cout << "[CBF-U] Min signed distance to U-shape: " << min_sd << "\n";

    return 0;
}


