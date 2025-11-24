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
constexpr int H_OBS = 18;
const tinytype PREDICTION_INFLATION = tinytype(0.01);

struct QPConstraint {
    Row2 a;
    tinytype b;
};

struct QPResult {
    Vec2 u;
    bool feasible = false;
};

struct MovingDisk {
    tinytype cx0;
    tinytype cy0;
    tinytype vx;
    tinytype vy;
    tinytype radius;
    tinytype wobble_x;
    tinytype wobble_x_freq;
    tinytype wobble_x_phase;
    tinytype wobble_y;
    tinytype wobble_y_freq;
    tinytype wobble_y_phase;

    std::array<tinytype,3> disk_at_time(tinytype t) const {
        double td = static_cast<double>(t);
        double sin_arg = static_cast<double>(wobble_x_freq) * td + static_cast<double>(wobble_x_phase);
        double cos_arg = static_cast<double>(wobble_y_freq) * td + static_cast<double>(wobble_y_phase);
        tinytype cx = cx0 + vx * t + wobble_x * tinytype(std::sin(sin_arg));
        tinytype cy = cy0 + vy * t + wobble_y * tinytype(std::cos(cos_arg));
        return {cx, cy, radius};
    }
};

struct DynamicObstacles {
    std::vector<MovingDisk> agents;
    tinytype dt = tinytype(1.0);

    std::vector<std::array<tinytype,3>> disks_at_step(int step) const {
        tinytype t = dt * tinytype(step);
        std::vector<std::array<tinytype,3>> disks;
        disks.reserve(agents.size());
        for (const auto& agent : agents) {
            disks.push_back(agent.disk_at_time(t));
        }
        return disks;
    }
};

std::vector<std::vector<std::array<tinytype,3>>> build_prediction(
    const DynamicObstacles& obs,
    int step,
    int horizon,
    tinytype inflation_rate) {
    std::vector<std::vector<std::array<tinytype,3>>> per_stage;
    per_stage.reserve(horizon);
    for (int h = 0; h < horizon; ++h) {
        auto disks = obs.disks_at_step(step + h);
        tinytype inflate = inflation_rate * tinytype(std::sqrt(static_cast<double>(h)));
        for (auto& d : disks) {
            d[2] += inflate;
        }
        per_stage.push_back(std::move(disks));
    }
    return per_stage;
}

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
    KKT.block<2,2>(0,0) = H;
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
        if (lambda(i) < -tol) return false;
    }
    if (u_out) *u_out = u;
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
        if (!std::isfinite(static_cast<double>(cand(0))) ||
            !std::isfinite(static_cast<double>(cand(1)))) return;
        if (!satisfies(cand, constraints, tol)) return;
        tinytype c = cost(cand);
        if (c < best_cost) {
            best.u = cand;
            best.feasible = true;
            best_cost = c;
        }
    };

    Eigen::LDLT<Mat22> ldlt(H);
    if (ldlt.isPositive()) {
        Vec2 u_free = ldlt.solve(-f);
        consider(u_free);
    }
    for (int i = 0; i < static_cast<int>(constraints.size()); ++i) {
        Vec2 cand;
        if (solve_active_set(H, f, constraints, {i}, &cand, tol)) {
            consider(cand);
        }
    }
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
                                   const std::vector<std::array<tinytype,3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& d : disks) {
        double dx = x - static_cast<double>(d[0]);
        double dy = y - static_cast<double>(d[1]);
        double r  = static_cast<double>(d[2]);
        double sd = std::sqrt(dx*dx + dy*dy) - r;
        if (sd < best) best = sd;
    }
    return best;
}

double signed_distance_segment_disks(
    const Eigen::Matrix<tinytype, Eigen::Dynamic, 1>& p0,
    const Eigen::Matrix<tinytype, Eigen::Dynamic, 1>& p1,
    const std::vector<std::array<tinytype,3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    double x0 = static_cast<double>(p0(0));
    double y0 = static_cast<double>(p0(1));
    double x1 = static_cast<double>(p1(0));
    double y1 = static_cast<double>(p1(1));
    double dx = x1 - x0;
    double dy = y1 - y0;
    double len2 = dx*dx + dy*dy;
    for (const auto& d : disks) {
        double cx = static_cast<double>(d[0]);
        double cy = static_cast<double>(d[1]);
        double r  = static_cast<double>(d[2]);
        double t = 0.0;
        if (len2 > 0.0) {
            t = ((cx - x0)*dx + (cy - y0)*dy) / len2;
            t = std::max(0.0, std::min(1.0, t));
        }
        double px = x0 + t*dx;
        double py = y0 + t*dy;
        double sd = std::sqrt((px - cx)*(px - cx) + (py - cy)*(py - cy)) - r;
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
    x0 << -10.0, 0.0, 0.0, 0.0;
    Vec x = x0;

    DynamicObstacles obstacles;
    obstacles.dt = tinytype(1.0);
    obstacles.agents = {
        { tinytype(-7.0), tinytype( 0.0), tinytype(0.0), tinytype(0.0),
          tinytype(1.0), tinytype(0.02), tinytype(0.3), tinytype(0.0),
          tinytype(0.02), tinytype(0.4), tinytype(0.0) },
        { tinytype(-4.2), tinytype( 1.7), tinytype(0.02), tinytype(-0.08),
          tinytype(0.9), tinytype(0.05), tinytype(0.4), tinytype(0.3),
          tinytype(0.06), tinytype(0.7), tinytype(0.2) },
        { tinytype(-3.8), tinytype(-1.7), tinytype(0.015), tinytype(0.08),
          tinytype(0.9), tinytype(0.05), tinytype(0.4), tinytype(0.9),
          tinytype(0.06), tinytype(0.7), tinytype(0.5) }
    };

    const int total_steps = 90;
    std::vector<Vec> Xdyn(total_steps + 1);
    std::vector<Vec2> U(total_steps, Vec2::Zero());
    std::vector<tinytype> relax_hist(total_steps, tinytype(0));
    std::vector<tinytype> margin_hist(total_steps, tinytype(0));
    Xdyn[0] = x;

    const Vec2 u_min = (Vec2() << -3.0, -3.0).finished();
    const Vec2 u_max = (Vec2() << 3.0, 3.0).finished();

    const Mat22 R = (Mat22() << 1.5, 0,
                                 0, 1.5).finished();
    const Mat22 Qgoal = tinytype(4.0) * Mat22::Identity();
    const Vec2 goal = (Vec2() << 0.0, 0.0).finished();

    const tinytype alpha0 = tinytype(2.0);
    const tinytype alpha1 = tinytype(3.0);
    const tinytype base_relax = tinytype(0.0);
    const int relax_trials = 1;
    const tinytype relax_growth = tinytype(1.0);

    std::ofstream csv("../cbf_dynamic_tracking.csv");
    if (!csv.is_open()) {
        std::cout << "[CBF-DYN] Failed to open cbf_dynamic_tracking.csv\n";
        return 1;
    }
    csv << "k,x1,x2,x3,x4,u1,u2,signed_dist,seg_signed_dist,cbf_relax,cbf_margin\n";
    std::ofstream csv_obs("../cbf_dynamic_obstacles.csv");
    if (csv_obs.is_open()) {
        csv_obs << "k,disk,cx,cy,r\n";
    }

    const tinytype goal_pos_tol = tinytype(0.15);
    const tinytype goal_vel_tol = tinytype(0.05);
    auto log_obstacles = [&](int step) {
        if (!csv_obs.is_open()) return;
        auto disks = obstacles.disks_at_step(step);
        for (size_t j = 0; j < disks.size(); ++j) {
            csv_obs << step << "," << j << ","
                    << disks[j][0] << "," << disks[j][1] << "," << disks[j][2] << "\n";
        }
    };

    auto goal_reached = [&](const Vec& state) -> bool {
        tinytype pos_norm = state.topRows(2).norm();
        tinytype vel_norm = state.bottomRows(2).norm();
        return (pos_norm < goal_pos_tol) && (vel_norm < goal_vel_tol);
    };

    auto disks0 = obstacles.disks_at_step(0);
    double sd0 = signed_distance_point_disks(x(0), x(1), disks0);
    csv << 0 << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3)
        << ",0,0," << sd0 << "," << sd0 << ",0,0\n";
    log_obstacles(0);
    double min_sd = sd0;
    Vec prev_state = x;

    for (int k = 0; k < total_steps; ++k) {
        auto prediction = build_prediction(obstacles, k, H_OBS, PREDICTION_INFLATION);
        const auto& disks = prediction.empty()
            ? obstacles.disks_at_step(k)
            : prediction.front();
        Vec2 z = x.topRows(2);
        Vec2 v = x.bottomRows(2);

        Mat22 H = R + tinytype(0.125) * Qgoal;
        Vec2 f = tinytype(0.5) * Qgoal * ((z + v) - goal);

        bool solved = false;
        Vec2 u = Vec2::Zero();
        tinytype used_relax = -1;
        tinytype min_margin = std::numeric_limits<tinytype>::infinity();

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
                constraints.push_back({a, rhs + relax});
            }

            Row2 a;
            a << 1.0, 0.0;  constraints.push_back({a, u_max(0)});
            a << -1.0, 0.0; constraints.push_back({a, -u_min(0)});
            a << 0.0, 1.0;  constraints.push_back({a, u_max(1)});
            a << 0.0, -1.0; constraints.push_back({a, -u_min(1)});

            QPResult cand = solve_small_qp(H, f, constraints);
            if (cand.feasible) {
                u = cand.u;
                solved = true;
                used_relax = relax;
                for (const auto& c : constraints) {
                    tinytype margin = c.b - c.a.dot(u);
                    if (margin < min_margin) min_margin = margin;
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
            used_relax = tinytype(0);
        }

        U[k] = u;
        relax_hist[k] = used_relax;
        margin_hist[k] = min_margin;

        prev_state = x;
        x = Ad * x + Bd * u;
        Xdyn[k + 1] = x;

        int step_idx = k + 1;
        log_obstacles(step_idx);
        auto disks_next = obstacles.disks_at_step(step_idx);
        double sd_point = signed_distance_point_disks(x(0), x(1), disks_next);
        double sd_segment = signed_distance_segment_disks(prev_state, x, disks_next);
        if (sd_segment < min_sd) min_sd = sd_segment;

        csv << step_idx << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3)
            << "," << u(0) << "," << u(1) << "," << sd_point << "," << sd_segment
            << "," << used_relax << "," << min_margin << "\n";

        if (goal_reached(x)) {
            std::cout << "[CBF-DYN] Goal reached at step " << step_idx
                      << " (pos_norm=" << x.topRows(2).norm()
                      << ", vel_norm=" << x.bottomRows(2).norm() << ")\n";
            break;
        }
    }

    csv.close();
    if (csv_obs.is_open()) csv_obs.close();

    std::cout << "[CBF-DYN] Exported cbf_dynamic_tracking.csv\n";
    std::cout << "[CBF-DYN] Min signed distance: " << min_sd << "\n";
    return 0;
}


