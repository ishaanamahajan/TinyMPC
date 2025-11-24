#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>
#include "../src/tinympc/psd_support.hpp"

namespace {

constexpr int NX0 = 4;
constexpr int NU0 = 2;
constexpr int N   = 45;

using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

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

    std::array<tinytype, 3> disk_at_time(tinytype t) const {
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

    std::vector<std::array<tinytype,3>> horizon_disks(int step,
                                                      int horizon,
                                                      tinytype inflation_rate) const {
        std::vector<std::array<tinytype,3>> disks;
        disks.reserve(static_cast<std::size_t>(agents.size()) * static_cast<std::size_t>(horizon));
        for (int h = 0; h < horizon; ++h) {
            tinytype t = dt * tinytype(step + h);
            tinytype inflate = inflation_rate * tinytype(h);
            for (const auto& agent : agents) {
                auto d = agent.disk_at_time(t);
                d[2] += inflate;
                disks.push_back(d);
            }
        }
        return disks;
    }

    std::vector<std::vector<std::array<tinytype,3>>> horizon_disks_per_stage(
        int step,
        int horizon,
        tinytype inflation_rate) const {
        std::vector<std::vector<std::array<tinytype,3>>> per_stage;
        per_stage.reserve(horizon);
        for (int h = 0; h < horizon; ++h) {
            auto disks = disks_at_step(step + h);
            tinytype inflate = inflation_rate * tinytype(std::sqrt(static_cast<double>(h)));
            for (auto& d : disks) {
                d[2] += inflate;
            }
            per_stage.push_back(std::move(disks));
        }
        return per_stage;
    }
};

Vec build_lifted(const Vec& base_state) {
    const int nxL = NX0 + NX0 * NX0;
    Vec lifted(nxL);
    lifted.setZero();
    lifted.topRows(NX0) = base_state;
    Mat outer = base_state * base_state.transpose();
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            lifted(NX0 + j*NX0 + i) = outer(i, j);
        }
    }
    return lifted;
}

std::pair<Mat, Mat> build_diag_refs(TinySolver* solver) {
    const int nxL = solver->work->nx;
    const int nuL = solver->work->nu;
    Mat Xref = Mat::Zero(nxL, N);
    Mat Uref = Mat::Zero(nuL, N-1);
    const tinytype q_xx = tinytype(1.0);
    const tinytype r_uu = tinytype(10.0);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int baseUU = NU0 + nxu + nux;

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX0; ++i) {
            int idx = NX0 + i*NX0 + i;
            tinytype denom = solver->work->Q(idx);
            if (denom != tinytype(0)) {
                Xref(idx, k) = -q_xx / denom;
            }
        }
    }
    for (int k = 0; k < N-1; ++k) {
        for (int j = 0; j < NU0; ++j) {
            int idx = baseUU + j*NU0 + j;
            tinytype denom = solver->work->R(idx);
            if (denom != tinytype(0)) {
                Uref(idx, k) = -r_uu / denom;
            }
        }
    }
    return {Xref, Uref};
}

double signed_distance_point_disks(const Vec& x,
                                   const std::vector<std::array<tinytype,3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& d : disks) {
        double dx = static_cast<double>(x(0) - d[0]);
        double dy = static_cast<double>(x(1) - d[1]);
        double r  = static_cast<double>(d[2]);
        double sd = std::sqrt(dx*dx + dy*dy) - r;
        if (sd < best) {
            best = sd;
        }
    }
    return best;
}

double signed_distance_segment_disks(const Vec& p0,
                                     const Vec& p1,
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

double min_prediction_signed_distance(
    const Vec& x,
    const std::vector<std::vector<std::array<tinytype,3>>>& disks_per_stage) {
    if (disks_per_stage.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double best = std::numeric_limits<double>::infinity();
    for (const auto& stage : disks_per_stage) {
        double sd = signed_distance_point_disks(x, stage);
        if (sd < best) {
            best = sd;
        }
    }
    return best;
}

int clamp_index(int idx, int lo, int hi) {
    if (idx < lo) return lo;
    if (idx > hi) return hi;
    return idx;
}

enum class PlanMode {
    PSD,
    NOMINAL,
};

struct PlanCache {
    std::vector<Vec> states;
    std::vector<Vec> inputs;
    int start_step = 0;
    int last_iters = 0;
    PlanMode mode = PlanMode::PSD;
};

const char* plan_mode_str(PlanMode mode) {
    switch (mode) {
        case PlanMode::PSD: return "psd";
        case PlanMode::NOMINAL: return "nominal";
        default: return "unknown";
    }
}

void rollout_plan(const Mat& Ad,
                  const Mat& Bd,
                  const Vec& x_start,
                  TinySolver* solver,
                  PlanCache* cache) {
    if (!cache) return;
    cache->states.assign(N, Vec::Zero(NX0));
    cache->inputs.assign(N-1, Vec::Zero(NU0));
    Vec x = x_start;
    cache->states[0] = x;
    for (int k = 0; k < N-1; ++k) {
        Vec u = solver->solution->u.col(k).topRows(NU0);
        cache->inputs[k] = u;
        x = Ad * x + Bd * u;
        cache->states[k+1] = x;
    }
    cache->last_iters = solver->solution->iter;
}

void set_tracking_refs(TinySolver* solver,
                       const PlanCache& cache,
                       int current_step,
                       const Mat& stab_Xref,
                       const Mat& stab_Uref) {
    Mat Xref = stab_Xref;
    Mat Uref = stab_Uref;
    if (!cache.states.empty()) {
        const int max_idx = static_cast<int>(cache.states.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx);
            Xref.col(i).topRows(NX0) = cache.states[idx];
        }
    }
    if (!cache.inputs.empty()) {
        const int max_idx_u = static_cast<int>(cache.inputs.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N-1; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx_u);
            Uref.col(i).topRows(NU0) = cache.inputs[idx];
        }
    }
    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);
}

}  // namespace

extern "C" int main() {
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

    Mat A, B;
    tiny_build_lifted_from_base(Ad, Bd, A, B);
    const int nxL = A.rows();
    const int nuL = B.cols();

    Mat Q = Mat::Zero(nxL, nxL);
    Q(0,0) = 8.0; Q(1,1) = 8.0; Q(2,2) = 0.8; Q(3,3) = 0.8;
    Q.diagonal().segment(NX0, NX0*NX0).array() = tinytype(5e-3);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int nuu = NU0 * NU0;
    R.diagonal().head(NU0).array() = tinytype(1.5);
    R.diagonal().segment(NU0, nxu).array() = tinytype(6.0);
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(6.0);
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(250.0);

    Vec fdyn = Vec::Zero(nxL);
    const tinytype rho_base = tinytype(5.0);
    const tinytype rho_psd_penalty = tinytype(0.95);

    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    x_min.middleRows(NX0, NX0*NX0).setConstant(-1500.0);
    x_max.middleRows(NX0, NX0*NX0).setConstant( 1500.0);

    Mat u_min = Mat::Constant(nuL, N-1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N-1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    u_min.bottomRows(nxu + nux + nuu).setConstant(-120.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 120.0);

    Vec x0(NX0);
    x0 << -10.0, 0.0, 0.0, 0.0;

    DynamicObstacles obstacles;
    obstacles.dt = tinytype(1.0);
    obstacles.agents = {
        // Static blocker far upstream
        { tinytype(-7.0), tinytype( 0.0), tinytype(0.0), tinytype(0.0),
          tinytype(1.0), tinytype(0.02), tinytype(0.3), tinytype(0.0),
          tinytype(0.02), tinytype(0.4), tinytype(0.0) },
        // Upper mover near x ≈ -4 wobbling up/down
        { tinytype(-4.2), tinytype( 1.7), tinytype(0.02), tinytype(-0.08),
          tinytype(0.9), tinytype(0.05), tinytype(0.4), tinytype(0.3),
          tinytype(0.06), tinytype(0.7), tinytype(0.2) },
        // Lower mover mirroring the upper one
        { tinytype(-3.8), tinytype(-1.7), tinytype(0.015), tinytype(0.08),
          tinytype(0.9), tinytype(0.05), tinytype(0.4), tinytype(0.9),
          tinytype(0.06), tinytype(0.7), tinytype(0.5) }
    };

    TinySolver* solver_psd = nullptr;
    if (tiny_setup(&solver_psd, A, B, fdyn, Q, R, rho_base, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_psd->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_psd, x_min, x_max, u_min, u_max);
    tiny_enable_psd(solver_psd, NX0, NU0, rho_psd_penalty);
    auto diag_refs_psd = build_diag_refs(solver_psd);
    tiny_set_x_ref(solver_psd, diag_refs_psd.first);
    tiny_set_u_ref(solver_psd, diag_refs_psd.second);

    TinySolver* solver_track = nullptr;
    if (tiny_setup(&solver_track, A, B, fdyn, Q, R, rho_base, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_track->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_track, x_min, x_max, u_min, u_max);
    auto diag_refs_track = build_diag_refs(solver_track);

    PlanCache plan;
    Vec x_track = x0;

    const int total_steps = 90;
    const int replan_stride = 5;
    const int horizon_guard = 5;
    const double psd_on_distance = 2.5;
    const double psd_off_distance = 2.5;
    bool psd_constraints_active = false;

    std::ofstream csv_plan("../psd_dynamic_plan_log.csv");
    if (csv_plan.is_open()) {
        csv_plan << "replan_step,plan_type,iter,num_disks,min_sd_seed,min_sd_prediction\n";
    }
    std::ofstream csv_track("../psd_dynamic_tracking.csv");
    if (csv_track.is_open()) {
        csv_track << "k,x1,x2,x3,x4,u1,u2,signed_dist,seg_signed_dist,plan_age,solver_iter\n";
    }
    std::ofstream csv_obstacles("../psd_dynamic_obstacles.csv");
    if (csv_obstacles.is_open()) {
        csv_obstacles << "k,disk,cx,cy,r\n";
    }

    const tinytype goal_pos_tol = tinytype(0.15);
    const tinytype goal_vel_tol = tinytype(0.05);

    auto log_obstacles = [&](int step) {
        if (!csv_obstacles.is_open()) return;
        auto disks = obstacles.disks_at_step(step);
        for (size_t j = 0; j < disks.size(); ++j) {
            csv_obstacles << step << "," << j << ","
                          << disks[j][0] << "," << disks[j][1] << "," << disks[j][2] << "\n";
        }
    };

    auto log_tracking_row = [&](int step, const Vec& state, const Vec& input,
                                double sd_point, double sd_segment,
                                int plan_age, int iters) {
        if (!csv_track.is_open()) return;
        csv_track << step << "," << state(0) << "," << state(1) << "," << state(2) << "," << state(3)
                  << "," << input(0) << "," << input(1) << "," << sd_point
                  << "," << sd_segment << "," << plan_age << "," << iters << "\n";
    };

    auto goal_reached = [&](const Vec& state) -> bool {
        tinytype pos_norm = state.topRows(2).norm();
        tinytype vel_norm = state.bottomRows(2).norm();
        return (pos_norm < goal_pos_tol) && (vel_norm < goal_vel_tol);
    };

    auto replan_plan = [&](int step, const Vec& x_seed) {
        auto disks_now = obstacles.disks_at_step(step);
        double min_sd = signed_distance_point_disks(x_seed, disks_now);

        if (!psd_constraints_active && min_sd < psd_on_distance) {
            psd_constraints_active = true;
        } else if (psd_constraints_active && min_sd > psd_off_distance) {
            psd_constraints_active = false;
        }

        if (psd_constraints_active) {
            solver_psd->settings->en_psd = 1;
            tiny_set_lifted_disks(solver_psd, disks_now);
        } else {
            solver_psd->settings->en_psd = 0;
        }

        Vec x_lift = build_lifted(x_seed);
        tiny_set_x0(solver_psd, x_lift);
        tiny_solve(solver_psd);
        rollout_plan(Ad, Bd, x_seed, solver_psd, &plan);
        plan.start_step = step;
        plan.mode = psd_constraints_active ? PlanMode::PSD : PlanMode::NOMINAL;

        double sd_seed = signed_distance_point_disks(x_seed, disks_now);
        std::size_t disk_count = psd_constraints_active ? disks_now.size() : 0;

        if (csv_plan.is_open()) {
            csv_plan << step << "," << plan_mode_str(plan.mode) << ","
                     << plan.last_iters << "," << disk_count << ","
                     << sd_seed << "," << min_sd << "\n";
        }
        std::cout << "[PSD-DYN] Replan at k=" << step
                  << " mode=" << plan_mode_str(plan.mode)
                  << " iter=" << plan.last_iters
                  << " disks=" << disk_count
                  << " min_sd=" << min_sd << "\n";
    };

    auto disks_init = obstacles.disks_at_step(0);
    double sd0 = signed_distance_point_disks(x_track, disks_init);
    Vec zero_u = Vec::Zero(NU0);
    log_obstacles(0);
    log_tracking_row(0, x_track, zero_u, sd0, sd0, /*plan_age=*/0, /*iters=*/0);

    double min_sd_track = sd0;

    replan_plan(0, x_track);

    Vec prev_state = x_track;
    for (int k = 0; k < total_steps; ++k) {
        bool need_replan = (k == 0)
                        || (k - plan.start_step >= replan_stride)
                        || (k >= plan.start_step + N - horizon_guard);
        if (need_replan && k > 0) {
            replan_plan(k, x_track);
        }

        set_tracking_refs(solver_track, plan, k, diag_refs_track.first, diag_refs_track.second);
        tiny_set_x0(solver_track, build_lifted(x_track));
        tiny_solve(solver_track);
        Vec u0 = solver_track->solution->u.col(0).topRows(NU0);
        prev_state = x_track;
        x_track = Ad * x_track + Bd * u0;

        int step_idx = k + 1;
        log_obstacles(step_idx);
        auto disks_now = obstacles.disks_at_step(step_idx);
        double sd_point = signed_distance_point_disks(x_track, disks_now);
        double sd_segment = signed_distance_segment_disks(prev_state, x_track, disks_now);
        if (sd_segment < min_sd_track) {
            min_sd_track = sd_segment;
        }

        int plan_age = step_idx - plan.start_step;
        log_tracking_row(step_idx, x_track, u0, sd_point, sd_segment,
                         plan_age, solver_track->solution->iter);

        if (goal_reached(x_track)) {
            std::cout << "[PSD-DYN] Goal reached at step " << step_idx
                      << " (pos_norm=" << x_track.topRows(2).norm()
                      << ", vel_norm=" << x_track.bottomRows(2).norm() << ")\n";
            break;
        }
    }

    if (csv_plan.is_open()) csv_plan.close();
    if (csv_track.is_open()) csv_track.close();
    if (csv_obstacles.is_open()) csv_obstacles.close();

    std::cout << "[PSD-DYN] Closed-loop min signed distance: " << min_sd_track << "\n";
    return 0;
}


