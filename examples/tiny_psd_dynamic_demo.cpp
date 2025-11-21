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

int clamp_index(int idx, int lo, int hi) {
    if (idx < lo) return lo;
    if (idx > hi) return hi;
    return idx;
}

struct PlanCache {
    std::vector<Vec> states;
    std::vector<Vec> inputs;
    int start_step = 0;
    int last_iters = 0;
};

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
    const tinytype rho_psd = tinytype(5.0);
    const tinytype rho_track = tinytype(5.0);
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
    x0 << -8.0, 0.0, 0.0, 0.0;

    DynamicObstacles obstacles;
    obstacles.dt = tinytype(1.0);
    obstacles.agents = {
        // Downward crossing traffic near the origin
        { tinytype(0.5), tinytype( 3.0), tinytype(0.0), tinytype(-0.12),
          tinytype(0.9), tinytype(0.15), tinytype(0.8), tinytype(0.0),
          tinytype(0.05), tinytype(0.6), tinytype(0.0) },
        // Upward crossing traffic slightly to the right
        { tinytype(2.5), tinytype(-3.5), tinytype(0.0), tinytype( 0.10),
          tinytype(0.8), tinytype(0.12), tinytype(0.7), tinytype(1.2),
          tinytype(0.04), tinytype(0.7), tinytype(0.5) },
        // Slightly drifting obstacle to bias detours in +y
        { tinytype(4.0), tinytype( 1.5), tinytype(0.02), tinytype(-0.06),
          tinytype(0.7), tinytype(0.1), tinytype(0.5), tinytype(-0.4),
          tinytype(0.03), tinytype(0.5), tinytype(0.9) }
    };

    TinySolver* solver_psd = nullptr;
    if (tiny_setup(&solver_psd, A, B, fdyn, Q, R, rho_psd, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_psd->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_psd, x_min, x_max, u_min, u_max);
    tiny_enable_psd(solver_psd, NX0, NU0, rho_psd_penalty);
    auto diag_refs_psd = build_diag_refs(solver_psd);
    tiny_set_x_ref(solver_psd, diag_refs_psd.first);
    tiny_set_u_ref(solver_psd, diag_refs_psd.second);

    TinySolver* solver_track = nullptr;
    if (tiny_setup(&solver_track, A, B, fdyn, Q, R, rho_track, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_track->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_track, x_min, x_max, u_min, u_max);
    auto diag_refs_track = build_diag_refs(solver_track);

    PlanCache plan;
    Vec x_track = x0;

    const int total_steps = 90;
    const int replan_stride = 6;
    const tinytype inflation_rate = tinytype(0.02);
    const int horizon_guard = 5;

    std::ofstream csv_plan("../psd_dynamic_plan_log.csv");
    if (csv_plan.is_open()) {
        csv_plan << "replan_step,iter,num_disks,min_sd_seed\n";
    }
    std::ofstream csv_track("../psd_dynamic_tracking.csv");
    if (csv_track.is_open()) {
        csv_track << "k,x1,x2,x3,x4,u1,u2,signed_dist,plan_age,solver_iter\n";
    }
    std::ofstream csv_obstacles("../psd_dynamic_obstacles.csv");
    if (csv_obstacles.is_open()) {
        csv_obstacles << "k,disk,cx,cy,r\n";
    }

    auto replan_psd = [&](int step, const Vec& x_seed) {
        Vec x_lift = build_lifted(x_seed);
        tiny_set_x0(solver_psd, x_lift);
        auto horizon = obstacles.horizon_disks(step, N, inflation_rate);
        tiny_set_lifted_disks(solver_psd, horizon);
        tiny_solve(solver_psd);
        rollout_plan(Ad, Bd, x_seed, solver_psd, &plan);
        plan.start_step = step;
        if (csv_plan.is_open()) {
            auto disks_now = obstacles.disks_at_step(step);
            double sd_seed = signed_distance_point_disks(x_seed, disks_now);
            csv_plan << step << "," << plan.last_iters << "," << horizon.size()
                     << "," << sd_seed << "\n";
        }
        std::cout << "[PSD-DYN] Replan at k=" << step
                  << " iter=" << plan.last_iters
                  << " disks=" << horizon.size() << "\n";
    };

    auto log_obstacles = [&](int step) {
        if (!csv_obstacles.is_open()) return;
        auto disks = obstacles.disks_at_step(step);
        for (size_t j = 0; j < disks.size(); ++j) {
            csv_obstacles << step << "," << j << ","
                          << disks[j][0] << "," << disks[j][1] << "," << disks[j][2] << "\n";
        }
    };

    auto log_tracking_row = [&](int step, const Vec& state, const Vec& input,
                                double sd, int plan_age, int iters) {
        if (!csv_track.is_open()) return;
        csv_track << step << "," << state(0) << "," << state(1) << "," << state(2) << "," << state(3)
                  << "," << input(0) << "," << input(1) << "," << sd
                  << "," << plan_age << "," << iters << "\n";
    };

    replan_psd(0, x_track);

    auto disks_init = obstacles.disks_at_step(0);
    double sd0 = signed_distance_point_disks(x_track, disks_init);
    Vec zero_u = Vec::Zero(NU0);
    log_obstacles(0);
    log_tracking_row(0, x_track, zero_u, sd0, /*plan_age=*/0, /*iters=*/0);

    double min_sd_track = sd0;

    for (int k = 0; k < total_steps; ++k) {
        bool need_replan = (k == 0)
                        || (k - plan.start_step >= replan_stride)
                        || (k >= plan.start_step + N - horizon_guard);
        if (need_replan && k > 0) {
            replan_psd(k, x_track);
        }

        set_tracking_refs(solver_track, plan, k, diag_refs_track.first, diag_refs_track.second);
        tiny_set_x0(solver_track, build_lifted(x_track));
        tiny_solve(solver_track);
        Vec u0 = solver_track->solution->u.col(0).topRows(NU0);
        x_track = Ad * x_track + Bd * u0;

        int step_idx = k + 1;
        log_obstacles(step_idx);
        auto disks_now = obstacles.disks_at_step(step_idx);
        double sd = signed_distance_point_disks(x_track, disks_now);
        if (sd < min_sd_track) {
            min_sd_track = sd;
        }

        int plan_age = step_idx - plan.start_step;
        log_tracking_row(step_idx, x_track, u0, sd, plan_age, solver_track->solution->iter);
    }

    if (csv_plan.is_open()) csv_plan.close();
    if (csv_track.is_open()) csv_track.close();
    if (csv_obstacles.is_open()) csv_obstacles.close();

    std::cout << "[PSD-DYN] Closed-loop min signed distance: " << min_sd_track << "\n";
    return 0;
}


