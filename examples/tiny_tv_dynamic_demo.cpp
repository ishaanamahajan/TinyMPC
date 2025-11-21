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

double signed_distance_point_disks(const Vec& x,
                                   const std::vector<std::array<tinytype,3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& d : disks) {
        double dx = static_cast<double>(x(0) - d[0]);
        double dy = static_cast<double>(x(1) - d[1]);
        double r  = static_cast<double>(d[2]);
        double sd = std::sqrt(dx*dx + dy*dy) - r;
        if (sd < best) best = sd;
    }
    return best;
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
    const tinytype rho = tinytype(5.0);
    const tinytype margin = tinytype(0.0);

    TinySolver* solver = nullptr;
    if (tiny_setup(&solver, A, B, fdyn, Q, R, rho, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver->settings->adaptive_rho = 0;

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
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    Vec x0(NX0);
    x0 << -8.0, 0.0, 0.0, 0.0;

    Mat Xref = Mat::Zero(nxL, N);
    Mat Uref = Mat::Zero(nuL, N-1);
    const tinytype q_xx = tinytype(1.0);
    const tinytype r_uu = tinytype(10.0);
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX0; ++i) {
            int idx = NX0 + i*NX0 + i;
            tinytype d = solver->work->Q(idx);
            if (d != tinytype(0)) Xref(idx, k) = -q_xx / d;
        }
    }
    const int baseUU = NU0 + nxu + nux;
    for (int k = 0; k < N-1; ++k) {
        for (int j = 0; j < NU0; ++j) {
            int idx = baseUU + j*NU0 + j;
            tinytype d = solver->work->R(idx);
            if (d != tinytype(0)) Uref(idx, k) = -r_uu / d;
        }
    }
    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);

    DynamicObstacles obstacles;
    obstacles.dt = tinytype(1.0);
    obstacles.agents = {
        { tinytype(0.5), tinytype( 3.0), tinytype(0.0), tinytype(-0.12),
          tinytype(0.9), tinytype(0.15), tinytype(0.8), tinytype(0.0),
          tinytype(0.05), tinytype(0.6), tinytype(0.0) },
        { tinytype(2.5), tinytype(-3.5), tinytype(0.0), tinytype( 0.10),
          tinytype(0.8), tinytype(0.12), tinytype(0.7), tinytype(1.2),
          tinytype(0.04), tinytype(0.7), tinytype(0.5) },
        { tinytype(4.0), tinytype( 1.5), tinytype(0.02), tinytype(-0.06),
          tinytype(0.7), tinytype(0.1), tinytype(0.5), tinytype(-0.4),
          tinytype(0.03), tinytype(0.5), tinytype(0.9) }
    };

    const int total_steps = 90;

    std::ofstream csv("../tv_dynamic_tracking.csv");
    if (csv.is_open()) {
        csv << "k,x1,x2,x3,x4,u1,u2,signed_dist,iter\n";
    } else {
        std::cout << "[TV-DYN] Failed to open tv_dynamic_tracking.csv\n";
        return 1;
    }
    std::ofstream csv_obs("../tv_dynamic_obstacles.csv");
    if (csv_obs.is_open()) {
        csv_obs << "k,disk,cx,cy,r\n";
    }

    auto log_obstacles = [&](int step) {
        if (!csv_obs.is_open()) return;
        auto disks = obstacles.disks_at_step(step);
        for (size_t j = 0; j < disks.size(); ++j) {
            csv_obs << step << "," << j << ","
                    << disks[j][0] << "," << disks[j][1] << "," << disks[j][2] << "\n";
        }
    };

    Vec x_dyn = x0;
    Vec zero_u = Vec::Zero(NU0);
    auto disks0 = obstacles.disks_at_step(0);
    double sd0 = signed_distance_point_disks(x_dyn, disks0);
    csv << 0 << "," << x_dyn(0) << "," << x_dyn(1) << "," << x_dyn(2) << "," << x_dyn(3)
        << "," << zero_u(0) << "," << zero_u(1) << "," << sd0 << ",0\n";
    log_obstacles(0);

    double min_sd = sd0;
    for (int k = 0; k < total_steps; ++k) {
        auto disks_now = obstacles.disks_at_step(k);
        tiny_enable_base_tangent_avoidance_2d_multi(solver, disks_now, margin);

        tiny_set_x0(solver, build_lifted(x_dyn));
        tiny_solve(solver);

        Vec u0 = solver->solution->u.col(0).topRows(NU0);
        x_dyn = Ad * x_dyn + Bd * u0;

        int step_idx = k + 1;
        log_obstacles(step_idx);
        auto disks_next = obstacles.disks_at_step(step_idx);
        double sd = signed_distance_point_disks(x_dyn, disks_next);
        if (sd < min_sd) min_sd = sd;

        csv << step_idx << "," << x_dyn(0) << "," << x_dyn(1) << "," << x_dyn(2) << "," << x_dyn(3)
            << "," << u0(0) << "," << u0(1) << "," << sd << "," << solver->solution->iter << "\n";
    }

    csv.close();
    if (csv_obs.is_open()) csv_obs.close();

    std::cout << "[TV-DYN] Exported tv_dynamic_tracking.csv\n";
    std::cout << "[TV-DYN] Min signed distance: " << min_sd << "\n";
    return 0;
}


