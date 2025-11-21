#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>
#include "../src/tinympc/psd_support.hpp"

extern "C" int main() {
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

    static constexpr int NX0 = 4;
    static constexpr int NU0 = 2;
    static constexpr int N   = 45;   // shorter horizon tends to stall near the U opening

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
    const tinytype rho = 5.0;
    const tinytype margin = tinytype(0.0);

    TinySolver* solver = nullptr;
    if (tiny_setup(&solver, A, B, fdyn, Q, R, rho, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver->settings->adaptive_rho = 0;
    std::cout << "[TV-U] Using fixed rho=" << solver->cache->rho << "\n";

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
    x0 << 6.0, -1.0, 0.0, 0.0;

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

    const tinytype r_wall = tinytype(0.8);
    std::vector<std::array<tinytype,3>> disks = {
        { tinytype(2.5), tinytype( 0.0), r_wall },
        { tinytype(2.5), tinytype( 1.2), r_wall },
        { tinytype(2.5), tinytype(-1.2), r_wall },
        { tinytype(3.8), tinytype( 1.2), r_wall },
        { tinytype(3.8), tinytype(-1.2), r_wall },
        { tinytype(5.0), tinytype( 1.2), r_wall },
        { tinytype(5.0), tinytype(-1.2), r_wall }
    };
    tiny_enable_base_tangent_avoidance_2d_multi(solver, disks, margin);
    std::cout << "[TV-U] Base tangents enabled for " << disks.size() << " disks\n";

    auto signed_distance = [&](double x, double y) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& d : disks) {
            double dx = x - static_cast<double>(d[0]);
            double dy = y - static_cast<double>(d[1]);
            double r  = static_cast<double>(d[2]);
            double sd = std::sqrt(dx*dx + dy*dy) - r;
            if (sd < best) best = sd;
        }
        return best;
    };

    auto build_lifted = [&](const Vec& base_state) {
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
    };

    Vec x_dyn = x0;
    const int steps = N - 1;
    Vec zero_u = Vec::Zero(NU0);
    double min_sd = signed_distance(x_dyn(0), x_dyn(1));

    std::ofstream csv("../tv_ushape_trajectory.csv");
    if (!csv.is_open()) {
        std::cout << "[TV-U] Failed to open tv_ushape_trajectory.csv\n";
        return 1;
    }

    csv << "k,x1,x2,x3,x4,u1,u2,signed_dist,iter\n";

    auto write_row = [&](int k, const Vec& xstate, const Vec& ustate, double sd, int iters) {
        csv << k << "," << xstate(0) << "," << xstate(1) << "," << xstate(2) << "," << xstate(3)
            << "," << ustate(0) << "," << ustate(1) << "," << sd << "," << iters << "\n";
    };

    write_row(0, x_dyn, zero_u, min_sd, 0);

    for (int k = 0; k < steps; ++k) {
        Vec x_lift = build_lifted(x_dyn);
        tiny_set_x0(solver, x_lift);
        tiny_solve(solver);

        Vec u0 = solver->solution->u.col(0).topRows(NU0);
        x_dyn = Ad * x_dyn + Bd * u0;
        double sd = signed_distance(x_dyn(0), x_dyn(1));
        if (sd < min_sd) min_sd = sd;

        write_row(k+1, x_dyn, u0, sd, solver->solution->iter);
    }

    csv.close();
    std::cout << "[TV-U] Exported tv_ushape_trajectory.csv\n";
    std::cout << "[TV-U] Min signed distance to U-shape: " << min_sd << "\n";

    return 0;
}


