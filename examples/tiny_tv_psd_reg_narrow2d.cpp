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

    static constexpr int NX0 = 4;  // [x, y, vx, vy]
    static constexpr int NU0 = 2;  // [ux, uy]
    static constexpr int N   = 40;

    // Base dynamics: double integrator with dt = 1
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

    // Cost weights (match the linear TV demo)
    Mat Q = Mat::Zero(nxL, nxL);
    Q(0, 0) = 10.0;
    Q(1, 1) = 10.0;
    Q(2, 2) = 1.0;
    Q(3, 3) = 1.0;
    Q.diagonal().segment(NX0, NX0*NX0).array() = tinytype(1e-2);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int nuu = NU0 * NU0;
    R.diagonal().head(NU0).array() = tinytype(2.0);
    R.diagonal().segment(NU0, nxu).array() = tinytype(10.0);
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(10.0);
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(500.0);

    Vec fdyn = Vec::Zero(nxL);

    const tinytype rho = 5.0;
    const tinytype rho_psd = 1.0;

    TinySolver* solver = nullptr;
    int status = tiny_setup(&solver, A, B, fdyn, Q, R, rho, nxL, nuL, N, /*verbose=*/1);
    if (status) return status;

    solver->settings->adaptive_rho = 0;
    std::cout << "[TV-PSD-REG] Using fixed rho=" << solver->cache->rho << "\n";

    // Bounds on lifted states and inputs
    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    x_min.middleRows(NX0, NX0*NX0).setConstant(-1000.0);
    x_max.middleRows(NX0, NX0*NX0).setConstant( 1000.0);

    Mat u_min = Mat::Constant(nuL, N-1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N-1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    u_min.bottomRows(nxu + nux + nuu).setConstant(-100.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 100.0);
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // Enable PSD coupling and lifted disk constraints
    tiny_enable_psd(solver, NX0, NU0, rho_psd);

    const tinytype disk_margin = tinytype(0.0);
    const tinytype disk_r = tinytype(3.0);
    std::vector<std::array<tinytype,3>> disks = {
        { tinytype(-3.0), tinytype( 3.25), disk_r + disk_margin },
        { tinytype(-3.0), tinytype(-3.25), disk_r + disk_margin }
    };
    //tiny_set_lifted_disks(solver, disks);
    std::cout << "[TV-PSD-REG] PSD disks set for corridor obstacles\n";

    // Time-varying base tangent planes for the same disks
    tiny_enable_base_tangent_avoidance_2d_multi(solver, disks, disk_margin);
    std::cout << "[TV-PSD-REG] Base tangents enabled for " << disks.size() << " disks\n";

    // Initial lifted state from base x0
    Vec x0(NX0);
    x0 << -8.0, -4.0, 0.0, 0.0;

    Vec x0_lift(nxL);
    x0_lift.setZero();
    x0_lift.topRows(NX0) = x0;
    Mat X0 = x0 * x0.transpose();
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            x0_lift(NX0 + j*NX0 + i) = X0(i, j);
        }
    }
    tiny_set_x0(solver, x0_lift);

    // Linear references on diag(XX) and diag(UU) to match other demos
    const tinytype q_xx = tinytype(1.0);
    const tinytype r_uu = tinytype(10.0);
    Mat Xref = Mat::Zero(nxL, N);
    Mat Uref = Mat::Zero(nuL, N-1);

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX0; ++i) {
            int idx = NX0 + i*NX0 + i;
            tinytype d = solver->work->Q(idx);
            if (d != tinytype(0)) {
                Xref(idx, k) = -q_xx / d;
            }
        }
    }

    const int baseUU = NU0 + nxu + nux;
    for (int k = 0; k < N-1; ++k) {
        for (int j = 0; j < NU0; ++j) {
            int idx = baseUU + j*NU0 + j;
            tinytype d = solver->work->R(idx);
            if (d != tinytype(0)) {
                Uref(idx, k) = -r_uu / d;
            }
        }
    }

    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);

    // Solve and record diagnostics
    tiny_solve(solver);
    int iters = solver->solution->iter;

    // Build dynamics-consistent rollout using base controls
    Vec x_dyn = x0;
    std::vector<Vec> Xdyn(N);
    Xdyn[0] = x_dyn;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver->solution->u.col(k).topRows(NU0);
        x_dyn = Ad * x_dyn + Bd * u_base;
        Xdyn[k+1] = x_dyn;
    }

    auto signed_distance = [&](double x, double y) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& d : disks) {
            double dx = x - static_cast<double>(d[0]);
            double dy = y - static_cast<double>(d[1]);
            double r_eff = static_cast<double>(d[2]);
            double sd = std::sqrt(dx*dx + dy*dy) - r_eff;
            if (sd < best) best = sd;
        }
        return best;
    };

    // Export rollout, controls, lifted diagnostics
    std::ofstream csv("../psd_tv_linear_narrow2d_trajectory.csv");
    if (csv.is_open()) {
        csv << "k,x1,x2,x3,x4,u1,u2,XX_11,XX_22,rank1_gap,signed_dist,iter\n";
        for (int k = 0; k < N; ++k) {
            Vec xk_dyn = Xdyn[k];
            double x1 = xk_dyn(0);
            double x2 = xk_dyn(1);
            double x3 = xk_dyn(2);
            double x4 = xk_dyn(3);

            Vec xk = solver->solution->x.col(k);
            Mat XX_mat(NX0, NX0);
            for (int j = 0; j < NX0; ++j) {
                for (int i = 0; i < NX0; ++i) {
                    XX_mat(i, j) = xk(NX0 + j*NX0 + i);
                }
            }
            double XX_11 = XX_mat(0, 0);
            double XX_22 = XX_mat(1, 1);
            Vec x_base = xk.topRows(NX0);
            double gap = (XX_mat - x_base * x_base.transpose()).norm();

            csv << k << "," << x1 << "," << x2 << "," << x3 << "," << x4;
            if (k < N-1) {
                Vec uk = solver->solution->u.col(k);
                csv << "," << uk(0) << "," << uk(1);
            } else {
                csv << ",0,0";
            }

            double sd = signed_distance(x1, x2);
            csv << "," << XX_11 << "," << XX_22 << "," << gap << "," << sd << "," << iters << "\n";
        }
        csv.close();
        std::cout << "[TV-PSD-REG] Exported psd_tv_linear_narrow2d_trajectory.csv\n";
    } else {
        std::cout << "[EXPORT] Failed to open psd_tv_linear_narrow2d_trajectory.csv for writing\n";
    }

    // Report minimum signed distance from slack state for quick safety check
    double min_sd_slack = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        Vec xk = solver->solution->x.col(k).topRows(NX0);
        double sd = signed_distance(static_cast<double>(xk(0)), static_cast<double>(xk(1)));
        if (sd < min_sd_slack) min_sd_slack = sd;
    }
    std::cout << "[TV-PSD-REG] Min signed distance (slack state): " << min_sd_slack << "\n";

    return 0;
}


