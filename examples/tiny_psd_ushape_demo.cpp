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
    static constexpr int N   = 45;

    // Double-integrator base dynamics (dt = 1)
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
    const tinytype rho_psd = 1.5;

    TinySolver* solver = nullptr;
    if (tiny_setup(&solver, A, B, fdyn, Q, R, rho, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver->settings->adaptive_rho = 0;
    std::cout << "[PSD-U] Using fixed rho=" << solver->cache->rho << "\n";

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
    x0 << 6.0, 0.0, 0.0, 0.0;  // start deeper inside the cul-de-sac
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

    // U-shaped obstacle from disks (two arms + base)
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

    tiny_enable_psd(solver, NX0, NU0, rho_psd);
    tiny_set_lifted_disks(solver, disks);

    // Add gentle linear refs on diag(XX)/diag(UU) to stabilize lifted slack variables
    {
        Mat Xref = Mat::Zero(nxL, N);
        Mat Uref = Mat::Zero(nuL, N-1);
        const tinytype q_xx = tinytype(1.0);
        const tinytype r_uu = tinytype(10.0);
        const int baseUU = NU0 + nxu + nux;
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < NX0; ++i) {
                int idx = NX0 + i*NX0 + i;
                tinytype d = solver->work->Q(idx);
                if (d != tinytype(0)) Xref(idx, k) = -q_xx / d;
            }
        }
        for (int k = 0; k < N-1; ++k) {
            for (int j = 0; j < NU0; ++j) {
                int idx = baseUU + j*NU0 + j;
                tinytype d = solver->work->R(idx);
                if (d != tinytype(0)) Uref(idx, k) = -r_uu / d;
            }
        }
        tiny_set_x_ref(solver, Xref);
        tiny_set_u_ref(solver, Uref);
    }

    tiny_solve(solver);
    int iters = solver->solution->iter;

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

    Vec x_dyn = x0;
    std::vector<Vec> Xdyn(N);
    Xdyn[0] = x_dyn;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver->solution->u.col(k).topRows(NU0);
        x_dyn = Ad * x_dyn + Bd * u_base;
        Xdyn[k+1] = x_dyn;
    }

    std::ofstream csv("../psd_ushape_trajectory.csv");
    if (csv.is_open()) {
        csv << "k,x1,x2,x3,x4,u1,u2,XX_11,XX_22,rank1_gap,signed_dist,iter\n";
        for (int k = 0; k < N; ++k) {
            Vec xk_dyn = Xdyn[k];
            Vec xk = solver->solution->x.col(k);
            Mat XX_mat(NX0, NX0);
            for (int j = 0; j < NX0; ++j) {
                for (int i = 0; i < NX0; ++i) {
                    XX_mat(i,j) = xk(NX0 + j*NX0 + i);
                }
            }
            double gap = (XX_mat - xk.topRows(NX0) * xk.topRows(NX0).transpose()).norm();
            double sd = signed_distance(xk_dyn(0), xk_dyn(1));

            csv << k << "," << xk_dyn(0) << "," << xk_dyn(1) << "," << xk_dyn(2) << "," << xk_dyn(3);
            if (k < N-1) {
                Vec uk = solver->solution->u.col(k);
                csv << "," << uk(0) << "," << uk(1);
            } else {
                csv << ",0,0";
            }
            csv << "," << XX_mat(0,0) << "," << XX_mat(1,1) << "," << gap << "," << sd << "," << iters << "\n";
        }
        csv.close();
        std::cout << "[PSD-U] Exported psd_ushape_trajectory.csv\n";
    } else {
        std::cout << "[PSD-U] Failed to open psd_ushape_trajectory.csv\n";
    }

    double min_sd = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        double sd = signed_distance(solver->solution->x(0, k), solver->solution->x(1, k));
        if (sd < min_sd) min_sd = sd;
    }
    std::cout << "[PSD-U] Min signed distance to U-shape: " << min_sd << "\n";

    return 0;
}


