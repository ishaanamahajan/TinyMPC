#include <iostream>
#include <fstream>
#include <limits>
#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>
#include "../src/tinympc/psd_support.hpp"

extern "C" int main() {
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

    const int NX0 = 4;  // [x, y, vx, vy]
    const int NU0 = 2;  // [ux, uy]
    const int N   = 40;  // match TV narrow2d demo

    // Base dynamics (double integrator, dt=1)
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

    // Weights: match tiny_psd_demo.cpp
    Mat Q = Mat::Zero(nxL, nxL);
    // Base state: [x, y, vx, vy]
    Q(0, 0) = 10.0;
    Q(1, 1) = 10.0;
    Q(2, 2) = 1.0;
    Q(3, 3) = 1.0;
    // Lifted vec(XX)

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

    // ADMM penalties
    const tinytype rho = 5.0;
    const tinytype rho_psd = 1.5;

    TinySolver* solver = nullptr;
    int status = tiny_setup(&solver, A, B, fdyn, Q, R, rho, nxL, nuL, N, 1);
    if (status) return status;

    solver->settings->adaptive_rho = 0;

    // Bounds
    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    x_min.middleRows(NX0, NX0*NX0).setConstant(-100.0);
    x_max.middleRows(NX0, NX0*NX0).setConstant( 100.0);

    Mat u_min = Mat::Constant(nuL, N-1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N-1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    u_min.bottomRows(nxu + nux + nuu).setConstant(-100.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 100.0);

    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // PSD block
    tiny_enable_psd(solver, NX0, NU0, rho_psd);

    // Initial lifted state: start at (-8, 4) with zero velocity
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

    // Linear lift-costs: same structure as tiny_psd_demo.cpp
    {
        const int nx0 = NX0;
        const int nu0 = NU0;
        const int nxu_loc = nx0 * nu0;
        const int nux_loc = nu0 * nx0;
        const int baseUU = nu0 + nxu_loc + nux_loc;  // start index of vec(UU)
        const tinytype q_xx = tinytype(1.0);   // linear weight on diag(XX)
        const tinytype r_uu = tinytype(10.0);  // linear weight on diag(UU)

        Mat Xref = Mat::Zero(nxL, N);
        Mat Uref = Mat::Zero(nuL, N-1);

        // State: put q_xx on XX_ii by setting Xref(ii) = -q_xx / Q(ii)
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < nx0; ++i) {
                int idx_xx_ii = nx0 + i*nx0 + i;
                tinytype denom = solver->work->Q(idx_xx_ii);
                if (denom != tinytype(0)) {
                    Xref(idx_xx_ii, k) = -q_xx / denom;
                }
            }
        }

        // Input: put r_uu on UU_jj via Uref
        for (int k = 0; k < N-1; ++k) {
            for (int j = 0; j < nu0; ++j) {
                int idx_uu_jj = baseUU + j*nu0 + j;
                tinytype denom = solver->work->R(idx_uu_jj);
                if (denom != tinytype(0)) {
                    Uref(idx_uu_jj, k) = -r_uu / denom;
                }
            }
        }

        tiny_set_x_ref(solver, Xref);
        tiny_set_u_ref(solver, Uref);
    }

    // Two disks forming a narrow vertical corridor to the LEFT of the origin:
    //   Disk 1: center (-3.0, +3.25), radius r = 3.0
    //   Disk 2: center (-3.0, -3.25), radius r = 3.0
    // This leaves the origin (0,0) free to be approached.
    const tinytype r = tinytype(3.0);
    std::vector<std::array<tinytype, 3>> disks;
    disks.push_back({ -3.0,  3.25, r});
    disks.push_back({ -3.0, -3.25, r});

    tiny_set_lifted_disks(solver, disks);

    // Solve
    tiny_solve(solver);
    int iters = solver->solution->iter;

    // Build a dynamics-consistent rollout under the solved base controls
    Vec x_dyn = x0;  // base initial state [x,y,vx,vy]
    std::vector<Vec> Xdyn(N);
    Xdyn[0] = x_dyn;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver->solution->u.col(k).topRows(NU0);
        x_dyn = Ad * x_dyn + Bd * u_base; // + fdyn.topRows(NX0) if nonzero
        Xdyn[k+1] = x_dyn;
    }

    // Export trajectory and signed distance to obstacles
    std::ofstream csv("../psd_narrow2d_trajectory.csv");
    if (csv.is_open()) {
        csv << "k,x1,x2,u1,u2,sd_min,iter\n";

        for (int k = 0; k < N; ++k) {
            Vec xk = Xdyn[k];
            double x1 = xk(0);
            double x2 = xk(1);

            // Signed distance to the two obstacles:
            // disks centered at (-3.0, +3.25) and (-3.0, -3.25) with radius r = 3.0
            double sd1 = std::sqrt((x1 + 3.0)*(x1 + 3.0) + (x2 - 3.25)*(x2 - 3.25)) - r;
            double sd2 = std::sqrt((x1 + 3.0)*(x1 + 3.0) + (x2 + 3.25)*(x2 + 3.25)) - r;
            double sdmin = std::min(sd1, sd2);

            csv << k << "," << x1 << "," << x2;

            if (k < N-1) {
                Vec uk = solver->solution->u.col(k);
                csv << "," << uk(0) << "," << uk(1);
            } else {
                csv << ",0,0";
            }

            csv << "," << sdmin << "," << iters << "\n";
        }

        csv.close();
        std::cout << "[EXPORT] psd_narrow2d_trajectory.csv written\n";
    }

    return 0;
}
