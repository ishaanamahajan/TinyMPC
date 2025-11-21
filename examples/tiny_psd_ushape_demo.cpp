#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
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

    auto build_lift_diag_refs = [&](TinySolver* solver_handle) {
        Mat Xref = Mat::Zero(nxL, N);
        Mat Uref = Mat::Zero(nuL, N-1);
        const tinytype q_xx = tinytype(1.0);
        const tinytype r_uu = tinytype(10.0);
        const int baseUU = NU0 + nxu + nux;
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < NX0; ++i) {
                int idx = NX0 + i*NX0 + i;
                tinytype d = solver_handle->work->Q(idx);
                if (d != tinytype(0)) Xref(idx, k) = -q_xx / d;
            }
        }
        for (int k = 0; k < N-1; ++k) {
            for (int j = 0; j < NU0; ++j) {
                int idx = baseUU + j*NU0 + j;
                tinytype d = solver_handle->work->R(idx);
                if (d != tinytype(0)) Uref(idx, k) = -r_uu / d;
            }
        }
        return std::make_pair(Xref, Uref);
    };

    const tinytype rho = 5.0;
    const tinytype rho_psd = 0.96;

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
    x0 << 6.0, -1.0, 0.0, 0.0;  // start deeper inside the cul-de-sac
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
        auto diag_refs_psd = build_lift_diag_refs(solver);
        Mat Xref_psd = diag_refs_psd.first;
        Mat Uref_psd = diag_refs_psd.second;
        tiny_set_x_ref(solver, Xref_psd);
        tiny_set_u_ref(solver, Uref_psd);
    }

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

    tiny_solve(solver);
    int iters = solver->solution->iter;

    Vec x_dyn = x0;
    std::vector<Vec> Xdyn(N);
    std::vector<Vec> Udyn(N-1, Vec::Zero(NU0));
    Xdyn[0] = x_dyn;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver->solution->u.col(k).topRows(NU0);
        Udyn[k] = u_base;
        x_dyn = Ad * x_dyn + Bd * u_base;
        Xdyn[k+1] = x_dyn;
    }

    std::ofstream csv("../psd_ushape_trajectory.csv");
    if (csv.is_open()) {
        csv << "k,x1,x2,x3,x4,u1,u2,XX_11,XX_22,rank1_gap,signed_dist,iter\n";
        double min_sd = std::numeric_limits<double>::infinity();
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
            if (sd < min_sd) min_sd = sd;

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
        double min_sd_traj = std::numeric_limits<double>::infinity();
        for (const auto& xk_dyn : Xdyn) {
            double sd = signed_distance(xk_dyn(0), xk_dyn(1));
            if (sd < min_sd_traj) min_sd_traj = sd;
        }
        std::cout << "[PSD-U] Min signed distance to U-shape: " << min_sd_traj << "\n";
    } else {
        std::cout << "[PSD-U] Failed to open psd_ushape_trajectory.csv\n";
    }

    // Closed-loop tracking: plan once with PSD, then track with unconstrained TinyMPC
    TinySolver* solver_track = nullptr;
    if (tiny_setup(&solver_track, A, B, fdyn, Q, R, rho, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_track->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_track, x_min, x_max, u_min, u_max);

    auto diag_refs_track = build_lift_diag_refs(solver_track);
    Mat Xref_stab_track = diag_refs_track.first;
    Mat Uref_stab_track = diag_refs_track.second;
    Mat Xref_track = Xref_stab_track;
    Mat Uref_track = Uref_stab_track;

    Vec x_track = x0;
    Vec zero_u = Vec::Zero(NU0);
    double min_sd_track = signed_distance(x_track(0), x_track(1));
    const int steps = N - 1;

    std::ofstream csv_track("../psd_ushape_tracking.csv");
    if (csv_track.is_open()) {
        csv_track << "k,x1,x2,x3,x4,u1,u2,signed_dist,iter\n";
        csv_track << 0 << "," << x_track(0) << "," << x_track(1) << "," << x_track(2) << "," << x_track(3)
                  << "," << zero_u(0) << "," << zero_u(1) << "," << min_sd_track << ",0\n";

        for (int k = 0; k < steps; ++k) {
            Vec x_lift = build_lifted(x_track);
            tiny_set_x0(solver_track, x_lift);

            Xref_track = Xref_stab_track;
            for (int i = 0; i < N; ++i) {
                int plan_idx = std::min(k + i, N-1);
                Xref_track.col(i).topRows(NX0) = Xdyn[plan_idx];
            }

            Uref_track = Uref_stab_track;
            for (int i = 0; i < N-1; ++i) {
                int plan_idx = k + i;
                if (plan_idx < N-1) {
                    Uref_track.col(i).topRows(NU0) = Udyn[plan_idx];
                } else {
                    Uref_track.col(i).topRows(NU0).setZero();
                }
            }
            tiny_set_x_ref(solver_track, Xref_track);
            tiny_set_u_ref(solver_track, Uref_track);

            tiny_solve(solver_track);
            Vec u0 = solver_track->solution->u.col(0).topRows(NU0);
            x_track = Ad * x_track + Bd * u0;
            double sd = signed_distance(x_track(0), x_track(1));
            if (sd < min_sd_track) min_sd_track = sd;

            csv_track << (k+1) << "," << x_track(0) << "," << x_track(1) << "," << x_track(2) << "," << x_track(3)
                      << "," << u0(0) << "," << u0(1) << "," << sd << "," << solver_track->solution->iter << "\n";
        }
        csv_track.close();
        std::cout << "[PSD-U] Exported psd_ushape_tracking.csv\n";
        std::cout << "[PSD-U] Closed-loop min signed distance: " << min_sd_track << "\n";
    } else {
        std::cout << "[PSD-U] Failed to open psd_ushape_tracking.csv\n";
    }

    return 0;
}


