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

    static constexpr int NX0 = 4;   // [x, y, vx, vy]
    static constexpr int NU0 = 2;   // [ux, uy]
    static constexpr int N   = 40;

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
    Q(0, 0) = 10.0; Q(1, 1) = 10.0; Q(2, 2) = 1.0; Q(3, 3) = 1.0;
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

    const tinytype rho_psd = tinytype(5.0);
    const tinytype rho_tv = tinytype(5.0);
    const tinytype rho_psd_penalty = tinytype(1.0);

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

    Vec x0(NX0); x0 << -8.0, -4.0, 0.0, 0.0;
    Vec x0_lift(nxL); x0_lift.setZero();
    x0_lift.topRows(NX0) = x0;
    Mat X0 = x0 * x0.transpose();
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            x0_lift(NX0 + j*NX0 + i) = X0(i, j);
        }
    }

    std::vector<std::array<tinytype, 3>> disks = {
        { tinytype(-3.0), tinytype( 3.25), tinytype(3.0) },
        { tinytype(-3.0), tinytype(-3.25), tinytype(3.0) }
    };

    auto signed_distance = [&](double x, double y) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& d : disks) {
            double dx = x - static_cast<double>(d[0]);
            double dy = y - static_cast<double>(d[1]);
            double r = static_cast<double>(d[2]);
            double sd = std::sqrt(dx*dx + dy*dy) - r;
            if (sd < best) best = sd;
        }
        return best;
    };

    TinySolver* solver_psd = nullptr;
    if (tiny_setup(&solver_psd, A, B, fdyn, Q, R, rho_psd, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_psd->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_psd, x_min, x_max, u_min, u_max);
    tiny_enable_psd(solver_psd, NX0, NU0, rho_psd_penalty);
    tiny_set_lifted_disks(solver_psd, disks);
    tiny_set_x0(solver_psd, x0_lift);

    {
        Mat Xref = Mat::Zero(nxL, N);
        Mat Uref = Mat::Zero(nuL, N-1);
        const tinytype q_xx = tinytype(1.0);
        const tinytype r_uu = tinytype(10.0);
        const int baseUU = NU0 + nxu + nux;

        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < NX0; ++i) {
                int idx = NX0 + i*NX0 + i;
                tinytype denom = solver_psd->work->Q(idx);
                if (denom != tinytype(0)) Xref(idx, k) = -q_xx / denom;
            }
        }
        for (int k = 0; k < N-1; ++k) {
            for (int j = 0; j < NU0; ++j) {
                int idx = baseUU + j*NU0 + j;
                tinytype denom = solver_psd->work->R(idx);
                if (denom != tinytype(0)) Uref(idx, k) = -r_uu / denom;
            }
        }
        tiny_set_x_ref(solver_psd, Xref);
        tiny_set_u_ref(solver_psd, Uref);
    }

    tiny_solve(solver_psd);
    int iters_psd = solver_psd->solution->iter;

    std::vector<Vec> Xpsd(N, Vec::Zero(NX0));
    std::vector<Vec> Upsd(N-1, Vec::Zero(NU0));
    Vec x_dyn_psd = x0;
    Xpsd[0] = x_dyn_psd;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver_psd->solution->u.col(k).topRows(NU0);
        Upsd[k] = u_base;
        x_dyn_psd = Ad * x_dyn_psd + Bd * u_base;
        Xpsd[k+1] = x_dyn_psd;
    }

    {
        std::ofstream csv("../psd_tv_pipeline_narrow2d_stage1_psd.csv");
        if (csv.is_open()) {
            csv << "k,x1,x2,x3,x4,u1,u2,signed_dist,iter\n";
            for (int k = 0; k < N; ++k) {
                const Vec& xk = Xpsd[k];
                double sd = signed_distance(xk(0), xk(1));
                csv << k << "," << xk(0) << "," << xk(1) << "," << xk(2) << "," << xk(3);
                if (k < N-1) {
                    const Vec& uk = Upsd[k];
                    csv << "," << uk(0) << "," << uk(1);
                } else {
                    csv << ",0,0";
                }
                csv << "," << sd << "," << iters_psd << "\n";
            }
            csv.close();
            std::cout << "[PSD+TV-N2D] Stage1 rollout -> psd_tv_pipeline_narrow2d_stage1_psd.csv\n";
        } else {
            std::cout << "[EXPORT] Failed to open Stage1 CSV\n";
        }
    }

    TinySolver* solver_tv = nullptr;
    if (tiny_setup(&solver_tv, A, B, fdyn, Q, R, rho_tv, nxL, nuL, N, /*verbose=*/1)) {
        return 1;
    }
    solver_tv->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_tv, x_min, x_max, u_min, u_max);
    tiny_set_x0(solver_tv, x0_lift);

    Mat Xref_tv = Mat::Zero(nxL, N);
    Mat Uref_tv = Mat::Zero(nuL, N-1);
    for (int k = 0; k < N; ++k) {
        Xref_tv.col(k).topRows(NX0) = Xpsd[k];
    }
    for (int k = 0; k < N-1; ++k) {
        Uref_tv.col(k).topRows(NU0) = Upsd[k];
    }
    tiny_set_x_ref(solver_tv, Xref_tv);
    tiny_set_u_ref(solver_tv, Uref_tv);

    const tinytype margin = tinytype(0.0);
    tiny_enable_base_tangent_avoidance_2d_multi(solver_tv, disks, margin);

    tiny_solve(solver_tv);
    int iters_tv = solver_tv->solution->iter;

    std::vector<Vec> Xtv(N, Vec::Zero(NX0));
    Vec x_dyn_tv = x0;
    Xtv[0] = x_dyn_tv;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver_tv->solution->u.col(k).topRows(NU0);
        x_dyn_tv = Ad * x_dyn_tv + Bd * u_base;
        Xtv[k+1] = x_dyn_tv;
    }

    double min_sd_tv = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        double sd = signed_distance(Xtv[k](0), Xtv[k](1));
        if (sd < min_sd_tv) min_sd_tv = sd;
    }
    std::cout << "[PSD+TV-N2D] Stage2 min signed distance: " << min_sd_tv << "\n";

    {
        std::ofstream csv("../psd_tv_pipeline_narrow2d_stage2_tv.csv");
        if (csv.is_open()) {
            csv << "k,x1,x2,x3,x4,u1,u2,XX_11,XX_22,rank1_gap,signed_dist,iter\n";
            for (int k = 0; k < N; ++k) {
                Vec xk_dyn = Xtv[k];
                double x1 = xk_dyn(0);
                double x2 = xk_dyn(1);
                double x3 = xk_dyn(2);
                double x4 = xk_dyn(3);

                Vec xk = solver_tv->solution->x.col(k);
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
                    Vec uk = solver_tv->solution->u.col(k);
                    csv << "," << uk(0) << "," << uk(1);
                } else {
                    csv << ",0,0";
                }

                double sd = signed_distance(x1, x2);
                csv << "," << XX_11 << "," << XX_22 << "," << gap << "," << sd << "," << iters_tv << "\n";
            }
            csv.close();
            std::cout << "[PSD+TV-N2D] Stage2 rollout -> psd_tv_pipeline_narrow2d_stage2_tv.csv\n";
        } else {
            std::cout << "[EXPORT] Failed to open Stage2 CSV\n";
        }
    }

    return 0;
}


