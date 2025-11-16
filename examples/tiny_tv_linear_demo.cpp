#include <iostream>
#include <fstream>
#include <limits>
#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>
#include "../src/tinympc/psd_support.hpp"
#include "demo_config.hpp"

// Bind config macros to local constants for readability
static constexpr int NX0 = DEMO_NX0;
static constexpr int NU0 = DEMO_NU0;
static constexpr int N   = DEMO_N;

extern "C" int main() {
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

    // Base dynamics (double integrator)
    Mat Ad(NX0, NX0); Ad << 1,0,1,0,  0,1,0,1,  0,0,1,0,  0,0,0,1;
    Mat Bd(NX0, NU0); Bd << 0.5,0,  0,0.5,  1,0,  0,1;

    // Lifted model
    Mat A, B;
    tiny_build_lifted_from_base(Ad, Bd, A, B);
    const int nxL = A.rows();
    const int nuL = B.cols();

    // Weights similar to PSD demo
    Mat Q = Mat::Zero(nxL, nxL);
    Q(0,0) = 10.0; Q(1,1) = 10.0; Q(2,2) = 1.0; Q(3,3) = 1.0;
    Q.diagonal().segment(NX0, NX0*NX0).array() = tinytype(1e-6);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0*NU0, nux = NU0*NX0, nuu = NU0*NU0;
    R.diagonal().head(NU0).array() = tinytype(2.0);
    R.diagonal().segment(NU0, nxu).array() = tinytype(10.0);
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(10.0);
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(500.0);
    Vec fdyn = Vec::Zero(nxL);

    // ADMM penalty for default TV-linear demo
    const tinytype rho = 5.0;
    const tinytype margin = tinytype(DEMO_OBS_MARGIN); // keep consistent with PSD demo radius
    
    TinySolver *solver = nullptr;
    int status = tiny_setup(&solver,
                            A, B, fdyn, Q, R,
                            rho, nxL, nuL, N,
                            /*verbose=*/1);

    // Adaptive rho DISABLED - it doesn't converge well for TV problems  
    // Use fixed rho instead
    solver->settings->adaptive_rho = 0;
    std::cout << "[TV-LIN] Using fixed rho=" << solver->cache->rho << "\n";
    
    if (status) return status;

    // Bounds
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
    u_min.bottomRows(nxu + nux + nuu).setConstant(-1000.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 1000.0);
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // Lifted initial condition
    Vec x0(NX0); x0 << -10, 0.1, 0, 0;
    Vec x0_lift(nxL); x0_lift.setZero();
    x0_lift.topRows(NX0) = x0;
    Mat X0 = x0 * x0.transpose();
    for (int j = 0; j < NX0; ++j)
        for (int i = 0; i < NX0; ++i)
            x0_lift(NX0 + j*NX0 + i) = X0(i,j);
    tiny_set_x0(solver, x0_lift);

    // Small linear terms on XX and UU via references
    Mat Xref = Mat::Zero(nxL, N);
    Mat Uref = Mat::Zero(nuL, N-1);
    const tinytype q_xx = tinytype(1.0);
    const tinytype r_uu = tinytype(10.0);
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX0; ++i) {
            int idx_xx_ii = NX0 + i*NX0 + i;
            tinytype denom = solver->work->Q(idx_xx_ii);
            if (denom != tinytype(0)) Xref(idx_xx_ii, k) = -q_xx / denom;
        }
    }
    const int baseUU = NU0 + nxu + nux;
    for (int k = 0; k < N-1; ++k) {
        for (int j = 0; j < NU0; ++j) {
            int idx_uu_jj = baseUU + j*NU0 + j;
            tinytype denom = solver->work->R(idx_uu_jj);
            if (denom != tinytype(0)) Uref(idx_uu_jj, k) = -r_uu / denom;
        }
    }
    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);

    // Per-iteration re-tangent hyperplane projection (no trust region, no outer loop)
    // Enable base-tangent updates so planes are refreshed every ADMM iteration
    const tinytype ox = tinytype(DEMO_OBS_X), oy = tinytype(DEMO_OBS_Y), r = tinytype(DEMO_OBS_R);
    tiny_enable_base_tangent_avoidance(solver, ox, oy, r, margin);

    // Solve
    tiny_solve(solver);
    int iters = solver->solution->iter;

    // Export trajectory and signed distance to obstacle
    std::ofstream csv("../tv_linear_trajectory.csv");
    if (csv.is_open()) {
        csv << "k,x1,x2,u1,u2,signed_dist,iter\n";
        for (int k = 0; k < N; ++k) {
            Vec xk = solver->solution->x.col(k);
            double x1 = xk(0), x2 = xk(1);
            double sd = std::sqrt((x1-ox)*(x1-ox) + (x2-oy)*(x2-oy)) - r;
            csv << k << "," << x1 << "," << x2;
            if (k < N-1) {
                Vec uk = solver->solution->u.col(k);
                csv << "," << uk(0) << "," << uk(1);
            } else {
                csv << ",0,0";
            }
            csv << "," << sd << "," << iters << "\n";
        }
        csv.close();
        std::cout << "\n[EXPORT] TV-linear trajectory saved to tv_linear_trajectory.csv\n";
    }

    // Simple summary
    double min_sd = 1e9;
    for (int k = 0; k < N; ++k) {
        Vec xk = solver->solution->x.col(k);
        double sd = std::sqrt((xk(0)-ox)*(xk(0)-ox) + (xk(1)-oy)*(xk(1)-oy)) - r;
        if (sd < min_sd) min_sd = sd;
    }
    std::cout << "[TV-LIN] Min signed distance to obstacle: " << min_sd << "\n";

    return 0;
}
