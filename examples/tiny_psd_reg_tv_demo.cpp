#include <iostream>
#include <fstream>
#include <limits>
#include <Eigen/Dense>
#include <vector>
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

    // Base dynamics like the Julia script
    Mat Ad(NX0, NX0); Ad << 1,0,1,0,  0,1,0,1,  0,0,1,0,  0,0,0,1;
    Mat Bd(NX0, NU0); Bd << 0.5,0,  0,0.5,  1,0,  0,1;

    Mat A,B;
    tiny_build_lifted_from_base(Ad, Bd, A, B); // prints sizes

    const int nxL = A.rows();
    const int nuL = B.cols();

    // Weights: emphasize base motion; tiny quadratic on vec(XX); strong on UU
    Mat Q = Mat::Zero(nxL, nxL);
    // Base state: [x, y, vx, vy]
    Q(0,0) = 10.0; Q(1,1) = 10.0; Q(2,2) = 1.0; Q(3,3) = 1.0;
    // Lifted vec(XX): small quadratic to avoid over-tightening
    Q.diagonal().segment(NX0, NX0*NX0).array() = tinytype(1e-2);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0*NU0, nux = NU0*NX0, nuu = NU0*NU0;
    // Base input: moderate cost
    R.diagonal().head(NU0).array() = tinytype(2.0);
    // Lifted input blocks: r_xx ~ 10 on XU/UX, R_xx ~ 500 on UU
    R.diagonal().segment(NU0, nxu).array() = tinytype(10.0);            // vec(XU)
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(10.0);      // vec(UX)
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(500.0); // vec(UU)
    Vec fdyn = Vec::Zero(nxL);

    // ADMM penalties for combined PSD + TV demo
    const tinytype rho = 5.0;
    const tinytype rho_psd = 1.0;
    
    TinySolver *solver = nullptr;
    int status = tiny_setup(&solver,
                            A, B, fdyn, Q, R,
                            rho, nxL, nuL, N,
                            /*verbose=*/1); // prints A,B,Q,R, Kinf, Pinf
    
    // Adaptive rho DISABLED - it doesn't converge well for PSD/TV problems
    solver->settings->adaptive_rho = 0;
    std::cout << "[PSD+TV-REG] Using fixed rho=" << solver->cache->rho << "\n";
    
    if (status) return status;

    // Bounds: cap base states/controls; add generous caps on lifted blocks
    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    // Gentle caps on vec(XX)
    x_min.middleRows(NX0, NX0*NX0).setConstant(-1000.0);
    x_max.middleRows(NX0, NX0*NX0).setConstant( 1000.0);

    Mat u_min = Mat::Constant(nuL, N-1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N-1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    // Gentle caps on lifted inputs
    u_min.bottomRows(nxu + nux + nuu).setConstant(-100.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 100.0);
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // Enable PSD coupling and store rho used
    {
        const bool ENABLE_PSD = true;
        if (ENABLE_PSD) {
            tiny_enable_psd(solver, NX0, NU0, rho_psd);
            std::cout << "[PSD+TV-REG] PSD coupling enabled (rho_psd=" << rho_psd << ")\n";
        }
    }

    // Lifted initial condition: [x0; vec(x0*x0')]
    Vec x0(NX0); x0 << -10, 0.1, 0, 0;
    Vec x0_lift(nxL); x0_lift.setZero();
    x0_lift.topRows(NX0) = x0;
    Mat X0 = x0 * x0.transpose();
    for (int i = 0; i < NX0; i++) {
        for (int j = 0; j < NX0; j++) {
            // column-major vec: stacks columns of X0
            x0_lift(NX0 + j*NX0 + i) = X0(i,j);
        }
    }
    tiny_set_x0(solver, x0_lift);

    // Linear lift-costs (make demo closer to julia_sdp.jl):
    // Add linear terms on diag(XX) and diag(UU) via Xref/Uref.
    // Encoding: q = -(Q .* Xref), r = -(R .* Uref) in update_linear_cost.
    {
        const int nx0 = NX0, nu0 = NU0;
        const int nxu_loc = nx0*nu0, nux_loc = nu0*nx0;
        const int baseUU = nu0 + nxu_loc + nux_loc; // start index of vec(UU)
        const tinytype q_xx = tinytype(1.0);  // linear weight on diag(XX)
        const tinytype r_uu = tinytype(10.0); // linear weight on diag(UU)

        Mat Xref = Mat::Zero(nxL, N);
        Mat Uref = Mat::Zero(nuL, N-1);

        // State: put q_xx on XX_ii by setting Xref(ii) = -q_xx / Q(ii)
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < nx0; ++i) {
                int idx_xx_ii = nx0 + i*nx0 + i;
                tinytype denom = solver->work->Q(idx_xx_ii);
                if (denom != tinytype(0)) Xref(idx_xx_ii, k) = -q_xx / denom;
            }
        }

        // Input: put r_uu on UU_jj via Uref
        for (int k = 0; k < N-1; ++k) {
            for (int j = 0; j < nu0; ++j) {
                int idx_uu_jj = baseUU + j*nu0 + j;
                tinytype denom = solver->work->R(idx_uu_jj);
                if (denom != tinytype(0)) Uref(idx_uu_jj, k) = -r_uu / denom;
            }
        }
        tiny_set_x_ref(solver, Xref);
        tiny_set_u_ref(solver, Uref);
    }

    // Pure-PSD lifted disks + TV-style base tangents sharing same effective radius
    const tinytype ox             = tinytype(DEMO_OBS_X);
    const tinytype oy             = tinytype(DEMO_OBS_Y);
    const tinytype obs_r          = tinytype(DEMO_OBS_R);
    const tinytype obs_margin     = tinytype(DEMO_OBS_MARGIN);
    const tinytype r_eff          = obs_r + obs_margin;
    const tinytype tangent_margin = tinytype(0.0); // PSD disks already inflate radius

    std::vector<std::array<tinytype,3>> disks = {{ {ox, oy, r_eff} }}; // one circle
    //tiny_set_lifted_disks(solver, disks);

    {
        const bool ENABLE_TV_TANGENTS = true;
        if (ENABLE_TV_TANGENTS) {
            tiny_enable_base_tangent_avoidance(solver, ox, oy, r_eff, tangent_margin);
            std::cout << "[PSD+TV-REG] Base tangents enabled (r_eff=" << r_eff << ")\n";
        }
    }

    // Solve once—watch the PSD eigen prints and TV tangent refresh logs
    tiny_solve(solver);
    int iters = solver->solution->iter;

    // Build a dynamics-consistent base rollout under the solved base controls
    Vec x_dyn = x0;  // base initial state [x,y,vx,vy]
    std::vector<Vec> Xdyn(N);
    Xdyn[0] = x_dyn;
    for (int k = 0; k < N-1; ++k) {
        Vec u_base = solver->solution->u.col(k).topRows(NU0);
        x_dyn = Ad * x_dyn + Bd * u_base; // + fdyn.topRows(NX0) if nonzero
        Xdyn[k+1] = x_dyn;
    }
    
    // Export solution to CSV for plotting
    std::ofstream csv_file("../psd_tv_combo_trajectory.csv");
    if (csv_file.is_open()) {
        // Header
        csv_file << "k,x1,x2,x3,x4,u1,u2,XX_11,XX_22,rank1_gap,signed_dist,iter\n";
        
        for (int k = 0; k < N; ++k) {
            // Dynamics-consistent base state
            Vec xk_dyn = Xdyn[k];
            double x1 = xk_dyn(0), x2 = xk_dyn(1), x3 = xk_dyn(2), x4 = xk_dyn(3);
            
            // Slack-based lifted state for diagnostics (XX, rank-1 gap)
            Vec xk = solver->solution->x.col(k);
            
            // XX matrix diagonal elements (for checking rank-1)
            Mat XX_vec(NX0, NX0);
            for (int j = 0; j < NX0; ++j) {
                for (int i = 0; i < NX0; ++i) {
                    XX_vec(i,j) = xk(NX0 + j*NX0 + i); // column-major
                }
            }
            double XX_11 = XX_vec(0,0);
            double XX_22 = XX_vec(1,1);
            
            // Rank-1 gap for this stage
            Vec x_base = xk.topRows(NX0);
            double gap = (XX_vec - x_base * x_base.transpose()).norm();
            
            csv_file << k << "," << x1 << "," << x2 << "," << x3 << "," << x4;
            
            if (k < N-1) {
                Vec uk = solver->solution->u.col(k);
                csv_file << "," << uk(0) << "," << uk(1);
            } else {
                csv_file << ",0,0";  // No control at terminal stage
            }
            
            // Signed distance to obstacle used in demo (measured to the effective radius r_eff)
            double sd = std::sqrt((x1-ox)*(x1-ox) + (x2-oy)*(x2-oy)) - r_eff;

            csv_file << "," << XX_11 << "," << XX_22 << "," << gap << "," << sd << "," << iters << "\n";
        }
        csv_file.close();
        std::cout << "\n[PSD+TV-REG] Trajectory saved to psd_tv_combo_trajectory.csv\n";
    }

    // Simple summary of safety margin
    double min_sd = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        Vec xk = solver->solution->x.col(k);
        double sd = std::sqrt((xk(0)-ox)*(xk(0)-ox) + (xk(1)-oy)*(xk(1)-oy)) - r_eff;
        if (sd < min_sd) min_sd = sd;
    }
    std::cout << "[PSD+TV-REG] Min signed distance to obstacle: " << min_sd << "\n";
    
    return 0;
}


