#include <iostream>
#include <fstream>
#include <limits>
#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>
#include "../src/tinympc/psd_support.hpp"

#define NX0 4
#define NU0 2
#define N   31

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
    // Base state: [x, y, vx, vy] (slightly stronger to avoid saturation)
    Q(0,0) = 10.0; Q(1,1) = 10.0; Q(2,2) = 1.0; Q(3,3) = 1.0;
    // Lifted vec(XX): tiny quadratic (q_xx ~ 1e-6) to avoid over‑tightening
    Q.diagonal().segment(NX0, NX0*NX0).array() = tinytype(1e-6);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0*NU0, nux = NU0*NX0, nuu = NU0*NU0;
    // Base input: moderate cost
    R.diagonal().head(NU0).array() = tinytype(2.0);
    // Lifted input blocks: r_xx ~ 10 on XU/UX, R_xx ~ 500 on UU
    R.diagonal().segment(NU0, nxu).array() = tinytype(10.0);            // vec(XU)
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(10.0);      // vec(UX)
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(500.0); // vec(UU)
    Vec fdyn = Vec::Zero(nxL);

    TinySolver *solver = nullptr;
    int status = tiny_setup(&solver,
                            A, B, fdyn, Q, R,
                            /*rho*/ tinytype(12.0), nxL, nuL, N,
                            /*verbose=*/1); // prints A,B,Q,R, Kinf, Pinf
    if (status) return status;

    // Bounds: cap base states/controls; add generous caps on lifted blocks to avoid numeric blow-up
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
    u_min.bottomRows(nxu + nux + nuu).setConstant(-1000.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 1000.0);
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // Enable PSD coupling with a small rho_psd initially (0.5–1.0)
    const bool ENABLE_PSD = true;
    if (ENABLE_PSD) {
        // Start modest; can ramp to 5 later if calm
        tiny_enable_psd(solver, NX0, NU0, /*rho_psd*/ tinytype(3.0));
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

    // No linear reference terms
    tiny_set_x_ref(solver, Mat::Zero(nxL, N));
    tiny_set_u_ref(solver, Mat::Zero(nuL, N-1));

    // Pure-PSD variant: lifted disks + light RLT (no TV tangents)
    const tinytype ox = -5.0, oy = 0.0, r = 2.0;
    std::vector<std::array<tinytype,3>> disks = {{ {ox, oy, r} }}; // one circle
    tiny_set_lifted_disks(solver, disks);
    tiny_add_rlt_position_xx(solver);

    // Solve once—watch the PSD eigen prints
    tiny_solve(solver);
    
    // Export solution to CSV for plotting
    std::ofstream csv_file("../psd_trajectory.csv");
    if (csv_file.is_open()) {
        // Header
        csv_file << "k,x1,x2,x3,x4,u1,u2,XX_11,XX_22,rank1_gap\n";
        
        for (int k = 0; k < N; ++k) {
            // Extract solution (from solution->x which has final projected values)
            Vec xk = solver->solution->x.col(k);
            
            // Position and velocity (first 4 states)
            double x1 = xk(0), x2 = xk(1), x3 = xk(2), x4 = xk(3);
            
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
            
            csv_file << "," << XX_11 << "," << XX_22 << "," << gap << "\n";
        }
        csv_file.close();
        std::cout << "\n[EXPORT] Trajectory saved to ../psd_trajectory.csv\n";
    }
    
    return 0;
}
