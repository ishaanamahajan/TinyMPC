#include <iostream>
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

    // Tiny diagonal Q/R (keep tiny for now)
    Mat Q = Mat::Identity(nxL, nxL) * tinytype(1e-6);
    Mat R = Mat::Identity(nuL, nuL) * tinytype(1e-6);
    Vec fdyn = Vec::Zero(nxL);

    TinySolver *solver = nullptr;
    int status = tiny_setup(&solver,
                            A, B, fdyn, Q, R,
                            /*rho*/ tinytype(1.0), nxL, nuL, N,
                            /*verbose=*/1); // prints A,B,Q,R, Kinf, Pinf
    if (status) return status;

    // No bounds
    Mat x_min = Mat::Constant(nxL, N, -1e17);
    Mat x_max = Mat::Constant(nxL, N,  1e17);
    Mat u_min = Mat::Constant(nuL, N-1, -1e17);
    Mat u_max = Mat::Constant(nuL, N-1,  1e17);
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    tiny_enable_psd(solver, NX0, NU0, /*rho_psd*/ tinytype(1.0));

    // Lifted initial condition: [x0; vec(x0*x0')]
    Vec x0(NX0); x0 << -10, 0.1, 0, 0;
    Vec x0_lift(nxL); x0_lift.setZero();
    x0_lift.topRows(NX0) = x0;
    Mat X0 = x0 * x0.transpose();
    for (int i = 0; i < NX0; i++) {
        for (int j = 0; j < NX0; j++) {
            x0_lift(NX0 + i*NX0 + j) = X0(i,j);
        }
    }

    tiny_set_x0(solver, x0_lift);

    // Solve once—watch the PSD eigen prints
    tiny_solve(solver);
    return 0;
}
