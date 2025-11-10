#include <iostream>
#include <fstream>
#include <limits>
#include <Eigen/Dense>
#include <tinympc/tiny_api.hpp>
#include "../src/tinympc/psd_support.hpp" // for TV linearization helpers

#define NX 4
#define NU 2
#define N  31

extern "C" int main() {
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

    // Base double integrator dynamics (like julia_sdp.jl base part)
    Mat Ad(NX, NX); Ad << 1,0,1,0,  0,1,0,1,  0,0,1,0,  0,0,0,1;
    Mat Bd(NX, NU); Bd << 0.5,0,  0,0.5,  1,0,  0,1;

    // Quadratic weights on base states/inputs
    Mat Q = Mat::Zero(NX, NX);
    Q(0,0) = 10.0; Q(1,1) = 10.0; Q(2,2) = 1.0; Q(3,3) = 1.0;
    Mat R = Mat::Zero(NU, NU);
    R(0,0) = 2.0; R(1,1) = 2.0;
    Vec fdyn = Vec::Zero(NX);

    // Setup solver
    TinySolver *solver = nullptr;
    int status = tiny_setup(&solver,
                            Ad, Bd, fdyn, Q, R,
                            /*rho*/ tinytype(12.0), NX, NU, N,
                            /*verbose=*/1);
    if (status) return status;

    // Bounds: cap base states/controls
    Mat x_min = Mat::Constant(NX, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(NX, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX).setConstant(-30.0);
    x_max.topRows(NX).setConstant( 30.0);

    Mat u_min = Mat::Constant(NU, N-1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(NU, N-1,  std::numeric_limits<tinytype>::infinity());
    u_min.setConstant(-3.0);
    u_max.setConstant( 3.0);
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // Initial condition
    Vec x0(NX); x0 << -10, 0.1, 0, 0;
    tiny_set_x0(solver, x0);

    // No linear references (pure quadratic cost)
    tiny_set_x_ref(solver, Mat::Zero(NX, N));
    tiny_set_u_ref(solver, Mat::Zero(NU, N-1));

    // Enable time-varying base tangent half-spaces around a circular obstacle
    const tinytype ox = -5.0, oy = 0.0, r = 2.0, margin = 0.0;
    tiny_enable_base_tangent_avoidance(solver, ox, oy, r, margin);

    // Solve
    tiny_solve(solver);

    // Export solution to CSV
    std::ofstream csv_file("../tv_linear_trajectory.csv");
    if (csv_file.is_open()) {
        csv_file << "k,x1,x2,x3,x4,u1,u2\n";
        for (int k = 0; k < N; ++k) {
            Vec xk = solver->solution->x.col(k);
            csv_file << k << "," << xk(0) << "," << xk(1) << "," << xk(2) << "," << xk(3);
            if (k < N-1) {
                Vec uk = solver->solution->u.col(k);
                csv_file << "," << uk(0) << "," << uk(1) << "\n";
            } else {
                csv_file << ",0,0\n";
            }
        }
        csv_file.close();
        std::cout << "\n[EXPORT] TV-linear trajectory saved to ../tv_linear_trajectory.csv\n";
    }

    return 0;
}

