#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <tinympc/tiny_api.hpp>

// ---------------- Time-varying linear constraint helpers --------------------
inline int tiny_enable_tv_state_linear(TinySolver* solver, int n_constr) {
    if (!solver) { std::cout << "tiny_enable_tv_state_linear: solver nullptr\n"; return 1; }
    solver->settings->en_tv_state_linear = 1;
    solver->work->numtvStateLinear = n_constr;
    solver->work->tv_Alin_x = tinyMatrix::Zero(n_constr * solver->work->N, solver->work->nx);
    solver->work->tv_blin_x = tinyMatrix::Zero(n_constr, solver->work->N);
    solver->work->vlnew_tv = solver->work->x;
    solver->work->gl_tv    = tinyMatrix::Zero(solver->work->nx, solver->work->N);
    return 0;
}

// Base-level tangent half-space update per-stage using the latest rollout x
// a^T z <= b form, where only base (x,y) entries in a are nonzero.
inline void tiny_update_base_tangent_avoidance_tv(
    TinySolver* solver, tinytype ox, tinytype oy, tinytype r, tinytype margin)
{
    const int N   = solver->work->N;
    const int nxL = solver->work->nx;
    const int nc  = std::max(1, solver->work->numtvStateLinear);

    for (int k = 0; k < N; ++k) {
        tinytype x = solver->work->x(0,k);
        tinytype y = solver->work->x(1,k);
        tinytype dx = x - ox;
        tinytype dy = y - oy;
        tinytype d  = std::sqrt(dx*dx + dy*dy);

        // Normal n = (dx,dy)/||dx,dy||. Use a safe default when near zero.
        tinytype nx = 1.0, ny = 0.0;
        if (d > tinytype(1e-8)) { nx = dx / d; ny = dy / d; }

        // Half-space: n^T [x;y] >= n^T [ox;oy] + r + margin
        // Convert to a^T z <= b with a = -[n_x, n_y, 0,...], b = -(n^T o + r + margin)
        tinyVector a = tinyVector::Zero(nxL);
        a(0) = -nx; a(1) = -ny;
        tinytype b = - (nx*ox + ny*oy + r + margin);

        const int row = k*nc + 0;
        if (row >= 0 && row < solver->work->tv_Alin_x.rows()) {
            solver->work->tv_Alin_x.row(row) = a.transpose();
        }
        if (solver->work->tv_blin_x.rows() >= 1 && k < solver->work->tv_blin_x.cols()) {
            solver->work->tv_blin_x(0,k) = b;
        }
    }
}

// Convenience: enable base-tangent avoidance from user code.
inline int tiny_enable_base_tangent_avoidance(
    TinySolver* solver, tinytype ox, tinytype oy, tinytype r, tinytype margin)
{
    if (!solver) { std::cout << "tiny_enable_base_tangent_avoidance: solver nullptr\n"; return 1; }
    // Ensure time-varying state linear constraints are allocated (1 per stage)
    tiny_enable_tv_state_linear(solver, 1);
    solver->settings->en_tv_state_linear = 1;
    solver->settings->en_base_tangent_tv = 1;
    solver->settings->obs_x = ox;
    solver->settings->obs_y = oy;
    solver->settings->obs_r = r;
    solver->settings->obs_margin = margin;
    return 0;
}

