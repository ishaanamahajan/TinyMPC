#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>
#include <tinympc/tiny_api.hpp>

// Manual Kronecker product implementation
template<typename Derived1, typename Derived2>
Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, Eigen::Dynamic>
kronecker(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& B) {
    Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, Eigen::Dynamic> result(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            result.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j) * B;
        }
    }
    return result;
}

// Build lifted (A,B) exactly like the Julia file
inline void tiny_build_lifted_from_base(
    const Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& Ad,
    const Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& Bd,
    Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& A_out,
    Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& B_out)
{
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    const int nx0 = Ad.rows();
    const int nu0 = Bd.cols();
    const int nxx = nx0*nx0;
    const int nxu = nx0*nu0;
    const int nux = nu0*nx0;
    const int nuu = nu0*nu0;

    A_out.setZero(nx0 + nxx, nx0 + nxx);
    A_out.block(0,0,nx0,nx0)            = Ad;
    A_out.block(nx0,nx0,nxx,nxx)        = kronecker(Ad, Ad);

    B_out.setZero(nx0 + nxx, nu0 + nxu + nux + nuu);
    B_out.block(0,0,nx0,nu0)            = Bd;
    B_out.block(nx0,     nu0,           nxx, nxu) = kronecker(Bd, Ad);
    B_out.block(nx0,     nu0 + nxu,     nxx, nux) = kronecker(Ad, Bd);
    B_out.block(nx0,     nu0 + nxu+nux, nxx, nuu) = kronecker(Bd, Bd);

    std::cout << "[PSD] Built lifted A(" << A_out.rows() << "x" << A_out.cols()
              << ") B(" << B_out.rows() << "x" << B_out.cols() << ")\n";
}

// Enable PSD block (stores nx0, nu0 and allocates S/H buffers)
inline int tiny_enable_psd(TinySolver* solver, int nx0, int nu0, tinytype rho_psd)
{
    if (!solver) { std::cout << "tiny_enable_psd: solver nullptr\n"; return 1; }
    solver->settings->en_psd   = 1;
    solver->settings->nx0_psd  = nx0;
    solver->settings->nu0_psd  = nu0;
    solver->cache->rho_psd     = rho_psd;

    const int psd_dim = 1 + nx0 + nu0;
    const int N = solver->work->N;

    solver->work->Spsd     = tinyMatrix::Zero(psd_dim*psd_dim, N);
    solver->work->Spsd_new = tinyMatrix::Zero(psd_dim*psd_dim, N);
    solver->work->Hpsd     = tinyMatrix::Zero(psd_dim*psd_dim, N);

    std::cout << "[PSD] Enabled: nx0=" << nx0 << " nu0=" << nu0
              << " psd_dim=" << psd_dim << " rho_psd=" << rho_psd << "\n";
    return 0;
}

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

// Enable non-time-varying linear state constraints with n_constr rows
inline int tiny_enable_state_linear(TinySolver* solver, int n_constr) {
    if (!solver) { std::cout << "tiny_enable_state_linear: solver nullptr\n"; return 1; }
    solver->settings->en_state_linear = 1;
    solver->work->numStateLinear = n_constr;
    solver->work->Alin_x = tinyMatrix::Zero(n_constr, solver->work->nx);
    solver->work->blin_x = tinyVector::Zero(n_constr);
    // Initialize associated slack/dual storage
    solver->work->vlnew = solver->work->x;
    solver->work->gl    = tinyMatrix::Zero(solver->work->nx, solver->work->N);
    return 0;
}

// Pure-PSD variant: add lifted disk constraints (no TV relinearization)
// Each disk j with center (ox,oy) and radius r contributes a row
// m^T [x; vec(XX)] >= n  where m = [-2ox, -2oy, e11, e22] and n = r^2 - ||o||^2.
// We store a^T z <= b via a = -m, b = -n into Alin_x, blin_x and enable en_state_linear.
inline int tiny_set_lifted_disks(
    TinySolver* solver,
    const std::vector<std::array<tinytype,3>>& disks)
{
    if (!solver) { std::cout << "tiny_set_lifted_disks: solver nullptr\n"; return 1; }
    const int nx0 = solver->settings->nx0_psd;
    const int nxL = solver->work->nx;
    if (nx0 <= 0) { std::cout << "tiny_set_lifted_disks: nx0_psd not set (>0)\n"; return 1; }
    const int m = static_cast<int>(disks.size());
    if (m == 0) return 0;

    tinyMatrix Alin_x = tinyMatrix::Zero(m, nxL);
    tinyVector blin_x = tinyVector::Zero(m);

    const int idx_xx11 = nx0 + 0 + 0*nx0;
    const int idx_xx22 = nx0 + 1 + 1*nx0;

    for (int j = 0; j < m; ++j) {
        const tinytype ox = disks[j][0];
        const tinytype oy = disks[j][1];
        const tinytype r  = disks[j][2];

        tinyVector mrow = tinyVector::Zero(nxL);
        mrow(0) = -2 * ox; mrow(1) = -2 * oy;
        mrow(idx_xx11) = 1; mrow(idx_xx22) = 1;
        tinytype n = (r*r - (ox*ox + oy*oy));

        tinyVector a = -mrow;
        tinytype   b = -n;
        Alin_x.row(j) = a.transpose();
        blin_x(j) = b;
    }

    // Enable and set in solver
    tiny_enable_state_linear(solver, m);
    return tiny_set_linear_constraints(
        solver,
        Alin_x,
        blin_x,
        tinyMatrix::Zero(0, solver->work->nu),
        tinyVector::Zero(0));
}

// Append additional non-TV state linear rows to current Alin_x/blin_x
inline int tiny_append_state_linear_rows(
    TinySolver* solver,
    const tinyMatrix& A_extra,
    const tinyVector& b_extra)
{
    if (!solver) { std::cout << "tiny_append_state_linear_rows: solver nullptr\n"; return 1; }
    if (A_extra.rows() == 0) return 0;
    if (A_extra.cols() != solver->work->nx) { std::cout << "tiny_append_state_linear_rows: bad cols\n"; return 1; }
    if (b_extra.rows() != A_extra.rows()) { std::cout << "tiny_append_state_linear_rows: mismatched b rows\n"; return 1; }

    tinyMatrix A_old = solver->work->Alin_x;
    tinyVector b_old = solver->work->blin_x;

    const int m_old = A_old.rows();
    const int m_new = m_old + A_extra.rows();
    tinyMatrix A_cat = tinyMatrix::Zero(m_new, solver->work->nx);
    tinyVector b_cat = tinyVector::Zero(m_new);
    if (m_old > 0) {
        A_cat.topRows(m_old) = A_old;
        b_cat.topRows(m_old) = b_old;
    }
    A_cat.bottomRows(A_extra.rows()) = A_extra;
    b_cat.bottomRows(b_extra.rows()) = b_extra;

    solver->work->numStateLinear = m_new;
    solver->settings->en_state_linear = 1;
    return tiny_set_linear_constraints(
        solver,
        A_cat,
        b_cat,
        solver->work->Alin_u,
        solver->work->blin_u);
}

// Add light McCormick/RLT constraints tying XX(0:1,0:1) to x(0:1) using current bounds.
// For i==j (diagonal): add upper bound w_ii <= (u_i + l_i) x_i - u_i l_i.
// For i!=j: add 4 McCormick inequalities for w_ij = x_i x_j.
inline int tiny_add_rlt_position_xx(TinySolver* solver)
{
    if (!solver) { std::cout << "tiny_add_rlt_position_xx: solver nullptr\n"; return 1; }
    const int nx0 = solver->settings->nx0_psd;
    if (nx0 < 2) { std::cout << "tiny_add_rlt_position_xx: nx0_psd<2\n"; return 1; }
    const int nxL = solver->work->nx;

    // Assume constant bounds over N; read from k=0
    auto get_bounds = [&](int idx)->std::pair<tinytype,tinytype>{
        tinytype l = solver->work->x_min(idx, 0);
        tinytype u = solver->work->x_max(idx, 0);
        return {l,u};
    };

    auto idx_xx = [&](int i, int j)->int { return nx0 + j*nx0 + i; };

    std::vector<Eigen::Matrix<tinytype, Eigen::Dynamic, 1>> rows;
    std::vector<tinytype> bs;

    for (int i = 0; i < 2; ++i) {
        auto [li, ui] = get_bounds(i);
        // Diagonal upper bound: w_ii - (ui+li) x_i <= -ui*li
        tinyVector a = tinyVector::Zero(nxL);
        a(i) = -(ui + li);
        a(idx_xx(i,i)) = 1.0;
        rows.push_back(a);
        bs.push_back(-ui*li);
    }

    // Off-diagonal (0,1) and (1,0)
    for (int p = 0; p < 2; ++p) {
        int i = (p == 0) ? 0 : 1;
        int j = (p == 0) ? 1 : 0;
        auto [li, ui] = get_bounds(i);
        auto [lj, uj] = get_bounds(j);
        int w = idx_xx(i,j);

        // 1) -w + lj x_i + li x_j <= li*lj
        {
            tinyVector a = tinyVector::Zero(nxL);
            a(w) = -1.0;
            a(i) = lj;
            a(j) = li;
            rows.push_back(a);
            bs.push_back(li*lj);
        }
        // 2) -w + uj x_i + ui x_j <= ui*uj
        {
            tinyVector a = tinyVector::Zero(nxL);
            a(w) = -1.0;
            a(i) = uj;
            a(j) = ui;
            rows.push_back(a);
            bs.push_back(ui*uj);
        }
        // 3) w - uj x_i - li x_j <= -li*uj
        {
            tinyVector a = tinyVector::Zero(nxL);
            a(w) = 1.0;
            a(i) = -uj;
            a(j) = -li;
            rows.push_back(a);
            bs.push_back(-li*uj);
        }
        // 4) w - lj x_i - ui x_j <= -ui*lj
        {
            tinyVector a = tinyVector::Zero(nxL);
            a(w) = 1.0;
            a(i) = -lj;
            a(j) = -ui;
            rows.push_back(a);
            bs.push_back(-ui*lj);
        }
    }

    const int m = static_cast<int>(rows.size());
    if (m == 0) return 0;
    tinyMatrix Aadd = tinyMatrix::Zero(m, nxL);
    tinyVector Badd = tinyVector::Zero(m);
    for (int r = 0; r < m; ++r) { Aadd.row(r) = rows[r].transpose(); Badd(r) = bs[r]; }
    return tiny_append_state_linear_rows(solver, Aadd, Badd);
}

// Add only diagonal RLT caps for position block: w_ii <= (u_i + l_i) x_i - u_i l_i
inline int tiny_add_rlt_position_xx_diag_only(TinySolver* solver)
{
    if (!solver) { std::cout << "tiny_add_rlt_position_xx_diag_only: solver nullptr\n"; return 1; }
    const int nx0 = solver->settings->nx0_psd;
    if (nx0 < 2) { std::cout << "tiny_add_rlt_position_xx_diag_only: nx0_psd<2\n"; return 1; }
    const int nxL = solver->work->nx;

    auto get_bounds = [&](int idx)->std::pair<tinytype,tinytype>{
        tinytype l = solver->work->x_min(idx, 0);
        tinytype u = solver->work->x_max(idx, 0);
        return {l,u};
    };
    auto idx_xx = [&](int i, int j)->int { return nx0 + j*nx0 + i; };

    const int m = 2; // only i=0,1
    tinyMatrix Aadd = tinyMatrix::Zero(m, nxL);
    tinyVector Badd = tinyVector::Zero(m);
    for (int i = 0; i < 2; ++i) {
        auto [li, ui] = get_bounds(i);
        tinyVector a = tinyVector::Zero(nxL);
        a(i) = -(ui + li);
        a(idx_xx(i,i)) = 1.0;
        Aadd.row(i) = a.transpose();
        Badd(i) = -ui*li;
    }
    return tiny_append_state_linear_rows(solver, Aadd, Badd);
}

inline void tiny_set_circle_avoidance(TinySolver* solver, tinytype ox, tinytype oy, tinytype r) {
    const int nx0 = solver->settings->nx0_psd; // base state dim
    const int nxL = solver->work->nx;          // lifted state dim
    const int N   = solver->work->N;

    tinyVector m = tinyVector::Zero(nxL);
    m(0) = -2 * ox; m(1) = -2 * oy;
    const int idx_xx11 = nx0 + 0 + 0*nx0;
    const int idx_xx22 = nx0 + 1 + 1*nx0;
    m(idx_xx11) = 1; m(idx_xx22) = 1;
    // From (x-obs)'(x-obs) >= r^2  => (x'x - 2 obs'x) >= r^2 - ||obs||^2
    tinytype n = (r*r - (ox*ox + oy*oy));

    tinyVector a = -m; tinytype b = -n; // a^T z <= b
    for (int k = 0; k < N; ++k) {
        solver->work->tv_Alin_x.row(k * 1 + 0) = a.transpose();
        solver->work->tv_blin_x(0, k) = b;
    }
}

// Optional: base corridor x >= xmin as a second TV-linear constraint row
inline void tiny_set_xmin_halfspace(TinySolver* solver, tinytype xmin) {
    const int nxL = solver->work->nx;
    const int N   = solver->work->N;
    const int stride = std::max(1, solver->work->numtvStateLinear);
    if (stride < 2) return; // require allocation with at least 2 rows per stage

    tinyVector a = tinyVector::Zero(nxL);
    a(0) = -1; // -x <= -xmin  =>  x >= xmin
    tinytype b = -xmin;
    for (int k = 0; k < N; ++k) {
        solver->work->tv_Alin_x.row(k * stride + 1) = a.transpose();
        solver->work->tv_blin_x(1, k) = b;
    }
}
