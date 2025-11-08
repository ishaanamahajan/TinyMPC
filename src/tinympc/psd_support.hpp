#pragma once

#include <Eigen/Dense>
#include <iostream>
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
