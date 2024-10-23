#include "RK4.h"

RK4::RK4(MKL_INT dim)
:dim_(dim)
{
  k1_.resize(dim_, 0.0);
  k2_.resize(dim_, 0.0);
  k3_.resize(dim_, 0.0);
  k4_.resize(dim_, 0.0);
  Rho0c1_.resize(dim_, 0.0);
  Rho0c2_.resize(dim_, 0.0);
  Rho0c3_.resize(dim_, 0.0);
}

RK4::~RK4()
{}

void RK4::multiply_(MZType &Res,
                    MZType &Red,
                    MZType &Rho)
{
  std::complex<double> alpp(1.0, 0.0);
  std::complex<double> bet(0.0, 0.0);

  cblas_zgemv(CblasRowMajor, CblasNoTrans, dim_, dim_, &alpp, &Red[0],
              dim_, &Rho[0], 1, &bet, &Res[0], 1);
}

void RK4::rk4_redfield(double delta_t,
                       MZType &Rho0,
                       MZType &Red)
{
  cblas_zcopy(dim_, &Rho0[0], 1, &Rho0c1_[0], 1);
  cblas_zcopy(dim_, &Rho0[0], 1, &Rho0c2_[0], 1);
  cblas_zcopy(dim_, &Rho0[0], 1, &Rho0c3_[0], 1);
  // k1 
  multiply_(k1_, Red, Rho0);
  
  // k2
  std::complex<double> h2(delta_t / 2.0, 0.0);
  cblas_zaxpy(dim_, &h2, &k1_[0], 1, &Rho0c1_[0], 1);
  multiply_(k2_, Red, Rho0c1_);

  // k3
  cblas_zaxpy(dim_, &h2, &k2_[0], 1, &Rho0c2_[0], 1);
  multiply_(k3_, Red, Rho0c2_);

  // k4
  std::complex<double> h(delta_t, 0.0);
  cblas_zaxpy(dim_, &h, &k3_[0], 1, &Rho0c3_[0], 1);
  multiply_(k4_, Red, Rho0c3_);

  // yt = y0 + h/6 (k1 + 2k2 + 2k3 + k4)
  std::complex<double> h6(delta_t / 6.0, 0.0);
  std::complex<double> h3(delta_t / 3.0, 0.0);
  cblas_zaxpy(dim_, &h6, &k1_[0], 1, &Rho0[0], 1);
  cblas_zaxpy(dim_, &h3, &k2_[0], 1, &Rho0[0], 1);
  cblas_zaxpy(dim_, &h3, &k3_[0], 1, &Rho0[0], 1);
  cblas_zaxpy(dim_, &h6, &k4_[0], 1, &Rho0[0], 1);
}
