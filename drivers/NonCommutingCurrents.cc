// Redfield calculations
/*******************************************************************************/
// NESS heat current
// Spin-boson model with two reservoirs
// Reservoirs at different temperature
// The couplings are the twisted cos(t)sz + sin(t)sx
// Different values of t for each side
/*******************************************************************************/

#include <iostream>
#include <numeric>
#include <algorithm>
#include <complex>
#include <sys/time.h>
#include <omp.h>

#include "../Utils/Utils.h"
#include "../Redfield/Redfield.h"
#include "../RungeKutta/RK4.h"

double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

std::vector<double> linspace(double a, double b, size_t n)
{
  double h = (b - a) / static_cast<double>(n - 1);
  std::vector<double> vec(n);
  typename std::vector<double>::iterator x;
  double val;
  for (x = vec.begin(), val = a; x != vec.end(); ++x, val += h)
    *x = val;
  return vec;
}

double ohmic_spectrum(double w, double t)
{
  double ga = 0.05; // Coupling to the residual reservoir
  double wc = 1000.0; // Cut-off frequency
  double aw = std::abs(w);
  double n_b = 1.0 / (std::exp(aw / t) - (1.0 * (w != 0)));
  double j_s = ga * aw * std::exp(-1.0 * aw / wc);
  return (j_s * (n_b + 1.0) * (w > 0)) + (j_s * n_b * (w < 0)) + (ga * t * (w == 0.0));
}

int main(int argc, char **argv)
{
  MKL_INT rc = 777; // Local dimension of the reaction cordinates
  double eps = 1.0; // Splitting of the spin
  // z is the bath with sz, x is the bath with sx
  double om = 8.0; // Self-energy of the reaction coordinates
  double la_l = 0.777; // Coupling strength of between spin and reaction coordinate left
  double la_r = 0.777; // Coupling strength of between spin and reaction coordinate right
  double t_l = 0.777; // Temperature left
  double t_r = 0.777; // Temperature right
  double theta_l = 0.777; // Phase of the coupling left
  double theta_r = 0.777; //Phase of the coupling right

  // Parsing arguments
  if(argc != 15){
    std::cerr << "Usage: " << argv[0] << " --rc [RC] --la_l [la_l] --la_r [la_r] --t_l [t_l] --t_r [t_r] --theta_l [theta_l] --theta_r [theta_r]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--rc") rc = atoi(argv[i + 1]);
    else if(str == "--la_l") la_l = atof(argv[i + 1]);
    else if(str == "--la_r") la_r = atof(argv[i + 1]);
    else if(str == "--t_l") t_l = atof(argv[i + 1]);
    else if(str == "--t_r") t_r = atof(argv[i + 1]);
    else if(str == "--theta_l") theta_l = atof(argv[i + 1]);
    else if(str == "--theta_r") theta_r = atof(argv[i + 1]);
    else continue;
  }
  if(rc == 777 || la_l == 0.777 || la_r == 0.777 || t_l == 0.777 || t_r == 0.777 || theta_l == 0.777 || theta_r == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --rc [RC] --la_l [la_l] --la_r [la_r] --t_l [t_l] --t_r [t_r] --theta_l [theta_l] --theta_r [theta_r]" << std::endl;
    exit(1);
  }

  std::cout << std::fixed;
  std::cout.precision(4);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# RC = " << rc << std::endl;
  std::cout << "# Om = " << om << std::endl;
  std::cout << "# la_l = " << la_l << " la_r = " << la_r << std::endl;
  std::cout << "# t_l = " << t_l << " t_r = " << t_r << std::endl;
  std::cout << "# theta_l = " << theta_l << " theta_r = " << theta_r << std::endl;

  // Coupling operators
  CType zero_t(0.0, 0.0);
  MZType stheta_l = {zero_t, zero_t, zero_t, zero_t};
  MZType stheta_r = {zero_t, zero_t, zero_t, zero_t};
  for(MKL_INT i = 0; i < 4; ++i){
    stheta_l[i] = (std::cos(theta_l) * Utils::sigz[i]) + (std::sin(theta_l) * Utils::sigx[i]);
    stheta_r[i] = (std::cos(theta_r) * Utils::sigz[i]) + (std::sin(theta_r) * Utils::sigx[i]);
  }

  // Hamiltonian
  MKL_INT size_s = 1 << 1;
  MZType SigmaZ = Utils::sigz;
  MZType Ham = Utils::kron(eps, 
                           Utils::sigz,
                           Utils::kron(1.0, Utils::eye(rc), Utils::eye(rc)));
  MZType t2 = Utils::kron(om,
                          Utils::eye(2),
                          Utils::kron(1.0, Utils::ata(rc), Utils::eye(rc)));
  MZType t3 = Utils::kron(om, 
                          Utils::eye(2),
                          Utils::kron(1.0, Utils::eye(rc), Utils::ata(rc)));
  MZType t4 = Utils::kron(la_l, 
                          stheta_l,
                          //Utils::sigx,
                          Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType t5 = Utils::kron(la_r, 
                          stheta_r,
                          //Utils::sigy,
                          Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  MZType null_s(size_s * size_s, 0.0);
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  Utils::add(Ham, alpha, t2, t3);
  Utils::add(Ham, alpha, t4, t5);

  // Interaction Hamiltonian between RCs and residual baths, system OPs
  MZType V_l = Utils::kron(1.0, 
                     Utils::eye(2),
                     Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType V_r = Utils::kron(1.0, 
                     Utils::eye(2),
                     Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  // Redfield tensor
  double time1 = seconds();
  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  Redfield redT(ham_dim);

  MZType Red_l((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Red_r((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Coh((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  redT.construct_redfield_2r_phonon(Coh, Red_l, Red_r, Ham, V_l, V_r, 
          &ohmic_spectrum, &ohmic_spectrum, t_l, t_r);
  double time2 = seconds();

  std::cout << "# Time Redfield: " << time2 - time1 << std::endl;
  // Write a superoperator onto Coh
  cblas_zaxpy(Coh.size(), &alpha, &Red_l[0], 1, &Coh[0], 1);
  cblas_zaxpy(Coh.size(), &alpha, &Red_r[0], 1, &Coh[0], 1);

  // Compute steady state
  time1 = seconds();
  MZType steady_state = redT.get_steady_state(Coh);
  time2 = seconds();

  std::cout << "# Time SS solution: " << time2 - time1 << std::endl;

  // Compute heat current
  time1 = seconds();
  MZType rho_diss(ham_dim * ham_dim, 0.0);
  cblas_zgemv(CblasRowMajor,
              CblasNoTrans,
              ham_dim * ham_dim,
              ham_dim * ham_dim,
              &alpha,
              &Red_l[0],
              ham_dim * ham_dim,
              &steady_state[0],
              1,
              &beta,
              &rho_diss[0],
              1);
  // Trace
  CType h_curr(0.0, 0.0);
  for(MKL_INT i = 0; i < ham_dim; ++i){
    h_curr += rho_diss[(i * ham_dim) + i] * redT.eigvals[i];
  }
  time2 = seconds();

  std::cout << "# Time Expc value: " << time2 - time1 << std::endl;

#if 0
  MZType Rho(ham_dim * ham_dim, 0.0);
  for(MKL_INT i = 0; i < ham_dim; ++i){
    for(MKL_INT j = 0; j < ham_dim; ++j){
      Rho[(i * ham_dim) + j] = steady_state[(j * ham_dim) + i];
    }
  }
  // Reduced state
  // Rotate back to tensor basis
  MZType buffer(ham_dim * ham_dim, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
              &Ham[0], ham_dim, &Rho[0], ham_dim, &beta, &buffer[0], ham_dim);  
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, ham_dim, ham_dim, ham_dim, &alpha,
              &buffer[0], ham_dim, &Ham[0], ham_dim, &beta, &Rho[0], ham_dim);
  // Partial trace over RC
  MZType RedRho = Utils::partial_trace_rc(Rho, size_s, rc * rc); 
#endif
  std::cout << std::scientific;
  std::cout.precision(6);
  //std::cout << theta_l << " " << theta_r << " " << h_curr.real() << std::endl;
  std::cout << la_l << " " << la_r << " " << h_curr.real() << std::endl;
  //CType im_const = {0.0, 1.0};
  //CType x_val = RedRho[1] + RedRho[2];
  //CType y_val = im_const * (RedRho[1] - RedRho[2]);
  //CType z_val = RedRho[0] - RedRho[3];
  //std::cout << la_l << " " << la_r << " " << x_val.real() << " " << y_val.real() << " " << z_val.real() << std::endl;

  return 0;
}
