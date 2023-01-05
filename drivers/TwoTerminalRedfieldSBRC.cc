// Redfield calculations
/*******************************************************************************/
// Computes the heat current in the two-terminal set-up
// A reaction coordinate is extracted from each reservoir, each of which is a
// single bosonic mode with finite dimension m
/*******************************************************************************/

#include <iostream>
#include <numeric>
#include <algorithm>
#include <complex>
#include <sys/time.h>
#include <omp.h>

#include "../Utils/Utils.h"
#include "../Redfield/Redfield.h"

// To evaluate the time a routine takes
double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

// Spectral function of the residual baths
double ohmic_spectrum(double w, double t)
{
  double ga = 0.0071; // Coupling to the residual reservoir
  double wc = 1000.0; // Cut-off frequency
  double aw = std::abs(w);
  double n_b = 1.0 / (std::exp(aw / t) - (1.0 * (w != 0)));
  double j_s = ga * aw * std::exp(-1.0 * aw / wc);
  return (j_s * (n_b + 1.0) * (w > 0)) + (j_s * n_b * (w < 0)) + (ga * t * (w == 0.0));
}

int main(int argc, char **argv)
{
  MKL_INT rc = 777; // Local dimension of the reaction cordinates
  double eps = 0.0; // Pre-factor of the sigz term in H
  double delta = 1.0; // Pre-factor of the sigx term in H
  double om1 = 28.0; // Self-energy of the reaction coordinate on the left
  double om2 = 28.0; // Self-energy of the reaction coordinate on the right
  double la = 0.777; // Coupling strength of between spin and reaction coordinates
  double t_l = 1.0; // Temperature left
  double t_r = 0.5; // Temperature right

  // Parsing arguments
  if(argc != 5){
    std::cerr << "Usage: " << argv[0] << " --rc [RC] --la [la]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--rc") rc = atoi(argv[i + 1]);
    else if(str == "--la") la = atof(argv[i + 1]);
    else continue;
  }
  if(rc == 777 || la == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --rc [RC] --la [la]" << std::endl;
    exit(1);
  }
  double la1 = la;
  double la2 = la;

  std::cout << std::fixed;
  std::cout.precision(4);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# RC = " << rc << std::endl;

  // Hamiltonian. Matrices implemented in Utils.
  // Careful! Spin matrices in Utils are mutable.
  // Types defined in Utils: MZType is std::vector< std::complex<double> >
  // CType is std::complex<double>
  MZType Ham = Utils::kron(0.5 * eps, 
                           Utils::sigz,
                           Utils::kron(1.0, Utils::eye(rc), Utils::eye(rc)));
  MZType t2 = Utils::kron(0.5 * delta, 
                          Utils::sigx,
                          Utils::kron(1.0, Utils::eye(rc), Utils::eye(rc)));
  MZType t3 = Utils::kron(om1, 
                          Utils::eye(2),
                          Utils::kron(1.0, Utils::ata(rc), Utils::eye(rc)));
  MZType t4 = Utils::kron(om2, 
                          Utils::eye(2),
                          Utils::kron(1.0, Utils::eye(rc), Utils::ata(rc)));
  MZType t5 = Utils::kron(la1, 
                          Utils::sigz,
                          Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType t6 = Utils::kron(la2, 
                          Utils::sigz,
                          Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  Utils::add(Ham, t2, t3);
  Utils::add(Ham, t4, t5);
  MZType n_vec(Ham.size(), 0.0);
  Utils::add(Ham, t6, n_vec);

  // Interaction Hamiltonian between RCs and residual baths, system OPs
  MZType Vl = Utils::kron(1.0, 
                     Utils::eye(2),
                     Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType Vr = Utils::kron(1.0, 
                     Utils::eye(2),
                     Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  // Redfield tensor
  double time1 = seconds();
  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  Redfield redT(ham_dim);

  MZType RedL((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType RedR((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Coh((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  redT.construct_redfield_2r(Coh, RedL, RedR, Ham, Vl, Vr, &ohmic_spectrum, t_l, t_r);
  double time2 = seconds();

  std::cout << "# Time Redfield: " << time2 - time1 << std::endl;
  // Write a superoperator onto Coh
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  cblas_zaxpy(Coh.size(), &alpha, &RedL[0], 1, &Coh[0], 1);
  cblas_zaxpy(Coh.size(), &alpha, &RedR[0], 1, &Coh[0], 1);

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
              &RedL[0],
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

  std::cout << std::scientific;
  std::cout.precision(6);
  std::cout << la << " " << h_curr.real() << " " << h_curr.imag() << std::endl;

  return 0;
}
