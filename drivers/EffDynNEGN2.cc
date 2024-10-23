// Redfield calculations
// Dynamics
// Effective
/*******************************************************************************/
// Single reservoir
// Phonon mapping with sigma x
// Reaction coordinate at strong coupling
/*******************************************************************************/

#include <iostream>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <complex>
#include <sys/time.h>
#include <omp.h>

#include "../Utils/Utils.h"
#include "../Redfield/Redfield.h"
#include "../RungeKutta/RK4.h"

// To evaluate the time a routine takes
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

// Spectral function of the residual baths
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
  double la = 0.777; // System-Reservoir coupling
  double t_r = 0.777; // Temperature
  double om_r = 8.0; // Self-energy of the RC

  // Parsing arguments
  if(argc != 5){
    std::cerr << "Usage: " << argv[0] << " --la [la] --tr [t_h]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--la") la = atof(argv[i + 1]);
    if(str == "--tr") t_r = atof(argv[i + 1]);
    else continue;
  }
  if(t_r == 0.777 || la == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --la [la] --tr [t_r]" << std::endl;
    exit(1);
  }

  std::vector< std::complex<double> > eps(2, 1.0); // Local qubit splitting
  for(MKL_INT i = 0; i < 2; ++i) eps[i] = eps[i] - (0.1 * i);

  std::cout << std::fixed;
  std::cout.precision(5);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# la = " << la << std::endl;
  std::cout << "# om_r = " << om_r << std::endl;
  std::cout << "# t_r = " << t_r << std::endl;
  std::cout << "# ga = 0.05" << std::endl;

  // Effective Hamiltonian
  MKL_INT size_s = 1 << 2;
  MZType t1 = Utils::kron(eps[0] * std::exp(-2.0 * la * la / (om_r * om_r)),
                          Utils::sigz,
                          Utils::eye(2));
  MZType t2 = Utils::kron(eps[1] * std::exp(-2.0 * la * la / (om_r * om_r)),
                          Utils::eye(2),
                          Utils::sigz);
  MZType t3 = Utils::kron(1.0,
                          Utils::sigx,
                          Utils::eye(2));
  MZType t4 = Utils::kron(1.0,
                          Utils::eye(2),
                          Utils::sigx);

  MZType null_s(size_s * size_s, 0.0);
  CType alpha(1.0, 0.0);
  CType alpham(-1.0, 0.0);
  CType laomt = alpham * la * la / om_r;
  CType beta(0.0, 0.0);
  MZType Ham(size_s * size_s, 0.0);
  MZType H1(size_s * size_s, 0.0);
  MZType Sint(size_s * size_s, 0.0);

  Utils::add(H1, alpha, t1, t2);
  Utils::add(Sint, alpha, t3, t4);
  
  MZType Sint2(size_s * size_s, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alpha,
              &Sint[0], size_s, &Sint[0], size_s, &beta, &Sint2[0], size_s);

  Utils::add(Ham, laomt, Sint2, H1);

  // Interaction Hamiltonian between sytem and bath, system OPs
  // Phonon coupling
  MZType Vr(size_s * size_s, 0.0);
  for(MKL_INT i = 0; i < (size_s * size_s); ++i){
    Vr[i] = (-2.0 * la / om_r) * Sint[i];
  }

  // Redfield tensor
  double time1 = seconds();
  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  Redfield redT(ham_dim);

  MZType RedR((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Coh((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  redT.construct_redfield_1r_phonon(Coh, RedR, Ham, Vr, &ohmic_spectrum, t_r);
  double time2 = seconds();

  std::cout << "# Time Redfield: " << time2 - time1 << std::endl;
  // Write a superoperator onto Coh
  cblas_zaxpy(Coh.size(), &alpha, &RedR[0], 1, &Coh[0], 1);

  // Dynamics
  // Runge Kutta
  RK4 rk4(ham_dim * ham_dim);

  std::vector<double> times = linspace(0.0, 5000.0, 500001);
  double delta_t = times[1] - times[0];
  MKL_INT tsteps = times.size();

  // Initial State
  // For the system, both spins at ground state
  MZType SysRho(size_s * size_s, 0.0);
  SysRho[5] = 1.0;

  // Time Evo
  double time = 0.0;
  MZType buffer(ham_dim * ham_dim, 0.0);
  MZType RedRho = SysRho;
  std::vector<double> eigvals_rho_pt(size_s, 0.0);
  for(MKL_INT tt = 0; tt < tsteps; ++tt){
  
    // From this point on, this is hardcoded to N=2 and SigmaX coupling
    // Entanglement negativity
    MZType RedRho_PT(RedRho.size(), 0.0);
    // Partial Transpose
    RedRho_PT[0] = RedRho[0]; RedRho_PT[1] = RedRho[1]; RedRho_PT[2] = RedRho[8]; RedRho_PT[3] = RedRho[9];
    RedRho_PT[4] = RedRho[4]; RedRho_PT[5] = RedRho[5]; RedRho_PT[6] = RedRho[12]; RedRho_PT[7] = RedRho[13];
    RedRho_PT[8] = RedRho[2]; RedRho_PT[9] = RedRho[3]; RedRho_PT[10] = RedRho[10]; RedRho_PT[11] = RedRho[11];
    RedRho_PT[12] = RedRho[6]; RedRho_PT[13] = RedRho[7]; RedRho_PT[14] = RedRho[14]; RedRho_PT[15] = RedRho[15];
    // Eigenvalues of PT
    MKL_INT info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'N', 'U', size_s, &RedRho_PT[0], size_s, &eigvals_rho_pt[0]);
    // Negativity
    double negativity = 0.0;
    for(MKL_INT i = 0; i < size_s; ++i){
      if(eigvals_rho_pt[i] < 0) negativity += eigvals_rho_pt[i];
    }

    std::cout << time << " " << negativity << std::endl;
    
    // Rotate state into eigenbasis of Ham to evolve it
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &Ham[0], ham_dim, &RedRho[0], ham_dim, &beta, &buffer[0], ham_dim);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &buffer[0], ham_dim, &Ham[0], ham_dim, &beta, &RedRho[0], ham_dim);  
    // Now we can evolve
    rk4.rk4_redfield(delta_t, RedRho, Coh);
    time += delta_t;
    // We have to rotate the state back to original basis so we can take the partial trace
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &Ham[0], ham_dim, &RedRho[0], ham_dim, &beta, &buffer[0], ham_dim);  
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &buffer[0], ham_dim, &Ham[0], ham_dim, &beta, &RedRho[0], ham_dim);
  }

  return 0;
}
