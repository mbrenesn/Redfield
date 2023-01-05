// Redfield calculations
/*******************************************************************************/
// Computes the heat current in the two-terminal set-up
// Double qubit configuration, one coupled to hot the other to cold
// Phonon mapping with sigma x
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

// To evaluate the time a routine takes
double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

std::vector<double> linspace(double a, double b, size_t n) {
  double h = (b - a) / static_cast<double>(n - 1);
  std::vector<double> vec(n);
  typename std::vector<double>::iterator x;
  double val;
  for (x = vec.begin(), val = a; x != vec.end(); ++x, val += h)
    *x = val;
  return vec;
}

// Flat spectrum
double flat_spectrum(double w, double t)
{
  double ga = 0.00036 / M_PI;
  double aw = std::abs(w);
  double n_b = 1.0 / (std::exp(aw / t) - (1.0 * (w != 0)));
  double j_s = ga;
  
  return (M_PI * j_s * (n_b + 1.0) * (w > 0)) 
       + (M_PI * j_s * n_b * (w < 0)) 
       + (M_PI * j_s * (w == 0.0));
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

// Brownian spectral function
double brownian_spectrum(double w, double t)
{
  double ga = 0.02 / M_PI;
  double Om = 15.0;
  double aw = std::abs(w);
  double n_b = 1.0 / (std::exp(aw / t) - (1.0 * (w != 0)));
  double j_s = (4.0 * ga * Om * Om * aw)  
            / ( (((Om * Om) - (aw * aw)) * ((Om * Om) - (aw * aw)))
            + ((2.0 * M_PI * ga * Om * aw) * (2.0 * M_PI * ga * Om * aw)) );

  return (M_PI * j_s * (n_b + 1.0) * (w > 0)) 
       + (M_PI * j_s * n_b * (w < 0)) 
       + ((4.0 * M_PI * ga / (Om * Om / t)) * (w == 0.0));
}

int main(int argc, char **argv)
{
  MKL_INT rc = 777; // Local dimension of the reaction cordinates
  double delta = 0.777;
  double g = 0.777; // Internal coupling between two qubits
  double la_h = 1.0; double la_c = 3.0; // System-Reservoir couplings
  double t_h = 0.777; // Temperature right
  double t_c = 1.0; // Temperature left

  // Parsing arguments
  if(argc != 7){
    std::cerr << "Usage: " << argv[0] << " --delta [del] --g [g] --th [t_h]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--delta") delta = atof(argv[i + 1]);
    if(str == "--g") g = atof(argv[i + 1]);
    if(str == "--th") t_h = atof(argv[i + 1]);
    else continue;
  }
  if(delta == 0.777 || t_h == 0.777 || g == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --delta [del] --g [g] --th [t_h]" << std::endl;
    exit(1);
  }

  double eps_h = 1.0; double eps_c = delta + eps_h; // Local term H two-qubit
  std::cout << std::fixed;
  std::cout.precision(5);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# eps_h = " << eps_h << std::endl;
  std::cout << "# delta = " << delta << std::endl;
  std::cout << "# g = " << g << std::endl;
  std::cout << "# la_h = " << la_h << " la_c = " << la_c << std::endl;
  std::cout << "# t_h = " << t_h << " t_c = " << t_c << std::endl;
  std::cout << "# ga = 0.00036 / pi" << std::endl;

  // Hamiltonian. Matrices implemented in Utils.
  // Careful! Spin matrices in Utils are mutable.
  // Types defined in Utils: MZType is std::vector< std::complex<double> >
  // CType is std::complex<double>
  // Two-qubit Hamiltonian
  MZType sigma_pm = {Utils::p_one_r, Utils::zero, Utils::zero, Utils::zero};
  MZType Ham = Utils::kron(eps_h, 
                           sigma_pm,
                           Utils::iden);
  MZType t1 = Utils::kron(eps_c, 
                          Utils::iden,
                          sigma_pm);
  MZType t2 = Utils::kron(g,
                          Utils::sigp,
                          Utils::sigm);
  MZType t3 = Utils::kron(g,
                          Utils::sigm,
                          Utils::sigp);
  

  MZType n_vec(Ham.size(), 0.0);
  Utils::add(Ham, t1, t2);
  Utils::add(Ham, t3, n_vec);
  
  // Interaction Hamiltonian between system and baths
  // Phonon coupling
  // V = la * sigz \otimes (a^dag + a)
  MZType Vl = Utils::kron(la_h,
                          Utils::sigx,
                          Utils::iden);
  MZType Vr = Utils::kron(la_c,
                          Utils::iden,
                          Utils::sigx);

  // Redfield tensor
  double time1 = seconds();
  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  Redfield redT(ham_dim);

  MZType RedL((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType RedR((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Coh((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  redT.construct_redfield_2r_phonon(Coh, RedL, RedR, Ham, Vl, Vr, &flat_spectrum, t_h, t_c);
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

  // Negativity measure
  // Rotate state back to spin basis
  MZType buffer(ham_dim * ham_dim, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ham_dim, ham_dim, ham_dim, &alpha, &steady_state[0], ham_dim, &Ham[0], ham_dim, &beta, &buffer[0], ham_dim);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha, &Ham[0], ham_dim, &buffer[0], ham_dim, &beta, &steady_state[0], ham_dim);
 
  // Partial transpose
  MZType ss_copy = steady_state;
  ss_copy[1] = steady_state[4]; ss_copy[3] = steady_state[6];
  ss_copy[4] = steady_state[1]; ss_copy[6] = steady_state[3];
  ss_copy[9] = steady_state[12]; ss_copy[11] = steady_state[14];
  ss_copy[12] = steady_state[9]; ss_copy[14] = steady_state[11];
  // Eigvals
  std::vector<double> lambdas(ham_dim, 0.0);
  MKL_INT info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', ham_dim, &ss_copy[0], ham_dim, &lambdas[0]);

  double negativity = 0.0;
  for(MKL_INT i = 0; i < ham_dim; ++i){
    if(lambdas[i] < 0) negativity += std::abs(lambdas[i]);   
  }

  std::cout << std::scientific;
  std::cout.precision(15);
  std::cout << g << " " << h_curr.real() << " " << negativity << std::endl;

  return 0;
}
