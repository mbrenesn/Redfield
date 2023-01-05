// Redfield calculations
/*******************************************************************************/
// Computes the heat current in the two-terminal set-up
// Double qubit configuration, one coupled to hot the other to cold
// Phonon mapping with sigma x
// Strong Coupling
// Rectification
// Asymmetry is in the self-energy of the qubits
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

// Spectral function of the residual baths
double ohmic_spectrum(double w, double t)
{
  double ga = 0.02 / M_PI; // Coupling to the residual reservoir
  double Om = 15.0;
  ga = ga * (4.0 / (Om * Om)); 
  double wc = 1000.0; // Cut-off frequency
  double aw = std::abs(w);
  double n_b = 1.0 / (std::exp(aw / t) - (1.0 * (w != 0)));
  double j_s = ga * aw * std::exp(-1.0 * aw / wc);
  
  return (M_PI * j_s * (n_b + 1.0) * (w > 0)) 
       + (M_PI * j_s * n_b * (w < 0)) 
       + (M_PI * ga * t * (w == 0.0));
}

int main(int argc, char **argv)
{
  MKL_INT rc = 777; // Local dimension of the reaction cordinates
  double jx = 0.777;
  double jy = 0.777;
  double jz = 0.777;
  double la = 0.777;
  double dt = 0.777;
  double as = 0.777; // Temperature right
  double om_h = 15.0; double om_c = 15.0; // Self-energy of the RCs

  // Parsing arguments
  if(argc != 15){
    std::cerr << "Usage: " << argv[0] << " --rc [rc] --jx [jx] --jy [jy] --jz [jz] --la [la] --as [as] --dt [dt]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--jx") jx = atof(argv[i + 1]);
    if(str == "--jy") jy = atof(argv[i + 1]);
    if(str == "--jz") jz = atof(argv[i + 1]);
    if(str == "--la") la = atof(argv[i + 1]);
    if(str == "--as") as = atof(argv[i + 1]);
    if(str == "--dt") dt = atof(argv[i + 1]);
    if(str == "--rc") rc = atoi(argv[i + 1]);
    else continue;
  }
  if(jx == 0.777 || jy == 0.777 || jz == 0.777 || la == 0.777 || as == 0.777 || dt == 0.777 || rc == 777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --rc [rc] --jx [jx] --jy [jy] --jz [jz] --la [la] --as [as] --dt [dt]" << std::endl;
    exit(1);
  }

  double eps_h = 1.0 * (1.0 - as); double eps_c = 1.0 * (1.0 + as); // Local term H two-qubit
  double la_h = la; double la_c = la; // System-Reservoir couplings
  double t_0 = 2.0;
  double t_h = t_0 + (dt / 2.0);
  double t_c = t_0 - (dt / 2.0);
  std::cout << std::fixed;
  std::cout.precision(5);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# eps_h = " << eps_h << " eps_c = " << eps_c << std::endl;
  std::cout << "# jx = " << jx << std::endl;
  std::cout << "# jy = " << jy << std::endl;
  std::cout << "# jz = " << jz << std::endl;
  std::cout << "# la_h = " << la_h << " la_c = " << la_c << std::endl;
  std::cout << "# om_h = " << om_h << " om_c = " << om_c << std::endl;
  std::cout << "# t_h = " << t_h << " t_c = " << t_c << std::endl;
  std::cout << "# rc = " << rc << std::endl;
  std::cout << "# Ohmic ga = (0.02 / pi) * (4 / Om^2)" << std::endl;

  // Hamiltonian. Matrices implemented in Utils.
  // Careful! Spin matrices in Utils are mutable.
  // Types defined in Utils: MZType is std::vector< std::complex<double> >
  // CType is std::complex<double>
  // Two-qubit Hamiltonian
  MZType sigma_pm = {Utils::p_one_r, Utils::zero, Utils::zero, Utils::zero};
  MZType TwoBit = Utils::kron(eps_h, 
                              Utils::sigz,
                              Utils::iden);
  MZType t1 = Utils::kron(eps_c, 
                          Utils::iden,
                          Utils::sigz);
  MZType t2 = Utils::kron(jx,
                          Utils::sigx,
                          Utils::sigx);
  MZType t3 = Utils::kron(jy,
                          Utils::sigy,
                          Utils::sigy);
  MZType t4 = Utils::kron(jz,
                          Utils::sigz,
                          Utils::sigz);
  
  MZType n_vec(TwoBit.size(), 0.0);
  Utils::add(TwoBit, t1, t2);
  Utils::add(TwoBit, t3, t4);
  // Hamiltonian system + RCs
  MZType Ham = Utils::kron(1.0,
                           TwoBit,
                           Utils::kron(1.0, Utils::eye(rc), Utils::eye(rc)));
  MZType t5 = Utils::kron(om_h,
                          Utils::eye(4),
                          Utils::kron(1.0, Utils::ata(rc), Utils::eye(rc)));
  MZType t6 = Utils::kron(om_c,
                          Utils::eye(4),
                          Utils::kron(1.0, Utils::eye(rc), Utils::ata(rc)));
  MZType t7 = Utils::kron(la_h,
                          Utils::kron(1.0, Utils::sigx, Utils::iden),
                          Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType t8 = Utils::kron(la_c,
                          Utils::kron(1.0, Utils::iden, Utils::sigx),
                          Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  Utils::add(Ham, t5, t6);
  Utils::add(Ham, t7, t8);

  MZType Hamr(Ham);

  // Interaction Hamiltonian between system and baths
  // Phonon coupling
  MZType Vl = Utils::kron(1.0,
                          Utils::eye(4),
                          Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType Vr = Utils::kron(1.0,
                          Utils::eye(4),
                          Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));

  MZType Vlr(Vl);
  MZType Vrr(Vr);
  // Redfield tensor
  double time1 = seconds();
  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  Redfield redT(ham_dim);
  Redfield redTr(ham_dim);

  MZType RedL((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType RedR((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Coh((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType RedLr((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType RedRr((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Cohr((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  redT.construct_redfield_2r_phonon(Coh, RedL, RedR, Ham, Vl, Vr, &ohmic_spectrum, t_h, t_c);
  redTr.construct_redfield_2r_phonon(Cohr, RedLr, RedRr, Hamr, Vlr, Vrr, &ohmic_spectrum, t_c, t_h);
  double time2 = seconds();

  std::cout << "# Time Redfield: " << time2 - time1 << std::endl;
  // Write a superoperator onto Coh
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  cblas_zaxpy(Coh.size(), &alpha, &RedL[0], 1, &Coh[0], 1);
  cblas_zaxpy(Coh.size(), &alpha, &RedR[0], 1, &Coh[0], 1);
  cblas_zaxpy(Cohr.size(), &alpha, &RedLr[0], 1, &Cohr[0], 1);
  cblas_zaxpy(Cohr.size(), &alpha, &RedRr[0], 1, &Cohr[0], 1);

  // Compute steady state
  time1 = seconds();
  MZType steady_state = redT.get_steady_state(Coh);
  MZType steady_state_r = redTr.get_steady_state(Cohr);
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
  MZType rho_diss_r(ham_dim * ham_dim, 0.0);
  cblas_zgemv(CblasRowMajor,
              CblasNoTrans,
              ham_dim * ham_dim,
              ham_dim * ham_dim,
              &alpha,
              &RedLr[0],
              ham_dim * ham_dim,
              &steady_state_r[0],
              1,
              &beta,
              &rho_diss_r[0],
              1);

  // Trace
  CType h_curr(0.0, 0.0);
  CType h_curr_r(0.0, 0.0);
  for(MKL_INT i = 0; i < ham_dim; ++i){
    h_curr += rho_diss[(i * ham_dim) + i] * redT.eigvals[i];
    h_curr_r += rho_diss_r[(i * ham_dim) + i] * redTr.eigvals[i];
  }
  time2 = seconds();

  std::cout << "# Time Expc value: " << time2 - time1 << std::endl;

  std::cout << std::scientific;
  std::cout.precision(15);
  std::cout << la << " " << h_curr.real() << " " << h_curr_r.real() << std::endl;

  return 0;
}
