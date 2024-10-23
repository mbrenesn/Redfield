// Redfield calculations
// Dynamics of one spin
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
  MKL_INT rc = 777; // Local dimension of the reaction cordinates
  MKL_INT Nbits = 777; // Number of qubits
  double la = 0.777; // System-Reservoir coupling
  double t_r = 0.777; // Temperature
  double om_r = 8.0; // Self-energy of the RC

  // Parsing arguments
  if(argc != 9){
    std::cerr << "Usage: " << argv[0] << " --N [N] --rc [rc] --la [la] --tr [t_h]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--rc") rc = atoi(argv[i + 1]);
    if(str == "--N") Nbits = atoi(argv[i + 1]);
    if(str == "--la") la = atof(argv[i + 1]);
    if(str == "--tr") t_r = atof(argv[i + 1]);
    else continue;
  }
  if(Nbits == 777 || rc == 777 || t_r == 0.777 || la == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --N [N] --rc [rc] --la [la] --tr [t_r]" << std::endl;
    exit(1);
  }

  std::vector< std::complex<double> > eps(Nbits, 1.0); // Local qubit splitting
  //for(MKL_INT i = 0; i < Nbits; ++i) eps[i] = eps[i] - (0.5 * i);
  eps[0] = 0.5;

  std::cout << std::fixed;
  std::cout.precision(5);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# RC = " << rc << std::endl;
  std::cout << "# la = " << la << std::endl;
  std::cout << "# om_r = " << om_r << std::endl;
  std::cout << "# t_r = " << t_r << std::endl;
  std::cout << "# ga = 0.05" << std::endl;

  // NBit Hamiltonian
  MKL_INT size_s = 1LL << Nbits;
  MZType Hbits(size_s * size_s, 0.0);
  std::vector<MZType> SigmaX;
  std::vector<MZType> SigmaZ;
  std::vector<MZType> SigmaXZ;
  SigmaX.resize(Nbits);
  SigmaZ.resize(Nbits);
  SigmaXZ.resize(Nbits);
  // Mats
  std::vector< std::complex<double> > sigxz(4, 0.0);
  for(MKL_INT i = 0; i < 4; ++i) sigxz[i] = (1.0 / std::sqrt(2)) * (Utils::sigx[i] + Utils::sigz[i]);
  // First SX and SZ
  SigmaZ[0] = Utils::kron(1.0, Utils::sigz, Utils::eye(1LL << (Nbits - 1)));
  SigmaX[0] = Utils::kron(1.0, Utils::sigx, Utils::eye(1LL << (Nbits - 1)));
  SigmaXZ[0] = Utils::kron(1.0, sigxz, Utils::eye(1LL << (Nbits - 1)));
  // Last SX and SZ
  SigmaZ[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigz);
  SigmaX[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigx);
  SigmaXZ[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), sigxz);
  // The ones in the middle
  for(MKL_INT i = 1; i < (Nbits - 1); ++i){
    SigmaZ[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigz), Utils::eye(1LL << (Nbits - i - 1)));
    SigmaX[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigx), Utils::eye(1LL << (Nbits - i - 1)));
    SigmaXZ[i] = Utils::kron(1.0, 
                 Utils::kron(1.0, Utils::eye(1LL << i), sigxz), Utils::eye(1LL << (Nbits - i - 1)));
  }

  // System Hamiltonian
  MZType null_s(size_s * size_s, 0.0);
  CType dull = {1.0, 0.0};
  for(MKL_INT i = 0; i < Nbits; ++i){
    Utils::add(Hbits, eps[i], SigmaZ[i], null_s); 
  }

  // Hamiltonian system + RCs
  MZType Ham = Utils::kron(1.0,
                           Hbits,
                           Utils::eye(rc));
  MZType t1 = Utils::kron(om_r,
                          Utils::eye(size_s),
                          Utils::ata(rc));
  MZType null_m(Ham.size(), 0.0);
  for(MKL_INT i = 0; i < Nbits; ++i){
    MZType t2 = Utils::kron(la, SigmaX[i], Utils::atpa(rc));
    //MZType t2 = Utils::kron(la, SigmaZ[i], Utils::atpa(rc));
    //MZType t2 = Utils::kron(la, SigmaXZ[i], Utils::atpa(rc));

    Utils::add(Ham, dull, t2, null_m);
  }
  Utils::add(Ham, dull, t1, null_m);

  // Interaction Hamiltonian between RCs and residual baths, system OPs
  // Phonon coupling
  MZType Vr = Utils::kron(1.0,
                          Utils::eye(size_s),
                          Utils::atpa(rc));
 
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
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  cblas_zaxpy(Coh.size(), &alpha, &RedR[0], 1, &Coh[0], 1);

  // Dynamics
  // Runge Kutta
  RK4 rk4(ham_dim * ham_dim);

  std::vector<double> times = linspace(0.0, 2500.0, 250001);
  double delta_t = times[1] - times[0];
  MKL_INT tsteps = times.size();

  // Initial State
  // For the system, let's just do a 1 on the first entry
  MZType SysRho(size_s * size_s, 0.0);
  SysRho[0] = 1.0;

  // Product state with a thermal state of the RC
  MZType RCRho = Utils::ata(rc);
  for(MKL_INT i = 0; i < rc; ++i){
    RCRho[(i * rc) + i] = std::exp((-1.0 * om_r * RCRho[(i * rc) + i]) / t_r);
  }
  CType z_sum = 0.0;
  for(MKL_INT i = 0; i < rc; ++i){
    z_sum += RCRho[(i * rc) + i];
  }
  for(MKL_INT i = 0; i < rc; ++i){
    RCRho[(i * rc) + i] = RCRho[(i * rc) + i] / z_sum;
  }

  // Initial State is a product state between system and RC
  MZType Rho = Utils::kron(1.0, SysRho, RCRho);

  // Time Evo
  double time = 0.0;
  MZType buffer(ham_dim * ham_dim, 0.0);
  for(MKL_INT tt = 0; tt < tsteps; ++tt){
  
    MZType RedRho = Utils::partial_trace_rc(Rho, size_s, rc); 

    // From this point on, this is hardcoded to N=1
    // Expectation values
    CType sigmaz_1 = {0.0, 0.0};
    cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alpha,
                 &SigmaZ[0][0], size_s, &RedRho[0], size_s, &beta, &null_s[0], size_s );
    for(MKL_INT i = 0; i < size_s; ++i){
      sigmaz_1 += null_s[(i * size_s) + i];  
    }

    std::cout << time << " " << std::real(sigmaz_1) << std::endl;

    // Rotate state into eigenbasis of Ham to evolve it
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &Ham[0], ham_dim, &Rho[0], ham_dim, &beta, &buffer[0], ham_dim);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &buffer[0], ham_dim, &Ham[0], ham_dim, &beta, &Rho[0], ham_dim);  
    // Now we can evolve
    rk4.rk4_redfield(delta_t, Rho, Coh);
    time += delta_t;
    // We have to rotate the state back to original basis so we can take the partial trace
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &Ham[0], ham_dim, &Rho[0], ham_dim, &beta, &buffer[0], ham_dim);  
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, ham_dim, ham_dim, ham_dim, &alpha,
                &buffer[0], ham_dim, &Ham[0], ham_dim, &beta, &Rho[0], ham_dim);
  }

  return 0;
}
