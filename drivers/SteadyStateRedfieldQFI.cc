// Redfield calculations
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
  double ga = 0.005; // Coupling to the residual reservoir
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
  double om_r = 15.0; // Self-energy of the RC

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
  for(MKL_INT i = 0; i < Nbits; ++i) eps[i] = eps[i] - (0.0025 * i);

  std::cout << std::fixed;
  std::cout.precision(5);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# RC = " << rc << std::endl;
  std::cout << "# la = " << la << std::endl;
  std::cout << "# om_r = " << om_r << std::endl;
  std::cout << "# t_r = " << t_r << std::endl;
  std::cout << "# ga = 0.005" << std::endl;

  // NBit Hamiltonian and observable
  MKL_INT size_s = 1LL << Nbits;
  MZType Hbits(size_s * size_s, 0.0);
  MZType TotZ(size_s * size_s, 0.0);
  std::vector<MZType> SigmaX;
  std::vector<MZType> SigmaZ;
  SigmaX.resize(Nbits);
  SigmaZ.resize(Nbits);
  // Mats
  // First SX and SZ
  SigmaZ[0] = Utils::kron(1.0, Utils::sigz, Utils::eye(1LL << (Nbits - 1)));
  SigmaX[0] = Utils::kron(1.0, Utils::sigx, Utils::eye(1LL << (Nbits - 1)));
  // Last SX and SZ
  SigmaZ[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigz);
  SigmaX[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigx);
  // The ones in the middle
  for(MKL_INT i = 1; i < (Nbits - 1); ++i){
    SigmaZ[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigz), Utils::eye(1LL << (Nbits - i - 1)));
    SigmaX[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigx), Utils::eye(1LL << (Nbits - i - 1)));
  }

  // System Hamiltonian and observable
  MZType null_s(size_s * size_s, 0.0);
  CType dull = {1.0, 0.0};
  for(MKL_INT i = 0; i < Nbits; ++i){
    Utils::add(Hbits, eps[i], SigmaZ[i], null_s); 
    Utils::add(TotZ, dull, SigmaZ[i], null_s);
  }

#if 0
  for(MKL_INT state = 0; state < size_s; ++state){
    MKL_INT bs = state;

    std::complex<double> mag_term = 0.0;
    for(MKL_INT site = 0; site < Nbits; ++site){
      MKL_INT bitset = bs;

      if(bitset & (1LL << site)){
        SigmaZ[(state * size_s) + state] += 1.0;
        mag_term += eps[site];
      }
      else{
        SigmaZ[(state * size_s) + state] -= 1.0;
        mag_term -= eps[site];
     }

      bitset ^= 1LL << site;
      SigmaX[site][(state * size_s) + bitset] = 1.0;
    }
    Hbits[(state * size_s) + state] = mag_term;
  }
#endif

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
  
  // Compute steady state
  time1 = seconds();
  MZType steady_state = redT.get_steady_state(Coh);
  time2 = seconds();
  std::cout << "# Time SS solution: " << time2 - time1 << std::endl;

  time1 = seconds();
  // Rotate state back to spin basis
  MZType buffer(ham_dim * ham_dim, 0.0);
  cblas_zgemm( CblasRowMajor,
               CblasNoTrans,
               CblasNoTrans,
               ham_dim,
               ham_dim,
               ham_dim,
               &alpha,
               &Ham[0],
               ham_dim,
               &steady_state[0],
               ham_dim,
               &beta,
               &buffer[0],
               ham_dim );
  cblas_zgemm( CblasRowMajor,
               CblasNoTrans,
               CblasConjTrans,
               ham_dim,
               ham_dim,
               ham_dim,
               &alpha,
               &buffer[0],
               ham_dim,
               &Ham[0],
               ham_dim,
               &beta,
               &steady_state[0],
               ham_dim );

  MZType rho = Utils::partial_trace_rc(steady_state, size_s, rc);
  time2 = seconds();
  std::cout << "# Time rotation to spin basis and partial trace: " << time2 - time1 << std::endl;

  // Rho is now the state of the system in computational basis
  // Diagonalise system
  time1 = seconds();
  std::vector<double> eigvals(size_s, 0.0);
  MKL_INT info;
  info = LAPACKE_zheevd( LAPACK_ROW_MAJOR,
                         'V', 
                         'U',
                         size_s, 
                         &rho[0], 
                         size_s, 
                         &eigvals[0] );
  time2 = seconds();
  std::cout << "# Time diagonalisation of rho: " << time2 - time1 << std::endl;

  // Rotate observable into the basis that diagonalises rho
  time1 = seconds();
  MZType buffer_s(size_s * size_s, 0.0);
  cblas_zgemm( CblasRowMajor,
               CblasConjTrans,
               CblasNoTrans,
               size_s,
               size_s,
               size_s,
               &alpha,
               &rho[0],
               size_s,
               &TotZ[0],
               size_s,
               &beta,
               &buffer_s[0],
               size_s );
  cblas_zgemm( CblasRowMajor,
               CblasNoTrans,
               CblasNoTrans,
               size_s,
               size_s,
               size_s,
               &alpha,
               &buffer_s[0],
               size_s,
               &rho[0],
               size_s,
               &beta,
               &TotZ[0],
               size_s );
  time2 = seconds();
  std::cout << "# Time rotation of observable: " << time2 - time1 << std::endl;

  std::cout.precision(8);
  // Compute Quantum Fisher information
  std::complex<double> fish_t(0.0, 0.0);
  for(MKL_INT n = 0; n < size_s; ++n){
    for(MKL_INT m = 0; m < n; ++m){
      std::complex<double> o_nm = TotZ[(n * size_s) + m];
      double en = eigvals[n];
      double em = eigvals[m];
      fish_t += 2.0 * (((en - em) * (en - em)) / (en + em)) * std::norm(o_nm);
    }
  }
  std::cout << "# N Fisher" << std::endl;
  std::cout << t_r << " " << (2.0 * fish_t.real()) / Nbits << std::endl;

  return 0;
}
