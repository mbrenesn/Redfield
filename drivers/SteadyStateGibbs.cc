// Thermal state calculations
// Instead of doing Redfield, the SS is thermal so one can just build the Gibbs
// state of total system+RC Hamiltonian
// QFI calculations
/*******************************************************************************/
// Single reservoir
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
#include "../Utils/UtilsReal.h"
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

  std::vector<double> eps(Nbits, 1.0); // Local qubit splitting
  //for(MKL_INT i = 0; i < Nbits; ++i) eps[i] = eps[i] - (0.0025 * i);

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
  std::vector<double> Hbits(size_s * size_s, 0.0);
  std::vector<double> TotZ(size_s * size_s, 0.0);
  std::vector< std::vector<double> > SigmaXZ;
  std::vector< std::vector<double> > SigmaZ;
  SigmaXZ.resize(Nbits);
  SigmaZ.resize(Nbits);
  // Mats
  std::vector<double> sigxz(4, 0.0);
  for(MKL_INT i = 0; i < 4; ++i) sigxz[i] = (1.0 / std::sqrt(2)) * (UtilsReal::sigx[i] + UtilsReal::sigz[i]);

  // First SXZ and SZ
  SigmaZ[0] = UtilsReal::kron(1.0, UtilsReal::sigz, UtilsReal::eye(1LL << (Nbits - 1)));
  SigmaXZ[0] = UtilsReal::kron(1.0, UtilsReal::sigx, UtilsReal::eye(1LL << (Nbits - 1)));
  // Last SX and SZ
  SigmaZ[Nbits - 1] = UtilsReal::kron(1.0, UtilsReal::eye(1LL << (Nbits - 1)), UtilsReal::sigz);
  SigmaXZ[Nbits - 1] = UtilsReal::kron(1.0, UtilsReal::eye(1LL << (Nbits - 1)), UtilsReal::sigx);
  // The ones in the middle
  for(MKL_INT i = 1; i < (Nbits - 1); ++i){
    SigmaZ[i] = UtilsReal::kron(1.0, 
                UtilsReal::kron(1.0, UtilsReal::eye(1LL << i), UtilsReal::sigz), 
                                     UtilsReal::eye(1LL << (Nbits - i - 1)));
    SigmaXZ[i] = UtilsReal::kron(1.0, 
                 UtilsReal::kron(1.0, UtilsReal::eye(1LL << i), UtilsReal::sigx), 
                                      UtilsReal::eye(1LL << (Nbits - i - 1)));
  }

  // System Hamiltonian and observable
  std::vector<double> null_s(size_s * size_s, 0.0);
  double dull = 1.0;
  for(MKL_INT i = 0; i < Nbits; ++i){
    UtilsReal::add(Hbits, eps[i], SigmaZ[i], null_s); 
    UtilsReal::add(TotZ, dull, SigmaZ[i], null_s);
  }

  // Hamiltonian system + RCs
  std::vector<double> Ham = UtilsReal::kron(1.0,
                                            Hbits,
                                            UtilsReal::eye(rc));
  std::vector<double> t1 = UtilsReal::kron(om_r,
                                           UtilsReal::eye(size_s),
                                           UtilsReal::ata(rc));
  std::vector<double> null_m(Ham.size(), 0.0);
  for(MKL_INT i = 0; i < Nbits; ++i){
    std::vector<double> t2 = UtilsReal::kron(la, SigmaXZ[i], UtilsReal::atpa(rc));

    UtilsReal::add(Ham, dull, t2, null_m);
  }
  UtilsReal::add(Ham, dull, t1, null_m);

  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  // Diagonalise full system Hamiltonian
  double time1 = seconds();
  std::vector<double> eigvals(ham_dim, 0.0);
  MKL_INT info;
  info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR,
                         'V', 
                         'U',
                         ham_dim, 
                         &Ham[0], 
                         ham_dim, 
                         &eigvals[0] );
  double time2 = seconds();
  std::cout << "# Time diagonalisation of Ham: " << time2 - time1 << std::endl;

  time1 = seconds();
  // Build thermal state
  std::vector<double> rho(ham_dim * ham_dim, 0.0);
  //std::vector<double> probs(ham_dim, 0.0);
  double beta_t = 1.0 / t_r;
  
  for(MKL_INT i = 0; i < ham_dim; ++i){
    double sum = 1.0;
    for(MKL_INT j = 0; j < ham_dim; ++j){
      if( i != j ){
        sum += std::exp(-1.0 * beta_t * (eigvals[j] - eigvals[i]));
        std::cout << i << " " << j << " " << eigvals[j] - eigvals[i] << std::endl;
      }
    }
    rho[(i * ham_dim) + i] = 1.0 / sum;
  }

  // Rotate state back to spin basis
  std::vector<double> buffer(ham_dim * ham_dim, 0.0);
  cblas_dgemm( CblasRowMajor,
               CblasNoTrans,
               CblasNoTrans,
               ham_dim,
               ham_dim,
               ham_dim,
               1.0,
               &Ham[0],
               ham_dim,
               &rho[0],
               ham_dim,
               0.0,
               &buffer[0],
               ham_dim );
  cblas_dgemm( CblasRowMajor,
               CblasNoTrans,
               CblasTrans,
               ham_dim,
               ham_dim,
               ham_dim,
               1.0,
               &buffer[0],
               ham_dim,
               &Ham[0],
               ham_dim,
               0.0,
               &rho[0],
               ham_dim );
  
  for(MKL_INT i = 0; i < 10; ++i){
    for(MKL_INT j = 0; j < 10; ++j){
      std::cout << rho[(i * ham_dim) + j] << " ";
    }
    std::cout << std::endl;
  }
  
#if 0
  std::vector<double> red_rho = UtilsReal::partial_trace_rc(rho, size_s, rc);
  time2 = seconds();
  std::cout << "# Time rotation to spin basis and partial trace: " << time2 - time1 << std::endl;

  UtilsReal::print_mat(red_rho);
  // Red Rho is now the state of just system in computational basis, without RC
  // Diagonalise system
  time1 = seconds();
  std::vector<double> eigvals_s(size_s, 0.0);
  info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR,
                         'V', 
                         'U',
                         size_s, 
                         &red_rho[0], 
                         size_s, 
                         &eigvals_s[0] );
  time2 = seconds();
  std::cout << "# Time diagonalisation of reduced rho: " << time2 - time1 << std::endl;

  // Rotate observable into the basis that diagonalises rho
  time1 = seconds();
  std::vector<double> buffer_s(size_s * size_s, 0.0);
  cblas_dgemm( CblasRowMajor,
               CblasTrans,
               CblasNoTrans,
               size_s,
               size_s,
               size_s,
               1.0,
               &red_rho[0],
               size_s,
               &TotZ[0],
               size_s,
               0.0,
               &buffer_s[0],
               size_s );
  cblas_dgemm( CblasRowMajor,
               CblasNoTrans,
               CblasNoTrans,
               size_s,
               size_s,
               size_s,
               1.0,
               &buffer_s[0],
               size_s,
               &red_rho[0],
               size_s,
               0.0,
               &TotZ[0],
               size_s );
  time2 = seconds();
  std::cout << "# Time rotation of observable: " << time2 - time1 << std::endl;

  std::cout.precision(8);
  // Compute Quantum Fisher information
  double fish_t = 0.0;
  for(MKL_INT n = 0; n < size_s; ++n){
    for(MKL_INT m = 0; m < n; ++m){
      double o_nm = TotZ[(n * size_s) + m];
      double en = eigvals_s[n];
      double em = eigvals_s[m];
      fish_t += 2.0 * (((en - em) * (en - em)) / (en + em)) * o_nm * o_nm;
    }
  }
  fish_t = fish_t * 2.0;

  std::cout << "# T T/dT"  << std::endl;
  std::cout << t_r << " " << std::sqrt(fish_t) * t_r << std::endl;
#endif
  return 0;
}
