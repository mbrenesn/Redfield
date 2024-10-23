// Equilibrium state calculations
// Dissipative phase transitions XYZ model
// Effective model
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
#include "../Utils/UtilsReal.h"

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

int main(int argc, char **argv)
{
  MKL_INT rc = 777; // Local dimension of the reaction cordinates
  MKL_INT Nbits = 777; // Number of qubits
  double la = 0.777; // System-Reservoir coupling
  double t_r = 0.777; // Temperature
  double om_r = 8.0; // Self-energy of the RC
  double jx = 0.777;
  double jy = 0.777;
  double jz = 0.777; // Hamiltonian parameters XYZ

  // Parsing arguments
  if(argc != 15){
    std::cerr << "Usage: " << argv[0] << " --N [N] --rc [rc] --la [la] --tr [t_h] --jx [jx] --jy [jy] --jz [jz]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--rc") rc = atoi(argv[i + 1]);
    if(str == "--N") Nbits = atoi(argv[i + 1]);
    if(str == "--la") la = atof(argv[i + 1]);
    if(str == "--tr") t_r = atof(argv[i + 1]);
    if(str == "--jx") jx = atof(argv[i + 1]);
    if(str == "--jy") jy = atof(argv[i + 1]);
    if(str == "--jz") jz = atof(argv[i + 1]);
    else continue;
  }
  if(Nbits == 777 || rc == 777 || t_r == 0.777 || la == 0.777 || jx == 0.777 || jy == 0.777 || jz == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --N [N] --rc [rc] --la [la] --tr [t_h] --jx [jx] --jy [jy] --jz [jz]" << std::endl;
    exit(1);
  }

  std::cout << std::fixed;
  std::cout.precision(5);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# RC = " << rc << std::endl;
  std::cout << "# la = " << la << std::endl;
  std::cout << "# om_r = " << om_r << std::endl;
  std::cout << "# t_r = " << t_r << std::endl;
  std::cout << "# jx = " << jx << std::endl;
  std::cout << "# jy = " << jy << std::endl;
  std::cout << "# jz = " << jz << std::endl;

  // NBit Hamiltonian
  MKL_INT size_s = 1LL << Nbits;
  MZType Hbits(size_s * size_s, 0.0);
  std::vector<MZType> SigmaXX;
  std::vector<MZType> SigmaYY;
  std::vector<MZType> SigmaZZ;
  std::vector<MZType> SigmaX;
  std::vector<MZType> SigmaY;
  std::vector<MZType> SigmaZ;
  SigmaX.resize(Nbits);
  SigmaY.resize(Nbits);
  SigmaZ.resize(Nbits);
  SigmaXX.resize(Nbits - 1);
  SigmaYY.resize(Nbits - 1);
  SigmaZZ.resize(Nbits - 1);
  // Mats
  // Single-body matrices
  SigmaX[0] = Utils::kron(1.0, Utils::sigx, Utils::eye(1LL << (Nbits - 1)));
  SigmaY[0] = Utils::kron(1.0, Utils::sigy, Utils::eye(1LL << (Nbits - 1)));
  SigmaZ[0] = Utils::kron(1.0, Utils::sigz, Utils::eye(1LL << (Nbits - 1)));
  SigmaX[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigx);
  SigmaY[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigy);
  SigmaZ[Nbits - 1] = Utils::kron(1.0, Utils::eye(1LL << (Nbits - 1)), Utils::sigz);
  for(MKL_INT i = 1; i < (Nbits - 1); ++i){
    SigmaX[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigx), Utils::eye(1LL << (Nbits - i - 1)));
    SigmaY[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigy), Utils::eye(1LL << (Nbits - i - 1)));
    SigmaZ[i] = Utils::kron(1.0, 
                Utils::kron(1.0, Utils::eye(1LL << i), Utils::sigz), Utils::eye(1LL << (Nbits - i - 1)));
  }
  // Two-body matrices
  MZType XX = Utils::kron(1.0, Utils::sigx, Utils::sigx);
  MZType YY = Utils::kron(1.0, Utils::sigy, Utils::sigy);
  MZType ZZ = Utils::kron(1.0, Utils::sigz, Utils::sigz);
  SigmaXX[0] = Utils::kron(1.0, XX, Utils::eye(1LL << (Nbits - 2)));
  SigmaYY[0] = Utils::kron(1.0, YY, Utils::eye(1LL << (Nbits - 2)));
  SigmaZZ[0] = Utils::kron(1.0, ZZ, Utils::eye(1LL << (Nbits - 2)));
  for(MKL_INT i = 1; i < (Nbits - 1); ++i){
    SigmaXX[i] = Utils::kron(1.0, 
                 Utils::kron(1.0, Utils::eye(1LL << i), XX), Utils::eye(1LL << (Nbits - i - 2)));
    SigmaYY[i] = Utils::kron(1.0, 
                 Utils::kron(1.0, Utils::eye(1LL << i), YY), Utils::eye(1LL << (Nbits - i - 2)));
    SigmaZZ[i] = Utils::kron(1.0, 
                 Utils::kron(1.0, Utils::eye(1LL << i), ZZ), Utils::eye(1LL << (Nbits - i - 2)));
  }

  // System Hamiltonian: EFFECTIVE
  double jy_tilde = (0.5 * jy * (1.0 + std::exp(-8.0 * la * la / (om_r * om_r))))
                  + (0.5 * jz * (1.0 - std::exp(-8.0 * la * la / (om_r * om_r))));
  double jz_tilde = (0.5 * jz * (1.0 + std::exp(-8.0 * la * la / (om_r * om_r))))
                  + (0.5 * jy * (1.0 - std::exp(-8.0 * la * la / (om_r * om_r))));
  
  MZType null_s(size_s * size_s, 0.0);
  CType dull = {1.0, 0.0};
  CType jxx = {jx, 0.0}; CType jyy = {jy_tilde, 0.0}; CType jzz = {jz_tilde, 0.0}; 
  MZType TotX(size_s * size_s, 0.0);
  MZType TotX2(size_s * size_s, 0.0);
  for(MKL_INT i = 0; i < (Nbits - 1); ++i){
    Utils::add(Hbits, jxx, SigmaXX[i], null_s); 
    Utils::add(Hbits, jyy, SigmaYY[i], null_s); 
    Utils::add(Hbits, jzz, SigmaZZ[i], null_s); 
    Utils::add(TotX, dull, SigmaX[i], null_s);
  }
  Utils::add(TotX, dull, SigmaX[Nbits - 1], null_s);

  // S^2
  CType alph = {1.0, 0.0};
  CType bet = {0.0 , 0.0};
  cblas_zgemm( CblasRowMajor,
               CblasNoTrans,
               CblasNoTrans,
               size_s,
               size_s,
               size_s,
               &alph,
               &TotX[0],
               size_s,
               &TotX[0],
               size_s,
               &bet,
               &TotX2[0],
               size_s );

  MZType EffH(size_s * size_s, 0.0);
  CType la_term = {-1.0 * la * la / om_r, 0.0};

  Utils::add(EffH, la_term, TotX2, Hbits);

  // Diagonalise effective Hamiltonian
  double time1 = seconds();
  std::vector<double> eigvals(size_s, 0.0);
  MKL_INT info;
  info = LAPACKE_zheevd( LAPACK_ROW_MAJOR,
                         'V', 
                         'U',
                         size_s, 
                         &EffH[0], 
                         size_s, 
                         &eigvals[0] );
  double time2 = seconds();
  std::cout << "# Time diagonalisation of effective H: " << time2 - time1 << std::endl;

  time1 = seconds();
  // Build thermal state
  MZType rho(size_s * size_s, 0.0);
  double beta_t = 1.0 / t_r;
  
  for(MKL_INT i = 0; i < size_s; ++i){
    double sum = 1.0;
    for(MKL_INT j = 0; j < size_s; ++j){
      if( i != j ){
        sum += std::exp(-1.0 * beta_t * (eigvals[j] - eigvals[i]));
      }
    }
    rho[(i * size_s) + i] = 1.0 / sum;
  }

  // Rotate state back to spin basis
  cblas_zgemm( CblasRowMajor,
               CblasNoTrans,
               CblasNoTrans,
               size_s,
               size_s,
               size_s,
               &alph,
               &EffH[0],
               size_s,
               &rho[0],
               size_s,
               &bet,
               &null_s[0],
               size_s );
  cblas_zgemm( CblasRowMajor,
               CblasNoTrans,
               CblasTrans,
               size_s,
               size_s,
               size_s,
               &alph,
               &null_s[0],
               size_s,
               &EffH[0],
               size_s,
               &bet,
               &rho[0],
               size_s );

  time2 = seconds();
  std::cout << "# Time thermal state and rotation: " << time2 - time1 << std::endl;

  // Compute expectation values on reduced state
  MZType buffer(size_s * size_s, 0.0);
  CType spin_struct_xx = {0.0, 0.0};
  CType spin_struct_yy = {0.0, 0.0};
  CType spin_struct_zz = {0.0, 0.0};
  for(MKL_INT i = 0; i < Nbits; ++i){
    for(MKL_INT j = 0; j < Nbits; ++j){
      cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alph,
                   &SigmaX[i][0], size_s, &SigmaX[j][0], size_s, &bet, &null_s[0], size_s );
      cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alph,
                   &rho[0], size_s, &null_s[0], size_s, &bet, &buffer[0], size_s );
      for(MKL_INT k = 0; k < size_s; ++k)
        spin_struct_xx += buffer[(k * size_s) + k];
      
      cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alph,
                   &SigmaY[i][0], size_s, &SigmaY[j][0], size_s, &bet, &null_s[0], size_s );
      cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alph,
                   &rho[0], size_s, &null_s[0], size_s, &bet, &buffer[0], size_s );
      for(MKL_INT k = 0; k < size_s; ++k)
        spin_struct_yy += buffer[(k * size_s) + k];
      
      cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alph,
                   &SigmaZ[i][0], size_s, &SigmaZ[j][0], size_s, &bet, &null_s[0], size_s );
      cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alph,
                   &rho[0], size_s, &null_s[0], size_s, &bet, &buffer[0], size_s );
      for(MKL_INT k = 0; k < size_s; ++k)
        spin_struct_zz += buffer[(k * size_s) + k]; 
    }
  }
  
  std::cout << la << " " << std::real(spin_struct_xx) / Nbits << " " << std::real(spin_struct_yy) / Nbits << " " << std::real(spin_struct_zz) / Nbits << std::endl;

  return 0;
}
