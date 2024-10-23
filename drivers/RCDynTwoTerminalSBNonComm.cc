// Redfield calculations
/*******************************************************************************/
// Dynamics
// Spin-boson model with two reservoirs
// Reservoirs at the same temperature
// One of the coupling is sx and the other is sz
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
  double om_z = 8.0; // Self-energy of the reaction coordinate on the left
  double om_x = 8.0; // Self-energy of the reaction coordinate on the right
  double la_z = 0.777; // Coupling strength of between spin and reaction coordinate
  double la_x = 0.777; // Coupling strength of between spin and reaction coordinate
  double t_r = 0.777; // Temperature left

  // Parsing arguments
  if(argc != 9){
    std::cerr << "Usage: " << argv[0] << " --rc [RC] --laz [la_z] --lax [la_x] --tr [t_r]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--rc") rc = atoi(argv[i + 1]);
    else if(str == "--laz") la_z = atof(argv[i + 1]);
    else if(str == "--lax") la_x = atof(argv[i + 1]);
    else if(str == "--tr") t_r = atof(argv[i + 1]);
    else continue;
  }
  if(rc == 777 || la_x == 0.777 || la_z == 0.777 || t_r == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --rc [RC] --laz [la_z] --lax [la_x] --tr [t_r]" << std::endl;
    exit(1);
  }

  std::cout << std::fixed;
  std::cout.precision(4);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# RC = " << rc << std::endl;
  std::cout << "# la_x = " << la_x << std::endl;
  std::cout << "# la_z = " << la_z << std::endl;
  std::cout << "# t_r = " << t_r << std::endl;

  // Hamiltonian
  MKL_INT size_s = 1 << 1;
  MZType SigmaZ = Utils::sigz;
  MZType Ham = Utils::kron(eps, 
                           Utils::sigz,
                           Utils::kron(1.0, Utils::eye(rc), Utils::eye(rc)));
  MZType t2 = Utils::kron(om_z, 
                          Utils::eye(2),
                          Utils::kron(1.0, Utils::ata(rc), Utils::eye(rc)));
  MZType t3 = Utils::kron(om_x, 
                          Utils::eye(2),
                          Utils::kron(1.0, Utils::eye(rc), Utils::ata(rc)));
  MZType t4 = Utils::kron(la_z, 
                          Utils::sigz,
                          Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType t5 = Utils::kron(la_x, 
                          Utils::sigx,
                          Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  MZType null_s(size_s * size_s, 0.0);
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  Utils::add(Ham, alpha, t2, t3);
  Utils::add(Ham, alpha, t4, t5);

  // Interaction Hamiltonian between RCs and residual baths, system OPs
  MZType V_z = Utils::kron(1.0, 
                     Utils::eye(2),
                     Utils::kron(1.0, Utils::atpa(rc), Utils::eye(rc)));
  MZType V_x = Utils::kron(1.0, 
                     Utils::eye(2),
                     Utils::kron(1.0, Utils::eye(rc), Utils::atpa(rc)));
  
  // Redfield tensor
  double time1 = seconds();
  MKL_INT ham_dim = static_cast<MKL_INT>(std::sqrt(Ham.size()));
  Redfield redT(ham_dim);

  MZType Red_z((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Red_x((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  MZType Coh((ham_dim * ham_dim) * (ham_dim * ham_dim), 0.0);
  redT.construct_redfield_2r_phonon(Coh, Red_z, Red_x, Ham, V_z, V_x, 
          &ohmic_spectrum, &ohmic_spectrum, t_r, t_r);
  double time2 = seconds();

  std::cout << "# Time Redfield: " << time2 - time1 << std::endl;
  // Write a superoperator onto Coh
  cblas_zaxpy(Coh.size(), &alpha, &Red_z[0], 1, &Coh[0], 1);
  cblas_zaxpy(Coh.size(), &alpha, &Red_x[0], 1, &Coh[0], 1);

  // Dynamics
  // Runge Kutta
  RK4 rk4(ham_dim * ham_dim);

  std::vector<double> times = linspace(0.0, 40.0, 4001);
  double delta_t = times[1] - times[0];
  MKL_INT tsteps = times.size();

  // Initial State for the system
  MZType SysRho(size_s * size_s, 0.0);
  // |psi> = (|up> + |down>) / sqrt(2). rho = |psi><psi|
  // rho(t = 0) 
  //SysRho[0] = 0.5; SysRho[1] = 0.5; SysRho[2] = 0.5; SysRho[3] = 0.5;
  SysRho[0] = 1.0; 
  // Initial state for the RC Z
  MZType RCRhoZ = Utils::ata(rc);
  for(MKL_INT i = 0; i < rc; ++i){
    RCRhoZ[(i * rc) + i] = std::exp((-1.0 * om_z * RCRhoZ[(i * rc) + i]) / t_r);
  }
  CType z_sum = 0.0;
  for(MKL_INT i = 0; i < rc; ++i){
    z_sum += RCRhoZ[(i * rc) + i];
  }
  for(MKL_INT i = 0; i < rc; ++i){
    RCRhoZ[(i * rc) + i] = RCRhoZ[(i * rc) + i] / z_sum;
  }
  // Initial state for the RC X
  MZType RCRhoX = Utils::ata(rc);
  for(MKL_INT i = 0; i < rc; ++i){
    RCRhoX[(i * rc) + i] = std::exp((-1.0 * om_z * RCRhoX[(i * rc) + i]) / t_r);
  }
  z_sum = 0.0;
  for(MKL_INT i = 0; i < rc; ++i){
    z_sum += RCRhoX[(i * rc) + i];
  }
  for(MKL_INT i = 0; i < rc; ++i){
    RCRhoX[(i * rc) + i] = RCRhoX[(i * rc) + i] / z_sum;
  }
  // Total initial state
  MZType Rho = Utils::kron(1.0, SysRho, Utils::kron(1.0, RCRhoZ, RCRhoX));

  // Time Evo
  double time = 0.0;
  MZType buffer(ham_dim * ham_dim, 0.0);
  std::cout.precision(6);
  for(MKL_INT tt = 0; tt < tsteps; ++tt){
  
    MZType RedRho = Utils::partial_trace_rc(Rho, size_s, rc * rc); 
    CType sigmaz = {0.0, 0.0};
    cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size_s, size_s, size_s, &alpha,
                 &SigmaZ[0], size_s, &RedRho[0], size_s, &beta, &null_s[0], size_s );
    for(MKL_INT i = 0; i < size_s; ++i){
      sigmaz += null_s[(i * size_s) + i];  
    }

    std::cout << time << " " << (std::real(sigmaz) + 1.0) / 2.0 << std::endl;

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
