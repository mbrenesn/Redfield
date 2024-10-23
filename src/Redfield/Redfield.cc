#include "Redfield.h"

/*******************************************************************************/
// Custom constructor, requires Hamiltonian dimension
// Default instantiation is not allowed.
/*******************************************************************************/
Redfield::Redfield(MKL_INT dim)
: ham_dim_(dim)
{
  if(ham_dim_ == 0){
    std::cerr << "Class Redfield cannot be instantiated with default parameters" << std::endl;
    exit(1);
  }
}

Redfield::~Redfield()
{}

/*******************************************************************************/
// Redfield tensor method
// This function computes the coherent (Coh) and the Redfield tensors
// for the single-terminal configuration, i.e., the one in which a
// reservoir is coupled to a central system.
// Arguments:
// 1st - Tensor describing the coherent part of the Redfield equation (dim = dim{H}^2)
// 2nd - Tensor describing the incoherent part of the Redfield equation due to the 
// reservoir (dim = dim{H}^2)
// 3rd - Hamiltonian of the central system (dim = dim{H})
// 4th - Interaction Hamiltonian with the reservoir
// Interaction is assumed of the form A \otimes B, where A are system operators
// This argument takes the form of A (dim = dim{H})
// 5th - A function of two arguments, frequency and temperature that returns
// the spectral function of the traced-out reservoirs. Could be modifified for more
// arguments
// 6th - Temperature of the reservoir
/*******************************************************************************/
void Redfield::construct_redfield_1r_phonon(MZType &Coh,
                                            MZType &Redfield,
                                            MZType &Ham,
                                            MZType &Vr,
                                            std::function<double(double, double)> spec_den,
                                            double t_r)
{
  assertm(Ham.size() == Vr.size(), "Dimension mismatch in Redfield construction");
  
  // Diagonalise Ham
  eigvals.resize(ham_dim_);
  MKL_INT info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', ham_dim_, &Ham[0], ham_dim_, &eigvals[0]);

  // Rotate Vr
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  MZType buffer(ham_dim_ * ham_dim_, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Vr[0], ham_dim_, &Ham[0], ham_dim_, &beta, &buffer[0], ham_dim_);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Ham[0], ham_dim_, &buffer[0], ham_dim_, &beta, &Vr[0], ham_dim_);
  
  // Redfield Tensor
  MKL_INT dim = ham_dim_;
  MKL_INT dim2 = ham_dim_ * ham_dim_;

  #pragma omp parallel for collapse (2)
  for(MKL_INT i = 0; i < dim; ++i){
    for(MKL_INT j = 0; j < dim; ++j){
      
      double en = eigvals[i] - eigvals[j];
      CType enn(0.0, -1.0 * en);
      
      Coh[(((i * dim) + j) * dim2) + ((i * dim) + j)] = enn;

      for(MKL_INT k = 0; k < dim; ++k){
        for(MKL_INT l = 0; l < dim; ++l){
        
          double del_lk = eigvals[l] - eigvals[k];
          double del_ki = eigvals[k] - eigvals[i];
          double del_lj = eigvals[l] - eigvals[j];
          double del_kl = eigvals[k] - eigvals[l];

          CType R1r = Vr[(i * dim) + k] * Vr[(k * dim) + l] * spec_den(del_lk, t_r);

          Redfield[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1r;
      
          CType R2r = Vr[(l * dim) + j] * Vr[(i * dim) + k] * spec_den(del_ki, t_r);
          
          Redfield[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2r;
          
          CType R3r = Vr[(k * dim) + i] * Vr[(j * dim) + l] * spec_den(del_lj, t_r);
          
          Redfield[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3r);
          
          CType R4r = Vr[(j * dim) + l] * Vr[(l * dim) + k] * spec_den(del_kl, t_r);
          
          Redfield[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4r);
        }
      }
    }
  }
}

/*******************************************************************************/
// Redfield tensor method
// This function computes the coherent (Coh), left and right redfield tensors
// for the two-terminal configuration, i.e., the one in which a left and right
// reservoirs are coupled to a central system.
// Arguments:
// 1st - Tensor describing the coherent part of the Redfield equation (dim = dim{H}^2)
// 2nd - Tensor describing the incoherent part of the Redfield equation due to the 
// left reservoir (dim = dim{H}^2)
// 3rd - Tensor describing the incoherent part of the Redfield equation due to the 
// right reservoir (dim = dim{H}^2)
// 4th - Hamiltonian of the central system (dim = dim{H})
// 5th - Interaction Hamiltonian with the left reservoir
// Interaction is assumed of the form A \otimes B, where A are system operators
// This argument takes the form of A (dim = dim{H})
// 6th - Interaction Hamiltonian of the right reservoir, see above
// 7th - A function of two arguments, frequency and temperature that returns
// the spectral function of the traced-out reservoirs. Could be modifified for more
// arguments
// 8th - Temperature of left reservoir
// 9th - Temperature of right reservoir
/*******************************************************************************/
void Redfield::construct_redfield_2r_phonon(MZType &Coh,
                                            MZType &RedfieldLeft,
                                            MZType &RedfieldRight,
                                            MZType &Ham,
                                            MZType &Vl,
                                            MZType &Vr,
                                            std::function<double(double, double)> spec_den_l,
                                            std::function<double(double, double)> spec_den_r,
                                            double t_l,
                                            double t_r)
{
  assertm(Ham.size() == Vl.size(), "Dimension mismatch in Redfield construction");
  assertm(Ham.size() == Vr.size(), "Dimension mismatch in Redfield construction");
  
  // Diagonalise Ham
  eigvals.resize(ham_dim_);
  MKL_INT info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', ham_dim_, &Ham[0], ham_dim_, &eigvals[0]);

  // Rotate Vl
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  MZType buffer(ham_dim_ * ham_dim_, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Vl[0], ham_dim_, &Ham[0], ham_dim_, &beta, &buffer[0], ham_dim_);
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Ham[0], ham_dim_, &buffer[0], ham_dim_, &beta, &Vl[0], ham_dim_);
  
  // Rotate Vr
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Vr[0], ham_dim_, &Ham[0], ham_dim_, &beta, &buffer[0], ham_dim_);
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Ham[0], ham_dim_, &buffer[0], ham_dim_, &beta, &Vr[0], ham_dim_);

  // Redfield Tensor
  MKL_INT dim = ham_dim_;
  MKL_INT dim2 = ham_dim_ * ham_dim_;

  #pragma omp parallel for collapse (2)
  for(MKL_INT i = 0; i < dim; ++i){
    for(MKL_INT j = 0; j < dim; ++j){
      
      double en = eigvals[i] - eigvals[j];
      CType enn(0.0, -1.0 * en);
      
      Coh[(((i * dim) + j) * dim2) + ((i * dim) + j)] = enn;

      for(MKL_INT k = 0; k < dim; ++k){
        for(MKL_INT l = 0; l < dim; ++l){
        
          double del_lk = eigvals[l] - eigvals[k];
          double del_ki = eigvals[k] - eigvals[i];
          double del_lj = eigvals[l] - eigvals[j];
          double del_kl = eigvals[k] - eigvals[l];

          CType R1l = Vl[(i * dim) + k] * Vl[(k * dim) + l] * spec_den_l(del_lk, t_l);
          CType R1r = Vr[(i * dim) + k] * Vr[(k * dim) + l] * spec_den_r(del_lk, t_r);

          RedfieldLeft[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1l;
          RedfieldRight[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1r;
      
          CType R2l = Vl[(l * dim) + j] * Vl[(i * dim) + k] * spec_den_l(del_ki, t_l);
          CType R2r = Vr[(l * dim) + j] * Vr[(i * dim) + k] * spec_den_r(del_ki, t_r);
          
          RedfieldLeft[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2l;
          RedfieldRight[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2r;
          
          CType R3l = Vl[(k * dim) + i] * Vl[(j * dim) + l] * spec_den_l(del_lj, t_l);
          CType R3r = Vr[(k * dim) + i] * Vr[(j * dim) + l] * spec_den_r(del_lj, t_r);
          
          RedfieldLeft[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3l);
          RedfieldRight[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3r);
          
          CType R4l = Vl[(j * dim) + l] * Vl[(l * dim) + k] * spec_den_l(del_kl, t_l);
          CType R4r = Vr[(j * dim) + l] * Vr[(l * dim) + k] * spec_den_r(del_kl, t_r);
          
          RedfieldLeft[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4l);
          RedfieldRight[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4r);
        }
      }
    }
  }
}

/*******************************************************************************/
// Redfield tensor method: Secular approximation
// This function computes the coherent (Coh), left and right redfield tensors
// for the two-terminal configuration, i.e., the one in which a left and right
// reservoirs are coupled to a central system.
// Arguments:
// 1st - Tensor describing the coherent part of the Redfield equation (dim = dim{H}^2)
// 2nd - Tensor describing the incoherent part of the Redfield equation due to the 
// left reservoir (dim = dim{H}^2)
// 3rd - Tensor describing the incoherent part of the Redfield equation due to the 
// right reservoir (dim = dim{H}^2)
// 4th - Hamiltonian of the central system (dim = dim{H})
// 5th - Interaction Hamiltonian with the left reservoir
// Interaction is assumed of the form A \otimes B, where A are system operators
// This argument takes the form of A (dim = dim{H})
// 6th - Interaction Hamiltonian of the right reservoir, see above
// 7th - A function that of two arguments, frequency and temperature that returns
// the spectral function of the traced-out reservoirs. Could be modifified for more
// arguments
// 8th - Temperature of left reservoir
// 9th - Temperature of right reservoir
/*******************************************************************************/
void Redfield::construct_redfield_2r_phonon_sec(MZType &Coh,
                                                MZType &RedfieldLeft,
                                                MZType &RedfieldRight,
                                                MZType &Ham,
                                                MZType &Vl,
                                                MZType &Vr,
                                                std::function<double(double, double)> spec_den_l,
                                                std::function<double(double, double)> spec_den_r,
                                                double t_l,
                                                double t_r)
{
  assertm(Ham.size() == Vl.size(), "Dimension mismatch in Redfield construction");
  assertm(Ham.size() == Vr.size(), "Dimension mismatch in Redfield construction");
  
  // Diagonalise Ham
  eigvals.resize(ham_dim_);
  MKL_INT info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', ham_dim_, &Ham[0], ham_dim_, &eigvals[0]);
  
  // Rotate Vl
  CType alpha(1.0, 0.0);
  CType beta(0.0, 0.0);
  MZType buffer(ham_dim_ * ham_dim_, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Vl[0], ham_dim_, &Ham[0], ham_dim_, &beta, &buffer[0], ham_dim_);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Ham[0], ham_dim_, &buffer[0], ham_dim_, &beta, &Vl[0], ham_dim_);
  
  // Rotate Vr
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Vr[0], ham_dim_, &Ham[0], ham_dim_, &beta, &buffer[0], ham_dim_);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ham_dim_, ham_dim_, ham_dim_, &alpha, &Ham[0], ham_dim_, &buffer[0], ham_dim_, &beta, &Vr[0], ham_dim_);

  // Redfield Tensor
  MKL_INT dim = ham_dim_;
  MKL_INT dim2 = ham_dim_ * ham_dim_;
  
  #pragma omp parallel for collapse (2)
  for(MKL_INT i = 0; i < dim; ++i){
    for(MKL_INT j = 0; j < dim; ++j){
      
      double en = eigvals[i] - eigvals[j];
      CType enn(0.0, -1.0 * en);
      
      Coh[(((i * dim) + j) * dim2) + ((i * dim) + j)] = enn;

      for(MKL_INT k = 0; k < dim; ++k){
        for(MKL_INT l = 0; l < dim; ++l){
        
          double del_lk = eigvals[l] - eigvals[k];
          double del_ki = eigvals[k] - eigvals[i];
          double del_lj = eigvals[l] - eigvals[j];
          double del_kl = eigvals[k] - eigvals[l];

          CType R1l = Vl[(i * dim) + k] * Vl[(k * dim) + l] * spec_den_l(del_lk, t_l);
          CType R1r = Vr[(i * dim) + k] * Vr[(k * dim) + l] * spec_den_r(del_lk, t_r);

          if(i == j && l == j){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1l;
            RedfieldRight[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1r;
          }
          else if(i != j && l != j){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1l;
            RedfieldRight[(((i * dim) + j) * dim2) + ((l * dim) + j)] += -1.0 * R1r;
          }

          CType R2l = Vl[(l * dim) + j] * Vl[(i * dim) + k] * spec_den_l(del_ki, t_l);
          CType R2r = Vr[(l * dim) + j] * Vr[(i * dim) + k] * spec_den_r(del_ki, t_r);

          if(i == j && k == l){ 
            RedfieldLeft[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2l;
            RedfieldRight[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2r;
          }
          else if(i !=j && k != l){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2l;
            RedfieldRight[(((i * dim) + j) * dim2) + ((k * dim) + l)] += R2r;
          }

          CType R3l = Vl[(k * dim) + i] * Vl[(j * dim) + l] * spec_den_l(del_lj, t_l);
          CType R3r = Vr[(k * dim) + i] * Vr[(j * dim) + l] * spec_den_r(del_lj, t_r);
          
          if(i == j && k == l){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3l);
            RedfieldRight[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3r);
          }
          else if(i != j && k != l){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3l);
            RedfieldRight[(((i * dim) + j) * dim2) + ((k * dim) + l)] += std::conj(R3r);
          }

          CType R4l = Vl[(j * dim) + l] * Vl[(l * dim) + k] * spec_den_l(del_kl, t_l);
          CType R4r = Vr[(j * dim) + l] * Vr[(l * dim) + k] * spec_den_r(del_kl, t_r);
          
          if(i == j && i == k){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4l);
            RedfieldRight[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4r);
          }
          else if(i != j && i != k){
            RedfieldLeft[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4l);
            RedfieldRight[(((i * dim) + j) * dim2) + ((i * dim) + k)] += -1.0 * std::conj(R4r);
          }
        }
      }
    }
  }
}

/*******************************************************************************/
// Method to obtain the steady state
// This function returns the steady-state solution from a total tensor RTensor
// that describes both coherent and incoherent effects
// Arguments:
// 1st - Tensor describing both the coherent and incoherent parts of the Redfield 
// equation (dim = dim{H}^2)
// Returns - Steady state (dim = dim{H})
/*******************************************************************************/
MZType Redfield::get_steady_state(MZType &RTensor)
{
  assertm(static_cast<MKL_INT>(std::sqrt(RTensor.size())) == ham_dim_ * ham_dim_, 
              "Dimension mismatch in steady state routine: Redfield tensor and Hamiltonian");

  // \overline{W} = W + |0>><<1|
  for(MKL_INT i = 0; i < ham_dim_; ++i){
    RTensor[(i * ham_dim_) + i] += 1.0;
  }

  // Solution vector
  MKL_INT dim = ham_dim_ * ham_dim_;
  MZType SState(dim, 0.0);
  SState[0] = 1.0;
  
  // LU factorisation
  std::vector<MKL_INT> ipiv(dim, 0);
  MKL_INT info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR,
                                dim,
                                dim,
                                &RTensor[0],
                                dim,
                                &ipiv[0]);
  if(info != 0){
    std::cerr << "LU factorisation failed" << std::endl;
    std::cerr << "Error number: " << info << std::endl;
    exit(1);
  }
  
  info = LAPACKE_zgetrs(LAPACK_ROW_MAJOR,
                        'N',
                        dim,
                        1,
                        &RTensor[0],
                        dim,
                        &ipiv[0],
                        &SState[0],
                        1);
  if(info != 0){
    std::cerr << "Solution to zgetrs failed" << std::endl;
    std::cerr << "Error number: " << info << std::endl;
    exit(1);
  }

  return SState;
}
