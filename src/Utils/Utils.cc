#include "Utils.h"

namespace Utils
{
  /*******************************************************************************/
  // Pauli matrices
  /*******************************************************************************/
  CType zero(0.0, 0.0);
  CType p_one_r(1.0, 0.0);
  CType m_one_r(-1.0, 0.0);
  CType p_one_c(0.0, 1.0);
  CType m_one_c(0.0, -1.0);
  MZType sigx = {zero, p_one_r, p_one_r, zero};
  MZType sigy = {zero, m_one_c, p_one_c, zero};
  MZType sigz = {p_one_r, zero, zero, m_one_r};
  MZType sigp = {zero, p_one_r, zero, zero};
  MZType sigm = {zero, zero, p_one_r, zero};
  MZType iden = {p_one_r, zero, zero, p_one_r};

  /*******************************************************************************/
  // Identity
  /*******************************************************************************/
  MZType eye(MKL_INT m)
  {
    MZType vec(m * m, 0.0);
    for(MKL_INT i = 0; i < m; ++i){
      vec[(i * m) + i] = 1.0;
    }

    return vec;
  }
  
  /*******************************************************************************/
  // adagger*a matrix for m levels
  /*******************************************************************************/
  MZType ata(MKL_INT m)
  {
    MZType vec(m * m, 0.0);
    for(MKL_INT i = 0; i < m; ++i){
      vec[(i * m) + i] = 0.5 * ((2.0 * i) + 1.0);
    }

    return vec;
  }
  
  /*******************************************************************************/
  // adagger + a matrix for m levels
  /*******************************************************************************/
  MZType atpa(MKL_INT m)
  {
    MZType vec(m * m, 0.0);
    for(MKL_INT i = 0; i < (m - 1); ++i){
      vec[(i * m) + (i + 1)] = std::sqrt(i + 1);
      vec[((i + 1) * m) + i] = std::sqrt(i + 1);
    }
    
    return vec;
  }
  
  /*******************************************************************************/
  // Kron product of second and third argument, times a pre-factor alpha
  // on the first argument.
  // This routine assumes a and b are contigous memory segments representing
  // a *square* matrix.
  /*******************************************************************************/
  MZType kron(CType alpha,
              MZType const &a,
              MZType const &b)
  {
    MKL_INT len_n = a.size();
    MKL_INT len_m = b.size();
    MKL_INT n = static_cast<MKL_INT>(std::sqrt(len_n));
    MKL_INT m = static_cast<MKL_INT>(std::sqrt(len_m));
    MKL_INT len = n * m;
    MZType res(len * len, 0.0);
    for(MKL_INT i = 0; i < n; ++i){
      for(MKL_INT j = 0; j < n; ++j){
        MKL_INT startrow = i * m;
        MKL_INT startcol = j * m;
        for(MKL_INT k = 0; k < m; ++k){
          for(MKL_INT l = 0; l < m; ++l){
            res[((startrow + k) * len) + (startcol + l)] = alpha * a[(i * n) + j] * b[(k * m) + l];
          }
        }
      }
    }

    return res;
  }
  
  /*******************************************************************************/
  // Vanilla matrix addition for square matrices.
  // Adds second and third arguments to first argument.
  // Prefact is a factor on the first argument
  /*******************************************************************************/
  void add(MZType &res,
           CType &prefact,
           MZType const &a,
           MZType const &b)
  {
    assertm( a.size() == b.size(), "Utils::add, matrices don't match size" );
    assertm( a.size() == res.size(), "Utils::add, matrices don't match size" );
    MKL_INT len = static_cast<MKL_INT>(std::sqrt(a.size()));
    for(MKL_INT i = 0; i < len; ++i){
      for(MKL_INT j = 0; j < len; ++j){
        res[(i * len) + j] += (prefact * a[(i * len) + j]) + b[(i * len) + j];
      }
    }
  }
  
  /*******************************************************************************/
  // A very basic implementation of a partial trace
  // This assumes a tensor structure of the type A x B x C x ... x RC
  // Given that structure, this routine returns the Tr_{RC}[\rho] where
  // \rho is the full density matrix that contains RC degrees of freedom
  // Arguments:
  // rho: Full density matrix
  // keep: Dimension of the subspace to keep
  // rc: Local space of the RC
  /*******************************************************************************/
  MZType partial_trace_rc(MZType &rho,
                          MKL_INT keep,
                          MKL_INT rc)
  {
    MZType red_rho(keep * keep, 0.0);
    MKL_INT global_dim = keep * rc;
  
    for(MKL_INT i = 0; i < keep; ++i){
      for(MKL_INT j = 0; j < keep; ++j){
        for(MKL_INT k = 0; k < rc; ++k){
          MKL_INT sub_block_i = k + (i * rc);
          MKL_INT sub_block_j = k + (j * rc);

          red_rho[(i * keep) + j] += rho[(sub_block_i * global_dim) + sub_block_j];
        }
      }
    }

    return red_rho;
  }

  /*******************************************************************************/
  // Print matrix. Second argument establishes if printing complex values or not
  /*******************************************************************************/
  void print_mat(MZType &mat,
                 bool comp)
  {
    MKL_INT len = static_cast<MKL_INT>(std::sqrt(mat.size()));
    for(MKL_INT i = 0; i < len; ++i){
      for(MKL_INT j = 0; j < len; ++j){
        if(comp) std::cout << mat[(i * len) + j] << " ";
        else std::cout << mat[(i * len) + j].real() << " ";
      }
      std::cout << std::endl;
    }
  }
}
