#ifndef __UTILS_H
#define __UTILS_H

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <functional>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

#define MKL_Complex16 std::complex<double>

#include "mkl.h"
#include "mkl_types.h"

typedef std::vector< std::complex<double> > MZType;
typedef std::complex<double> CType;

namespace Utils
{
  extern CType zero;
  extern CType p_one_r;
  extern CType m_one_r;
  extern CType p_one_c;
  extern CType m_one_c;
  extern MZType sigx;
  extern MZType sigy;
  extern MZType sigz;
  extern MZType sigp;
  extern MZType sigm;
  extern MZType iden;

  MZType eye(MKL_INT m);
  MZType ata(MKL_INT m);
  MZType atpa(MKL_INT m);
  MZType kron(CType alpha, 
              MZType const &a, 
              MZType const &b);
  MZType partial_trace_rc(MZType &rho,
                          MKL_INT keep,
                          MKL_INT rc);
  void add(MZType &res,
           CType &prefact,
           MZType const &a,
           MZType const &b);
  void print_mat(MZType &mat,
                 bool comp);
}
#endif
