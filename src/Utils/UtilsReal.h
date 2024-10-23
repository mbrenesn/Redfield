#ifndef __UTILS_REAL_H
#define __UTILS_REAL_H

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <functional>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

#include "mkl.h"
#include "mkl_types.h"

namespace UtilsReal
{
  extern std::vector<double> sigx;
  extern std::vector<double> sigy;
  extern std::vector<double> sigz;
  extern std::vector<double> sigp;
  extern std::vector<double> sigm;
  extern std::vector<double> iden;

  std::vector<double> eye(MKL_INT m);
  std::vector<double> ata(MKL_INT m);
  std::vector<double> atpa(MKL_INT m);
  std::vector<double> kron(double alpha, 
                           std::vector<double> const &a, 
                           std::vector<double> const &b);
  std::vector<double> partial_trace_rc(std::vector<double> &rho,
                                       MKL_INT keep,
                                       MKL_INT rc);
  void add(std::vector<double> &res,
           double &prefact,
           std::vector<double> const &a,
           std::vector<double> const &b);
  void print_mat(std::vector<double> &mat);
}
#endif
