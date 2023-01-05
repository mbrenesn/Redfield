#ifndef __REDFIELD_H
#define __REDFIELD_H

#include "../Utils/Utils.h"
#include "mkl_lapacke.h"

class Redfield
{
  public:
    // Methods
    Redfield(MKL_INT dim = 0);
    ~Redfield();
    // Single reservoir
    void construct_redfield_1r_phonon(MZType &Coh,
                                      MZType &Redfield,
                                      MZType &Ham,
                                      MZType &Vr,
                                      std::function<double(double, double)> spec_den,
                                      double t_r);
    // Two reservoirs
    void construct_redfield_2r_phonon(MZType &Coh,
                                      MZType &RedfieldLeft,
                                      MZType &RedfieldRight,
                                      MZType &Ham,
                                      MZType &Vl,
                                      MZType &Vr,
                                      std::function<double(double, double)> spec_den_l,
                                      std::function<double(double, double)> spec_den_r,
                                      double t_l,
                                      double t_r);
    void construct_redfield_2r_phonon_sec(MZType &Coh,
                                          MZType &RedfieldLeft,
                                          MZType &RedfieldRight,
                                          MZType &Ham,
                                          MZType &Vl,
                                          MZType &Vr,
                                          std::function<double(double, double)> spec_den_l,
                                          std::function<double(double, double)> spec_den_r,
                                          double t_l,
                                          double t_r);
    MZType get_steady_state(MZType &RTensor);
    // Members
    std::vector<double> eigvals;

  private:
    MKL_INT ham_dim_;
};
#endif
