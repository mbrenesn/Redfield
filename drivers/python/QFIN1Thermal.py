#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm
from scipy.linalg import solve_continuous_lyapunov
import time
import argparse

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims,2))
    rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)

def c_t_analytic(delta, beta):
    return 4.0 * (delta**2) * (beta**2) * np.exp(2 * beta * delta) / ((1.0 + np.exp(2 * beta * delta))**2)

# Parsing arguments
parser = argparse.ArgumentParser(description='Redfield \
        Single reservoir, two qubits connected by RC')

parser.add_argument('--rc', action='store', dest='rc', type=int,
                    help='[INT] Number of RC levels')
parser.add_argument('--la', action='store', dest='la', type=float,
                    help='[FLOAT] Coupling to  RC')
parser.add_argument('--eps', action='store', dest='eps', type=float,
                    help='[FLOAT] Spin splitting')
parser.add_argument('--tr', action='store', dest='t_r', type=float,
                    help='[FLOAT] Temperature of reservoir')

args = parser.parse_args()
# Parameters
rc = args.rc # Number of levels in RC
eps1 = args.eps # Spin
m = [rc]
om1 = 15.0 # self-energy of RC
la1 = args.la # coupling spin and RC
t_r = args.t_r # temperature of bath

# Matrices
sigx = np.array([[0.0, 1.0], [1.0, 0.0]])
sigy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sigz = np.array([[1.0, 0.0], [0.0, -1.0]])
sigp = 0.5 * (sigx + (1.0j * sigy))
sigm = 0.5 * (sigx - (1.0j * sigy))
ata = lambda m : 0.5 * np.diag([i for i in range(1, 2 * m, 2)])
atpa = lambda m : np.diag( np.sqrt([i for i in range(1, m)]), 1) \
                  + np.diag( np.sqrt([i for i in range(1, m)]), -1)

np.set_printoptions(suppress=True, precision=5, linewidth=100)
time1 = time.time()
# Hamiltonian RC with two reservoirs
# Order: system + reservoir
Ham = eps1 * np.kron(sigz, np.eye(m[0])) \
    + om1 * np.kron(np.eye(2), ata(m[0])) \
    + la1 * np.kron(sigx, atpa(m[0]))
    #+ la1 * np.kron(((1.0 / np.sqrt(2)) * sigx) + ((1.0 / np.sqrt(2)) * sigz), atpa(m[0]))

# Weak coupling thermal Gibbs
beta = 1.0 / t_r
HamBit = eps1 * sigz
ThermalWeak = expm(-1.0 * HamBit * beta)
tra = np.trace(ThermalWeak)
ThermalWeak = ThermalWeak / tra

# Heat Capacity Weak
ener = np.trace(ThermalWeak.dot(HamBit))
ener2 = np.trace(ThermalWeak.dot(HamBit.dot(HamBit)))
var = ener2 - (ener * ener)
heat_capacity = var * beta * beta

# Weak Coupling Fisher
# Q
ExpEnerWeak = np.trace(ThermalWeak.dot(HamBit))
ExpEnerIdWeak = ExpEnerWeak * np.diag(np.ones(2))
dRhodBetaWeak = ThermalWeak.dot(ExpEnerIdWeak - HamBit)
QWeak = 2.0 * dRhodBetaWeak

# X
SLDWeak = solve_continuous_lyapunov(ThermalWeak, QWeak)

# Fisher
FisherWeak = np.trace(SLDWeak.dot(SLDWeak.dot(ThermalWeak)))

# Strong coupling: Fisher
# State of qubit + RC
ThermalRhoCB = expm(-1.0 * Ham * beta)
tra = np.trace(ThermalRhoCB)
ThermalRhoCB = ThermalRhoCB / tra

# State of system alone
ThermalRhoSys = partial_trace(ThermalRhoCB, [0], [2, rc])

# Heat Capacity
ener_str = np.trace(ThermalRhoSys.dot(HamBit))
ener2_str = np.trace(ThermalRhoSys.dot(HamBit.dot(HamBit)))
var_str = ener2_str - (ener_str * ener_str)
heat_capacity_str = var_str * beta * beta

# Q
# This one here is subtle, in particular the derivative
ExpEnerTotal = np.trace(ThermalRhoCB.dot(Ham))
HRhoTotal = Ham.dot(ThermalRhoCB)
TrRCHRhoTotal = partial_trace(HRhoTotal, [0], [2, rc])
dRhodBeta = (ExpEnerTotal * ThermalRhoSys) - TrRCHRhoTotal 
Q = 2.0 * dRhodBeta

# X
SLD = solve_continuous_lyapunov(ThermalRhoSys, Q)

# Fisher
Fisher = np.trace(SLD.dot(SLD.dot(ThermalRhoSys)))

# Effective model RC order 2
block11 = eps1 * np.exp(-2.0 * la1 * la1 / (om1 * om1)) * sigz
block12 = 1.0j * eps1 * (2.0 * la1 / om1) * np.exp(-2.0 * la1 * la1 / (om1 * om1)) * sigy
block21 = -1.0j * eps1 * (2.0 * la1 / om1) * np.exp(-2.0 * la1 * la1 / (om1 * om1)) * sigy
block22 = eps1 * np.exp(-2.0 * la1 * la1 / (om1 * om1)) * sigz * (1.0 - ((2.0*la1/om1) * (2.0*la1/om1)))
HamEff = np.vstack([np.hstack([block11, block12]), np.hstack([block21, block22])])
ThermalEff = expm(-1.0 * HamEff * beta)
traeff = np.trace(ThermalEff)
ThermalEff = ThermalEff / traeff

ThermalSysEff = partial_trace(ThermalEff, [1], [2, 2])

# Heat Capacity Effective
ener_eff = np.trace(ThermalEff.dot(HamEff))
ener2_eff = np.trace(ThermalEff.dot(HamEff.dot(HamEff)))
var_eff = ener2_eff - (ener_eff * ener_eff)
heat_capacity_eff = var_eff * beta * beta

# Effective Coupling Fisher
# Q
ExpEnerEff = np.trace(ThermalEff.dot(HamEff))
HRhoEff = HamEff.dot(ThermalEff)
TrRCHRhoEff = partial_trace(HRhoEff, [1], [2, 2])
dRhodBetaEff = (ExpEnerEff * ThermalSysEff) - TrRCHRhoEff
QEff = 2.0 * dRhodBetaEff

# X
SLDEff = solve_continuous_lyapunov(ThermalSysEff, QEff)

# Fisher
FisherEff = np.trace(SLDEff.dot(SLDEff.dot(ThermalSysEff)))

print("%.8f %.8f %.8f %.8f" % (t_r, beta * np.sqrt(FisherWeak), beta * np.sqrt(Fisher), beta * np.sqrt(np.real(FisherEff))))
