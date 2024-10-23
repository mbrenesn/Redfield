#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm
from scipy.linalg import solve_continuous_lyapunov
import time
import argparse

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    Parameters
    ----------
    rho : 2D array
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
    rho_a : 2D array
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

# Parsing arguments
parser = argparse.ArgumentParser(description='Redfield \
        Single reservoir, two qubits connected by RC')

parser.add_argument('--N', action='store', dest='Nbits', type=int,
                    help='[INT] Number of qubits')
parser.add_argument('--rc', action='store', dest='rc', type=int,
                    help='[INT] Number of RC levels')
parser.add_argument('--la', action='store', dest='la', type=float,
                    help='[FLOAT] Coupling to  RC')
parser.add_argument('--tr', action='store', dest='t_r', type=float,
                    help='[FLOAT] Temperature of reservoir')
parser.add_argument('--om', action='store', dest='om', type=float,
                    help='[FLOAT] RC frequency')

args = parser.parse_args()
# Parameters
Nbits = args.Nbits
rc = args.rc # Number of levels in RC
#eps = [(1.0 - (0.5 * i)) for i in range(Nbits)] # Spin
eps = [1.0 for _ in range(Nbits)] # Spin
m = [rc]
om1 = args.om # self-energy of RC
la1 = args.la # coupling spin and RC
t_r = args.t_r # temperature of bath

# Matrices
size_s = 1 << Nbits
sigx = np.array([[0.0, 1.0], [1.0, 0.0]])
sigy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sigz = np.array([[1.0, 0.0], [0.0, -1.0]])
sigxz = (1.0 / np.sqrt(2)) * (sigx + sigz)
sigp = 0.5 * (sigx + (1.0j * sigy))
sigm = 0.5 * (sigx - (1.0j * sigy))
ata = lambda m : 0.5 * np.diag([i for i in range(1, 2 * m, 2)])
atpa = lambda m : np.diag( np.sqrt([i for i in range(1, m)]), 1) \
                  + np.diag( np.sqrt([i for i in range(1, m)]), -1)

np.set_printoptions(suppress=True, precision=5, linewidth=100)
time1 = time.time()
# Hamiltonian RC of system alone
SigmaZ = [0.0 for _ in range(Nbits)]
# RC-system coupling term
Scoupling = [0.0 for _ in range(Nbits)]
# First and last
SigmaZ[0] = np.kron(sigz, np.eye(1 << (Nbits - 1)))
SigmaZ[Nbits - 1] = np.kron(np.eye(1 << (Nbits - 1)), sigz)
Scoupling[0] = np.kron(sigx, np.eye(1 << (Nbits - 1)))
Scoupling[Nbits - 1] = np.kron(np.eye(1 << (Nbits - 1)), sigx)
#Scoupling[0] = np.kron(sigxz, np.eye(1 << (Nbits - 1)))
#Scoupling[Nbits - 1] = np.kron(np.eye(1 << (Nbits - 1)), sigxz)
# The rest
for i in range(1, Nbits - 1):
    SigmaZ[i] = np.kron(np.kron(np.eye(1 << i), sigz), np.eye(1 << (Nbits - i - 1)))
    Scoupling[i] = np.kron(np.kron(np.eye(1 << i), sigx), np.eye(1 << (Nbits - i - 1)))
    #Scoupling[i] = np.kron(np.kron(np.eye(1 << i), sigxz), np.eye(1 << (Nbits - i - 1)))

# System Hamiltonian
Hbits = eps[0] * SigmaZ[0]
Obs = SigmaZ[0]
for i in range(1, Nbits):
    Hbits += eps[i] * SigmaZ[i]
    Obs += SigmaZ[i]

# System + RC Hamiltonian
Ham = np.kron(Hbits, np.eye(rc)) + np.kron(om1 * np.eye(size_s), ata(rc))
for i in range(Nbits):
    Ham += np.kron(la1 * Scoupling[i], atpa(rc))

# Suboptimal measurement
eigva_obs, eigve_obs = np.linalg.eigh(Obs)

# Weak coupling thermal Gibbs
beta = 1.0 / t_r
eigvals_weak = np.linalg.eigvalsh(Hbits)
ThermalWeak = np.diag(1.0 / (np.sum(np.exp( -1.0 * beta * (eigvals_weak - eigvals_weak[:, np.newaxis])), 
    axis = 1)))

# Weak Coupling Fisher
# Q
ExpEnerWeak = np.sum(np.diag(ThermalWeak) * eigvals_weak)
SLDWeak = (ExpEnerWeak - eigvals_weak) * np.eye(size_s)

# Fisher
FisherWeak = np.trace(SLDWeak.dot(SLDWeak.dot(ThermalWeak)))

# Strong coupling: Fisher
# State of qubit + RC
eigvals, eigvecs = np.linalg.eigh(Ham)
ThermalRhoEn = np.diag(1.0 / (np.sum(np.exp( -1.0 * beta * (eigvals - eigvals[:, np.newaxis])), axis = 1)))
# Rotate back to spin basis
ThermalRhoCB = eigvecs.dot(ThermalRhoEn.dot(eigvecs.transpose().conjugate()))

# State of system alone
ThermalRhoSys = partial_trace(ThermalRhoCB, [0], [size_s, rc])
print(ThermalRhoSys)
# Q
# This one here is subtle, in particular the derivative
ExpEnerTotal = np.trace(ThermalRhoCB.dot(Ham))
HRhoTotal = Ham.dot(ThermalRhoCB)
TrRCHRhoTotal = partial_trace(HRhoTotal, [0], [size_s, rc])

dRhodBeta = (ExpEnerTotal * ThermalRhoSys) - TrRCHRhoTotal 
Q = 2.0 * dRhodBeta

# X
SLD = solve_continuous_lyapunov(ThermalRhoSys, Q)

print(SLD)
print()
# Fisher
Fisher = np.trace(SLD.dot(SLD.dot(ThermalRhoSys)))

# Fisher projected into \sum \hat{sigma}^z_i basis\
ProjectedRho = np.zeros(Hbits.shape) 
ProjectedTrRCHRhoTotal = np.zeros(Hbits.shape)
for i in range(Hbits.shape[0]):
    ProjectedRho += (eigve_obs[i].conjugate().transpose().dot(ThermalRhoSys.dot(eigve_obs[i]))) \
                    * np.outer(eigve_obs[i].conjugate().transpose(), eigve_obs[i])  
    ProjectedTrRCHRhoTotal += (eigve_obs[i].conjugate().transpose().dot(TrRCHRhoTotal.dot(eigve_obs[i]))) \
                    * np.outer(eigve_obs[i].conjugate().transpose(), eigve_obs[i])  
ProjecteddRhodBeta = (ExpEnerTotal * ProjectedRho) - ProjectedTrRCHRhoTotal
ProjectedQ = 2.0 * ProjecteddRhodBeta

# X
ProjectedSLD = solve_continuous_lyapunov(ProjectedRho, ProjectedQ)

# Projected Fisher
ProjectedFisher = np.trace(ProjectedSLD.dot(ProjectedSLD.dot(ProjectedRho)))

# Suboptimal from \chi
dRhodT = (1.0 / (t_r * t_r)) * (TrRCHRhoTotal - (ExpEnerTotal * ThermalRhoSys))
QT = 2.0 * dRhodT

# X
SLDT = solve_continuous_lyapunov(ThermalRhoSys, QT)

OL = np.trace(Obs.dot(SLDT.dot(ThermalRhoSys)))
LO = np.trace(SLDT.dot(Obs.dot(ThermalRhoSys)))
chi_t = 0.5 * (OL + LO)
OExp = np.trace(Obs.dot(ThermalRhoSys))
O2Exp = np.trace(Obs.dot(Obs.dot(ThermalRhoSys)))
deltaO = np.sqrt(O2Exp - (OExp * OExp))
SNR = t_r * np.abs(chi_t) / deltaO

# Effective
Sint = np.kron(sigx, np.eye(2)) + np.kron(np.eye(2), sigx)
HEff = (eps[0] * np.exp(-2.0 * la1 * la1 / (om1 * om1))) * np.kron(sigz, np.eye(2)) \
     + (eps[1] * np.exp(-2.0 * la1 * la1 / (om1 * om1))) * np.kron(np.eye(2), sigz) \
     - (((la1 * la1) / om1) * Sint.dot(Sint)) 
#HEff = HEff + (om1 * np.kron(ata(1), np.eye(4))) 
#HEff = HEff + (((2.0 * la1 * la1) / om1) * np.eye(4))
ThermalEff = expm(-1.0 * HEff * beta)
tra = np.trace(ThermalEff)
ThermalEff = ThermalEff / tra

print(ThermalEff)
# SLD Effective
ExpEnerEff = np.trace(ThermalEff.dot(HEff))
SLDEff = (ExpEnerEff * np.eye(size_s)) - HEff

# SLD thermal
dRhodBetaEff = ThermalEff.dot((ExpEnerEff * np.eye(size_s)) - HEff)
QEff = 2.0 * dRhodBetaEff

SLDEff2 = solve_continuous_lyapunov(ThermalEff, QEff)

print(SLDEff)
# Fisher
FisherEff = np.trace(SLDEff.dot(SLDEff.dot(ThermalEff)))

# Heat capacity effective
ExpVH2 = np.trace(HEff.dot(HEff.dot(ThermalEff)))
ExpVH = np.trace(HEff.dot(ThermalEff))
HeatCapEff = (ExpVH2 - (ExpVH * ExpVH)) / (t_r * t_r)

print(np.sqrt(HeatCapEff))
def heat_cap_eff(la, delta, om, beta):
    delta_eff = np.exp(-2.0 * la * la / (om * om)) * delta
    e1 = 2.0 * la * la / om
    e2 = 2.0 * np.sqrt((la*la*la*la) + (delta_eff * delta_eff * om * om)) / om

    term1 = (e1 * e1) + (e2 * e2)
    term2 = 1.0 + (np.cosh(beta * e1) * np.cosh(beta * e2))
    term3 = 2.0 * e1 * e2 * np.sinh(beta * e1) * np.sinh(beta * e2)
    term4 = np.cosh(beta * e1) + np.cosh(beta * e2)

    return ((term1 * term2) - term3) / (term4 * term4)

print("%.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (t_r, beta * np.sqrt(FisherWeak), beta * np.sqrt(Fisher), beta * np.sqrt(ProjectedFisher), SNR, beta * np.sqrt(FisherEff), beta * np.sqrt(heat_cap_eff(la1, eps[0], om1, beta))))
#print("%.8f %.8f %.8f %.8f %.8f" % (t_r, beta * np.sqrt(FisherWeak), beta * np.sqrt(Fisher), beta * np.sqrt(ProjectedFisher), SNR))
