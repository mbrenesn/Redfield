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

args = parser.parse_args()
# Parameters
Nbits = args.Nbits
rc = args.rc # Number of levels in RC
#eps = [(1.0 - (0.25 * i)) for i in range(Nbits)] # Spin
eps = [1.0 for _ in range(Nbits)] # Spin
m = [rc]
om1 = 15.0 # self-energy of RC
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
for i in range(1, Nbits):
    Hbits += eps[i] * SigmaZ[i]

# System + RC Hamiltonian
Ham = np.kron(Hbits, np.eye(rc)) + np.kron(om1 * np.eye(size_s), ata(rc))
for i in range(Nbits):
    Ham += np.kron(la1 * Scoupling[i], atpa(rc))

print(np.kron(np.eye(size_s), atpa(rc)))
# Weak coupling thermal Gibbs
beta = 1.0 / t_r
ThermalWeak = expm(-1.0 * Hbits * beta)
tra = np.trace(ThermalWeak)
ThermalWeak = np.divide(ThermalWeak, tra)

# Weak Coupling Fisher
# Q
ExpEnerWeak = np.trace(ThermalWeak.dot(Hbits))

ExpEnerIdWeak = ExpEnerWeak * np.eye(size_s)
dRhodBetaWeak = ThermalWeak.dot(ExpEnerIdWeak - Hbits)
QWeak = 2.0 * dRhodBetaWeak

# X
SLDWeak = solve_continuous_lyapunov(ThermalWeak, QWeak)

# Fisher
FisherWeak = np.trace(SLDWeak.dot(SLDWeak.dot(ThermalWeak)))

# Strong coupling: Fisher
# State of qubit + RC
ThermalRhoCB = expm(-1.0 * Ham * beta)
tra = np.trace(ThermalRhoCB)
ThermalRhoCB = np.divide(ThermalRhoCB, tra)

# State of system alone
ThermalRhoSys = partial_trace(ThermalRhoCB, [0], [size_s, rc])

# Q
# This one here is subtle, in particular the derivative
ExpEnerTotal = np.trace(ThermalRhoCB.dot(Ham))
HRhoTotal = Ham.dot(ThermalRhoCB)
TrRCHRhoTotal = partial_trace(HRhoTotal, [0], [size_s, rc])
dRhodBeta = (ExpEnerTotal * ThermalRhoSys) - TrRCHRhoTotal 
Q = 2.0 * dRhodBeta

# X
SLD = solve_continuous_lyapunov(ThermalRhoSys, Q)

# Fisher
Fisher = np.trace(SLD.dot(SLD.dot(ThermalRhoSys)))

print("%.8f %.8f %.8f" % (t_r, beta * np.sqrt(FisherWeak), beta * np.sqrt(Fisher)))
