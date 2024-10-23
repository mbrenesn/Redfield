#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm
from scipy.linalg import solve_continuous_lyapunov
import time
import argparse

def partial_trace(rho, keep, dims, optimize=False):
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
eps = [(1.0 - (0.5 * i)) for i in range(Nbits)] # Spin
#eps = [0.5 for _ in range(Nbits)] # Spin
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
Obs = np.array(SigmaZ[0])
for i in range(1, Nbits):
    Hbits += eps[i] * SigmaZ[i]
    Obs += SigmaZ[i]

# System + RC Hamiltonian
Ham = np.kron(Hbits, np.eye(rc)) + np.kron(om1 * np.eye(size_s), ata(rc))
for i in range(Nbits):
    Ham += np.kron(la1 * Scoupling[i], atpa(rc))

beta = 1.0 / t_r
eigvals, eigvecs = np.linalg.eigh(Ham)
ThermalRhoEn = np.diag(1.0 / (np.sum(np.exp( -1.0 * beta * (eigvals - eigvals[:, np.newaxis])), axis = 1)))
# Rotate back to spin basis
ThermalRhoCB = eigvecs.dot(ThermalRhoEn.dot(eigvecs.transpose().conjugate()))

# State of system alone
ThermalRhoSys = partial_trace(ThermalRhoCB, [0], [size_s, rc])

obsz1 = SigmaZ[0]
obsz2 = SigmaZ[1]

z1 = np.trace(ThermalRhoSys.dot(obsz1))
z2 = np.trace(ThermalRhoSys.dot(obsz2))

print("%.12f %.12f %.12f" % (la1, np.real(z1), np.real(z2)))
