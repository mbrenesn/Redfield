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

parser.add_argument('--rc', action='store', dest='rc', type=int,
                    help='[INT] Number of RC levels')
parser.add_argument('--la', action='store', dest='la', type=float,
                    help='[FLOAT] Coupling to  RC')
parser.add_argument('--tr', action='store', dest='t_r', type=float,
                    help='[FLOAT] Temperature of reservoir')

args = parser.parse_args()
# Parameters
rc = args.rc # Number of levels in RC
m = [rc]
om1 = 8.0 # self-energy of RC
la1 = args.la # coupling spin and RC
t_r = args.t_r # temperature of bath
jxx = 0.77
jyy = 1.23
jzz = 0.89

# Matrices
size_s = 1 << 2
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
Hbits = jxx * np.kron(sigx, sigx) + jyy * np.kron(sigy, sigy) + jzz * np.kron(sigz, sigz)

# System + RC Hamiltonian
Ham = np.kron(Hbits, np.eye(rc)) + np.kron(om1 * np.eye(size_s), ata(rc))
Ham += np.kron(la1 * np.kron(sigx, np.eye(2)), atpa(rc))
Ham += np.kron(la1 * np.kron(np.eye(2), sigx), atpa(rc))

beta = 1.0 / t_r

ThermalRhoCB = expm(-1.0 * Ham * beta)
tra = np.trace(ThermalRhoCB)
ThermalRhoCB = np.divide(ThermalRhoCB, tra)

# State of system alone
ThermalRhoSys = partial_trace(ThermalRhoCB, [0], [size_s, rc])

print(ThermalRhoSys)

sx1 = np.kron(sigz, np.eye(2))
sx2 = np.kron(np.eye(2), sigz)
sx_11 = np.trace(sx1.dot(sx1.dot(ThermalRhoSys)))
sx_12 = np.trace(sx1.dot(sx2.dot(ThermalRhoSys)))
sx_22 = np.trace(sx2.dot(sx2.dot(ThermalRhoSys)))

print((sx_11 + sx_12 + sx_12 + sx_22) / 2)
