#!/usr/bin/env python

from qutip import *
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
import argparse

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
eps1 = 1.0
g = 0.0
om1 = 15.0 # self-energy of RC
la1 = args.la # coupling spin and RC
t_r = args.t_r # temperature of bath
ga = 0.005 # residual coupling
wc = 1000 # cut-off frequency

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
    + la1 * np.kron(((1.0 / np.sqrt(2)) * sigx) + ((1.0 / np.sqrt(2)) * sigz), atpa(m[0]))

# Weak coupling thermal Gibbs
HamBit = sigz
thermal_weak = expm(-1.0 * HamBit / t_r)
tra = np.trace(thermal_weak)
thermal_weak = thermal_weak / tra

ener = np.trace(thermal_weak.dot(HamBit))
ener2 = np.trace(thermal_weak.dot(HamBit.dot(HamBit)))
var = ener2 - (ener * ener)
heat_capacity = var / (t_r * t_r)

# Interaction Hamiltonian between RCs and residual baths, system OPs
Vr = np.kron(np.eye(2), atpa(m[0]))
time2 = time.time()
print("# Time Hamiltonian:", time2 - time1)

# Observable
Sz = sigz

def ohmic_spectrum(w):
    n_b = 1.0 / (np.exp( np.absolute(w) / t_r ) - (1.0 * (w != 0)))
    j_s = ga * np.absolute(w) * np.exp(-1.0 * np.absolute(w) / wc)
    return (j_s * (n_b + 1.0) * (w > 0)) + (j_s * n_b * (w < 0)) + (ga * t_r * (w == 0.0))

H = Qobj(Ham)
Vro = Qobj(Vr)

time1 = time.time()
R, ekets = bloch_redfield_tensor(H, [[Vro, ohmic_spectrum]], sec_cutoff = 1.0e16)
time2 = time.time()
print("# Time Redfield tensor:", time2 - time1)

# \overline{W} = W + |0>><<1|
RTensor = np.array((np.real(R) * 2.0) + (1.0j * np.imag(R)))

size = 2 * rc
for i in range(size):
    RTensor[0, (i * size) + i] += 1.0

# Solution vector
Sol = np.zeros(size * size)
Sol[0] = 1.0

Rinv = np.linalg.inv(RTensor)

SState = Rinv.dot(Sol)

SState = np.reshape(SState,(size,size))
SStateQ = Qobj(SState)

SStateCB = SStateQ.transform(ekets, inverse = True)
SStateR = np.array(SStateCB)

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


rho = partial_trace(SStateR, [0], [2, rc])

eigvals, eigvecs = np.linalg.eigh(rho)

Sz = eigvecs.transpose().conjugate().dot(Sz.dot(eigvecs))

rho_str = np.diag(eigvals)
HamStr = eigvecs.transpose().conjugate().dot(HamBit.dot(eigvecs))
ener_str = np.trace(rho_str.dot(HamStr))
ener2_str = np.trace(rho_str.dot(HamStr.dot(HamStr)))
var_str = ener2_str - (ener_str * ener_str)
heat_capacity_str = var_str / (t_r * t_r)

'''
fisher = 0.0
for n in range(2):
    for m in range(2):
        o_nm = Sx[n,m]
        fisher += 2.0 * (o_nm * o_nm) * (((eigvals[n] - eigvals[m]) * (eigvals[n] - eigvals[m])) / (eigvals[n] + eigvals[m]))
'''

print("%.8f %.8f %.8f" % (t_r, np.sqrt(heat_capacity), np.sqrt(np.real(heat_capacity_str))))
#print(np.sqrt(fisher) * t_r)
