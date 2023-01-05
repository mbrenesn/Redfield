#!/usr/bin/env python

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Spin-boson model

# Parameters
rc = 2 # Number of levels in RC
m = [rc, rc]
eps = 0.5
delta = 0.5
om1 = 20.0 # self-energy of left RC
om2 = 20.0 # self-energy of right RC
la1 = 1.5 # coupling spin and left RC
la2 = 1.5 # coupling spin and right RC
t_l = 1.0 # temperature of left bath
t_r = 0.5 # temperature of right bath
ga = 0.0071 # residual coupling
wc = 1000 # cut-off frequency

# Matrices
sigx = np.array([[0.0, 1.0], [1.0, 0.0]])
sigy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sigz = np.array([[1.0, 0.0], [0.0, -1.0]])
ata = lambda m : 0.5 * np.diag([i for i in range(1, 2 * m, 2)])
atpa = lambda m : np.diag( np.sqrt([i for i in range(1, m)]), 1) \
                  + np.diag( np.sqrt([i for i in range(1, m)]), -1)

time1 = time.time()
# Hamiltonian RC with two reservoirs
# Order: system + reservoir1 + reservoir2
Ham = 0.5 * eps * np.kron(sigz, np.kron(np.eye(m[0]), np.eye(m[1]))) \
    + 0.5 * delta * np.kron(sigx, np.kron(np.eye(m[0]), np.eye(m[1]))) \
    + om1 * np.kron(np.eye(2), np.kron(ata(m[0]), np.eye(m[1]))) \
    + om2 * np.kron(np.eye(2), np.kron(np.eye(m[0]), ata(m[1]))) \
    + la1 * np.kron(sigz, np.kron(atpa(m[0]), np.eye(m[1]))) \
    + la2 * np.kron(sigz, np.kron(np.eye(m[0]), atpa(m[1])))

# Interaction Hamiltonian between RCs and residual baths, system OPs
Vl = np.kron(np.eye(2), np.kron(atpa(m[0]), np.eye(m[1])))
Vr = np.kron(np.eye(2), np.kron(np.eye(m[0]), atpa(m[1])))
time2 = time.time()
print("# Time Hamiltonian:", time2 - time1)

"""
# There's a mismatch between paper and code for this func
def ohmic_spectrum_l(w):
    n_b = 1.0 / (np.exp( np.absolute(w) / t_l ) - (1.0 * (w != 0)))
    j_s = ga * np.absolute(w) * np.exp(-1.0 * np.absolute(w) / wc)
    return (j_s * (n_b + 1.0) * (w > 0)) + (j_s * n_b * (w < 0)) + (ga * t_l * (w == 0.0))

def ohmic_spectrum_r(w):
    n_b = 1.0 / (np.exp( np.absolute(w) / t_r ) - (1.0 * (w != 0)))
    j_s = ga * np.absolute(w) * np.exp(-1.0 * np.absolute(w) / wc)
    return (j_s * (n_b + 1.0) * (w > 0)) + (j_s * n_b * (w < 0)) + (ga * t_r * (w == 0.0))

H = Qobj(Ham)
Vlo = Qobj(Vl)
Vro = Qobj(Vr)

time1 = time.time()
R, ekets = bloch_redfield_tensor(H, [[Vlo, ohmic_spectrum_l], [Vro, ohmic_spectrum_r]], sec_cutoff = 1.0)
time2 = time.time()
print("# Time Redfield tensor:", time2 - time1)

plt.spy(R.data)
plt.savefig('sparse.png')
#tlist = np.linspace(0, 15.0, 1000)
#
#psi0 = rand_ket(2)
#
#e_ops = [sigmax(), sigmay(), sigmaz()]
#
#expt_list = bloch_redfield_solve(R, ekets, psi0, tlist, e_ops)
#
#print(R[0])
"""
