#!/usr/bin/env python

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time

import argparse
import sys
import os.path
# Two qubits coupled to reservoirs weak coupling Redfield

# Parameters
parser = argparse.ArgumentParser(description='Compute heat currents SB model Redfield WC')
parser.add_argument('--l', action='store', dest='la', type=float, help='Coupling')

args = parser.parse_args()

la_h = args.la # coupling spin and hot bath
la_c = args.la # coupling spin and cold bath

# Matrices
sigx = np.array([[0.0, 1.0], [1.0, 0.0]])
sigy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sigz = np.array([[1.0, 0.0], [0.0, -1.0]])

time1 = time.time()
# Hamiltonian RC with two reservoirs
# Order: system + reservoir1 + reservoir2
Ham = 0.5 * np.array(sigx)

# Interaction Hamiltonian between RCs and residual baths, system OPs
Vh = la_h * np.array(sigz)
Vc = la_c * np.array(sigz)
time2 = time.time()
print("# Time Hamiltonian:", time2 - time1)

def brownian_spectrum_h(w):
    t_h = 1.0 # temperature of left bath
    ga = 0.0071 / np.pi # residual coupling
    om_h = 28.0 # self-energy of hot bath
    aw = np.absolute(w)
    n_b = 1.0 / (np.exp( aw / t_h ) - (1.0 * (w != 0)))
    j_s = (4.0 * ga * om_h * om_h * aw) / ( (((om_h * om_h) - (aw * aw)) * ((om_h * om_h) - (aw * aw))) + ((2.0 * np.pi * ga * om_h * aw) * (2.0 * np.pi * ga * om_h * aw)) )

    return (np.pi * j_s * (n_b + 1.0) * (w > 0)) + (np.pi * j_s * n_b * (w < 0)) + ((4.0 * np.pi * ga / (om_h * om_h / t_h)) * (w == 0.0))
    
def brownian_spectrum_c(w):
    t_c = 0.5 # temperature of right bath
    ga = 0.0071 / np.pi # residual coupling
    om_c = 28.0 # self-energy of cold bath
    aw = np.absolute(w)
    n_b = 1.0 / (np.exp( aw / t_c ) - (1.0 * (w != 0)))
    j_s = (4.0 * ga * om_c * om_c * aw) / ( (((om_c * om_c) - (aw * aw)) * ((om_c * om_c) - (aw * aw))) + ((2.0 * np.pi * ga * om_c * aw) * (2.0 * np.pi * ga * om_c * aw)) )

    return (np.pi * j_s * (n_b + 1.0) * (w > 0)) + (np.pi * j_s * n_b * (w < 0)) + ((4.0 * np.pi * ga / (om_c * om_c / t_c)) * (w == 0.0))
    

H = Qobj(Ham)
H0 = np.array([[0.0, 0.0], [0.0, 0.0]])
H0o = Qobj(H0)
Vho = Qobj(Vh)
Vco = Qobj(Vc)

time1 = time.time()
R, ekets = bloch_redfield_tensor(H, [[Vho, brownian_spectrum_h], [Vco, brownian_spectrum_c]], sec_cutoff = 1.0e16)
Rh, eketsh = bloch_redfield_tensor(H, [[Vho, brownian_spectrum_h]], sec_cutoff = 1.0e16)
time2 = time.time()
print("# Time Redfield tensor:", time2 - time1)

# \overline{W} = W + |0>><<1|
RTensor = np.array((np.real(R) * 2.0) + (1.0j * np.imag(R)))
RhTensor = np.array(np.real(Rh) * 2.0)
RTensor[0,0] += 1.0
RTensor[0,3] += 1.0

# Solution vector
Sol = np.array([1.0, 0.0, 0.0, 0.0])

Rinv = np.linalg.inv(RTensor)

SState = Rinv.dot(Sol)

Dt = np.reshape(RhTensor.dot(SState),(2,2))
Hd = np.diag(H.eigenenergies())
print("%.6f %.10f" % (args.la, np.real(np.trace(Hd.dot(Dt)))))

