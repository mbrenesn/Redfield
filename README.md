# Redfield
Equilibrium/Out-of-equilibrium calculations using Redfield master equation with reaction coordinates

## Description

This application solves for steady states of a central quantum system coupled to either 1 or 2 reservoirs. Several drivers are implemented in [drivers](https://github.com/mbrenesn/Redfield/tree/master/drivers) for either different quantum systems or different reservoir configurations.

The coupling between the system and the reservoir has been implemented in the so-called 'phonon' coupling, whereby the coupling between S and R is of the form of A \otimes B, with just *one term*. Different types of couplings require implementation on the [Redfield class](https://github.com/mbrenesn/Redfield/blob/master/src/Redfield/Redfield.h).

This application depends on the Intel MKL libraries. The Makefile is hard-coded for Intel compilers.
