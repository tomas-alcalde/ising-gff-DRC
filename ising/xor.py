"""
Sampling of the XOR-Ising model, the product of two independent Ising models.

Provides the periodic and + boundary condition versions, each of which draws two
independent Swendsen-Wang samples and multiplies them spin by spin.
"""

import numpy as np
from ising_sw_per import Spins, plot_bare, plot_bare_triple
from ising_sw_plus import SpinsPlusBC


def xor_per(N, J=0.5*np.log(1+np.sqrt(2)), n_iter=1e2, show=False, multishow=False):
    """
    Sample the XOR-Ising model on the N x N torus (periodic boundary conditions).
    """

    s1 = Spins(N=N, J=J, n_iter=n_iter)
    s1.iter()
    s2 = Spins(N=N, J=J, n_iter=n_iter)
    s2.iter()
    t = s1.state * s2.state
    if show:
        plot_bare(t)
    if multishow:
        plot_bare_triple(s1.state, s2.state, t)
    return t


def xor_plus(N, J=0.5*np.log(1+np.sqrt(2)), n_iter=1e2, show=False, multishow=False):
    """
    Sample the XOR-Ising model on the N x N square with + boundary conditions.
    """

    s1 = SpinsPlusBC(N=N, J=J, n_iter=n_iter)
    s1.iter()
    s2 = SpinsPlusBC(N=N, J=J, n_iter=n_iter)
    s2.iter()
    t = s1.state * s2.state
    if show:
        plot_bare(t)
    if multishow:
        plot_bare_triple(s1.state, s2.state, t)
    return t


if __name__ == "__main__":
    xor_plus(N=100, multishow=True)
