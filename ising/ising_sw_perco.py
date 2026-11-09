"""
Swendsen-Wang sampling of the Ising model on the N x N square with Bernoulli percolation
boundary conditions of parameter p.

Specialises the alternating sampler of ising_sw_plus_minus by changing the spins pinned on
the outer ring of the thickened (N+2) x (N+2) grid. Each boundary vertex is independently
+ with probability p and - with probability 1 - p. The boundary is sampled once, when the
sampler is built, and stays fixed (quenched) for every iteration.
"""

import numpy as np
import matplotlib.pyplot as plt

from ising_sw_per import plot_bare
from ising_sw_plus_minus import SpinsPlusMinusBC


class SpinsPercoBC(SpinsPlusMinusBC):
    """
    Samples the Ising model on the N x N square with i.i.d. Bernoulli(p) +/- boundary conditions
    Default J is the critical temperature
    """

    def __init__(self, N, p=0.5, **kwargs):
        super().__init__(N, **kwargs)
        self.p = p

        # Sample the boundary once: + with probability p, - otherwise.
        # Only the outer ring of the template is ever read, the interior is overwritten
        M = N + 2
        self.template = np.where(np.random.rand(M, M) < p, 1, -1).astype(self.state.dtype)

    def padded(self, state):
        """
        Embeds state in the (N+2) x (N+2) grid, whose outer ring holds the
        Bernoulli(p) boundary spins sampled at construction.
        """

        N = self.N
        padded = self.template.copy()
        padded[1:N+1, 1:N+1] = state
        return padded


if __name__ == "__main__":
    s = SpinsPercoBC(N=1000, p=0.5)
    s.iter()
    plot_bare(s.state)
