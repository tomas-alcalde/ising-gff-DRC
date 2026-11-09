"""
Swendsen-Wang sampling of the Ising model on the N x N square with boundary conditions
alternating between + and - blocks of fixed length L.

Specialises the alternating sampler of ising_sw_plus_minus by changing the spins pinned on
the outer ring of the thickened (N+2) x (N+2) grid. The 4N boundary vertices adjacent to the
square are traversed clockwise from the top-left, and the k-th one is given the spin
(-1)^(k // L). If L does not divide 4N, the last block is shorter.
For L = 1 this agrees with the checkerboard boundary of ising_sw_plus_minus along each side,
but alternates straight through the corners, where the checkerboard repeats a sign.
"""

import numpy as np
import matplotlib.pyplot as plt

from ising_sw_per import plot_bare
from ising_sw_plus_minus import SpinsPlusMinusBC


def boundary_ring(N):
    """
    Row and column indices, in the (N+2) x (N+2) grid, of the 4N boundary vertices adjacent
    to the square, traversed clockwise from the top-left. The four corners are left out,
    since they have no interior neighbour.
    """

    k = np.arange(1, N + 1)
    rows = np.concatenate([np.zeros(N, int), k, np.full(N, N + 1), k[::-1]])
    cols = np.concatenate([k, np.full(N, N + 1), k[::-1], np.zeros(N, int)])
    return rows, cols


class SpinsPlusMinusBlocksBC(SpinsPlusMinusBC):
    """
    Samples the Ising model on the N x N square with boundary conditions alternating
    between + and - blocks of length L
    Default J is the critical temperature
    """

    def __init__(self, N, L, **kwargs):
        super().__init__(N, **kwargs)
        self.L = L

        # The boundary is fixed, so build the padded template once
        rows, cols = boundary_ring(N)
        self.template = np.ones((N + 2, N + 2), dtype=self.state.dtype)
        self.template[rows, cols] = np.where((np.arange(4 * N) // L) % 2 == 0, 1, -1)

    def padded(self, state):
        """
        Embeds state in the (N+2) x (N+2) grid, whose outer ring holds the
        alternating boundary blocks of length L.
        """

        N = self.N
        padded = self.template.copy()
        padded[1:N+1, 1:N+1] = state
        return padded


if __name__ == "__main__":
    s = SpinsPlusMinusBlocksBC(N=1000, L=100)
    s.iter()
    plot_bare(s.state)
