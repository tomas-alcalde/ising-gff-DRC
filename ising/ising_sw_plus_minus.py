"""
Swendsen-Wang sampling of the Ising model on the N x N square with alternating +/- boundary conditions.

Specialises the periodic sampler of ising_sw_per by thickening the lattice into an
(N+2) x (N+2) grid whose outer ring is pinned to the checkerboard spin (-1)^(i+j),
so that consecutive boundary vertices alternate between + and -.
"""

import numpy as np
import matplotlib.pyplot as plt

from ising_sw_per import Spins, plot_bare, bond_grid, label_bond_grid


class SpinsPlusMinusBC(Spins):
    """
    Samples the Ising model on the N x N square with alternating +/- boundary conditions
    Default J is the critical temperature
    """

    def padded(self, state):
        """
        Embeds state in the (N+2) x (N+2) grid, whose outer ring holds the
        alternating boundary spins (-1)^(i+j).
        """

        N = self.N
        M = N + 2

        i, j = np.indices((M, M))
        padded = np.where((i + j) % 2 == 0, 1, -1).astype(state.dtype)
        padded[1:N+1, 1:N+1] = state
        return padded

    def agree(self, state):
        """
        Agreement of every vertex with its right and bottom neighbours, on the
        thickened (N+2) x (N+2) grid.
        """

        padded = self.padded(state)

        h_agree = padded[:, :-1] == padded[:, 1:]   # (M, M-1): agreement with vertex right
        v_agree = padded[:-1, :] == padded[1:, :]   # (M-1, M): agreement with vertex below

        # Edges between two boundary vertices carry no interaction, so they stay closed
        # (with alternating spins they disagree anyway, this just makes it explicit)
        h_agree[0, :] = False
        h_agree[-1, :] = False
        v_agree[:, 0] = False
        v_agree[:, -1] = False

        return h_agree, v_agree

    def _bonds(self, state):
        """
        Opens each agreeing edge independently with probability 1 - exp(-2J).
        """

        p = 1 - np.exp(-2 * self.J)

        h_agree, v_agree = self.agree(state)

        h_bonds = h_agree & (np.random.rand(*h_agree.shape) < p)
        v_bonds = v_agree & (np.random.rand(*v_agree.shape) < p)

        return h_bonds, v_bonds

    def _update(self, state):
        """
        Performs a single step of resampling
        """

        N = self.N
        h_bonds, v_bonds = self._bonds(state)

        grid = bond_grid(h_bonds, v_bonds)
        labeled_grid, vertex_labels, n_labels = label_bond_grid(grid, N + 2)

        cluster_signs = np.random.choice([1, -1], size=n_labels + 1)

        # Pin the sign of every cluster touching the boundary to that boundary spin.
        # Open bonds only join agreeing spins, so no cluster touches both a + and a - boundary vertex.
        ring = np.ones((N + 2, N + 2), dtype=bool)
        ring[1:N+1, 1:N+1] = False
        cluster_signs[vertex_labels[ring]] = self.padded(state)[ring]

        return cluster_signs[vertex_labels[1:N+1, 1:N+1]]


if __name__ == "__main__":
    s = SpinsPlusMinusBC(N=3000)
    s.iter()
    plot_bare(s.state)
