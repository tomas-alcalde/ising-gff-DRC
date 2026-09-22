"""
Swendsen-Wang sampling of the Ising model on the N x N square with + boundary conditions.

Specialises the periodic sampler of ising_sw_per by thickening the lattice into an
(N+2) x (N+2) grid whose outer ring is pinned to the + spin.
"""

import numpy as np
import matplotlib.pyplot as plt

from ising_sw_per import Spins, plot_bare, bond_grid, label_bond_grid


class SpinsPlusBC(Spins):
    """
    Samples the Ising model on the N x N square with + boundary conditions
    Default J is the critical temperature
    """

    def agree(self, state):
        """
        Agreement of every vertex with its right and bottom neighbours, on the
        thickened (N+2) x (N+2) grid. 
        """

        N = self.N

        # Add boundary with all +1 spins
        padded = np.ones((N + 2, N + 2), dtype=state.dtype)
        padded[1:N+1, 1:N+1] = state

        bottom = padded[1:N+1, 1:N+1] == padded[2:N+2, 1:N+1]   # (N,N): agreement with vertex below, interior vertices only
        right = padded[1:N+1, 1:N+1] == padded[1:N+1, 2:N+2]    # (N, N): agreement with vertex right, interior vertices only

        h_core, right_cap = right[:, :N - 1], right[:, N - 1]
        v_core, bottom_cap = bottom[:N - 1, :], bottom[N - 1, :]
        top_cap = padded[0, 1:N+1] == padded[1, 1:N+1]
        left_cap = padded[1:N+1, 0] == padded[1:N+1, 1]

        # Thicken into an (N+2) x (N+2) vertex grid, where the outer ring is the wired boundary condition.
        # Thickening is convenient for plotting the boundary cluster of wired percolation models, see perco_plotting.py
        M = N + 2
        h_agree = np.zeros((M, M - 1), dtype=bool)
        h_agree[0, :] = True
        h_agree[M - 1, :] = True
        h_agree[1:N+1, 1:N] = h_core
        h_agree[1:N+1, 0] = left_cap
        h_agree[1:N+1, N] = right_cap

        v_agree = np.zeros((M - 1, M), dtype=bool)
        v_agree[:, 0] = True
        v_agree[:, M - 1] = True
        v_agree[1:N, 1:N+1] = v_core
        v_agree[0, 1:N+1] = top_cap
        v_agree[N, 1:N+1] = bottom_cap

        return h_agree, v_agree

    def _bonds(self, state):
        """
        Opens each agreeing edge independently with probability 1 - exp(-2J),
        then reopens the wired boundary ring.
        """

        p = 1 - np.exp(-2 * self.J)

        h_agree, v_agree = self.agree(state)

        h_bonds = h_agree & (np.random.rand(*h_agree.shape) < p)
        v_bonds = v_agree & (np.random.rand(*v_agree.shape) < p)

        h_bonds[0, :] = True
        h_bonds[-1, :] = True
        v_bonds[:, 0] = True
        v_bonds[:, -1] = True

        return h_bonds, v_bonds

    def _update(self, state):
        """
        Performs a single step of resampling
        """

        N = self.N
        h_bonds, v_bonds = self._bonds(state)

        # h_bonds/v_bonds already include the thickened wired boundary, so this is
        # exactly the same embed-then-label recipe as the periodic case.
        grid = bond_grid(h_bonds, v_bonds)
        labeled_grid, vertex_labels, n_labels = label_bond_grid(grid, N + 2)

        ghost_label = vertex_labels[0, 0]
        real_labels = vertex_labels[1:N+1, 1:N+1]

        cluster_signs = np.random.choice([1, -1], size=n_labels + 1)
        cluster_signs[ghost_label] = 1   # Pin sign of boundary cluster
        return cluster_signs[real_labels]


if __name__ == "__main__":
    s = SpinsPlusBC(N=100)
    s.iter()
    plot_bare(s.state)
