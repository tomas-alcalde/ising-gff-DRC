"""
Swendsen-Wang sampling of the Ising model on the N x N torus.
By default, sampled at the critical temperature

Also holds the grid utilities shared with the other boundary conditions: bond_grid
and label_bond_grid. Plotting helpers for spin configurations live here too.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Plus-shaped connectivity: each vertex has edges to top-right-down-left
BOND_STRUCTURE = np.array([[0, 1, 0], 
                           [1, 1, 1], 
                           [0, 1, 0]])


def bond_grid(h_bonds, v_bonds, pad=0):
    """
    Embeds vertices and edges on the same grid. Vertices sit at even row/column indices, and 
    each bond sits at the odd index between the two vertices it connects
    A cell is 0 iff. it is a closed bond

    h_bonds is N x (N-1) (bond to the right of vertex) 
    v_bonds is (N-1) x N (bond to the bottom of vertex)

    The padding reserves all-zero cells on every side, to implement boundary conditions eventually
    """
    
    N = h_bonds.shape[0]
    size = 2 * N - 1 + 2 * pad
    grid = np.zeros((size, size), dtype=int)

    grid[pad:pad + 2 * N - 1:2, pad:pad + 2 * N - 1:2] = 1
    grid[pad:pad + 2 * N - 1:2, pad + 1:pad + 2 * N - 2:2] = h_bonds
    grid[pad + 1:pad + 2 * N - 2:2, pad:pad + 2 * N - 1:2] = v_bonds

    return grid


def label_bond_grid(grid, N, pad=0):
    """
    Labels the clusters of a grid (i.e. bond percolation model) 
    Each cell (whether edge or vertex) holds the id of its unique cluster, including isolated vertices
    Closed bonds have label zero 

    vertex_labels is the (N, N) sub-array of labeled_grid holding only vertex id's
    n_labels is the number of clusters, including isolated vertices
    """

    labeled_grid, n_labels = label(grid, structure=BOND_STRUCTURE)
    vertex_labels = labeled_grid[pad:pad + 2 * N - 1:2, pad:pad + 2 * N - 1:2]
    return labeled_grid, vertex_labels, n_labels


class Spins:
    """
    Samples the Ising model on the N x N torus 
    Default J is the critical temperature.
    """

    def __init__(self, N, J = 0.5*np.log(1+np.sqrt(2)), n_iter=1e2, snaps=1):
        self.N = N
        self.J = J
        self.n_iter = int(n_iter)
        self.snaps = snaps
        self.snap_iters = set(self.n_iter * k for k in range(1, self.snaps + 1))
        self.samples = []
        self.state = np.random.choice([1,-1], (N,N))
        self.start_state = self.state.copy()
        self.iters_done = 0

    def agree(self, state):
        """
        Agreement of every vertex with its right and bottom neighbours, with the
        periodic wrapping edges split off into their own arrays.
        """

        N = self.N

        bottom = state == np.roll(state, -1, axis=0)  # (N,N): agreement with vertex below, wrapping
        right = state == np.roll(state, -1, axis=1)   # (N, N): agreement with vertex right, wrapping

        # Split off the periodic wrapping edges 
        # Convention: h_agree is N x (N-1), v_agree is (N-1) x N.
        # This matches the input of bond_grid() 
        h_agree, wrap_h_agree = right[:, :N - 1], right[:, N - 1]
        v_agree, wrap_v_agree = bottom[:N - 1, :], bottom[N - 1, :]
        return h_agree, v_agree, wrap_h_agree, wrap_v_agree

    def _bonds(self, state):
        """
        Opens each agreeing edge independently with probability 1 - exp(-2J).
        """

        N = self.N
        p = 1 - np.exp(-2 * self.J)

        h_agree, v_agree, wrap_h_agree, wrap_v_agree = self.agree(state)

        h_bonds = h_agree & (np.random.rand(N, N - 1) < p)
        v_bonds = v_agree & (np.random.rand(N - 1, N) < p)
        wrap_h_bonds = wrap_h_agree & (np.random.rand(N) < p)
        wrap_v_bonds = wrap_v_agree & (np.random.rand(N) < p)
        return h_bonds, v_bonds, wrap_h_bonds, wrap_v_bonds

    def _update(self, state):
        """
        Performs single step of resampling
        """

        N = self.N
        h_bonds, v_bonds, wrap_h_bonds, wrap_v_bonds = self._bonds(state)

        grid = bond_grid(h_bonds, v_bonds)
        labeled_grid, vertex_labels, n_labels = label_bond_grid(grid, N)

        # Manually merge clusters across wrapping bonds 
        # Then small union-find over the resulting labels.
        parent = np.arange(n_labels + 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in np.nonzero(wrap_h_bonds)[0]:
            union(vertex_labels[i, N - 1], vertex_labels[i, 0])
        for j in np.nonzero(wrap_v_bonds)[0]:
            union(vertex_labels[N - 1, j], vertex_labels[0, j])

        roots = np.array([find(lbl) for lbl in range(n_labels + 1)])
        cluster_signs = np.random.choice([1, -1], size=n_labels + 1)
        return cluster_signs[roots[vertex_labels]]

    def iter(self):
        """
        Iteratates the resampling in _update()
        """

        for i in range(self.n_iter * self.snaps):
            self.state = self._update(self.state)
            self.iters_done += 1
            if self.iters_done in self.snap_iters:
                self.samples.append((self.iters_done, self.state.copy()))
        return self.state

    def plot(self):
        """
        Default plotting option for sampled states
        """

        if not self.samples:
            return
        fig, axes = plt.subplots(1, len(self.samples), figsize=(4 * len(self.samples), 4))
        if len(self.samples) == 1:
            axes = [axes]
        for ax, (it, state) in zip(axes, self.samples):
            ax.imshow(state, cmap="Greys", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Iterations={it}")
        fig.suptitle(f"Swendsen-Wang Ising sampler (N={self.N}, J={self.J:.3f})")
        plt.tight_layout()
        plt.show()


def plot_bare(state):
    """
    Plots a fixed sample, with no title or text.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(state, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_bare_triple(state1, state2, state3):
    """
    Plots three fixed samples next to each other, with no title or text.
    Used for plotting the XOR-Ising model with its two respective Ising models (see xor.py)
    """

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, state in zip(axes, (state1, state2, state3)):
        ax.imshow(state, cmap="Greys", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s = Spins(N=100, snaps=3)
    s.iter()
    s.plot()
    plot_bare(s.state)
