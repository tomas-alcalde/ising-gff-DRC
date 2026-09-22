"""
Sampling and plotting of the associated height function on the N x N squarelattice

The height function is reconstructed from the coupling of currents.py: it
is constant on the clusters of omega and DRC, so the reconstruction is a breadth-first search 
over the bipartite graph of incident omega and DRC clusters,
started from the wired boundary cluster where h = 0.
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / "ising"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "perco_plotting"))

from ising_sw_per import bond_grid, label_bond_grid
from currents import sample_traces


def sample_height(N, J=0.5*np.log(1+np.sqrt(2)), n_iter=1e2):
    """
    Sample the height function on the N x N square.

    tau is the product of two independent + b.c. Ising models (see xor.py)
    omega is their doubled-FK trace, both on the thickened (N+2) x (N+2) grid
    DRC is Percolation.dual() of omega, on the (N+1) x (N+1) grid of bounded faces
    tau* is an iid symmetric sign per DRC cluster.

    The height function has h = 0 on the wired boundary and gradient
          h(v) - h(v*) = 0.5 * tau(v) * tau*(v*)
    across every incident primal/dual pair (i.e. corner).
    """

    M = N + 2

    states = sample_traces(N, J, n_iter, omega=True)
    tau = states["tau"]
    omega = states["omega"]
    drc = states["DRC"]

    # omega_labels is (M,M) array such that element [i, j] is the cluster label of vertex (i,j)
    # n_omega is the number of unique clusters (so labels above take value in 1, ..., n_omega)
    _, omega_labels, n_omega = label_bond_grid(bond_grid(omega.h_bonds, omega.v_bonds), M)

    # drc_labels is (M-1, M-1) since passed to (bounded) faces of the dual, but same structure
    _, drc_labels, n_drc = label_bond_grid(bond_grid(drc.h_bonds, drc.v_bonds), M - 1)

    # tau is constant on omega clusters, tau* is an iid sign per DRC cluster
    cluster_tau = np.zeros(n_omega + 1, dtype=int)
    cluster_tau[omega_labels] = tau
    cluster_tau_star = np.random.choice([1, -1], size=n_drc + 1)

    # NOTE: drc_labels[a,b] is the DRC label of the face bounded by
    # primal vertices omega_labels[a,b] (top-left), omega_labels[a, b+1] (top-right), 
    # omega_labels[a+1, b] (bottom-left), omega_labels[a+1, b+1] (bottom-right).

    # corners is a (4*(M-1)**2,) int array. 
    # It is made up of four raveled (M-1, M-1) blocks concatenated, where 
    #   block 0 = omega_labels[:-1, :-1] -> labels of primal vertices to top-left
    #   block 1 = omega_labels[:-1,  1:] -> label of primal vertices to top-right
    #   block 2 = omega_labels[ 1:, :-1] -> label of primal vertices to bottom-left
    #   block 3 = omega_labels[ 1:,  1:] -> label of primal verties to bottom-right 

    # faces has the same shape, but the four blocks are identical and 
    # such that faces[k] is the DRC label of the dual vertex whose corner is corners[k]
    # i.e. (corners[k], faces[k]) is one incident (primal corner, dual face) pair.
    corners = np.concatenate([omega_labels[:-1, :-1].ravel(), omega_labels[:-1, 1:].ravel(),
                              omega_labels[1:, :-1].ravel(), omega_labels[1:, 1:].ravel()])
    faces = np.tile(drc_labels.ravel(), 4)

    # pairs is a (K, 2) int array. 
    # It is made up of *distinct* rows [omega_label, drc_label] sorted lexicographically
    # Note K <= 4*(M-1)**2 and is typically much smaller 
    pairs = np.unique(np.stack([corners, faces], axis=1), axis=0)

    # adj_omega[u] is a list of the DRC labels p incident to omega cluster u,
    # sorted in ascending order. 
    # adj_drc[p] is the list of omega labels u incident to DRC cluster p, 
    # sorted in ascending order.
    # Equivalently adj_omega[u] is the list of column indices p where the adjacency
    # matrix A[u,p] on the diamond graph would be non-zero.
    # Since this graph is very sparse, counting only non-zero entries is much more efficient
    adj_omega = [[] for _ in range(n_omega + 1)]
    adj_drc = [[] for _ in range(n_drc + 1)]
    for u, p in pairs:
        adj_omega[u].append(p)
        adj_drc[p].append(u)

    h_omega = np.full(n_omega + 1, np.nan)
    h_drc = np.full(n_drc + 1, np.nan)

    root = omega_labels[0, 0]   # The wired boundary cluster, pinned to zero
    h_omega[root] = 0.0

    # Height values are computed via a BFS over the bipartite graph structure above. 
    # Each queue entry is a pair (cluster_label, on_primal): on_primal=True means the
    # label indexes into h_omega/adj_omega, False means it indexes into h_drc/adj_drc . 
    # Each iteration pops one already-visited cluster, walks its neighbors via 
    # the adjacency lists (thus swapping primal <-> dual), and for every neighbor not yet visited (h is np.nan) 
    # computes its height and pushes it onto the queue.
    queue = deque([(root, True)])
    while queue:
        c, on_primal = queue.popleft()
        if on_primal:
            for p in adj_omega[c]:
                if np.isnan(h_drc[p]):
                    h_drc[p] = h_omega[c] - 0.5 * cluster_tau[c] * cluster_tau_star[p]
                    queue.append((p, False))
        else:
            for u in adj_drc[c]:
                if np.isnan(h_omega[u]):
                    h_omega[u] = h_drc[c] + 0.5 * cluster_tau[u] * cluster_tau_star[c]
                    queue.append((u, True))

    return h_omega[omega_labels], h_drc[drc_labels]


def plot_height(h_primal, h_dual):
    """
    Plot the primal and dual height functions side by side
    """

    scale = max(np.abs(h_primal).max(), np.abs(h_dual).max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, h in zip(axes, (h_primal, h_dual)):
        im = ax.imshow(h, cmap="viridis", vmin=-scale, vmax=scale)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.show()


def _joint_grid(h_primal, h_dual):
    """
    Mix the primal and dual height functions onto a single (2M-1) x (2M-1) grid:
    primal vertices land on the even indices, dual vertices on the odd indices.

    The leftover cells are the midpoints of the primal (equiv. dual) edges, where h is
    not defined; interpolated here so the field has no gaps.
    """

    M = h_primal.shape[0]

    joint = np.empty((2*M - 1, 2*M - 1))
    joint[::2, ::2] = h_primal
    joint[1::2, 1::2] = h_dual
    joint[::2, 1::2] = 0.5 * (h_primal[:, :-1] + h_primal[:, 1:])
    joint[1::2, ::2] = 0.5 * (h_primal[:-1, :] + h_primal[1:, :])

    return joint


def plot_height_joint(h_primal, h_dual):
    """
    Plot the primal and dual height functions together on the full lattice
    """

    joint = _joint_grid(h_primal, h_dual)
    scale = np.abs(joint).max()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(joint, cmap="viridis", vmin=-scale, vmax=scale)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_height_joint_3d(h_primal, h_dual):
    """
    Same as plot_height_joint, but as a 3D bar plot
    """

    joint = _joint_grid(h_primal, h_dual)
    n = joint.shape[0]
    scale = np.abs(joint).max()

    x, y = np.meshgrid(np.arange(n), np.arange(n))
    z = joint.ravel()
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=-scale, vmax=scale)
    colors = cmap(norm(z))

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(projection="3d")
    ax.bar3d(x.ravel(), y.ravel(), np.minimum(z, 0), 1, 1, np.abs(z), color=colors, shade=True)
    ax.set_zlim(-scale, scale)
    ax.axis("off")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    h_primal, h_dual = sample_height(N=100)
    plot_height(h_primal, h_dual)
    plot_height_joint(h_primal, h_dual)
    plot_height_joint_3d(h_primal, h_dual)   # More than N=100 is too much for 3d plot
