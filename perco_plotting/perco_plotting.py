"""
Handling and plotting of bond percolations on the square lattice.

The Percolation class stores a configuration as its horizontal and vertical bond
arrays, and knows how to take its geometric dual, draw itself, and draw the cluster
of a given vertex. It is the common currency of the current and height samplers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import label


def _format_ax(ax):
    """
    Strips an axis down to the bare lattice drawing.
    """

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


class Percolation:
    """
    Aid for handling and plotting percolation models and their duals, on the square lattice
    
    Enforces N x N grid, otherwise prompts error
    h_bonds is N x (N-1) array of Booleans representing state of bond to the right of vertex
    v_bonds is (N-1) x N array of Booleans representing state of bond to the bottom of vertex

    origin is the (x, y) position of vertex [0, 0], which is what lets a dual state, whose
    vertices sit at the centres of the primal faces, plot in its true place. See dual().

    Note the index/coordinate discrepancy: entry [y, x] of h_bonds/v_bonds corresponds to
    vertex (x, y), i.e. axis 0 (rows) is treated as the y-coordinate and axis 1 (columns) as
    the x-coordinate. This matches how the underlying bond arrays are naturally indexed as
    [row, col] elsewhere (e.g. Ising state arrays).
    """

    def __init__(self, h_bonds, v_bonds, origin=(0.0, 0.0)):
        h_bonds = np.asarray(h_bonds)
        v_bonds = np.asarray(v_bonds)

        N = h_bonds.shape[0]
        if h_bonds.shape != (N, N - 1) or v_bonds.shape != (N - 1, N):
            raise ValueError(
                "Percolation requires an N x N grid: expected h_bonds of shape "
                f"({N}, {N - 1}) and v_bonds of shape ({N - 1}, {N}), got "
                f"h_bonds{h_bonds.shape} and v_bonds{v_bonds.shape}."
            )

        self.h_bonds = h_bonds
        self.v_bonds = v_bonds
        self.N = N
        self.origin = origin

    def bonds_to_lines(self):
        """
        Formats bonds into lines, in preparation of plotting
        """

        x0, y0 = self.origin

        y_h, x_h = np.nonzero(self.h_bonds)
        lines = [[(x0+x, y0+y), (x0+x+1, y0+y)] for x,y in zip(x_h, y_h)]

        y_v, x_v = np.nonzero(self.v_bonds)
        lines.extend([[(x0+x, y0+y), (x0+x, y0+y+1)] for x, y in zip(x_v, y_v)])

        return lines

    def dual(self):
        """
        Returns the geometric dual as its own Percolation instance, on the (N-1) x (N-1) grid
        of bounded faces: dual vertex [a, b] is the centre of the primal face with top-left
        corner [a, b], hence the (0.5, 0.5) shift of the origin. Since the dual is a genuine
        Percolation state, plot it and cluster it with the ordinary bonds_to_lines/plot_cluster.

        Not involutive: every dual discards that outer ring, so N drops by one each time.
        """

        x0, y0 = self.origin
        return Percolation(~self.v_bonds[:, 1:-1], ~self.h_bonds[1:-1, :],
                           origin=(x0 + 0.5, y0 + 0.5))

    def plot(self, show_primal=True, show_dual=False, main_color="black", second_color="blue"):
        """
        Plots the primal and/or dual percolation.

        If both plotted, the dual model shows in blue
        """

        fig, ax = plt.subplots(figsize=(4,4))

        if show_primal:
            ax.add_collection(LineCollection(self.bonds_to_lines(), colors=main_color, linewidths=1.5, zorder=2))

        if show_dual:
            dual_color = second_color if show_primal else main_color
            ax.add_collection(LineCollection(self.dual().bonds_to_lines(), colors=dual_color, linewidths=1.5, zorder=2))

        _format_ax(ax)
        plt.tight_layout()
        plt.show()

    def plot_cluster(self, target=None):
        """
        Plot the *primal* cluster intersecting a given vertex, defaults to centre vertex.

        target is an (x, y) coordinate pair, following the class convention above: it indexes
        the bond arrays as [y, x], so vertex (x, y) sits at labeled_grid[2*y, 2*x].
        When implemented with wired b.c., sample boundary cluster by setting target=(0,0),
        i.e. bottom-left vertex.
        """

        if target is None:
            target = (self.N // 2, self.N // 2)
        tx, ty = target

        grid = np.zeros((2*self.N-1, 2*self.N-1), dtype=int)
        grid[::2, ::2] = 1   # Vertices
        grid[::2, 1::2] = self.h_bonds
        grid[1::2, ::2] = self.v_bonds

        structure = np.array([[0,1,0], [1,1,1], [0,1,0]])
        labeled_grid, num_cluster = label(grid, structure=structure)

        cluster_id = labeled_grid[2*ty, 2*tx]
        h_cluster_ids = labeled_grid[::2, 1::2]
        v_clusters_ids = labeled_grid[1::2, ::2]
        h_target_find = np.isin(h_cluster_ids, cluster_id)
        v_target_find = np.isin(v_clusters_ids, cluster_id)

        x0, y0 = self.origin

        y_h, x_h = np.nonzero(h_target_find)
        cluster_lines = [[(x0 + x, y0 + y), (x0 + x + 1, y0 + y)] for x, y in zip(x_h, y_h)]

        y_v, x_v = np.nonzero(v_target_find)
        cluster_lines.extend([[(x0 + x, y0 + y), (x0 + x, y0 + y + 1)] for x, y in zip(x_v, y_v)])

        fig, ax = plt.subplots(figsize=(4,4))
        ax.add_collection(LineCollection(cluster_lines, colors="black", linewidths=1, zorder=2))

        _format_ax(ax)
        plt.tight_layout()
        plt.show()


# Simulate Bernoulli bond percolation
if __name__ == "__main__":
    N = 100
    p = 0.5

    h_bonds = np.random.rand(N, N-1) < p
    v_bonds = np.random.rand(N-1, N) < p
    p = Percolation(h_bonds, v_bonds)
    p.plot(show_primal=True, show_dual=True)
    p.plot_cluster()
