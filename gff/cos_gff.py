"""
Sampling and plotting of cos(alpha * GFF) for a Dirichlet GFF on the unit square.

The sign of the field is the continuum analogue of an Ising spin configuration,
which show2d_spin plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from gff_fft_zero import gff


def cos_gff(N, alpha, show2d=False, show2d_spin=False, show3d=False):
    """
    Sample cos(alpha * GFF) for a Dirichlet GFF on the unit square.
    """

    field = np.cos(alpha * gff(N))

    if show2d:
        fig, ax = plt.subplots()
        im = ax.imshow(
            field,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            interpolation="bilinear",
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    if show2d_spin:
        fig, ax = plt.subplots()
        spin = np.where(field > 0, 1, 0)
        im = ax.imshow(
            spin,
            cmap=ListedColormap(["white", "black"]),
            interpolation="nearest",
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        plt.tight_layout()
        plt.show()

    if show3d:
        I, J = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(I / N, J / N, field, cmap="viridis", vmin=-1, vmax=1)
        plt.tight_layout()
        plt.show()

    return field


if __name__ == "__main__":
    N = 100
    cos_gff(N, alpha=1 / np.sqrt(2), show2d=True, show3d=True)
