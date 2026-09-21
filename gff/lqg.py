"""
Sampling and plotting of the Liouville quantum gravity measure on the unit square.

The density exp(gamma * GFF) is taken with respect to a Dirichlet GFF.
"""

import numpy as np
import matplotlib.pyplot as plt
from gff_fft_zero import gff


def lqg(gamma, N, show2d=False, show3d=False):
    """ 
    Sample the Liouville quantum gravity measure exp(gamma * GFF) on the unit square.
    """
    
    field = np.exp(gamma * gff(N))

    if show2d:
        fig, ax = plt.subplots()
        vmax = field.max()
        im = ax.imshow(
            field,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            interpolation="auto",
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    if show3d:
        I, J = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
        vmax = field.max()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(I / N, J / N, field, cmap="viridis", vmin=0, vmax=vmax)
        plt.tight_layout()
        plt.show()

    return field


if __name__ == "__main__":
    N = 100
    lqg(gamma=1, N=N, show2d=True, show3d=True)
