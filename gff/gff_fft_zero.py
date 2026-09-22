"""
Sampling of the Gaussian free field on the unit square, with zero boundary conditions

The field is sampled in the Dirichlet eigenbasis of the Laplacian and mapped back
to real space with a type-I discrete sine transform, then padded with its zero
boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dstn


def gff(N, show2d=False, show3d=False):
    """
    Sample the Gaussian free field on an (N+1)x(N+1) grid of the unit square
    """

    # Define indexing
    i = np.arange(1, N)
    j = np.arange(1, N)
    I, J = np.meshgrid(i, j, indexing="ij")

    # Define eigenvalues
    evals = np.pi**2 * (I**2 + J**2)

    # Define iid standard normals, and rescale them
    # NOTE: Dirichlet eigenfunctions on [0,1]^2 are 2*sin(m*pi*x)*sin(n*pi*y),
    # so need to add extra 1/2 to match DST-I convetions
    normals = np.random.normal(size=(N-1, N-1))
    normals = normals / (2 * np.sqrt(evals))

    # Map to real space via discrete sine transform, 
    # then pad the boundary conditions
    padded = np.zeros((N+1, N+1))
    gff = np.sqrt(2*np.pi)*dstn(normals, type=1)
    padded[1:N, 1:N] = gff

    if show2d:
        fig, ax = plt.subplots()
        vmax = np.abs(padded).max()
        im = ax.imshow(
            padded,
            cmap="viridis",
            vmin=-vmax,
            vmax=vmax,
            interpolation="auto",
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    if show3d:
        fix, ax = plt.subplots(subplot_kw={"projection": "3d"})
        vmax = np.abs(gff).max()
        ax.plot_surface(
            I/N,
            J/N,
            gff,
            cmap="viridis",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.tight_layout()
        plt.show()

    return padded


if __name__ == "__main__":
    gff(100, show2d=True, show3d=True)
