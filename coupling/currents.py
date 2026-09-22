"""
Random current representations of the Ising model, along with related percolation models

All models are built out of Swendsen-Wang samples of the + boundary condition Ising model, 
sprinkled with independent Bernoulli percolation. 
Each current is returned as its trace, odd part and even part.

IMPORTANT NOTE: When off-critical temperature, currents are sampled at the dual 
temperature J^* satisfying exp(-2J^*) = tanh(J).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.append(str(Path(__file__).resolve().parent.parent / "ising"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "perco_plotting"))

from ising_sw_plus import SpinsPlusBC
from perco_plotting import Percolation, _format_ax


def _bernoulli(N, p):
    """
    Independent Bernoulli percolation of parameter p
    """

    return np.random.rand(N, N-1) < p, np.random.rand(N-1, N) <p


def _sprinkle(N, p):
    """
    Bernoulli(p) sprinkling on the thickened (N+2) x (N+2) grid that s.agree() returns,
    with the wired boundary ring reopened afterwards.
    """

    eta_h, eta_v = _bernoulli(N+2, p)
    eta_h[0, :] = True; eta_h[-1, :] = True
    eta_v[:, 0] = True; eta_v[:, -1] = True
    return eta_h, eta_v


def _even_part(trace, odd):
    """
    Given the odd part of a current (SRC or DRC) and its trace, find its even trace.
    """

    return Percolation(trace.h_bonds | ~odd.h_bonds,
                       trace.v_bonds | ~odd.v_bonds)


def _single_current(prefix, xi, eta):
    """
    Trace / odd part / even part of the SRC.
    Obtained by first sampling the dual of SRC, then passed through Percolation.dual()
    """

    xi_h, xi_v = xi
    eta_h, eta_v = eta

    trace = Percolation(xi_h & eta_h, xi_v & eta_v)
    odd = Percolation(xi_h, xi_v)
    return {f"{prefix}_dual": trace,
            f"{prefix}_odd_dual": odd,
            f"{prefix}_even_dual": _even_part(trace, odd)}


def sample_traces(N, J=0.5*np.log(1+np.sqrt(2)), n_iter=1e2, omega=False, SRC=False, DRC=False):
    """
    Sample the doubled-FK model omega, the trace of the SRC, and/or the trace of the DRC.
    Returns a dict of Percolation states, keyed by "SRC", "SRC2", "omega", and "DRC". 
    Also returns the (primal) XOR-Ising spin field under "tau", used in height.py

    The setup is such that trace(DRC) = trace(SRC) u trace(SRC2).
    The odd parts compose as odd(DRC) = odd(SRC) \Delta odd(SRC2),
    and even(DRC) = ((even(SRC) u even(SRC2)) \ odd(DRC)) u (odd(SRC) n odd(SRC2)).
    """

    if not (SRC or DRC or omega):
        print("No percolation model has been selected for sampling")
        return {}

    states = {}

    J_dual = 0.5 * np.log(1/np.tanh(J))
    p_sprinkle = 1 / np.cosh(J_dual)

    s1 = SpinsPlusBC(N=N, J=J, n_iter=n_iter)
    s1_sample = s1.iter()
    xi1 = s1.agree(s1_sample)
    eta1 = _sprinkle(N, p_sprinkle)

    if SRC:
        states.update(_single_current("SRC", xi1, eta1))

    if omega or DRC:
        s2 = SpinsPlusBC(N=N, J=J, n_iter=n_iter)
        s2_sample = s2.iter()
        xi2 = s2.agree(s2_sample)
        eta2 = _sprinkle(N, p_sprinkle)

        if SRC:
            states.update(_single_current("SRC2", xi2, eta2))

        states["omega"] = Percolation(xi1[0] & eta1[0] & xi2[0] & eta2[0],
                                      xi1[1] & eta1[1] & xi2[1] & eta2[1])

        # DRC is exactly the dual of omega
        states["DRC"] = states["omega"].dual()

        t_sample = s1_sample * s2_sample
        tau = np.ones((N+2, N+2), dtype=int)
        tau[1:N+1, 1:N+1] = t_sample
        states["tau"] = tau

        # The odd part of the DRC is the domain walls of tau 
        states["DRC_odd_dual"] = Percolation(*s1.agree(t_sample))
        states["DRC_even_dual"] = _even_part(states["omega"], states["DRC_odd_dual"])

    return states


def _src_prefixes(states):
    """
    Check which single currents present in the states dict, in sampling order
    """

    return [prefix for prefix in ("SRC", "SRC2") if f"{prefix}_dual" in states]


def plot_traces(states, omega=False, SRC=False, DRC=False):
    """
    Plot the states returned by sample_traces
    """

    if SRC:
        for prefix in _src_prefixes(states):
            states[f"{prefix}_dual"].plot(show_primal=False, show_dual=True)
    if omega or DRC:
        states["omega"].plot(show_primal=omega, show_dual=DRC)


def plot_omega_bdy(N, J=0.5*np.log(1+np.sqrt(2)), n_iter=1e2):
    """
    Plot the boundary cluster of omega
    """

    states = sample_traces(N, J, n_iter, omega=True)
    states["omega"].plot_cluster(target=(0,0))


def _plot_parity(states, prefix):
    """
    Plot the parity of one current: odd part in blue, even part in red
    """

    fig, ax = plt.subplots(figsize=(4,4))
    ax.add_collection(LineCollection(states[f"{prefix}_even_dual"].dual().bonds_to_lines(), colors="red", linewidths=1.5, zorder=2))
    ax.add_collection(LineCollection(states[f"{prefix}_odd_dual"].dual().bonds_to_lines(), colors="blue", linewidths=1.5, zorder=3))
    _format_ax(ax)
    plt.tight_layout()
    plt.show()


def plot_parities(states, SRC=False, DRC=False):
    """
    Plot parities of the single currents and of the double current
    """

    if SRC:
        for prefix in _src_prefixes(states):
            _plot_parity(states, prefix)
    if DRC:
        _plot_parity(states, "DRC")


if __name__ == "__main__":
    omega, SRC, DRC = True, True, True
    states = sample_traces(N=100, omega=omega, SRC=SRC, DRC=DRC) 
    plot_traces(states, omega=omega, SRC=SRC, DRC=DRC)
    plot_omega_bdy(N=100)
    plot_parities(states, SRC=SRC, DRC=DRC)
