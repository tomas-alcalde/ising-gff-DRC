# The Ising model and the Gaussian free field

Basic Python scripts for simulations of (most of) the models featuring in
[arXiv:2602.05886](https://arxiv.org/abs/2602.05886) and
[arXiv:2602.06011](https://arxiv.org/abs/2602.06011).
The behaviour and usage of each `.py` file is explained within.

Let me know if you spot any mistakes!

## Directory structure

### [`ising/`](ising/)

Scripts for the Ising (and XOR-Ising) model with plus or periodic boundary
conditions, using the Swendsen–Wang algorithm.

- [`ising_sw_per.py`](ising/ising_sw_per.py) — Swendsen–Wang sampler on the `N x N`
  torus, plus other the grid utilities and plotting helpers shared throughout.
- [`ising_sw_plus.py`](ising/ising_sw_plus.py) — Swendsen–Wang sampler on the `N x N`
  square grid with `+` boundary conditions.
- [`xor.py`](ising/xor.py) — Convenient plotting of the XOR-Ising model, the pointwise product
  of two independent Ising models.

### [`gff/`](gff/)

Scripts for the Gaussian free field and its (real or imaginary) multiplicative chaos,
sampled via a Karhunen–Loève expansion, summed with a FFT.

- [`gff_fft_zero.py`](gff/gff_fft_zero.py) — GFF with zero boundary conditions on the unit square.
- [`lqg.py`](gff/lqg.py) — Liouville quantum gravity measure `exp(gamma * GFF)`.
- [`cos_gff.py`](gff/cos_gff.py), [`sin_gff.py`](gff/sin_gff.py) — Imaginary multiplicative chaos
  `cos(alpha * GFF)` and `sin(alpha * GFF)`.

### [`coupling/`](coupling/)

Scripts for single Ising currents, double Ising currents and the associated height
function. For more details on the construction (and history) of the coupling, we refer
to Sections 2.1–2.2 of [arXiv:2602.05886](https://arxiv.org/abs/2602.05886) and
references therein.

- [`currents.py`](coupling/currents.py) — Single and double Ising currents,
  built out of `+` boundary condition Swendsen–Wang samples sprinkled with
  independent Bernoulli percolation. Returns each current's trace, odd part and even
  part. Also samples the coupled double-FK representation `omega`
- [`height.py`](coupling/height.py) — the associated height function, computed by a BFS on
  the clusters of `omega` and its coupled double current. 

### [`perco_plotting/`](perco_plotting/)

Definition of a `Percolation` class (of more general interest) to aid moving
between primal and dual models, plotting the cluster of a fixed point, and other
features. Its `__main__` block simulates and plots Bernoulli percolation.

### [`images/`](images/)

Examples of plots obtained from the above simulations, sorted into subdirectories
([`gff`](images/gff/), [`height`](images/height/), [`ising`](images/ising/),
[`omega`](images/omega/)).

## Requirements

`numpy`, `scipy` and `matplotlib`. Most scripts call on others, so run scripts from inside their own directory (each has a `__main__` block with a sample call).
