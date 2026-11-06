# tipbox

A collection of JAX-based, differentiable box models for tipping point time series generation and analysis.

## Developer's Guide

```bash
conda create -n tipbox python=3.10
conda activate tipbox
git clone https://github.com/juannat7/tipbox
cd tipbox
pip install -e .
```

The named box models require no external continuation software. The former random-polynomial generators (`b_fold`, `b_hopf`, `b_transcritical`) have been removed; their implementation is preserved in [the repository history](https://github.com/juannat7/tipbox/tree/6b6443e).

## Available Box Models

### Bifurcation-Induced Tipping

| System | Type | Description | Reference |
|--------|------|-------------|-----------|
| `b_harvesting` | Semi-real | May's harvesting model (fold) | May 1977 |
| `b_rosenzweig_macarthur_tc` | Semi-real | Consumer-resource model (transcritical) | Bury et al. 2021 |
| `b_rosenzweig_macarthur_hopf` | Semi-real | Consumer-resource model (Hopf) | Bury et al. 2021 |
| `b_seirx_tc` | Semi-real | SEIRx disease–vaccination behaviour (transcritical) | Bury et al. 2021 |
| `b_amoc` | Semi-real | AMOC 2-box model (bifurcation-induced) | Ritchie et al. 2023; Alkhayuon et al. 2019 |

### Rate-Induced Tipping

| System | Description | Reference |
|--------|-------------|-----------|
| `r_bautin` | Bautin normal form | Ritchie et al. 2023 |
| `r_compost_bomb` | Compost bomb instability | Wieczorek et al. 2011 |
| `r_saddle_node` | Saddle-node normal form | Ashwin et al. 2012 |
| `r_amoc` | AMOC 2-box model (rate-induced) | Ritchie et al. 2023; Alkhayuon et al. 2019 |

## Quickstart

```python
import jax
from tipbox import generate

# Generate a tipping trajectory
full, part, info = generate.generate(
    key=jax.random.PRNGKey(42),
    tipping_type='b_harvesting',          # or 'b_harvesting', 'r_compost_bomb', etc.
    null_simulation=False,           # True for non-tipping trajectory
    num_timesteps=500,
    sigma_tilde=0.01,
)

# full.trajectory  - complete simulation
# part.trajectory  - extracted pre-transition segment
# info             - metadata (transition_index, sigma, etc.)
```

The SEIRx simulator now applies `sigma_tilde` to its configured noise amplitudes, as documented. To reproduce its former noise level, use `sigma_tilde=1.0`; the high-level default is `0.01`. Rate models use their configured time grid; change their typed configuration to adjust integration steps or physical parameters.

## Tutorial

| Notebook | Description |
|----------|-------------|
| [`01_generate_batch.ipynb`](notebook/01_generate_batch.ipynb) | Batch generation of tipping trajectories, including tutorial on how to use the package generally |
| [`02_sensitivity_analysis.ipynb`](notebook/02_sensitivity_analysis.ipynb) | Parameter sensitivity analysis using autodiff |

## TODO

- [ ] Add three-species chaotic food chain model (Hopf) — Zhuge 2025
- [ ] Add Plant-herbivore model (Rate) — Ritchie 2023
- [ ] Add AMOC 5-box model (Rate) — Ritchie 2023
