# tipbox

Named stochastic box models and classical early-warning benchmarks for tipping-point time series, with JAX integration kernels.

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
| `b_ricker` | Discrete map | Density-dependent population map (growth crosses zero) | Ricker 1954 |
| `b_thermoacoustic` | Oscillator | Thermoacoustic subcritical Hopf | Bonciolini et al. 2018 |

### Rate-Induced Tipping

| System | Description | Reference |
|--------|-------------|-----------|
| `r_bautin` | Bautin normal form | Ritchie et al. 2023 |
| `r_compost_bomb` | Compost bomb instability | Wieczorek et al. 2011 |
| `r_saddle_node` | Saddle-node normal form | Ashwin et al. 2012 |
| `r_amoc` | AMOC 2-box model (rate-induced) | Ritchie et al. 2023; Alkhayuon et al. 2019 |

## Model API

```python
from tipbox.models import MayHarvesting

model = MayHarvesting(r=1.0, k=1.0, s=0.1)
result = model.make_trajectory(1000, noise=0.01, seed=42)
print(model.get_parameters())
```

Every model exposes typed configuration, physical state and observable names, and an exact-grid trajectory method. See the [model reference](docs/index.rst) for equations, units, defaults, numerical conventions, and historical compatibility. To build the documentation locally, install `pip install -e ".[docs]"` and run `python -m sphinx -b html docs docs/_build/html`.

## Classical warning indicators

```python
from tipbox.models import MayHarvesting
from tipbox.indicators import rolling_ar1, rolling_variance
from tipbox.metrics import kendall_tau

result = MayHarvesting().make_trajectory(1000, seed=42)
signal = result.trajectory[:, 0]
ar1 = rolling_ar1(signal, window=200, detrend="linear")
variance = rolling_variance(signal, window=200, detrend="linear")
print(kendall_tau(variance))
```

Trailing AR1 regression slope, Pearson autocorrelation, variance, skewness, and kurtosis share explicit window, step, and detrending controls. Results keep the input length, with `NaN` for warm-up or missing-data windows. See [definitions and defaults](docs/indicators.rst).

For a labeled ensemble, `ensemble_roc(labels, {"Variance": variance_series})` reduces each trajectory's indicator to a Kendall trend and returns ROC/AUROC with valid and missing counts. It also accepts precomputed scalar scores. See the [complete benchmark example](docs/benchmarks.rst) for generation, labels, event conventions, and plotting.

```python
from tipbox.datasets import generate_dataset

dataset = generate_dataset(
    MayHarvesting(), num_timesteps=300,
    n_tipping=10, n_null=10, n=1000, noise=0.01, seed=42,
)
# dataset.observations: (20, 300, 1); dataset.labels: (20,)
```

Batch generation preserves the model configuration and records accepted seeds, event references, physical lead times, and rejection counts. `n` controls the full simulation length; `num_timesteps` controls the retained context. The default `lead_steps=1` ends immediately before the model's reference marker.

## Empirical records

```python
from tipbox.datasets import load_empirical

record = load_empirical("voice")
time, signal = record.time, record.observations[:, 0]
print(record.metadata["sampling"])
```

Five cases are available: `microcosm`, `voice`, `mitochondria`, `climate`, and `blackout`. Records retain their original time axes, missing readings, and source metadata. Data download once into a user cache and are not bundled. The [dataset reference](docs/datasets.rst) explains analysis intervals, offline use, citations, and the upstream data-use terms, which are separate from TipBox's MIT software license.

`Ricker` and `Thermoacoustic` provide simulation components for the Daphnia and TAC sim-to-real cases. The [paired-case reference](docs/sim_to_real.rst) identifies original observations, event conventions, and calibration requirements. Their raw Dryad archives could not be accessed from the development environment; automatic observation parsers and the private TipPFN calibration are not included.

## Quickstart

```python
import jax
from tipbox.generate import generate

# Generate a tipping trajectory
full, part, info = generate(
    key=jax.random.PRNGKey(42),
    tipping_type='b_harvesting',          # or 'b_harvesting', 'r_compost_bomb', etc.
    null_simulation=False,           # True for non-tipping trajectory
    num_timesteps=500,
    sigma_tilde=0.01,
)

# full.trajectory  - complete simulation
# part.trajectory  - extracted pre-transition segment
# info             - metadata (transition_index, system_name, etc.)
# full.sigma       - integration-step noise scale
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

## Used in research

- Nathaniel et al. (2026), [Koopman early warning signals for bifurcation and rate-induced tipping](https://arxiv.org/abs/2608.14716), with the [KEWS implementation](https://github.com/juannat7/kews).
- Sevinchan et al. (2026), [In-context learning to predict critical transitions in dynamical systems](https://arxiv.org/abs/2605.12308).

These studies used earlier versions; the [compatibility reference](docs/compatibility.rst) records numerical corrections and historical settings.
