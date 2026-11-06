# TipBox

TipBox is a collection of tipping systems, combining stochastic box models, empirical observations, and simulation components for sim-to-real studies. It covers bifurcation- and rate-induced tipping and provides classical early-warning indicators, including lag-1 autocorrelation and variance, for studying their warning signals.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://juannat7.github.io/tipbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## Installation

```bash
git clone https://github.com/juannat7/tipbox.git
cd tipbox
pip install -e .
```

## Generate or load a time series

TipBox provides a common interface to simulate tipping systems and load observations. See the [system collection](#systems) for available models and their documentation.

```python
from tipbox.models import MayHarvesting, SaddleNode
from tipbox.datasets import load_veraart2012

# Bifurcation-induced tipping: n is the number of sampled observations.
harvesting = MayHarvesting()
bifurcation = harvesting.make_trajectory(n=1200, noise=1.0, seed=42)

# Rate-induced tipping: simulate 2,401 observations of a moving equilibrium.
rate = SaddleNode(epsilon=1.4).make_trajectory(n=2401, noise=0.01, seed=43)

# Empirical observations: Veraart et al. (2012), chemostats M1 and M2.
observed = load_veraart2012()
```

Simulation results contain `time` and `trajectory` arrays; loaded observations contain `time` and `observations`. Equations, parameters, and simulation controls are in the [documentation](https://juannat7.github.io/tipbox/usage.html).

![Bifurcation-induced tipping, rate-induced tipping, and empirical observations](docs/_static/figures/trajectories.png)

Empirical panels: [CC BY-NC 4.0](docs/_static/figures/LICENSE.md).

## Compute early-warning indicators

Choose an observable and apply a trailing-window statistic before the fold:

```python
from tipbox.indicators import rolling_ar1, rolling_variance

before_fold = bifurcation.param_values < harvesting.critical_parameter
signal = bifurcation.trajectory[before_fold, 0]
ar1 = rolling_ar1(signal, window=200, detrend="linear")
variance = rolling_variance(signal, window=200, detrend="linear")
```

Here `window=200` means 200 observations. The [indicator reference](https://juannat7.github.io/tipbox/indicators.html) also covers autocorrelation, skewness, and kurtosis.

![AR1 and variance for the same three time series](docs/_static/figures/indicators.png)

## Evaluate with AUROC

Compare indicators across independent tipping and non-tipping trajectories using the area under the ROC curve (AUROC). Each trajectory contributes one indicator-trend score; the curve measures how well those scores distinguish the two classes.

### Generate a batch

Generate labeled batches for repeated experiments or indicator comparisons:

```python
from tipbox.models import MayHarvesting
from tipbox.datasets import generate_dataset
from tipbox.indicators import rolling_ar1, rolling_variance
from tipbox.metrics import ensemble_roc, plot_roc

batch = generate_dataset(
    MayHarvesting(), n_tipping=5, n_null=5,
    n=1000, num_timesteps=300, noise=1.0, seed=42,
)
signals = batch.observations[:, :, 0]  # 10 trajectories, 300 retained samples each.
indicators = {
    "AR1": [rolling_ar1(x, window=100, detrend="linear") for x in signals],
    "Variance": [rolling_variance(x, window=100, detrend="linear") for x in signals],
}
report = ensemble_roc(batch.labels, indicators)
print(report.curves["Variance"].auroc)
ax = plot_roc(report)
```

`n` controls the full simulation length; `num_timesteps` selects the context retained before each reference event. See the [benchmark guide](https://juannat7.github.io/tipbox/benchmarks.html) for labels, trend scores, and evaluation conventions.

## Systems

Each model name links to its equations, defaults, and usage.

| Model | Tipping mechanism | Reference |
|---|---|---|
| [MayHarvesting](https://juannat7.github.io/tipbox/models/harvesting.html) | Fold | May (1977) |
| [Rosenzweig<br>MacArthur](https://juannat7.github.io/tipbox/models/consumer_resource.html) | Transcritical / Hopf | Bury et al. (2021) |
| [SEIRx](https://juannat7.github.io/tipbox/models/seirx.html) | Transcritical | Bury et al. (2021) |
| [AMOC](https://juannat7.github.io/tipbox/models/amoc.html) | Fold | Alkhayuon et al. (2019) |
| [SaddleNode](https://juannat7.github.io/tipbox/models/saddle_node.html) | Rate-induced | Ashwin et al. (2012) |
| [Bautin](https://juannat7.github.io/tipbox/models/bautin.html) | Rate-induced | Huang et al. (2024) |
| [CompostBomb](https://juannat7.github.io/tipbox/models/compost.html) | Rate-induced | Wieczorek et al. (2011) |
| [AMOCRate](https://juannat7.github.io/tipbox/models/amoc.html) | Rate-induced | Alkhayuon et al. (2019) |
| [Ricker](https://juannat7.github.io/tipbox/models/ricker.html) | Transcritical | Ricker (1954) |
| [Thermoacoustic](https://juannat7.github.io/tipbox/models/thermoacoustic.html) | Subcritical Hopf | Bonciolini et al. (2018) |

![Phase portraits and simulated trajectories, with start and end markers](docs/_static/figures/overview.png)

Empirical cases include the microcosm, voice onset, cellular ATP, paleoclimate, and electrical blackout records. The [dataset guide](https://juannat7.github.io/tipbox/datasets.html) lists sources and loaders; the [sim-to-real guide](https://juannat7.github.io/tipbox/sim_to_real.html) covers Daphnia and thermoacoustic studies.

## Documentation and examples

[Quickstart](https://juannat7.github.io/tipbox/quickstart.html) · [API](https://juannat7.github.io/tipbox/api.html) · [Figure reproduction](https://juannat7.github.io/tipbox/figures.html)

To build the documentation or reproduce the figures locally:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
python examples/figures.py
```

Runnable notebooks:

- [Generation and evaluation](notebook/01_generate_batch.ipynb)
- [Parameter sensitivity](notebook/02_sensitivity_analysis.ipynb)

## Used in research

- Nathaniel et al. (2026), [Koopman early warning signals for bifurcation and rate-induced tipping](https://arxiv.org/abs/2608.14716). Code: [KEWS](https://github.com/juannat7/kews).
- Sevinchan, Nathaniel, et al. (2026), [In-context learning to predict critical transitions in dynamical systems](https://arxiv.org/abs/2605.12308). NeurIPS 2026.
