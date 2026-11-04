# TipBox

Named stochastic box models, empirical time series, and classical early-warning benchmarks for critical transitions.

![TipBox model gallery: eleven named configurations drawn from actual simulations](docs/_static/figures/overview.png)

Configure a model, generate reproducible trajectories, and compare warning indicators through ROC/AUROC. The compact model interface and trajectory gallery take inspiration from [dysts](https://github.com/GilpinLab/dysts). Integration kernels use JAX; continuation software is not required.

## Get started

```bash
git clone https://github.com/juannat7/tipbox.git
cd tipbox
pip install -e .
```

Python 3.10 or newer is required.

```python
from tipbox.models import MayHarvesting

model = MayHarvesting()
result = model.make_trajectory(1000, noise=1.0, seed=42)
print(result.time.shape, result.trajectory.shape)  # (1000,) (1000, 1)
print(model.get_parameters())
```

Models expose physical coefficients, initial states, forcing, noise, integration and sampling controls. Trailing AR1 regression slope, Pearson autocorrelation, variance, skewness, and kurtosis share explicit window, step, and detrending arguments. Results retain the input length, with `NaN` for warm-up or missing-data windows.

Start with the [complete walkthrough](docs/quickstart.rst), or use the [API map](docs/api.rst) to find imports and return shapes. The [documentation](docs/index.rst) explains each model's equations, parameter defaults, units, numerical conventions, and event rules. Noise controls have model-specific physical meanings; the [shared API](docs/usage.rst) describes their use.

## See the indicators

![Bifurcation-induced, rate-induced, and empirical signals with trailing AR1 and variance](docs/_static/figures/indicators.png)

Harvesting and moving-equilibrium simulations illustrate bifurcation- and rate-induced tipping. Their indicators use observations strictly before each model's reference marker. The measured microcosm uses six separate recovery experiments, plotted against elapsed recovery time in the lower panels. Indicator windows never cross recovery boundaries.

Microcosm: Veraart et al. (2012), [original DANS deposit](https://doi.org/10.17026/dans-ztg-93aw), CC BY-NC 4.0; recovery annotations follow [Grziwotz et al. (2023)](https://doi.org/10.1126/sciadv.abq4558). This composite figure has a [separate attribution and license](docs/_static/figures/LICENSE.md). [Figure settings and reproduction](docs/figures.rst) document all controls and display choices.

## Build and evaluate a benchmark

```python
import matplotlib.pyplot as plt
from tipbox.datasets import generate_dataset
from tipbox.indicators import rolling_ar1, rolling_variance
from tipbox.metrics import ensemble_roc, plot_roc
from tipbox.models import MayHarvesting

dataset = generate_dataset(
    MayHarvesting(), num_timesteps=300,
    n_tipping=5, n_null=5, n=1000,
    noise=1.0, lead_steps=1, seed=42,
)
signals = dataset.observations[:, :, 0]  # (10, 300)
indicators = {
    "AR1": [rolling_ar1(x, window=100, detrend="linear") for x in signals],
    "Variance": [rolling_variance(x, window=100, detrend="linear") for x in signals],
}
result = ensemble_roc(dataset.labels, indicators)
print(result.curves["Variance"].auroc)
ax = plot_roc(result)
plt.show()
```

![AR1 and variance ROC curves for two simulated ensembles, with retained-score counts](docs/_static/figures/roc.png)

Each trajectory supplies one Kendall trend score. The quick example uses five tipping and five null trajectories; the figure uses 50 of each per model, predeclared windows and increasing score directions, and report score coverage. They describe these outcome-conditioned batches; models have different forcing histories, time units, and event rules. See [benchmark conventions](docs/benchmarks.rst) for labels, accepted seeds, rejection sampling, and physical lead times.

## Load empirical observations

```python
from tipbox.datasets import load_microcosm_original

record = load_microcosm_original()
print(record.observations.shape)  # (7914, 2), columns M1 and M2
print(record.metadata["sampling_by_observable"])
```

`load_empirical(name)` also loads the five cases used by [KEWS](https://github.com/juannat7/kews): `microcosm`, `voice`, `mitochondria`, `climate`, and `blackout`. Loaders preserve missing readings, original time axes, and source metadata. Data download into a user cache and are not bundled. The [dataset reference](docs/datasets.rst) covers offline use, analysis intervals, citations, and source-specific data terms.

`Ricker` and `Thermoacoustic` provide simulation components for Daphnia extinction and thermoacoustic combustion. The [sim-to-real reference](docs/sim_to_real.rst) identifies original observations, event conventions, and calibration requirements. Automatic Dryad observation parsers and the private TipPFN calibration remain pending source access.

## Model collection

| Model | Registry name | Mechanism | Reference |
|---|---|---|---|
| `MayHarvesting` | `b_harvesting` | Fold | May 1977 |
| `RosenzweigMacArthur(bifurcation_type="transcritical")` | `b_rosenzweig_macarthur_tc` | Transcritical | Bury et al. 2021 |
| `RosenzweigMacArthur(bifurcation_type="hopf")` | `b_rosenzweig_macarthur_hopf` | Hopf | Bury et al. 2021 |
| `SEIRx` | `b_seirx_tc` | Transcritical | Bury et al. 2021 |
| `AMOC` | `b_amoc` | Fold | Alkhayuon et al. 2019; Ritchie et al. 2023 |
| `Ricker` | `b_ricker` | Transcritical population map | Ricker 1954 |
| `Thermoacoustic` | `b_thermoacoustic` | Subcritical Hopf | Bonciolini et al. 2018 |
| `SaddleNode` | `r_saddle_node` | Rate-induced | Ashwin et al. 2012 |
| `Bautin` | `r_bautin` | Rate-induced | Ritchie et al. 2023 |
| `CompostBomb` | `r_compost_bomb` | Rate-induced | Wieczorek et al. 2011 |
| `AMOCRate` | `r_amoc` | Rate-induced | Alkhayuon et al. 2019; Ritchie et al. 2023 |

The [model reference](docs/models/index.rst) includes equations and defaults. The original nine configurations remain, alongside the two sim-to-real simulation components. The former AUTO-dependent random-polynomial generators (`b_fold`, `b_hopf`, `b_transcritical`) are preserved in [repository history](https://github.com/juannat7/tipbox/tree/6b6443e).

## Documentation and examples

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
JAX_PLATFORMS=cpu python examples/figures.py
```

The figure script produces PNG, SVG, PDF, and a manifest recording parameters, seeds, source checksums, software versions, and analysis choices. See [reproduction controls](docs/figures.rst). The built site also includes a [plain-text documentation index](docs/llms.txt) for coding assistants.

Runnable notebooks cover [batch generation](notebook/01_generate_batch.ipynb) and [parameter sensitivity](notebook/02_sensitivity_analysis.ipynb). The original `tipbox.generate.generate` entry point remains available; [compatibility notes](docs/compatibility.rst) explain numerical corrections and settings for earlier experiments.

## Used in research

- Nathaniel et al. (2026), [Koopman early warning signals for bifurcation and rate-induced tipping](https://arxiv.org/abs/2608.14716), with the [KEWS implementation](https://github.com/juannat7/kews).
- Sevinchan et al. (2026), [In-context learning to predict critical transitions in dynamical systems](https://arxiv.org/abs/2605.12308).

These studies used earlier versions of TipBox. Historical settings are documented in the [compatibility reference](docs/compatibility.rst).
