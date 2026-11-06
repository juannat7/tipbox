# Figure attribution and licenses

`overview.{png,svg,pdf}` and `roc.{png,svg,pdf}` contain only TipBox simulations
and are covered by the repository's MIT license. `examples/figures.py` and
`manifest.json` are also covered by that license.

`trajectories.{png,svg,pdf}` and `indicators.{png,svg,pdf}` include a visualization of the original chemostat
observations and is distributed under **CC BY-NC 4.0**:
https://creativecommons.org/licenses/by-nc/4.0/

Source: Veraart et al. (2012), *Recovery rates reflect distance to a tipping
point in a living system*, Nature 481, 357–359,
https://doi.org/10.1038/nature10723.

Data: *Data from: Recovery rates reflect distance to a tipping point in a
living system*, DANS, https://doi.org/10.17026/dans-ztg-93aw,
`Correct data both chemostats.ods` (datafile 14452). The original deposit
specifies CC BY-NC 4.0. The plotted series is M1, retaining source timestamps
and missing values. Six 250-point recovery intervals are analyzed separately
with trailing AR1 and variance after per-window linear detrending; no raw
observations are included in the package. The manifest records selected rows,
analysis controls, and the original source SHA256.

Recovery interval annotations follow Grziwotz et al. (2023), *Anticipating the
occurrence and type of critical transitions*, Science Advances 9, eabq4558,
https://doi.org/10.1126/sciadv.abq4558; associated analysis archive:
https://doi.org/10.5281/zenodo.7379358. Numerical observations for this figure
come directly from the original DANS deposit, whose terms are distinct from
the aggregated archive's empirical-data terms.

The trajectory-gallery presentation is inspired by the dysts package:
https://github.com/GilpinLab/dysts. No dysts artwork or simulations are copied.
