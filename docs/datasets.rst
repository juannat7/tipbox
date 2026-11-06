Empirical records
=================

TipBox loads the five empirical cases used in
`KEWS <https://github.com/juannat7/kews/tree/main/experiments/empirical>`_. They
come from Grziwotz et al. (2023), *Anticipating the occurrence and type of
critical transitions*, `Science Advances 9, eabq4558
<https://doi.org/10.1126/sciadv.abq4558>`_, archived as
`Zenodo v1, 7379358 <https://doi.org/10.5281/zenodo.7379358>`_.
Coupled GISS AMOC is outside this initial collection.

.. code-block:: python

   from tipbox.datasets import get_empirical_info, list_empirical, load_empirical

   print(list_empirical())
   print(get_empirical_info("voice"))  # Inspect reference settings offline.
   record = load_empirical("voice")   # Download once, then use the cache.
   print(record.time.shape, record.observations.shape)
   print(record.metadata["sampling"])

Data remain in a user cache; no source data are bundled with the package.
The loader validates the pinned archive size and SHA256 on every load and reads
only the requested member and source README, without extracting the archive.
Original row order, time coordinates, duplicates, and missing observations are
preserved. In particular, the climate file's value/time column order is
converted explicitly to ``time`` and ``observations[:, 0]``.

Sources and terms
-----------------

The archive README contains specific empirical-data terms: acknowledge the
source, use monitoring data for nonprofit activities, and contact the relevant
data administrators listed in the paper's Table S2 before publications or
presentations; coauthorship may be requested depending on use. Read the
`source record <https://zenodo.org/records/7379358>`_ and
``record.metadata["terms"]`` for the complete original terms. The cached ZIP
retains the README and upstream software license. TipBox's MIT license applies
to its software; it does not replace those data terms. Loading records does not
establish permission to republish the data or derived figures.

Metadata records the source DOI, paper DOI, contributor, original member name,
archive location and SHA256, terms, and ``preprocessing="none"``. Source
contributions are Egbert H. van Nes (microcosm; Veraart et al. 2012), Isao T.
Tokuda (voice), Markus Schwarzländer (mitochondria), the World Data Center for
Paleoclimatology (climate; also analyzed by Dakos et al. 2008), and the Bonneville
Power Administration (blackout). Cite the archive and paper together with the
original source appropriate to your record.

Cases and example settings
--------------------------

Reference markers, trend-start times, and example window/step sample counts
follow the archive plotting script and KEWS. These are retrospective annotations
and analysis choices, not estimates made by the loader or universal EWS defaults.

.. list-table:: Pinned raw records and reference settings
   :header-rows: 1
   :widths: 15 19 19 12 12 12 11

   * - Case
     - Observable
     - Time unit
     - Rows
     - Marker
     - Trend start
     - Window / step
   * - ``microcosm``
     - Light attenuation (m^-1)
     - day
     - 7914
     - 27
     - 0
     - 200 / 5
   * - ``voice``
     - Subglottal pressure (kPa)
     - s
     - 5640
     - 0.49
     - 0.3
     - 200 / 50
   * - ``mitochondria``
     - Relative fluorescence ratio
     - unresolved; raw source coordinates
     - 271
     - 4.7
     - 3.2
     - 125 / 1
   * - ``climate``
     - CaCO3 (%)
     - million years, signed age
     - 483
     - -34
     - -38
     - 150 / 5
   * - ``blackout``
     - Frequency (Hz)
     - s
     - 11300
     - 550
     - 200
     - 1000 / 50

Climate time increases toward the present from -39.93 to -32 million years;
the negative sign is retained. The microcosm record includes 130 missing
observations, climate includes one adjacent duplicate timestamp, and
mitochondria is irregularly sampled. Voice and blackout are regular to the
reported tolerance. Microcosm recovery times are also rounded in the supplied
file. ``metadata["sampling"]`` reports adjacent minimum/median/maximum time
increments, duplicate and backward counts, missing counts, and regularity using
``rtol=1e-5, atol=0`` with strictly positive increments. No interval is inferred
across dropped rows because no rows are dropped.

The mitochondrial case has inconsistent upstream time labels. The archive
script and published Fig. 3G–H label seconds; the
`cellular-energy methods <https://pmc.ncbi.nlm.nih.gov/articles/PMC9821862/>`_
describe minutes and also report a 40-second acquisition interval. The median
raw increment is approximately 0.011111. TipBox therefore reports
``time_unit="unresolved"`` and retains the discrepancy in
``time_unit_notes``; it does not rescale the timestamps or reference values.
Resolve the physical clock before reporting durations or lead times for this
record. Its original experiment is Wagner et al. (2019),
`New Phytologist 224, 1668–1684 <https://doi.org/10.1111/nph.16093>`_.

Indicator example
-----------------

Use observations strictly before the retrospective marker, retain warm-up
context, and score the specified trend interval. Voice has regular timestamps:

.. code-block:: python

   from tipbox.datasets import load_empirical
   from tipbox.indicators import rolling_ar1, rolling_variance
   from tipbox.metrics import kendall_tau

   record = load_empirical("voice")
   metadata = record.metadata
   before = record.time < metadata["transition_time"]
   time = record.time[before]
   signal = record.observations[before, 0]
   controls = dict(window=metadata["example_window"], step=metadata["example_step"])
   ar1 = rolling_ar1(signal, **controls)
   variance = rolling_variance(signal, **controls)
   trend = time >= metadata["trend_start"]
   print(kendall_tau(ar1[trend], time=time[trend]))

This is a single observed transition, so it does not provide a binary AUROC.
The available reference marker is used to select a retrospective analysis
window, not supplied to an online warning indicator. Choose and document a
sampling policy before interpreting AR lags on irregular records. The loader
does not interpolate, remove duplicates, infer events, assign null labels, or
fit a dynamical model.

Microcosm recovery intervals
--------------------------------

The microcosm experiment has repeated perturbations. Its six analysis intervals
are P1 [0.57092, 1.4539], P2 [4.5496, 5.4326], P3 [10.642, 11.525],
P4 [14.454, 15.337], P5 [19.499, 20.382], and P6 [24.502, 25.385] days,
with inclusive endpoints. Each contains 250 finite observations in this archive.
Compute within each interval; concatenating them would create artificial
lagged pairs across separate recoveries:

.. code-block:: python

   from tipbox.datasets import load_empirical
   from tipbox.indicators import rolling_variance

   record = load_empirical("microcosm")
   results = {}
   for label, start, stop in record.metadata["recovery_intervals"]:
       keep = (record.time >= start) & (record.time <= stop)
       results[label] = rolling_variance(
           record.observations[keep, 0], window=200, step=5,
       )

Stored times have finite rounding precision. Treating index-lags as a uniform
physical lag inside a recovery interval is an explicit approximation, rather
than a correction silently performed by TipBox.

.. _original-microcosm-source:

Veraart et al. (2012)
---------------------

``load_veraart2012`` reads the full-precision Veraart et al. (2012) deposit,
`doi:10.17026/dans-ztg-93aw <https://doi.org/10.17026/dans-ztg-93aw>`_. It
retains full-precision times and both measured chemostats, ``M1`` and ``M2``.
``load_microcosm_original`` remains an alias; ``load_empirical("microcosm")``
loads the rounded, single-chemostat archive copy instead:

.. code-block:: python

   from tipbox.datasets import load_veraart2012

   record = load_veraart2012()
   time = record.time                         # (7914,), days
   m1, m2 = record.observations.T             # light attenuation, m^-1
   print(record.metadata["sampling_by_observable"])

The pinned file is ``Correct data both chemostats.ods`` (897,788 bytes),
retrieved directly from DANS datafile 14452. TipBox reads the first sheet's
measurement rows and first three columns with the Python standard library;
baseline columns are not subtracted. M1 contains 130 missing readings and M2
contains 103. All rows and missing values remain in place. This original
deposit uses `CC BY-NC 4.0
<https://creativecommons.org/licenses/by-nc/4.0/>`_; cite Veraart et al.,
`Nature 481, 357–359 <https://doi.org/10.1038/nature10723>`_ and the data DOI.
These terms differ from the aggregated Zenodo record above.

M1 matches the aggregated microcosm series after that file's five-significant-
digit rounding. The six recovery intervals used in :doc:`figures` correspond
to zero-based, stop-exclusive row slices ``[161:411]``, ``[1283:1533]``,
``[3001:3251]``, ``[4076:4326]``, ``[5273:5523]``, and ``[6684:6934]``.
Each contains 250 finite M1 readings, uniformly spaced by approximately
0.0035461 day. Applying the rounded aggregate time bounds to the original
times would omit a boundary row. The full experiment has interruptions;
recovery intervals are analyzed separately. No reference event or recovery
annotations are inferred for M2.

``cache_dir=None`` and ``download=True`` follow the shared cache conventions
below, using the ``dans_ztg_93aw`` subdirectory. ``source_path=None`` can be
replaced with a local unmodified ODS path for offline use; it overrides cache
lookup and never downloads. Every load verifies file size and SHA256
``cec2680a8af7cc13e8afe31f92b9315d9c481b23523d73f9c47b552f75132f3e``.

Cache and offline use
---------------------

All loader arguments are keyword-only except the case name:

.. list-table:: Loader controls
   :header-rows: 1

   * - Argument
     - Default
     - Behavior
   * - ``cache_dir``
     - ``None``
     - ``$XDG_CACHE_HOME/tipbox`` or ``~/.cache/tipbox``; the archive is placed in
       a ``zenodo_7379358`` subdirectory. Empty or relative XDG values are ignored.
   * - ``download``
     - ``True``
     - Fetch the 1,016,323-byte archive if absent. False requires a cached copy.
   * - ``archive_path``
     - ``None``
     - Use a local, unmodified copy of the pinned ZIP; overrides cache lookup
       and never downloads.

.. code-block:: python

   from tipbox.datasets import load_empirical

   record = load_empirical("voice", cache_dir="/path/to/cache", download=False)
   record = load_empirical("voice", archive_path="/path/to/DEV_Bifurcation-main.zip")

A missing offline archive raises ``FileNotFoundError``. Corrupt or different
versions raise ``ValueError`` and are not silently overwritten. Downloads are
verified before atomic placement in the cache; interruptions do not leave a
partial file at the expected path. HTTP and network failures propagate to the
caller. The pinned SHA256 is
``5e5fb299890014f6053e98c3e63b93a21dd9c7bff9212391c9cfd2e7d5dc9275``;
the Zenodo record also supplies MD5 ``0e7977de28d1f22f33da38325a8c8a43``.

API
---

.. automodule:: tipbox.datasets.empirical
   :members: EmpiricalSpec, EmpiricalRecord, list_empirical, get_empirical_info, load_empirical

.. autofunction:: tipbox.datasets.load_veraart2012
