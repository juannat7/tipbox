Historical compatibility
========================

The implementation before this refactor is preserved at
`6b6443e <https://github.com/juannat7/tipbox/tree/6b6443e>`_. The AUTO-dependent
random-polynomial generators ``b_fold``, ``b_hopf``, and ``b_transcritical`` and
their short aliases have been removed. All nine named model configurations
remain available.
The Ricker map and thermoacoustic oscillator add two named simulation components;
their source and calibration boundaries are documented in :doc:`sim_to_real`.

The established import ``from tipbox.generate import generate`` continues to
work, as does ``from tipbox import generate; generate.generate(...)``.
``make_trajectory`` is a new interface and does not reinterpret the legacy
simulator's fixed-duration or context-buffer conventions. Reproducibility is
per interface: the new method uses a shared random-increment layout, so the
same integer seed does not imply the same noise realization as legacy simulate.
The legacy harvesting, Rosenzweig–MacArthur, and SEIRx simulators interpret
``total_len`` as an interval count and return ``total_len+1`` samples, including
the initial state. The new ``make_trajectory(n)`` returns exactly ``n`` samples.

For a forced bifurcation path that never reaches its critical parameter,
``locate_transition`` returns ``None``. It no longer substitutes the last sample
for an absent crossing. Null paths retain their midpoint comparison reference.

Scientific corrections
----------------------

SEIRx now applies the documented ``sigma_tilde`` multiplier. Use
``sigma_tilde=1.0`` to recover its former amplitude; the generation default is
``0.01``. Rosenzweig–MacArthur now applies its formerly ignored ``sigma1`` and
``sigma2`` values. To recover the former legacy diffusion, use
``RosenzweigMacArthurConfig(sigma1=1.0, sigma2=1.0)`` with the former
``sigma_tilde`` value. With the default coefficients, the corrected diffusion is
100 times smaller than before these parameters became effective.

Harvesting now derives its upper fold (approximately ``0.260436``) instead of
using the rounded ``0.26`` marker. Rosenzweig–MacArthur derives its invasion
threshold (approximately ``5.8823529`` for the default coefficients) instead of
using ``5.60``; its Hopf threshold is approximately ``15.686275``. These changes
alter null caps and the exact extracted context. Explicit
``h_bifurcation=0.26``, ``a_transcritical_bifurcation=5.60`` and
``a_hopf_bifurcation=15.69`` retain the former marker conventions.

SEIRx's automatic marker follows ``delta``. AMOC's ``H_fold`` remains explicitly
calibrated; changing its physical coefficients or initial ON state requires a
new scientific calibration. Its shared physical equations preserve the previous
transport laws and year conversion.

Rate-AMOC's positive reference remains the first sample at or after time zero.
If the simulated interval does not include zero, ``locate_transition`` now
returns ``None`` rather than substituting an unrelated endpoint. The default
simulation interval spans zero and retains its previous marker.
Its collapse classifier now requires a finite transport value below the
configured threshold; negative infinity alone is numerical failure, not an event.
The legacy rate generator rejects nonfinite null paths, including physical
states when returned separately, so that numerical failure cannot become a
negative label after this classifier correction. Positive normal-form escape
after a finite transition remains supported.

Control and output changes
--------------------------

Supported rate-model initial states and supplied noise now reach their
simulators. Unsupported keywords raise errors. The legacy rate generators
retain their model-specific time grids and reject ``dt_sample``; the new model
interface supports explicit integration and output intervals for every model.

The new interface returns scalar observations with one column, SEIRx as
``[I,x]``, and both AMOC models as ``[Q,SN,ST]``. The legacy rate-AMOC simulator
continues to return scalar transport and a separate physical path. The
bifurcation AMOC output includes transport and the two physical coordinates.

Unused ``tip_ramp_end`` has been removed. ``a_*_null`` and ``omega_null`` are now
optional cap overrides; ``None`` computes the previous symmetric-cap formula.
Initial-state dispersion ``sigma_ic`` is active in AMOC ``make_trajectory``;
legacy ``simulate`` retains its configured mean unless an initial state is given.

The empirical mitochondrial record's ``time_unit`` is ``"unresolved"`` because
the source figure, methods, and stated acquisition interval conflict. Raw time
coordinates and reference markers remain unchanged; ``time_unit_notes`` retains
the evidence. See :doc:`datasets` before interpreting a physical lead time.
