Daphnia and thermoacoustics
===========================

TipBox provides two simulation components discussed in
`TipPFN, Appendix C <https://arxiv.org/html/2605.12308v1#A3>`_.
Original observation parsers and the private fitted calibration are not included.

Daphnia extinction
------------------

:doc:`models/ricker` is a positive-population surrogate with a zero-log-growth
reference. It does not model integer demographic extinction. Its illustrative
coefficients and time units require calibration before pairing with observations.

Original observations: Drake and Griffen (2010),
`doi:10.1038/nature09389 <https://doi.org/10.1038/nature09389>`_;
`Dryad archive <https://doi.org/10.5061/dryad.q3p64>`_ (CC0).
Preserve population identities and observation times. An observed extinction
and the estimated bifurcation interval are different evaluation targets.

Thermoacoustic combustion
-------------------------

:doc:`models/thermoacoustic` implements Bonciolini et al.'s pressure oscillator.
Its numerical-example coefficients are not a fitted microphone model.
Original study: `doi:10.1098/rsos.172078
<https://doi.org/10.1098/rsos.172078>`_;
`Dryad pressure recordings <https://doi.org/10.5061/dryad.4cj4k>`_ (CC0, TDMS).

Calibrate pressure units, growth schedules, noise, and amplitude extraction.
The simulated phase-plane radius uses velocity; it differs from an envelope
reconstructed from pressure. Whole-record Hilbert transforms and zero-phase
filtering use future observations. Keep experiments/channels grouped in splits.

Source access
-------------

Public metadata were verified, but raw downloads returned HTTP 403/401 during
development. Neither archive was inspected here. Automatic parsers and matched
sim-to-real reproduction require those original files and a verified calibration.
The five loadable records in :doc:`datasets` are available independently.
