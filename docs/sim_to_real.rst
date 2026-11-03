Simulation-to-observation cases
===============================

The TipPFN study describes two paired cases: Daphnia extinction and a
thermoacoustic combustor (TAC). TipBox provides their reusable simulation
components, :doc:`models/ricker` and :doc:`models/thermoacoustic`, through the
same configuration, trajectory, and labeled-batch interfaces as other models.
This page identifies the original observations and the calibration and event
decisions needed to pair them. It does not reproduce private TipPFN dataset
construction code or claim its fitted parameters.

Daphnia extinction
-------------------

Original study: Drake and Griffen (2010), *Early warning signals of extinction
in deteriorating environments*,
`Nature 467, 456–459 <https://doi.org/10.1038/nature09389>`_. Original data and
code are archived at `Dryad, doi:10.5061/dryad.q3p64
<https://doi.org/10.5061/dryad.q3p64>`_. The published archive is
``data-and-code.zip`` (13,273,254 bytes; MD5
``923e08e6bafae412c43250fef0752ac7``), with a separate
``README_for_data-and-code.txt``. The Dryad metadata identifies a CC0 dedication.

Experimental populations experience either deteriorating food provision or a
constant-environment control. Preserve population identities and observation
times; do not treat moving windows from one population as independent replicates.
The TipPFN appendix discusses a fitted Ricker-map surrogate and distinguishes
an observed extinction anchor from a reference drawn in an estimated
``[271, 316]``-day bifurcation interval. These are different targets.

TipBox's Ricker map has a zero-log-growth marker, positive population outputs,
and explicit growth, density dependence, noise, and time-spacing controls. Its
coefficients and time units must be fitted or otherwise justified using
training observations before calling it a matched Daphnia simulation. It does
not simulate integer demographic counts or infer an observed extinction time.
The defaults are an illustrative regulated population, not the fitted context
ensemble from TipPFN.

Thermoacoustic combustor
------------------------

Original study: Bonciolini et al. (2018),
`doi:10.1098/rsos.172078 <https://doi.org/10.1098/rsos.172078>`_. Pressure
recordings are archived at `Dryad, doi:10.5061/dryad.4cj4k
<https://doi.org/10.5061/dryad.4cj4k>`_. The archive is
``Experimental_time_traces_tdms.zip`` (302,008,984 bytes; MD5
``82cc5298c245ad509e1ee414e2c34941``), containing stationary and ramp experiments
in TDMS format. Dryad metadata identifies a CC0 dedication.

The TipPFN description uses four microphone channels at 2 kHz, with Mic1 as the
primary signal and a 100–200 Hz band-pass Hilbert amplitude envelope. Its
experimental reference is a fitted Hopf crossing, which differs from the
possibly delayed state jump. Preserve experiment, recording, ramp, and channel
identifiers when building train/test groups. A stationary low-state trace can
provide a control; splitting adjacent portions of the same recording does not
create independent experiments.

The TipBox oscillator implements the original study's pressure equation with
its published numerical-example coefficients. Simulated pressure units,
growth-rate schedules, noise intensity, initial-state distribution, and amplitude
extraction must be calibrated or explicitly mapped before comparison to real
microphone envelopes. The simulated phase-plane radius uses velocity and is not
identical to an envelope reconstructed from pressure. Whole-record Hilbert
transforms and forward/backward filtering use future samples; such offline
processing is not automatically a causal warning pipeline.

Source access and current boundary
----------------------------------

The DOI pages and public metadata were verified. During development, the public
download routes returned HTTP 403 from this execution environment; Dryad's REST
download API returned HTTP 401 without an API account. The original raw ZIPs
could not be downloaded and inspected here. Therefore, automatic observation
parsers, fitted parameter sets, and exact TipPFN paired-episode reconstruction
are **not included**. No substitute observations or guessed file schemas are
silently supplied.

Obtain the original archives through their DOI pages or an authorized Dryad API
account, then retain their README, source version, checksum, and provenance
alongside any exported numeric records. Original data import and a verified
calibration remain necessary to complete each observation/simulation pair.
The five directly loadable empirical records in :doc:`datasets` are independent
of this access boundary.

Reference for the paired cases
------------------------------

Sevinchan et al. (2026), *In-context learning to predict critical transitions
in dynamical systems*, `Appendix C.1 and C.4
<https://arxiv.org/html/2605.12308v1#A3>`_. The simulation components here add no
Koopman, neural-network, or pretrained-model dependency.
