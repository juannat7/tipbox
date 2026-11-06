Using the models
================

Configure and simulate
----------------------

Each named model accepts its typed configuration or keyword overrides of its
configuration fields. Unknown fields raise an error. Configuration is immutable
and is shared by simulation and the generated parameter reference.

.. code-block:: python

   from tipbox.models import MayHarvesting, MayHarvestingConfig

   model = MayHarvesting(r=1.0, sigma=0.02)
   same_model = MayHarvesting(MayHarvestingConfig(r=1.0, sigma=0.02))
   print(model.get_parameters())

   result = model.make_trajectory(
       1000, dt=0.01, dt_sample=1.0, noise=0.01,
       initial_state=[0.8], seed=42,
   )

``n`` is the exact number of observations, including the initial state. The
physical interval is ``[t_start, t_start + (n-1)*dt_sample]``. If ``n`` is omitted,
the configured ``t_end`` determines the interval. The integration step ``dt`` and
output spacing ``dt_sample`` are distinct; output spacing must be an integer
multiple of the integration step. AMOC defaults to ``dt_inner`` for integration
and its configuration's ``dt`` for output sampling.
For the discrete Ricker map, ``dt`` is the time represented by one update;
changing it is not an ODE accuracy refinement.

Ecological bifurcation forcing is normalized to the simulated duration. Changing
``n`` at fixed sampling interval changes its ramp rate. Rate-induced forcing
uses absolute time: shortening a trajectory that begins far before time zero
may never reach the forcing change. Supply a custom forcing callable to define
a schedule independently of the requested duration.

.. code-block:: python

   result = model.make_trajectory(
       1000, initial_state=[0.8], seed=42,
       forcing=lambda t: 0.15 + 0.0001*t,
   )

The callable receives the integration times and, for RK4, its midpoints. It must
return a finite scalar at each supplied time.

Noise and initial states
------------------------

The argument ``noise`` has the historical ``sigma_tilde`` meaning. It multiplies
the configured diffusion amplitudes, except in the saddle-node model, where it
is a diffusion intensity and the amplitude is ``sqrt(2*noise)``. The model pages
state the exact stochastic equations. For continuous-time models,
``result.sigma`` includes ``sqrt(dt)`` and reports an integration-step scale.
For Ricker it is the per-update Gaussian log-growth scale, without ``sqrt(dt)``.

``noise=0`` disables process noise. An initial-state sampler can still depend on
the seed; provide ``initial_state`` for a deterministic initial condition. The
seed is local to the call and does not alter global random state.

``innovations`` optionally replaces random standard-normal increments. Its shape
is ``((n-1)*stride, model.noise_dimension)``, where ``stride=dt_sample/dt``. This
supports reproducible forcing experiments and coupled-noise comparisons. Retain
custom forcing and innovation inputs when reproducing such a run; the integer
seed alone does not encode user-supplied inputs.

Physical states and observations
--------------------------------

All new-model observations have shape ``(n, n_observables)``. Inspect
``model.state_names`` and ``model.observable_names`` before choosing a variable.
SEIRx has four physical states but returns infectious count and vaccinator
fraction. Both AMOC models return transport followed by the two salinity-anomaly
states. Scalar models have one observed column.

The sampled physical state is retained in ``result.metadata['state_trajectory']``.
Initial states and innovations refer to physical dimensions, which can differ
from the number of output columns.
Ricker's state and initial-state override are log population, with population
as the observable. Thermoacoustic state is pressure and scaled velocity, with
their phase-plane amplitude included as a third observation.

Transition markers and numerical methods
----------------------------------------

A forcing choice is not an observed outcome: ``null_simulation=True`` selects
the model's null schedule, while sufficiently large noise may still produce a
transition. Analytic bifurcation markers are recomputed for harvesting,
Rosenzweig–MacArthur and SEIRx; an explicit marker overrides that calculation.
AMOC's fold marker is calibrated and must be reconsidered when changing its
physical parameters.

The continuous-time ecological models and normal forms use Euler–Maruyama.
Compost updates carbon before evaluating that step's temperature drift. AMOC
and the thermoacoustic oscillator use RK4 drift with true forcing midpoints,
then add their diffusion increments. Ricker uses discrete map updates in log
coordinates. Projections and numerical constraints are described on each model page.

``make_trajectory`` performs eager input validation and is not a JAX-transformable
function. The underlying kernels remain JAX implementations. Historical models'
low-level ``simulate`` methods support model-dependent differentiable workflows,
as demonstrated in the parameter-sensitivity notebook. For Ricker and the
thermoacoustic oscillator, ``simulate`` is an eager adapter to the shared method;
derivatives require their underlying integration kernels.

Generate benchmark segments
---------------------------

The established generation interface returns a full simulation, a context
segment before its marker, and metadata. It retries rejected simulations only;
invalid inputs and model errors propagate.

.. code-block:: python

   import jax
   from tipbox.generate import generate
   from tipbox.models import MayHarvestingConfig

   full, context, info = generate(
       jax.random.PRNGKey(42), tipping_type='b_harvesting',
       config=MayHarvestingConfig(sigma=0.02),
       num_timesteps=500, steps_before_transition=10,
   )

The metadata's ``transition_index`` refers to the full trajectory. Ecological
bifurcation markers identify parameter crossings. Rate models generally use
state thresholds; rate AMOC uses the forcing's time-zero marker after checking
that a collapse occurs. A null trajectory's marker is a reference point, not an
observed transition. These definitions must be fixed before comparing scores.
