SEIRx disease and vaccination behaviour
=======================================

``b_seirx_tc`` · ``tipbox.models.SEIRx``

The physical state is susceptible :math:`S`, exposed :math:`E`, infectious
:math:`I`, and vaccinator fraction :math:`x`. Recovered individuals are implicit
in the total population :math:`N`:

.. math::

   dS &= [\mu N(1-x)-\beta SI/N-\mu S]dt+\eta\sigma_S dW_S,\\
   dE &= [\beta SI/N-(\epsilon+\mu)E]dt+\eta\sigma_E dW_E,\\
   dI &= [\epsilon E-(\gamma+\mu)I]dt+\eta\sigma_I dW_I,\\
   dx &= \kappa x(1-x)[-\omega(t)+I+\delta(2x-1)]dt
          +\eta\sigma_x dW_x.

Time is measured in weeks. The vaccination-risk forcing ramps from
``omega_min`` to ``omega_max`` over :math:`T/\text{param_end_multiplier}` and then
stays constant. The automatic null cap is
:math:`\max(\omega_{\min},2\omega_c-\omega_{\max})`; ``omega_null`` overrides it.
The disease-free behavioural marker is :math:`\omega_c=\delta`, unless an
explicit ``omega_bifurcation`` is supplied.

The initial state is :math:`(0,0,0,1)`. Each Euler–Maruyama update first floors
all four coordinates at ``1e-8``, then clips :math:`x` to :math:`[0,1]` and
rescales :math:`(S,E,I)` if their sum exceeds :math:`N`. This retained numerical
floor seeds tiny exposed and infectious populations even with ``noise=0``;
the implemented disease-free boundary is therefore approximate.
Observations contain :math:`(I,x)`; all four sampled physical coordinates remain
in result metadata.

.. model-parameters:: b_seirx_tc

References: `Oraby et al. (2014) <https://doi.org/10.1098/rspb.2013.3172>`_,
`Pananos et al. (2017) <https://doi.org/10.1073/pnas.1704093114>`_, and the
benchmark formulation in `Bury et al. (2021)
<https://doi.org/10.1073/pnas.2106140118>`_.
