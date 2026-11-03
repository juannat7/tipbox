Rate-induced saddle-node normal form
====================================

``r_saddle_node`` · ``tipbox.models.SaddleNode``

.. math::

   dX &= [(X+\Lambda(t))^2-1]dt+\sqrt{2\eta}\,dW_t,\\
   \Lambda(t)&=\frac{\Lambda_{\max}}2
      \left[\tanh\left(\frac{\Lambda_{\max}\epsilon t}{2}\right)+1\right].

The frozen equilibria translate as the forcing increases. Sufficiently rapid
translation can drive a trajectory away from the stable branch without a local
loss of frozen-system stability. The null scenario multiplies ``epsilon`` by
``null_rate_factor``. The time origin is absolute; the main forcing change is
near zero. Initial states are uniform on the configured interval.

Here ``noise`` is :math:`\eta`, a diffusion intensity. Its per-step scale is
:math:`\sqrt{2\eta\,dt}`. The tipping marker is the first state crossing of
``tip_threshold``. Scalar observations have one column in ``make_trajectory``;
the historical simulator retains its dummy second column.

.. model-parameters:: r_saddle_node

References: `Ashwin et al. (2012)
<https://doi.org/10.1098/rsta.2011.0306>`_ and `Ritchie et al. (2023)
<https://doi.org/10.5194/esd-14-669-2023>`_.
