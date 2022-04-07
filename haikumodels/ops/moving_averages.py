"""Moving Averages."""

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from haiku._src import base, initializers, module

hk.get_state = base.get_state
hk.set_state = base.set_state
hk.Module = module.Module
hk.initializers = initializers
del base, module, initializers


class ExponentialMovingAverage(hk.Module):
  """Maintains an exponential moving average.
    This uses the Adam debiasing procedure.
    See https://arxiv.org/pdf/1412.6980.pdf for details.
    """

  def __init__(
      self,
      decay,
      zero_debias: bool = True,
      warmup_length: int = 0,
      name: Optional[str] = None,
  ):
    """Initializes an ExponentialMovingAverage module.
        Args:
          decay: The chosen decay. Must in ``[0, 1)``. Values close to 1 result in
            slow decay; values close to ``0`` result in fast decay.
          zero_debias: Whether to run with zero-debiasing.
          warmup_length: A positive integer, EMA has no effect until
            the internal counter has reached ``warmup_length`` at which point the
            initial value for the decaying average is initialized to the input value
            after `warmup_length` iterations.
          name: The name of the module.
        """
    super().__init__(name=name)
    self.decay = decay
    self.warmup_length = warmup_length
    self.zero_debias = zero_debias

    if warmup_length < 0:
      raise ValueError(
          f"`warmup_length` is {warmup_length}, but should be non-negative.")

    if warmup_length and zero_debias:
      raise ValueError(
          "Zero debiasing does not make sense when warming up the value of the "
          "average to an initial value. Set zero_debias=False if setting "
          "warmup_length to a non-zero value.")

  def __call__(
      self,
      value: jnp.ndarray,
      update_stats: bool = True,
      avg_init: bool = False,
  ) -> jnp.ndarray:
    """Updates the EMA and returns the new value.
        Args:
          value: The array-like object for which you would like to perform an
            exponential decay on.
          update_stats: A Boolean, whether to update the internal state
            of this object to reflect the input value. When ``update_stats`` is False
            the internal stats will remain unchanged.
          avg_init: A Boolean, to initialize ``average`` when ``avg_init`` is True
            from pretrained weights with tensorflow dimension ordering.

        Returns:
          The exponentially weighted average of the input value.
        """
    if not isinstance(value, jnp.ndarray):
      value = jnp.asarray(value)

    counter = hk.get_state("counter", (),
                           jnp.int32,
                           init=hk.initializers.Constant(-self.warmup_length))
    counter = counter + 1

    decay = jax.lax.convert_element_type(self.decay, value.dtype)
    if self.warmup_length > 0:
      decay = jax.lax.select(counter <= 0, 0.0, decay)

    one = jnp.ones([], value.dtype)
    hidden = hk.get_state("hidden", value.shape, value.dtype, init=jnp.zeros)
    hidden = hidden * decay + value * (one - decay)

    average = hidden
    if self.zero_debias:
      average /= one - jnp.power(decay, counter)

    if update_stats:
      hk.set_state("counter", counter)
      hk.set_state("hidden", hidden)
      if avg_init:
        hk.set_state("average", value)
      else:
        hk.set_state("average", average)

    return average

  @property
  def average(self):
    return hk.get_state("average")
