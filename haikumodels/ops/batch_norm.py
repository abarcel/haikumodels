"""Batch Norm."""

from typing import Optional, Sequence, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import base, initializers, module, utils
from ..ops import moving_averages

hk.get_parameter = base.get_parameter
hk.initializers = initializers
hk.Module = module.Module
hk.ExponentialMovingAverage = moving_averages.ExponentialMovingAverage
del base, initializers, module, moving_averages


class BatchNorm(hk.Module):
  """Normalizes inputs to maintain a mean of ~0 and stddev of ~1.
    See: https://arxiv.org/abs/1502.03167.
    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:
      - No scale/offset in which case ``create_*`` should be set to ``False`` and
        ``scale``/``offset`` aren't passed when the module is called.
      - Trainable scale/offset in which case ``create_*`` should be set to
        ``True`` and again ``scale``/``offset`` aren't passed when the module is
        called. In this case this module creates and owns the ``scale``/``offset``
        variables.
      - Externally generated ``scale``/``offset``, such as for conditional
        normalization, in which case ``create_*`` should be set to ``False`` and
        then the values fed in at call time.
    NOTE: ``jax.vmap(hk.transform(BatchNorm))`` will update summary statistics and
    normalize values on a per-batch basis; we currently do *not* support
    normalizing across a batch axis introduced by vmap.
    """

  def __init__(
      self,
      create_scale: bool,
      create_offset: bool,
      decay_rate: float,
      eps: float = 1e-3,
      scale_init: Optional[hk.initializers.Initializer] = None,
      offset_init: Optional[hk.initializers.Initializer] = None,
      mean_init: Optional[hk.initializers.Initializer] = None,
      var_init: Optional[hk.initializers.Initializer] = None,
      axis: Optional[Sequence[int]] = None,
      cross_replica_axis: Optional[Union[str, Sequence[str]]] = None,
      cross_replica_axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
      data_format: str = "channels_last",
      name: Optional[str] = None,
  ):
    """Constructs a BatchNorm module.
        Args:
          create_scale: Whether to include a trainable scaling factor.
          create_offset: Whether to include a trainable offset.
          decay_rate: Decay rate for EMA.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). Can only be set
            if ``create_scale=True``. By default, ``1``.
          offset_init: Optional initializer for bias (aka offset). Can only be set
            if ``create_offset=True``. By default, ``0``.
          mean_init: Optional initializer for mean.
          var_init: Optional initializer for variance.
          axis: Which axes to reduce over. The default (``None``) signifies that all
            but the channel axis should be normalized. Otherwise this is a list of
            axis indices which will have normalization statistics calculated.
          cross_replica_axis: If not ``None``, it should be a string representing
            the axis name over which this module is being run within a ``jax.pmap``.
            Supplying this argument means that batch statistics are calculated
            across all replicas on that axis.
          cross_replica_axis_index_groups: Specifies how devices are grouped.
          data_format: The data format of the input. Can be either
            ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
            default it is ``channels_last``.
          name: The module name.
        """
    super().__init__(name=name)
    if cross_replica_axis is None and cross_replica_axis_index_groups is not None:
      raise ValueError("`cross_replica_axis` name must be specified"
                       "if `cross_replica_axis_index_groups` are used.")

    self.create_scale = create_scale
    self.create_offset = create_offset
    self.eps = eps
    self.scale_init = None if not create_scale else scale_init
    self.offset_init = None if not create_offset else offset_init
    self.mean_init = mean_init
    self.var_init = var_init
    self.axis = axis
    self.cross_replica_axis = cross_replica_axis
    self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
    self.channel_index = utils.get_channel_index(data_format)
    self.mean_ema = hk.ExponentialMovingAverage(decay_rate, name="mean_ema")
    self.var_ema = hk.ExponentialMovingAverage(decay_rate, name="var_ema")

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool,
      test_local_stats: bool = False,
      scale: Optional[jnp.ndarray] = None,
      offset: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Computes the normalized version of the input.
        Args:
          inputs: An array, where the data format is ``[..., C]``.
          is_training: Whether this is during training.
          test_local_stats: Whether local stats are used when is_training=False.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.
        Returns:
          The array, normalized across all but the last dimension.
        """
    if self.create_scale and scale is not None:
      raise ValueError(
          "Cannot pass `scale` at call time if `create_scale=True`.")
    if self.create_offset and offset is not None:
      raise ValueError(
          "Cannot pass `offset` at call time if `create_offset=True`.")

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    if is_training or test_local_stats:
      mean = jnp.mean(inputs, axis, keepdims=True)
      mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)
      if self.cross_replica_axis:
        mean = jax.lax.pmean(
            mean,
            axis_name=self.cross_replica_axis,
            axis_index_groups=self.cross_replica_axis_index_groups,
        )
        mean_of_squares = jax.lax.pmean(
            mean_of_squares,
            axis_name=self.cross_replica_axis,
            axis_index_groups=self.cross_replica_axis_index_groups,
        )
      var = mean_of_squares - jnp.square(mean)
    else:
      mean = self.mean_ema.average
      var = self.var_ema.average

    w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
    w_dtype = inputs.dtype

    if is_training:
      if self.mean_init is not None:
        self.mean_ema(self.mean_init(w_shape, w_dtype), avg_init=True)
        self.mean_init = None
      if self.var_init is not None:
        self.var_ema(self.var_init(w_shape, w_dtype), avg_init=True)
        self.var_init = None
      else:
        self.mean_ema(mean)
        self.var_ema(var)

    if self.create_scale:
      scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
    elif scale is None:
      scale = np.ones([], dtype=w_dtype)

    if self.create_offset:
      offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
    elif offset is None:
      offset = np.zeros([], dtype=w_dtype)

    eps = jax.lax.convert_element_type(self.eps, var.dtype)
    inv = scale * jax.lax.rsqrt(var + eps)
    return (inputs - mean) * inv + offset
