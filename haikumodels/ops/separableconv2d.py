"""Separable Conv2D."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import haiku as hk
import jax.numpy as jnp


class SeparableConv2D(hk.Module):
  """Separable 2-D Depthwise Convolution using the same method with
    Xception function. This module Computes seperable convolution by
    first applying `DepthwiseConv2D` and then applying `Conv2D` as
    pointwise convolution with kernel shape of ``1``.
    """

  def __init__(
      self,
      output_channels: int,
      channel_multiplier: int = 1,
      kernel_shape: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      weights_init: Optional[dict] = None,
      w_init: Callable[[Sequence[int], Any], jnp.ndarray] = None,
      b_init: Callable[[Sequence[int], Any], jnp.ndarray] = None,
      name: Optional[str] = None,
  ):
    """Construct a Separable 2D Depthwise Convolution module.
        Args:
          output_channels: Number of output channels of pointwise convolution.
          channel_multiplier: Multiplicity of output channels of depthwise convolution.
            To keep the number of output channels the same as
            the number of input channels, set 1.
          kernel_shape: The shape of the kernel. Either an integer or a sequence of
            length 2. Defaults to 3.
          stride: Optional stride for the kernel of deptwise convolution.
            Either an integer or a sequence of length 2. Defaults to 1.
          padding: Optional padding algorithm for depthwise convolution.
            Either ``VALID`` or ``SAME`` or a callable or sequence of callables of
            length 2. Any callables must take a single integer argument equal to
            the effective kernel size and return a list of two integers representing
            the padding before and after. See haiku.pad.* for more details and
            example functions. Defaults to ``SAME``. See:
            https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
          with_bias: Whether to add a bias. By default, False.
          weights_init: Dict of optional pretrained weights.
          w_init: Optional weight initialization for both depthwise and
            pointwise convolutions. By default, truncated normal.
          b_init: Optional bias initialization. By default, zeros.
          name: The name of the module.
        """
    super().__init__(name=name)

    self.depthwise = hk.DepthwiseConv2D(
        channel_multiplier=channel_multiplier,
        kernel_shape=kernel_shape,
        stride=stride,
        padding=padding,
        with_bias=False,
        w_init=weights_init["dw_w_init"]
        if "dw_w_init" in weights_init else w_init,
        name="depthwise_kernel",
    )

    self.pointwise = hk.Conv2D(
        output_channels=output_channels,
        kernel_shape=1,
        padding="VALID",
        with_bias=with_bias,
        w_init=weights_init["pw_w_init"]
        if "pw_w_init" in weights_init else w_init,
        b_init=weights_init["b_init"] if "b_init" in weights_init else b_init,
        name="pointwise_kernel",
    )

  def __call__(self, inputs: jnp.ndarray):
    out = inputs

    out = self.depthwise(out)
    out = self.pointwise(out)

    return out
