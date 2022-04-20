from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

import h5py
import haiku as hk
import jax
import jax.numpy as jnp

from .. import utils
from ..ops import BatchNorm, SeparableConv2D

hk.BatchNorm = BatchNorm
hk.SeparableConv2D = SeparableConv2D

URL = ("https://github.com/abarcel/haikumodels/releases/download/v0.1/"
       "xception_weights_tf_dim_ordering_tf_kernels.h5")


class entry_flow_1(hk.Module):

  def __init__(
      self,
      output_channels: Sequence[int],
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.conv1 = hk.Conv2D(
        output_channels=output_channels[0],
        kernel_shape=3,
        stride=2,
        padding="VALID",
        with_bias=False,
        name="conv1",
        **next(weights_init),
        **wb_init,
    )
    self.conv1_bn = hk.BatchNorm(
        name="conv1_bn",
        **next(weights_init),
        **bn_config,
    )

    self.conv2 = hk.Conv2D(
        output_channels=output_channels[1],
        kernel_shape=3,
        padding="VALID",
        with_bias=False,
        name="conv2",
        **next(weights_init),
        **wb_init,
    )
    self.conv2_bn = hk.BatchNorm(
        name="conv2_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    out = self.conv1(out)
    out = self.conv1_bn(out, is_training)
    out = jax.nn.relu(out)
    out = self.conv2(out)
    out = self.conv2_bn(out, is_training)
    out = jax.nn.relu(out)

    return out


class entry_flow_2(hk.Module):

  def __init__(
      self,
      output_channels: int,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.sepconv1 = hk.SeparableConv2D(
        output_channels=output_channels,
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv1",
        **wb_init,
    )
    self.sepconv1_bn = hk.BatchNorm(
        name="sepconv1_bn",
        **next(weights_init),
        **bn_config,
    )

    self.sepconv2 = hk.SeparableConv2D(
        output_channels=output_channels,
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv2",
        **wb_init,
    )
    self.sepconv2_bn = hk.BatchNorm(
        name="sepconv2_bn",
        **next(weights_init),
        **bn_config,
    )

    self.conv2d = hk.Conv2D(
        output_channels=output_channels,
        kernel_shape=1,
        stride=2,
        with_bias=False,
        name="residual",
        **next(weights_init),
        **wb_init,
    )
    self.conv2d_bn = hk.BatchNorm(
        name="residual_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    out = jax.nn.relu(out)
    out = self.sepconv1(out)
    out = self.sepconv1_bn(out, is_training)

    out = jax.nn.relu(out)
    out = self.sepconv2(out)
    out = self.sepconv2_bn(out, is_training)

    out = hk.max_pool(out, window_shape=3, strides=2, padding="SAME")

    residual = self.conv2d(inputs)
    residual = self.conv2d_bn(residual, is_training)

    out = jnp.add(out, residual)

    return out


class middle_flow_1(hk.Module):

  def __init__(
      self,
      output_channels: int,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.sepconv1 = hk.SeparableConv2D(
        output_channels=output_channels,
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv1",
        **wb_init,
    )
    self.sepconv1_bn = hk.BatchNorm(
        name="sepconv1_bn",
        **next(weights_init),
        **bn_config,
    )

    self.sepconv2 = hk.SeparableConv2D(
        output_channels=output_channels,
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv2",
        **wb_init,
    )
    self.sepconv2_bn = hk.BatchNorm(
        name="sepconv2_bn",
        **next(weights_init),
        **bn_config,
    )

    self.sepconv3 = hk.SeparableConv2D(
        output_channels=output_channels,
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv3",
        **wb_init,
    )
    self.sepconv3_bn = hk.BatchNorm(
        name="sepconv3_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    out = jax.nn.relu(out)
    out = self.sepconv1(out)
    out = self.sepconv1_bn(out, is_training)

    out = jax.nn.relu(out)
    out = self.sepconv2(out)
    out = self.sepconv2_bn(out, is_training)

    out = jax.nn.relu(out)
    out = self.sepconv3(out)
    out = self.sepconv3_bn(out, is_training)

    out = jnp.add(out, inputs)

    return out


class exit_flow_1(hk.Module):

  def __init__(
      self,
      output_channels: Sequence[int],
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.sepconv1 = hk.SeparableConv2D(
        output_channels=output_channels[0],
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv1",
        **wb_init,
    )
    self.sepconv1_bn = hk.BatchNorm(
        name="sepconv1_bn",
        **next(weights_init),
        **bn_config,
    )

    self.sepconv2 = hk.SeparableConv2D(
        output_channels=output_channels[1],
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv2",
        **wb_init,
    )
    self.sepconv2_bn = hk.BatchNorm(
        name="sepconv2_bn",
        **next(weights_init),
        **bn_config,
    )

    self.conv2d = hk.Conv2D(
        output_channels=output_channels[1],
        kernel_shape=1,
        stride=2,
        with_bias=False,
        name="residual",
        **next(weights_init),
        **wb_init,
    )
    self.conv2d_bn = hk.BatchNorm(
        name="residual_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    out = jax.nn.relu(out)
    out = self.sepconv1(out)
    out = self.sepconv1_bn(out, is_training)

    out = jax.nn.relu(out)
    out = self.sepconv2(out)
    out = self.sepconv2_bn(out, is_training)

    out = hk.max_pool(out, window_shape=3, strides=2, padding="SAME")

    residual = self.conv2d(inputs)
    residual = self.conv2d_bn(residual, is_training)

    out = jnp.add(out, residual)

    return out


class exit_flow_2(hk.Module):

  def __init__(
      self,
      output_channels: Sequence[int],
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.sepconv1 = hk.SeparableConv2D(
        output_channels=output_channels[0],
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv1",
        **wb_init,
    )
    self.sepconv1_bn = hk.BatchNorm(
        name="sepconv1_bn",
        **next(weights_init),
        **bn_config,
    )

    self.sepconv2 = hk.SeparableConv2D(
        output_channels=output_channels[1],
        with_bias=False,
        weights_init=next(weights_init),
        name="sepconv2",
        **wb_init,
    )
    self.sepconv2_bn = hk.BatchNorm(
        name="sepconv2_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    out = self.sepconv1(out)
    out = self.sepconv1_bn(out, is_training)
    out = jax.nn.relu(out)

    out = self.sepconv2(out)
    out = self.sepconv2_bn(out, is_training)
    out = jax.nn.relu(out)

    return out


class Xception(hk.Module):
  """Instantiates the Xception architecture.
    See https://arxiv.org/pdf/1610.02357.pdf for details.
    Optionally loads weights pre-trained on ImageNet.
    Note that the default input image size for this model is 299x299.
    """

  def __init__(
      self,
      include_top: bool = True,
      weights: str = "imagenet",
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      ckpt_dir: Optional[str] = None,
      name: str = "xception",
  ):
    """Initializes a Xception function.
        Args:
          include_top: Whether to include the fully-connected layer at the top
            of the network.
            By default, True.
          weights: One of None (random initialization) or ``imagenet``
            (pre-trained on ImageNet).
            By default, ``imagenet``.
          pooling: Pooling for feature extraction when `include_top` is False.
            (`pooling`, when `include_top` is True, defaults to ``avg`` and
            changes to `pooling` will be uneffective):
            - None means that the output of the model will be the 4D tensor
              output of the last convolutional block.
            - ``avg`` means that global average pooling will be applied to the
              output of the last convolutional block, and thus the output of
              the model will be a 2D tensor.
            - ``max`` means that global max pooling will be applied.
            By default, when `include_top` is False, `pooling` is None.
          classes: Number of classes to classify images into,
            only to be specified if `include_top` is True
            and `weights` argument is None.
            By default, ``1000``.
          classifier_activation: A ``jax.nn`` activation function to use on the
            "top" layer. Ignored unless `include_top` is True.
            Set `classifier_activation` to None to return the logits of the
            "top" layer. When `weights` is ``imagenet``,
            `classifier_activation` can only be set to None or ``jax.nn.softmax``.
            By default, ``jax.nn.softmax``.
          wb_init: Dictionary of two elements, ``w_init`` and ``b_init``
            weight initializers for both dense layer and convolution layers.
            Optionally specified only when `weights` is None.
            By default, ``w_init`` is truncated normal and ``b_init`` is zeros.
          bn_config: Dictionary of two elements, ``decay_rate`` and
            ``eps`` to be passed on to the :class:``~haiku.BatchNorm`` layers.
            By default, ``decay_rate`` is ``0.99`` and ``eps`` is ``1e-3``.
          ckpt_dir: Optional path to download pretrained weights.
            By default, temporary system file directory.
          name: Optional name name of the module.
            By default, ``xception``.
        """
    super().__init__(name=name)
    self.blocks = []
    self.default_size = 299
    self.min_size = 71
    self.include_top = include_top
    self.weights = weights
    self.pooling = pooling
    self.classes = classes
    self.classifier_activation = classifier_activation
    self.ckpt_dir = ckpt_dir

    if weights == "imagenet":
      wb_init, bn_config = None, None

    self.wb_init = dict(wb_init or {})

    self.bn_config = dict(bn_config or {})
    self.bn_config.setdefault("decay_rate", 0.99)
    self.bn_config.setdefault("eps", 1e-3)
    self.bn_config.setdefault("create_scale", False)
    self.bn_config.setdefault("create_offset", True)

  def init_blocks(self, inputs: jnp.ndarray):
    if self.weights == "imagenet" and self.include_top:
      if self.classes != 1000:
        print("When setting `include_top` as True "
              "and loading from ``imagenet`` weights, "
              "`classes` must be ``1000``."
              "\tEntered value ``" + str(self.classes) +
              "`` is replaced with ``1000``.")
        self.classes = 1000
      if self.classifier_activation is not (None or jax.nn.softmax):
        print("When setting `include_top` as True and loading "
              "from ``imagenet`` weights, `classifier_activation` "
              "must be None or ``jax.nn.softmax``."
              "\tEntered setting is replaced with ``jax.nn.softmax``.")
        self.classifier_activation = jax.nn.softmax
      if inputs.shape[1:] != (self.default_size, self.default_size, 3):
        raise ValueError("When setting `include_top` as True "
                         "and loading ``imagenet`` weights, "
                         "`inputs` shape should be " +
                         str((None, self.default_size, self.default_size, 3)) +
                         " where None can be any natural number.")
    if (inputs.shape[1] or inputs.shape[2]) < self.min_size:
      raise ValueError("Input size must be at least " + str(self.min_size) +
                       "x" + str(self.min_size) + "; got `inputs` shape as ``" +
                       str(inputs.shape[1:3]) + "``.")

    model_weights = None
    if self.weights == "imagenet":
      ckpt_file = utils.download(self.ckpt_dir, URL)
      model_weights = h5py.File(ckpt_file, "r")
    weights_init = utils.load_weights(model_weights, url_type="github")

    i = 1
    self.blocks.append(
        entry_flow_1(
            output_channels=[32, 64],
            weights_init=weights_init,
            wb_init=self.wb_init,
            bn_config=self.bn_config,
            name=f"block{i:02d}",
        ))

    for filters in [128, 256, 728]:
      i += 1
      self.blocks.append(
          entry_flow_2(
              output_channels=filters,
              weights_init=weights_init,
              wb_init=self.wb_init,
              bn_config=self.bn_config,
              name=f"block{i:02d}",
          ))

    for filters in [728] * 8:
      i += 1
      self.blocks.append(
          middle_flow_1(
              output_channels=filters,
              weights_init=weights_init,
              wb_init=self.wb_init,
              bn_config=self.bn_config,
              name=f"block{i:02d}",
          ))

    i += 1
    self.blocks.append(
        exit_flow_1(
            output_channels=[728, 1024],
            weights_init=weights_init,
            wb_init=self.wb_init,
            bn_config=self.bn_config,
            name=f"block{i:02d}",
        ))

    i += 1
    self.blocks.append(
        exit_flow_2(
            output_channels=[1536, 2048],
            weights_init=weights_init,
            wb_init=self.wb_init,
            bn_config=self.bn_config,
            name=f"block{i:02d}",
        ))

    if self.include_top:
      self.top_layer = hk.Linear(
          output_size=self.classes,
          with_bias=True,
          name="top_layer",
          **next(weights_init),
          **self.wb_init,
      )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    if not self.blocks:
      self.init_blocks(inputs)

    for block in self.blocks:
      out = block(out, is_training)

    if self.include_top:
      out = jnp.mean(out, axis=(1, 2))
      out = self.top_layer(out)
      if self.classifier_activation:
        out = self.classifier_activation(out)
    else:
      if self.pooling == "avg":
        out = jnp.mean(out, axis=(1, 2))
      elif self.pooling == "max":
        out = jnp.max(out, axis=(1, 2))

    return out


def preprocess_input(x):
  return utils.preprocess_input(x, mode="tf")
