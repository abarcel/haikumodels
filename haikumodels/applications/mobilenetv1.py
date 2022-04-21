from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

import h5py
import haiku as hk
import jax
import jax.numpy as jnp

from .. import utils
from ..ops import BatchNorm
from haiku._src.typing import PRNGKey

hk.BatchNorm = BatchNorm

BASE_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/"


class block(hk.Module):

  def __init__(
      self,
      output_channels: int,
      alpha: float,
      channel_multiplier: int = 1,
      stride: int = 1,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.filters = int(output_channels * alpha)
    self.stride = stride

    self.depthwise_conv = hk.DepthwiseConv2D(
        channel_multiplier=channel_multiplier,
        kernel_shape=3,
        padding="SAME" if stride in [1, (1, 1)] else "VALID",
        stride=stride,
        with_bias=False,
        name="depthwise_conv",
        **next(weights_init),
        **wb_init,
    )
    self.depthwise_conv_bn = hk.BatchNorm(
        name="depthwise_bn",
        **next(weights_init),
        **bn_config,
    )

    self.pointwise_conv = hk.Conv2D(
        output_channels=self.filters,
        kernel_shape=1,
        with_bias=False,
        name="pointwise_conv",
        **next(weights_init),
        **wb_init,
    )
    self.pointwise_conv_bn = hk.BatchNorm(
        name="pointwise_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    out = inputs

    if self.stride in [2, (2, 2)]:
      out = jnp.pad(
          out,
          ((0, 0), (0, 1), (0, 1), (0, 0)),
          "constant",
          constant_values=(0, 0),
      )

    out = self.depthwise_conv(out)
    out = self.depthwise_conv_bn(out, is_training)
    out = jax.nn.relu6(out)
    out = self.pointwise_conv(out)
    out = self.pointwise_conv_bn(out, is_training)
    out = jax.nn.relu6(out)

    return out


class MobileNetV1(hk.Module):
  """Instantiates the MobileNetV1 architecture.
    See https://arxiv.org/pdf/1704.04861.pdf for details.
    Optionally loads weights pre-trained on ImageNet.
    Note that the default input image size for this model is 224x224.
    """

  def __init__(
      self,
      include_top: bool = True,
      weights: str = "imagenet",
      alpha: float = 1.0,
      channel_multiplier: int = 1,
      dropout_rate: float = 1e-3,
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      ckpt_dir: Optional[str] = None,
      name: str = "mobilenet_v1",
  ):
    """Initializes a MobileNetV1 function.
        Args:
          include_top: Whether to include the fully-connected layer at the top
            of the network.
            By default, True.
          weights: One of None (random initialization) or ``imagenet``
            (pre-trained on ImageNet).
            By default, ``imagenet``.
          alpha: Controls the width of the network. This is known as the width
            multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
            decreases the number of filters in each layer. - If `alpha` > 1.0,
            proportionally increases the number of filters in each layer. - If
            `alpha` = 1, default number of filters from the paper are used at
            each layer.
            By default, ``1.0``.
          channel_multiplier: Depth multiplier for depthwise convolution. This is
            called the resolution multiplier in the MobileNet paper.
            By default, ``1``.
          dropout_rate: Dropout rate.
            By default, ``1e-3``.
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
            only to be optionally specified if `include_top` is True
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
            By default, ``mobilenet_v1``.
        """

    super().__init__(name=name)
    self.dw_blocks = []
    self.default_sizes = [128, 160, 192, 224]
    self.min_size = 32
    self.include_top = include_top
    self.weights = weights
    self.alpha = alpha
    self.channel_multiplier = channel_multiplier
    self.dropout_rate = dropout_rate
    self.pooling = pooling
    self.classes = 1000 if weights == "imagenet" else classes
    self.classifier_activation = classifier_activation
    self.ckpt_dir = ckpt_dir

    if weights == "imagenet":
      wb_init, bn_config = None, None

    self.wb_init = dict(wb_init or {})

    self.bn_config = dict(bn_config or {})
    self.bn_config.setdefault("decay_rate", 0.99)
    self.bn_config.setdefault("eps", 1e-3)
    self.bn_config.setdefault("create_scale", True)
    self.bn_config.setdefault("create_offset", True)

  def init_blocks(self, inputs: jnp.ndarray):
    if self.weights == "imagenet":
      if self.channel_multiplier != 1:
        print("When loading from ``imagenet`` weights, "
              "`channel_multiplier` must be ``1``."
              "\tEntered value " + str(self.channel_multiplier) +
              "is replaced with ``1``")
        self.channel_multiplier = 1
      if self.include_top:
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
        if (inputs.shape[1] and inputs.shape[2]
            ) not in self.default_sizes or inputs.shape[1] != inputs.shape[2]:
          raise ValueError(
              "When setting `include_top` as True "
              "and loading ``imagenet`` weights, "
              "`inputs` height and width should be same and one of " +
              str(self.default_sizes))
    if (inputs.shape[1] or inputs.shape[2]) < self.min_size:
      raise ValueError("Input size must be at least " + str(self.min_size) +
                       "x" + str(self.min_size) + "; got `inputs` shape as ``" +
                       str(inputs.shape[1:3]) + "``.")

    if self.weights == "imagenet":
      if self.alpha >= 1.0:
        alpha_text = "1_0"
      elif self.alpha == 0.75:
        alpha_text = "7_5"
      elif self.alpha == 0.50:
        alpha_text = "5_0"
      else:
        alpha_text = "2_5"

    model_weights = None
    if self.weights == "imagenet":
      net_shape = 224
      for size in self.default_sizes:
        if size in [inputs.shape[1], inputs.shape[2]]:
          net_shape = size
      model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, net_shape)
      ckpt_file = utils.download(self.ckpt_dir, BASE_URL + model_name)
      model_weights = h5py.File(ckpt_file, "r")
    weights_init = utils.load_weights(model_weights)

    filters = int(32 * self.alpha)
    self.conv1 = hk.Conv2D(
        output_channels=filters,
        kernel_shape=3,
        stride=2,
        with_bias=False,
        name="conv1",
        **next(weights_init),
        **self.wb_init,
    )
    self.conv1_bn = hk.BatchNorm(
        name="conv1_bn",
        **next(weights_init),
        **self.bn_config,
    )

    self.dw_blocks = []

    for i, (filters, stride) in enumerate(
        zip(
            [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024],
            [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1],
        )):
      self.dw_blocks.append(
          block(
              output_channels=filters,
              alpha=self.alpha,
              channel_multiplier=self.channel_multiplier,
              stride=stride,
              weights_init=weights_init,
              wb_init=self.wb_init,
              bn_config=self.bn_config,
              name=f"block{i+1:02d}",
          ))

    if self.include_top:
      self.conv_preds = hk.Conv2D(
          output_channels=self.classes,
          kernel_shape=1,
          name="conv_preds",
          **next(weights_init),
          **self.wb_init,
      )

  def __call__(self,
               inputs: jnp.ndarray,
               is_training: bool,
               rng: Optional[PRNGKey] = None):
    out = inputs

    if not self.dw_blocks:
      self.init_blocks(inputs)

    out = self.conv1(out)
    out = self.conv1_bn(out, is_training)
    out = jax.nn.relu6(out)

    for dw_block in self.dw_blocks:
      out = dw_block(out, is_training)

    if self.include_top:
      out = jnp.mean(out, axis=(1, 2), keepdims=True)
      if is_training and rng:
        out = hk.dropout(rng, self.dropout_rate, out)
      out = self.conv_preds(out)
      out = jnp.squeeze(out, axis=(1, 2))
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
