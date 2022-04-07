from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

import h5py
import haiku as hk
import jax
import jax.numpy as jnp

from .. import utils
from ..ops import BatchNorm

hk.BatchNorm = BatchNorm

BASE_URL = "https://github.com/abarcel/haikumodels/releases/download/v0.1/"


class block1(hk.Module):

  def __init__(
      self,
      output_channels: int,
      conv_shortcut: bool = True,
      kernel_shape: int = 3,
      stride: int = 1,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)
    self.conv_shortcut = conv_shortcut

    if conv_shortcut is True:
      self.shortcut_conv = hk.Conv2D(
          output_channels=4 * output_channels,
          kernel_shape=1,
          stride=stride,
          padding="VALID",
          name="conv_shortcut",
          **next(weights_init),
          **wb_init,
      )
      self.shortcut_conv_bn = hk.BatchNorm(
          name="conv_shortcut_bn",
          **next(weights_init),
          **bn_config,
      )

    self.conv1 = hk.Conv2D(
        output_channels=output_channels,
        kernel_shape=1,
        stride=stride,
        padding="VALID",
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
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name="conv2",
        **next(weights_init),
        **wb_init,
    )
    self.conv2_bn = hk.BatchNorm(
        name="conv2_bn",
        **next(weights_init),
        **bn_config,
    )

    self.conv3 = hk.Conv2D(
        output_channels=4 * output_channels,
        kernel_shape=1,
        padding="VALID",
        name="conv3",
        **next(weights_init),
        **wb_init,
    )
    self.conv3_bn = hk.BatchNorm(
        name="conv3_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = shortcut = inputs

    if self.conv_shortcut is True:
      shortcut = self.shortcut_conv(inputs)
      shortcut = self.shortcut_conv_bn(shortcut, is_training)

    out = self.conv1(out)
    out = self.conv1_bn(out, is_training)
    out = jax.nn.relu(out)
    out = self.conv2(out)
    out = self.conv2_bn(out, is_training)
    out = jax.nn.relu(out)
    out = self.conv3(out)
    out = self.conv3_bn(out, is_training)
    out = jax.nn.relu(out + shortcut)

    return out


class stack1(hk.Module):

  def __init__(
      self,
      output_channels: int,
      blocks: int,
      stride1: int = 2,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.blocks = []

    self.blocks.append(
        block1(
            output_channels=output_channels,
            stride=stride1,
            weights_init=weights_init,
            wb_init=wb_init,
            bn_config=bn_config,
            name="block01",
        ))

    for i in range(2, blocks + 1):
      self.blocks.append(
          block1(
              output_channels=output_channels,
              conv_shortcut=False,
              weights_init=weights_init,
              wb_init=wb_init,
              bn_config=bn_config,
              name=f"block{i:02d}",
          ))

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    for block in self.blocks:
      out = block(out, is_training)

    return out


class block2(hk.Module):

  def __init__(
      self,
      output_channels: int,
      conv_shortcut: bool = False,
      kernel_shape: int = 3,
      stride: int = 1,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)
    self.stride = stride
    self.conv_shortcut = conv_shortcut

    self.preact_bn = hk.BatchNorm(
        name="preact_bn",
        **next(weights_init),
        **bn_config,
    )

    if conv_shortcut is True:
      self.shortcut_conv = hk.Conv2D(
          output_channels=4 * output_channels,
          kernel_shape=1,
          stride=stride,
          padding="VALID",
          name="shortcut_conv",
          **next(weights_init),
          **wb_init,
      )

    self.conv1 = hk.Conv2D(
        output_channels=output_channels,
        kernel_shape=1,
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
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
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

    self.conv3 = hk.Conv2D(
        output_channels=4 * output_channels,
        kernel_shape=1,
        padding="VALID",
        name="conv3",
        **next(weights_init),
        **wb_init,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    preact = self.preact_bn(inputs, is_training)
    preact = jax.nn.relu(preact)

    if self.conv_shortcut is True:
      shortcut = self.shortcut_conv(preact)
    else:
      shortcut = (hk.max_pool(
          inputs, window_shape=1, strides=self.stride, padding="VALID")
                  if self.stride > 1 else inputs)

    out = self.conv1(preact)
    out = self.conv1_bn(out, is_training)
    out = jax.nn.relu(out)

    out = jnp.pad(out, ((0, 0), (1, 1), (1, 1), (0, 0)),
                  "constant",
                  constant_values=(0, 0))

    out = self.conv2(out)
    out = self.conv2_bn(out, is_training)
    out = jax.nn.relu(out)
    out = self.conv3(out)

    out = jnp.add(out, shortcut)

    return out


class stack2(hk.Module):

  def __init__(
      self,
      output_channels: int,
      blocks: int,
      stride1: int = 2,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.blocks = []

    self.blocks.append(
        block2(
            output_channels=output_channels,
            conv_shortcut=True,
            weights_init=weights_init,
            wb_init=wb_init,
            bn_config=bn_config,
            name="block01",
        ))

    for i in range(2, blocks):
      self.blocks.append(
          block2(
              output_channels=output_channels,
              weights_init=weights_init,
              wb_init=wb_init,
              bn_config=bn_config,
              name=f"block{i:02d}",
          ))

    self.blocks.append(
        block2(
            output_channels=output_channels,
            stride=stride1,
            weights_init=weights_init,
            wb_init=wb_init,
            bn_config=bn_config,
            name=f"block{blocks:02d}",
        ))

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    for block in self.blocks:
      out = block(out, is_training)

    return out


class ResNet(hk.Module):
  """Instantiates the ResNet architecture.
    See https://arxiv.org/pdf/1512.03385.pdf for details.
    Optionally loads weights pre-trained on ImageNet.
    Note that the default input image size for this model is 224x224.
    """

  CONFIGS = {
      "ResNet50": {
          "blocks_per_group": (3, 4, 6, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (1, 2, 2, 2),
      },
      "ResNet101": {
          "blocks_per_group": (3, 4, 23, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (1, 2, 2, 2),
      },
      "ResNet152": {
          "blocks_per_group": (3, 8, 36, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (1, 2, 2, 2),
      },
      "ResNet50V2": {
          "blocks_per_group": (3, 4, 6, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (2, 2, 2, 1),
      },
      "ResNet101V2": {
          "blocks_per_group": (3, 4, 23, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (2, 2, 2, 1),
      },
      "ResNet152V2": {
          "blocks_per_group": (3, 8, 36, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (2, 2, 2, 1),
      },
  }

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      channels_per_group: Sequence[int],
      strides_per_group: Sequence[int],
      include_top: bool = True,
      weights: str = "imagenet",
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      resnet_v2: bool = False,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      ckpt_dir: Optional[str] = None,
      name: Optional[str] = None,
  ):
    """Initializes a ResNet function.
        Args:
          blocks_per_group: A sequence of length 4 that indicates the number of
            blocks created in each group.
          channels_per_group: A sequence of length 4 that indicates the channel size
            for blocks created in each group.
          strides_per_group: A sequence of length 4 that indicates the stride
            for blocks created in each group.
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
            only to be optionally specified if `include_top` is True
            and `weights` argument is None.
            By default, ``1000``.
          classifier_activation: A ``jax.nn`` activation function to use on the
            "top" layer. Ignored unless `include_top` is True.
            Set `classifier_activation` to None to return the logits of the
            "top" layer. When `weights` is ``imagenet``,
            `classifier_activation` can only be set to None or ``jax.nn.softmax``.
            By default, ``jax.nn.softmax``.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation.
            By default, False.
          wb_init: Dictionary of two elements, ``w_init`` and ``b_init``
            weight initializers for both dense layer and convolution layers.
            Optionally specified only when `weights` is None.
            By default, ``w_init`` is truncated normal and ``b_init`` is zeros.
          bn_config: Dictionary of two elements, ``decay_rate`` and
            ``eps`` to be passed on to the :class:``~haiku.BatchNorm`` layers.
            By default, ``decay_rate`` is ``0.99`` and ``eps`` is ``1e-5``.
          ckpt_dir: Optional path to download pretrained weights.
            By default, temporary system file directory.
          name: Optional name name of the module.
        """

    super().__init__(name=name)
    self.stack_groups = []
    self.default_size = 224
    self.min_size = 32
    self.blocks_per_group = blocks_per_group
    self.channels_per_group = channels_per_group
    self.strides_per_group = strides_per_group
    self.include_top = include_top
    self.weights = weights
    self.pooling = pooling
    self.classes = 1000 if weights == "imagenet" else classes
    self.classifier_activation = classifier_activation
    self.ckpt_dir = ckpt_dir
    self.name = name
    self.preact = resnet_v2

    if weights == "imagenet":
      wb_init, bn_config = None, None

    self.wb_init = dict(wb_init or {})

    self.bn_config = dict(bn_config or {})
    self.bn_config.setdefault("decay_rate", 0.99)
    self.bn_config.setdefault("eps", 1e-5)
    self.bn_config.setdefault("create_scale", True)
    self.bn_config.setdefault("create_offset", True)

  def init_stacks(self, inputs: jnp.ndarray):
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
    if (inputs.shape[1] < self.min_size) or (inputs.shape[2] < self.min_size):
      raise ValueError("Input size must be at least " + str(self.min_size) +
                       "x" + str(self.min_size) + "; got `inputs` shape as ``" +
                       str(inputs.shape[1:3]) + "``.")

    model_weights = None
    if self.weights == "imagenet":
      file_name = self.name + "_weights_tf_dim_ordering_tf_kernels.h5"
      ckpt_file = utils.download(self.ckpt_dir, BASE_URL + file_name)
      model_weights = h5py.File(ckpt_file, "r")
    weights_init = utils.load_weights(model_weights)

    stack = stack2 if self.preact else stack1

    self.conv1 = hk.Conv2D(
        output_channels=64,
        kernel_shape=7,
        stride=2,
        padding="VALID",
        name="group01_conv1",
        **next(weights_init),
        **self.wb_init,
    )

    if self.preact is False:
      self.conv1_bn = hk.BatchNorm(
          name="group01_conv1_bn",
          **next(weights_init),
          **self.bn_config,
      )

    for i in range(4):
      self.stack_groups.append(
          stack(
              output_channels=self.channels_per_group[i],
              blocks=self.blocks_per_group[i],
              stride1=self.strides_per_group[i],
              weights_init=weights_init,
              wb_init=self.wb_init,
              bn_config=self.bn_config,
              name=f"group{i+2:02d}",
          ))

    if self.preact is True:
      self.post_bn = hk.BatchNorm(
          name="post_bn",
          **next(weights_init),
          **self.bn_config,
      )

    if self.include_top:
      self.top_layer = hk.Linear(
          output_size=self.classes,
          name="top_layer",
          **next(weights_init),
          **self.wb_init,
      )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = inputs

    if not self.stack_groups:
      self.init_stacks(inputs)

    out = jnp.pad(out, ((0, 0), (3, 3), (3, 3), (0, 0)),
                  "constant",
                  constant_values=(0, 0))
    out = self.conv1(out)
    if self.preact is False:
      out = self.conv1_bn(out, is_training)
      out = jax.nn.relu(out)
    out = jnp.pad(out, ((0, 0), (1, 1), (1, 1), (0, 0)),
                  "constant",
                  constant_values=(0, 0))
    out = hk.max_pool(out, window_shape=3, strides=2, padding="VALID")

    for stack_group in self.stack_groups:
      out = stack_group(out, is_training)

    if self.preact is True:
      out = self.post_bn(out, is_training)
      out = jax.nn.relu(out)

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


class ResNet50(ResNet):
  """Instantiates the ResNet50 architecture.
    NOTE: Information about Args can be found in main ResNet
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
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        resnet_v2=False,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet50",
        **ResNet.CONFIGS["ResNet50"],
    )


class ResNet101(ResNet):
  """Instantiates the ResNet101 architecture.
    NOTE: Information about Args can be found in main ResNet
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
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        resnet_v2=False,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet101",
        **ResNet.CONFIGS["ResNet101"],
    )


class ResNet152(ResNet):
  """Instantiates the ResNet152 architecture.
    NOTE: Information about Args can be found in main ResNet
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
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        resnet_v2=False,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet152",
        **ResNet.CONFIGS["ResNet152"],
    )


class ResNet50V2(ResNet):
  """Instantiates the ResNet50V2 architecture.
    NOTE: Information about Args can be found in main ResNet
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
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        resnet_v2=True,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet50v2",
        **ResNet.CONFIGS["ResNet50V2"],
    )


class ResNet101V2(ResNet):
  """Instantiates the ResNet101V2 architecture.
    NOTE: Information about Args can be found in main ResNet
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
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        resnet_v2=True,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet101v2",
        **ResNet.CONFIGS["ResNet101V2"],
    )


class ResNet152V2(ResNet):
  """Instantiates the ResNet152V2 architecture.
    NOTE: Information about Args can be found in main ResNet
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
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        resnet_v2=True,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet152v2",
        **ResNet.CONFIGS["ResNet152V2"],
    )


def preprocess_input(x):
  return utils.preprocess_input(x, mode="caffe")
