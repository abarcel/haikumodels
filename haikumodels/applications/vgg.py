from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import h5py
import haiku as hk
import jax
import jax.numpy as jnp

from .. import utils

BASE_URL = "https://storage.googleapis.com/tensorflow/keras-applications/"


class conv_block(hk.Module):

  def __init__(
      self,
      block_size: int,
      output_channels: int,
      with_bias: Optional[bool] = None,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.layers = []

    for i in range(block_size):
      self.layers.append(
          hk.Conv2D(
              output_channels=output_channels,
              kernel_shape=3,
              with_bias=with_bias,
              name=f"conv{i+1}",
              **next(weights_init),
              **wb_init,
          ))

  def __call__(self, inputs: jnp.ndarray):
    out = inputs

    for layer in self.layers:
      out = layer(out)
      out = jax.nn.relu(out)

    out = hk.max_pool(out, window_shape=2, strides=2, padding="VALID")

    return out


class top_block(hk.Module):

  def __init__(
      self,
      output_sizes: Sequence[int],
      classifier_activation: Callable[[jnp.ndarray], jnp.ndarray],
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)
    self.classifier_activation = classifier_activation

    self.layers = []

    for i, output_size in enumerate(output_sizes):
      self.layers.append(
          hk.Linear(
              output_size=output_size,
              name=f"dense{i+1}",
              **next(weights_init),
              **wb_init,
          ))

  def __call__(self, inputs: jnp.ndarray):
    out = inputs

    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < len(self.layers) - 1:
        out = jax.nn.relu(out)
      else:
        out = self.classifier_activation(out)

    return out


class VGG(hk.Module):
  """Instantiates the VGG architecture.
    See https://arxiv.org/pdf/1409.1556.pdf for details.
    Optionally loads weights pre-trained on ImageNet.
    Note that the default input image size for this model is 224x224.
    """

  CONFIGS = {
      "VGG16": {
          "convs_per_block": (2, 2, 3, 3, 3),
          "channels_per_block": (64, 128, 256, 512, 512),
          "sizes_top_block": (4096, 4096),
      },
      "VGG19": {
          "convs_per_block": (2, 2, 4, 4, 4),
          "channels_per_block": (64, 128, 256, 512, 512),
          "sizes_top_block": (4096, 4096),
      },
  }

  def __init__(
      self,
      convs_per_block: Sequence[int],
      channels_per_block: Sequence[int],
      sizes_top_block: Sequence[int],
      include_top: bool = True,
      weights: Optional[str] = None,
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      with_bias: Optional[bool] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      ckpt_dir: Optional[str] = None,
      name: Optional[str] = None,
  ):
    """Initializes a VGG function.
        Args:
          convs_per_block: Number of convolutions per block.
          channels_per_block: Output channels for convolutions in per block.
          sizes_top_block: Output size of each dense layer(Except "top" layer
            which depends on number of classes).
          include_top: Whether to include the fully-connected layer at the top
            of the network.
            By default, True.
          weights: One of None (random initialization) or ``imagenet``
            (pre-trained on ImageNet).
            By default, ``imagenet`` if using `VGG16-19`, else None.
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
          with_bias: Whether to add a bias to convolution layers.
            Optionally specified only when `weights` is None.
            By default, True.
          wb_init: Dictionary of two elements, ``w_init`` and ``b_init``
            weight initializers for both dense layer and convolution layers.
            Optionally specified only when `weights` is None.
            By default, ``w_init`` is truncated normal and ``b_init`` is zeros.
          ckpt_dir: Optional path to download pretrained weights.
            By default, temporary system file directory.
          name: Optional name name of the module.
        """
    super().__init__(name=name)
    self.blocks = []
    self.default_size = 224
    self.min_size = 32
    self.convs_per_block = convs_per_block
    self.channels_per_block = channels_per_block
    self.sizes_top_block = sizes_top_block
    self.include_top = include_top
    self.weights = weights
    self.pooling = pooling
    self.classes = 1000 if weights == "imagenet" else classes
    self.classifier_activation = classifier_activation
    self.with_bias = with_bias
    self.ckpt_dir = ckpt_dir
    self.name = name

    if weights == "imagenet":
      wb_init = None

    self.wb_init = dict(wb_init or {})

  def init_blocks(self, inputs: jnp.ndarray):
    if self.weights == "imagenet":
      if self.with_bias != True:
        print("When loading from ``imagenet`` weights, "
              "``with_bias`` must be True."
              "\tEntered value " + str(self.with_bias) +
              "is replaced with True.")
        self.with_bias = True
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
        if inputs.shape[1:] != (self.default_size, self.default_size, 3):
          raise ValueError("When setting `include_top` as True "
                           "and loading ``imagenet`` weights, "
                           "`inputs` shape should be " +
                           str((None, self.default_size, self.default_size,
                                3)) + " where None can be any natural number.")
    if (inputs.shape[1] < self.min_size) or (inputs.shape[2] < self.min_size):
      raise ValueError("Input size must be at least " + str(self.min_size) +
                       "x" + str(self.min_size) + "; got `inputs` shape as ``" +
                       str(inputs.shape[1:3]) + "``.")

    model_weights = None
    if self.weights == "imagenet":
      file_name = self.name + "_weights_tf_dim_ordering_tf_kernels.h5"
      ckpt_file = utils.download(self.ckpt_dir,
                                 BASE_URL + self.name + "/" + file_name)
      model_weights = h5py.File(ckpt_file, "r")
    weights_init = utils.load_weights(model_weights)

    for i in range(len(self.convs_per_block)):
      self.blocks.append(
          conv_block(
              block_size=self.convs_per_block[i],
              output_channels=self.channels_per_block[i],
              with_bias=self.with_bias,
              weights_init=weights_init,
              wb_init=self.wb_init,
              name=f"block{i:02d}",
          ))

    if self.include_top:
      self.dense_stack = top_block(
          output_sizes=self.sizes_top_block + (self.classes, ),
          classifier_activation=self.classifier_activation,
          weights_init=weights_init,
          wb_init=self.wb_init,
          name="top_block",
      )

  def __call__(self, inputs: jnp.ndarray):
    out = inputs

    if not self.blocks:
      self.init_blocks(inputs)

    for block in self.blocks:
      out = block(out)

    out = hk.Flatten()(out)

    if self.include_top:
      out = self.dense_stack(out)
    else:
      if self.pooling == "avg":
        out = jnp.mean(out, axis=(1, 2))
      elif self.pooling == "max":
        out = jnp.max(out, axis=(1, 2))

    return out


class VGG16(VGG):
  """Instantiates the VGG16 architecture.
    NOTE: Information about Args can be found in main VGG.
    """

  def __init__(
      self,
      include_top: bool = True,
      weights: str = "imagenet",
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      with_bias: bool = True,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      ckpt_dir: Optional[str] = None,
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        with_bias=with_bias,
        wb_init=wb_init,
        ckpt_dir=ckpt_dir,
        name="vgg16",
        **VGG.CONFIGS["VGG16"],
    )


class VGG19(VGG):
  """Instantiates the VGG19 architecture.
    NOTE: Information about Args can be found in main VGG.
    """

  def __init__(
      self,
      include_top: bool = True,
      weights: str = "imagenet",
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      with_bias: bool = True,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      ckpt_dir: Optional[str] = None,
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        with_bias=with_bias,
        wb_init=wb_init,
        ckpt_dir=ckpt_dir,
        name="vgg19",
        **VGG.CONFIGS["VGG19"],
    )


def preprocess_input(x):
  return utils.preprocess_input(x, mode="caffe")
