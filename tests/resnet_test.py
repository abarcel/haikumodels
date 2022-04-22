import haiku as hk
import jax
import jax.numpy as jnp

from PIL import Image

import haikumodels as hm

resnet_configs = {
    "include_top": True,
    "weights": "imagenet",
    "pooling": None,
    "classes": 1000,
    "classifier_activation": jax.nn.softmax,
}


def ResNet_TEST(name):

  def _forward(images, is_training):
    net = getattr(hm, name)(**resnet_configs)
    return net(images, is_training)

  forward = hk.transform_with_state(_forward)

  img = Image.open("tests/reference_images/elephant.jpg").resize((224, 224))

  x = jnp.asarray(img, dtype=jnp.float32)
  x = jnp.expand_dims(x, axis=0)
  x = hm.resnet.preprocess_input(x)

  params, state = forward.init(None, x, is_training=True)

  haiku_out, _ = forward.apply(params, state, None, x, is_training=False)

  path = ('tests/reference_outputs/resnet/%s_elephant_reference.npy' %
          name.lower())
  keras_ref_out = jnp.load(path)
  diff = jnp.mean(jnp.abs(keras_ref_out - haiku_out))
  assert diff < 1e-6, "Mean absolute difference is higher than: 1e-6"


def ResNet50_TEST():
  return ResNet_TEST("ResNet50")


def ResNet101_TEST():
  return ResNet_TEST("ResNet101")


def ResNet152_TEST():
  return ResNet_TEST("ResNet152")


def ResNet50V2_TEST():
  return ResNet_TEST("ResNet50V2")


def ResNet101V2_TEST():
  return ResNet_TEST("ResNet101V2")


def ResNet152V2_TEST():
  return ResNet_TEST("ResNet152V2")