import haiku as hk
import jax
import jax.numpy as jnp

from PIL import Image

import haikumodels as hm

vgg_configs = {
    "include_top": True,
    "weights": "imagenet",
    "pooling": None,
    "classes": 1000,
    "classifier_activation": jax.nn.softmax,
}


def VGG_TEST(name):

  def _forward(images):
    net = getattr(hm, name)(**vgg_configs)
    return net(images)

  forward = hk.transform(_forward)

  img = Image.open("tests/reference_images/elephant.jpg").resize((224, 224))

  x = jnp.asarray(img, dtype=jnp.float32)
  x = jnp.expand_dims(x, axis=0)
  x = hm.vgg.preprocess_input(x)

  params = forward.init(jax.random.PRNGKey(42), x)

  haiku_out = forward.apply(params, None, x)

  path = 'tests/reference_outputs/vgg/%s_elephant_reference.npy' % name.lower()
  keras_ref_out = jnp.load(path)
  diff = jnp.mean(jnp.abs(keras_ref_out - haiku_out))
  assert diff < 1e-6, "Mean absolute difference is higher than: 1e-6"


def VGG16_TEST():
  return VGG_TEST("VGG16")


def VGG19_TEST():
  return VGG_TEST("VGG19")