import haiku as hk
import jax
import jax.numpy as jnp

from PIL import Image

import haikumodels as hm

xception_configs = {
    "include_top": True,
    "weights": "imagenet",
    "pooling": None,
    "classes": 1000,
    "classifier_activation": jax.nn.softmax,
}


def Xception_TEST():

  def _forward(images, is_training):
    net = hm.Xception(**xception_configs)
    return net(images, is_training)

  forward = hk.transform_with_state(_forward)

  img = Image.open("tests/reference_images/elephant.jpg").resize((299, 299))

  x = jnp.asarray(img, dtype=jnp.float32)
  x = jnp.expand_dims(x, axis=0)
  x = hm.xception.preprocess_input(x)

  params, state = forward.init(None, x, is_training=True)

  haiku_out, _ = forward.apply(params, state, None, x, is_training=False)

  path = 'tests/reference_outputs/xception/xception_elephant_reference.npy'
  keras_ref_out = jnp.load(path)
  diff = jnp.mean(jnp.abs(keras_ref_out - haiku_out))
  assert diff < 1e-6, "Mean absolute difference is higher than: 1e-6"
