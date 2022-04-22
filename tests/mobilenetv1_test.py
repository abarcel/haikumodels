import haiku as hk
import jax
import jax.numpy as jnp

from PIL import Image

import haikumodels as hm

mobilenetv1_configs = {
    "include_top": True,
    "weights": "imagenet",
    "pooling": None,
    "classes": 1000,
    "classifier_activation": jax.nn.softmax,
}


def MobileNetV1_TEST(alpha=1.):

  def _forward(images, is_training):
    net = hm.MobileNetV1(alpha=alpha, **mobilenetv1_configs)
    return net(images, is_training)

  forward = hk.transform_with_state(_forward)

  img = Image.open("tests/reference_images/elephant.jpg").resize((224, 224))

  x = jnp.asarray(img, dtype=jnp.float32)
  x = jnp.expand_dims(x, axis=0)
  x = hm.mobilenetv1.preprocess_input(x)

  params, state = forward.init(None, x, is_training=True)

  haiku_out, _ = forward.apply(params, state, None, x, is_training=False)

  if alpha >= 1.0:
    alpha_text = "1_0"
  elif alpha == 0.75:
    alpha_text = "7_5"
  elif alpha == 0.50:
    alpha_text = "5_0"
  else:
    alpha_text = "2_5"

  path = 'tests/reference_outputs/mobilenetv1/mobilenetv1_%s_elephant_reference.npy' % alpha_text
  keras_ref_out = jnp.load(path)
  diff = jnp.mean(jnp.abs(keras_ref_out - haiku_out))
  assert diff < 1e-6, "Mean absolute difference is higher than: 1e-6"
