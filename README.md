# Pre-trained image classification models for Jax/Haiku

Jax/Haiku Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning.

## Available Models

- MobileNetV1
- ResNet, ResNetV2
- VGG16, VGG19
- Xception

## Planned Releases

- MobileNetV2, MobileNetV3
- InceptionResNetV2, InceptionV3
- EfficientNetV1, EfficientNetV2

# Installation

Haikumodels require Python 3.7 or later.

1. Needed libraries can be installed using "installation.txt".
2. If [Jax GPU](https://github.com/google/jax#installation) support desired, must be installed seperately according to system needs.

# Usage examples for image classification models

## Classify ImageNet classes with ResNet50

```python
import haiku as hk
import jax
import jax.numpy as jnp
from PIL import Image

import haikumodels as hm

rng = jax.random.PRNGKey(42)


def _model(images, is_training):
  net = hm.ResNet50()
  return net(images, is_training)


model = hk.transform_with_state(_model)

img_path = "elephant.jpg"
img = Image.open(img_path).resize((224, 224))

x = jnp.asarray(img, dtype=jnp.float32)
x = jnp.expand_dims(x, axis=0)
x = hm.resnet.preprocess_input(x)

params, state = model.init(rng, x, is_training=True)

preds, _ = model.apply(params, state, None, x, is_training=False)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("Predicted:", hm.decode_predictions(preds, top=3)[0])
# Predicted:
# [('n02504013', 'Indian_elephant', 0.8784022),
# ('n01871265', 'tusker', 0.09620289),
# ('n02504458', 'African_elephant', 0.025362419)]
```
## Extract features with VGG16

```python
import haiku as hk
import jax
import jax.numpy as jnp
from PIL import Image

import haikumodels as hm

rng = jax.random.PRNGKey(42)

model = hk.without_apply_rng(hk.transform(hm.VGG16(include_top=False)))

img_path = "elephant.jpg"
img = Image.open(img_path).resize((224, 224))

x = jnp.asarray(img, dtype=jnp.float32)
x = jnp.expand_dims(x, axis=0)
x = hm.vgg.preprocess_input(x)

params = model.init(rng, x)

features = model.apply(params, x)
```
## Fine-tune Xception on a new set of classes

```python
from typing import Callable, Any, Sequence, Optional

import optax
import haiku as hk
import jax
import jax.numpy as jnp

import haikumodels as hm

rng = jax.random.PRNGKey(42)


class Freezable_TrainState(NamedTuple):
  trainable_params: hk.Params
  non_trainable_params: hk.Params
  state: hk.State
  opt_state: optax.OptState


# create your custom top layers and include the desired pretrained model
class ft_xception(hk.Module):

  def __init__(
      self,
      classes: int,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      with_bias: bool = True,
      w_init: Callable[[Sequence[int], Any], jnp.ndarray] = None,
      b_init: Callable[[Sequence[int], Any], jnp.ndarray] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.classifier_activation = classifier_activation

    self.xception_no_top = hm.Xception(include_top=False)
    self.dense_layer = hk.Linear(
        output_size=1024,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        name="trainable_dense_layer",
    )
    self.top_layer = hk.Linear(
        output_size=classes,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        name="trainable_top_layer",
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    out = self.xception_no_top(inputs, is_training)
    out = jnp.mean(out, axis=(1, 2))
    out = self.dense_layer(out)
    out = jax.nn.relu(out)
    out = self.top_layer(out)
    out = self.classifier_activation(out)


# use `transform_with_state` if models has batchnorm in it
# else use `transform` and then `without_apply_rng`
def _model(images, is_training):
  net = ft_xception(classes=200)
  return net(images, is_training)


model = hk.transform_with_state(_model)

# create your desired optimizer using Optax or alternatives
opt = optax.rmsprop(learning_rate=1e-4, momentum=0.90)


# this function will initialize params and state
# use the desired keyword to divide params to trainable and non_trainable
def initial_state(x_y, nonfreeze_key="trainable"):
  x, _ = x_y
  params, state = model.init(rng, x, is_training=True)

  trainable_params, non_trainable_params = hk.data_structures.partition(
      lambda m, n, p: nonfreeze_key in m, params)

  opt_state = opt.init(params)

  return Freezable_TrainState(trainable_params, non_trainable_params, state,
                              opt_state)


train_state = initial_state(next(gen_x_y))


# create your own custom loss function as desired
def loss_function(trainable_params, non_trainable_params, state, x_y):
  x, y = x_y
  params = hk.data_structures.merge(trainable_params, non_trainable_params)
  y_, state = model.apply(params, state, None, x, is_training=True)

  cce = categorical_crossentropy(y, y_)

  return cce, state


# to update params and optimizer, a train_step function must be created
@jax.jit
def train_step(train_state: Freezable_TrainState, x_y):
  trainable_params, non_trainable_params, state, opt_state = train_state
  trainable_params_grads, _ = jax.grad(loss_function,
                                       has_aux=True)(trainable_params,
                                                     non_trainable_params,
                                                     state, x_y)

  updates, new_opt_state = opt.update(trainable_params_grads, opt_state)
  new_trainable_params = optax.apply_updates(trainable_params, updates)

  train_state = Freezable_TrainState(new_trainable_params, non_trainable_params,
                                     state, new_opt_state)
  return train_state


# train the model on the new data for few epochs
train_state = train_step(train_state, next(gen_x_y))

# after training is complete it possible to merge
# trainable and non_trainable params to use for prediction
trainable_params, non_trainable_params, state, _ = train_state
params = hk.data_structures.merge(trainable_params, non_trainable_params)
preds, _ = model.apply(params, state, None, x, is_training=False)
```
