"""Utils for Haiku Applications."""

import json
import os
import re
import tempfile
from typing import Optional

import h5py
import jax.numpy as jnp
import requests
from tqdm import tqdm

URL = ("https://storage.googleapis.com/download.tensorflow.org/"
       "data/imagenet_class_index.json")


def download(ckpt_dir: Optional[str] = None,
             url: Optional[str] = None,
             fn_extension: str = "h5"):
  """If sub-folder path ``ckpt_dir`` is specified, creates a sub-folder
    named "haikumodels" inside specified folder, if not specified
    creates the sub-folder inside temporary folder and downloads
    pretrained weights from the given url to the sub-folder.
    """
  name = re.findall(r"\/([\w]*." + fn_extension + ")", url)[-1]
  if ckpt_dir is None:
    ckpt_dir = tempfile.gettempdir()
  ckpt_dir = os.path.join(ckpt_dir, "haikumodels")
  ckpt_file = os.path.join(ckpt_dir, name)
  if not os.path.exists(ckpt_file):
    print(
        f'Downloading from: {re.findall(r"(.*"+fn_extension+")", url)[-1]} to {ckpt_file}'
    )
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    ckpt_file_temp = os.path.join(ckpt_dir, name + ".temp")
    with open(ckpt_file_temp, "wb") as file:
      for data in response.iter_content(chunk_size=1024):
        progress_bar.update(len(data))
        file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
      print("An error occured while downloading, please try again.")
      if os.path.exists(ckpt_file_temp):
        os.remove(ckpt_file_temp)
    else:
      os.rename(ckpt_file_temp, ckpt_file)
  return ckpt_file


class _init_from_weights:

  def __init__(self, weights):
    self.weights = weights

  def __call__(self, shape, dtype):
    return jnp.asarray(self.weights, dtype=dtype).reshape(shape)


def load_attributes(group, name):
  data = [
      n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs[name]
  ]
  return data


def load_weights(f: Optional[h5py.File] = None, url_type="googleapis"):
  """Iterates and returns pretrained params from "h5" file."""
  if f:
    layer_names = load_attributes(f, "layer_names")

    if url_type == "googleapis":
      filtered_layer_names = []
      for name in layer_names:
        g = f[name]
        weight_names = load_attributes(g, "weight_names")
        if weight_names:
          filtered_layer_names.append(name)

      for layer_name in filtered_layer_names:
        g = f[layer_name]
        weight_names = load_attributes(g, "weight_names")
        layer_keys = ["w_init", "b_init"]
        bn_keys = ["scale_init", "offset_init", "mean_init", "var_init"]

        weights = {}
        for i, weight_name in enumerate(weight_names):
          if len(weight_names) == 4:
            weights[bn_keys[i]] = _init_from_weights(g[weight_name])
          else:
            weights[layer_keys[i]] = _init_from_weights(g[weight_name])

        yield weights

    else:
      for layer_name in layer_names:
        g = f[layer_name]
        weight_names = load_attributes(g, "weight_names")

        weights = {}
        for weight_name in weight_names:
          weights[weight_name] = _init_from_weights(g[weight_name])

        yield weights

  else:
    yield {}


def preprocess_input(x, mode="caffe"):
  """Preprocesses a Jax Numpy array encoding a batch of images."""
  if mode not in {"caffe", "tf", "torch"}:
    raise ValueError("Expected mode to be one of `caffe`, `tf` or `torch`. "
                     f"Received: mode={mode}")

  if not issubclass(x.dtype.type, jnp.floating):
    x = x.astype(jnp.float32, copy=False)

  if mode == "tf":
    x /= 127.5
    x -= 1.0
    return x
  elif mode == "torch":
    x /= 255.0
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
  else:
    # 'RGB'->'BGR'
    x = x[..., ::-1]
    mean = jnp.array([103.939, 116.779, 123.68])
    std = None

  # Zero-center by mean pixel
  x = jnp.subtract(x, mean)
  if std is not None:
    x = jnp.divide(x, std)
  return x


def decode_predictions(preds, top=5):
  """Decodes the prediction of an ImageNet model.
    Args:
      preds: Numpy array encoding a batch of predictions.
      top: Integer, how many top-guesses to return. Defaults to 5.
    Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
    Raises:
      ValueError: In case of invalid shape of the `pred` array
        (must be 2D).
    """

  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError("`decode_predictions` expects "
                     "a batch of predictions "
                     "(i.e. a 2D array of shape (samples, 1000)). "
                     "Found array with shape: " + str(preds.shape))

  fpath = download(None, URL, fn_extension="json")
  with open(fpath) as f:
    CLASS_INDEX = json.load(f)
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [
        tuple(CLASS_INDEX[str(i)]) + (float(pred[i]), ) for i in top_indices
    ]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results
