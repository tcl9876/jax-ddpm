# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities."""

# pylint: disable=invalid-name

import functools

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as onp
import PIL
from tensorflow.io import gfile
import time

def unreplicate(x):
    return jax.device_get(flax.jax_utils.unreplicate(x))


def normalize_data(x):
	return x / 127.5 - 1.


def unnormalize_data(x):
	return (x + 1.) * 127.5

def to_bf16(params):
	def bf16(x):
		return x.astype(jnp.bfloat16)
	return jax.tree_util.tree_map(bf16, params)

def to_fp32(params):
	def fp32(x):
		return x.astype(jnp.float32)
	return jax.tree_util.tree_map(fp32, params)

def nearest_neighbor_upsample(x, k=2):
	B, H, W, C = x.shape
	x = x.reshape(B, H, 1, W, 1, C)
	x = jnp.broadcast_to(x, (B, H, k, W, k, C))
	return x.reshape(B, H * k, W * k, C)


def space_to_depth(x, k=2):
	B, H, W, C = x.shape
	assert H % k == 0 and W % k == 0
	x = x.reshape((B, H // k, k, W // k, k, C))
	x = x.transpose((0, 1, 3, 2, 4, 5))
	x = x.reshape((B, H // k, W // k, k * k * C))
	return x


def depth_to_space(x, k=2):
	B, h, w, c = x.shape
	x = x.reshape((B, h, w, k, k, c // (k * k)))
	x = x.transpose((0, 1, 3, 2, 4, 5))
	x = x.reshape((B, h * k, w * k, c // (k * k)))
	return x


def np_tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
	"""NumPy utility: tile a batch of images into a single image.

	Args:
		imgs: np.ndarray: a uint8 array of images of shape [n, h, w, c]
		pad_pixels: int: number of pixels of padding to add around each image
		pad_val: int: padding value
		num_col: int: number of columns in the tiling; defaults to a square

	Returns:
		np.ndarray: one tiled image: a uint8 array of shape [H, W, c]
	"""
	if pad_pixels < 0:
		raise ValueError('Expected pad_pixels >= 0')
	if not 0 <= pad_val <= 255:
		raise ValueError('Expected pad_val in [0, 255]')

	imgs = onp.asarray(imgs)
	if imgs.dtype != onp.uint8:
		raise ValueError('Expected uint8 input')
	# if imgs.ndim == 3:
	#   imgs = imgs[..., None]
	n, h, w, c = imgs.shape
	if c not in [1, 3]:
		raise ValueError('Expected 1 or 3 channels')

	if num_col <= 0:
		# Make a square
		ceil_sqrt_n = int(onp.ceil(onp.sqrt(float(n))))
		num_row = ceil_sqrt_n
		num_col = ceil_sqrt_n
	else:
		# Make a B/num_per_row x num_per_row grid
		assert n % num_col == 0
		num_row = int(onp.ceil(n / num_col))

	imgs = onp.pad(
			imgs,
			pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels),
						(pad_pixels, pad_pixels), (0, 0)),
			mode='constant',
			constant_values=pad_val
	)
	h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
	imgs = imgs.reshape(num_row, num_col, h, w, c)
	imgs = imgs.transpose(0, 2, 1, 3, 4)
	imgs = imgs.reshape(num_row * h, num_col * w, c)

	if pad_pixels > 0:
		imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
	if c == 1:
		imgs = imgs[Ellipsis, 0]
	return imgs


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
	#creates a grid of images and saves them to a file. also returns the created PIL image.
	imgs_grid = np_tile_imgs(
					imgs, pad_pixels=pad_pixels, pad_val=pad_val,
					num_col=num_col)

	if not (filename.startswith("gs://") or filename.startswith("gcs://")):
		image = PIL.Image.fromarray(imgs_grid)
		image.save(filename)
	else:
		#save a temporary version locally, then move to GCS remote storage, then delete the local copy.
		image = PIL.Image.fromarray(imgs_grid)
		image.save("./tmp_figure.png")
		gfile.copy("./tmp_figure.png", filename)
		time.sleep(1.5)
		gfile.remove("./tmp_figure.png")

	return image




@functools.partial(jax.pmap, axis_name='batch')
def _check_synced(pytree):
	mins = jax.lax.pmin(pytree, axis_name='batch')
	equals = jax.tree_map(jnp.array_equal, pytree, mins)
	return jnp.all(jnp.asarray(jax.tree_leaves(equals)))


def assert_synced(pytree):
	"""Check that `pytree` is the same across all replicas.

	Args:
		pytree: the pytree to check (should be replicated)

	Raises:
		RuntimeError: if sync check failed
	"""
	equals = _check_synced(pytree)
	assert equals.shape == (jax.local_device_count(),)
	equals = all(jax.device_get(equals))  # no unreplicate
	logging.info('Sync check result: %d', equals)
	if not equals:
		raise RuntimeError('Sync check failed!')


@functools.partial(jax.pmap, axis_name='batch')
def _barrier(x):
	return jax.lax.psum(x, axis_name='batch')


def barrier():
	"""MPI-like barrier."""
	jax.device_get(_barrier(jnp.ones((jax.local_device_count(),))))


def allgather_and_reshape(x, axis_name='batch'):
	"""Allgather and merge the newly inserted axis w/ the original batch axis."""
	y = jax.lax.all_gather(x, axis_name=axis_name)
	assert y.shape[1:] == x.shape
	return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])


def np_treecat(xs):
	return jax.tree_map(lambda *zs: onp.concatenate(zs, axis=0), *xs)


def dist(fn, accumulate, axis_name='batch'):
	"""Wrap a function in pmap and device_get(unreplicate(.)) its return value."""

	if accumulate == 'concat':
		accumulate_fn = functools.partial(
				allgather_and_reshape, axis_name=axis_name)
	elif accumulate == 'mean':
		accumulate_fn = functools.partial(
				jax.lax.pmean, axis_name=axis_name)
	elif accumulate == 'none':
		accumulate_fn = None
	else:
		raise NotImplementedError(accumulate)

	@functools.partial(jax.pmap, axis_name=axis_name)
	def pmapped_fn(*args, **kwargs):
		out = fn(*args, **kwargs)
		return out if accumulate_fn is None else jax.tree_map(accumulate_fn, out)

	def wrapper(*args, **kwargs):
		return jax.device_get(
				flax.jax_utils.unreplicate(pmapped_fn(*args, **kwargs)))

	return wrapper


def tf_to_numpy(tf_batch):
	"""TF to NumPy, using ._numpy() to avoid copy."""
	# pylint: disable=protected-access,g-long-lambda
	return jax.tree_map(lambda x: (x._numpy()
						if hasattr(x, '_numpy') else x), tf_batch)


def numpy_iter(tf_dataset):
	return map(tf_to_numpy, iter(tf_dataset))


def sumflat(x):
	return x.sum(axis=tuple(range(1, len(x.shape))))


def meanflat(x):
	return x.mean(axis=tuple(range(1, len(x.shape))))


def flatten(x):
	return x.reshape(x.shape[0], -1)



def count_params(pytree):
	return sum([x.size for x in jax.tree_leaves(pytree)])


def copy_pytree(pytree):
	return jax.tree_map(jnp.array, pytree)

def zero_pytree(pytree):
	return jax.tree_map(jnp.zeros_like, pytree)


def global_norm(pytree):
	return jnp.sqrt(jnp.sum(jnp.asarray(
			[jnp.sum(jnp.square(x)) for x in jax.tree_leaves(pytree)])))


def clip_by_global_norm(pytree, clip_norm, use_norm=None):
	if use_norm is None:
		use_norm = global_norm(pytree)
		assert use_norm.shape == ()  # pylint: disable=g-explicit-bool-comparison
	scale = clip_norm * jnp.minimum(1.0 / use_norm, 1.0 / clip_norm)
	return jax.tree_map(lambda x: x * scale, pytree), use_norm


def apply_ema(decay, avg, new):
	return jax.tree_map(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def scale_init(scale, init_fn, dtype=jnp.float32):
	"""Scale the output of an initializer."""

	def init(key, shape, dtype=dtype):
		return scale * init_fn(key, shape, dtype)

	return init


@functools.partial(jax.jit, static_argnums=(2,))
def _foldin_and_split(rng, foldin_data, num):
	return jax.random.split(jax.random.fold_in(rng, foldin_data), num)


class RngGen(object):
	"""Random number generator state utility for Jax."""

	def __init__(self, init_rng):
		self._base_rng = init_rng
		self._counter = 0

	def __iter__(self):
		return self

	def __next__(self):
		return self.advance(1)

	def advance(self, count):
		self._counter += count
		return jax.random.fold_in(self._base_rng, self._counter)

	def split(self, num):
		self._counter += 1
		return _foldin_and_split(self._base_rng, self._counter, num)


def jax_randint(key, minval=0, maxval=2**20):
	return int(jax.random.randint(key, shape=(), minval=minval, maxval=maxval))


def broadcast_from_left(x, shape):
	assert len(shape) >= x.ndim
	return jnp.broadcast_to(
			x.reshape(x.shape + (1,) * (len(shape) - x.ndim)),
			shape)

def list_devices(force_no_cpu=True):
	devices = jax.devices()

	if devices[0].platform == 'cpu' and force_no_cpu:
		error_msg = \
		"""Stopping process as Jax couldn't detect TPU/GPU. 
		If on a TPU, make sure to run pip install "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
		If you did that, perhaps try doing 'pkill -9 python && sudo rm -rf /tmp/tpu_logs /tmp/libtpu_lockfile' as per https://github.com/google/jax/issues/10192"""
		raise RuntimeError(error_msg)

	if jax.process_index() == 0:
		print("DEVICES: ", devices)
	
	print(f"global device count: {len(devices)}, device count on node {jax.process_index()}: {jax.local_device_count}")
