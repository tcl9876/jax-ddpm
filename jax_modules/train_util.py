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

#THIS CODE FILE WAS MODIFIED

"""Diffusion model training and distillation."""

# pylint: disable=g-long-lambda,g-complex-comprehension,g-long-ternary
# pylint: disable=invalid-name,logging-format-interpolation


import functools
from typing import Any, Union

from diffusion.dpm import DiffusionWrapper
from .unet import UNetTextConditioned
from . import utils
from .optimizer import make_adam, unshard_pytree
from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate

#are all the nones needed? todo for later
@jax.jit
def replace_params(state, new_params):
	state = state.replace(params=None)
	return state.replace(
		step=state.step + 1,
		params=new_params
	)

@jax.jit
def replace_sharded_params(state, new_sharded_params):
	state = state.replace(sharded_params=None)
	return state.replace(sharded_params=new_sharded_params)

@jax.jit
def replace_ema_params(state, new_ema_params):
	state = state.replace(ema_params=None)
	return state.replace(ema_params=new_ema_params)

@jax.jit
def replace_opt(state, new_opt_state):
	state = state.replace(optimizer_state=None)
	return state.replace(optimizer_state=new_opt_state)

#TPU's pad if ch<128, this becomes a problem if optimizer states have ch of 512 or 768
#since its sharded version (that divides by 8) will result in 64 or 96 channels
#to avoid padding, we move from the second last to last axis so its divisible by 128 
def move_to_last_axis(pytree, mul=2):
	def movefn(x):
		xshape = x.shape
		if len(xshape) >= 2 and xshape[-1] * x.shape[-2] > 16384:
			newshape = list(xshape[:-2]) + [x.shape[-2] //mul, x.shape[-1]*mul]
			return jnp.reshape(x, newshape)
		else:
			return x
	return jax.tree_map(lambda a: movefn(a), pytree)

def move_from_last_axis(pytree, mul=2):
	def movefn(x):
		xshape = x.shape
		if len(xshape) >= 2 and xshape[-1] * x.shape[-2] > 16384:
			newshape = list(xshape[:-2]) + [x.shape[-2] *mul, x.shape[-1] //mul]
			return jnp.reshape(x, newshape)
		else:
			return x
	return jax.tree_map(lambda a: movefn(a), pytree)



@flax.struct.dataclass
class TrainState:
	step: int #current iteration
	params: Any #replicated BF16 copy of model parameters
	sharded_params: Any #sharded FP32 copy of model parameters
	optimizer_state: Any #sharded FP32 copy of adam states
	ema_params: Any #sharded FP32 copy of EMA parameters
	

def get_single_pytree_shard(pytree, i, device_count=8):
	def slicefn(x, i):
		dim = x.shape[-1] // device_count
		return x[..., dim*i:dim*i+dim]
	return jax.tree_map(lambda a: slicefn(a, i), pytree)

#this is working as intended, right? --> maybe test it
def reshape_and_transpose(x):
	shapelen = len(x.shape)
	newshape = list(x.shape[:-1]) + [8, x.shape[-1] // 8]
	x = jnp.reshape(x, newshape)
	
	#hwi8j -> 8hwij,  01234 -> 30124  shapelen 4
	#8j -> 8j  01 -> 01 shapelen 1
	#i8j -> 8ij  012 -> 102 shapelen 2
	newperm = [shapelen-1] + list(range(shapelen-1)) + [shapelen]
	x = jnp.transpose(x, newperm)
	return x         

class Trainer:
	"""Diffusion model."""

	def __init__(self, config, dataset=None):
		self.config = config

		if dataset is not None:
			self.dataset = dataset
		else:
			raise Exception('dataset must be provided')

		self.model = UNetTextConditioned(**config.model.args)
		self.local_devices = jax.local_devices()

	@property
	def current_num_steps(self):
		return self.config.model.train_num_steps

	def make_init_params(self, global_rng):
		init_kwargs = dict(
			x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.float32),
			context={"clip_emb": jnp.zeros((1, 77, 1024), dtype=jnp.float32), "t5_emb": jnp.zeros((1, 77, 1024), dtype=jnp.float32)},
			alpha=jnp.ones((1,), dtype=jnp.float32)*0.5,
			train=False,
		)
		return self.model.init({'params': global_rng}, **init_kwargs)['params']

	def make_init_state(self):
		"""Make an initial TrainState."""
		# Init model params (same rng across hosts)
		init_params = self.make_init_params(
			global_rng=jax.random.PRNGKey(self.config.seed))
		logging.info('Number of trainable parameters: {:,}'.format(
			utils.count_params(init_params)))

		self.local_device_count = len(self.local_devices)
		param_shards = []
		ema_param_shards = []
		opt_shards = []
		self.tx = make_adam(self.config)

		for i in range(self.local_device_count):
			print('p', i)
			param_shard = get_single_pytree_shard(init_params, i, self.local_device_count)
			param_shard = move_to_last_axis(param_shard)
			param_shard = utils.to_fp32(param_shard)
			param_shards.append(param_shard)
		sharded_params = jax.device_put_sharded(param_shards, self.local_devices)

		for i in range(self.local_device_count):
			print('ema p', i)
			ema_param_shards.append(utils.copy_pytree(param_shards[i]))
		sharded_ema_params = jax.device_put_sharded(ema_param_shards, self.local_devices)
		del ema_param_shards

		for i in range(self.local_device_count):
			print('o', i, 'len arr', len(param_shards))
			opt_shards.append(self.tx.init(param_shards.pop(0)))
		sharded_optimizer_states = jax.device_put_sharded(opt_shards, self.local_devices)
		del opt_shards
			
		init_params = replicate(init_params)
		#self.update = self.make_update_fn()
		return TrainState(
			step=replicate(0),
			params=init_params,
			sharded_params=sharded_params,
			optimizer_state=sharded_optimizer_states,
			ema_params=sharded_ema_params
		)

	def loss_fn(self, rng, train, batch, params):
		"""Training loss for diffusion model."""
		rng = utils.RngGen(rng)

		# Input: image
		img = batch['image']
		assert img.dtype == jnp.float32
		
		#drop the text sequence by zero-ing it out. jointly drops clip and t5 embeddings.
		drop_mask = jnp.greater(jax.random.uniform(next(rng), (img.shape[0], 1, 1)), 0.9).astype(jnp.int32)
		context = {}
		for key in ["clip_emb", "t5_emb"]:
			label = batch[key]
			uncond_label = jnp.zeros_like(label)
			mask = jnp.broadcast_to(drop_mask, label.shape)
			label = label*(1-mask) + mask*uncond_label
			context[key] = label

		def model_fn(x, alpha):
			
			return self.model.apply(
				{'params': params}, x=x, alpha=alpha, context=context, train=train,
				rngs={'dropout': next(rng)} if train else None)


		logging.info(
			f'train_alpha_schedule: {self.config.model.train_alpha_schedule}')
		model = DiffusionWrapper(
			model_fn=model_fn,
			mean_type=self.config.model.mean_type,
			logvar_type=self.config.model.logvar_type,
			logvar_coeff=self.config.model.get('logvar_coeff', 0.),
			alpha_schedule=self.config.model.train_alpha_schedule,
			tmin=self.config.model.tmin)
		
		loss = model.training_losses(
			x=img,
			rng=next(rng),
			num_steps=self.current_num_steps,
			mean_loss_weight_type=self.config.model.mean_loss_weight_type)

		return loss

	def forward_backward(self, rng, batch, state, local_core_on_chip, loss_metric, gnorm_metric):
		"""
		performs the forward and backward pass, but does not update the optimizer state, returning the gradients (and metrics) only.
		the gradients are casted to FP32 before being averaged across *all* hosts via jax.lax.pmean.
		"""
		rng = utils.RngGen(rng)

		# Loss and gradient
		loss_fn = functools.partial(self.loss_fn, next(rng), True, batch)
		loss, grad = jax.value_and_grad(loss_fn)(state.params) #state.params is a replicated copy of mostly BF16 model parameters.

		# Average grad across shards after casting to fp32.
		grad = utils.to_fp32(grad)
		grad, gnorm = utils.clip_by_global_norm(
			grad, clip_norm=self.config.train.grad_clip)
		grad = jax.lax.pmean(grad, axis_name='batch')


		avg_loss = jax.lax.pmean(loss, axis_name='batch')
		avg_gnorm = jax.lax.pmean(gnorm, axis_name='batch')
		loss_metric += avg_loss
		gnorm_metric += avg_gnorm
		        
		#TODO: grad skipping?
		grad = jax.tree_map(lambda g: reshape_and_transpose(g)[local_core_on_chip], grad)
		grad = move_to_last_axis(grad)
		return grad, loss_metric, gnorm_metric

	def update_fn(self, state, grad):
		"""
		applies the adam update on a sharded FP32 state, including updating the EMA.
		it then downcasts and unshards this sharded copy of the parameters.
		"""
		flat = lambda tree: jax.tree_util.tree_flatten(tree)[0]
		g, o, s = flat(grad), flat(state.optimizer_state), flat(state.sharded_params)
		print("LENGTHS", len(g), len(o), len(s))

		updates, new_opt_state = self.tx.update(
			grad, state.optimizer_state, state.sharded_params
		)
		new_sharded_params = optax.apply_updates(state.sharded_params, updates)

		new_ema_params = utils.apply_ema(self.config.train.ema_decay, 
				avg=state.ema_params, new=new_sharded_params)
		
		casted_new_params = jax.tree_util.tree_map(
			lambda x, y: x.astype(y.dtype),
			new_sharded_params, state.params
		)
		casted_new_params = move_from_last_axis(casted_new_params)
		new_params = unshard_pytree(casted_new_params)

		state = replace_params(state, new_params)
		state = replace_sharded_params(state, new_sharded_params)
		state = replace_ema_params(state, new_ema_params)
		state = replace_opt(state, new_opt_state)
		return state
