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

"""Diffusion model training and distillation."""

# pylint: disable=g-long-lambda,g-complex-comprehension,g-long-ternary
# pylint: disable=invalid-name,logging-format-interpolation


import functools
from typing import Any, Union

from diffusion.dpm import DiffusionWrapper
from .unet import UNetTextConditioned
from . import utils
from .optimizer import make_adam
from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate
from .dist_util import reshape_and_transpose, get_single_pytree_shard, move_to_last_axis, move_from_last_axis, unshard_pytree


CLIP_CHANNELS = {
	"openai/clip-vit-large-patch14": 768,
	"laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 1024
}
T5_CHANNELS = {
	"t5-large": 1024,
	"t5-3b": 1024,
	"t5-11b": 1024,
	"google/t5-v1_1-xl": 2048,
	"google/t5-v1_1-xxl": 4096
}


@flax.struct.dataclass
class TrainState:
	step: int
	params: Any
	sharded_params: Any #sharded FP32 copy of model parameters
	optimizer_state: Any
	ema_params: Any
	
class Trainer:
	"""Diffusion model."""

	def __init__(self, config, dataset=None):
		self.config = config

		if dataset is not None:
			self.dataset = dataset
		else:
			raise Exception('dataset must be provided')

		self._eval_step = None

		self.model = UNetTextConditioned(**config.model.args)
		self.devices = jax.local_devices()

	@property
	def current_num_steps(self):
		return self.config.model.train_num_steps

	def make_init_params(self, global_rng):
		context = dict(
			clip_emb=jnp.zeros((1, 77, CLIP_CHANNELS[self.config.model.clip_model_id]), dtype=jnp.float32),
			t5_emb=jnp.zeros((1, 77, T5_CHANNELS[self.config.model.t5_model_id]), dtype=jnp.float32),
			clip_image_emb=jnp.zeros((1, CLIP_CHANNELS[self.config.model.clip_model_id]), dtype=jnp.float32),
			aesth_score=jnp.zeros((1,), dtype=jnp.float32)
		)
		init_kwargs = dict(
			x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.float32),
			context=context,
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

		self.device_count = len(self.devices)
		param_shards = []
		ema_param_shards = []
		opt_shards = []
		self.tx = make_adam(self.config)

		for i in range(self.device_count):
			param_shard = get_single_pytree_shard(init_params, i, self.device_count)
			param_shard = move_to_last_axis(param_shard)
			param_shard = utils.to_fp32(param_shard)
			param_shards.append(param_shard)
		sharded_params = jax.device_put_sharded(param_shards, self.devices)

		for i in range(self.device_count):
			ema_param_shards.append(utils.copy_pytree(param_shards[i]))
		sharded_ema_params = jax.device_put_sharded(ema_param_shards, self.devices)
		del ema_param_shards

		for i in range(self.device_count):
			opt_shards.append(self.tx.init(param_shards.pop(0)))
		sharded_optimizer_states = jax.device_put_sharded(opt_shards, self.devices)
		del opt_shards
			
		init_params = replicate(init_params)
		print('Replicated params & sharded opt states have been placed on devices')
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
		
		context = {}
		"""
		p > 0.9 nothing
		p in [0.85, 0.9] clip image only
		p in [0.8, 0.85] clip image and captions
		p < 0.8 captions only
		"""
		text_and_image_p = jax.random.uniform(next(rng), (img.shape[0], 1, 1))
		if "clip_seq" in batch:
			keep_val = jnp.less(text_and_image_p, 0.85).astype(jnp.float32)
			context["clip_seq"] = batch["clip_seq"] * jnp.broadcast_to(keep_val, batch["clip_seq"].shape)

		if "t5_seq" in batch:
			keep_val = jnp.less(text_and_image_p, 0.85).astype(jnp.float32)
			context["t5_seq"] = batch["t5_seq"] * jnp.broadcast_to(keep_val, batch["t5_seq"].shape)
		
		if "clip_img" in batch:
			keep_val = jnp.logical_and(
				jnp.less(text_and_image_p, 0.9), jnp.greater(text_and_image_p, 0.8)
			).astype(jnp.float32)
			context["clip_img"] = batch["clip_img"] * jnp.broadcast_to(keep_val, batch["clip_img"].shape)
		
		#aesth_p > 0.8 include aesthetic; aesth_p is independent of text_and_image_p
		aesth_p = jax.random.uniform(next(rng), (img.shape[0],))
		if "aesth_score" in batch:
			keep_val = jnp.less(aesth_p, 0.8).astype(jnp.float32)
			context["aesth_score"] = batch["aesth_score"] * jnp.broadcast_to(keep_val, batch["aesth_score"].shape)
		
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
		rng = utils.RngGen(rng)

		# Loss and gradient
		loss_fn = functools.partial(self.loss_fn, next(rng), True, batch)
		loss, grad = jax.value_and_grad(loss_fn)(state.params)

		# Average grad across shards after casting to fp32.
		grad = utils.to_fp32(grad)
		grad, gnorm = utils.clip_by_global_norm(
			grad, clip_norm=self.config.train.grad_clip)
		grad = jax.lax.pmean(grad, axis_name='all_devices')

		avg_loss = jax.lax.pmean(loss, axis_name='all_devices')
		avg_gnorm = jax.lax.pmean(gnorm, axis_name='all_devices')
		loss_metric += avg_loss
		gnorm_metric += avg_gnorm
		        
		#TODO: grad skipping? --> is this necessary if we clip?
		grad = jax.tree_map(lambda g: reshape_and_transpose(g)[local_core_on_chip], grad)
		grad = move_to_last_axis(grad)
		return grad, loss_metric, gnorm_metric

	def update_fn(self, state, grad):
		#apply adam update on sharded FP32 state, then downcast and unshard the fp32 model params.
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

		state = state.replace(
			step=state.step + 1,
			params=new_params,
			sharded_params=new_sharded_params,
			ema_params=new_ema_params,
			optimizer_state=new_opt_state,
		)
		return state
		