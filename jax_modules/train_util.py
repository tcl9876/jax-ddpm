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

from datasets import datasets
from diffusion.dpm import DiffusionWrapper
from diffusion.schedules import get_logsnr_schedule
from .unet import UNet
from . import utils
from .optimizer import make_adam, shard_pytree, unshard_pytree
from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate


local_shard_pytree = jax.pmap(shard_pytree, axis_name='i', devices=jax.local_devices())
#local_unshard_pytree = jax.pmap(unshard_pytree, axis_name='i', devices=jax.local_devices())

@flax.struct.dataclass
class TrainState:
	step: int
	params: Any
	sharded_params: Any #possible sharded FP32 copy of model parameters
	optimizer_state: Any
	ema_params: Any


class Trainer:
	"""Diffusion model."""

	def __init__(self, config, dataset=None):
		self.config = config

		if dataset is not None:
			self.dataset = dataset
		else:
			self.dataset = getattr(datasets, config.dataset.name)(
				**config.dataset.args)

		self._eval_step = None

		# infer number of output channels for UNet
		x_ch = self.dataset.data_shape[-1]
		out_ch = x_ch
		if config.model.mean_type == 'both':
			out_ch += x_ch
		if 'learned' in config.model.logvar_type:
			out_ch += x_ch

		self.model = UNet(**config.model.args)

	@property
	def current_num_steps(self):
		return self.config.model.train_num_steps

	def make_init_params(self, global_rng):
		init_kwargs = dict(
			x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.float32),
			y=jnp.zeros((1,), dtype=jnp.int32),
			logsnr=jnp.zeros((1,), dtype=jnp.float32),
			train=False,
		)
		return self.model.init({'params': global_rng}, **init_kwargs)['params']

	def make_init_state(self):
		"""Make an initial TrainState."""
		# Init model params (same rng across hosts)
		init_params = self.make_init_params(
			global_rng=jax.random.PRNGKey(self.config.seed))
		logging.info('Param shapes: {}'.format(
			jax.tree_map(lambda a: a.shape, init_params)))
		logging.info('Number of trainable parameters: {:,}'.format(
			utils.count_params(init_params)))

		init_params = replicate(init_params)
		sharded_params = local_shard_pytree(utils.to_fp32(init_params))
		ema_params = utils.copy_pytree(sharded_params)
		self.tx = make_adam(self.config)
		optimizer_state = replicate(self.tx.init(unreplicate(ema_params)))
		self.update = self.make_update_fn()

		return TrainState(
			step=replicate(0),
			params=init_params,
			sharded_params=sharded_params,
			optimizer_state=optimizer_state,
			ema_params=ema_params
		)

	def loss_fn(self, rng, train, batch, params):
		"""Training loss for diffusion model."""
		#NOTE: 'label' can mean conditioning sequence. in future maybe split into T5 and clip.
		rng = utils.RngGen(rng)

		# Input: image
		img = batch['image']
		assert img.dtype == jnp.float32
		
		# Input: label
		label = batch.get('label', None)
		if label is not None:
			assert label.shape == (img.shape[0],), (label.shape, (img.shape[0],))
			assert label.dtype == jnp.int32

			#drop randomly for CFG
			uncond_label = jnp.full_like(label, self.model.num_classes)
			mask = jnp.greater(jax.random.uniform(next(rng), label.shape), 0.9).astype(jnp.int32)
			label = label*(1-mask) + mask*uncond_label

		def model_fn(x, logsnr):
			return self.model.apply(
				{'params': params}, x=x, logsnr=logsnr, y=label, train=train,
				rngs={'dropout': next(rng)} if train else None)

		target_model_fn = None

		logging.info(
			f'train_logsnr_schedule: {self.config.model.train_logsnr_schedule}')
		model = DiffusionWrapper(
			model_fn=model_fn,
			target_model_fn=target_model_fn,
			mean_type=self.config.model.mean_type,
			logvar_type=self.config.model.logvar_type,
			logvar_coeff=self.config.model.get('logvar_coeff', 0.))
		loss_dict = model.training_losses(
			x=img,
			rng=next(rng),
			logsnr_schedule_fn=get_logsnr_schedule(
				**self.config.model.train_logsnr_schedule),
			num_steps=self.current_num_steps,
			mean_loss_weight_type=self.config.model.mean_loss_weight_type)

		assert all(v.shape == (img.shape[0],) for v in loss_dict.values())
		loss_dict = {k: v.mean() for (k, v) in loss_dict.items()}
		return loss_dict['loss'], loss_dict

	def forward_backward(self, rng, batch, params):
		rng = utils.RngGen(rng)

		# Loss and gradient
		loss_fn = functools.partial(self.loss_fn, next(rng), True, batch)

		# Training mode
		(_, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)

		# Average grad across shards
		grad, metrics['gnorm'] = utils.clip_by_global_norm(
			grad, clip_norm=self.config.train.grad_clip)
		grad = jax.lax.pmean(grad, axis_name='batch')

		# Average metrics across shards
		metrics = jax.lax.pmean(metrics, axis_name='batch')
		assert all(v.shape == () for v in metrics.values())
		metrics = {  # prepend prefix to names of metrics
			f"train/{k}": v for k, v in metrics.items()
		}

		return grad, metrics
	
	def make_update_fn(self):
		def update_fn(state, grad):
			#first convert grad to fp32, then apply adam update on sharded FP32 state, then unshard and downcast the fp32 model params.
			
			sharded_grad = utils.to_fp32(shard_pytree(grad))

			updates, new_opt_state = self.tx.update(
				sharded_grad, state.optimizer_state, state.sharded_params
			)
			new_sharded_params = optax.apply_updates(state.sharded_params, updates)

			new_ema_params = utils.apply_ema(self.config.train.ema_decay, 
				avg=state.ema_params, new=new_sharded_params)

			new_params = unshard_pytree(utils.to_bf16(new_sharded_params))
			return state.replace(
				step=state.step + 1,
				params=new_params,
				sharded_params=new_sharded_params,
				optimizer_state=new_opt_state,
				ema_params=new_ema_params
			)
		
		return update_fn
		

class MeanObject(object):
    def __init__(self):
        self.reset_states()
    
    def __repr__(self):
        return repr(self._mean)
     
    def reset_states(self):
        self._mean = 0.
        self._count = 0
        
    def update(self, new_entry):
        assert isinstance(new_entry, float)# or L.shape == () #what is L? commenting out.
        self._count = self._count + 1
        self._mean = (1-1/self._count)*self._mean + new_entry/self._count
        
    def result(self):
        return self._mean
        
class Metrics(object):
	def __init__(self, metric_names):
		self.names = metric_names
		self._metric_dict = {
			name: MeanObject() for name in self.names
		}

	def to_dict(self):
		out_dict = {}
		for k, v in self._metric_dict.items():
			out_dict[k] = float(v._mean)
		return out_dict
		
	def __repr__(self):
		return repr(self._metric_dict)

	def update(self, new_metrics):
		for name in self.names:
			self._metric_dict[name].update(new_metrics[name])
		
	def reset_states(self):
		for name in self.names:
			self._metric_dict[name].reset_states()