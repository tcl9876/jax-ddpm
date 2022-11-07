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
from typing import Any

from datasets import datasets
from diffusion.dpm import DiffusionWrapper
from diffusion.schedules import get_logsnr_schedule
from .unet import UNet
from . import utils
from .optimizer import make_optimizer
from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax


@flax.struct.dataclass
class TrainState:
	step: int
	params: Any
	optimizer_state: Any
	ema_params: Any
	num_sample_steps: int


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

		self.model = UNet(
			num_classes=self.dataset.num_classes,
			out_ch=out_ch,
			**config.model.args)

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

		# Make the optimizer
		self.tx = make_optimizer(self.config)
		optimizer_state = self.tx.init(init_params)

		# For ema_params below, copy so that pmap buffer donation doesn't donate the
		# same buffer twice
		return TrainState(
			step=0,
			params=init_params,
			optimizer_state=optimizer_state,
			ema_params=utils.copy_pytree(init_params),
			num_sample_steps=self.config.model.train_num_steps)

	def loss_fn(self, rng, train, batch, params):
		"""Training loss for diffusion model."""
		#TODO: remove dataset normalization/preprocessing here?
		rng = utils.RngGen(rng)

		# Input: image
		img = batch['image']
		assert img.dtype == jnp.float32
		img = utils.normalize_data(img)  # scale image to [-1, 1]

		# Input: label
		label = batch.get('label', None)
		if label is not None:
			assert label.shape == (img.shape[0],)
			assert label.dtype == jnp.int32

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

	def step_fn(self, base_rng, train, state, batch):
		"""One training/eval step."""
		config = self.config

		# RNG for this step on this host
		step = state.step
		rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
		rng = jax.random.fold_in(rng, step)
		rng = utils.RngGen(rng)

		# Loss and gradient
		loss_fn = functools.partial(self.loss_fn, next(rng), train, batch)

		if train:
			# Training mode
			(_, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

			# Average grad across shards
			grad, metrics['gnorm'] = utils.clip_by_global_norm(
				grad, clip_norm=1.0)
			grad = jax.lax.pmean(grad, axis_name='batch')

			# Update optimizer and EMA params
			updates, new_opt_state = self.tx.update(
				grad, state.optimizer_state, state.params)
			new_params = optax.apply_updates(state.params, updates)

			if hasattr(config.train, 'ema_decay'):
				ema_decay = config.train.ema_decay
			elif config.train.avg_type == 'ema':
				ema_decay = 1. - (1. / config.train.avg_steps)
			elif config.train.avg_type == 'aa':
				t = step % config.train.avg_steps
				ema_decay = t / (t + 1.)
			elif config.train.avg_type is None:
				ema_decay = 0.
			else:
				raise NotImplementedError(config.train.avg_type)
			if ema_decay == 0:
				new_ema_params = new_params
			else:
				new_ema_params = utils.apply_ema(
					decay=jnp.where(step == 0, 0.0, ema_decay),
					avg=state.ema_params,
					new=new_params)

			state = state.replace(
				step=step + 1,
				params=new_params,
				optimizer_state=new_opt_state,
				ema_params=new_ema_params)
			
		else:
		# Eval mode with EMA params
			_, metrics = loss_fn(state.ema_params)

		# Average metrics across shards
		metrics = jax.lax.pmean(metrics, axis_name='batch')
		assert all(v.shape == () for v in metrics.values())
		metrics = {  # prepend prefix to names of metrics
			f"{'train' if train else 'eval'}/{k}": v for k, v in metrics.items()
		}
		return (state, metrics) if train else metrics

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