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
from .unet import UNetTextConditioned
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
	accum_step: int
	params: Any
	sharded_params: Any #sharded FP32 copy of model parameters
	accum_grad: Any #sharded aggregate gradient for accumulation
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

		self.model = UNetTextConditioned(**config.model.args)

	@property
	def current_num_steps(self):
		return self.config.model.train_num_steps

	def make_init_params(self, global_rng):
		init_kwargs = dict(
			x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.float32),
			context={"clip_emb": jnp.zeros((1, 77, 1024), dtype=jnp.float32), "t5_emb": jnp.zeros((1, 77, 1024), dtype=jnp.float32)},
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
			accum_step=replicate(0),
			params=init_params,
			sharded_params=sharded_params,
			accum_grad=utils.zero_pytree(ema_params),
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
		
		drop_mask = jnp.greater(jax.random.uniform(next(rng), (img.shape[0], 1, 1)), 0.9).astype(jnp.int32)
		context = {}
		for key in ["clip_emb", "t5_emb"]:
			label = batch[key]
			uncond_label = jnp.zeros_like(label)
			mask = jnp.broadcast_to(drop_mask, label.shape)
			label = label*(1-mask) + mask*uncond_label
			context[key] = label

		def model_fn(x, logsnr):
			
			return self.model.apply(
				{'params': params}, x=x, logsnr=logsnr, context=context, train=train,
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

		# Average grad across shards after casting to fp32.
		grad = utils.to_fp32(grad)
		grad, metrics['gnorm'] = utils.clip_by_global_norm(
			grad, clip_norm=self.config.train.grad_clip)
		grad = jax.lax.pmean(grad, axis_name='batch')

		# Average metrics across shards
		metrics = utils.to_fp32(metrics)
		metrics = jax.lax.pmean(metrics, axis_name='batch')
		assert all(v.shape == () for v in metrics.values())
		metrics = {  # prepend prefix to names of metrics
			f"train/{k}": v for k, v in metrics.items()
		}

		return grad, metrics
	
	def make_update_fn(self):
			
		def update_fn(state, grad):
			n_accums = self.config.train.n_accums

			sharded_grad = shard_pytree(grad)
			new_accum_grad = jax.tree_map(
				lambda x, y: (x+y)/n_accums,
				state.accum_grad, sharded_grad
			)

			#update accumulated gradient if&onlyif gradient is all finite (no inf/Nan)
			all_finite = jnp.all(
        		jnp.array([jnp.all(jnp.isfinite(p)) for p in jax.tree_util.tree_flatten(new_accum_grad)[0]])
			)
			truefn = lambda state: state.replace(accum_grad=new_accum_grad, accum_step=state.accum_step+1)
			falsefn = lambda state: state
			state = jax.lax.cond(all_finite, truefn, falsefn, state)
			
			def update(_):
				#apply adam update on sharded FP32 state, then downcast and unshard the fp32 model params.
				updates, new_opt_state = self.tx.update(
					sharded_grad, state.optimizer_state, state.sharded_params
				)
				new_sharded_params = optax.apply_updates(state.sharded_params, updates)

				new_ema_params = utils.apply_ema(self.config.train.ema_decay, 
					avg=state.ema_params, new=new_sharded_params)
				
				casted_new_params = jax.tree_util.tree_map(
					lambda x, y: x.astype(y.dtype),
					new_sharded_params, state.params
				)
				new_params = unshard_pytree(casted_new_params)
				return state.replace(
					step=state.step + 1,
					accum_step=0,
					params=new_params,
					sharded_params=new_sharded_params,
					optimizer_state=new_opt_state,
					ema_params=new_ema_params
				)
			
			def do_nothing(_):
				return state
			
			return jax.lax.cond(state.accum_step >= n_accums, update, do_nothing, operand=None)

		return update_fn
	
	"""
	def state_dict(self, state):
		#returns a saveable dict of params, ema_params and optimizer_state that's stored on the CPU and is fully unsharded.
		cpu = lambda x: utils.unreplicate(x)
		return {
			"step": state.step,
			"params": cpu(unshard_pytree(state.sharded_params)),
			"ema_params": cpu(unshard_pytree(state.ema_params)),
			"optimizer_state": cpu(unshard_pytree(state.optimizer_state))
		}

	def load_state_dict(self, state_dict):
		if state_dict is None:
			print("No restored state dict.")
			return None

		self.step = replicate(jnp.int32(state_dict["step"]))
		self.params = replicate(state_dict["params"])
		self.optimizer.ema_params = local_shard_pytree(replicate(state_dict["ema_params"]))
		self.optimizer.optimizer_state = local_shard_pytree(replicate(state_dict["optimizer_state"]))

		return TrainState(
			step=replicate(jnp.int32(state_dict["step"])),
			params=replicate(utils.to_bf16(state_dict["params"])),
			sharded_params=local_shard_pytree(replicate(state_dict["params"])),
			optimizer_state=local_shard_pytree(replicate(state_dict["optimizer_state"])),
			ema_params=local_shard_pytree(replicate(state_dict["ema_params"]))
		)
	"""
		

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