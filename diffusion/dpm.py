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

"""Simple diffusion implementation."""

# pylint:disable=missing-class-docstring,missing-function-docstring
# pylint:disable=logging-format-interpolation
# pylint:disable=g-long-lambda

from jax_modules import utils
from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

### Basic diffusion process utilities

# NOTE: the variable "alpha" is the same as alpha_bar in the original DDPM paper
# this is different from the diffusion-distillation repo, which uses "alpha" to refer to sqrt(alpha_bar)


def get_alpha_set(name, beta_start, beta_end, steps, b1=None):
	if name=='linear':
		beta_schedule = onp.linspace(beta_start, beta_end, steps)		
		alpha_schedule = onp.cumprod(1 - beta_schedule)
		return jnp.array(alpha_schedule.astype('float32'))
	else:
		assert 0, "only linear support right now"


def diffusion_forward(*, x, alpha):
	return {
		'mean': x * jnp.sqrt(alpha),
		'std': jnp.sqrt(1 - alpha),
		'var': (1 - alpha),
		'logvar': jnp.log(1 - alpha)
	}

def predict_x_from_eps(*, xt, eps, alpha):
	"""x = (xt - sqrt(1-alpha)*eps) / sqrt(alpha)."""
	alpha = utils.broadcast_from_left(alpha, xt.shape)
	sigma = jnp.sqrt(1 - alpha)
	return (xt - sigma * eps) / jnp.sqrt(alpha)


def predict_eps_from_x(*, xt, x, alpha):
	"""x = (xt - sqrt(1-alpha)*eps) / sqrt(alpha)."""
	"""eps = (xt - sqrt(alpha)*x) / sqrt(1-alpha)"""
	alpha = utils.broadcast_from_left(alpha, xt.shape)
	sigma = jnp.sqrt(1 - alpha)
	return (xt - jnp.sqrt(alpha) * x) / sigma

def predict_x_from_v(*, xt, v, alpha):
	alpha = utils.broadcast_from_left(alpha, xt.shape)
	sigma = jnp.sqrt(1 - alpha)
	return jnp.sqrt(alpha) * xt - sigma * v


class DiffusionWrapper:

	def __init__(self, model_fn, *, mean_type, logvar_type, logvar_coeff, alpha_schedule, tmin):
		self.model_fn = model_fn
		self.mean_type = mean_type
		self.logvar_type = logvar_type
		self.logvar_coeff = logvar_coeff
		self.alpha_set = get_alpha_set(**alpha_schedule)
		self.tmin = tmin

	def _run_model(self, *, xt, alpha, model_fn):
		model_output = model_fn(xt, alpha)
		if self.mean_type == 'eps':
			model_eps = model_output
		elif self.mean_type == 'x':
			model_x = model_output
		elif self.mean_type == 'v':
			model_v = model_output
		else:
			raise NotImplementedError(self.mean_type)

		# get prediction of x at t=0
		if self.mean_type == 'eps':
			model_x = predict_x_from_eps(xt=xt, eps=model_eps, alpha=alpha)
		elif self.mean_type == 'v':
			model_x = predict_x_from_v(xt=xt, v=model_v, alpha=alpha)
		return model_x


	def training_losses(self, *, x, rng, num_steps, mean_loss_weight_type):
		assert x.dtype == jnp.float32
		assert isinstance(num_steps, int)
		rng = utils.RngGen(rng)
		eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
		bc = lambda z: utils.broadcast_from_left(z, x.shape)


		t = jax.random.randint(
				next(rng), shape=(x.shape[0],), minval=self.tmin, maxval=num_steps)
		#alpha = self.alpha_set[t]
		alpha = jnp.take(self.alpha_set, t)
		print('gather shapes', alpha.shape, t.shape, bc(alpha).shape)

		xt_dist = diffusion_forward(x=x, alpha=bc(alpha))
		xt = xt_dist['mean'] + xt_dist['std'] * eps

		#broadcast here?
		x_target = x
		model_output = self._run_model(
				xt=xt, alpha=alpha, model_fn=self.model_fn)
		x_mse = utils.meanflat(jnp.square(model_output - x_target))
		snr = alpha / (1 - alpha)

		if mean_loss_weight_type == 'p2':
			reweighting_factor = snr * (1 - alpha)
		elif mean_loss_weight_type == 'p2_half':
			reweighting_factor = snr * jnp.sqrt(1 - alpha)
		elif mean_loss_weight_type == 'constant':  
			reweighting_factor = 1.
		elif mean_loss_weight_type == 'snr': 
			reweighting_factor = snr
		elif mean_loss_weight_type == 'snr_trunc':  
			reweighting_factor = jnp.maximum(snr, 1.)
		elif mean_loss_weight_type == 'v_mse':
			reweighting_factor = snr + 1.
		elif mean_loss_weight_type == 'root_snr':
			reweighting_factor = jnp.sqrt(snr)
		else:
			raise NotImplementedError(mean_loss_weight_type)
		
		loss = x_mse * reweighting_factor
		loss = jnp.mean(loss)
		return loss
