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


def diffusion_forward(*, x, logsnr):
	"""q(z_t | x)."""
	return {
			'mean': x * jnp.sqrt(nn.sigmoid(logsnr)),
			'std': jnp.sqrt(nn.sigmoid(-logsnr)),
			'var': nn.sigmoid(-logsnr),
			'logvar': nn.log_sigmoid(-logsnr)
	}


def predict_x_from_eps(*, z, eps, logsnr):
	"""x = (z - sigma*eps)/alpha."""
	logsnr = utils.broadcast_from_left(logsnr, z.shape)
	return jnp.sqrt(1. + jnp.exp(-logsnr)) * (
			z - eps * jax.lax.rsqrt(1. + jnp.exp(logsnr)))


def predict_xlogvar_from_epslogvar(*, eps_logvar, logsnr):
	"""Scale Var[eps] by (1+exp(-logsnr)) / (1+exp(logsnr)) = exp(-logsnr)."""
	return eps_logvar - logsnr


def predict_eps_from_x(*, z, x, logsnr):
	"""eps = (z - alpha*x)/sigma."""
	logsnr = utils.broadcast_from_left(logsnr, z.shape)
	return jnp.sqrt(1. + jnp.exp(logsnr)) * (
			z - x * jax.lax.rsqrt(1. + jnp.exp(-logsnr)))


def predict_epslogvar_from_xlogvar(*, x_logvar, logsnr):
	"""Scale Var[x] by (1+exp(logsnr)) / (1+exp(-logsnr)) = exp(logsnr)."""
	return x_logvar + logsnr


def predict_x_from_v(*, z, v, logsnr):
	logsnr = utils.broadcast_from_left(logsnr, z.shape)
	alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
	sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
	return alpha_t * z - sigma_t * v


def predict_v_from_x_and_eps(*, x, eps, logsnr):
	logsnr = utils.broadcast_from_left(logsnr, x.shape)
	alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
	sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
	return alpha_t * eps - sigma_t * x


class DiffusionWrapper:

	def __init__(self, model_fn, *, mean_type, logvar_type, logvar_coeff,
							 target_model_fn=None, loss_scale=1.0):
		self.model_fn = model_fn
		self.mean_type = mean_type
		self.logvar_type = logvar_type
		self.logvar_coeff = logvar_coeff
		self.target_model_fn = target_model_fn
		self.loss_scale = loss_scale

	def _run_model(self, *, z, logsnr, model_fn, clip_x):
		model_output = model_fn(z, logsnr)
		if self.mean_type == 'eps':
			model_eps = model_output
		elif self.mean_type == 'x':
			model_x = model_output
		elif self.mean_type == 'v':
			model_v = model_output
		elif self.mean_type == 'both':
			_model_x, _model_eps = jnp.split(model_output, 2, axis=-1)  # pylint: disable=invalid-name
		else:
			raise NotImplementedError(self.mean_type)

		# get prediction of x at t=0
		if self.mean_type == 'both':
			# reconcile the two predictions
			model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
			wx = utils.broadcast_from_left(nn.sigmoid(-logsnr), z.shape)
			model_x = wx * _model_x + (1. - wx) * model_x_eps
		elif self.mean_type == 'eps':
			model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
		elif self.mean_type == 'v':
			model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)

		# clipping
		if clip_x:
			model_x = jnp.clip(model_x, -1., 1.)

		# get eps prediction if clipping or if mean_type != eps
		if self.mean_type != 'eps' or clip_x:
			model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)

		# get v prediction if clipping or if mean_type != v
		if self.mean_type != 'v' or clip_x:
			model_v = predict_v_from_x_and_eps(
					x=model_x, eps=model_eps, logsnr=logsnr)

		return {'model_x': model_x,
						'model_eps': model_eps,
						'model_v': model_v}

	def training_losses(self, *, x, rng, logsnr_schedule_fn,
											num_steps, mean_loss_weight_type):
		assert x.dtype in [jnp.float32, jnp.float64]
		assert isinstance(num_steps, int)
		rng = utils.RngGen(rng)
		eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
		bc = lambda z: utils.broadcast_from_left(z, x.shape)

		# sample logsnr
		if num_steps > 0:
			logging.info('Discrete time training: num_steps=%d', num_steps)
			assert num_steps >= 1
			t = jax.random.randint(
					next(rng), shape=(x.shape[0],), minval=0, maxval=num_steps)
			u = (t+1).astype(x.dtype) / num_steps
		else:
			logging.info('Continuous time training')
			# continuous time
			u = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
		logsnr = logsnr_schedule_fn(u)
		assert logsnr.shape == (x.shape[0],)

		# sample z ~ q(z_logsnr | x)
		z_dist = diffusion_forward(x=x, logsnr=bc(logsnr))
		z = z_dist['mean'] + z_dist['std'] * eps

		x_target = x
		eps_target = eps
		v_target = predict_v_from_x_and_eps(
				x=x_target, eps=eps_target, logsnr=logsnr)

		# denoising loss
		model_output = self._run_model(
				z=z, logsnr=logsnr, model_fn=self.model_fn, clip_x=False)

		x_mse = utils.meanflat(jnp.square(model_output['model_x'] - x_target))
		eps_mse = utils.meanflat(jnp.square(model_output['model_eps'] - eps_target))
		v_mse = utils.meanflat(jnp.square(model_output['model_v'] - v_target))
		
		if mean_loss_weight_type == 'p2':
			#p2 weighting with gamma = 1
			assert logsnr.shape == eps_mse.shape
			reweighting_factor = 1 / (1 + jnp.exp(logsnr))
			loss = eps_mse * reweighting_factor
		elif reweighting_factor == 'p2_half':
			#p2 with gamma = 0.5. This might be better as its in the latent space where imperceptible information has been removed already.
			reweighting_factor = 1 / jnp.sqrt(1 + jnp.exp(logsnr))
			loss = eps_mse * reweighting_factor
		elif mean_loss_weight_type == 'constant':  # constant weight on x_mse
			loss = x_mse
		elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
			loss = eps_mse
		elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
			loss = jnp.maximum(x_mse, eps_mse)
		elif mean_loss_weight_type == 'v_mse':
			loss = v_mse
		else:
			raise NotImplementedError(mean_loss_weight_type)
		
		loss = loss * self.loss_scale
		return {'loss': loss}
