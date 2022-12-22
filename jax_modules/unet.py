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

"""Linen version of unet similar to that from Improved DDPM."""

# pytype: disable=wrong-keyword-args,wrong-arg-count
# pylint: disable=logging-format-interpolation,g-long-lambda

from typing import Tuple, Optional, Any

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from .attention import FlaxTransformer2DModel

def nonlinearity(x):
	return x * nn.sigmoid(1.702 * x)

class Normalize(nn.Module):
	name: str

	@nn.compact
	def __call__(self, x):
		x_dtype = x.dtype
		normx_32 = nn.normalization.GroupNorm()(x.astype('float32'))
		return normx_32.astype(x_dtype)


def get_timestep_embedding(timesteps, embedding_dim,
							max_time=1000., dtype=jnp.float32):
	"""Build sinusoidal embeddings (from Fairseq).

	This matches the implementation in tensor2tensor, but differs slightly
	from the description in Section 3.5 of "Attention Is All You Need".

	Args:
		timesteps: jnp.ndarray: generate embedding vectors at these timesteps
		embedding_dim: int: dimension of the embeddings to generate
		max_time: float: largest time input
		dtype: data type of the generated embeddings

	Returns:
		embedding vectors with shape `(len(timesteps), embedding_dim)`
	"""
	assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
	timesteps *= (1000. / max_time)

	half_dim = embedding_dim // 2
	emb = np.log(10000) / (half_dim - 1)
	emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
	emb = timesteps.astype(dtype)[:, None] * emb[None, :]
	emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
	if embedding_dim % 2 == 1:  # zero pad
		emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
	assert emb.shape == (timesteps.shape[0], embedding_dim)
	return emb

"""
2d positional embedding like ViT. adapted from https://github.com/ericl122333/PatchDiffusion-Pytorch/blob/main/patch_diffusion/nn.py
a model that's given absolute positions might be better at spatial relationships, eg for prompts like 'a red cube *on top of* a blue cube'
positional encoding is injected right after the input conv. 
"""
def timestep_embedding_2d(dim, resolution):
    omega = 64 / resolution   #higher resolutions need longer wavelengths
    half_dim = dim // 2
    arange = jnp.arange(resolution, dtype=jnp.float32)

    emb = (np.log(10000) / (half_dim - 1))
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    
    emb = arange[:, None] * emb[None, :]
    emb = jnp.sin(emb * omega)
    
    emb_x = jnp.repeat(emb[None, ...], resolution, axis=0)
    emb_y = jnp.repeat(emb[:, None, :], resolution, axis=1)
    emb = jnp.concatenate([emb_x, emb_y], axis=-1)
    return emb[None, ...]

def nearest_neighbor_upsample(x):
	B, H, W, C = x.shape  # pylint: disable=invalid-name
	x = x.reshape(B, H, 1, W, 1, C)
	x = jnp.broadcast_to(x, (B, H, 2, W, 2, C))
	return x.reshape(B, H * 2, W * 2, C)


class ResnetBlock(nn.Module):
	"""Convolutional residual block."""

	dropout: float
	out_ch: Optional[int] = None
	resample: Optional[str] = None
	param_dtype: Optional[Any] = jnp.float32

	@nn.compact
	def __call__(self, x, *, emb, deterministic):
		B, _, _, C = x.shape  # pylint: disable=invalid-name
		assert emb.shape[0] == B and len(emb.shape) == 2
		out_ch = C if self.out_ch is None else self.out_ch

		h = nonlinearity(Normalize(name='norm1')(x))
		if self.resample is not None:
			updown = lambda z: {
					'up': nearest_neighbor_upsample(z),
					'down': nn.avg_pool(z, (2, 2), (2, 2))
			}[self.resample]
			h = updown(h)
			x = updown(x)
		h = nn.Conv(
				features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1', param_dtype=self.param_dtype)(h)

		# add in timestep/class embedding
		emb_out = nn.Dense(features=2 * out_ch, name='temb_proj', dtype=self.param_dtype, param_dtype=self.param_dtype)(
				nonlinearity(emb))[:, None, None, :]
		scale, shift = jnp.split(emb_out, 2, axis=-1)
		h = Normalize(name='norm2')(h) * (1 + scale) + shift
		# rest
		h = nonlinearity(h)
		h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
		h = nn.Conv(
				features=out_ch,
				kernel_size=(3, 3),
				strides=(1, 1),
				kernel_init=nn.initializers.zeros,
				name='conv2', 
				param_dtype=self.param_dtype)(h)

		if C != out_ch:
			x = nn.Dense(features=out_ch, name='nin_shortcut', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)

		assert x.shape == h.shape
		return x + h


class SequenceProcessor(nn.Module):
    seq_width: Optional[int] = 1024
    t5_mult: float = 4.0
    @nn.compact
    def __call__(self, context):
        """
        does several things:
        1) applies linear transformation separately to clip_seq and t5_seq. t5_seq has less variance so we multiply it by a constant
        2) prepends a null embedding to the sequence, similar to https://arxiv.org/pdf/2211.01324.pdf. 
        3) concatenates the sequences.
        """
        clip_seq, t5_seq = context["clip_seq"], context["t5_seq"]

        b, hc, ht5 = clip_seq.shape[0], clip_seq.shape[-1], t5_seq.shape[-1]
        if self.seq_width is None: 
            seq_width = hc
        else:
            seq_width = self.seq_width

        assert hc%seq_width == 0 and ht5%seq_width == 0
        clip_seq = nn.Dense(seq_width)(clip_seq)
        t5_seq = nn.Dense(seq_width)(t5_seq * self.t5_mult)
        
        null_emb = nn.Embed(1, seq_width)(jnp.zeros([b, 1], dtype=jnp.int32))
        return jnp.concatenate([null_emb, clip_seq, t5_seq], axis=1)


class UNetTextConditioned(nn.Module):
	"""The UNet architecture w/ t5 and clip encoding handling."""

	ch: int
	emb_ch: int
	out_ch: int
	ch_mult: Tuple[int]
	num_res_blocks: int
	attn_resolutions: Tuple[int]
	num_heads: Optional[int]
	dropout: float
	
	logsnr_scale_range: Tuple[float, float] = (-10., 10.)

	resblock_resample: bool = False
	head_dim: Optional[int] = None  # alternative to num_heads
	param_dtype: Any = 'fp32'
	use_glu: bool = False
	t5_mult: float = 4.0

	@nn.compact
	def __call__(self, x, alpha, context, *, train):
		text_processor = SequenceProcessor(seq_width=self.seq_width, t5_mult=self.t5_mult)
		context = text_processor(context)
		
		B, H, W, _ = x.shape  # pylint: disable=invalid-name
		assert H == W
		assert x.dtype == jnp.float32
		assert alpha.shape == (B,) and alpha.dtype == jnp.float32
		num_resolutions = len(self.ch_mult)
		ch = self.ch
		emb_ch = self.emb_ch


		logsnr = jnp.log(alpha / (1 - alpha))
		logsnr_input = (logsnr - self.logsnr_scale_range[0]) / (
			self.logsnr_scale_range[1] - self.logsnr_scale_range[0])


		if self.param_dtype == 'fp32':
			param_dtype = jnp.float32
		else:
			param_dtype = jnp.bfloat16
		
		print("PARAM DTYPE:", param_dtype)

		emb = get_timestep_embedding(logsnr_input, embedding_dim=emb_ch, max_time=1.)
		emb = nn.Dense(features=emb_ch, name='dense0')(emb)
		emb = nn.Dense(features=emb_ch, name='dense1')(nonlinearity(emb))
		assert emb.shape == (B, emb_ch)

		emb = emb.astype(param_dtype)
		context = context.astype(param_dtype)
		
		hs = [nn.Conv(
				features=ch, kernel_size=(3, 3), strides=(1, 1), name='conv_in')(x)]
		pe = timestep_embedding_2d(ch, x.shape[1])
		assert pe.shape[1:] == hs[0].shape[1:]
		hs= [hs[0].astype(param_dtype) + pe.astype(param_dtype)]

		def maybe_remat(block, i_level):
			#dont remat the bottom block because it doesn't use much memory.
			if i_level <= 2:
				return nn.remat(block)
			else:
				return block

		# Downsampling
		for i_level in range(num_resolutions):
			# Residual blocks for this resolution
			for i_block in range(self.num_res_blocks):
				h = maybe_remat(ResnetBlock, i_level)(
						out_ch=ch * self.ch_mult[i_level],
						dropout=self.dropout,
						name=f'down_{i_level}.block_{i_block}',
						param_dtype=param_dtype)(
								hs[-1], emb=emb, deterministic=not train)
				if h.shape[1] in self.attn_resolutions:
					h = maybe_remat(FlaxTransformer2DModel, i_level)(
							in_channels=h.shape[-1], 
							n_heads=h.shape[-1]//self.head_dim, 
							d_head=self.head_dim, 
							only_cross_attention=(i_level == 0),
							param_dtype=param_dtype)(h, context)
				hs.append(h)
			# Downsample
			if i_level != num_resolutions - 1:
				hs.append(self._downsample(
						hs[-1], name=f'down_{i_level}.downsample', emb=emb, train=train, param_dtype=param_dtype))

		# Middle
		h = hs[-1]
		h = ResnetBlock(dropout=self.dropout, name='mid.block_1', param_dtype=param_dtype)(
				h, emb=emb, deterministic=not train)
		h = FlaxTransformer2DModel(
				in_channels=h.shape[-1], n_heads=h.shape[-1]//self.head_dim, d_head=self.head_dim, param_dtype=param_dtype)(h, context)
		h = ResnetBlock(dropout=self.dropout, name='mid.block_2', param_dtype=param_dtype)(
				h, emb=emb, deterministic=not train)

		# Upsampling
		for i_level in reversed(range(num_resolutions)):
			# Residual blocks for this resolution
			for i_block in range(self.num_res_blocks + 1):
				h = maybe_remat(ResnetBlock, i_level)(
						out_ch=ch * self.ch_mult[i_level],
						dropout=self.dropout,
						name=f'up_{i_level}.block_{i_block}',
                        param_dtype=param_dtype)(
								jnp.concatenate([h, hs.pop()], axis=-1),
								emb=emb, deterministic=not train)
				if h.shape[1] in self.attn_resolutions:
					h = maybe_remat(FlaxTransformer2DModel, i_level)(
							in_channels=h.shape[-1], 
							n_heads=h.shape[-1]//self.head_dim, 
							d_head=self.head_dim, 
							only_cross_attention=(i_level == 0),
							param_dtype=param_dtype)(h, context)
			# Upsample
			if i_level != 0:
				h = self._upsample(
						h, name=f'up_{i_level}.upsample', emb=emb, train=train, param_dtype=param_dtype)
		assert not hs

		# End
		h = h.astype('float32')
		h = nonlinearity(Normalize(name='norm_out')(h))
		least_multof8 = int(np.ceil(self.out_ch/8) * 8) #force output to have 8 channels to allow sharding across the 8 cores. then discard the extra channels.
		h = nn.Conv(
				features=least_multof8,
				kernel_size=(3, 3),
				strides=(1, 1),
				kernel_init=nn.initializers.zeros,
				dtype=jnp.float32,
				name='conv_out')(h)[..., :self.out_ch]

		assert h.shape == (*x.shape[:3], self.out_ch)
		return h

	def _downsample(self, x, *, name, emb, train, param_dtype=jnp.float32):
		B, H, W, C = x.shape  # pylint: disable=invalid-name
		if self.resblock_resample:
			x = nn.remat(ResnetBlock)(
					dropout=self.dropout, resample='down', name=name, param_dtype=param_dtype)(
							x, emb=emb, deterministic=not train)
		else:
			x = nn.Conv(features=C, kernel_size=(3, 3), strides=(2, 2), name=name, param_dtype=param_dtype)(x)
		assert x.shape == (B, H // 2, W // 2, C)
		return x

	def _upsample(self, x, *, name, emb, train, param_dtype=jnp.float32):
		B, H, W, C = x.shape  # pylint: disable=invalid-name
		if self.resblock_resample:
			x = nn.remat(ResnetBlock)(
					dropout=self.dropout, resample='up', name=name, param_dtype=param_dtype)(
							x, emb=emb, deterministic=not train)
		else:
			x = nearest_neighbor_upsample(x)
			x = nn.Conv(features=C, kernel_size=(3, 3), strides=(1, 1), name=name, param_dtype=param_dtype)(x)
		assert x.shape == (B, H * 2, W * 2, C)
		return x
