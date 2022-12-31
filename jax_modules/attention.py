# This file contains code adapted from both Huggingface Transformers, and Google research's memory_efficient_attention: https://github.com/google-research/google-research/tree/master/memory_efficient_attention
# Both were licensed under apache 2, lines 38-103 were from google research, lines 104 and after from huggingface.
# Copyright 2022 The Google Research Authors.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import lax
import math
import functools


def nonlinearity(x):
	return x * nn.sigmoid(1.702 * x)

class LN32(nn.Module):
    
	@nn.compact
	def __call__(self, x):
		x_dtype = x.dtype
		normx_32 = nn.normalization.LayerNorm(epsilon=1e-5)(x.astype('float32'))
		return normx_32.astype(x_dtype)

#use bfloat and lax.Precision.DEFAULT, as well as add support for batching.
def _query_chunk_attention(query,
                            key,
                            value,
                            key_chunk_size=512, #the one specified in the paper/code was for much bigger attention matrices, so we made it smaller.
                            precision=lax.Precision.DEFAULT,
                            dtype=jnp.bfloat16):
    batch_size, num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features).astype(dtype)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum(
            '...qhd,...khd->...qhk', query, key, precision=precision).astype(dtype)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            '...vhf,...qhv->...qhf', value, exp_weights, precision=precision).astype(dtype)
        max_score = jnp.einsum('...qhk->...qh', max_score)
        return (exp_values, exp_weights.sum(axis=-1), max_score)
    
    def chunk_scanner(chunk_idx):
        key_chunk = lax.dynamic_slice(
            key, (0, chunk_idx, 0, 0),
            slice_sizes=(batch_size, key_chunk_size, num_heads, k_features))
        value_chunk = lax.dynamic_slice(
            value, (0, chunk_idx, 0, 0),
            slice_sizes=(batch_size, key_chunk_size, num_heads, v_features))
        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights

def mefficient_attention(query,
                        key,
                        value,
                        query_chunk_size=1024,
                        precision=jax.lax.Precision.DEFAULT,
                        dtype=jnp.bfloat16):
    batch_size, num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx, _):
        query_chunk = lax.dynamic_slice(
            query, (0, chunk_idx, 0, 0),
            slice_sizes=(batch_size, min(query_chunk_size, num_q), num_heads, q_features))
        return (chunk_idx + query_chunk_size,
                _query_chunk_attention(
                    query_chunk, key, value, precision=precision, dtype=dtype))

    _, res = lax.scan(
        chunk_scanner,
        init=0,
        xs=None,
        length=math.ceil(num_q / query_chunk_size))
    return jnp.concatenate(res, axis=0)

        
class FlaxAttentionBlock(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762
    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.param_dtype, param_dtype=self.param_dtype, name="to_q")
        self.kv = nn.Dense(inner_dim*2, use_bias=False, dtype=self.param_dtype, param_dtype=self.param_dtype, name="to_kv")

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.param_dtype, param_dtype=self.param_dtype, name="to_out_0")

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, head_size, head_dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj, value_proj = jnp.split(self.kv(context), 2, axis=-1)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        hidden_states = mefficient_attention(query_states, key_states, value_states)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        return hidden_states


class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762
    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    param_dtype: jnp.dtype = jnp.float32
    use_glu: bool = True

    def setup(self):
        # self attention (or nothing if only_cross_attention is True)
        if not self.only_cross_attention:
            self.attn1 = FlaxAttentionBlock(self.dim, self.n_heads, self.d_head, self.dropout, param_dtype=self.param_dtype)
            self.norm1 = LN32()
        else:
            self.attn1 = None
            self.norm1 = None
        # cross attention
        self.attn2 = nn.remat(FlaxAttentionBlock)(self.dim, self.n_heads, self.d_head, self.dropout, param_dtype=self.param_dtype)
        self.ff = nn.remat(FlaxFeedForward)(dim=self.dim, dropout=self.dropout, param_dtype=self.param_dtype, use_glu=self.use_glu)
        self.norm2 = LN32()
        self.norm3 = LN32()

    def __call__(self, hidden_states, context, deterministic=True):
        # possible self attention
        residual = hidden_states
        if not self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
            hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states


class FlaxTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf
    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        if self.use_linear_projection:
            self.proj_in = nn.Dense(inner_dim, dtype=self.param_dtype, param_dtype=self.param_dtype)
        else:
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.param_dtype,
                param_dtype=self.param_dtype,
            )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                param_dtype=self.param_dtype,
            )
            for _ in range(self.depth)
        ]

        if self.use_linear_projection:
            self.proj_out = nn.Dense(inner_dim, dtype=self.param_dtype, param_dtype=self.param_dtype)
        else:
            self.proj_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.param_dtype,
                param_dtype=self.param_dtype,
            )

    def __call__(self, hidden_states, context, deterministic=True):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states


class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a gated linear unit activation from:
    https://arxiv.org/abs/2002.05202
    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    param_dtype: jnp.dtype = jnp.float32
    use_glu: bool = True

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        if self.use_glu:
            self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.param_dtype)
        else:
            self.net_0 = nn.Dense(self.dim * 4, dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.param_dtype, param_dtype=self.param_dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.
    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.param_dtype, param_dtype=self.param_dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=-1)
        return hidden_linear * nonlinearity(hidden_gelu)
