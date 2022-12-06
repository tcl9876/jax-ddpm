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
import numpy as np
from functools import partial
from typing import Optional


def nonlinearity(x):
	return x * nn.sigmoid(1.702 * x)

class LN32(nn.Module):
    
	@nn.compact
	def __call__(self, x):
		x_dtype = x.dtype
		normx_32 = nn.normalization.LayerNorm(epsilon=1e-5)(x.astype('float32'))
		return normx_32.astype(x_dtype)

class SequenceProcessor(nn.Module):
    seq_width: Optional[int] = 1024
    @nn.compact
    def __call__(self, clip_seq, t5_seq):
        """
        does several things:
        1) applies layernormalization + linear transformation separately to both, as t5 seq may have different variance, and they'll eventually be projected by the same weight matrices.
        2) prepends a null embedding to the sequence, similar to https://arxiv.org/pdf/2211.01324.pdf. 
        3) concatenates the sequences.
        """
        
        b, hc, ht5 = clip_seq.shape[0], clip_seq.shape[-1], t5_seq.shape[-1]
        if self.seq_width is None: 
            seq_width = hc
        else:
            seq_width = self.seq_width

        assert hc%seq_width == 0 and ht5%seq_width == 0
        clip_seq = nn.Dense(seq_width)(LN32()(clip_seq))
        t5_seq = nn.Dense(seq_width)(LN32()(t5_seq))
        
        null_emb = nn.Embed(1, seq_width)(jnp.zeros([b, 1], dtype=jnp.int32))
        return jnp.concatenate([null_emb, clip_seq, t5_seq], axis=1)

def max_pow2_that_evenly_divides(n):
    i = 1
    while n%2 ==0:
        i *= 2
        n = n//2
    return i

def scaled_dp_attention(q, k, v, scale):
    attention_scores = jnp.einsum("b i d, b j d->b i j", q, k)
    attention_scores = attention_scores * scale
    attention_probs = jax.nn.softmax(attention_scores, axis=2)
    return jnp.einsum("b i j, b j d -> b i d", attention_probs, v)

def att_with_carry(carry, args, scale): #make compatible with jax.lax.scan
    return None, scaled_dp_attention(*args, scale)

def scan_att(q, k, v, scale, n_splits): 
    assert q.shape[0]%n_splits == 0
    nq, nk, nv = [jnp.reshape(h, (n_splits, h.shape[0]//n_splits, h.shape[-2], h.shape[-1])) for h in [q, k, v]]
    scanfn = partial(att_with_carry, scale=scale)
    outs = jax.lax.scan(scanfn, None, [nq, nk, nv])[1]
    return outs.reshape(-1, outs.shape[-2], outs.shape[-1])

#a self attention that limits memory usage dynamically by looking at the sizes of q and k. it is exactly equivalent to scaled_dp_attention.
#by default, dont store more than ~= a single 8-headed 32^2 SA matrix.
def lowmem_dot_product_attention(q, k, v, scale, max_allowable=(1024 * 1024 * 8)):
    #SHAPE: (batch_size * head_size, seq_len, dim // head_size)

    dimprod = q.shape[0] * q.shape[1] * k.shape[1] #full memory cost of SA matrix all heads.
    true_divisor = dimprod/max_allowable
    if true_divisor < 1:
        n_splits = 1 #dont break it down at all, because batching it more is faster.
    elif true_divisor >= q.shape[0]:
        n_splits = q.shape[0] #break up q so it has shape [1, ...] , the most possible you can break it down.
    else:
        nearest_pow2 = 2 ** np.round(np.log2(true_divisor))
        max_pow2 = max_pow2_that_evenly_divides(q.shape[0])
        n_splits = int(min(max_pow2, nearest_pow2))
    
    return scan_att(q, k, v, scale, n_splits)

        
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
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj, value_proj = jnp.split(self.kv(context), 2, axis=-1)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        hidden_states = lowmem_dot_product_attention(query_states, key_states, value_states, self.scale)

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
        self.ff = nn.remat(FlaxGluFeedForward)(dim=self.dim, dropout=self.dropout, param_dtype=self.param_dtype)
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


class FlaxGluFeedForward(nn.Module):
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

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.param_dtype)
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
