import jax
import optax
import jax.numpy as jnp

class CosineDecay:
    def __init__(self, startlr, maxlr, minlr, warmup_steps, decay_steps):
        self.startlr = startlr
        self.maxlr = maxlr
        self.minlr = minlr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        step = jnp.minimum(step, self.decay_steps)
        startlr, maxlr, minlr = self.startlr, self.maxlr, self.minlr
        warmup = startlr + step/self.warmup_steps * (maxlr - startlr)

        decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * step/self.decay_steps))
        decay_factor = (1 - minlr/maxlr) * decay_factor + minlr/maxlr
        lr = maxlr * decay_factor
        return jnp.minimum(warmup, lr)

def make_adam(config):
    """Make the optimizer."""

    optimizer_kwargs = {}
    if config.train.weight_decay > 0.:
        optimizer_kwargs['weight_decay'] = config.train.weight_decay

    learning_rate = CosineDecay(0.0, config.train.learning_rate, config.train.learning_rate, 1000, 1000000)
    if config.train.optimizer == 'adam':
        optimizer = optax.adam(
            **optimizer_kwargs,
            b1=config.train.get('adam_beta1', 0.9),
            b2=config.train.get('adam_beta2', 0.999),
            learning_rate=learning_rate)
    else:
        raise NotImplementedError()

    return optimizer


def shard_pytree(pytree):
    def shard_tensor(inputs):
        axis_index = jax.lax.axis_index('i')
        sharded_shape = list(inputs.shape[:-1]) + [8, inputs.shape[-1]//8] #reshapes a [..., C] tensor to a [..., 8, C//8] tensor, allowing for indexing on the last axis.
        inputs_sharded = inputs.reshape(sharded_shape)
        return inputs_sharded[..., axis_index, :]
    return jax.tree_util.tree_map(shard_tensor, pytree)

def unshard_pytree(pytree):
    def unshard_tensor(inputs):
        gather_axis = len(inputs.shape) - 1
        gathered_inputs = jax.lax.all_gather(inputs, axis_name='i', axis=gather_axis)
        unsharded_shape = list(inputs.shape[:-1]) + [-1] #reshapes a [..., 8, C//8] tensor to a [..., C] tensor, which was its original shape
        return gathered_inputs.reshape(unsharded_shape)
    return jax.tree_util.tree_map(unshard_tensor, pytree)
