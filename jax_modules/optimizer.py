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

    learning_rate = CosineDecay(0.0, config.train.max_learning_rate, config.train.min_learning_rate, config.train.learning_rate_warmup_steps, config.train.get('decay_steps', config.train.iterations))
    if config.train.optimizer == 'adam':
        optimizer = optax.adamw(
            **optimizer_kwargs,
            b1=config.train.get('adam_beta1', 0.9),
            b2=config.train.get('adam_beta2', 0.999),
            learning_rate=learning_rate)
    else:
        raise NotImplementedError()

    if config.train.enable_update_skip:
        optimizer = optax.apply_if_finite(optimizer, 100) #any more than 100 NaN's and something is definitely wrong.
    return optimizer
