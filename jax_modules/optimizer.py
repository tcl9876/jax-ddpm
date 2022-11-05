import jax.numpy as jnp
import optax

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

#TODO: implement ZeRO sharded optimizer and the needed pjit wrapper
def make_optimizer(config):
    """Make the optimizer."""

    optimizer_kwargs = {}
    if config.train.weight_decay > 0.:
        optimizer_kwargs['weight_decay'] = config.train.weight_decay

    learning_rate = CosineDecay(0.0, config.train.learning_rate, config.train.learning_rate, 1000, 1000000)
    if config.train.optimizer == 'adam' or config.train.optimizer == 'adamw':
        optimizer = optax.adamw(
            **optimizer_kwargs,
            b1=config.train.get('adam_beta1', 0.9),
            b2=config.train.get('adam_beta2', 0.999),
            learning_rate=learning_rate)
    else:
        raise NotImplementedError()

    return optimizer