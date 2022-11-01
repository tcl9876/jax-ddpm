import diffusers
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints
from tensorflow.io import gfile, write_file

from typing import Any, Callable

def count_params(pytree):
    return sum([x.size for x in jax.tree_leaves(pytree)])

def copy_pytree(pytree):
    return jax.tree_util.tree_map(jnp.array, pytree)

def save_checkpoint(state, ckpt_dir, ckpt_path=None, unreplicate=False, keep=99):
    if unreplicate: state = unreplicate(state)
    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=state.step, keep=keep)

def restore_checkpoint(empty_state, ckpt_dir=None, ckpt_path=None, step=None):
    return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=empty_state, step=step)

def compute_global_norm(grads):
    norms, _ = jax.flatten_util.ravel_pytree(jax.tree_map(jnp.linalg.norm, grads))
    return jnp.linalg.norm(norms)

def unreplicate(x):
    return jax.device_get(flax.jax_utils.unreplicate(x))

#prints to the console and appends to the logfile at logfile_path
def print_and_log(*args, logfile_path):
    print(*args)
    for a in args:
        with gfile.GFile(logfile_path, mode='a') as f:
            f.write(str(a))

    with gfile.GFile(logfile_path, mode='a') as f:
        f.write('\n')

class EMATrainState(TrainState):    
    ema_decay: float
    ema_params: flax.core.FrozenDict[str, Any]

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.
        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.
        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.
        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = jax.tree_map(lambda ema, p: ema * self.ema_decay + (1 - self.ema_decay) * p,
                                      self.ema_params, new_params)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, ema_params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            ema_decay=ema_decay,
            **kwargs,
        )


def train_loss_fn(params, key, state, train_inputs, schedule_alphas_cumprod):
    img, label = train_inputs
    time_key, noise_key = jax.random.split(key, 2)

    timesteps = jax.random.randint(time_key, minval=0, maxval=1000, shape=(img.shape[0],), dtype=jnp.int32)
    alphas_cumprod = schedule_alphas_cumprod[timesteps]
    alphas_cumprod = alphas_cumprod[:, None, None, None]
    eps = jax.random.normal(noise_key, shape=img.shape, dtype=img.dtype)
    noisy_img = img * jnp.sqrt(alphas_cumprod) + eps * jnp.sqrt(1 - alphas_cumprod)

    pred_eps = state.apply_fn({'params': params}, sample=noisy_img, timesteps=timesteps, encoder_hidden_states=None).sample 
    loss = jnp.sum(jnp.square(eps - pred_eps))
    return loss, {'loss': loss}
    

def train_step_fn(key, state, train_inputs, train_lossfn):
    grad_fn = jax.value_and_grad(train_lossfn, has_aux=True, argnums=0)
    (loss, metrics), grads = grad_fn(state.params, key, state, train_inputs)

    grads = jax.lax.pmean(grads, axis_name='shards')
    global_norm = compute_global_norm(grads)

    state = state.apply_gradients(grads=grads)
    return state, metrics, global_norm


#Metrics related utils
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
    
    def __repr__(self):
        return repr(self._metric_dict)
    
    def update(self, new_metrics):
        for name in self.names:
            self._metric_dict[name].update(new_metrics[name])
        
    def reset_states(self):
        for name in self.names:
            self._metric_dict[name].reset_states()