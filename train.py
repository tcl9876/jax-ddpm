import jax
import jax.numpy as jnp
import numpy as np
import flax
from jax import random
import time
import os
from functools import partial
from tensorflow.config import set_visible_devices as tf_set_visible_devices
from tensorflow.io import gfile, write_file
from absl import app, flags
from ml_collections.config_flags import config_flags

from training_utils import EMATrainState, unreplicate, copy_pytree, count_params, save_checkpoint, train_step_fn, train_loss_fn, print_and_log, Metrics
from unet_condition_2d_flax import FlaxUNet2DConditionModel
from dataset_utils import create_dataset
import diffusers
import optax

tf_set_visible_devices([], device_type="GPU")
np.set_printoptions(precision=4)
jnp.set_printoptions(precision=4)


args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
#flags.DEFINE_string("data_dir", None, "the directory where your data is stored (or where it will be downloaded into).")
flags.mark_flags_as_required(["config", "global_dir"])


def main(_):
    #set up configs and dataset
    config, global_dir = args.config, args.global_dir
    config.unlock()

    margs = config.model
    dargs = config.dataset
    sargs = config.schedule
    oargs = config.optimizer
    targs = config.training_args

    if not gfile.isdir(global_dir):
        gfile.makedirs(global_dir)

    #dargs.data_dir = dargs.data_dir.format(args.data_dir)
    dargs.batch_size = targs.batch_size #set tfds dataloader batch size based on specified batch size in training args.
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)

    dataset = create_dataset(dargs)

    #create logfile
    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path):
        write_file(logfile_path, "")
    printl = partial(print_and_log, logfile_path=logfile_path)

    if config.schedule_name == "ddpm":
        schedule = diffusers.FlaxDDPMScheduler(**sargs)

    #create model, prng train state, train functions that we'll use for training
    model = FlaxUNet2DConditionModel(**margs)
    global_key = jax.random.PRNGKey(seed=0)
    global_key, init_key = jax.random.split(global_key, 2)
    params = model.init_weights(init_key)
    print('Total Parameters:', count_params(params))
    
    optimizer = optax.adam(oargs.lr, b1=oargs.b1, b2=oargs.b2, eps=oargs.eps)
    state = EMATrainState.create(
        apply_fn=model.apply,
        params=params,
        ema_params=copy_pytree(params),
        tx=optimizer,
        ema_decay=0.9999,
    )
    train_lossfn = partial(train_loss_fn, schedule_alphas_cumprod=schedule.alphas_cumprod)
    train_step = partial(train_step_fn, train_lossfn=train_lossfn)

    #replicate across tpu cores
    devices = jax.devices()
    print("Devices:", devices)
    state = flax.jax_utils.replicate(state, devices=devices)
    p_train_step = jax.pmap(
        fun=jax.jit(train_step),
        axis_name='shards',
    )

    #run the actual training process
    s = time.time()
    metrics = Metrics(['loss'])
    for global_step, (train_inputs) in zip(range(int(unreplicate(state.step)), targs.iterations), dataset):
        # Train step
        global_key, *device_keys = random.split(global_key, num=jax.local_device_count() + 1)
        device_keys = jax.device_put_sharded(device_keys, devices)

        state, new_metrics, global_norm = p_train_step(
            device_keys,
            state,
            train_inputs
        )
        
        if global_step%20==0:
            new_metrics = unreplicate(new_metrics)
            new_metrics = jax.tree_map(lambda x: float(x.mean()), new_metrics)
            gnorm = unreplicate(global_norm)
            gnorm = jax.tree_map(lambda x: float(x.mean()), gnorm)

            new_metrics['global_norm'] = gnorm
            metrics.update(new_metrics)

        if global_step % targs.log_freq==0: 
            printl(f'Real Step: {unreplicate(state.step)}, Batches passed this session: {global_step},  Metrics: {metrics}, Gnorm: {gnorm}, Time {round(time.time()-s)}s')

            metrics.reset_states()
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0:
                save_checkpoint(state, checkpoint_dir, unreplicate=True, keep=num_checkpoints)


if __name__ == '__main__':
    app.run(main)