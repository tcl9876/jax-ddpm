import os
import time
import functools
import jax
import flax
from absl import app, flags
from ml_collections.config_flags import config_flags
from jax_modules.train_util import Trainer, Metrics
from jax_modules.checkpoints import save_checkpoint, restore_checkpoint
from jax_modules.utils import numpy_iter, unreplicate
from tensorflow.io import gfile, write_file

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored (or where it will be downloaded into).")
flags.mark_flags_as_required(["config", "global_dir"])

def print_and_log(*args, logfile_path):
    print(*args)
    for a in args:
        with gfile.GFile(logfile_path, mode='a') as f:
            f.write(str(a))

    with gfile.GFile(logfile_path, mode='a') as f:
        f.write('\n')


def main(_):
    config, global_dir = args.config, args.global_dir
    config.unlock()

    if not gfile.isdir(global_dir):
        gfile.makedirs(global_dir)
    
    targs = config.train
    #dargs.batch_size = targs.batch_size #set tfds dataloader batch size based on specified batch size in training args.
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)
    
    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path):
        write_file(logfile_path, "")
    printl = functools.partial(print_and_log, logfile_path=logfile_path)

    #TODO: add checkpoint restoration.
    trainer = Trainer(config)
    state = jax.device_get(trainer.make_init_state())
    state = restore_checkpoint(targs.checkpoint_dirs[0], state)
    print(f"Current iteration after restore: {state.step}")
    state = flax.jax_utils.replicate(state)

    train_step = functools.partial(trainer.step_fn, jax.random.PRNGKey(0), True)
    train_step = functools.partial(jax.lax.scan, train_step)  # for substeps
    train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

    total_bs = config.train.batch_size
    device_bs = total_bs // jax.device_count()
    train_ds = trainer.dataset.get_shuffled_repeated_dataset(
        split='train',
        batch_shape=(
            jax.local_device_count(),  # for pmap
            config.train.substeps,  # for lax.scan over multiple substeps
            device_bs,  # batch size per device
        ),
        local_rng=jax.random.PRNGKey(0),
        augment=True)
    train_iter = numpy_iter(train_ds)

    s = time.time()
    metrics = Metrics(["train/gnorm", "train/loss"])

    for global_step in range(unreplicate(state.step), targs.iterations + targs.substeps, targs.substeps):
        batch = next(train_iter)
        state, new_metrics = train_step(state, batch)
        if global_step%2==0 or global_step < 100:
            new_metrics = unreplicate(new_metrics)
            new_metrics = jax.tree_map(lambda x: float(x.mean()), new_metrics)
            metrics.update(new_metrics)

        if global_step % targs.log_loss_every_steps==0 or global_step < 100: 
            printl(f'Real Step: {unreplicate(state.step)}, Batches passed this session: {global_step},  Metrics: {metrics}, Time {round(time.time()-s)}s')
            metrics.reset_states()
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0:
                unreplicated_state = unreplicate(state)
                save_checkpoint(checkpoint_dir, unreplicated_state, keep=num_checkpoints, step=unreplicated_state.step)
    

if __name__ == '__main__':
    app.run(main)