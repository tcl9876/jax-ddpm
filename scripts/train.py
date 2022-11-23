import os
import time
import functools
import jax
import flax
from absl import app, flags
from ml_collections.config_flags import config_flags
from jax_modules.train_util import Trainer, Metrics, local_shard_pytree
from jax_modules.checkpoints import save_checkpoint, restore_checkpoint
from jax_modules.utils import numpy_iter, unreplicate, barrier
from tensorflow.io import gfile, write_file
import wandb

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored (or where it will be downloaded into).")
flags.DEFINE_string("wandb_project", None, "if you are using wandb to manage experiments, the project name.")
flags.DEFINE_string("wandb_run", None, "if you are using wandb to manage experiments, the experiment run name.")
flags.mark_flags_as_required(["config", "global_dir"])

def print_and_log_dict(logfile_path, kwargs):
    #print and log a dict of kwargs.
    metric_dict = kwargs.pop("metrics")
    wandb.log(metric_dict.to_dict())
    wandb.log(kwargs)
    printed_string = ""
    for k, v in kwargs.items():
        printed_string += f"{k}: {v}, "
    
    fstr = f"{printed_string[:-2]}, metrics: {repr(metric_dict)}"
    print(fstr)
    with gfile.GFile(logfile_path, mode='a') as f:
        f.write(fstr + '\n')

def main(_):
    config, global_dir = args.config, args.global_dir
    config.unlock()

    if not gfile.isdir(global_dir):
        gfile.makedirs(global_dir)
    
    targs = config.train
    #dargs.batch_size = targs.batch_size #set tfds dataloader batch size based on specified batch size in training args.
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)
    ismain = (jax.process_index() == 0)
    
    use_wandb = (args.wandb_project is not None and args.wandb_run is not None)
    if use_wandb and ismain:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=config.to_dict(), 
            resume=True
        )

    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path) and ismain:
        write_file(logfile_path, "")

    if ismain:
        print("MODEL CONFIG:", config.model)
        print("TRAINING CONFIG:", targs)

    trainer = Trainer(config)
    state = trainer.make_init_state()
    state = restore_checkpoint(targs.checkpoint_dirs[0], state)
    
    devices = jax.devices()
    start_step = int(unreplicate(state.step))
    barrier()
    if ismain:
        print(f"Current iteration after restore: {start_step}")
        print("training devices:", devices)
        print("device count:", len(devices))

    fb_step = jax.pmap(trainer.forward_backward, axis_name='batch')
    update_func = jax.pmap(trainer.update, axis_name='i')

    total_bs = config.train.batch_size
    device_bs = total_bs // len(devices)
    train_ds = trainer.dataset.get_shuffled_repeated_dataset(
        split='train',
        batch_shape=(
            len(devices),  # should it be device_count or local_device_count for multinode?
            device_bs,  # batch size per device
        ),
        local_rng=jax.random.PRNGKey(0),
        augment=True,
        data_dir=args.data_dir)
    train_iter = numpy_iter(train_ds)

    s = time.time()
    metrics = Metrics(["train/gnorm", "train/loss"])

    global_rng = jax.random.PRNGKey(0)
    for global_step in range(start_step, targs.iterations + 1):
        batch = next(train_iter)
        global_rng, *train_step_rng = jax.random.split(global_rng, num=jax.device_count() + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, devices)

		#first run forward and backwards pass, and all-reduce the grads across ALL nodes. 
		#then self.optimizer.update takes the grads and applies them seperately per node. then we wait for all nodes to finish via a barrier.
        grad, new_metrics = fb_step(train_step_rng, batch, state.params)

        """
        sg = jax.tree_util.tree_flatten(grad)[0]
        sp = jax.tree_util.tree_flatten(state.sharded_params)[0]
        for a, b in zip(sg, sp):
            print(a.shape, b.shape)
        
        for i in jax.tree_util.tree_flatten(state.optimizer_state[0])[0]:
            if len(i.shape) == 0:
                print(i)
        """

        state = update_func(state, grad)
        barrier() #benchmark speed w/o barrier, in case jax.device_get is slow

        if global_step%20==0:
            new_metrics = unreplicate(new_metrics)
            new_metrics = jax.tree_map(lambda x: float(x.mean()), new_metrics)
            metrics.update(new_metrics)

        if global_step % targs.log_loss_every_steps==0: 
            real_step = unreplicate(state.step)
            kwargs = {
                "real step": real_step,
                "total images seen": real_step * targs.batch_size,
                "metrics": metrics,
                "seconds elapsed": round(time.time()-s)
            }
            if ismain:
                print_and_log_dict(logfile_path, kwargs)
            metrics.reset_states()
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0 and ismain:
                save_checkpoint(checkpoint_dir, state, keep=num_checkpoints, step=unreplicate(state.step))
        

if __name__ == '__main__':
    app.run(main)