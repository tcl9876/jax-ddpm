import os
import time
import functools
import jax
import flax
from absl import app, flags
from ml_collections.config_flags import config_flags
from jax_modules.train_util import Trainer, Metrics
from jax_modules.checkpoints import save_checkpoint, restore_checkpoint, state_make_unreplicated
from jax_modules.utils import unreplicate, barrier, list_devices
from tensorflow.io import gfile, write_file
from t2i_datasets.utils import read_encoded, build_tfrecord_dataset
from argparse import Namespace

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored (or where it will be downloaded into).")
flags.DEFINE_string("wandb_project", None, "if you are using wandb to manage experiments, the project name.")
flags.DEFINE_string("wandb_run", None, "if you are using wandb to manage experiments, the experiment run name.")
flags.mark_flags_as_required(["config", "global_dir"])



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

    list_devices()
    
    use_wandb = (args.wandb_project is not None and args.wandb_run is not None)
    if ismain:
        if use_wandb: 
            import wandb
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=config.to_dict(), 
                resume=True
            )

        def print_and_log_dict(logfile_path, kwargs, use_wandb):
            #print and log a dict of kwargs.
            metric_dict = kwargs.pop("metrics")
            if use_wandb:
                wandb.log(metric_dict.to_dict())
                wandb.log(kwargs)
            printed_string = ""
            for k, v in kwargs.items():
                printed_string += f"{k}: {v}, "
            
            fstr = f"{printed_string[:-2]}, metrics: {repr(metric_dict)}"
            print(fstr)
            with gfile.GFile(logfile_path, mode='a') as f:
                f.write(fstr + '\n')
    else:
        print_and_log_dict = lambda a,b,c: None

    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path) and ismain:
        write_file(logfile_path, "")

    if ismain:
        print("MODEL CONFIG:", config.model)
        print("TRAINING CONFIG:", targs)

    dataset_info_obj = Namespace(data_shape=[32, 32, 4]) #we manually build tfrecord, can maybe clear up the legacy dataset code later.
    trainer = Trainer(config, dataset=dataset_info_obj)
    state = trainer.make_init_state()
    state = restore_checkpoint(targs.checkpoint_dirs[0], state, make_replicated=True)
    
    devices = jax.devices()
    #start_step = int(unreplicate(state.step))
    start_step = 0
    barrier()

    print(f"Current iteration after restore on node {jax.process_index()}: {start_step}")
    if ismain:
        print("training devices:", devices)
        print("device count:", len(devices))

    fb_step = jax.pmap(trainer.forward_backward, axis_name='batch')
    update_func = jax.pmap(trainer.update_fn, axis_name='i')

    train_iter = build_tfrecord_dataset(args.data_dir, batch_sizes=[targs.batch_size//jax.local_device_count(), jax.local_device_count()],
        map_fn=read_encoded, process_index=jax.process_index(), process_count=jax.process_count())

    s = time.time()
    metrics = Metrics(["train/gnorm", "train/loss"])

    if ismain:
        print("batch shapes:")
        jax.tree_map(lambda x: print(x.shape), next(train_iter))

    process_ids = jax.device_put_sharded([jax.numpy.int32(i) for i in range(8)], jax.local_devices())  
    global_rng = jax.random.PRNGKey(jax.process_index()) #set seed to process index so different nodes dont receive same rngs
    for global_step in range(start_step, targs.iterations + 1):
        batch = next(train_iter)
        global_rng, *train_step_rng = jax.random.split(global_rng, num=jax.local_device_count() + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, devices)

		#first run forward and backwards pass, and all-reduce the grads across ALL nodes. 
		#then self.optimizer.update takes the grads and applies them seperately per node. then we wait for all nodes to finish via a barrier.
        state, new_metrics = fb_step(train_step_rng, batch, state, process_ids) 

        if global_step == start_step:
            #flatgrad = jax.tree_util.tree_flatten(grad)[0]
            flatparams = jax.tree_util.tree_flatten(state.params)[0]
            flatmu = jax.tree_util.tree_flatten(state.optimizer_state[0].mu)[0]
            flatnu = jax.tree_util.tree_flatten(state.optimizer_state[0].nu)[0]
            for p, m, n in zip(flatparams, flatmu, flatnu):
                print(p.dtype, m.shape, n.shape)
            print('Trying with everything, with optimizer and whatnot')
        
        if (global_step+1)%config.train.n_accums == 0:
            state = update_func(state)
            # barrier() #TODO: experiment both with and without barrier on v3-32, benchmark speeds for each. 
            # also does it recompile? even just a regular psum if recompiled would be slow. could also be device_get() thats slow - check this too, prob better not use device_get

        if global_step < 10: print('A step was successfully completed')

        if global_step%100==0:
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
                print_and_log_dict(logfile_path, kwargs, use_wandb)
            metrics.reset_states()
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0 and ismain:
                state_unrep = state_make_unreplicated(state)
                save_checkpoint(checkpoint_dir, state_unrep, keep=num_checkpoints, step=state_unrep.step)
        
#checkpoints saving has a very large alloc? fix this?
#INIT STEP IS ALWAYS ZERO: FIX THIS

if __name__ == '__main__':
    app.run(main)
