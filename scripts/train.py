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
flags.DEFINE_string("sampling_probs", None, "if data_dir is a comma separated list of image directories, the respective probabilities to sample from each (also as comma separated list)")
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

    dataset_info_obj = Namespace(data_shape=config.in_dimensions) #we manually build tfrecord, can maybe clear up the legacy dataset code later.
    trainer = Trainer(config, dataset=dataset_info_obj)
    state = trainer.make_init_state()
    state = restore_checkpoint(targs.checkpoint_dirs[0], state, make_replicated=True)
    
    devices = jax.local_devices()
    start_step = int(unreplicate(state.step))
    barrier()

    print(f"Current iteration after restore on node {jax.process_index()}: {start_step}")
    if ismain:
        print("training devices:", devices)
        print("device count:", len(devices))

    fb_step = jax.pmap(trainer.forward_backward, axis_name='batch')
    update_func = jax.pmap(trainer.update_fn, axis_name='i')

    train_iter = build_tfrecord_dataset(args.data_dir, batch_sizes=[targs.batch_size//jax.local_device_count(), jax.local_device_count()],
        map_fn=read_encoded, process_index=jax.process_index(), process_count=jax.process_count(), repeating=True, sampling_probs=args.sampling_probs)

    s = time.time()
    if ismain:
        print("batch shapes:")
        jax.tree_map(lambda x: print(x.shape), next(train_iter))
    
    devices = jax.devices()
    local_devices = jax.local_devices()
    local_device_count = jax.local_device_count()

    local_core_on_chip = jax.device_put_sharded([jax.numpy.int32(i) for i in range(8)], local_devices)  
    global_rng = jax.random.PRNGKey(jax.process_index()) #set seed to process index so different nodes dont receive same rngs
    loss_metric = jax.device_put_replicated(jax.numpy.float32(0.), local_devices)
    gnorm_metric = jax.device_put_replicated(jax.numpy.float32(0.), local_devices)

    for global_step in range(start_step, targs.iterations + 1):
        batch = next(train_iter)
        global_rng, *train_step_rng = jax.random.split(global_rng, num=local_device_count + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, devices)

        grad, loss_metric, gnorm_metric = fb_step(train_step_rng, batch, state, local_core_on_chip, loss_metric, gnorm_metric)         
        state = update_func(state, grad)   
        
        # barrier() #TODO: experiment both with and without barrier on v3-32, benchmark speeds for each. 
        #why does it work for 2 steps but not 3? this happens with multiple models

        if global_step < 10: print('A step was successfully completed')

        if global_step%targs.log_loss_every_steps==0:
            loss_unrep = unreplicate(loss_metric) / targs.log_loss_every_steps
            gnorm_unrep = unreplicate(gnorm_metric) / targs.log_loss_every_steps
            kwargs = {
                "step": global_step + start_step,
                "metrics": {
                    "loss": loss_unrep,
                    "gnorm": gnorm_unrep,
                },
                "time": round(time.time()-s) 
            }

            if ismain:
                print_and_log_dict(logfile_path, kwargs, use_wandb)
            loss_metric *= 0
            gnorm_metric *= 0
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0 and ismain:
                state_unrep = state_make_unreplicated(state)
                save_checkpoint(checkpoint_dir, state_unrep, keep=num_checkpoints, step=state_unrep.step)
                
#checkpoints saving has a very large alloc? fix this?
#INIT STEP IS ALWAYS ZERO: FIX THIS

if __name__ == '__main__':
    app.run(main)
