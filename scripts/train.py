import os
import time
import jax
from absl import app, flags
from ml_collections.config_flags import config_flags
from jax_modules.train_util import Trainer, T5_CHANNELS, CLIP_CHANNELS
from jax_modules.checkpoints import save_checkpoint, restore_checkpoint, state_make_unreplicated
from jax_modules.dist_util import unreplicate, barrier, list_devices, assert_synced
from jax_modules.utils import print_and_log
from tensorflow.io import gfile, write_file
from t2i_datasets.utils import read_encoded, build_tfrecord_dataset
from argparse import Namespace
import numpy as np
import jax.numpy as jnp
import flax
import diffusers
from diffusion.flax_pipeline import FlaxGeneralDiffusionPipeline
from functools import partial


args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored (or where it will be downloaded into).")
flags.DEFINE_string("sampling_probs", None, "if data_dir is a comma separated list of image directories, the respective probabilities to sample from each (also as comma separated list)")
flags.DEFINE_string("wandb_project", None, "if you are using wandb to manage experiments, the project name.")
flags.DEFINE_string("wandb_run", None, "if you are using wandb to manage experiments, the experiment run name.")
flags.DEFINE_string("captions_path", None, "if you are evaluating during training, the path for the npz array that contains text embeddings of the desired captions.")
flags.DEFINE_string("start_tfrecord_index", "auto", "starts from the n'th tfrecord file on each node. so if you want to skip the first 10000 files in all, and have 4 nodes, set --start_tfrecord_index 2500")
flags.mark_flags_as_required(["config", "global_dir"])


def evaluate_model_on_captions(state, trainer, config, captions_path, imsave_path, default_aesth_score=6.5, clip_image_emb=None):
    print(f'evaluating model on captions at {captions_path}')
    if os.path.isfile("tmpfile.npz"): os.remove("tmpfile.npz")
    gfile.copy(captions_path, "tmpfile.npz")
    captions_arr = np.load("tmpfile.npz")
    
    if clip_image_emb is None:
        clip_image_emb = jnp.zeros((captions_arr['clip_emb'].shape[0], captions_arr['clip_emb'].shape[-1]))
    default_aesth_score = jnp.ones((captions_arr['clip_emb'].shape[0], )) * default_aesth_score
    context = {
        "clip_emb": flax.jax_utils.replicate(captions_arr['clip_emb'], jax.local_devices()),
        "t5_emb": flax.jax_utils.replicate(captions_arr['t5_emb'], jax.local_devices()),
        "clip_image_emb": flax.jax_utils.replicate(clip_image_emb, jax.local_devices()),
        "aesth_score": flax.jax_utils.replicate(default_aesth_score, jax.local_devices())
    }

    noise_schedule = config.model.eval_alpha_schedule
    scheduler = diffusers.FlaxDDIMScheduler(beta_schedule="linear", 
        beta_start=noise_schedule.beta_start, beta_end=noise_schedule.beta_end)
    scheduler_state = scheduler.create_state()
    scheduler_state = flax.jax_utils.replicate(scheduler_state, jax.local_devices())
    global_rng = jax.random.PRNGKey(0)
    global_rng, *rng = jax.random.split(global_rng, jax.device_count() + 1)
    rng = jax.device_put_sharded(rng, jax.devices())
    
    params = {
        "unet": state.params,
        "scheduler": scheduler_state
    }
    pipe = FlaxGeneralDiffusionPipeline(
        vae=None, 
        text_encoder=None, 
        tokenizer=None, 
        unet=trainer.model,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        dtype=jax.numpy.float32,
        model_config=config.model
    )

    B, H, W, C = captions_arr['clip_emb'].shape[0], config.in_dimensions[0],  config.in_dimensions[1], config.model.args.out_ch
    _, latents = pipe(
        prompt_ids=context,
        params=params,
        prng_seed=rng,
        num_inference_steps=50,
        height=H,
        width=W,
        guidance_scale=5.,
        jit=True
    )
    latents = np.array(latents.reshape(-1, H, W, C))

    #note: these will have to be decoded by a VAE, in a separate session
    np.savez("tmpfile.npz", latents=latents)
    gfile.copy("tmpfile.npz", imsave_path)
    time.sleep(3.0)
    gfile.remove("tmpfile.npz")
    print(f'Successfully saved latents of shape {latents.shape} at {imsave_path}')
    print(f"Mean {latents.mean()} , Std {latents.std()} , Min {latents.min()} , Max {latents.max()}")



def main(_):
    config, global_dir = args.config, args.global_dir
    config.unlock()

    if not gfile.isdir(global_dir):
        gfile.makedirs(global_dir)
    
    targs = config.train
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)
    ismain = (jax.process_index() == 0)

    list_devices()
    
    captions_path = args.captions_path
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
    assert_synced(state.params)

    devices = jax.devices()
    local_devices = jax.local_devices()
    local_device_count = jax.local_device_count()
    start_step = int(unreplicate(state.step))
    barrier()


    print(f"Current iteration after restore on node {jax.process_index()}: {start_step}")
    if ismain:
        print("training devices:", devices)
        print("device count:", len(devices))

    fb_step = jax.pmap(trainer.forward_backward, axis_name='all_devices', devices=devices) #fb_step will all-reduce grads over all nodes, so devices includes all devices
    update_func = jax.pmap(trainer.update_fn, axis_name='local_devices', devices=local_devices) #the state update is run independently on each node, so devices only include local devices
    printl = partial(print_and_log, logfile_path=logfile_path)

    if args.start_tfrecord_index.lower() == "auto":
        steps_per_tfrecord = config.dataset.args.number_encodings_per_shard/targs.batch_size
        num_tfrecords_passed = int(np.ceil(start_step/steps_per_tfrecord))
    else:
        num_tfrecords_passed = int(args.start_tfrecord_index)

    
    map_fn = partial(read_encoded, remove_keys=config.dataset.args.remove_keys, clip_channels=CLIP_CHANNELS[config.model.clip_model_id], t5_channels=T5_CHANNELS[config.model.t5_model_id])
    train_iter = build_tfrecord_dataset(args.data_dir, batch_sizes=[targs.batch_size//local_device_count, local_device_count],
        map_fn=map_fn, process_index=jax.process_index(), process_count=jax.process_count(), repeating=False, num_tfrecords_passed=num_tfrecords_passed,
        sampling_probs=args.sampling_probs, print_func=printl)

    s = time.time()
    if ismain:
        print("batch shapes:")
        jax.tree_map(lambda x: print(x.shape), next(train_iter))
    
    local_core_on_chip = jax.device_put_sharded([jax.numpy.int32(i) for i in range(local_device_count)], local_devices)  
    global_rng = jax.random.PRNGKey(jax.process_index()) #set seed to process index so different nodes dont receive same rngs
    loss_metric = jax.device_put_replicated(jax.numpy.float32(0.), local_devices)
    gnorm_metric = jax.device_put_replicated(jax.numpy.float32(0.), local_devices)
    
    print('starting training')
    for global_step in range(start_step, targs.iterations + 1):
        try:
            batch = next(train_iter)
            if global_step == start_step:
                jax.tree_map(lambda x: print(x.shape) if hasattr(x, "shape") else _, batch)
                print(jnp.var(batch["t5_emb"]), jnp.var(batch["clip_emb"]), jnp.var(batch["image"]) )
        except StopIteration:
            print("This epoch is complete!") #manual restart it if its done and you want >1 epochs.
            break

        global_rng, *train_step_rng = jax.random.split(global_rng, num=local_device_count + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, local_devices)

        grad, loss_metric, gnorm_metric = fb_step(train_step_rng, batch, state, local_core_on_chip, loss_metric, gnorm_metric)         
        state = update_func(state, grad)
        # barrier() #is a barrier needed or will it wait normally? 

        if global_step < 10: print('A step was successfully completed')

        if global_step%targs.log_loss_every_steps==0:
            loss_unrep = unreplicate(loss_metric) / targs.log_loss_every_steps
            gnorm_unrep = unreplicate(gnorm_metric) / targs.log_loss_every_steps
            kwargs = {
                "step": global_step,
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
                assert_synced(state.params)
                state_unrep = state_make_unreplicated(state)
                save_checkpoint(checkpoint_dir, state_unrep, keep=num_checkpoints, step=state_unrep.step)

        if global_step%targs.snapshot_freq==0 and ismain:
            imsave_path = os.path.join(os.path.join(global_dir, "results"), f'images_{global_step}.npz')
            if ismain and args.captions_path is not None:
                evaluate_model_on_captions(state, trainer, config, captions_path, imsave_path)


if __name__ == '__main__':
    app.run(main)
