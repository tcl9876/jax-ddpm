import jax
import jax.numpy as jnp
import numpy as np
import flax
from jax import random
import time
import os
from tensorflow.config import set_visible_devices as tf_set_visible_devices
from tensorflow.io import gfile, write_file, read_file
from absl import app, flags
from ml_collections.config_flags import config_flags
from checkpoints import latest_checkpoint_path
from unet import UNet
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import diffusers
from flax_pipeline import FlaxGeneralDiffusionPipeline

tf_set_visible_devices([], device_type="GPU")
np.set_printoptions(precision=4)
jnp.set_printoptions(precision=4)

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("checkpoint_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("save_dir", None, "the directory to save your results to.")
flags.DEFINE_string("n_samples", "36", "the number of samples you want to create. can be comma-separated list.")
flags.DEFINE_integer("n_steps", 1000, "how many evaluation steps you want to use")
flags.DEFINE_integer("max_batch_size", 64, "the maximum allowable batch size for sampling.")
flags.DEFINE_integer("height", 32, "image height.")
flags.DEFINE_integer("width", 32, "image width.")
flags.DEFINE_integer("nrow", 6, "if you are making a grid, the number of columns in the grid. By default, we use 6 columns.")
flags.DEFINE_string("guidance_scale", "1.0", "the guidance weight for classifier-free guidance.")
flags.DEFINE_string("save_format", "grid", "either 'grid' or 'npz'. determines whether to save results as a grid of images (default, best for <= 100 images), or as an .npz file (for evaluation).")
flags.mark_flags_as_required(["config", "checkpoint_dir", "save_dir"])


def plt_savefig(figure_path):
    if not (figure_path.startswith("gs://") or figure_path.startswith("gcs://")):
        plt.savefig(figure_path)
        return
    
    plt.savefig("./tmp_figure.png")

    write_file(
        figure_path, read_file("./tmp_figure.png")
    )

def save_images(images, save_path, nrow=6, scale=5):
    if nrow is None:
        m = int(np.ceil(len(images)/10))
        n = 10
    else:
        m = nrow
        n = int(np.ceil(len(images)/nrow))

    plt.figure(figsize=(scale*n, scale*m))

    for i in range(len(images)):
        plt.subplot(m, n, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    plt_savefig(save_path)
    plt.close('all')

def main(_):
    config = args.config
    config.unlock()

    ckpt_path = latest_checkpoint_path(args.checkpoint_dir, prefix='checkpoint_')
    with gfile.GFile(ckpt_path, 'rb') as fp:
        restored_sd = flax.serialization.from_bytes(None, fp.read())
    
    
    scheduler = diffusers.FlaxDDIMScheduler(beta_schedule="squaredcos_cap_v2")
    scheduler_state = scheduler.create_state()
    #we only support using DDIM, in practice we'd use Pytorch version of diffusers schedulers, a lot more support for those.
    
    params = {
        "unet": restored_sd["ema_params"],
        "scheduler": scheduler_state
    }
    unet = UNet(
        num_classes=1, out_ch=3, #TODO: put this within the config.
        **config.model.args
    )

    devices = jax.devices()
    print("DEVICES:", devices)
    params = flax.jax_utils.replicate(params, devices=devices)

    pipe = FlaxGeneralDiffusionPipeline(
        vae=None, 
        text_encoder=None, 
        tokenizer=None, 
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        dtype=jnp.bfloat16,
        model_config=config.model
    )

    #TODO: remove magic numbers
    n_samples_list = [int(n) for n in args.n_samples.split(",")]
    samples = np.zeros(shape=(0, 32, 32, 3)).astype('uint8')
    global_rng = jax.random.PRNGKey(0)

    for n_samples in n_samples_list:
        for n in tqdm(range(0, n_samples, args.max_batch_size)):
            batch_size = min(args.max_batch_size, n_samples - n)
            #prompt_ids = [None] * len(devices)
            global_rng, *rng = jax.random.split(global_rng, jax.device_count() + 1)
            rng = jax.device_put_sharded(rng, jax.devices())

            per_replica_bs = args.max_batch_size//jax.device_count()
            prompt_ids = [None] * per_replica_bs
            current_images = pipe(
                prompt_ids=prompt_ids,
                params=params,
                prng_seed=rng,
                num_inference_steps=args.n_steps,
                height=32,
                width=32,
                guidance_scale=1.0,
                jit=True
            ).block_until_ready()

            current_images = np.array(current_images.reshape(-1, 32, 32, 3))[:batch_size]
            current_images = np.clip(current_images * 255, 0, 255).astype('uint8')
            samples = np.concatenate((samples, current_images), axis=0)
        
        ext = "png" if args.save_format.lower() == "grid" else "npz"
        label_string = "uncond"

        samples_identifier = f"{len(gfile.glob(f'{args.save_dir}/*.{ext}'))}_{label_string}"
        samples_path = os.path.join(args.save_dir, f"samples_{samples_identifier}.{ext}")
        
        if args.save_format.lower() == "grid":
            save_images(samples, samples_path, nrow=args.nrow)
        else:
            np.savez("tmpfile.npz", arr0=samples)
            gfile.copy("tmpfile.npz", samples_path)
            time.sleep(3.0)
            gfile.remove("tmpfile.npz")

        print(f"Saved {len(samples)} samples to {samples_path}")

"""
LEGACY EVALUATOR USING PROGRESSIVE DISTILLATION CODE.

def main(_):
    config = args.config
    config.unlock()

    model = Model(config)
    ckpt_path = latest_checkpoint_path(args.checkpoint_dir, prefix='checkpoint_')
    with gfile.GFile(ckpt_path, 'rb') as fp:
        restored_sd = flax.serialization.from_bytes(None, fp.read())
    
    params = {
        "unet": restored_sd["ema_params"],
        #"scheduler": scheduler.create_state()
    }

    devices = jax.devices()
    print("DEVICES:", devices)
    params = flax.jax_utils.replicate(params, devices=devices)

    #TODO: remove magic numbers
    n_samples_list = [int(n) for n in args.n_samples.split(",")]
    samples = np.zeros(shape=(0, 32, 32, 3)).astype('uint8')
    
    global_rng = jax.random.PRNGKey(0)

    def samples_fn(rng, params, labels, num_steps, num_samples): 
        return model.samples_fn(rng=rng,
            params=params,
            labels=labels,
            num_steps=num_steps,
            num_samples=num_samples
        )
    samples_fn = jax.pmap(samples_fn, axis_name='batch', static_broadcasted_argnums=(2, 3, 4))

    for n_samples in n_samples_list:
        for n in tqdm(range(0, n_samples, args.max_batch_size)):
            batch_size = min(args.max_batch_size, n_samples - n)
            #prompt_ids = [None] * len(devices)
            global_rng, *rng = jax.random.split(global_rng, jax.device_count() + 1)
            rng = jax.device_put_sharded(rng, jax.devices())

            per_replica_bs = args.max_batch_size//jax.device_count()
            current_images = samples_fn(
                rng,
                params["unet"],
                None,
                args.n_steps,
                per_replica_bs
            ).block_until_ready()

            current_images = np.array(current_images.reshape(-1, 32, 32, 3))[:batch_size]
            current_images = np.clip(current_images * 127.5 + 127.5, 0, 255).astype('uint8')
            samples = np.concatenate((samples, current_images), axis=0)
        
        ext = "png" if args.save_format.lower() == "grid" else "npz"
        label_string = "uncond"

        samples_identifier = f"{len(gfile.glob(f'{args.save_dir}/*.{ext}'))}_{label_string}"
        samples_path = os.path.join(args.save_dir, f"samples_{samples_identifier}.{ext}")
        
        if args.save_format.lower() == "grid":
            save_images(samples, samples_path, nrow=args.nrow)
        else:
            np.savez("tmpfile.npz", arr0=samples)
            gfile.copy("tmpfile.npz", samples_path)
            time.sleep(3.0)
            gfile.remove("tmpfile.npz")

        print(f"Saved {len(samples)} samples to {samples_path}")
"""

if __name__ == '__main__':
    app.run(main)

    