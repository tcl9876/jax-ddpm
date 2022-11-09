import jax
import jax.numpy as jnp
import numpy as np
import flax
import time
import os
from tensorflow.config import set_visible_devices as tf_set_visible_devices
from tensorflow.io import gfile
from absl import app, flags
from ml_collections.config_flags import config_flags
from jax_modules.checkpoints import restore_checkpoint
from jax_modules.unet import UNet
from jax_modules.utils import save_tiled_imgs
from tqdm.auto import tqdm
import diffusers
from flax_pipeline import FlaxGeneralDiffusionPipeline
from diffusers import FlaxAutoencoderKL
import wandb

tf_set_visible_devices([], device_type="GPU")
np.set_printoptions(precision=4)
jnp.set_printoptions(precision=4)

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("checkpoint_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("wandb_project", None, "if you are using wandb to manage experiments, the project name.")
flags.DEFINE_string("wandb_run", None, "if you are using wandb to manage experiments, the experiment run name.")
flags.DEFINE_string("save_dir", None, "the directory to save your results to.")
flags.DEFINE_string("n_samples", "36", "the number of samples you want to create. can be comma-separated list.")
flags.DEFINE_integer("n_steps", 1000, "how many evaluation steps you want to use")
flags.DEFINE_integer("max_batch_size", 64, "the maximum allowable batch size for sampling.")
flags.DEFINE_string("auth token", None, "hugging face authentication token for Stable Diffusion.")
flags.DEFINE_integer("height", 256, "image height.")
flags.DEFINE_integer("width", 256, "image width.")
flags.DEFINE_integer("ncol", 6, "if you are making a grid, the number of columns in the grid. By default, we use 6 columns.")
flags.DEFINE_string("guidance_scale", "1.0", "the guidance weight for classifier-free guidance.")
flags.DEFINE_string("save_format", "grid", "either 'grid' or 'npz'. determines whether to save results as a grid of images (default, best for <= 100 images), or as an .npz file (for evaluation).")
flags.mark_flags_as_required(["config", "checkpoint_dir", "save_dir"])


def main(_):
    config = args.config
    config.unlock()

    if config.eval_snr_schedule.name == "cosine":
        scheduler = diffusers.FlaxDDIMScheduler(beta_schedule="squaredcos_cap_v2")
    elif config.eval_snr_schedule.name == "linear":
        scheduler = diffusers.FlaxDDIMScheduler(beta_schedule="linear", beta_start=config.eval_snr_schedule.beta_start, beta_end=config.eval_snr_schedule.beta_end)
    else:
        raise NotImplementedError

    scheduler_state = scheduler.create_state()
    #we only support using DDIM, in practice we'd use Pytorch version of diffusers schedulers, a lot more support for those.
    
    restored_sd = restore_checkpoint(args.checkpoint_dir, None) #restore the checkpoint as a dict.
    del restored_sd["optimizer_state"]
    print(f"Restored Checkpoint from {restored_sd.step} steps")
    params = {
        "unet": restored_sd["ema_params"],
        "scheduler": scheduler_state
    }
    unet = UNet(**config.model.args)

    devices = jax.devices()
    print("DEVICES:", devices)
    params = flax.jax_utils.replicate(params, devices=devices)

    vae = FlaxAutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=args.auth_token)
    pipe = FlaxGeneralDiffusionPipeline(
        vae=vae, 
        text_encoder=None, 
        tokenizer=None, 
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        dtype=jnp.bfloat16,
        model_config=config.model
    )

    h, w, c = args.height, args.width, unet.out_ch
    n_samples_list = [int(n) for n in args.n_samples.split(",")]
    samples = np.zeros(shape=(0, h, w, c)).astype('uint8')
    global_rng = jax.random.PRNGKey(0)

    for n_samples in n_samples_list:
        for n in tqdm(range(0, n_samples, args.max_batch_size)):
            batch_size = min(args.max_batch_size, n_samples - n)
            #prompt_ids = [None] * len(devices)
            global_rng, *rng = jax.random.split(global_rng, jax.device_count() + 1)
            rng = jax.device_put_sharded(rng, jax.devices())

            per_replica_bs = args.max_batch_size//jax.device_count()

            if unet.num_classes == 1:
                prompt_ids = [None] * per_replica_bs
            else:
                global_rng, class_key = jax.random.split(global_rng)
                prompt_ids = jax.random.randint(class_key, shape=(jax.device_count(), per_replica_bs), minval=0, maxval=unet.num_classes)

            current_images = pipe(
                prompt_ids=prompt_ids,
                params=params,
                prng_seed=rng,
                num_inference_steps=args.n_steps,
                height=h,
                width=w,
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
            pil_image_result = save_tiled_imgs(samples_path, samples, num_col=args.ncol)
            if args.wandb_project is not None:
                wandb.login()
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run,
                    config=config.to_dict(), 
                    resume=True
                )
                table = wandb.Table(columns=["image", "checkpoint_step", "sampler", "n_steps"])
                table.add_data(wandb.Image(pil_image_result), restored_sd["step"], "DDIM", args.n_steps)
                wandb.log({"generated images": table})
        else:
            np.savez("tmpfile.npz", arr0=samples)
            gfile.copy("tmpfile.npz", samples_path)
            time.sleep(3.0)
            gfile.remove("tmpfile.npz")

        print(f"Saved {len(samples)} samples to {samples_path}")


if __name__ == '__main__':
    app.run(main)

    