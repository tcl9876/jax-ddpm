### DDPM training with JAX + TPU
The goal of this repository is to do scalable diffusion model training on TPUs with JAX. 

To get started, first SSH into the TPU-VM, then clone this repository and navigate into it. Then:

``pip install "jax[tpu]==0.3.17" -f [https://storage.googleapis.com/jax-releases/libtpu_releases.html](https://storage.googleapis.com/jax-releases/libtpu_releases.html "https://storage.googleapis.com/jax-releases/libtpu_releases.html") ``

`` pip install -r requirements.txt ``

``python3 train.py --config config/expm1_config.py --global_dir "gs://bucket_name"``

Roadmap:

- [x] write basic training code
- [ ] test training code on small-scale (CIFAR-10) and achieve good result
- [ ] improve logging / metrics (e.g. tensorboard) and inference/evaluation code
- [ ] implement and test v-prediction, and training re-weighting (probably either P2 or Karras). Ideally achieve better result
- [ ] write training script for a text-to-image latent diffusion model, with text embeddings from Huggingface CLIP models
- [ ] implement gradient checkpointing and mixed-precision training (possibly including optimizer state)

We use Flax-related things from diffusers for parts of the codebase, so it should be relatively simple to integrate stable diffusion into the training code.
