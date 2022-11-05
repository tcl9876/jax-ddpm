### DDPM training with JAX + TPU
The goal of this repository is to do scalable diffusion model training on TPUs with JAX. 

First, create a TPU-VM, e.g:
``gcloud compute tpus tpu-vm create jaxddpm --zone us-central1-f --accelerator-type v2-8 --version tpu-vm-base``

SSH into the TPU-VM via: 
``gcloud alpha compute tpus tpu-vm ssh jaxddpm --zone us-central1-f``

then clone this repository and navigate into it. Run the following:

``pip install "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ``

`` pip install -r requirements.txt ``

``pip install -e .``

``python3 train.py --config config/expm1_config.py --global_dir "gs://bucket_name"``

For now, our training checkpoints and logs are publicly available at gs://jax-ddpm-training

To-do's (not strictly in order)

- [x] write basic training code
- [x] test training code on small-scale (CIFAR-10) and achieve good result
- [ ] make compatible with diffusers, specifically the samplers, and possibly incorporate their U-Net as well.
- [ ] improve logging, metrics, and evaluation (probably wandb so that experiment runs can be shared online)
- [ ] implement everything karras-EDM related and achieve good result
- [ ] train latent diffusion model on imagenet 256 at f=8
- [ ] implement gradient checkpointing and mixed-precision training, and any further memory reductions (especially with respect to optimizer state)
- [ ] work on dataloader for LAION dataset, that might include pre-computed text and/or image embeddings
- [ ] write training script for a text-to-image latent diffusion model

We adapted the code from the [Progressive distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation) repo. Hopefully, this code require minimal changes to scale up to more TPU chips as ```jax.pmap``` should handle most of the distribution work for us. The authors of this codebase reportedly trained models on 64 TPU-v4 chips.