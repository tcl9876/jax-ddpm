### DDPM training with JAX + TPU
The goal of this repository is to do scalable diffusion model training on TPUs with JAX. 

To get started, first SSH into the TPU-VM, then clone this repository and navigate into it. Then:

``pip install "jax[tpu]==0.3.17" -f [https://storage.googleapis.com/jax-releases/libtpu_releases.html](https://storage.googleapis.com/jax-releases/libtpu_releases.html "https://storage.googleapis.com/jax-releases/libtpu_releases.html") ``
`` pip install -r requirements.txt ``
``python3 train.py --config config/expm1_config.py --global_dir "gs://bucket_name"``
