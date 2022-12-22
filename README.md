### DDPM training with JAX + TPU
The goal of this repository is to do scalable diffusion model training on TPUs with JAX. 

First, create a TPU-VM, e.g:
``gcloud compute tpus tpu-vm create tpu-name --zone europe-west4-a --accelerator-type v3-8 --version tpu-vm-base``

You will also need a GCS bucket with read/write access, as *all* data will be stored in GCS.

SSH into the TPU-VM via: 
``gcloud alpha compute tpus tpu-vm ssh tpu-name --zone europe-west4-a``

then clone this repository and navigate into it. Run the following:

``pip install "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ``

``pip install -r requirements.txt``

``pip install -e .``

There's 4 basic steps in the pipeline: 
1) downloading the dataset:
``python3 scripts/download_ds.py --data_dir gs://bucket_name/parquets_folder --write_dir gs://bucket_name/images_folder --image_size 256 --min_image_size 64 --processes_count 16 --thread_count 256 --caption_col CAPTION_COL``
This portion mostly relies on [img2dataset](https://github.com/rom1504/img2dataset). download_ds.py is mostly useful for downloading images using the CPUs attached to a TPU VM/Node. You can skip this step if you've already downloaded images with this tool. currently only tfrecord format is supported

2) creating image and text encodings for the dataset:
``python3 scripts/encode_ds.py --data_dir gs://bucket_name/images_folder --write_dir gs://bucket_name/data_folder``
This will run the VAE encoder to store a compressed version of the image, as well as t5 and/or clip text models to store text encodings. Everything is stored in bfloat16 format to save storage.

3) training:
``python3 train.py --config config/my_config.py --global_dir "gs://bucket_name/folder" "gs://bucket_name/data_folder"``
to track metrics, there will be a logfile that tracks the loss and gradient norms during training. wandb support is also there, but is somewhat experimental. 

4) evaluation:
``python3 scripts/evaluate.py --config config/my_config.py --checkpoint_dir gs://bucket_name/folder/checkpoints_recent --save_dir gs://bucket_name/results_folder --max_batch_size 64 --guidance_scale 3.5 --auth_token HF_AUTH_TOKEN --prompt "prompt one; prompt two; prompt three``

Essentially all changes you need to make (except on the data side) can be controlled within the specified config file 


If training on multiple nodes, you won't SSH into an individual TPU node. Instead, follow the guide for training [JAX on multiple nodes](https://cloud.google.com/tpu/docs/jax-pods). Use the same commands as above, but in the following manner:
``gcloud compute tpus tpu-vm ssh tpu-name --zone europe-west4-a --worker=all --command "python3 [my command]"``


####
Training details:
-By default, we train in mixed-precision, where a bfloat16 copy of the model is replicated across all devices, and a FP32 copy of the weights, optimizer states and EMA weight is sharded across the 8 TPU cores in each host. Gradient checkpointing is used in the attention and feedforward blocks of the self/cross attention.

Training benchmarks:
| Model Size  | Resolution | batch size on v3-8 | Throughput on v3-8 (iters/sec) | Throughput on v3-8 (images/sec) |
| ---- | ---- | ---- | ---- | ---- |
| - | - | - | - | - |
| - | - | - | - | - |


To-do's:

- [x] write basic training code
- [x] test training code on small-scale (CIFAR-10) and achieve good result
- [x] improve logging, metrics, and evaluation
- [x] make compatible with diffusers, specifically the samplers, and possibly incorporate their U-Net as well.
- [x] train latent diffusion model on imagenet 256 at f=8
- [x] implement gradient checkpointing and mixed-precision training, and optimizer state partitioning
- [x] work on dataloader for LAION dataset, that might include pre-computed text and/or image embeddings
- [x] write training script for a text-to-image latent diffusion model
- [x] support multinode training
- [ ] allow for evaluator to run inside trainloop
- [ ] add inpainting and aesthetic score conditioning
- [ ] allow for conversion to pytorch so it can be integrated into diffusers


We adapted the code from the [Progressive distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation) repo. Portions of the codebase also use modules from the diffusers library.