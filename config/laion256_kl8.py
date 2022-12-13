# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ml_collections


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    return D(
        seed=0,
        dataset=D(
            name='encoded_t2i', #doesnt matter what the dataset itself is made of, eg laion, coco, cc3m are all ok
            args=D(
                class_conditional=False,
                randflip=False,
                image_size=32
            ),
        ),
        sampler='ddim',
        in_dimensions=[32,32,4],
        model=D(
            # architecture
            name='unet_iddpm',
            args=D(
                ch=256,
                emb_ch=512,  # default is ch * 4
                ch_mult=[1, 1, 2, 2],
                num_res_blocks=2,
                attn_resolutions=[16, 8, 4],
                head_dim=128,
                dropout=0.0,
                logsnr_scale_range=(-7., 7.),
                resblock_resample=False,
                out_ch=4,
                seq_width=1024,
                param_dtype='bf16',
            ),
            mean_type='v', # eps, x, both, v
            logvar_type='fixed_large',
            mean_loss_weight_type='snr',  # constant, snr, snr_trunc, v_mse

            train_num_steps=1000,  
            eval_sampling_num_steps=1000,
            train_alpha_schedule=D(name='linear', beta_start=0.00085, beta_end=0.012, steps=1000),
            eval_alpha_schedule=D(name='linear', beta_start=0.00085, beta_end=0.012, steps=1000),
            tmin=5,
            eval_clip_denoised=False,
            t5_model_id="t5-3b",
            clip_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            vae_id="CompVis/stable-diffusion-v1-4"
        ),
        train=D(
            # optimizer
            batch_size=512,
            optimizer='adam',
            adam_beta2=0.999,
            learning_rate=1.2e-4,
            learning_rate_warmup_steps=5000,
            weight_decay=0.,
            ema_decay=0.9999,
            grad_clip=1.0,
            enable_update_skip=False,
            # logging
            log_loss_every_steps=500,
            log_dir="{}/logs",
            #checkpoint_every_secs=900,  # 15 minutes
            #eval_every_steps=10000,
            checkpoint_dirs=["{}/checkpoints_recent", "{}/checkpoints_permanent"],  
            num_checkpoints=[10, 999999],
            save_freq=[10000, 100000],
            iterations=1000001

        ),
    )
