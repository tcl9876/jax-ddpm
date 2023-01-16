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
            args=D(
                image_size=256,
                resize_method='random_crop',
                image_format='jpg',
                number_sample_per_shard=10000,
                number_encodings_per_shard=5120,
                remove_keys=["image_smaller"]
            ),
        ),
        sampler='ddim',
        in_dimensions=[64,64,4],
        model=D(
            # architecture
            name='unet_iddpm',
            args=D(
                ch=128,
                emb_ch=384,  # default is ch * 4
                ch_mult=[1, 2, 3, 3],
                num_res_blocks=3,
                attn_resolutions=[32, 16, 8],
                head_dim=128,
                dropout=0.0,
                logsnr_scale_range=(-7., 7.),
                resblock_resample=False,
                out_ch=4,
                seq_width=1024,
                param_dtype='bf16',
                t5_mult=4.0,
                aesth_score_range=(2.0, 9.0),
                use_glu=False,
                use_pos_enc=True
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
            t5_model_id="google/t5-v1_1-xxl",
            clip_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            vae_id="CompVis/stable-diffusion-v1-4"
        ),
        train=D(
            # optimizer
            batch_size=256, #the PER-NODE batch size, NOT the global batch size across all nodes. if youre desired total batch size is N, set batch_size=(N//num_nodes)
            optimizer='adam',
            adam_beta2=0.99,
            max_learning_rate=1e-4,
            min_learning_rate=5e-5,
            learning_rate_warmup_steps=5000,
            weight_decay=0.01,
            ema_decay=0.9999,
            grad_clip=1.0,
            enable_update_skip=True,
            log_loss_every_steps=500,
            snapshot_freq=5000,
            log_dir="{}/logs",
            checkpoint_dirs=["{}/checkpoints_recent", "{}/checkpoints_permanent"],  
            num_checkpoints=[10, 999999],
            save_freq=[10000, 100000],
            iterations=1000001
        ),
    )
