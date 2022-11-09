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

"""ImageNet 64x64."""

# pylint: disable=invalid-name,line-too-long

import ml_collections


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    return D(
        seed=0,
        dataset=D(
            name='LatentImageNetEncodings',
            args=D(
                image_size=32,
                class_conditional=True,
                randflip=False
            ),
        ),
        sampler='ddim',
        model=D(
            # architecture
            name='unet_iddpm',
            args=D(
                out_ch=4,
                ch=256,
                emb_ch=1024,  # default is ch * 4
                ch_mult=[1, 2, 3],
                num_res_blocks=3,
                attn_resolutions=[8, 16, 32],
                num_heads=None,
                head_dim=128,
                dropout=0.0,
                logsnr_input_type='inv_cos',
                resblock_resample=True,
            ),
            mean_type='v',  # eps, x, both, v
            logvar_type='fixed_large',
            mean_loss_weight_type='p2',  # p2, p2_half, constant, snr, snr_trunc
            loss_scale=100, #multiply loss to potentially reduce underflow, although shouldn't be much of an issue with TPUs

            # logsnr schedule
            train_num_steps=1000, #make sure NOT to do continuous time, its not supported yet
            eval_sampling_num_steps=1000,
            train_logsnr_schedule=D(name='linear',
                                    beta_start=0.00085, beta_end=0.012, num_timesteps=1000),
            eval_logsnr_schedule=D(name='linear',
                                    beta_start=0.00085, beta_end=0.012, num_timesteps=1000),
            eval_clip_denoised=False,
        ),
        train=D(
            # optimizer
            batch_size=192,
            optimizer='adam',
            learning_rate=2e-4,
            learning_rate_warmup_steps=1000,
            weight_decay=0.001,
            ema_decay=0.9999,
            grad_clip=1.0,
            substeps=10,
            enable_update_skip=False,
            # logging
            log_loss_every_steps=100,
            log_dir="{}/logs",
            #checkpoint_every_secs=900,  # 15 minutes
            #eval_every_steps=10000,
            checkpoint_dirs=["{}/checkpoints_recent", "{}/checkpoints_permanent"],  
            num_checkpoints=[10, 999999],
            save_freq=[1000, 10000],
            iterations=800001
        ),
    )
