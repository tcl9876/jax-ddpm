import faulthandler

faulthandler.enable()

import tensorflow as tf
import jax
from tensorflow.io import gfile
import numpy as np
from transformers import FlaxT5EncoderModel, T5TokenizerFast, FlaxCLIPTextModel, CLIPTokenizerFast
from diffusers import FlaxStableDiffusionPipeline
import gc
import jax.numpy as jnp
import os
from jax_modules.utils import numpy_iter, to_bf16, list_devices
from t2i_datasets.utils import make_encoders_fn, read_pixels, build_tfrecord_dataset
from absl import app, flags
import logging
import transformers
from functools import partial

args = flags.FLAGS
flags.DEFINE_string("write_dir", None, "the global directory you will save the encodings into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored")
flags.DEFINE_string("clip_model_id", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "clip config (from HF transformers)")
flags.DEFINE_string("t5_model_id", "t5-3b", "t5 config (from HF transformers)") #SWITCH TO 11B!
flags.DEFINE_integer("batch_size", 512, "global batch size ")
flags.mark_flags_as_required(["write_dir", "data_dir"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
if jax.process_index() == 0:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()

def print_and_log(*args, logfile_path):
    print(*args)
    for a in args:
        with gfile.GFile(logfile_path, mode='a') as f:
            f.write(str(a))

    with gfile.GFile(logfile_path, mode='a') as f:
        f.write('\n')
        
def main(_):
    list_devices()

    stable_pipeline, stable_params = FlaxStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="flax", dtype=jnp.bfloat16
    )
    vae, vae_params = stable_pipeline.vae, stable_params["vae"]
    for key in ['unet', 'scheduler', 'safety_checker', 'text_encoder']:
        del stable_params[key] #save memory by removing everything but vae
    gc.collect()
    vae_params = jax.device_put_replicated(vae_params, jax.local_devices()) #previously failed here

    clip_text_model = FlaxCLIPTextModel.from_pretrained(args.clip_model_id, from_pt=True, dtype=jnp.bfloat16) #bfloat has no effect?
    clip_text_module, clip_text_params = clip_text_model.module, clip_text_model.params
    clip_text_params = to_bf16(clip_text_params)
    clip_text_params = jax.device_put_replicated(clip_text_params, jax.local_devices())
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_model_id)

    with jax.default_device(jax.devices("cpu")[0]):
        t5_model = FlaxT5EncoderModel.from_pretrained(args.t5_model_id, from_pt=True, dtype=jnp.bfloat16) #bfloat has no effect?
        
    t5_module, t5_params = t5_model.module, t5_model.params
    t5_params = to_bf16(t5_params)
    t5_params = jax.device_put_replicated(t5_params, jax.local_devices())
    t5_tokenizer = T5TokenizerFast.from_pretrained(args.t5_model_id, model_max_length=77)

    encoders_fn = make_encoders_fn(vae, clip_text_module, t5_module)
    encoders_fn = jax.pmap(encoders_fn)
    full_image_dataset = build_tfrecord_dataset(args.data_dir, batch_sizes=[args.batch_size], map_fn=read_pixels, process_index=jax.process_index(), process_count=jax.process_count(), repeating=False)

    logfile_path = os.path.join(args.write_dir, 'logfile.txt')
    if not gfile.exists(logfile_path) and jax.process_index() == 0:
        tf.io.write_file(logfile_path, "")
    printl = partial(print_and_log, logfile_path=logfile_path)

    for image_pixels, captions in full_image_dataset:
        _captions = [c.decode('utf-8') for c in captions]
        printl(f"process index {jax.process_index()}, num captions: {len(_captions)}")
        printl(_captions[:4])
        break

    TFRECORD_MIN_EXAMPLES = 5000
    
    all_latents, all_clip_embs, all_t5_embs = [], [], []

    num_records = 0
    for image_pixels, captions in full_image_dataset:
        captions = [c.decode('utf-8') for c in captions]
        processed_images = image_pixels / 127.5 - 1.
        clip_inputs = dict(clip_tokenizer(captions, truncation=True, return_tensors="np", max_length=77, padding='max_length'))
        t5_inputs = dict(t5_tokenizer(captions, truncation=True, return_tensors="np", padding='max_length'))

        #reshape for pmap
        n = jax.local_device_count()
        assert args.batch_size%n == 0
        reshaper = lambda x: x.reshape(n, args.batch_size//n, *x.shape[1:])
        processed_images = reshaper(processed_images.transpose(0, 3, 1, 2))
        clip_inputs = jax.tree_map(reshaper, clip_inputs)
        t5_inputs = jax.tree_map(reshaper, t5_inputs)

        latents, clip_emb, t5_emb = encoders_fn(processed_images, clip_inputs, t5_inputs, vae_params, clip_text_params, t5_params)
        undo_reshape = lambda x: np.array(x.reshape(-1, *x.shape[2:]))
        latents, clip_emb, t5_emb = undo_reshape(latents), undo_reshape(clip_emb), undo_reshape(t5_emb)
        clip_mask, t5_mask = undo_reshape(clip_inputs["attention_mask"]), undo_reshape(t5_inputs["attention_mask"])
        latents = np.transpose(latents, [0, 2, 3, 1])

        for i in range(len(latents)):
            all_latents.append(latents[i])
            clip_maskeds, t5_maskeds = np.where(clip_mask[i] == 0)[0], np.where(t5_mask[i] == 0)[0]
            if len(clip_maskeds) == 0:
                all_clip_embs.append(clip_emb[i])
            else:
                first_masked = clip_maskeds[0]
                all_clip_embs.append(clip_emb[i][:first_masked])
            if len(t5_maskeds) == 0:
                all_t5_embs.append(t5_emb[i])
            else:
                first_masked = t5_maskeds[0]
                all_t5_embs.append(t5_emb[i][:first_masked])
            
        if len(all_latents) >= TFRECORD_MIN_EXAMPLES:
            example_path = os.path.join(args.write_dir, f"example{num_records}_{jax.process_index()}.tfrecord")

            with tf.io.TFRecordWriter(example_path) as file_writer:
                for i in range(len(all_latents)):
                    def tofeature(x):
                        #store as bf16 to save storage
                        x = tf.io.serialize_tensor(tf.cast(x, tf.bfloat16)).numpy()
                        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

                    features_for_example = {
                        'latents': tofeature(all_latents[i]), 
                        'clip_emb': tofeature(all_clip_embs[i]), 
                        't5_emb': tofeature(all_t5_embs[i]),
                    }
                    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))
                    file_writer.write(example_proto.SerializeToString())

            printl(f"wrote tfrecord file to {example_path} on node {jax.process_index()}, all latents had len {len(all_latents)}")
            all_latents, all_clip_embs, all_t5_embs = [], [], []
            num_records += 1

    

if __name__ == '__main__':
    app.run(main)