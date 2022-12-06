import tensorflow as tf
import jax.numpy as jnp
from jax_modules.utils import numpy_iter
from tensorflow.io import gfile
import os

def read_encoded(example):
    features = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "clip_emb": tf.io.FixedLenFeature([], tf.string),
        "t5_emb": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    
    for key in features.keys():
        example[key] = tf.io.parse_tensor(example[key], tf.bfloat16)
        example[key] = tf.cast(example[key], tf.float32)
    
    example["image"] = tf.transpose(example["latents"], [2, 0, 1]) #TODO: see about removing the transpose by fixing dataset encoder builder.
    del example["latents"]
    
    for key in ["clip_emb", "t5_emb"]:
        example[key] = tf.concat([example[key], tf.zeros([77, 1024], dtype=tf.float32)], axis=0)[:77] #zero pad to 77

    return example

#decodes the image space dataset (usually result from img2dataset)
def read_pixels(example):
    features = {
        "key": tf.io.FixedLenFeature([], tf.string),
        "jpg": tf.io.FixedLenFeature([], tf.string),
        "caption": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_jpeg(example["jpg"], 3)
    caption = example['caption']
    return image, caption

#builds encodings
def make_encoders_fn(vae, clip_text_module, t5_module):
    
    def encoders_fn(processed_images, clip_inputs, t5_inputs, vae_params, clip_text_params, t5_params):
        latents = vae.apply({"params": vae_params}, processed_images, method=vae.encode)
        normalized_sample = latents.latent_dist.mean * 0.18215

        input_ids = clip_inputs["input_ids"]
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        clip_outputs = clip_text_module.apply({"params": clip_text_params}, **clip_inputs, position_ids=position_ids, output_hidden_states=True)
        clip_emb = clip_outputs.hidden_states[-2]

        t5_outputs = t5_module.apply({"params": t5_params}, **t5_inputs) #dont use penultimate as it has too high variance
        t5_emb = t5_outputs.last_hidden_state
        return normalized_sample, clip_emb, t5_emb

    return encoders_fn

def all_tfrecords(data_dir):
    tfrecords_list = []
    for dirpath, dirs, files in gfile.walk(data_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith('.tfrecord') or fname.endswith('.tfrecords'):
                tfrecords_list.append(fname)
    return tfrecords_list

def build_tfrecord_dataset(tfrecord_dir, batch_sizes, map_fn, process_index, process_count, repeating):
    filenames = all_tfrecords(tfrecord_dir)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE).map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shard(index=process_index, num_shards=process_count)
    if repeating:
        dataset = dataset.repeat()

    for batch_size in batch_sizes:
        dataset = dataset.batch(batch_size)
    dataset = numpy_iter(dataset.prefetch(tf.data.AUTOTUNE))
    return dataset