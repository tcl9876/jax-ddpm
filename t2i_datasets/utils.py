import tensorflow as tf
import jax.numpy as jnp
from jax_modules.utils import numpy_iter
from tensorflow.io import gfile
import os
import webdataset as wds

def read_encoded(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "clip_seq": tf.io.FixedLenFeature([], tf.string),
        "t5_seq": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    
    for key in features.keys():
        example[key] = tf.io.parse_tensor(example[key], tf.bfloat16)
        example[key] = tf.cast(example[key], tf.float32)
        
    for key in ["clip_seq", "t5_seq"]:
        example[key] = tf.concat([example[key], tf.zeros([77, 1024], dtype=tf.float32)], axis=0)[:77] #zero pad to 77

    return example

#decodes the image space dataset (usually result from img2dataset stored as tfrecord).
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
def make_encoders_fn(vae, clip_vision_module, clip_text_module, t5_module):
    
    def encoders_fn(processed_images, clip_image_inputs, clip_inputs, t5_inputs, vae_params, clip_vision_params, clip_text_params, t5_params):       
        if vae_params is not None:
            latents = vae.apply({"params": vae_params}, processed_images, method=vae.encode)
            normalized_sample = latents.latent_dist.mean * 0.18215
        else:
            normalized_sample = None

        clip_image_outputs = clip_vision_module.apply({"params": clip_vision_params}, **clip_image_inputs)
        clip_image_emb = clip_image_outputs.pooler_output 

        input_ids = clip_inputs["input_ids"]
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        clip_outputs = clip_text_module.apply({"params": clip_text_params}, **clip_inputs, position_ids=position_ids, output_hidden_states=True)
        clip_emb = clip_outputs.hidden_states[-2]

        t5_outputs = t5_module.apply({"params": t5_params}, **t5_inputs) #dont use penultimate as it has too high variance
        t5_emb = t5_outputs.last_hidden_state
        return normalized_sample, clip_image_emb, clip_emb, t5_emb

    return encoders_fn


def all_tfrecords(data_dir):
    tfrecords_list = []
    for dirpath, dirs, files in gfile.walk(data_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith('.tfrecord') or fname.endswith('.tfrecords'):
                tfrecords_list.append(fname)
    return tfrecords_list

def build_tfrecord_dataset(data_dirs, batch_sizes, map_fn, process_index, process_count, repeating, sampling_probs=None):
    dirs = [s.strip() for s in data_dirs.split(",")]
    if len(dirs) > 1:
        datasets = []
        for tfrecord_dir in dirs:
            filenames = all_tfrecords(tfrecord_dir)
            datasets.append(
                tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
            )
        weights = [float(p) for p in sampling_probs.split(",")]
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, weights=weights)
    else:
        filenames = all_tfrecords(dirs[0])
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shard(index=process_index, num_shards=process_count)
    if repeating:
        dataset = dataset.repeat()

    for batch_size in batch_sizes:
        dataset = dataset.batch(batch_size)
    dataset = numpy_iter(dataset.prefetch(tf.data.AUTOTUNE))
    return dataset


def build_webdataset_image_reader(filenames, batch_sizes, process_index, process_count, has_aesthetic_column=False, repeating=False, verbose=True):
    #columns should be [jpg, txt] or [jpg, txt, aesthetic]

    local_filenames = sorted(filenames)[process_index::process_count] #shard at the file level. each node will grab their slice of the .tar files.

    if verbose:
        print(f"On process index {process_index} out of {process_count}, using the following filenames: ")
        print(local_filenames)
    
    if has_aesthetic_column:
        columns = ["jpg", "txt", "aesthetic"]
    else:
        columns = ["jpg", "txt"]

    wds_dataset = wds.WebDataset(local_filenames, handler=wds.warn_and_continue).to_tuple(*columns)

    def yielder():
        for x in wds_dataset:
            yield x
    
    #if columns includes aesthetic score, will need to append tf.Tensorspec(shape=(), dtype=tf.float32) to the output_signature
    output_signature = [tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.string)]
    if len(columns) == 3:
        output_signature += [tf.TensorSpec(shape=(), dtype=tf.float64)]
    
    dataset = tf.data.Dataset.from_generator(
        yielder,
        output_signature=tuple(output_signature)
    )

    if len(columns) == 3:
        def read_wds_tuple(jpeg, caption, aesthetic):
            return tf.io.decode_jpeg(jpeg, 3), caption, tf.cast(aesthetic, tf.float32)
    else:
        def read_wds_tuple(jpeg, caption):
            return tf.io.decode_jpeg(jpeg, 3), caption
    
    dataset = dataset.map(read_wds_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    if repeating:
        dataset = dataset.repeat()

    for batch_size in batch_sizes:
        dataset = dataset.batch(batch_size)
    dataset = numpy_iter(dataset.prefetch(tf.data.AUTOTUNE))
    return dataset
