import tensorflow as tf
import jax.numpy as jnp
from jax_modules.utils import numpy_iter
from tensorflow.io import gfile
import os
import webdataset as wds
from functools import partial


def read_encoded(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "clip_emb": tf.io.FixedLenFeature([], tf.string),
        "t5_emb": tf.io.FixedLenFeature([], tf.string),
        "clip_image_emb": tf.io.FixedLenFeature([], tf.string),
        "aesth_score": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    
    for key in features.keys():
        example[key] = tf.io.parse_tensor(example[key], tf.bfloat16)
        example[key] = tf.cast(example[key], tf.float32)
        
    for key in ["clip_emb", "t5_emb"]:
        example[key] = tf.concat([example[key], tf.zeros([77, 1024], dtype=tf.float32)], axis=0)[:77] #zero pad to 77
    
    return example

def resize_image(image, image_size, resize_mode, resize_method='bilinear'):
    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    if resize_mode in ['center_crop', 'random_crop']:
        divide_by_dim = tf.minimum(h, w)
    else:
        divide_by_dim = tf.maximum(h, w)

    new_size = [tf.math.round(image_size * h / divide_by_dim), tf.math.round(image_size * w / divide_by_dim)]
    new_size = tf.convert_to_tensor(new_size, dtype=tf.int32)
    image = tf.image.resize(image, new_size, method=resize_method)
    image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.uint8)

    if resize_mode == 'center_crop':
        crop = tf.reduce_min(new_size)
        image = image[(new_size[0] - crop) // 2 : (new_size[0] + crop) // 2, (new_size[1] - crop) // 2 : (new_size[1] + crop) // 2]
    elif resize_mode == 'random_crop':
        image = tf.image.random_crop(image, [image_size, image_size, 3])
    else:
        raise RuntimeError(f"resize_mode must be either 'center_crop' or 'random_crop', got {resize_mode}")
    
    return image

def read_pixels(example, image_size, resize_method, image_format):
    if not isinstance(example, dict):
        features = {}
        columns = [image_format, "txt", "json"]
        for key in columns:
            features[key] = tf.io.FixedLenFeature([], tf.string)
        example = tf.io.parse_single_example(example, features)

    if image_format == 'jpg':
        image = tf.io.decode_jpeg(example[image_format], 3)
    elif image_format == 'png':
        image = tf.io.decode_png(example[image_format], 3)
    else:
        raise NotImplementedError(f"specified image format {image_format} not supported, must be either 'jpg' or 'png'.")

    if list(image.shape) != [image_size, image_size, 3]:
        image = resize_image(image, image_size, resize_method)

    example[image_format] = image
    return example

#builds encodings
def make_encoders_fn(vae, clip_module, t5_module):
    
    def encoders_fn(processed_images, clip_inputs, t5_inputs, vae_params, clip_params, t5_params):       
        if vae_params is not None:
            latents = vae.apply({"params": vae_params}, processed_images, method=vae.encode)
            normalized_sample = latents.latent_dist.mean * 0.18215
        else:
            normalized_sample = None

        input_ids = clip_inputs["input_ids"]
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        clip_outputs = clip_module.apply({"params": clip_params}, **clip_inputs, position_ids=position_ids, output_hidden_states=True)
        clip_image_emb = clip_outputs.image_embeds
        clip_emb = clip_outputs.text_model_output.hidden_states[-2]

        t5_outputs = t5_module.apply({"params": t5_params}, **t5_inputs) #dont use penultimate as it has too high variance
        t5_emb = t5_outputs.last_hidden_state
        return normalized_sample, clip_image_emb, clip_emb, t5_emb

    return encoders_fn

def all_files_with_ext(data_dir, ext):
    tfrecords_list = []
    for dirpath, dirs, files in gfile.walk(data_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(ext):
                tfrecords_list.append(fname)
                
    return sorted(tfrecords_list)

all_tfrecords = partial(all_files_with_ext, ext=".tfrecord")
all_wds_tars = partial(all_files_with_ext, ext=".tar")

def to_mapped_batched_numpy_iterator(dataset, map_fn, batch_sizes, repeating):
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if repeating:
        dataset = dataset.repeat()

    for batch_size in batch_sizes:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = numpy_iter(dataset.prefetch(tf.data.AUTOTUNE))
    return dataset

def print_local_filenames(print_func, local_filenames, process_index, process_count):
    if print_func is not None:
        print_func(f"On process index {process_index} out of {process_count}, using the following filenames: " + "\n")
        print_func(str(local_filenames) + "\n")

def build_tfrecord_dataset(data_dirs, batch_sizes, map_fn, process_index, process_count, repeating=False, sampling_probs=None, print_func=None):
    dirs = [s.strip() for s in data_dirs.split(",")]

    if len(dirs) > 1:
        datasets = []
        for tfrecord_dir in dirs:
            local_filenames = all_tfrecords(tfrecord_dir)[process_index::process_count]
            print_local_filenames(print_func, local_filenames, process_index, process_count)
            datasets.append(
                tf.data.TFRecordDataset(local_filenames, num_parallel_reads=tf.data.AUTOTUNE)
            )

        weights = [float(p) for p in sampling_probs.split(",")]
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, weights=weights)
    else:
        local_filenames = all_tfrecords(dirs[0])[process_index::process_count]
        print_local_filenames(print_func, local_filenames, process_index, process_count)
        dataset = tf.data.TFRecordDataset(local_filenames, num_parallel_reads=tf.data.AUTOTUNE)

    return to_mapped_batched_numpy_iterator(dataset, map_fn=map_fn, batch_sizes=batch_sizes, repeating=repeating)

def build_webdataset_image_reader(data_dirs, batch_sizes, map_fn, process_index, process_count, image_format, repeating=False, print_func=None):

    dirs = [s.strip() for s in data_dirs.split(",")]
    assert len(dirs) == 1, "Reading from multiple directories is not supported for webdataset"
    
    local_filenames = all_wds_tars(dirs[0])[process_index::process_count] #shard at the file level. each node will grab their slice of the .tar files.
    print_local_filenames(print_func, local_filenames, process_index, process_count)
    
    columns = [image_format, "txt", "json"]
    wds_dataset = wds.WebDataset(local_filenames, handler=wds.warn_and_continue).to_tuple(*columns)

    def yielder():
        for x in wds_dataset:
            yield x
    
    output_signature = [tf.TensorSpec(shape=(), dtype=tf.string) for _ in columns]
    
    dataset = tf.data.Dataset.from_generator(
        yielder,
        output_signature=tuple(output_signature)
    )

    def read_wds_tuple(*args):
        example = {}
        for a, col in zip(args, columns):
            example[col] = a
        
        return map_fn(example)
    
    return to_mapped_batched_numpy_iterator(dataset, map_fn=read_wds_tuple, batch_sizes=batch_sizes, repeating=repeating)
