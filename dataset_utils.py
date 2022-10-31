import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import flax

import random as python_random
import os
import functools

def cifar_mapfn(example):
    x = tf.cast(example["image"], tf.float32)
    x = x / 127.5 - 1.
    x = tf.transpose(x, [2, 0, 1]) #flax diffusers expects NCHW, not the usual NHWC of tensorflow/jax
    y = example["label"]
    return x, y    

def load_and_shard_tf_batch(xs, global_batch_size):
    local_device_count = jax.local_device_count()
    def _prepare(x):
        return x.reshape((local_device_count, global_batch_size // local_device_count) + x.shape[1:])
    return jax.tree_map(_prepare, xs)

def tfds_to_jax_dataset(dataset, batch_size):
    dataset = tfds.as_numpy(dataset)
    dataset = map(lambda x: load_and_shard_tf_batch(x, batch_size), dataset)
    dataset = flax.jax_utils.prefetch_to_device(dataset, 1) #one is probably okay? info here: https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html says 
    return dataset

def create_dataset(dargs):
    if dargs.dataset_name == "cifar10":
        raw_dataset = tfds.load('cifar10', split='train')

    raw_dataset = raw_dataset.map(
        cifar_mapfn, #functools.partial(cifar_mapfn, dargs=dargs),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    raw_dataset = raw_dataset.shuffle(dargs.batch_size*10).batch(dargs.batch_size, drop_remainder=True)
    raw_dataset = raw_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    
    dataset = tfds_to_jax_dataset(raw_dataset, dargs.batch_size)
    return dataset
