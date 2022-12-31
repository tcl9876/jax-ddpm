import tensorflow as tf
import jax
from tensorflow.io import gfile
import numpy as np
from transformers import FlaxT5EncoderModel, T5TokenizerFast, FlaxCLIPModel, CLIPTokenizerFast, FlaxCLIPVisionModel, CLIPProcessor
from diffusers import FlaxStableDiffusionPipeline
import gc
import jax.numpy as jnp
import os
from jax_modules.utils import to_bf16, print_and_log
from jax_modules.dist_util import list_devices
from t2i_datasets.utils import make_encoders_fn, read_pixels, build_tfrecord_dataset, build_webdataset_image_reader
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import transformers
from functools import partial
import json

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("write_dir", None, "the global directory you will save the encodings into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored")
flags.DEFINE_string("image_format", "frecord", "the format of your image data, either 'tfrecord' or 'webdataset'.")
flags.DEFINE_integer("batch_size", 512, "global batch size")
flags.mark_flags_as_required(["write_dir", "data_dir"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
if jax.process_index() == 0:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()

def get_aesthetic_scores(example):
    aesth_scores = []
    for metadata_string in example['json']:
        try:
            aesth_scores.append(
                json.loads(metadata_string.decode("utf-8"))["AESTHETIC_SCORE"]
            )
        except:
            aesth_scores.append(0.) #this is in case the metadata is missing aesthetic score for some reason. unlikely, but possible.
    return aesth_scores

def main(_):
    list_devices()
    
    config = args.config
    use_vae = config.model.vae_id is not None
    if use_vae:
        stable_pipeline, stable_params = FlaxStableDiffusionPipeline.from_pretrained(
            config.model.vae_id, revision="flax", dtype=jnp.bfloat16
        )
        vae, vae_params = stable_pipeline.vae, stable_params["vae"]
        vae_params = jax.device_put_replicated(vae_params, jax.local_devices())

        for key in ['unet', 'scheduler', 'safety_checker', 'text_encoder']:
            jax.device_get(stable_params[key])
            del stable_params[key] #save memory by removing everything but vae
        gc.collect()
    else:
        vae = None
    
    
    with jax.default_device(jax.devices("cpu")[0]):
        clip_model = FlaxCLIPModel.from_pretrained(config.model.clip_model_id, from_pt=True, dtype=jnp.bfloat16) #bfloat has no effect?
        clip_module, clip_params = clip_model.module, clip_model.params
        clip_params = to_bf16(clip_params)
        clip_tokenizer = CLIPTokenizerFast.from_pretrained(config.model.clip_model_id)
        clip_vision_processor = CLIPProcessor.from_pretrained(args.config.model.clip_model_id)

        t5_model = FlaxT5EncoderModel.from_pretrained(config.model.t5_model_id, from_pt=True, dtype=jnp.bfloat16) #bfloat has no effect?
        t5_module, t5_params = t5_model.module, t5_model.params
        t5_params = to_bf16(t5_params)
        t5_tokenizer = T5TokenizerFast.from_pretrained(config.model.t5_model_id, model_max_length=77)

    clip_params = jax.device_put_replicated(clip_params, jax.local_devices())
    t5_params = jax.device_put_replicated(t5_params, jax.local_devices())
    print("# t5 params:", sum([x.size for x in jax.tree_leaves(t5_params)]))

    encoders_fn = make_encoders_fn(vae, clip_module, t5_module)
    encoders_fn = jax.pmap(encoders_fn)

    logfile_path = os.path.join(args.write_dir, 'logfile.txt')
    if not gfile.exists(logfile_path) and jax.process_index() == 0:
        tf.io.write_file(logfile_path, "")
    printl = partial(print_and_log, logfile_path=logfile_path)

    dargs = config.dataset.args
    map_fn = partial(read_pixels, image_size=dargs.image_size, resize_method=dargs.resize_method, image_format=dargs.image_format)
    if args.image_format == 'tfrecord':
        full_image_dataset = build_tfrecord_dataset(args.data_dir, batch_sizes=[args.batch_size], map_fn=map_fn, process_index=jax.process_index(), process_count=jax.process_count(), repeating=False, print_func=printl)
    else:
        full_image_dataset = build_webdataset_image_reader(args.data_dir, batch_sizes=[args.batch_size], map_fn=map_fn, process_index=jax.process_index(), process_count=jax.process_count(), image_format=dargs.image_format, repeating=False, print_func=printl)

    for example in full_image_dataset:
        _captions = [c.decode('utf-8') for c in example["txt"]]
        printl(f"process index {jax.process_index()}, num captions: {len(_captions)}")
        printl(_captions[:4])
        printl(example[dargs.image_format].shape, example[dargs.image_format].dtype)
        printl(str(get_aesthetic_scores(example)[:16]))
        break

    TFRECORD_MIN_EXAMPLES = 5000
    num_records = 0
    all_images, all_clip_image_embs, all_clip_embs, all_t5_embs, all_aesth_scores = [], [], [], [], []

    for example in full_image_dataset:
        image_pixels = example[dargs.image_format]
        captions = example["txt"]
        captions = [c.decode('utf-8') for c in captions]
        all_aesth_scores += get_aesthetic_scores(example)

        image_pixels_T = image_pixels.transpose(0, 3, 1, 2)
        clip_inputs = dict(clip_tokenizer(captions, truncation=True, return_tensors="np", max_length=77, padding='max_length'))
        clip_inputs.update(
            dict(clip_vision_processor(images=list(image_pixels_T), return_tensors="np"))
        )
        t5_inputs = dict(t5_tokenizer(captions, truncation=True, return_tensors="np", padding='max_length'))

        #reshape for pmap
        n = jax.local_device_count()
        assert args.batch_size%n == 0
        reshaper = lambda x: x.reshape(n, args.batch_size//n, *x.shape[1:])
        if use_vae:
            processed_images = image_pixels / 127.5 - 1.
            processed_images = reshaper(processed_images.transpose(0, 3, 1, 2))
        else:
            vae_params, processed_images = None, None

        clip_inputs = jax.tree_map(reshaper, clip_inputs)
        clip_inputs['pixel_values'] = jnp.transpose(clip_inputs['pixel_values'], (0, 1, 3, 4, 2)) #should it be channels last?
        t5_inputs = jax.tree_map(reshaper, t5_inputs)
        
        images, clip_image_emb, clip_emb, t5_emb = encoders_fn(processed_images, clip_inputs, t5_inputs, vae_params, clip_params, t5_params)
        undo_reshape = lambda x: np.array(x.reshape(-1, *x.shape[2:]))
        clip_image_emb, clip_emb, t5_emb = undo_reshape(clip_image_emb), undo_reshape(clip_emb), undo_reshape(t5_emb)
        clip_mask, t5_mask = undo_reshape(clip_inputs["attention_mask"]), undo_reshape(t5_inputs["attention_mask"])
        if use_vae:
            images = list(np.transpose(undo_reshape(images), [0, 2, 3, 1]))
        else:
            images = list(image_pixels)
            
        for i in range(len(images)):
            images[i] = np.transpose(images[i], [2, 0, 1])
            assert list(images[i].shape) == list(config.in_dimensions), images[i].shape
            all_images.append(images[i])
            all_clip_image_embs.append(clip_image_emb[i])

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
            
        if len(all_images) >= TFRECORD_MIN_EXAMPLES:
            example_path = os.path.join(args.write_dir, f"example{num_records}_{jax.process_index()}.tfrecord")
            assert len(all_images) == len(all_clip_embs) == len(all_t5_embs) == len(all_clip_image_embs) == len(all_aesth_scores)

            with tf.io.TFRecordWriter(example_path) as file_writer:
                for i in range(len(all_images)):
                    def tofeature(x):
                        #store as bf16 to save storage
                        x = tf.io.serialize_tensor(tf.cast(x, tf.bfloat16)).numpy()
                        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
                    
                    if i == 0 and num_records == 0 and jax.process_index() == 0:
                        printl(f"image shape {all_images[i].shape}, clip_emb shape {all_clip_embs[i].shape}, t5_emb shape {all_t5_embs[i].shape}, clip_image_emb shape {all_clip_image_embs[i].shape}, aesth_score: {all_aesth_scores[i]}")

                    features_for_example = {
                        'image': tofeature(all_images[i]), 
                        'clip_emb': tofeature(all_clip_embs[i]), 
                        't5_emb': tofeature(all_t5_embs[i]),
                        'clip_image_emb': tofeature(all_clip_image_embs[i]),
                        'aesth_score': tofeature(all_aesth_scores[i])
                    }

                    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))
                    file_writer.write(example_proto.SerializeToString())

            printl(f"wrote tfrecord file to {example_path} on node {jax.process_index()}, all images had len {len(all_images)}")
            all_images, all_clip_image_embs, all_clip_embs, all_t5_embs, all_aesth_scores = [], [], [], [], []
            num_records += 1

    

if __name__ == '__main__':
    app.run(main)