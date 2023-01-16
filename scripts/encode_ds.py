import tensorflow as tf
import jax
from tensorflow.io import gfile
import numpy as np
from transformers import FlaxT5EncoderModel, T5Tokenizer, T5Config, FlaxCLIPModel, CLIPTokenizerFast, CLIPProcessor
from diffusers import FlaxStableDiffusionPipeline
import gc
import jax.numpy as jnp
import os
from jax_modules.utils import to_bf16, print_and_log, global_norm
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
flags.DEFINE_string("image_format", "tfrecord", "the format of your image data, either 'tfrecord' or 'webdataset'.")
flags.DEFINE_integer("batch_size", 512, "global batch size")
flags.mark_flags_as_required(["write_dir", "data_dir"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
if jax.process_index() == 0:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()

def get_metadata_entry(example, key):
    scores = []
    for metadata_string in example['json']:
        try:
            scores.append(
                json.loads(metadata_string.decode("utf-8"))[key]
            )
        except:
            scores.append(0.) #this is in case the metadata is missing aesthetic score for some reason. unlikely, but possible.
    return scores

def main(_):
    list_devices()
    config = args.config
    
    with jax.default_device(jax.devices("cpu")[0]):
        with jax.default_device(jax.devices("cpu")[0]):
            stable_pipeline, stable_params = FlaxStableDiffusionPipeline.from_pretrained(
                config.model.vae_id, revision="flax", dtype=jnp.bfloat16
            )
            vae, vae_params = stable_pipeline.vae, stable_params["vae"]

            for key in ['unet', 'scheduler', 'safety_checker', 'text_encoder']:
                jax.device_get(stable_params[key])
                del stable_params[key] #save memory by removing everything but vae
        
        clip_model = FlaxCLIPModel.from_pretrained(config.model.clip_model_id, from_pt=True, dtype=jnp.bfloat16) #bfloat has no effect?
        clip_module, clip_params = clip_model.module, clip_model.params
        clip_params = to_bf16(clip_params)
        clip_tokenizer = CLIPTokenizerFast.from_pretrained(config.model.clip_model_id)
        clip_vision_processor = CLIPProcessor.from_pretrained(args.config.model.clip_model_id)

        t5_config = T5Config.from_pretrained(config.model.t5_model_id)
        t5_model = FlaxT5EncoderModel.from_pretrained("/home/royal", config=t5_config, dtype=jnp.bfloat16) #MUST FIX THIS!
        t5_module, t5_params = t5_model.module, t5_model.params
        print(global_norm(t5_params))
        t5_params = to_bf16(t5_params)
        t5_tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_id, model_max_length=77)

    print("created objects")
    gc.collect()
    vae_params = jax.device_put_replicated(vae_params, jax.local_devices())
    clip_params = jax.device_put_replicated(clip_params, jax.local_devices())
    t5_params = jax.device_put_replicated(t5_params, jax.local_devices())
    print("# t5 params:", sum([x.size for x in jax.tree_leaves(t5_params)]))

    encoders_fn = make_encoders_fn(vae, clip_module, t5_module)
    encoders_fn = jax.pmap(encoders_fn)

    logfile_path = os.path.join(args.write_dir, f'encode_ds_logs_node_{jax.process_index()}.txt')
    if not gfile.exists(logfile_path):
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
        for key in "AESTHETIC_SCORE", "height", "width":
            printl(str(get_metadata_entry(example, key)[:16]))
        
        break

    num_records = 0
    record_keys =  ["image", "image_smaller", "clip_emb", "t5_emb", "clip_image_emb", "aesth_score", "height", "width"]
    all_datas = {}
    for key in record_keys:
        all_datas[key] = []

    for example in full_image_dataset:
        image_pixels = example[dargs.image_format]
        captions = example["txt"]
        captions = [c.decode('utf-8') for c in captions]
        all_datas["aesth_score"] += get_metadata_entry(example, "AESTHETIC_SCORE")
        all_datas["height"] += get_metadata_entry(example, "height")
        all_datas["width"] += get_metadata_entry(example, "width")

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
        undo_reshape = lambda x: np.array(x.reshape(-1, *x.shape[2:]))

        processed_images = image_pixels / 127.5 - 1.
        processed_images_smaller = tf.image.resize(processed_images, [256, 256]).numpy() #TODO: remove magic number and wrap in an if statement
        processed_images = reshaper(processed_images.transpose(0, 3, 1, 2))
        processed_images_smaller = reshaper(processed_images_smaller.transpose(0, 3, 1, 2))

        clip_inputs = jax.tree_map(reshaper, clip_inputs)
        clip_inputs['pixel_values'] = jnp.transpose(clip_inputs['pixel_values'], (0, 1, 3, 4, 2)) #should it be channels last?
        t5_inputs = jax.tree_map(reshaper, t5_inputs)
        
        images, images_smaller, clip_image_emb, clip_emb, t5_emb = encoders_fn(processed_images, processed_images_smaller, clip_inputs, t5_inputs, vae_params, clip_params, t5_params)
        
        clip_image_emb, clip_emb, t5_emb = undo_reshape(clip_image_emb), undo_reshape(clip_emb), undo_reshape(t5_emb)
        clip_mask, t5_mask = undo_reshape(clip_inputs["attention_mask"]), undo_reshape(t5_inputs["attention_mask"])
        
        images = list(np.transpose(undo_reshape(images), [0, 2, 3, 1]))
        images_smaller = list(np.transpose(undo_reshape(images_smaller), [0, 2, 3, 1]))
            
        for i in range(len(images)):
            images[i] = np.transpose(images[i], [2, 0, 1])
            images_smaller[i] = np.transpose(images_smaller[i], [2, 0, 1])
            assert list(images[i].shape) == list(config.in_dimensions), images[i].shape
            all_datas["image"].append(images[i])
            all_datas["image_smaller"].append(images_smaller[i])
            all_datas["clip_image_emb"].append(clip_image_emb[i])

            clip_maskeds, t5_maskeds = np.where(clip_mask[i] == 0)[0], np.where(t5_mask[i] == 0)[0]
            if len(clip_maskeds) == 0:
                all_datas["clip_emb"].append(clip_emb[i])
            else:
                first_masked = clip_maskeds[0]
                all_datas["clip_emb"].append(clip_emb[i][:first_masked])
            if len(t5_maskeds) == 0:
                all_datas["t5_emb"].append(t5_emb[i])
            else:
                first_masked = t5_maskeds[0]
                all_datas["t5_emb"].append(t5_emb[i][:first_masked])
        
        num_imgs = len(all_datas["image"])
        if num_imgs >= dargs.number_encodings_per_shard:
            example_path = os.path.join(args.write_dir, f"example{num_records}_{jax.process_index()}.tfrecord")
            for key in record_keys:
                assert len(all_datas[key]) == num_imgs

            with tf.io.TFRecordWriter(example_path) as file_writer:
                def tofeature(x):
                    #store as bf16 to save storage
                    x = tf.io.serialize_tensor(tf.cast(x, tf.bfloat16)).numpy()
                    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

                for i in range(num_imgs):
                    if i == 0 and num_records == 0 and jax.process_index() == 0:
                        for key in record_keys:
                            try:
                                printl(f"shape for {key} is {all_datas[key][i].shape}")
                            except:
                                printl(f"value for {key} is {all_datas[key][i]}")
                    
                    features_for_example = {}
                    for key in record_keys:
                        features_for_example[key] = tofeature(all_datas[key][i])
                    
                    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))
                    file_writer.write(example_proto.SerializeToString())

            printl(f"wrote tfrecord file to {example_path} on node {jax.process_index()}, all images had len {num_imgs}") 
           
            all_datas = {}
            for key in record_keys:
                all_datas[key] = []
            num_records += 1

    

if __name__ == '__main__':
    app.run(main)