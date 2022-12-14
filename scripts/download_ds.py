import os
from img2dataset import download
import jax
from tensorflow.io import gfile
import argparse
import time

# NOTE: running this downloader in multinode fashion only works within a TPU node, and is only meant to be used as such. 
# can maybe add explicit flags for process_index and process_count later
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download metadata')

    parser.add_argument('--data_dir', type=str, help='metadata location')
    parser.add_argument('--write_dir', type=str, help='output location')
    parser.add_argument('--url_col', type=str, help='url col. should be "URL" normally', default="URL")
    parser.add_argument('--caption_col', type=str, help='caption col. should be "TEXT" normally, or "top_caption" for laion coco.', default="TEXT")
    parser.add_argument('--image_size', type=int, help='img size', default=None)
    parser.add_argument('--min_image_size', type=int, default=0, help='min img size')
    parser.add_argument('--processes_count', type=int, default=16)
    parser.add_argument('--thread_count', type=int, default=256)
    parser.add_argument('--input_format', type=str, default='parquet')
    parser.add_argument('--output_format', type=str, help='output format, either tfrecord or webdataset', default='tfrecord')
    parser.add_argument('--additional_columns', type=str, help='additional columns', default='["similarity","punsafe","pwatermark","AESTHETIC_SCORE"]')
    parser.add_argument('--resize_mode', type=str, default='center_crop')
    parser.add_argument('--resize_only_if_bigger', action='store_true')
    parser.add_argument('--max_aspect_ratio', type=str, default='2.5')
    
    args = parser.parse_args()
    print(args)
    time.sleep(5)

    process_index, process_count = jax.process_index(), jax.process_count()
    data_files = sorted(gfile.glob(os.path.join(args.data_dir, f'*.{args.input_format}')))
    enumerated_data = list(enumerate(data_files))
    local_data_shards = enumerated_data[process_index::process_count]

    logfile_path = os.path.join(args.data_dir, f"downloader_infolog_node{process_index}.txt") #prevent overwrite
    mode = 'a' if gfile.exists(logfile_path) else 'w'
    with gfile.GFile(logfile_path, mode=mode) as f:
        string = f"downloading data from each of these files: {data_files}"
        print(string)
        f.write(string+"\n")
    with gfile.GFile(logfile_path, mode='a') as f:
        string = f"On {process_index} out of {process_count}, the relevant shards are: {local_data_shards}"
        f.write(string+"\n")
        print(string)

    for i, data_path in local_data_shards:
        try:
            download(
                processes_count=args.processes_count,
                thread_count=args.thread_count,
                url_list=data_path,
                image_size=args.image_size,
                output_folder=os.path.join(args.write_dir, f"{args.output_format}_folder_{i}"),
                output_format=args.output_format,
                input_format=args.input_format,
                url_col=args.url_col,
                caption_col=args.caption_col,
                enable_wandb=False,
                min_image_size=args.min_image_size,
                number_sample_per_shard=10000,
                distributor="multiprocessing",
                disallowed_header_directives=["noai", "noimageai", "noindex", "noimageindex"],
                resize_mode=args.resize_mode,
                resize_only_if_bigger=args.resize_only_if_bigger,
                max_aspect_ratio=float(args.max_aspect_ratio)
            )
        except BaseException as e:
            print(e)
            continue
