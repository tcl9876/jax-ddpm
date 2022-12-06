import os
from img2dataset import download
import jax
from tensorflow.io import gfile
import argparse


# NOTE: running this downloader in multinode fashion only works within a TPU node, and is only meant to be used as such. 
# can maybe add explicit flags for process_index and process_count later
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download metadata')
    parser.add_argument('--data_dir', type=str, help='metadata location')
    parser.add_argument('--write_dir', type=str, help='output location')
    parser.add_argument('--processes_count', type=int, help='processes count', default=16)
    parser.add_argument('--thread_count', type=int, help='thread count', default=256)
    parser.add_argument('--input_format', type=str, help='input format', default='parquet')
    args = parser.parse_args()

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
                image_size=256,
                output_folder=os.path.join(args.write_dir, f"tfr_{i}"),
                output_format="tfrecord",
                input_format=args.input_format,
                url_col="URL",
                caption_col="TEXT",
                enable_wandb=False,
                min_image_size=256,
                number_sample_per_shard=10000,
                distributor="multiprocessing",
                disallowed_header_directives=["noai", "noindex"],
            )
        except BaseException as e:
            print(e)
            continue



"""
run the following within a GCP VM (can be TPU VM) with enough disk space.

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install 
git clone https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus
gsutil -m mv improved_aesthetics_6plus gs://MY_BUCKET/

"""