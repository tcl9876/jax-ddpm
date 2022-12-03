import os
from img2dataset import download

if __name__ == '__main__':
    data_files = sorted(os.listdir("../improved_aesthetics_6plus/data"))
    print(f"downloading data from each of these files: {data_files}")
    for i, data_path in enumerate(data_files):
        try:
            download(
                processes_count=16,
                thread_count=256,
                url_list=os.path.join("../improved_aesthetics_6plus/data", data_path),
                image_size=256,
                output_folder=f"gs://jax-ddpm-eu/aesth_data/tfr_{i}",
                output_format="tfrecord",
                input_format="parquet",
                url_col="URL",
                caption_col="TEXT",
                enable_wandb=False,
                number_sample_per_shard=10000,
                distributor="multiprocessing",
                disallowed_header_directives=["noai", "noindex"],
            )
        except BaseException as e:
            print(e)
            continue



"""
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install 
git clone https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus
pip install tensorflow_io img2dataset gcsfs

img2dataset --url_list {} --input_format parquet --url_col URL --caption_col TEXT 
--output_format tfrecord  --output_folder {} --processes_count 8 --thread_count 64 --image_size 256 --enable_wandb False".format(urls_path, output_folder)
"""