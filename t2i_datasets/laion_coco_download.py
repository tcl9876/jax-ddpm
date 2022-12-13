import os

if __name__ == '__main__':
    for i in range(128):
        i = str(i).zfill(3)
        os.system(f"wget https://huggingface.co/datasets/laion/laion-coco/resolve/main/part-00{i}-2256f782-126f-4dc6-b9c6-e6757637749d-c000.snappy.parquet")
        os.system(f"gsutil -m cp part-00{i}-2256f782-126f-4dc6-b9c6-e6757637749d-c000.snappy.parquet gs://jax-ddpm-eu/laion-coco-records/")
        os.system(f"rm -rf part-00{i}-2256f782-126f-4dc6-b9c6-e6757637749d-c000.snappy.parquet")