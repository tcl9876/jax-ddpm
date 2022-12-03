import tensorflow as tf
from tensorflow.io import gfile
from tqdm import tqdm
import os
import argparse
parser = argparse.ArgumentParser(description='find total number of records in tfrecord')
parser.add_argument('tfrecord_dir', type=str, help='tfrecord dir')
parser.add_argument('--log_freq', type=int, default=1e5, help='log every how many records')
args = parser.parse_args()

files = gfile.glob(os.path.join(args.tfrecord_dir, '*.tfrecord'))
tfr_dataset =  tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
count = 0
for _ in tqdm(tfr_dataset):
    count += 1
    if count%args.log_freq == 0:
        tqdm.write(f"count at {count}")

string = f"there are a total of {count} records in {args.tfrecord_dir}"
print(string)
logfile_path = os.path.join(args.tfrecord_dir, "tfr_infolog.txt")
with gfile.GFile(logfile_path, mode='w') as f:
    f.write(string)
