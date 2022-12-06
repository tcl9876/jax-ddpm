# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Blocking checkpoint loading loops with flax/training/checkpoints.py.

Checkpointing helper functions.

Handles saving and restoring optimizer checkpoints based on step-number or
other numerical metric in filename.  Cleans up older / worse-performing
checkpoint files.
"""

import os
import re
import time
from absl import logging
from flax import serialization
from tensorflow.io import gfile
import jax
from flax.jax_utils import replicate, unreplicate


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
		r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
		r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')


def _checkpoint_path(ckpt_dir, step, prefix):
	return os.path.join(ckpt_dir, f'{prefix}{step}')


def natural_sort(file_list, signed=True):
	"""Natural sort for filenames with numerical substrings.

	Args:
		file_list: List[str]: list of paths to sort containing numerical
			substrings.
		signed: bool: if leading '-' (or '+') signs should be included in
			numerical substrings as a sign or treated as a separator.
	Returns:
		List of filenames sorted 'naturally', not lexicographically: any
		integer substrings are used to subsort numerically. e.g.
		file_1, file_10, file_2  -->  file_1, file_2, file_10
		file_0.1, file_-0.2, file_2.0  -->  file_-0.2, file_0.1, file_2.0
	"""
	float_re = SIGNED_FLOAT_RE if signed else UNSIGNED_FLOAT_RE
	def maybe_num(s):
		if float_re.match(s):
			return float(s)
		else:
			return s
	def split_keys(s):
		return [maybe_num(c) for c in float_re.split(s)]
	return sorted(file_list, key=split_keys)


def save_checkpoint(ckpt_dir,
					target,
					step,
					prefix='checkpoint_',
					keep=1,
					overwrite=False):
	"""Save a checkpoint of the model.

	Attempts to be pre-emption safe by writing to temporary before
	a final rename and cleanup of past files.

	Args:
		ckpt_dir: str: path to store checkpoint files in.
		target: serializable flax object, usually a flax optimizer.
		step: int or float: training step number or other metric number.
		prefix: str: checkpoint file name prefix.
		keep: number of past checkpoint files to keep.
		overwrite: bool: allow overwriting when writing a checkpoint.

	Returns:
		Filename of saved checkpoint.
	"""
	# Write temporary checkpoint file.
	logging.info('Saving checkpoint at step: %s', step)
	ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
	ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
	gfile.makedirs(os.path.dirname(ckpt_path))

	logging.info('Writing to temporary checkpoint location: %s', ckpt_tmp_path)
	with gfile.GFile(ckpt_tmp_path, 'wb') as fp:
		fp.write(serialization.to_bytes(target))

	# Rename once serialization and writing finished.
	gfile.rename(ckpt_tmp_path, ckpt_path, overwrite=overwrite)
	logging.info('Saved checkpoint at %s', ckpt_path)

	# Remove old checkpoint files.
	base_path = os.path.join(ckpt_dir, f'{prefix}')
	checkpoint_files = natural_sort(gfile.glob(base_path + '*'))
	if len(checkpoint_files) > keep:
		old_ckpts = checkpoint_files[:-keep]
		for path in old_ckpts:
			logging.info('Removing checkpoint at %s', path)
			gfile.remove(path)

	return ckpt_path


def latest_checkpoint_path(ckpt_dir, prefix):
	glob_path = os.path.join(ckpt_dir, f'{prefix}*')
	checkpoint_files = natural_sort(gfile.glob(glob_path))
	ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
	checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
	return checkpoint_files[-1] if checkpoint_files else None


def check_and_convert_gcs_filepath(filepath, raise_if_not_gcs=False):
	"""Utility for loading model checkpoints from GCS."""
	if filepath[:5] == 'gs://':
		local_filepath = '/temp/download/' + filepath[5:]
		if os.path.exists(local_filepath):
			print('loading from local copy of GCS file: ' + local_filepath)
		else:
			print('downloading file from GCS: ' + filepath)
			dir_index = local_filepath.rfind('/')
			os.system('mkdir -p ' + local_filepath[:dir_index])
			os.system('gsutil cp ' + filepath + ' ' + local_filepath)
		return local_filepath

	else:
		if raise_if_not_gcs:
			raise ValueError('input not recognized as a GCS path')
		return filepath


def restore_from_path(ckpt_path, target):
	ckpt_path = check_and_convert_gcs_filepath(ckpt_path)
	logging.info('Restoring checkpoint from %s', ckpt_path)
	with gfile.GFile(ckpt_path, 'rb') as fp:
		return serialization.from_bytes(target, fp.read())


def restore_checkpoint(ckpt_dir, target, step=None, prefix='checkpoint_', make_replicated=False):
	if step:
		ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
		if not gfile.exists(ckpt_path):
			raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
		else:
			print(f"Attempting to restore checkpoint from {ckpt_path}")
		return restore_from_path(ckpt_path, target)
	else:
		ckpt_path = latest_checkpoint_path(ckpt_dir, prefix=prefix)
		if ckpt_path is not None:
			print(f"Attempting to restore checkpoint from {ckpt_path}")
			with gfile.GFile(ckpt_path, 'rb') as fp:
				restored_state = serialization.from_bytes(target, fp.read())
			if make_replicated:
				restored_state = state_make_replicated(restored_state)
			return restored_state
		else:
			print(f"No checkpoint found in {ckpt_dir}. Restoring original state.")
			return target

#TODO: move these things along with stuff in train_util to a new file dist_util maybe?
def state_make_unreplicated(state):
	cpu_state = jax.device_get(state)
	return cpu_state.replace(
		step = unreplicate(cpu_state.step),
		accum_step = unreplicate(cpu_state.accum_step),
		params = unreplicate(cpu_state.params),
	)
	"""
	return state.replace(
        step = unreplicate(state.step),
        accum_step = unreplicate(state.accum_step),
        params = unreplicate(state.params),
    )
	"""
	
def state_make_replicated(state):
	return state.replace(
		step = replicate(state.step),
		accum_step = replicate(state.accum_step),
		params = replicate(state.params),
	)
