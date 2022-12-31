import jax
import jax.numpy as jnp
import functools
from absl import logging
import flax

def unreplicate(x):
    return jax.device_get(flax.jax_utils.unreplicate(x))

def shard_pytree(pytree, num_shards=8):
    def shard_tensor(inputs):
        assert inputs.shape[-1]%num_shards == 0, f"cannot evenly shard input with last dimension {inputs.shape[-1]} into {num_shards} shards"
        axis_index = jax.lax.axis_index('local_devices')
        sharded_shape = list(inputs.shape[:-1]) + [num_shards, inputs.shape[-1]//num_shards] #reshapes a [..., C] tensor to a [..., n, C//n] tensor, allowing for indexing on the last axis.
        inputs_sharded = inputs.reshape(sharded_shape)
        return inputs_sharded[..., axis_index, :]
    return jax.tree_util.tree_map(shard_tensor, pytree)

def unshard_pytree(pytree):
    def unshard_tensor(inputs):
        gather_axis = len(inputs.shape) - 1
        gathered_inputs = jax.lax.all_gather(inputs, axis_name='local_devices', axis=gather_axis)
        unsharded_shape = list(inputs.shape[:-1]) + [-1] #reshapes a [..., n, C//n] tensor to a [..., C] tensor, which was its original shape
        return gathered_inputs.reshape(unsharded_shape)
    return jax.tree_util.tree_map(unshard_tensor, pytree)

    
@functools.partial(jax.pmap, axis_name='all_devices')
def _check_synced(pytree):
	mins = jax.lax.pmin(pytree, axis_name='all_devices')
	equals = jax.tree_map(jnp.array_equal, pytree, mins)
	return jnp.all(jnp.asarray(jax.tree_leaves(equals)))

def assert_synced(pytree):
	"""Check that `pytree` is the same across all replicas.

	Args:
		pytree: the pytree to check (should be replicated)

	Raises:
		RuntimeError: if sync check failed
	"""
	equals = _check_synced(pytree)
	assert equals.shape == (jax.local_device_count(),)
	equals = all(jax.device_get(equals))  # no unreplicate
	logging.info('Sync check result: %d', equals)
	if not equals:
		raise RuntimeError('Sync check failed!')

@functools.partial(jax.pmap, axis_name='all_devices')
def _barrier(x):
	return jax.lax.psum(x, axis_name='all_devices')

def barrier():
	"""MPI-like barrier."""
	jax.device_get(_barrier(jnp.ones((jax.local_device_count(),))))

#TPU's pad if ch<128, this becomes a problem if optimizer states have ch of 512 or 768
#since its sharded version (that divides by 8) will result in 64 or 96 channels
#to avoid padding, we move from the second last to last axis so its divisible by 128 
def move_to_last_axis(pytree, mul=2):
	def movefn(x):
		xshape = x.shape
		if len(xshape) >= 2 and xshape[-1] * x.shape[-2] > 16384:
			newshape = list(xshape[:-2]) + [x.shape[-2] //mul, x.shape[-1]*mul]
			return jnp.reshape(x, newshape)
		else:
			return x
	return jax.tree_map(lambda a: movefn(a), pytree)

def move_from_last_axis(pytree, mul=2):
	def movefn(x):
		xshape = x.shape
		if len(xshape) >= 2 and xshape[-1] * x.shape[-2] > 16384:
			newshape = list(xshape[:-2]) + [x.shape[-2] *mul, x.shape[-1] //mul]
			return jnp.reshape(x, newshape)
		else:
			return x
	return jax.tree_map(lambda a: movefn(a), pytree)

def get_single_pytree_shard(pytree, i, device_count=8):
	def slicefn(x, i):
		dim = x.shape[-1] // device_count
		return x[..., dim*i:dim*i+dim]
	return jax.tree_map(lambda a: slicefn(a, i), pytree)

#this is working as intended, right? --> maybe test it
def reshape_and_transpose(x):
	shapelen = len(x.shape)
	newshape = list(x.shape[:-1]) + [8, x.shape[-1] // 8]
	x = jnp.reshape(x, newshape)
	
	#hwi8j -> 8hwij,  01234 -> 30124  shapelen 4
	#8j -> 8j  01 -> 01 shapelen 1
	#i8j -> 8ij  012 -> 102 shapelen 2
	newperm = [shapelen-1] + list(range(shapelen-1)) + [shapelen]
	x = jnp.transpose(x, newperm)
	return x         
    
def list_devices(force_no_cpu=True):
	devices = jax.devices()

	if devices[0].platform == 'cpu' and force_no_cpu:
		error_msg = \
		"""Stopping process as Jax couldn't detect TPU/GPU. 
		If on a TPU, make sure to run pip install "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
		If you did that, perhaps try doing 'pkill -9 python && sudo rm -rf /tmp/tpu_logs /tmp/libtpu_lockfile' as per https://github.com/google/jax/issues/10192"""
		raise RuntimeError(error_msg)

	if jax.process_index() == 0:
		print("DEVICES: ", devices)
	
	print(f"global device count: {len(devices)}, device count on node {jax.process_index()}: {jax.local_device_count}")
