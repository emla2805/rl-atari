import numpy as np
import tensorflow as tf
import random
import gym

from atari_wrappers import NoopResetEnv, MaxAndSkipEnv, Monitor, wrap_deepmind
from vec_env import SubprocVecEnv


def make_atari_env(env_id, num_env, seed):
    """Create a wrapped SubprocVecEnv for Atari."""
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            assert 'NoFrameskip' in env.spec.id
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env.seed(seed + rank)
            env = Monitor(env)
            return wrap_deepmind(env)
        return _thunk

    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i) for i in range(num_env)])


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
