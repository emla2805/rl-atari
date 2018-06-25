"""
Vanilla Policy Gradient Reinforcement Learning using Tensorflow
"""

import numpy as np
import tensorflow as tf
import time
import os
from collections import deque
from argparse import ArgumentParser
from common.utils import make_atari_env
from common.vec_env import VecFrameStack

tf.logging.set_verbosity(tf.logging.INFO)

EP_INFO_BUFF = deque(maxlen=100)
MEMORY_CAPACITY = 100_000

# MEMORY stores tuples:
# (observation, action, reward)
MEMORY = deque(maxlen=MEMORY_CAPACITY)


def gen():
    for m in list(MEMORY):
        yield m


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class CNNModel(tf.keras.Model):
    def __init__(self, ac_space):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(515)
        self.logits = tf.keras.layers.Dense(ac_space.n)

    def call(self, input):
        input_norm = tf.cast(input, tf.float32) / 255.
        result = self.conv1(input_norm)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.flatten(result)
        result = self.dense(result)
        logits = self.logits(result)
        return logits


class PGAgent(object):
    def __init__(self, model, ob_space, gamma, decay_steps, lr, batch_size):
        self.model = model
        self.gamma = gamma

        with tf.variable_scope('act'):
            self.act_obs = tf.placeholder(shape=(None,) + ob_space.shape, dtype=ob_space.dtype, name='observations')
            self.act_logits = self.model(self.act_obs)
            self.act_actions = tf.squeeze(tf.multinomial(logits=self.act_logits, num_samples=1))

        with tf.variable_scope('dataset'):
            dataset = tf.data.Dataset.from_generator(
                generator=gen,
                output_types=(tf.int32, tf.int32, tf.float32),
                output_shapes=(tf.TensorShape(ob_space.shape), tf.TensorShape([]), tf.TensorShape([])))
            dataset = dataset.shuffle(MEMORY_CAPACITY).repeat(None).batch(batch_size)
            self.iterator = dataset.make_one_shot_iterator()

            obs, actions, rewards = self.iterator.get_next()

            # Normalize rewards
            rew_mean, rew_var = tf.nn.moments(rewards, axes=[0])
            rewards = (rewards - rew_mean) / (tf.sqrt(rew_var) + 1e-8)

        logits = self.model(obs)

        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('learning_params'):
            learning_rate = tf.train.polynomial_decay(lr, self.global_step, decay_steps, 0)
            tf.summary.scalar('learning_rate', learning_rate)

        with tf.variable_scope('loss'):
            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            loss = tf.reduce_sum(rewards * cross_entropies)
            tf.summary.scalar('loss', loss)

        params = self.model.trainable_variables
        grads = tf.gradients(loss, params)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

        self.train_op = optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)
        self.merged = tf.summary.merge_all()

    def act(self, obs):
        return tf.get_default_session().run(self.act_actions, feed_dict={self.act_obs: obs})

    def train(self, obs, actions, rewards):
        obs = np.asarray(obs, dtype=np.int64)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float32)

        def _process_rewards(rwd):
            return discount_rewards(rwd, self.gamma)

        rewards_processed = np.apply_along_axis(_process_rewards, 0, rewards)
        obs_train, actions_train, rewards_train = map(flatten, (obs, actions, rewards_processed))

        # Add to replay memory
        MEMORY.extend(zip(obs_train, actions_train, rewards_train))

        # Take only on step
        _, summary, _global_step = tf.get_default_session().run([self.train_op, self.merged, self.global_step])
        return summary, _global_step


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-directory', default='/tmp/pg')
    parser.add_argument('--epochs', default=10_000, type=int)
    parser.add_argument('--env-id', default='PongNoFrameskip-v4')
    parser.add_argument('--num-envs', default=8, type=int)
    parser.add_argument('--num-rollout-steps', default=128, type=int)
    parser.add_argument('--train-batch-size', default=10_000, type=int)
    parser.add_argument('--learning-rate', default=5e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--save-checkpoint-steps', default=100, type=int)
    args = parser.parse_args()

    env = VecFrameStack(make_atari_env(args.env_id, num_env=args.num_envs, seed=0), num_stack=4)

    ac_space = env.action_space
    ob_space = env.observation_space

    cnn_model = CNNModel(ac_space)
    agent = PGAgent(model=cnn_model,
                    ob_space=ob_space,
                    gamma=args.gamma,
                    decay_steps=args.epochs,
                    lr=args.learning_rate,
                    batch_size=args.train_batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        restore_path = tf.train.latest_checkpoint(args.model_directory)
        if restore_path:
            saver.restore(sess, restore_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        summary_path = os.path.join(args.model_directory, 'summary')
        writer = tf.summary.FileWriter(summary_path, sess.graph)

        obs = env.reset()

        for update in range(1, args.epochs + 1):
            start_time = time.time()

            mb_obs, mb_actions, mb_rewards = [], [], []
            ep_infos = []
            for _ in range(args.num_rollout_steps):
                actions = agent.act(obs)

                mb_obs.append(obs.copy())
                mb_actions.append(actions)

                obs, rewards, dones, infos = env.step(actions)
                mb_rewards.append(rewards)

                for info in infos:
                    if 'episode' in info:
                        ep_infos.append(info['episode'])

            tf.logging.info('Step: {}, Memory size: {}'.format(update, len(MEMORY)))

            summary, _global_step = agent.train(mb_obs, mb_actions, mb_rewards)
            writer.add_summary(summary, _global_step)

            EP_INFO_BUFF.extend(ep_infos)

            end_time = time.time()
            fps = int(args.train_batch_size / (end_time - start_time))
            episode_len = safe_mean([info['l'] for info in EP_INFO_BUFF])
            episode_rew = safe_mean([info['r'] for info in EP_INFO_BUFF])

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                tag='episode/length', simple_value=episode_len)]), update)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                tag='episode/reward', simple_value=episode_rew)]), update)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='fps', simple_value=fps)]), update)

            if update % args.save_checkpoint_steps == 0:
                save_path = os.path.join(args.model_directory, 'model.ckpt')
                save_path = saver.save(sess, save_path, global_step=update)
                tf.logging.info('Model checkpoint saved: {}'.format(save_path))

        writer.close()
        env.close()


def flatten(x):
    """Swap and then flatten axes 0 and 1"""
    shp = x.shape
    return x.swapaxes(0, 1).reshape(shp[0] * shp[1], *shp[2:])


def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)


if __name__ == '__main__':
    main()
