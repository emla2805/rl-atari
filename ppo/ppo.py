"""
Proximal Policy Optimization (PPO) Reinforcement Learning using Tensorflow

https://arxiv.org/pdf/1707.06347.pdf

"""

import numpy as np
import tensorflow as tf
import time
import os
from collections import deque
from argparse import ArgumentParser
from utils import make_atari_env, ortho_init
from vec_env import VecFrameStack


EP_INFO_BUFF = deque(maxlen=100)

tf.logging.set_verbosity(tf.logging.INFO)


class CNNModel(tf.keras.Model):
    def __init__(self, ac_space):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,
                                            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)))
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(515, kernel_initializer=ortho_init(np.sqrt(2)))
        self.vf = tf.keras.layers.Dense(1, kernel_initializer=ortho_init(1.0))
        self.logits = tf.keras.layers.Dense(ac_space.n, kernel_initializer=ortho_init(0.01))

    def call(self, input):
        input_norm = tf.cast(input, tf.float32) / 255.
        result = self.conv1(input_norm)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.flatten(result)
        result = self.dense(result)
        value = self.vf(result)[:, 0]
        logits = self.logits(result)
        return [value, logits]


class PPOAgent(object):
    def __init__(self, model, env, gamma, lam, ent_coef, vf_coef, decay_steps, lr, clip_range,
                 max_grad_norm, batch_size, mini_batch_size):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.lam = lam

        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        self.obs = tf.placeholder(shape=(None,) + ob_space.shape, dtype=ob_space.dtype, name='observations')

        with tf.variable_scope('act'):
            self.act_vpred, self.act_logits = self.model(self.obs)
            self.act_actions = tf.squeeze(tf.multinomial(logits=self.act_logits, num_samples=1))

        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')
            self.logits_old = tf.placeholder(dtype=tf.float32, shape=[None] + [ac_space.n], name='logits_old')
            self.vpred_old = tf.placeholder(dtype=tf.float32, shape=[None], name='vpred_old')

        with tf.name_scope('dataset'):
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.obs, self.actions, self.returns, self.logits_old, self.vpred_old))
            dataset = dataset.shuffle(batch_size).repeat(4).batch(mini_batch_size)
            self.iterator = dataset.make_initializable_iterator()

            obs, actions, returns, logits_old, vpred_old = self.iterator.get_next()
            advantages = returns - vpred_old
            adv_mean, adv_var = tf.nn.moments(advantages, axes=[0])
            advantages = (advantages - adv_mean) / (tf.sqrt(adv_var) + 1e-8)

        vpred, logits = self.model(obs)

        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('learning_params'):
            learning_rate = tf.train.polynomial_decay(lr, self.global_step, decay_steps, 0)
            clip_value = tf.train.polynomial_decay(clip_range, self.global_step, decay_steps, 0)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('clip_value', clip_value)

        with tf.variable_scope('loss'):
            neglogpac_old = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_old, labels=actions)
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
            ratios = tf.exp(neglogpac_old - neglogpac)
            clipped_ratios = tf.clip_by_value(ratios, 1.0 - clip_value, 1.0 + clip_value)
            pg_loss = tf.reduce_mean(
                tf.maximum(-advantages * ratios, -advantages * clipped_ratios))
            tf.summary.scalar('loss_clip', pg_loss)

            vpred_clipped = vpred_old + tf.clip_by_value(vpred - vpred_old, -clip_value, clip_value)
            vf_loss = .5 * tf.reduce_mean(
                tf.maximum(tf.square(vpred - returns), tf.square(vpred_clipped - returns)))
            tf.summary.scalar('vf_loss', vf_loss)

            a0_e = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0_e)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0_e), axis=-1))
            tf.summary.scalar('entropy', entropy)

            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
            tf.summary.scalar('loss', loss)

        params = self.model.trainable_variables
        grads = tf.gradients(loss, params)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

        self.train_op = optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)
        self.merged = tf.summary.merge_all()

    def act(self, obs):
        return tf.get_default_session().run(
            [self.act_actions, self.act_vpred, self.act_logits], feed_dict={self.obs: obs})

    def train(self, writer, obs, actions, rewards, values, logits, dones, last_values, last_dones):
        obs = np.asarray(obs, dtype=np.int64)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        logits = np.asarray(logits, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool)

        dones_next = np.vstack((dones, last_dones))[1:, :]
        next_non_terminal = 1.0 - dones_next
        values_next = np.vstack((values, last_values))[1:, :] * next_non_terminal

        deltas = rewards + self.gamma * values_next - values
        gaes = deltas.copy()

        # calculate generative advantage estimator, see ppo paper eq(11)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * self.lam * gaes[t + 1]
        returns = gaes - values

        obs_train, returns_train, actions_train, values_train, logits_train = map(
            flatten, (obs, returns, actions, values, logits))

        tf.get_default_session().run(self.iterator.initializer, feed_dict={self.obs: obs_train,
                                                                           self.actions: actions_train,
                                                                           self.vpred_old: values_train,
                                                                           self.returns: returns_train,
                                                                           self.logits_old: logits_train})

        while True:
            try:
                _, summary, _global_step = tf.get_default_session().run(
                    [self.train_op, self.merged, self.global_step])
                writer.add_summary(summary, _global_step)
            except tf.errors.OutOfRangeError:
                break


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-directory', default='/tmp/ppo2-cnn-trash/5')
    parser.add_argument('--timesteps', default=10_000_000, type=int)
    parser.add_argument('--env-id', default='PongNoFrameskip-v4')
    parser.add_argument('--num-envs', default=8, type=int)
    parser.add_argument('--num-mini-batches', default=4, type=int)
    parser.add_argument('--num-rollout-steps', default=128, type=int)
    parser.add_argument('--num-opt-epochs', default=4, type=int)
    parser.add_argument('--learning-rate', default=2.5e-4, type=float)
    parser.add_argument('--clip-range', default=0.1, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--save-checkpoint-steps', default=100, type=int)
    args = parser.parse_args()

    env = VecFrameStack(make_atari_env(args.env_id, num_env=args.num_envs, seed=0), num_stack=4)

    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    num_envs = env.num_envs
    ac_space = env.action_space
    batch_size = num_envs * args.num_rollout_steps
    mini_batch_size = batch_size // args.num_mini_batches
    num_updates = args.timesteps // batch_size
    decay_steps = num_updates * args.num_mini_batches * args.num_mini_batches

    cnn_model = CNNModel(ac_space)
    ppo = PPOAgent(model=cnn_model,
                   env=env,
                   gamma=args.gamma,
                   lam=args.lam,
                   ent_coef=ent_coef,
                   vf_coef=vf_coef,
                   decay_steps=decay_steps,
                   lr=args.learning_rate,
                   clip_range=args.clip_range,
                   max_grad_norm=max_grad_norm,
                   batch_size=batch_size,
                   mini_batch_size=mini_batch_size)

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
        dones = np.array([False] * num_envs)

        for update in range(1, num_updates + 1):
            start_time = time.time()

            mb_obs, mb_actions, mb_rewards, mb_values, mb_logits, mb_dones = [], [], [], [], [], []
            ep_infos = []
            for _ in range(args.num_rollout_steps):
                actions, values, logits = ppo.act(obs)

                mb_obs.append(obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_logits.append(logits)
                mb_dones.append(dones)

                obs, rewards, dones, infos = env.step(actions)
                mb_rewards.append(rewards)

                for info in infos:
                    if 'episode' in info:
                        ep_infos.append(info['episode'])

            _, last_values, _ = ppo.act(obs)
            last_dones = dones

            ppo.train(writer, mb_obs, mb_actions, mb_rewards, mb_values, mb_logits, mb_dones, last_values, last_dones)

            EP_INFO_BUFF.extend(ep_infos)

            end_time = time.time()
            fps = int(batch_size / (end_time - start_time))
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
