from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from typing import Tuple, List

import gym
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack

from utils import ReplayBuffer, RunningMean


def make_env(env_name, seed):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84)
    env = FrameStack(env, num_stack=4)
    env.seed(seed)
    return env


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, reward, done, _ = env.step(action)
    return (
        np.array(obs, np.uint8),
        np.array(reward, np.int32),
        np.array(done, np.int32),
    )


@tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(
        env_step, [action], [tf.uint8, tf.int32, tf.int32]
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env-name", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--log-dir", default="logs/dqn")
    parser.add_argument("--num-iterations", default=20_000, type=int)
    parser.add_argument("--max-episode-frames", default=10_000, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=2.5e-4, type=float)
    parser.add_argument("--replay-buffer-size", default=100_000, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--min-epsilon", default=0.1, type=float)
    parser.add_argument("--max-epsilon", default=1.0, type=float)
    parser.add_argument("--epsilon-greedy-frames", default=1_000_000, type=int)
    parser.add_argument("--epsilon-random-frames", default=50_000, type=int)
    parser.add_argument("--update-target-freq", default=10_000, type=int)
    args = parser.parse_args()

    epsilon = args.max_epsilon

    env = make_env(args.env_name, seed=0)
    num_actions = 4
    input_shape = (84, 84, 4)
    update_after_actions = 4

    def create_q_net():
        return tf.keras.Sequential(
            [
                Rescaling(1.0 / 255, input_shape=input_shape),
                Conv2D(32, 8, strides=4, activation="relu"),
                Conv2D(64, 4, strides=2, activation="relu"),
                Conv2D(64, 3, strides=1, activation="relu"),
                Flatten(),
                Dense(512, activation="relu"),
                Dense(num_actions),
            ]
        )

    model = create_q_net()
    model_target = create_q_net()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, clipnorm=1.0
    )
    # Using huber loss for stability
    loss_fn = tf.keras.losses.Huber()

    replay_buffer = ReplayBuffer(
        capacity=args.replay_buffer_size, batch_size=args.batch_size
    )

    loss_metric = tf.keras.metrics.Mean(name="loss")
    episode_reward_metric = RunningMean(name="episode_reward", capacity=100)
    episode_length_metric = RunningMean(name="episode_length", capacity=100)
    summary_writer = tf.summary.create_file_writer(args.log_dir)

    @tf.function
    def train_step(obs, action, next_obs, rewards, done):
        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = model_target(next_obs, training=False)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = tf.cast(
            rewards, tf.float32
        ) + args.gamma * tf.reduce_max(future_rewards, axis=1)

        # If final frame set the last value to -1
        done = tf.cast(done, tf.float32)
        updated_q_values = updated_q_values * (1 - done) - done

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action, num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = model(obs, training=True)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = loss_fn(updated_q_values, q_action)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_metric(loss)

    frame = 0

    for episode in range(args.num_iterations):
        obs = tf.constant(env.reset(), dtype=tf.uint8)
        obs = tf.transpose(obs, perm=[1, 2, 0])  # Channel last
        episode_return = 0

        for episode_frame in range(1, args.max_episode_frames):
            frame += 1
            if (
                frame < args.epsilon_random_frames
                or tf.random.uniform([]) < epsilon
            ):
                action = tf.random.uniform(
                    [], maxval=num_actions, dtype=tf.int32
                )
            else:
                action_probs = model(
                    tf.expand_dims(obs, axis=0), training=False
                )
                action = tf.squeeze(
                    tf.argmax(action_probs, axis=1, output_type=tf.int32)
                )
            # Decay probability of taking random action
            epsilon -= (
                args.max_epsilon - args.min_epsilon
            ) / args.epsilon_greedy_frames
            epsilon = max(epsilon, args.min_epsilon)

            next_obs, reward, done = tf_env_step(action)
            next_obs = tf.transpose(next_obs, perm=[1, 2, 0])  # Channel last
            replay_buffer.add(obs, action, reward, next_obs, done)
            episode_return += reward
            obs = next_obs

            if frame % update_after_actions == 0 and frame > args.batch_size:
                (
                    obs_sample,
                    action_sample,
                    rewards_sample,
                    next_obs_sample,
                    done_sample,
                ) = replay_buffer.sample_batch()
                train_step(
                    obs_sample,
                    action_sample,
                    next_obs_sample,
                    rewards_sample,
                    done_sample,
                )

            if frame % args.update_target_freq == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())

                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss_metric.result(), step=frame)
                    tf.summary.scalar(
                        "episode/reward",
                        episode_reward_metric.result(),
                        step=frame,
                    )
                    tf.summary.scalar(
                        "episode/length",
                        episode_length_metric.result(),
                        step=frame,
                    )

                print(
                    f"Frame: {frame},",
                    f"Episode: {episode},",
                    f"Loss: {loss_metric.result():.4f},",
                    f"Ep Reward: {episode_reward_metric.result():.2f},",
                    f"Ep Length: {episode_length_metric.result():.0f}",
                )
                loss_metric.reset_states()

            if done:
                break

        episode_reward_metric(episode_return)
        episode_length_metric(episode_frame)
