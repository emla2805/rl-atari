from typing import Tuple, List
import tqdm

import gym
import numpy as np
import tensorflow as tf
from gym.vector import AsyncVectorEnv
from gym.wrappers import AtariPreprocessing, FrameStack
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input

eps = np.finfo(np.float32).eps.item()


def make_env(env_name, seed):
    def _make():
        env = gym.make(env_name)
        env = AtariPreprocessing(
            env, noop_max=30, frame_skip=4, screen_size=84
        )
        env = FrameStack(env, num_stack=4)
        env.seed(seed)
        return env

    return _make


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, reward, done, _ = env.step(action)
    return (
        obs.astype(np.uint8),
        np.array(reward, np.int32),
        np.array(done, np.int32),
    )


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(
        env_step, [action], [tf.uint8, tf.int32, tf.int32]
    )


def run_episode(
    initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int
) -> List[tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        # state = tf.expand_dims(state, 0)
        state = tf.transpose(state, perm=[0, 2, 3, 1])  # Channel last

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[:, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        index = tf.stack([tf.range(action.shape[0], dtype=tf.int64), action], axis=1)
        action_probs_t = tf.gather_nd(action_probs_t, index)
        action_probs = action_probs.write(t, action_probs_t)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        # if tf.cast(done, tf.bool):
        #     break

    action_probs = action_probs.concat()
    values = values.concat()
    rewards = rewards.concat()

    return action_probs, values, rewards


def get_expected_return(
    rewards: tf.Tensor, gamma: float, standardize: bool = True
) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (
            tf.math.reduce_std(returns) + eps
        )

    return returns


if __name__ == "__main__":
    env_name = "BreakoutNoFrameskip-v4"

    num_envs = 8
    max_steps = 3
    num_actions = 4
    input_shape = (84, 84, 4)
    
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    reward_threshold = 195
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    env = AsyncVectorEnv([make_env(env_name, i) for i in range(num_envs)])

    def create_model():
        inputs = Input(shape=(84, 84, 4))

        # Convolutions on the frames on the screen
        rescale = Rescaling(1. / 255.)(inputs)
        layer1 = Conv2D(32, 8, strides=4, activation="relu")(rescale)
        layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = Flatten()(layer3)

        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(num_actions)(layer5)
        value = Dense(1)(layer5)

        return tf.keras.Model(inputs=inputs, outputs=[action, value])

    model = create_model()

    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def compute_loss(
        action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor
    ) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int,
    ) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = run_episode(
                initial_state, model, max_steps_per_episode
            )

            # Calculate expected returns
            returns = get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]

            # Calculating loss values to update our network
            loss = compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward


    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.uint8)
            episode_reward = int(
                train_step(
                    initial_state,
                    model,
                    optimizer,
                    gamma,
                    max_steps_per_episode,
                )
            )

            running_reward = episode_reward * 0.01 + running_reward * 0.99

            t.set_description(f"Episode {i}")
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward
            )

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > reward_threshold:
                break

    print(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!")
