from collections import namedtuple, deque

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tf_agents.environments import suite_atari, tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

Experience = namedtuple(
    "Experience", ["observation", "action", "next_step_observation", "reward"]
)


if __name__ == "__main__":
    env_name = "BreakoutNoFrameskip-v4"
    num_iterations = 20000
    max_episode_frames = 108000
    replay_buffer_max_length = 100_000
    epsilon = 0.1  # Epsilon greedy parameter
    gamma = 0.99  # Discount factor for past rewards
    update_target_network = 10_000
    update_after_actions = 4
    ATARI_FRAME_SKIP = 4

    # replay_buffer = deque([], maxlen=replay_buffer_max_length)
    # def gen():
    #     for x in list(replay_buffer):
    #         yield x

    train_env = tf_py_environment.TFPyEnvironment(
        suite_atari.load(
            env_name,
            max_episode_steps=max_episode_frames / ATARI_FRAME_SKIP,
            gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING,
        )
    )
    num_actions = (
        train_env.action_spec().maximum - train_env.action_spec().minimum + 1
    )
    input_shape = train_env.observation_spec().shape

    def create_q_net():
        return tf.keras.Sequential(
            [
                Rescaling(1.0 / 255, input_shape=input_shape),
                Conv2D(32, 8, strides=4, activation="relu",),
                Conv2D(64, 4, strides=2, activation="relu"),
                Conv2D(64, 3, strides=1, activation="relu"),
                Flatten(),
                Dense(512, activation="relu"),
                Dense(num_actions),
            ]
        )

    model = create_q_net()
    model_target = create_q_net()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    # Using huber loss for stability
    loss_fn = tf.keras.losses.Huber()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=(
            tf.TensorSpec(input_shape, tf.uint8, "observation"),
            tf.TensorSpec([], tf.int64, "action"),
            tf.TensorSpec(input_shape, tf.uint8, "next_step_observation"),
            tf.TensorSpec([], tf.float32, "reward"),
            tf.TensorSpec([], tf.bool, "done"),
        ),
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length,
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=32
    ).prefetch(3)
    dataset = iter(dataset)

    # dataset = tf.data.Dataset.from_generator(
    #     gen,
    #     output_types=(tf.uint8, tf.int32, tf.uint8, tf.float32),
    #     output_shapes=(
    #         tf.TensorShape(input_shape),
    #         tf.TensorShape([]),
    #         tf.TensorShape(input_shape),
    #         tf.TensorShape([]),
    #     ),
    # )
    # dataset = dataset.shuffle(10_000).repeat()
    # dataset = dataset.batch(32)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = iter(dataset)

    loss_metric = tf.keras.metrics.Mean(name="loss")
    episode_reward_metric = tf.keras.metrics.Mean(name="episode_reward")
    episode_length_metric = tf.keras.metrics.Mean(name="episode_length")

    log_dir = "logs"
    summary_writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def train_step(obs, action, next_obs, rewards, done):
        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = model_target(next_obs, training=False)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards + gamma * tf.reduce_max(
            future_rewards, axis=1
        )

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

    frame_count = 0
    epsilon_random_frames = 50_000

    for episode in range(10000):
        time_step = train_env.reset()
        episode_return = 0.0
        episode_frames = 0

        while not time_step.is_last():
            frame_count += 1
            episode_frames += 1
            if (
                frame_count < epsilon_random_frames
                or tf.random.uniform([]) < epsilon
            ):
                action = tf.random.uniform(
                    [1], maxval=num_actions, dtype=tf.int64
                )
            else:
                action_probs = model(time_step.observation, training=False)
                action = tf.argmax(action_probs, axis=1)
            next_time_step = train_env.step(action)
            replay_buffer.add_batch(
                (
                    tf.squeeze(time_step.observation),
                    tf.squeeze(action),
                    tf.squeeze(next_time_step.observation),
                    tf.squeeze(next_time_step.reward),
                    tf.squeeze(next_time_step.is_last()),
                )
                # Experience(
                #     observation=tf.squeeze(time_step.observation),
                #     action=tf.squeeze(action),
                #     next_step_observation=tf.squeeze(
                #         next_time_step.observation
                #     ),
                #     reward=tf.squeeze(next_time_step.reward),
                # )
            )
            episode_return += next_time_step.reward
            time_step = next_time_step

            if (
                frame_count % update_after_actions == 0
                and frame_count > 50_000
            ):
                obs, action_sample, next_obs, rewards_sample, done_sample = next(dataset)[0]
                train_step(obs, action_sample, next_obs, rewards_sample, done_sample)

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())

                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss_metric.result(), step=frame_count)
                    tf.summary.scalar("episode/reward", episode_reward_metric.result(), step=frame_count)
                    tf.summary.scalar("episode/length", episode_length_metric.result(), step=frame_count)

                print(
                    f"Frame: {frame_count},",
                    f"Episode: {episode},",
                    f"Loss: {loss_metric.result():.4f},",
                    f"Ep Reward: {episode_reward_metric.result():.2f},",
                    f"Ep Length: {episode_length_metric.result():.0f}",
                )
                loss_metric.reset_states()
                episode_reward_metric.reset_states()
                episode_length_metric.reset_states()

            episode_reward_metric(episode_return)
            episode_length_metric(episode_frames)
