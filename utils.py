import tensorflow as tf


class RunningMean(tf.keras.metrics.Metric):
    def __init__(self, capacity, name="running_mean", **kwargs):
        super(RunningMean, self).__init__(name=name, **kwargs)
        self.capacity = capacity
        self.index = 0
        self.values = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=(),
        )

    def update_state(self, val):
        val = tf.cast(val, tf.float32)
        self.values = self.values.write(self.index, val)
        self.index = (self.index + 1) % self.capacity

    def result(self):
        return tf.reduce_mean(self.values.stack())

    def reset_states(self):
        self.values.close()


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.index = 0
        self.obs = tf.TensorArray(
            dtype=tf.uint8,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=(84, 84, 4),
        )
        self.actions = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=(),
        )
        self.rewards = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=(),
        )
        self.next_obs = tf.TensorArray(
            dtype=tf.uint8,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=(84, 84, 4),
        )
        self.dones = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=(),
        )

    def add(self, obs, action, reward, next_obs, done):
        self.obs = self.obs.write(self.index, obs)
        self.actions = self.actions.write(self.index, action)
        self.rewards = self.rewards.write(self.index, reward)
        self.next_obs = self.next_obs.write(self.index, next_obs)
        self.dones = self.dones.write(self.index, done)
        self.index = (self.index + 1) % self.capacity

    def sample_batch(self):
        indices = tf.random.uniform(
            [self.batch_size], maxval=self.rewards.size(), dtype=tf.int32
        )
        return (
            self.obs.gather(indices),
            self.actions.gather(indices),
            self.rewards.gather(indices),
            self.next_obs.gather(indices),
            self.dones.gather(indices),
        )
