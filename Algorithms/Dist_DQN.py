import numpy as np
from tensorflow import keras
import tensorflow as tf


class DistDQN:
    def __init__(self, env, model, target_model, optimizer, loss_fn, gamma, n_step, batch_size, max_replay_size):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.max_replay_size = max_replay_size

        self.replay_memory = []
        self.n_steps_memory = []
        self.n_step_buffer = []
        self.total_steps = 0

    def loss_fn(self, x, y):
        loss = tf.Variable(0.0, dtype=tf.float32)
        for i in x:
            loss_ = tf.math.log(tf.Variable(i, dtype=tf.float32)) * y
            loss_ = tf.reduce_sum(loss_)
            loss_ = -1 * loss_
            loss.assign_add(loss_)
        return loss

    def update_dis(self, r, support, probs)