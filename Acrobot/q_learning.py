import numpy as np
import random
import copy
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow import GradientTape
import tensorflow as tf


class ExperienceReplay:
    '''
    During gameplay all the states, actions, rewards, and new states are stored
    '''
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def add(self, s1, a, r, s2, done):
        '''
        Add the state, action, reward, and new state to the memory
        return 1 if full otherwise 0
        '''
        self.memory.append([s1, a, r, s2, done])
        if len(self.memory) > self.max_memory:
            return 1
        return 0
    
    def sample(self, n=90):
        '''
        Randomly sample a batch of size n from memory
        '''
        n = min(n, len(self.memory))
        random.shuffle(self.memory)
        ret = self.memory[:n]
        self.memory = self.memory[n:]
        return ret
    

class TargetNetwork:
    '''
    The target network is used to generate the target Q-values during training.
    '''
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def update_target(self, model):
        '''
        Update the target network
        '''
        self.target_model.set_weights(model.get_weights())

    def predict(self, state):
        '''
        Predict the Q-values for the given state
        '''
        return self.target_model(state)
    

class QLearningAgent:
    '''
    The Q-learning agent
    For now it only supports enviroment from OpenAI gym
    But it can be easily extended to other environments
    '''
    def __init__(self, model, target_model, env, exp_replay, step_size=0.9,  gamma=0.99,  batch_size=90, frequency=10, tau=1, epsilon=0.2, max_time=200):
        self.model = model
        self.target_model = target_model
        self.env = env
        self.exp_replay = exp_replay
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = epsilon
        self.gamma = gamma
        self.frequency = frequency
        self.step_size = step_size
        self.max_time = max_time

        self.history = []
    
    def train(self, num_episodes=1000):
        '''
        Train the agent for num_episodes
        '''
        for episode in range(num_episodes):
            s1 = self.env.reset()[0]
            done = False
            t = 0
            while (not done )and (t < self.max_time):
                t += 1
                q_vals = self.model(np.array([s1]))[0]
                a = self.epsilon_greedy(q_vals)
                s2, r, done, information , truncated = self.env.step(a)
                if done :
                    r = 500000
                if self.exp_replay.add(s1, a, r, s2, done):
                    # sample_exps = self.exp_replay.sample(self.batch_size)
                    # self.train_batch(sample_exps)
                    pass
                s1 = s2
            sample_exps = self.exp_replay.sample(self.batch_size)
            self.train_batch(sample_exps)
            if episode % self.frequency == 0:
                self.target_model.update_target(self.model)
            self.history.append(t)
            # print('Episode: {}, Steps: {}'.format(episode, t))
    
    def train_batch(self, sample_exps):
        '''
        Train the agent on a batch of experiences
        '''
        s1_batch = np.array([exp[0] for exp in sample_exps])
        a_batch = np.array([exp[1] for exp in sample_exps])
        r_batch = np.array([exp[2] for exp in sample_exps])
        s2_batch = np.array([exp[3] for exp in sample_exps])
        done_batch = np.array([exp[4] for exp in sample_exps])
        q_vals = self.model(s1_batch)
        q = []
        for i in range(len(q_vals)):
            q.append(q_vals[i][a_batch[i]])
        q_vals = np.array(q)
        next_state_q_vals = self.target_model.predict(s2_batch)
        y_batch = (r_batch + self.gamma * np.max(next_state_q_vals, axis=1)) * done_batch - 1
        y_batch = q_vals +  (y_batch - q_vals) * self.step_size

        if (len(self.history)%20==0):
            # print("q_vals",q_vals)
            # print("y_batch",y_batch )
            # print("r_batch",r_batch)
            # print("done_batch",done_batch)
            print("game_play : ", len(q_vals))
            # print("game_play : ", self.history[-1])
            print()
        
        with GradientTape() as tape:
            q_vals = self.model(s1_batch, training=True)
            q_vals = tf.gather(q_vals, a_batch, axis=1, batch_dims=1)
            # loss = tf.keras.losses.huber(y_batch, q_vals)
            # loss = tf.keras.losses.mean_absolute_error(y_batch, q_vals)
            # loss = tf.keras.losses.mean_squared_error(y_batch, q_vals)
            loss = tf.keras.losses.MSE(y_batch, q_vals)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def epsilon_greedy(self, q_vals):
        '''
        Choose the action greedily with probability 1-e
        and randomly with probability e.
        '''
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_vals))
        else:
            return np.argmax(q_vals)

