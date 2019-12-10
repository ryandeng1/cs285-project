import os
import numpy as np
import random
import time
from copy import copy
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten, Activation, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import argparse
import tensorflow as tf
from operator import itemgetter


class Agent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.total_reward = 0
        self.batch_size = 32
        self.lr = 0.00005
        self.value_size = 1
        self.memory = {}
        self.model_check = []
        self.model = self.A2C()


    def discount(self, r):
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r

        return discounted_r

    def A2C(self):

        I = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        own_state = tf.keras.layers.Lambda(lambda x: x[:, :7], output_shape=(7,))(I)
        other_state = tf.keras.layers.Lambda(lambda x: x[:, 7:], output_shape=(self.state_size-7,))(I)
        H1_int = tf.keras.layers.Dense(128, activation='relu')(other_state)
        combined = tf.keras.layers.concatenate([own_state, H1_int], axis=-1)
        H2 = tf.keras.layers.Dense(512, activation='relu')(I)
        H3 = tf.keras.layers.Dense(512, activation='relu')(H2)
        output = tf.keras.layers.Dense(self.action_size + 1, activation=None)(H3)
        policy = tf.keras.layers.Lambda(lambda x: x[:, :self.action_size], output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)
        policy_out = tf.keras.layers.Activation('softmax', name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear', name='value_out')(value)

        opt = tf.keras.optimizers.Adam(lr=self.lr)
        model = tf.keras.models.Model(inputs=I, outputs=[policy_out, value_out])
        model.compile(optimizer=opt, loss={'policy_out': 'categorical_crossentropy', 'value_out': 'mse'})

        return model

    def store(self,s,a,r,sp,done,old_id):
        for i in range(len(old_id)):
            id_ = old_id[i]
            try:
                self.memory[old_id[i]].append((s[i],a[old_id[i]],r[id_],done[id_]))
            except:
                self.memory[old_id[i]] = [(s[i],a[old_id[i]],r[id_],done[id_])]
    def train(self):
        for id_ in self.memory.keys():
            transitions = self.memory[id_]
            states_obs = np.array([rep[0] for rep in transitions]).reshape(-1,self.state_size)
            action_obs = np.array([rep[1] for rep in transitions])
            reward_obs = np.array([rep[2] for rep in transitions])
            q_n = self.discount(reward_obs)
            q_n = q_n.reshape(-1, 1)
            _, value_output = self.model.predict(states_obs)
            adv_n_baseline = q_n - value_output
            adv_n = adv_n_baseline
            advantages = np.zeros((states_obs.shape[0], self.action_size))
            for i in range(states_obs.shape[0]):
                advantages[i][action_obs[i]] = adv_n[i]

            self.model.fit({'states': states_obs},
                           {'policy_out': advantages, 'value_out': q_n},
                           epochs=1, verbose=0,shuffle=False)

        self.memory = {}

    def load(self, name):
        try:
            self.model.load_weights(name)
            print('model loaded successfully...')
        except OSError:
            print('did not find file %s' % name)
        except ValueError:
            print('load wrong model')

    def save(self, name):
        self.model.save_weights(name)

    def act(self, state,ids_):
        a = {}
        state = state.reshape([-1, self.state_size])
        policy, value = self.model.predict(state,batch_size=1)
        for i in range(policy.shape[0]):
            a[ids_[i]] = np.random.choice(self.action_size, 1, p=policy[i].flatten())[0]

        return a
