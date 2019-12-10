import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
from collections import deque

from ac_agent import Agent
import sys
sys.path.extend(['../Simulators'])
from multi_env import AirTrafficGym
np.set_printoptions(precision=2)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


def train(env, agent, save_path):
    agent.train_steps = 0
    min_timesteps_per_batch = 0
    total_timesteps = 0
    num_path = 0
    rolling_goals = deque(maxlen=500)
    best_score = -1000

    for e in range(100000): # number of episodes

        episode = False
        last_ob,old_id = env.reset()
        while not episode:

            if last_ob.shape[0] == 0:
                (ob,new_id), reward, done, info = env.step(np.array([1, 1]),last_ob)
                old_id = new_id
                last_ob = ob
                continue

            action = agent.act(last_ob,old_id)
            (ob,new_id), reward, done, info = env.step(action,last_ob)
            agent.store(last_ob,action,reward,ob,done,old_id)
            old_id = new_id
            last_ob = ob
            if env.id_tracker >= 100:
                episode = True
                print('========================================')
                print("Episode: {}".format(e))
                print("Goals: {}".format(env.goals))
                print("Conflicts: {}".format(env.conflicts))
                print("NMACs: {}".format(env.NMACs))
                print("BestReturn: {}".format(best_score))
                print('========================================')
                agent.train()
            if len(rolling_goals) == 500:
                if np.mean(rolling_goals) >= best_score:
                    best_score = np.mean(rolling_goals)
                    agent.save('save_model/multi_%.2f.h5' %np.mean(rolling_goals))
                    if np.mean(rolling_goals) == 500:
                        break
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--save_path', '-s', type=str, default='save_model/ckpt.h5')
    args = parser.parse_args()
    env = AirTrafficGym(args.seed)
    agent = Agent(state_size=env.observation_space[0], action_size=env.action_space.n)
    if args.train:
        train(env, agent, save_path=args.save_path)
if __name__ == '__main__':
    main()
