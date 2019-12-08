import numpy as np
import time
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend
# from keras.backend.tensorflow_backend import set_session
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
from collections import deque

from A2C_Agent import A2C_Agent
import sys
sys.path.extend(['../Simulators'])
from config_vertiport import Config
from MultiAircraftVertiportA2CEnv import MultiAircraftEnv

np.set_printoptions(precision=2)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
# backend.tensorflow_backend.set_session(sess)


def train(env, agent, save_path):
    agent.train_steps = 0
    min_timesteps_per_batch = 0
    total_timesteps = 0
    num_path = 0
    rolling_goals = deque(maxlen=500)
    best_score = -1000

    for e in range(Config.no_episodes):

        episode = False
        last_ob,old_id = env.reset()
        while not episode:

            if last_ob.shape[0] == 0:
                (ob,new_id), reward, done, info = env.step(1,last_ob)
                old_id = new_id
                last_ob = ob
                continue

            #env.render()
            action = agent.act(last_ob,old_id)
            #for j in range(3):
            (ob,new_id), reward, done, info = env.step(action,last_ob)
                #if done:
                #    break

            agent.store(last_ob,action,reward,ob,done,old_id)

            old_id = new_id
            last_ob = ob

            if env.id_tracker >= 100:
                episode = True
                print("Episode: {} | Goals: {} | Conflicts: {} | NMACs: {} | BestReturn: {}".format(e,
                                                                                                    env.goals,
                                                                                                    env.conflicts,
                                                                                                    env.NMACs,
                                                                                                    best_score))
                agent.train()
            #if step % 1000 == 0 and len(rolling_goals) == 500:
            #    print("AverageReturn ", np.mean(rolling_goals).round(2))
            #    print("BestReturn ", best_score)
                # print("StdReturn", np.std(returns).round(2))
                # print("MaxReturn", np.max(returns).round(2))
                # print("MinReturn", np.min(returns).round(2))

            if len(rolling_goals) == 500:
                if np.mean(rolling_goals) >= best_score:
                    best_score = np.mean(rolling_goals)
                    agent.save('save_model/multi_%.2f.h5' %np.mean(rolling_goals))
                    if np.mean(rolling_goals) == 500:
                        break

    print("Training Finished")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--save_path', '-s', type=str, default='save_model/ckpt.h5')
    args = parser.parse_args()

    env = MultiAircraftEnv(args.seed)
    agent = A2C_Agent(state_size=env.observation_space[0], action_size=env.action_space.n)

    if args.train:
        train(env, agent, save_path=args.save_path)

    #evaluate(env, agent, args.save_path)


if __name__ == '__main__':
    main()
