import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C


import gym
import traf_env
import multi_traf_env
import ray
from ray import tune
from ray.rllib.policy import Policy
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

ray.init()
register_env("multi_air-v0", lambda c: multi_traf_env.AirTrafficGym(num_agents=2))
trainer = PPOTrainer(env="multi_air-v0")
num_train_itr = 50
for i in range(num_train_itr):
  print("****************************Iteration: ", i, "****************************")
  print(trainer.train())

