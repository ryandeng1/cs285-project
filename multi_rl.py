import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C


import gym
import multi_env_rllib
import ray
from ray import tune
from ray.rllib.policy import Policy
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.tune.logger import pretty_print
from multi_env2 import AirTrafficGym

ray.init()
def policy_mapping_fn(agent_id):
	return "dqn_policy"


env = multi_env_rllib.AirTrafficGym(seed=0, num_agents=1)
register_env("multi_air-v0", lambda c: env)

num_train_itr = 500
policies = {"dqn_policy": (DQNTFPolicy, env.observation_space, env.action_space, {})}

trainer = DQNTrainer(
        env="multi_air-v0",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["dqn_policy"],
            },
            "gamma": 0.99,
            "n_step": 3,
        })

#trainer = DQNTrainer(env="multi_air-v0")
#trainer = PPOTrainer(env="multi_air-v0")
for i in range(num_train_itr):
    x = trainer.train()
    if i % 10 == 0:
        print("****************************Iteration: ", i, "****************************")
        print(pretty_print(x))

