import numpy as np
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.vec_env import DummyVecEnv
import simulation.traf_env
import gym

env = DummyVecEnv([lambda : simulation.traf_env.AirTrafficGym()])

generate_expert_traj(simulation.traf_env.autopilot, 'expert_airtraffic', env, n_episodes=5000)
