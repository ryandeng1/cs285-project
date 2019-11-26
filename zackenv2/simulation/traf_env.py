import numpy as np
import matplotlib.pyplot as plt
import math
import gym
import gym.spaces
from gym.utils import seeding
import shapely.geometry as shape
from simulation.autopilot import closed_form_autopilot

class AirTrafficGym(gym.Env):
    def __init__(self, airspace=shape.box(-10,-10,10,10), runways=[(0,0,np.pi)], timestep=1, vmin=150/3600,vmax=400/3600, airplane={'x':5,'y':1,'v':200/3600,'theta':np.pi,'dv':10/3600, 'dtheta':3*np.pi/180}):
        self._airspace = airspace
        self._runways = runways
        self._timestep = timestep
        self._airplane = airplane
        self._init_airplane = airplane.copy()
        self.vmin = vmin
        self.vmax = vmax
        self.time_elapsed = 0
        # heading, v
        # self.action_space = gym.spaces.Discrete(3) # gym.spaces.MultiDiscrete([4, 3])
        self.action_space = gym.spaces.Box(low=np.array([-1, -airplane['dtheta']]), high=np.array([1, airplane['dtheta']]))
        # x, y, hdg, v
        self.observation_space = gym.spaces.Box(
            low=np.array([self._airspace.bounds[0], self._airspace.bounds[1], 0, vmin]),
            high=np.array([self._airspace.bounds[2], self._airspace.bounds[3], 2*np.pi, vmax]))
        self.reward_range = (-100, 100)
        # self.log = np.reshape(np.array([airplane['x'],airplane['y'],airplane['theta']]), (1,3))
    def step(self, action):
        done = False
        reward = 0

        assert self.action_space.contains(action), "%r (%s) is invalid" % (action, type(action))

        # update airplane_position
        # if np.argmax(action[:3]) == 0:
        # if action == 0:
        #     self._airplane['v'] -= self._airplane['dv']
        # elif np.argmax(action[:3]) == 1:
        # elif action == 1:
        #     self._airplane['v'] += self._airplane['dv']
        '''
        if action[0] == 0:
            self._airplane['theta'] -= self._airplane['dtheta']
        elif action[0] == 1:
            self._airplane['theta'] += closed_form_autopilot(self._airplane['x'],self._airplane['y'], self._airplane['theta'], self._airplane['v'], self._airplane['dtheta'])
        elif action[0] == 2:
            self._airplane['theta'] += self._airplane['dtheta']
        '''
        self._airplane['v'] += action[0]*self._airplane['dv']
        if self._airplane['v'] < self.vmin:
            self._airplane['v'] = self.vmin
        elif self._airplane['v'] > self.vmax:
            self._airplane['v'] = self.vmax


        # action[1] = min(self._airplane['dtheta'], action[1]) if action[1] > 0 else max(self._airplane['dtheta'], action[1])
        ideal_theta = closed_form_autopilot(self._airplane['x'],self._airplane['y'], self._airplane['theta'], self._airplane['v'], self._airplane['dtheta'])

        if abs(action[1]-ideal_theta) < np.pi:
            idtheta = action[1]-ideal_theta
        elif action[1]-ideal_theta > 0:
            idtheta = action[1]-ideal_theta - 2*np.pi
        else:
            idtheta = action[1]-ideal_theta + 2*np.pi

        reward -= 100*np.square(idtheta/np.pi)
        # reward = max(-100, reward)


        self._airplane['theta'] += action[1]

        if self._airplane['theta'] < 0:
            self._airplane['theta'] += 2*np.pi
        if self._airplane['theta'] > 2*np.pi:
            self._airplane['theta'] -= 2*np.pi

        self._airplane['x'] = self._airplane['x'] + self._airplane['v']*self._timestep*np.cos(self._airplane['theta'])
        self._airplane['y'] = self._airplane['y'] + self._airplane['v']*self._timestep*np.sin(self._airplane['theta'])
        # self.log = np.append(self.log, np.reshape(np.array([self._airplane['x'], self._airplane['y'],self._airplane['theta']]), (1,3)), axis=0)

        self.time_elapsed += 1

        # Distribute Reward

        # Out of bounds
        if not self._airspace.contains(shape.Point(self._airplane['x'], self._airplane['y'])):
            reward = -100
            done = True
        # Too slow! falls out of sky
        if self._airplane['v'] < self.vmin:
            reward = -100
            done = True
        # Too fast
        if self._airplane['v'] > self.vmax:
            reward = -100

        # Won simulation
        if self._airplane['x'] + self._airplane['v']*self._timestep*np.cos(self._airplane['theta']) < 0 and self._airplane['x'] > 0 and abs(self._airplane['y']) < .40 and abs(self._airplane['theta']-np.pi) < .05:
            if self._airplane['v'] > self.vmax or self._airplane['v'] < self.vmin:
                reward = -90
            else:
                reward = max(10, 100 - self.time_elapsed*.01 - np.square(self._airplane['theta']-np.pi) - np.square(self._airplane['y']))

            done = True

        state = np.array([self._airplane['x'],self._airplane['y'], self._airplane['v'], self._airplane['theta']], dtype=np.float32)

        return state, reward, done, {}
    def reset(self):
        self._airplane = self._init_airplane.copy()
        state = (np.random.rand(4) * np.array([20,20,self.vmax-self.vmin, 2*np.pi]) + np.array([-10,-10,self.vmin, 0])).astype(np.float32)
        # state = np.array([self._airplane['x'],self._airplane['y'], self._airplane['v'], self._airplane['theta']], dtype=np.float32)
        self.time_elapsed = 0
        return state
def autopilot(_obs):
    _obs = _obs[0]
    dtheta = 3*np.pi/180
    action = np.zeros(2,dtype=np.float32)
    action[1] = closed_form_autopilot(_obs[0], _obs[1], _obs[3], _obs[2], dtheta)
    return np.array([action],dtype=np.float32)
