import numpy as np
import matplotlib.pyplot as plt
import math
import gym
import gym.spaces
from gym.utils import seeding
import shapely.geometry as shape
#from simulation.autopilot import closed_form_autopilot
from autopilot import closed_form_autopilot
from ray.rllib.env.multi_agent_env import MultiAgentEnv



class AirTrafficGym(MultiAgentEnv):
    def __init__(self, airspace=shape.box(-10,-10,10,10), runways=[(0,0,np.pi)], timestep=1, vmin=150/3600,vmax=400/3600, airplane={'x':5,'y':1,'v':200/3600,'theta':np.pi,'dv':10/3600, 'dtheta':3*np.pi/180}, num_agents=2):
        self.num_agents = num_agents
        self._airspace = airspace
        self._runways = runways
        self._timestep = timestep
        #self._airplane = airplane
        self._airplanes = [dict(airplane) for _ in range(self.num_agents)]
        self._init_airplane = airplane.copy()
        self._init_airplanes = [self._init_airplane.copy() for _ in range(self.num_agents)]
        self.vmin = vmin
        self.vmax = vmax
        self.time_elapsed = 0
        self.dv = 200/3600
        self.dtheta = 3*np.pi/180
        # heading, v
        #self.action_space = gym.spaces.Discrete(3) # gym.spaces.MultiDiscrete([4, 3])
        #self.action_space = gym.spaces.MultiDiscrete([3])
        self.action_space = gym.spaces.Discrete(3)
        # self.action_space = gym.spaces.Box(low=np.array([-1,-airplane['dtheta']]), high=np.array([1,airplane['dtheta']]))
        # x, y, hdg, v
        low = [self._airspace.bounds[0], self._airspace.bounds[1], vmin, 0]
        high = [self._airspace.bounds[2], self._airspace.bounds[3], vmax, 2*np.pi]
        self.observation_space = gym.spaces.Box(
            low=np.array(low),
            high=np.array(high))

        self.reward_range = (-100.0, 100.0)
        # self.log = np.reshape(np.array([airplane['x'],airplane['y'],airplane['theta']]), (1,3))
        




    def step_helper(self, action, airplane):
        #airplane = airplane.copy()
        done = False
        reward = 0
        # update airplane_position
        # if np.argmax(action[:3]) == 0:
        if action == 0:
            airplane['v'] -= self.dv #airplane['dv']
        # elif np.argmax(action[:3]) == 1:
        elif action == 1:
            airplane['v'] += self.dv #airplane['dv']
        '''
        if action[0] == 0:
            self._airplane['theta'] -= self._airplane['dtheta']
        elif action[0] == 1:
            self._airplane['theta'] += closed_form_autopilot(self._airplane['x'],self._airplane['y'], self._airplane['theta'], self._airplane['v'], self._airplane['dtheta'])
        elif action[0] == 2:
            self._airplane['theta'] += self._airplane['dtheta']
        '''
        #self._airplane['v'] += action[0]*self._airplane['dv']
        if airplane['v'] < self.vmin:
            airplane['v'] = self.vmin
        elif airplane['v'] > self.vmax:
            airplane['v'] = self.vmax

        # action[1] = min(self._airplane['dtheta'], action[1]) if action[1] > 0 else max(self._airplane['dtheta'], action[1])
        # airplane['theta'] += closed_form_autopilot(airplane['x'],airplane['y'], airplane['theta'], airplane['v'], airplane['dtheta'])
        airplane['theta'] += closed_form_autopilot(airplane['x'],airplane['y'], airplane['theta'], airplane['v'], self.dtheta)
        # reward -= 100*np.abs((action[1]-ideal_dtheta)/self._airplane['dtheta'])
        # reward = max(-100, reward)

        # self._airplane['theta'] += action[1]

        if airplane['theta'] < 0:
            airplane['theta'] += 2*np.pi
        if airplane['theta'] > 2*np.pi:
           airplane['theta'] -= 2*np.pi

        airplane['x'] = airplane['x'] + airplane['v']*self._timestep*np.cos(airplane['theta'])
        airplane['y'] = airplane['y'] + airplane['v']*self._timestep*np.sin(airplane['theta'])
        # self.log = np.append(self.log, np.reshape(np.array([self._airplane['x'], self._airplane['y'],self._airplane['theta']]), (1,3)), axis=0)

        
        # Distribute Reward

        # Out of bounds
        if not self._airspace.contains(shape.Point(airplane['x'], airplane['y'])):
            reward = -100
            done = True
        # Too slow! falls out of sky
        if airplane['v'] < self.vmin:
            reward = -100
            # done = True
        # Too fast
        if airplane['v'] > self.vmax:
            reward = -100

        # Won simulation
        if airplane['x'] + airplane['v']*self._timestep*np.cos(airplane['theta']) < 0 and airplane['x'] > 0 and abs(airplane['y']) < .40 and abs(airplane['theta']-np.pi) < .05:
            if airplane['v'] > self.vmax or airplane['v'] < self.vmin:
                reward = -90
            else:
                reward = max(10, 100 - self.time_elapsed*.01 - abs(airplane['theta']) - 5*abs(airplane['y']))

            done = True

        state = np.array([airplane['x'],airplane['y'], airplane['v'], airplane['theta']], dtype=np.float32)

        return state, reward, done, {}

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) is invalid" % (action, type(action))
        obs = {}
        dones = {}
        rews = {}
        #print("ACTION", action)
        for i in range(self.num_agents):
            airplane_name = "airplane_{0}".format(i)
            if airplane_name in action.keys():
                act = action[airplane_name]
                airplane = self._airplanes[i]
                state_, reward_, done_, _ = self.step_helper(act, airplane)
                obs[airplane_name] = state_
                rews[airplane_name] = reward_
                dones[airplane_name] = done_


        self.time_elapsed += 1
        done_vals = dones.values()
        if False in done_vals:
            dones["__all__"] = False
        else:
            dones["__all__"] = True
        return obs, rews, dones, {}


    def init_plane(self):
        if False:
            x = np.random.uniform(low=self._airspace.bounds[0], high=self._airspace.bounds[2])
            y = np.random.uniform(low=self._airspace.bounds[1], high=self._airspace.bounds[3])
            v = np.random.uniform(low=self.vmin, high=self.vmax)
            theta = np.random.uniform(low=0, high=2 * np.pi)
            #dtheta = self._init_airplane['dtheta']

            random_airplane = {"x": x, "y": y, "v": v, "theta": theta}
            #return random_airplane
            return random_airplane

        else:
            #airplane = [self._init_airplane['x'],self._init_airplane['y'], self._init_airplane['v'], self._init_airplane['theta']]
            airplane = {"x": self._init_airplane['x'], "y": self._init_airplane['y'], "v": self._init_airplane['v'], "theta": self._init_airplane['theta']}
            return airplane



    def reset(self):
        self._airplanes = []
        state = {}
        for i in range(self.num_agents):
            agent_name = "airplane_{}".format(i)
            plane = self.init_plane()
            state[agent_name] = [plane["x"], plane["y"], plane["v"], plane["theta"]]
            self._airplanes.append(plane)
        

        #self._airplanes = airplanes
        #state = np.array(airplanes, dtype=np.float32)
        #self.time_elapsed = 0
        #return state 
        #state = np.array([self._airplane['x'],self._airplane['y'], self._airplane['v'], self._airplane['theta']], dtype=np.float32)
        return state

