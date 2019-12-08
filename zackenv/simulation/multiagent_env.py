import math
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
from collections import OrderedDict

from config_ma import Config


class Multiagent(gym.Env):
    def __init__(self, sd):
        self.load_config()
        self.airport = AirPort(position = [400, 400])
        self.state = None

        # build observation space and action space
        self.own_state_size = 8
        self.int_state_size = 8
        self.observation_space = (self.own_state_size+Config.n_closest*self.int_state_size,)
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (2, ), dtype = np.float)

        self.conflicts = 0
        self.conflict_flag = None
        self.distance_mat = None
        self.seed(sd)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height
        self.num_aircraft = Config.num_aircraft
        self.EPISODES = Config.EPISODES
        self.tick = Config.tick
        self.minimum_separation = Config.minimum_separation
        self.NMAC_dist = Config.NMAC_dist
        self.horizon_dist = Config.horizon_dist
        self.initial_min_dist = Config.initial_min_dist
        self.goal_radius = Config.goal_radius
        self.init_speed = Config.init_speed
        self.min_speed = 80
        self.max_speed = 350

    def reset(self):
        # aircraft is stored in this list
        self.aircraft_dict = AircraftDict()
        self.id_tracker = 0

        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0

        return self._get_obs()

    def pressure_reset(self):
        self.conflicts = 0
        # aircraft is stored in this list
        self.aircraft_list = []

        for id in range(self.num_aircraft):
            theta = 2 * id * math.pi / self.num_aircraft
            r = self.window_width / 2 - 10
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            position = (self.window_width / 2 + x, self.window_height / 2 + y)
            goal_pos = (self.window_width / 2 - x, self.window_height / 2 - y)

            aircraft = Aircraft(
                id=id,
                position=position,
                speed=self.init_speed,
                heading=theta + math.pi,
                goal_pos=goal_pos
            )

            self.aircraft_list.append(aircraft)

        return self._get_obs()

    def _get_obs(self):
        s = []
        id = []
        for key, aircraft in self.aircraft_dict.ac_dict.items():

            own_s = []
            # (x, y, vx, vy, speed, heading, gx, gy)
            own_s.append(aircraft.position[0]/ Config.window_width)
            own_s.append(aircraft.position[1]/ Config.window_height)
            own_s.append((aircraft.speed - Config.min_speed) / (Config.max_speed - Config.min_speed))
            own_s.append(aircraft.heading/ (2 * math.pi))
            own_s.append((0.5*Config.NMAC_dist)/Config.diagonal)
            own_s.append(aircraft.goal.position[0]/ Config.window_width)
            own_s.append(aircraft.goal.position[1]/ Config.window_height)
            own_s.append(aircraft.prev_a/ (self.action_space.n-1))



            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            closest_ac = np.argsort(dist_array)
            for j in range(Config.n_closest):
                try:
                    aircraft = self.aircraft_dict.ac_dict[id_array[closest_ac[j]]]
                    own_s.append(aircraft.position[0]/ Config.window_width)
                    own_s.append(aircraft.position[1]/ Config.window_height)
                    own_s.append((aircraft.speed - Config.min_speed) / (Config.max_speed - Config.min_speed))
                    own_s.append(aircraft.heading/ (2 * math.pi))
                    own_s.append(dist_array[closest_ac[j]]/Config.diagonal)
                    own_s.append((0.5*Config.NMAC_dist)/Config.diagonal)
                    own_s.append((Config.NMAC_dist)/Config.diagonal)
                    own_s.append(aircraft.prev_a/ (self.action_space.n-1))

                except:
                    for k in range(self.int_state_size):
                        own_s.append(0)

            s.append(own_s)
            id.append(key)

        return np.reshape(s, (len(s), self.own_state_size+self.int_state_size*Config.n_closest)), id

    def step(self, a,last_ob, near_end=False):
        # a is a dictionary: {id, action, ...}
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            try:
                aircraft.step(a[id])
            except:
                pass
            #    aircraft.step()
        self.airport.step()
        if self.airport.clock_counter >= self.airport.time_next_aircraft and not near_end:
            aircraft = Aircraft(id = self.id_tracker, 
                                position = self.airport.position, #random position
                                speed = self.init_speed,
                                heading = self.random_heading(),
                                goal_pos = self.airport.position)
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
            if min_dist > 3 * self.minimum_separation:  # and self.aircraft_dict.num_aircraft < 10:
                self.aircraft_dict.add(aircraft)
                self.id_tracker += 1

                self.airport.generate_interval()

    
        reward, terminal, info = self._terminal_reward()
        obs = self._get_obs()

        return obs, reward, terminal, info

    def _terminal_reward(self):
        """
        determine the reward and terminal for the current transition, and use info. Main idea:
        1. for each aircraft:
          a. if there is no_conflict, return a large penalty and terminate
          b. elif it is out of map, assign its reward as self.out_of_map_penalty, prepare to remove it
          c. elif if it reaches goal, assign its reward as simulator, prepare to remove it
          d. else assign its reward as simulator
        2. accumulates the reward for all aircraft
        3. remove out-of-map aircraft and goal-aircraft
        4. if all aircraft are removed, return reward and terminate
           else return the corresponding reward and not terminate
        """
        reward = {}
        dones = {}
        info = {}
        # info = {'n': [], 'c': [], 'w': [], 'g': []}
        info_dist_list = []
        aircraft_to_remove = []  # add goal-aircraft and out-of-map aircraft to this list
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            # calculate min_dist and dist_goal for checking terminal
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
            #info_dist_list.append(min_dist)
            dist_goal = self.dist_goal(aircraft)
            aircraft.reward = 0

            conflict = False
            # set the conflict flag to false for aircraft
            # elif conflict, set penalty reward and conflict flag but do NOT remove the aircraft from list
            for id_, dist in zip(id_array, dist_array):
                if dist >= self.minimum_separation:  # safe
                    aircraft.conflict_id_set.discard(id_)  # discarding element not in the set won't raise error

                else:  # conflict!!
                    conflict = True
                    if id_ not in aircraft.conflict_id_set:
                        self.conflicts += 1
                        aircraft.conflict_id_set.add(id_)
                        # info['c'].append('%d and %d' % (aircraft.id, id))
                    if dist == min_dist:
                        aircraft.reward = -0.1 + Config.conflict_coeff*dist
                        #print(aircraft.reward)

            # if NMAC, set penalty reward and prepare to remove the aircraft from list
            if min_dist < self.NMAC_dist:
                # info['n'].append('%d and %d' % (aircraft.id, close_id))
                aircraft.reward = Config.NMAC_penalty
                aircraft_to_remove.append(aircraft)
                dones[id] = True
                info[id] = 'n'
                self.NMACs += 1
                # aircraft_to_remove.append(self.aircraft_dict.get_aircraft_by_id(close_id))

            # give out-of-map aircraft a penalty, and prepare to remove it
            elif not self.position_range.contains(np.array(aircraft.position)):
                aircraft.reward = Config.wall_penalty
                # info['w'].append(aircraft.id)
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)
                    dones[id] = True
                    info[id] = 'w'


            # set goal-aircraft reward according to simulator, prepare to remove it
            elif dist_goal < self.goal_radius:
                aircraft.reward = Config.goal_reward
                # info['g'].append(aircraft.id)
                self.goals += 1
                if aircraft not in aircraft_to_remove:
                    dones[id] = True
                    aircraft_to_remove.append(aircraft)
                    info[id] = 'g'


            # for aircraft without NMAC, conflict, out-of-map, goal, set its reward as simulator
            #elif not conflict:
                #aircraft.reward += Config.step_penalty

            # accumulates reward
            reward[id] = aircraft.reward ##removed
            if aircraft not in aircraft_to_remove:
                dones[id] = False
                info[id] = 't'

        # remove all the out-of-map aircraft and goal-aircraft
        for aircraft in aircraft_to_remove:
            self.aircraft_dict.remove(aircraft)
        # reward = [e.reward for e in self.aircraft_dict]

        return reward, dones, info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dist_to_all_aircraft(self, aircraft):
        id_list = []
        dist_list = []
        for id, intruder in self.aircraft_dict.ac_dict.items():
            if id != aircraft.id:
                id_list.append(id)
                dist_list.append(self.metric(aircraft.position, intruder.position))

        return np.array(dist_list), np.array(id_list)

    def dist_goal(self, aircraft):
        return self.metric(aircraft.position, aircraft.goal.position)

    def metric(self, pos1, pos2):
        # the distance between two points
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    # def dist(self, pos1, pos2):
    #     return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def random_pos(self):
        return np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height])
        )

    def random_speed(self):
        return np.random.uniform(low=self.min_speed, high=self.max_speed)

    def random_heading(self):
        return np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'pos_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'pos_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
            'vel_x': spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'vel_y': spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=self.min_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'heading': spaces.Box(low=0, high=2 * math.pi, shape=(1,), dtype=np.float32),
            'goal_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'goal_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
        })

        return spaces.Tuple((s,) * self.num_aircraft)


class AircraftDict:
    def __init__(self):
        self.ac_dict = OrderedDict()

    @property
    def num_aircraft(self):
        return len(self.ac_dict)

    def add(self, aircraft):
        assert aircraft.id not in self.ac_dict.keys(), 'aircraft id %d already in dict' % aircraft.id
        self.ac_dict[aircraft.id] = aircraft

    def remove(self, aircraft):
        try:
            del self.ac_dict[aircraft.id]
        except KeyError:
            pass

    def get_aircraft_by_id(self, aircraft_id):
        return self.ac_dict[aircraft_id]


class AircraftList:
    def __init__(self):
        self.ac_list = []
        self.id_list = []

    @property
    def num_aircraft(self):
        return len(self.ac_list)

    def add(self, aircraft):
        self.ac_list.append(aircraft)
        self.id_list.append(aircraft.id)
        assert len(self.ac_list) == len(self.id_list)

        unique, count = np.unique(np.array(self.id_list), return_counts=True)
        assert np.all(count < 2), 'ununique id added to list'

    def remove(self, aircraft):
        try:
            self.ac_list.remove(aircraft)
            self.id_list.remove(aircraft.id)
            assert len(self.ac_list) == len(self.id_list)
        except ValueError:
            pass

    def get_aircraft_by_id(self, aircraft_id):
        index = np.where(np.array(self.id_list) == aircraft_id)[0]
        assert index.shape[0] == 1, 'find multi aircraft with id %d' % aircraft_id
        return self.ac_list[int(index)]

        for aircraft in self.buffer_list:
            if aircraft.id == aircraft_id:
                return aircraft


class Goal:
    def __init__(self, position):
        self.position = position


class Aircraft:
    def __init__(self, id, position, speed, heading, goal_pos):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.heading = heading  # rad
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy], dtype=np.float32)
        self.prev_a = 0

        self.reward = 0
        self.goal = Goal(goal_pos)
        dx, dy = self.goal.position - self.position
        self.heading = math.atan2(dy, dx)

        self.load_config()

        self.conflict_id_set = set()

    def load_config(self):
        self.min_speed = 80
        self.max_speed = 350
        self.d_speed = 5
        self.speed_sigma = 0
        self.d_heading = math.radians(5)
        self.heading_sigma = math.radians(0)
        
    def step(self, a):
        # Speed change
        self.speed += self.d_speed * a[1]
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))
        self.speed += np.random.normal(0, self.speed_sigma) # errorness

        # Heading change
        self.heading += a[0] * self.d_heading
        self.heading += np.random.normal(0, self.heading_sigma)

        # Position change = f(speed, heading)
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])
        self.prev_a = a

        self.position += self.velocity


class AirPort:
    def __init__(self, position):
        self.position = np.array(position)
        self.clock_counter = 0
        self.time_next_aircraft = np.random.uniform(0, 60)

    # when the next aircraft will take off
    def generate_interval(self):
        self.time_next_aircraft = np.random.uniform(Config.time_interval_lower, Config.time_interval_upper)
        self.clock_counter = 0

    # add the clock counter by 1
    def step(self):
        self.clock_counter += 1