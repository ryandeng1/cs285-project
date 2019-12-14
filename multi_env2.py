import math
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
from collections import OrderedDict

map_width = 100
map_height = 100
diagonal = map_height * math.sqrt(2)
num_aircraft = 1 # How many aircrafts showing up in the map (list length)
n_closest = 0
rwy_degree = 90
rwy_degree_sigma = math.radians(30)

scale = 60  # 1 pixel = 30 meters
min_speed = 80/scale
max_speed = 350/scale
speed_sigma = 0/scale
d_speed = 5/scale
d_heading = math.radians(5)
heading_sigma = math.radians(0)

goal_radius = 10 # larger will get more goals, small value: agent lands accurately
time_interval_lower = 60
time_interval_upper = 120
conflict_coeff = 0.005
minimum_separation = 4
NMAC_dist = 150/scale


class AirTrafficGym(gym.GoalEnv):

    def __init__(self, seed):
        self.airport = Airport(position = np.array([50., 50.]))
        self.own_state_size = 8
        self.int_state_size = 0
        # self.observation_space = spaces.Box(0,1,shape=(8,),dtype=np.float32)
        self.observation_space = self.build_observation_space()
        self.position_range = spaces.Box(low=np.array([0, 0]),
                                         high=np.array([map_width, map_height]),
                                         dtype=np.float32)
        # discrete action space: -1, 0, 1
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Tuple((spaces.Discrete(3), ) * num_aircraft)

        # continuous action space: [-1, 1]
        # spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)

        self.conflicts = 0
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self.reset()

    def reset(self):
        self.aircraft_dict = AC_ActionDict()
        self.id_tracker = 0
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0
        aircraft = Aircraft(id = 0,
                            position = self.random_pos(),
                            speed = self.random_speed(),
                            heading = self.random_heading(),
                            goal_pos = self.airport.position)
        self.aircraft_dict.add(aircraft)
        self.airport.generate_interval()
        return self._get_obs()

    def _get_obs(self):
        s = []
        id = []
        for key, aircraft in self.aircraft_dict.ac_dict.items():
            # if self.aircraft_dict.ac_dict.items is empty (after reset()), this for loop will not be implemented
            own_s = []
            own_s.append(aircraft.position[0]/ map_width)
            own_s.append(aircraft.position[1]/ map_height)
            own_s.append((aircraft.speed - min_speed) / (max_speed - min_speed))
            own_s.append(aircraft.heading/ (2 * math.pi))
            # own_s.append((0.5 * NMAC_dist)/ diagonal)
            own_s.append(self.airport.position[0]/ map_width)
            own_s.append(self.airport.position[1]/ map_height)
            own_s.append(math.radians(rwy_degree)/(2*math.pi))
            own_s.append(aircraft.lifespan/aircraft.total_lifespan)


            # own_s.append(aircraft.prev_a)

            # dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            # closest_ac = np.argsort(dist_array)
            '''
            for j in range(n_closest):
                try:
                    aircraft = self.aircraft_dict.ac_dict[id_array[closest_ac[j]]]
                    own_s.append(aircraft.position[0]/ map_width)
                    own_s.append(aircraft.position[1]/ map_height)
                    own_s.append((aircraft.speed - min_speed) / (max_speed - min_speed))
                    own_s.append(aircraft.heading/ (2 * math.pi))
                    own_s.append(dist_array[closest_ac[j]]/ diagonal)
                    own_s.append((0.5 * NMAC_dist)/ diagonal)
                    own_s.append(NMAC_dist/ diagonal)
                    own_s.append(aircraft.prev_a)
                except:
                    for k in range(self.int_state_size):
                        own_s.append(0)
            '''
            return np.array(own_s)
            id.append(key)
        return np.reshape(s, (len(s), self.own_state_size+self.int_state_size* n_closest)), id

    def step(self, a):
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            try:
                aircraft.step(a) # Will not implement when aircraft_dict is empty
            except:
                pass

        self.airport.step()
        '''
        dist_array, id_array = self.dist_to_all_aircraft(aircraft) # will return empty list when aircraft_dict is empty
        min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
        if min_dist > 3 * minimum_separation and self.aircraft_dict.num_aircraft < num_aircraft:
            self.aircraft_dict.add(aircraft)
            self.id_tracker += 1
            self.airport.generate_interval()
        '''


        obs = self._get_obs()
        reward, terminal, info = self._terminal_reward()

        return obs, reward, terminal, info

    def _terminal_reward(self):
        reward = {}
        dones = {}
        info = {}
        info_dist_list = []
        aircraft_to_remove = []
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
            dist_goal = self.dist_2pt(aircraft.position, self.airport.position)
            aircraft.reward = 0
            conflict = False
            aircraft.lifespan -= 1

            # Conflict: aircraft will NOT be removed
            for id_, dist in zip(id_array, dist_array):
                if dist >= minimum_separation:
                    aircraft.conflict_id_set.discard(id_)
                else:  # conflict!!
                    conflict = True
                    if id_ not in aircraft.conflict_id_set:
                        self.conflicts += 1
                        aircraft.conflict_id_set.add(id_)
                    if dist == min_dist:
                        aircraft.reward = -0.1 + conflict_coeff * dist

            # NMAC: remove
            if min_dist < NMAC_dist:
                aircraft.reward = -1
                aircraft_to_remove.append(aircraft)
                dones[id] = True
                info[id] = False
                self.NMACs += 1

            # Out of map: remove
            elif not self.position_range.contains(np.array(aircraft.position)):
                aircraft.reward = -1
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)
                    dones[id] = True
                    info[id] = False

            # Goal: remove
            elif (dist_goal < goal_radius) \
                    and ((abs(aircraft.heading - math.radians(rwy_degree)) < rwy_degree_sigma) \
                         or (abs(aircraft.heading - math.radians(rwy_degree + 180)) < rwy_degree_sigma)):
            # and (((aircraft.heading >= rwy_degree - rwy_degree_sigma) & (aircraft.heading <= rwy_degree + rwy_degree_sigma)) \
            # or((aircraft.heading >= rwy_degree + math.radians(180) - rwy_degree_sigma) & (aircraft.heading <= rwy_degree + math.radians(180)+ rwy_degree_sigma))):
                aircraft.reward = 1 # aircraft.lifespan/aircraft.total_lifespan
                self.goals += 1
                if aircraft not in aircraft_to_remove:
                    dones[id] = True
                    aircraft_to_remove.append(aircraft)
                    info[id] = True
                '''
                elif aircraft.lifespan == 0:
                aircraft.reward = -1
                aircraft_to_remove.append(aircraft)
                dones[id] = True
                info[id] = False
                '''
            # Taking more steps to land has penalty
            elif not conflict:
                aircraft.reward = -0.001

            reward[id] = aircraft.reward
            if aircraft not in aircraft_to_remove:
                dones[id] = False
                info[id] = False


        return reward[id], dones[id], {'is_success':info[id]}

    def dist_to_all_aircraft(self, aircraft):
        id_list = []
        dist_list = []
        for id, intruder in self.aircraft_dict.ac_dict.items():
            if id != aircraft.id:
                id_list.append(id)
                dist_list.append(self.dist_2pt(aircraft.position, intruder.position))
        return np.array(dist_list), np.array(id_list)

    def dist_2pt(self, pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def random_pos(self):
        return np.random.uniform(low=np.array([0, 0]), high=np.array([map_width, map_height]))

    def random_speed(self):
        return np.random.uniform(low = min_speed, high = max_speed)

    def random_heading(self):
        return np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'pos_x': spaces.Box(low=0, high = map_width, shape=(1,), dtype=np.float32),
            'pos_y': spaces.Box(low=0, high = map_height, shape=(1,), dtype=np.float32),
            'vel_x': spaces.Box(low = -max_speed, high = max_speed, shape=(1,), dtype=np.float32),
            'vel_y': spaces.Box(low = -max_speed, high = max_speed, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low = min_speed, high = max_speed, shape=(1,), dtype=np.float32),
            'heading': spaces.Box(low=0, high=2 * math.pi, shape=(1,), dtype=np.float32),
            'goal_x': spaces.Box(low=0, high = map_width, shape=(1,), dtype=np.float32),
            'goal_y': spaces.Box(low=0, high = map_height, shape=(1,), dtype=np.float32),
        })

        return spaces.Tuple((s,) * num_aircraft)

class AC_ActionDict:
    def __init__(self):
        self.ac_dict = OrderedDict()

    @property
    def num_aircraft(self):
        return len(self.ac_dict)

    def add(self, aircraft):
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
        unique, count = np.unique(np.array(self.id_list), return_counts=True)

    def remove(self, aircraft):
        try:
            self.ac_list.remove(aircraft)
            self.id_list.remove(aircraft.id)
        except ValueError:
            pass

    def get_aircraft_by_id(self, aircraft_id):
        index = np.where(np.array(self.id_list) == aircraft_id)[0]
        return self.ac_list[int(index)]

        for aircraft in self.buffer_list:
            if aircraft.id == aircraft_id:
                return aircraft


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
        # self.prev_a = np.array([0, 0])
        self.reward = 0
        dx, dy = goal_pos - self.position
        self.heading = math.atan2(dy, dx)

        self.conflict_id_set = set()
        self.lifespan = 1000
        self.total_lifespan = self.lifespan

    def step(self, a):
        # self.speed += d_speed * a[1]
        self.speed = max(min_speed, min(self.speed, max_speed))
        self.speed += np.random.normal(0, speed_sigma)

        self.heading += d_heading * a
        # self.heading += d_heading * a[0]
        self.heading += np.random.normal(0, heading_sigma)
        if self.heading > 2*np.pi:
            self.heading -= 2*np.pi
        elif self.heading < 0:
            self.heading += 2*np.pi

        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])
        self.position += self.velocity
        self.prev_a = a

class Airport:
    def __init__(self, position):
        self.position = np.array(position)
        self.clock_counter = 0
        self.time_next_aircraft = np.random.uniform(0, 60)

    def generate_interval(self):
        self.time_next_aircraft = np.random.uniform(time_interval_lower, time_interval_upper)
        self.clock_counter = 0

    def step(self):
        self.clock_counter += 1