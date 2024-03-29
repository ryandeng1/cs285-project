import math
import numpy as np
import random
from random import randrange
import gym
from gym import spaces
from gym.utils import seeding
from collections import OrderedDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


map_width = 100
map_height = 100
diagonal = map_height * math.sqrt(2)
num_aircraft = 10 # How many aircrafts showing up in the map (list length)
detect_distance = 2
airport_radar = True

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
NMAC_dist = 1

radius = 40


class AirTrafficGym(MultiAgentEnv):

    def __init__(self, seed, num_agents):
        self.airport = Airport(position = np.array([50., 50.]))
        self.own_state_size = 8
        self.int_state_size = 0
        self.observation_space = self.build_observation_space() #spaces.Box(low=0,high=1,shape=(6,),dtype=np.float32)
        self.position_range = spaces.Box(low=np.array([0, 0]),
                                         high=np.array([map_width, map_height]),
                                         dtype=np.float32)
        # discrete action space: -1, 0, 1
        self.action_space = spaces.Discrete(3)
        # continuous action space: [-1, 1]
        # spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)
        self.conflicts = 0
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self.num_agents = num_agents
        self.boundary_epsilon = 0.0001
        self.reset()

    def reset(self):
        self.aircraft_dict = AC_ActionDict()
        self.id_tracker = 0
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0
        for i in range(self.num_agents):
            random_position = self.random_pos()
            aircraft = Aircraft(id = i,
                                position = random_position,
                                speed = self.random_speed(),
                                heading = self.random_heading(random_position),
                                goal_pos = self.airport.position)
            #print("RANDOM HEADING", aircraft.heading)
            #print("Actually random", self.random_heading())
            self.aircraft_dict.add(aircraft)

        self.airport.generate_interval()
        return self._get_obs([i for i in range(self.num_agents)])

    def _get_obs(self, lst_airplanes):
        state = {}
        others_state = {}
        ids = []
        #s = []
        #id = []
        #for aircraft_id, aircraft in self.aircraft_dict.ac_dict.items():
        for aircraft_id in lst_airplanes:
            aircraft = self.aircraft_dict.ac_dict[aircraft_id]
            # if self.aircraft_dict.ac_dict.items is empty (after reset()), this for loop will not be implemented
            own_s = []
            """
            own_s.append(aircraft.position[0]/ map_width)
            own_s.append(aircraft.position[1]/ map_height)
            own_s.append((aircraft.speed - min_speed) / (max_speed - min_speed))
            own_s.append(aircraft.heading/ (2 * math.pi))
            # own_s.append((0.5 * NMAC_dist)/ diagonal)
            #own_s.append(self.airport.position[0]/ map_width)
            #own_s.append(self.airport.position[1]/ map_height)
            own_s.append(math.radians(rwy_degree)/(2*math.pi))
            own_s.append(aircraft.lifespan/aircraft.total_lifespan)
            """
            own_s.append(aircraft.position[0])
            own_s.append(aircraft.position[1])
            own_s.append(aircraft.speed)
            own_s.append(aircraft.heading)
            state[aircraft_id] = own_s

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

        if num_aircraft > 1:

            if airport_radar:
                distances = np.array([[state[i][0]**2 + state[i][1]**2, i] for i in state.keys()])
                distances = distances[distances[:,0].argsort()]
                for aircraft_id in lst_airplanes:
                    for i in range(num_aircraft-1):
                        if i >= distances.shape[0]:
                            break
                        if distances[i,1] not in state.keys():
                            continue
                        if distances[i][1] != aircraft_id:
                            state[aircraft_id] += state[distances[i,1]][:4]
                    while len(state[aircraft_id]) < 4*num_aircraft:
                        state[aircraft_id] += [0,0,(max_speed-min_speed)/2+min_speed, np.pi]
            else:
                for aircraft_id in lst_airplanes:
                    distances = np.vstack(self.dist_to_all_aircraft(self.aircraft_dict.ac_dict[aircraft_id])).T
                    distances = distances[distances[:,0].argsort()]
                    this_aircraft = state[aircraft_id][:4]
                    for i in range(num_aircraft-1):
                        if distances[i,1] not in state.keys():
                            continue
                        if distances[i,0] > detect_distance:
                            break
                        other_aircraft = state[distances[i,1]][:4]

                        old_other_aircraft1 = other_aircraft[1]
                        other_aircraft[1]=(other_aircraft[0]-this_aircraft[0])*np.cos(this_aircraft[3]) + (other_aircraft[1]-this_aircraft[1])*np.cos(this_aircraft[3]-np.pi/2)
                        other_aircraft[0]=(other_aircraft[0]-this_aircraft[0])*np.cos(this_aircraft[3]-np.pi/2) + (old_other_aircraft1-this_aircraft[1])*np.cos(this_aircraft[3])

                        other_aircraft[3] = state[aircraft_id][3]-other_aircraft[3]
                        if other_aircraft[3] < 0:
                            other_aircraft[3] += 2*np.pi
                        elif other_aircraft[3] > 2*np.pi:
                            other_aircraft[3] -= 2*np.pi

                        state[aircraft_id] += other_aircraft
                        # state[aircraft_id] += state[distances[i,1]][:4]
                    while len(state[aircraft_id]) < 4*num_aircraft:
                        state[aircraft_id] += [0,0,(max_speed-min_speed)/2 + min_speed, np.pi]

        return state

    def step(self, a):
        # Only step for planes with a provided action given by a.
        lst_airplanes = []
        airplane_prev_pos = {}
        for action_id in a.keys():
            airplane_prev_pos[action_id] = self.aircraft_dict.ac_dict[action_id].position
        for action_id in a.keys():
            self.aircraft_dict.ac_dict[action_id].step(a[action_id])
            lst_airplanes.append(action_id)

        self.airport.step()
        '''
        dist_array, id_array = self.dist_to_all_aircraft(aircraft) # will return empty list when aircraft_dict is empty
        min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
        if min_dist > 3 * minimum_separation and self.aircraft_dict.num_aircraft < num_aircraft:
            self.aircraft_dict.add(aircraft)
            self.id_tracker += 1

            self.airport.generate_interval()
        '''
        obs = self._get_obs(lst_airplanes)
        reward, done, info = self._terminal_reward(lst_airplanes, airplane_prev_pos)
        if False in done.values():
            done["__all__"] = False
        else:
            done["__all__"] = True

        return obs, reward, done, info

    def _terminal_reward(self, lst_airplanes, airplane_prev_pos):
        reward = {}
        dones = {}
        info = {}
        info_dist_list = []
        #aircraft_to_remove = []
        #for aircraft_id, aircraft in self.aircraft_dict.ac_dict.items():
        for aircraft_id in lst_airplanes:
            aircraft = self.aircraft_dict.ac_dict[aircraft_id]
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = 9999
            for i in range(len(dist_array)):
                if id_array[i] in lst_airplanes:
                    min_dist = dist_array[i]
                    break
            dist_goal = self.dist_2pt(aircraft.position, self.airport.position)
            aircraft.reward = 0
            conflict = False
            #aircraft.lifespan = max(aircraft.lifespan - 1, 0)
            # Conflict: aircraft will NOT be removed
            """
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

            """


            """
            if not self.position_range.contains(np.array(aircraft.position)):
                aircraft.reward = -1
                #if aircraft not in aircraft_to_remove:
                    #aircraft_to_remove.append(aircraft)
                dones[aircraft_id] = True
                info[aircraft_id] = False
            """
            if self.check_near_boundary(aircraft.position[0], aircraft.position[1]):
                aircraft.reward = -100
                #if aircraft not in aircraft_to_remove:
                    #aircraft_to_remove.append(aircraft)
                dones[aircraft_id] = True
                info[aircraft_id] = False
            # Check near boundary
            elif aircraft.lifespan == 0:
                aircraft.reward = -100
                #if aircraft not in aircraft_to_remove:
                    #aircraft_to_remove.append(aircraft)
                dones[aircraft_id] = True
                info[aircraft_id] = False
            # NMAC: remove
            elif min_dist < NMAC_dist:
                aircraft.reward = -100
                #aircraft_to_remove.append(aircraft)
                dones[aircraft_id] = True
                info[aircraft_id] = False
                self.NMACs += 1

            # Out of map: remove
            # Goal: remove
            elif (dist_goal < goal_radius) \
                    and ((abs(aircraft.heading - math.radians(rwy_degree)) < rwy_degree_sigma) \
                         or (abs(aircraft.heading - math.radians(rwy_degree + 180)) < rwy_degree_sigma)):
            # and (((aircraft.heading >= rwy_degree - rwy_degree_sigma) & (aircraft.heading <= rwy_degree + rwy_degree_sigma)) \
            # or((aircraft.heading >= rwy_degree + math.radians(180) - rwy_degree_sigma) & (aircraft.heading <= rwy_degree + math.radians(180)+ rwy_degree_sigma))):
                aircraft.reward = 100 # aircraft.lifespan/aircraft.total_lifespan
                self.goals += 1
                #if aircraft not in aircraft_to_remove:
                dones[aircraft_id] = True
                #aircraft_to_remove.append(aircraft)
                info[aircraft_id] = True
                '''
                elif aircraft.lifespan == 0:
                aircraft.reward = -1
                aircraft_to_remove.append(aircraft)
                dones[id] = True
                info[id] = False
                '''
            # Taking more steps to land has penalty
            elif not conflict:
                aircraft.reward -= 0.1 #-0.001

            reward[aircraft_id] = aircraft.reward
            #if aircraft not in aircraft_to_remove:
            if aircraft_id not in dones:
                dones[aircraft_id] = False
                info[aircraft_id] = False

        #return reward[id], dones[id], {'is_success':info[id]}
        return reward, dones, self.process_info(info)


    # Change info dict to something compatible with rllib
    def process_info(self, info):
        revised_info = {}
        for k, v in info.items():
            revised_info[k] = {'is_success': v}
        return revised_info


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
        theta = np.linspace(0, 2 * np.pi, 1000)
        a, b = radius * np.cos(theta) + self.airport.position[0], radius * np.sin(theta) + self.airport.position[1]
        idx = randrange(1000)
        pos = np.array([a[idx], b[idx]])
        return pos
        # return np.array([10, 10]) #np.random.uniform(low=np.array([0, 0]), high=np.array([map_width, map_height]))

    def random_speed(self):
        return min_speed + 0.1 #np.random.uniform(low = min_speed, high = max_speed)

    def random_heading(self, random_position):
        rdn_heading = math.atan2(random_position[1] - self.airport.position[1], random_position[0] - self.airport.position[0]) + math.pi
        return rdn_heading #np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        """
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
        """

        # x, y, speed and heading
        max_dist = 1.2*np.sqrt(map_width**2 + map_height**2)
        # return spaces.Box(low=np.array([0, 0, min_speed, 0]*6), high=np.array([map_width, map_height, max_speed, 2 * np.pi]*6))
        return spaces.Box(low=np.array([0, 0, min_speed, 0] + (num_aircraft-1)*[-max_dist,-max_dist,min_speed,0]), high=np.array([map_width, map_height, max_speed, 2 * np.pi] + (num_aircraft-1)*[max_dist,max_dist,max_speed,2*np.pi]))



    def check_near_boundary(self, pos_x, pos_y):
        if abs(pos_x - 0) < self.boundary_epsilon or abs(pos_y - 0) < self.boundary_epsilon:
            return True

        if abs(pos_x - map_width) < self.boundary_epsilon or abs(pos_y - map_height) < self.boundary_epsilon:
            return True

        return False


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

        
        if self.heading > 2*np.pi:
            self.heading -= 2*np.pi
        elif self.heading < 0:
            self.heading += 2*np.pi
        

        self.conflict_id_set = set()
        self.lifespan = 10000
        self.total_lifespan = self.lifespan

    def step(self, a):
        # self.speed += d_speed * a[1]
        self.speed = max(min_speed, min(self.speed, max_speed))
        # self.speed += np.random.normal(0, speed_sigma)

        self.heading += d_heading * (a-1)
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

        # Bound the position
        if self.position[0] < 0:
            self.position[0] = 0

        if self.position[0] >= map_width:
            self.position[0] = map_width

        if self.position[1] < 0:
            self.position[1] = 0

        if self.position[1] >= map_height:
            self.position[1] = map_height
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
