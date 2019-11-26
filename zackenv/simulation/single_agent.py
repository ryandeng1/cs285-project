import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from .config import Config


class single_agent(gym.Env):
    """
    action: The change of heading [-1, 1] continuously
            The change of speed [-1, 1] continuously
    """

    def __init__(self):
        self.load_config()
        self.state = None

        # build observation space and action space
        state_dimension = self.intruder_size * 4 + 8
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(state_dimension,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32)

        self.seed(2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height
        self.intruder_size = Config.intruder_size
        self.EPISODES = Config.EPISODES
        self.G = Config.G
        self.tick = Config.tick
        self.scale = Config.scale
        self.minimum_separation = Config.minimum_separation
        self.horizon_dist = Config.horizon_dist
        self.initial_min_dist = Config.initial_min_dist
        self.goal_radius = Config.goal_radius
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed

    def reset(self):
        # ownship = recordtype('ownship', ['position', 'velocity', 'speed', 'heading', 'bank'])
        # intruder = recordtype('intruder', ['id', 'position', 'velocity'])
        # goal = recordtype('goal', ['position'])

        # initiate ownship to control
        self.subject = Ownship(
            position=(50, 50),
            speed=self.min_speed,
            heading=math.pi/4
        )

        # randomly generate intruder aircraft and store them in a list
        self.intruder_list = []
        for _ in range(self.intruder_size):
            intruder = Aircraft(
                position=self.random_pos(),
                speed=self.random_speed(),
                heading=self.random_heading(),
            )
            # new intruder aircraft can not appear too close to ownship
            # If the generated flight postition is too close at the begninig, or it is at the goal
            while (dist(self.subject, intruder) < self.initial_min_dist) or (intruder.position == np.array([0, 0])).all():
                intruder.position = self.random_pos()

            self.intruder_list.append(intruder)

        # generate a random goal: self.goal = Goal(position=self.random_pos())
        # Fix goal: runway
        self.goal = np.array([0, 0])

        # reset the number of conflicts to 0
        self.no_conflict = 0

        return self._get_ob()

    def _get_ob(self):
        # state contains pos, vel for all intruder aircraft
        # pos, vel, speed, heading for ownship
        # goal pos
        def normalize_velocity(velocity):
            translation = velocity + self.max_speed
            return translation / (self.max_speed * 2)

        s = []
        for aircraft in self.intruder_list:
            # (x, y, vx, vy)
            s.append(aircraft.position[0] / Config.window_width)
            s.append(aircraft.position[1] / Config.window_height)
            s.append(normalize_velocity(aircraft.velocity[0]))
            s.append(normalize_velocity(aircraft.velocity[1]))
        for i in range(1):
            # (x, y, vx, vy, speed, heading)
            s.append(self.subject.position[0] / Config.window_width)
            s.append(self.subject.position[1] / Config.window_height)
            s.append(normalize_velocity(self.subject.velocity[0]))
            s.append(normalize_velocity(self.subject.velocity[1]))
            s.append((self.subject.speed - Config.min_speed) / (Config.max_speed - Config.min_speed))
            s.append(self.subject.heading / (2 * math.pi))
        s.append(self.goal.position[0] / Config.window_width)
        s.append(self.goal.position[1] / Config.window_height)

        return np.array(s)

    def step(self, action):
        assert self.action_space.contains(action), 'given action is in incorrect shape'

        # next state of ownship
        self.subject.step(action)

        reward, terminal, info = self._terminal_reward()

        return self._get_ob(), reward, terminal, info

    def _terminal_reward(self):

        # step the intruder aircraft
        conflict = False
        # for each aircraft
        for idx in range(self.intruder_size):
            intruder = self.intruder_list[idx]
            intruder.position += intruder.velocity
            dist_intruder = dist(self.subject, intruder)
            # if this intruder out of map
            if not self.position_range.contains(intruder.position):
                self.intruder_list[idx] = self.reset_intruder()

            # if there is a conflict
            if dist_intruder < self.minimum_separation:
                conflict = True
                # if conflict status is True, monitor when this conflict status will be escaped
                if intruder.conflict == False:
                    self.no_conflict += 1 # number of conflicts +1
                    intruder.conflict = True
                else:
                    if not dist_intruder < self.minimum_separation:
                        intruder.conflict = False

        # if there is conflict
        if conflict:
            return -1, False, 'c'  # conflict

        if not self.position_range.contains(self.subject.position):
            return -100, True, 'w'  # out-of-map

        # if ownship reaches goal
        if dist(self.subject, self.goal) < self.goal_radius:
            return 1, True, 'g'  # goal
        return -dist(self.subject, self.goal)/1200, False, ''
        return 0, False, ''

    # reset pos, vel, heading of this aircraft
    def reset_intruder(self):
        intruder = Aircraft(
            position=self.random_pos(),
            speed=self.random_speed(),
            heading=self.random_heading(),
        )
        while dist(self.subject, intruder) < self.initial_min_dist:
            intruder.position = self.random_pos()

        return intruder

    def random_pos(self):
        return np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height])
        )

    def random_speed(self):
        return np.random.uniform(low=self.min_speed, high=self.max_speed)

    def random_heading(self):
        return np.random.uniform(low=0, high=2*math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'own_x': spaces.Box(low=0, high=self.window_width, dtype=np.float32),
            'own_y': spaces.Box(low=0, high=self.window_height, dtype=np.float32),
            'pos_x': spaces.Box(low=0, high=self.window_width, dtype=np.float32),
            'pos_y': spaces.Box(low=0, high=self.window_height, dtype=np.float32),
            'heading': spaces.Box(low=0, high=2*math.pi, dtype=np.float32),
            'speed': spaces.Box(low=self.min_speed, high=self.max_speed, dtype=np.float32),
        })
        return s


# class Goal:
#     def __init__(self, position):
#         self.position = position


class Aircraft:
    def __init__(self, position, speed, heading):
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.heading = heading  # rad
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy], dtype=np.float32)

        self.conflict = False  # track if this aircraft is in conflict with ownship


class Ownship(Aircraft):
    def __init__(self, position, speed, heading):
        Aircraft.__init__(self, position, speed, heading)
        self.load_config()

    def load_config(self):

        self.G = Config.G
        self.scale = Config.scale
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.d_speed = Config.d_speed
        self.speed_sigma = Config.speed_sigma
        self.position_sigma = Config.position_sigma

        self.d_heading = Config.d_heading
        self.heading_sigma = Config.heading_sigma

    def step(self, a):
        self.heading += self.d_heading * a[0]
        self.heading += np.random.normal(0, self.heading_sigma)   # randomness 0-2degree (manuver error)
        self.speed += self.d_speed * a[1]
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))  # project to range 
        self.speed += np.random.normal(0, self.speed_sigma)  # errorness

        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])
        self.position += self.velocity


def dist(object1, object2):
    return np.linalg.norm(object1.position - object2.position)
