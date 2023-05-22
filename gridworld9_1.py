import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
from gym import Env, spaces
import time
import pygame
import math

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
class GridWorldEnv9_1(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,render_mode=None,size=4,exploration_max=0.90,exploration_min=0,exploration_decay=1.0,gamma=1,max_steps=18,learning_rate=1):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.reward_matrix = np.array([[3,-1,-1,-1],[-1,-1,-1,10], [-1, -1,-1,-1],[-1, -1,-1,14]])
        self.diamond_matrix = np.array([["b",-1,-1,-1],[-1,-1,-1,"g"], [-1, -1,-1,-1],[-1, -1,-1,"y"]])
        # -1 means no diamond, "y" means yellow diamond, "b" means blue diamond and "g" means goal

        self.reach = self.size//2

        self.total_weight = 0
        for i in range(self.reach):
            self.total_weight += 1/(i+1)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # Generalize: calculate max and min reward for given range
                "mean_left": spaces.Box(-4,12,(1,)),
                "mean_right": spaces.Box(-4,12,(1,)),
                "mean_up": spaces.Box(-4,12,(1,)),
                "mean_down": spaces.Box(-4,12,(1,)),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.max_steps = max_steps
        self.gamma = gamma
        self.learning_rate = learning_rate
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([0, -1]),
        }

        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "mean_left": self.mean_left, "mean_right": self.mean_right, "mean_up": self.mean_up, "mean_down": self.mean_down}
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def position_to_left(self, position, reach):
        upper_bound = self.size - 1
        lower_bound = 0

        for i in range(reach):
            if (position[0] == lower_bound):
                position = np.array([upper_bound, position[1]])
        
            else:
                position = np.array([position[0]-1, position[1]])
        return position

    def position_to_right(self, position, reach):
        upper_bound = self.size - 1
        lower_bound = 0

        for i in range(reach):
            if (position[0] == upper_bound):
                position = np.array([lower_bound, position[1]])
        
            else:
                position = np.array([position[0]+1, position[1]])
        return position

    def position_to_up(self, position, reach):
        upper_bound = self.size - 1
        lower_bound = 0

        for i in range(reach):
            if (position[1] == upper_bound):
                position = np.array([position[0], lower_bound])
        
            else:
                position =  np.array([position[0], position[1]+1])

        return position
        
    def position_to_down(self, position, reach):
        upper_bound = self.size - 1
        lower_bound = 0

        for i in range(reach):

            if (position[1] == lower_bound):
                position = np.array([position[0], upper_bound])
        
            else:
                position = np.array([position[0], position[1]-1])

        return position


    def distance_from_agent(self, position):
        lower_bound = 0
        upper_bound = self.size - 1
        distance = 0

        if self._agent_location[0] > position[0]:
            agent_dist_wall_right = upper_bound - self._agent_location[0]
            coordinate_dist_wall_left = position[0]
            distance += min(self._agent_location[0] - position[0], agent_dist_wall_right + coordinate_dist_wall_left + 1)

        elif self._agent_location[0] < position[0]:
            agent_dist_wall_left = self._agent_location[0]
            coordinate_dist_wall_right = position[0]
            distance += min(position[0] - self._agent_location[0], agent_dist_wall_left + coordinate_dist_wall_right + 1)

        if self._agent_location[1] > position[1]:
            agent_dist_wall_up = upper_bound - self._agent_location[1]
            coordinate_dist_wall_down = position[1]
            distance += min(self._agent_location[1] - position[1], agent_dist_wall_up + coordinate_dist_wall_down + 1)

        elif self._agent_location[1] < position[1]:
            agent_dist_wall_down = self._agent_location[1]
            coordinate_dist_wall_up = position[1]
            distance += min(position[1] - self._agent_location[1], agent_dist_wall_down + coordinate_dist_wall_up + 1)  

        return distance

    def distance_to_weight(self, distance):

        weight = 1/distance

        return weight



    def _get_coordinates_weights_left(self):
        '''
        return a list of the tuple coordinate-weight needed to compute the weighted mean
        '''
        coordinates = []
        reach = self.size // 2
        for r in range(reach):
            for j in range(self.size):
                coord = np.array([self.position_to_left(self._agent_location, r+1)[0], j])
                dist = self.distance_from_agent(coord)
                weight = self.distance_to_weight(dist)
                coordinates.append([coord, weight])

        return coordinates
    

    def _get_coordinates_weights_right(self):
        '''
        return a list of the tuple coordinate-weight needed to compute the weighted mean
        '''
        coordinates = []
        reach = self.size // 2
        for r in range(reach):
            for j in range(self.size):
                coord = np.array([self.position_to_right(self._agent_location, r+1)[0], j])
                dist = self.distance_from_agent(coord)
                weight = self.distance_to_weight(dist)
                coordinates.append([coord, weight])

        return coordinates
    
    def _get_coordinates_weights_up(self):
        '''
        return a list of the tuple coordinate-weight needed to compute the weighted mean
        '''
        coordinates = []
        reach = self.size // 2
        for r in range(reach):
            for j in range(self.size):
                coord = np.array([j, self.position_to_up(self._agent_location, r+1)[1]])
                dist = self.distance_from_agent(coord)
                weight = self.distance_to_weight(dist)
                coordinates.append([coord, weight])

        return coordinates
    
    def _get_coordinates_weights_down(self):
        '''
        return a list of the tuple coordinate-weight needed to compute the weighted mean
        '''
        coordinates = []
        reach = self.size // 2
        for r in range(reach):
            for j in range(self.size):
                coord = np.array([j, self.position_to_down(self._agent_location, r+1)[1]])
                dist = self.distance_from_agent(coord)
                weight = self.distance_to_weight(dist)
                coordinates.append([coord, weight])

        return coordinates
    
    def _get_weighted_mean_left(self):
        coordinates = self._get_coordinates_weights_left()
        sum = 0
        for coord, weight in coordinates:
            sum += self.reward_matrix[coord[0], coord[1]] * weight

        mean = sum / self.total_weight
        return mean
    
    def _get_weighted_mean_right(self):
        coordinates = self._get_coordinates_weights_right()
        sum = 0
        for coord, weight in coordinates:
            sum += self.reward_matrix[coord[0], coord[1]] * weight

        mean = sum / self.total_weight
        return mean

    def _get_weighted_mean_up(self):
        coordinates = self._get_coordinates_weights_up()
        sum = 0
        for coord, weight in coordinates:
            sum += self.reward_matrix[coord[0], coord[1]] * weight

        mean = sum / self.total_weight
        return mean
    
    def _get_weighted_mean_down(self):
        coordinates = self._get_coordinates_weights_down()
        sum = 0
        for coord, weight in coordinates:
            sum += self.reward_matrix[coord[0], coord[1]] * weight

        mean = sum / self.total_weight
        return mean

    def reset(self, seed=None, options=None, exec=False):
        
    
        self.terminated = False
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.total_reward = 0
        self.reward_matrix = np.array([[3,-1,-1,-1],[-1,-1,-1,10], [-1, -1,-1,-1],[-1, -1,-1,14]])
        self.diamond_matrix = np.array([["b",-1,-1,-1],[-1,-1,-1,"g"], [-1, -1,-1,-1],[-1, -1,-1,"y"]])
        self._target_location = np.array([1, 3])

        

        # Choose the agent's location uniformly at random
        if not exec:

            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            # We will sample the target's location randomly until it does not coincide with the agent's location
            
            while np.array_equal(self._target_location, self._agent_location):
                self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        else: 
            self._agent_location = np.array([1,1])


        self._ydiamond_location = np.array([3,3])
        self._bdiamond_location = np.array([0,0])


        self.mean_left = np.array([self._get_weighted_mean_left()])
        self.mean_right = np.array([self._get_weighted_mean_right()])
        self.mean_up = np.array([self._get_weighted_mean_up()])
        self.mean_down = np.array([self._get_weighted_mean_down()])


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[int(action)]
        new_location = self._agent_location + direction

        # Tests in case of teleport to the other side
        if new_location[0]>3:
            self._agent_location += np.array([-3, 0])

        elif new_location[0]<0:
            self._agent_location += np.array([3, 0])

        elif new_location[1]>3:
            self._agent_location += np.array([0, -3])

        elif new_location[1]<0:
            self._agent_location += np.array([0, 3])

        # No teleports
        else:
            self._agent_location = new_location
        # An episode is done if the agent has reached the target
        self.terminated = np.array_equal(self._agent_location, self._target_location)


        reward = self.reward_matrix[self._agent_location[0], self._agent_location[1]]
        self.total_reward += reward

        # Update reward matrix by deleting collected diamond and the uncollected diamond counterpart
        if np.array_equal(self._agent_location, np.array([0,0])):
            self.reward_matrix[0,0] = -1
            self.reward_matrix[3,3] = -1
            self.diamond_matrix[0,0] = -1
            self.diamond_matrix[3,3] = -1

            # This exists to remove diamond from the rendered screen
            self._ydiamond_location = np.array([4,4])
            self._bdiamond_location = np.array([4,4])

        if np.array_equal(self._agent_location, np.array([3,3])):
            self.reward_matrix[3,3] = -1
            self.reward_matrix[0,0] = -1
            self.diamond_matrix[0,0] = -1
            self.diamond_matrix[3,3] = -1

            # This exists to remove diamond from the rendered screen
            self._bdiamond_location = np.array([4,4])
            self._ydiamond_location = np.array([4,4])


        self.mean_left = np.array([self._get_weighted_mean_left()])
        self.mean_right = np.array([self._get_weighted_mean_right()])
        self.mean_up = np.array([self._get_weighted_mean_up()])
        self.mean_down = np.array([self._get_weighted_mean_down()])  

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, self.terminated, info


    def get_total_reward(self):
        return int(self.total_reward)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        pygame.draw.rect(
            canvas,
            (0, 230, 230),
            pygame.Rect(
                pix_square_size * self._bdiamond_location,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.rect(
            canvas,
            (230, 230, 0),
            pygame.Rect(
                pix_square_size * self._ydiamond_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


from gym.envs.registration import register