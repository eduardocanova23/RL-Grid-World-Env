import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
from gym import Env, spaces
import time
import pygame

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
class GridWorldEnv3(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.reward_matrix = np.array([[-1,-1,-1,-1],[10,3,-1,-1], [-1, -1,-1,-1],[-1, -1,-1,14]])
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "left": spaces.Discrete(5),
                "right": spaces.Discrete(5),
                "up": spaces.Discrete(5),
                "down": spaces.Discrete(5)
                #"target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._reward_to_obs = {
            -1: 1,
            3: 2,
            13: 3,
            10: 4,
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
        return {"agent": self._agent_location, "left" : self._reward_left, "right": self._reward_right, "up": self._reward_up, "down": self._reward_down} 

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.total_reward = 0
        
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)



        self.reward_matrix = np.array([[-1,-1,-1,-1],[10,3,-1,-1], [-1, -1,-1,-1],[-1, -1,-1,13]])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([1, 0])
        while np.array_equal(self._target_location, self._agent_location):
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._ydiamond_location = np.array([3,3])
        self._bdiamond_location = np.array([1,1])

        agent_location_right = np.clip(
            self._agent_location + np.array([1, 0]), 0, self.size - 1
        )

        agent_location_left = np.clip(
            self._agent_location + np.array([-1, 0]), 0, self.size - 1
        )

        agent_location_up = np.clip(
            self._agent_location + np.array([0, -1]), 0, self.size - 1
        )

        agent_location_down = np.clip(
            self._agent_location + np.array([0, 1]), 0, self.size - 1
        )

        true_reward_right = self.reward_matrix[agent_location_right[0], agent_location_right[1]]
        true_reward_left = self.reward_matrix[agent_location_left[0], agent_location_left[1]]
        true_reward_up = self.reward_matrix[agent_location_up[0], agent_location_up[1]]
        true_reward_down = self.reward_matrix[agent_location_down[0], agent_location_down[1]]
        
        self._reward_right = self._reward_to_obs[int(true_reward_right)]
        self._reward_left = self._reward_to_obs[int(true_reward_left)]
        self._reward_up = self._reward_to_obs[int(true_reward_up)]
        self._reward_down = self._reward_to_obs[int(true_reward_down)]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[int(action)]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        reward = self.reward_matrix[self._agent_location[0], self._agent_location[1]]
        self.total_reward += reward
        if np.array_equal(self._agent_location, np.array([3,3])):
            if self.reward_matrix[3,3] != -1:
                self.reward_matrix[3,3] = -1

            self._ydiamond_location = np.array([4,4])
            self._bdiamond_location = np.array([4,4])
            

        if np.array_equal(self._agent_location, np.array([1,1])):
            if self.reward_matrix[1,1] != -1:
                self.reward_matrix[1,1] = -1
            self._bdiamond_location = np.array([4,4])
            self._ydiamond_location = np.array([4,4])
            self._collected_diamonds = 1
        
        
        agent_location_right = np.clip(
            self._agent_location + np.array([1, 0]), 0, self.size - 1
        )

        agent_location_left = np.clip(
            self._agent_location + np.array([-1, 0]), 0, self.size - 1
        )

        agent_location_up = np.clip(
            self._agent_location + np.array([0, -1]), 0, self.size - 1
        )

        agent_location_down = np.clip(
            self._agent_location + np.array([0, 1]), 0, self.size - 1
        )

        true_reward_right = self.reward_matrix[agent_location_right[0], agent_location_right[1]]
        true_reward_left = self.reward_matrix[agent_location_left[0], agent_location_left[1]]
        true_reward_up = self.reward_matrix[agent_location_up[0], agent_location_up[1]]
        true_reward_down = self.reward_matrix[agent_location_down[0], agent_location_down[1]]
        
        self._reward_right = self._reward_to_obs[int(true_reward_right)]
        self._reward_left = self._reward_to_obs[int(true_reward_left)]
        self._reward_up = self._reward_to_obs[int(true_reward_up)]
        self._reward_down = self._reward_to_obs[int(true_reward_down)]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

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