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
class GridWorldEnv5(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,render_mode=None,size=4,exploration_max=0.99,exploration_min=0,exploration_decay=0.999999,gamma=0.9,max_steps=16,learning_rate=0.9):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.reward_matrix = np.array([[-1,-1,-1,-1],[-1,-1,3,10], [-1, -1,-1,-1],[13, -1,-1,-1]])
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "left": spaces.Discrete(2),
                "right": spaces.Discrete(2),
                "up": spaces.Discrete(2),
                "down": spaces.Discrete(2),
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
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
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
        return {"agent": self._agent_location, "left": self._left, "right": self._right, "up": self._up, "down": self._down} # , "target": self._target_location

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None, exec=False):
        
        self.terminated = False
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.total_reward = 0
        self.reward_matrix = np.array([[-1,-1,-1,-1],[-1,-1,3,10], [-1, -1,-1,-1],[13, -1,-1,-1]])
        self._target_location = np.array([1, 3])
        # Choose the agent's location uniformly at random
        if not exec:

            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            # We will sample the target's location randomly until it does not coincide with the agent's location
            
            while np.array_equal(self._target_location, self._agent_location):
                self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        else: 
            self._agent_location = np.array([1,1])

        
        # RIGHT
        direction = self._action_to_direction[0]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._right = 0
        else:
            self._right = 1

        # LEFT
        direction = self._action_to_direction[2]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._left = 0
        else:
            self._left = 1



        # UP
        direction = self._action_to_direction[1]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._up = 0
        else:
            self._up = 1



        # DOWN
        direction = self._action_to_direction[3]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._down = 0
        else:
            self._down = 1


        self._ydiamond_location = np.array([3,0])
        self._bdiamond_location = np.array([1,2])

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
        self.terminated = np.array_equal(self._agent_location, self._target_location)
        


        # RIGHT
        direction = self._action_to_direction[0]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._right = 0
        else:
            self._right = 1

        # LEFT
        direction = self._action_to_direction[2]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._left = 0
        else:
            self._left = 1



        # UP
        direction = self._action_to_direction[1]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._up = 0
        else:
            self._up = 1



        # DOWN
        direction = self._action_to_direction[3]
        future_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.reward_matrix[future_location[0], future_location[1]] > 0:
            self._down = 0
        else:
            self._down = 1 

        reward = self.reward_matrix[self._agent_location[0], self._agent_location[1]]
        self.total_reward += reward
        if np.array_equal(self._agent_location, np.array([3,0])):
            self.reward_matrix[3,0] = -1
            self.reward_matrix[1,2] = -1
            self._ydiamond_location = np.array([4,4])
            self._bdiamond_location = np.array([4,4])

        if np.array_equal(self._agent_location, np.array([1,2])):
            self.reward_matrix[1,2] = -1
            self.reward_matrix[3,0] = -1
            self._bdiamond_location = np.array([4,4])
            self._ydiamond_location = np.array([4,4])


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

    def createTable(self):
        self.table = np.zeros((4, 4, 2,2,2,2, 4))

    def argmaxTable(self):
        # Returns action which results in highest Q value for current position
        compare = self.table[self._agent_location[0],self._agent_location[1], self._left, self._right, self._up, self._down]
        winners = np.argwhere(compare == np.amax(compare))
        winners_list = winners.flatten().tolist()
        return random.choice(winners_list)

    def maxTable(self, action):

        direction = self._action_to_direction[int(action)]
        # We use `np.clip` to make sure we don't leave the grid
        next_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        max = -math.inf
        for a in range(4):
            compare = self.table[next_location[0],next_location[1], self._left, self._right, self._up, self._down , a]

            if compare > max:
                max = compare

        return max

    def getaction(self, shouldexplore = True):

        if shouldexplore:
            self.exploration_max *= self.exploration_decay
            self.exploration_max = max(self.exploration_min, self.exploration_max)
            random_unvisited = []

            if random.uniform(0, 1) < self.exploration_max:

                for a in range(4):

                    if self.table[self._agent_location[0], self._agent_location[1], self._left, self._right, self._up, self._down , a] == 0:
                        random_unvisited.append(a)

                    if len(random_unvisited) == 0 or len(random_unvisited) == 4:
                        return random.randint(0,3)
                    
                    else :
                        random_index = random.randint(0, len(random_unvisited) - 1)
                        return random_unvisited[random_index]
        
        best_action = self.argmaxTable()
        return best_action


    def train_QLearning(self, n_episodes=1000000):

        self.createTable()
        for e in range(n_episodes):

            self.reset()
            steps = 0

            while (not self.terminated) and steps <= self.max_steps:

                steps += 1
                action = self.getaction()

                old_location = self._agent_location
                old_left = self._left
                old_right = self._right
                old_up = self._up
                old_down = self._down

                old_Q = self.table[self._agent_location[0],self._agent_location[1],self._left,self._right,self._up,self._down,action]

                next_state, reward, done, info = self.step(action)

                self.table[old_location[0],old_location[1],self._left,self._right,self._up,self._down,action] = ((self.learning_rate*(reward+self.gamma*self.maxTable() 
                - old_Q)) + old_Q)


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


from gym.envs.registration import register 