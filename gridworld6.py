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

BLACK = (0, 0, 0)

class GridWorldEnv6(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,render_mode=None,size=4,exploration_max=0.90,exploration_min=0,exploration_decay=1.0,gamma=0.9,max_steps=18,learning_rate=0.9):
        pygame.font.init()
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.reward_matrix = np.array([[0,0,0,0],[0,0,0,10], [0, 0,0,0],[13, 0,0,0]])
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.createTable()
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

        
        self.font = pygame.font.SysFont('timesnewroman.ttf', 18)

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
        return {"agent": self._agent_location} # , "target": self._target_location

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None, exec=False):
        
        self.terminated = False
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.total_reward = 0
        self.reward_matrix = np.array([[0,0,0,0],[0,0,0,10], [0, 0,0,0],[13, 0,0,0]])
        self._target_location = np.array([1, 3])
        # Choose the agent's location uniformly at random
        if not exec:

            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            # We will sample the target's location randomly until it does not coincide with the agent's location
            
            while np.array_equal(self._target_location, self._agent_location):
                self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        else: 
            self._agent_location = np.array([1,1])

        
        


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

        #pygame.draw.rect(
            #canvas,
            #(0, 230, 230),
            #pygame.Rect(
                #pix_square_size * self._bdiamond_location,
                #(pix_square_size, pix_square_size),
            #),
        #)

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

        q_00_R = self.table[0,0,0]
        q_00_U = self.table[0,0,1]
        q_00_L = self.table[0,0,2]
        q_00_D = self.table[0,0,3]

        q_01_R = self.table[0,1,0]
        q_01_U = self.table[0,1,1]
        q_01_L = self.table[0,1,2]
        q_01_D = self.table[0,1,3]


        q_01_L_img = self.font.render(str(q_01_L), True, BLACK)
        q_01_R_img = self.font.render(str(q_01_R), True, BLACK)
        q_01_U_img = self.font.render(str(q_01_U), True, BLACK)
        q_01_D_img = self.font.render(str(q_01_D), True, BLACK)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(q_01_L_img, (10, 180))
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
        self.table = np.zeros((4,4,4))

    def argmaxTable(self):
        # Returns action which results in highest Q value for current position
        winners = np.argwhere(self.table[self._agent_location[0], self._agent_location[1]] == np.amax(self.table[self._agent_location[0], self._agent_location[1]]))
        winners_list = winners.flatten().tolist()
        return random.choice(winners_list)

    def maxTable(self):
        # Returns max possible value of Q from current state
        max = -math.inf
        for a in range(4):
            compare = self.table[self._agent_location[0],self._agent_location[1], a]

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

                    if self.table[self._agent_location[0], self._agent_location[1], a] == 0:
                        random_unvisited.append(a)

                    if len(random_unvisited) == 0 or len(random_unvisited) == 4:
                        # No unvisited states or all unvisited states plus should explore -> Return random action
                        return random.randint(0,3)
                    
                    else :
                        # Return random action that hasn't been chosen yet
                        random_index = random.randint(0, len(random_unvisited) - 1)
                        return random_unvisited[random_index]
        
        # Return action which results in highest Q value
        best_action = self.argmaxTable()
        return best_action


    def train_QLearning(self, n_episodes=1000000):
    
        
        for e in range(n_episodes):
            
            
            self.reset()
            steps = 0

            while (not self.terminated) and steps <= self.max_steps:
                
                
                steps += 1
                action = self.getaction()

                old_location = self._agent_location
            

                old_Q = self.table[self._agent_location[0],self._agent_location[1],action]

                next_state, reward, done, info = self.step(action)

                # New Q(s,a) = learning_rate * ( R(s,a) + gamma * maxQ(s',a') - Q(s,a) ) + Q(s,a)
                self.table[old_location[0],old_location[1],action] = ((self.learning_rate*(reward+self.gamma*self.maxTable() 
                - old_Q)) + old_Q)  


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

