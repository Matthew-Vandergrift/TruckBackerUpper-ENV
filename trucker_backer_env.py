# Custom Truck Backer Upper Gymnasium Environment 
# Description of Truck Backer Upper from : An application of the temporal difference
# algorithm to the truck backer-upper problem
# Inspiration of PyGame Code from : https://github.com/johnnycode8/gym_custom_env

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import pygame 
import sys

# For Declaring Observation and Action Space
from gymnasium import spaces 

# My problem code
import trucker_backer_problem as tbu

# Begin by Registering the Environment 

register(id = 'TBU_v0', entry_point='trucker_backer_env:TruckBackerEnv')

# Environment Class
class TruckBackerEnv(gym.Env):
    # Required (idk why)
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    # INIT FUNCTION
    def __init__(self, trailer_length=14, cab_length=6, x_bounds=[0,200], y_bounds=[-100, 100], render_mode = None, fps = 1):
        self.render_mode = render_mode
        # Initalizing the Problem Object
        self.truck = tbu.TruckBackerUpper(trailer_length, cab_length, x_bounds, y_bounds)
        # Defining Action Space
        self.action_space = spaces.Box(low=-1, high=1)
        # Defining Obseration Space
        self.observation_space = spaces.Box(
            low=np.array([self.truck.x_bounds[0], self.truck.y_bounds[0], -2*np.pi, -2*np.pi]),
            high=np.array([self.truck.x_bounds[1], self.truck.y_bounds[1], 2*np.pi, 2*np.pi]),
            shape=(4,),
            dtype=np.float64
        )
        # Defining Max Number of Steps 
        self.step_counter = 0
        # Pygame stuff
        if render_mode == "human":
            self._init_pygame()
            self.fps = fps
            self.last_action=''


    # PYGAME FUNCTION
    def _init_pygame(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        # Define game window size (width, height)
        self.window_size = (self.cell_width * 10, self.cell_height * 3 + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load & resize sprites
        file_name = "truck.png"
        img = pygame.image.load(file_name)
        self.truck_img = pygame.transform.scale(img, self.cell_size)
        file_name2 = "target.png"
        img2 = pygame.image.load(file_name2)
        self.target_img = pygame.transform.scale(img2, self.cell_size)

    def render(self):
        self._process_events()
        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255,255,255))
            
        # Draw Truck
        pos = (self.truck.x, self.truck.y + 50)
        self.window_surface.blit(self.truck_img, pos)
        pos = (0,50)
        self.window_surface.blit(self.target_img, pos)

        # text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        # text_pos = (0, self.window_size[1] - self.action_info_height)
        # self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        # Limit frames per second
        self.clock.tick(self.fps)  

    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()


    # GYM FUNCTION 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.truck.reset_truck()
        self.step_counter = 0
        # Observation is simply the four state variables 
        obs = np.array([self.truck.x, self.truck.y, self.truck.theta_c, self.truck.theta_t])
        # Checking for rendering 
        if self.render_mode == "human":
            self.render()
        # I have no info
        info = {}
        # Return observation and info
        return obs, info

    def step(self, action):
        # Adding a step to the counter
        self.step_counter += 1

        # Taking the action
        terminated_goal, terminated_fail = self.truck.step(u = action[0]) # Need to ask why this is the case
        
        # Reward Function based on the paper in header of truck file
        if terminated_fail == True:
            reward = -0.1
            terminated = True  
        elif terminated_goal == True:
            reward = 10 
            terminated = True
        else:
            terminated = False
            reward = -0.03 * (self.truck.x**0.6) - 0.002*abs(self.truck.y)**1.2 - 0.1*abs(self.truck.theta_t) + 0.4
            
        # State Observation
        normed_x = (6*self.truck.x / self.truck.x_bounds[1] - 3)
        normed_y = (3* self.truck.y / self.truck.y_bounds[1])
        obs = np.array([normed_x, normed_y, self.truck.theta_c, self.truck.theta_t])

        # Additional info to return. For debugging or whatever.
        info = {}

        # Checking for Rendering
        if(self.render_mode=='human'):
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, (self.step_counter == 300), info
    
# For unit testing
if __name__=="__main__":
    env = gym.make('TBU_v0', render_mode=None)

    # Reset environment
    obs, info = env.reset()
    print("First Observation is : ", obs)
    rand_action = env.action_space.sample()
    print("Random Action is :", rand_action)

    # Reset environment
    obs,info = env.reset()

    # Take some random 
    num_actions_per_episode = 0
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)
        num_actions_per_episode += 1
        if reward > 0:
            print("reward is ", reward)
        if(terminated):
            print("Reset after %s actions" %num_actions_per_episode)
            obs, info  = env.reset()