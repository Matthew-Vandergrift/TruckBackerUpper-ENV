# Custom Truck Backer Upper Gymnasium Environment 
# Description of Truck Backer Upper from : An application of the temporal difference
# algorithm to the truck backer-upper problem
# Inspiration of Gynamisum Code from : https://github.com/johnnycode8/gym_custom_env

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

# For Declaring Observation and Action Space
from gymnasium import spaces 

# NYU Code (Can Replace with mine later)
import trucker_backer_problem as tbu

# Begin by Registering the Environment 
register(id = 'TBU_v0', entry_point='trucker_backer_env:TruckBackerEnv')

# Environment Class
class TruckBackerEnv(gym.Env):
    # Required (idk why)
    metadata = {"render_modes": [None], 'render_fps': 1}

    # INIT FUNCTION
    def __init__(self, trailer_length=14, cab_length=6, x_bounds=[0,200], y_bounds=[-100, 100], render_mode = None):
        self.render_mode = render_mode
        # Initalizing the Problem Object
        self.truck = tbu.TruckBackerUpper(trailer_length, cab_length, x_bounds, y_bounds)
        # Defining Action Space
        self.action_space = spaces.Box(low=-np.pi/4, high=np.pi/4)
        # Defining Obseration Space
        self.observation_space = spaces.Box(
            low=np.array([self.truck.x_bounds[0], self.truck.y_bounds[0], -2*np.pi, -2*np.pi]),
            high=np.array([self.truck.x_bounds[1], self.truck.y_bounds[1], 2*np.pi, 2*np.pi]),
            shape=(4,),
            dtype=np.float64
        )
    # GYM FUNCTION 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.truck.reset_truck()
        # Observation is simply the four state variables 
        obs = np.array([self.truck.x, self.truck.y, self.truck.theta_c, self.truck.theta_t])
        # I have no info
        info = {}
        # Return observation and info
        return obs, info

    def step(self, action):
        # Taking the action
        terminated_goal, terminated_fail = self.truck.step(u = action)
        
        # Reward Function based on the paper in header of truck file
        if terminated_fail == True:
            reward = -0.1
            terminated = True  
        elif terminated_goal == True:
            reward = 10 
            terminated = True
        else:
            terminated = False
            t = -0.03 * (self.truck.x**0.6) - 0.002*abs(self.truck.y)**1.2 - 0.1 * abs(self.truck.theta_t) + 0.4
            reward = -4

        # State Observation
        obs = np.array([self.truck.x, self.truck.y, self.truck.theta_c, self.truck.theta_t])

        # Additional info to return. For debugging or whatever.
        info = {}

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info
    
# For unit testing
if __name__=="__main__":
    env = gym.make('TBU_v0', render_mode=None)

    # Reset environment
    obs, info = env.reset()
    print("First Observation is : ", obs)
    rand_action = env.action_space.sample()[0] # This is a bug I need to fix
    print("Random Action is :", rand_action)

    # Reset environment
    obs,info = env.reset()

    # Take some random 
    num_actions_per_episode = 0
    while(True):
        rand_action = env.action_space.sample()[0]
        obs, reward, terminated, _, _ = env.step(rand_action)
        num_actions_per_episode += 1
        if(terminated):
            print("Reset after %s actions" %num_actions_per_episode)
            obs, info  = env.reset()