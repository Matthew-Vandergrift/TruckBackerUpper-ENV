import jax.numpy as jnp
from jax import random
from functools import partial
import math
import jax
from jax import jit
from gymnax.environments import spaces
#from gymnasium import spaces


#TODO: Add truncation to the environment.
#TODO: Normalize the environment? 

class TBU_Jax:
    """ Truck Backer Upper now in Jax!!!"""
    def __init__(self):
        # Setting the fully static parameters
        self.l_t = 14
        self.l_c = 6
        self.x_bounds = [0,200]
        self.y_bounds = [-100, 100]

        # Setting up observation and action spaces, using Gymnax spaces functions
        low=jnp.array([self.x_bounds[0], self.y_bounds[0], -2*jnp.pi, -2*jnp.pi], dtype=jnp.float32)
        high=jnp.array([self.x_bounds[1], self.y_bounds[1], 2*jnp.pi, 2*jnp.pi], dtype=jnp.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,1), dtype=jnp.float32)
        self.observation_space = spaces.Box(low, high, shape=(1,4), dtype=jnp.float32)

    @partial(jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, key = env_state
        x, y, theta_t, theta_c = state # Auto-unpacking the array

        # Applying Dynamic Functions

        # Intermediate Variables for computation
        a = 3 * jnp.cos(action)
        b = a * jnp.cos(theta_c)
        # Updating State Variables
        x += -1 * b * jnp.cos(theta_t)
        y += -1 * b * jnp.sin(theta_t)
        theta_t += -1 * jnp.arcsin(a * jnp.sin(theta_c) / self.l_t)
        theta_c += jnp.arcsin(3 * jnp.sin(action) / (self.l_c + self.l_t))
        # Checking for termination condition
        jack_knifed = theta_c > jnp.pi/2
        not_valid_loc = (x > self.x_bounds[1]) | (x < self.x_bounds[0]) | (y < self.y_bounds[0]) | (y > self.y_bounds[1])
        not_valid_angle = theta_t > 4*jnp.pi
        # For JIT reasons, this cannot simply use the and operator (https://github.com/jax-ml/jax/issues/3761)
        at_goal = jnp.logical_and(jnp.sqrt(x**2 + y**2) <= 3.0, jnp.abs(theta_t) <= 0.1)
        done = at_goal | jack_knifed | not_valid_loc | not_valid_angle
        # Goal-Contact Encoding with 10 for reaching the goal
        reward = (0 + at_goal)*10
        # Updating and returning the state
        env_state = jnp.array([x, y, theta_t, theta_c]), key
        env_state = self._maybe_reset(env_state, done)
        new_state = env_state[0]
        return env_state, self._get_obsv(new_state), reward, done, {}

    def _get_obsv(self, state):
      return state

    def _maybe_reset(self, env_state, done):
      key = env_state[1]
      return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
      # Randomly Setting Truck Position according to An application of TD ... paper
      new_y = random.uniform(key, minval=-10, maxval=10)
      next_key = random.split(key)[0]
      new_theta_t = random.uniform(next_key, minval=-1.5, maxval=1.5)
      new_state = jnp.array([160, new_y, new_theta_t, 0.0], dtype=jnp.float32)
      new_key = random.split(key)[0]
      return new_state, new_key

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
      env_state = self._reset(key)
      initial_state = env_state[0]
      return env_state, self._get_obsv(initial_state)

if __name__ == '__main__':
    seed = 0
    key = random.PRNGKey(seed)
    env = TBU_Jax()
    env_state, inital_obsv = env.reset(key)
    done = False 
    while not done:
        action = 1
        env_state, obsv, reward, done, info = env.step(env_state, 1)
        print(obsv)
        print(reward)
        print(done)
        print("-"*10)