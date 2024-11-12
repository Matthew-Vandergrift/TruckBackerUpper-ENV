"""JAX compatible version of Truck Backer Upper gym environment."""
from typing import Any, Dict, Optional, Tuple, Union
import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from jax import random
#jax.config.update("jax_disable_jit", True)

@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    theta_c: jnp.ndarray
    theta_t: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    l_t: float = 14.0
    l_c: float = 6.0
    dist_tol : float = 3.0
    angle_tol : float = 0.1
    x_bounds: tuple =  (0,200)
    y_bounds: tuple = (-100, 100) 
    max_steps_in_episode: int = 300  


class TBU_gymnax(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of TBU_contact, the gymnaisum env, 
    from https://github.com/Matthew-Vandergrift/TruckBackerUpper-ENV
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for TBU
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        action = jnp.clip(action, -1.0, 1.0)
        # Intermediate Variables for computation
        a = 3.0 * jnp.cos(action)
        b = a * jnp.cos(state.theta_c)
        # Updating State Variables
        x_new = state.x - b * jnp.cos(state.theta_t)
        y_new = state.y - b * jnp.sin(state.theta_t)
        theta_t_new = state.theta_t - jnp.arcsin(a * jnp.sin(state.theta_c) / params.l_t)
        theta_c_new = state.theta_c + jnp.arcsin(3.0 * jnp.sin(action) / (params.l_c + params.l_t))
 
        # Important: Reward is based on termination is previous step transition
        at_goal = jnp.logical_and((jnp.sqrt(x_new**2 + y_new**2) <= params.dist_tol), ((theta_t_new) <= params.angle_tol))
        reward = jax.lax.cond(at_goal.squeeze(), lambda p : 10, lambda p : 0, 5)
        #reward = reward.squeeze()

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x=x_new.squeeze(),
            y=y_new.squeeze(),
            theta_t=theta_t_new.squeeze(),
            theta_c=theta_c_new.squeeze(),
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done, {"discount": self.discount(state, params)}) 

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key_one, key_two = jax.random.split(key)
        state = EnvState(
            x= 160.0,
            y = jax.random.uniform(key_one, minval=-10, maxval=10, shape=(1,)).squeeze(),
            theta_c = 0.0,
            theta_t = jax.random.uniform(key_two, minval=-1.5, maxval=1.5, shape=(1,)).squeeze(),
            time = 0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params : EnvParams, key=None) -> chex.Array:
        """Applies observation function to state."""
        # We include state-scaling here and remove the time variable 
        normed_x = (6*state.x / params.x_bounds[1] - 3)
        normed_y = (3*state.y / params.y_bounds[1])
        obs = jnp.array([normed_x,normed_y, state.theta_c, state.theta_t])
        x = jnp.reshape(obs, (-1,))
        return x

    # def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
    #     """Check whether state is terminal."""
    #     # Check termination criteria
    #     at_goal = jnp.logical_and((jnp.sqrt(state.x**2 + state.y**2) <= params.dist_tol), (state.theta_t <= params.angle_tol))

    #     done_angles = jnp.logical_or(
    #         state.theta_c > jnp.pi/2, 
    #         state.theta_t > 4*jnp.pi
    #     )

    #     done_loc_x = jnp.logical_or(
    #         state.x > params.x_bounds[1], 
    #         state.x < params.x_bounds[0]
    #     )
    #     done_loc_y = jnp.logical_or(
    #         state.y < params.y_bounds[0],
    #         state.y > params.y_bounds[1]
    #     )

    #     # Check number of steps in episode termination condition
    #     done_steps = state.time >= params.max_steps_in_episode
    #     done_loc = jnp.logical_or(done_loc_x, done_loc_y)
    #     done = jnp.logical_or(at_goal, jnp.logical_or(jnp.logical_or(done_loc, done_angles), done_steps))
    #     return jnp.array(done)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "TBUax"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1.0, 1.0, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low=jnp.array([-3, -3, -2*jnp.pi, -2*jnp.pi], dtype=jnp.float32)
        high=jnp.array([3, 3, 2*jnp.pi, 2*jnp.pi], dtype=jnp.float32)
        return spaces.Box(low, high, shape=(4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(params.x_bounds[0], params.x_bounds[1], (), jnp.float32),
                "y": spaces.Box(params.y_bounds[0], params.y_bounds[1], (), jnp.float32),
                "theta_t": spaces.Box(-2*jnp.pi, 2*jnp.pi, (), jnp.float32),
                "theta_c": spaces.Box(-2*jnp.pi, 2*jnp.pi, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )