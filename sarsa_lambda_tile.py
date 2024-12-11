import numpy as np
from TBU_gymnax import TBU
import jax 
from jax import numpy as jnp
import typing
import tqdm
import matplotlib.pyplot as plt
# Andy Patterson's (https://github.com/andnp/PyFixedReps) Tile Coding. (My Tile coding has a bug) 
from PyFixedReps import TileCoder, TileCoderConfig

def a_numeric(a_index, possible_actions=np.arange(-1, 1, step=0.15)):
    return possible_actions[a_index]


def get_a(weights, s_obs, eps, key=None, possible_actions=np.arange(-1, 1, step=0.15)):
    key_one, key_two = jax.random.split(key)
    rnd_num = jax.random.uniform(key_one)
    if rnd_num <= eps:
        rnd_act = jax.random.randint(key_two, minval=0, maxval=len(possible_actions), shape=())
        return rnd_act
    else:
        qsa = np.sum(weights * s_obs[None, :], axis=-1)
        a = np.argmax(qsa)
        return a
    

# Tile Coding Function from RL-1 Assignment Six 
def tile_features(
    state: np.array,  # (N, S) 
    centers: np.array,  # (D, S)
    widths: float,
    offsets: list = [0],  # list of tuples of length S
) -> np.array:  # (N, D)
    """
    Given centers and widths, you first have to get an array of 0/1, with 1s
    corresponding to tile the state belongs to.
    If "offsets" is passed, it means we are using multiple tilings, i.e., we
    shift the centers according to the offsets and repeat the computation of
    the 0/1 array. The final output will sum the "activations" of all tilings.
    We recommend to normalize the output in [0, 1] by dividing by the number of
    tilings (offsets). 
    """
    result = np.zeros(centers.shape[0])
    for o in offsets:
        result += (np.linalg.norm(state[None, :] - (centers+o), ord=np.inf, axis=-1) <= widths) * 1.0
    return (result)

# Sarsa Lambda as described in the Intro to RL TextBook
def sarsa_lambda(lam, env, num_episodes, gamma, alpha, eps, tc, key=None, env_params=None, possible_actions=np.arange(-1, 1, step=0.15)):
    # Intializing weights + trace
    weights = np.zeros((len(possible_actions),) + (tc.features(),))
    # Intializing Returns 
    returns = np.zeros(num_episodes * 500)
    idx = 0
    # Looping over episodes 
    for e in tqdm.tqdm(range(0, num_episodes)):
        key, _key = jax.random.split(key)
        # Doing epsilon decay 
        eps = max(0.0001, eps - 1 / num_episodes) 
        # Init S
        s_obs, s_state = env.reset(key, env_params) 
        done = False
        phi_s = tc.encode(s_obs)
        # Choose A
        a = get_a(weights, phi_s, eps=eps, key = _key)
        # Init z = 0
        z = np.zeros(tc.features())
        while not done:
            idx += 1
            key, _key = jax.random.split(key)
            # take A, observe R,S`
            s_prime_obs, s_prime_state,  r, done, _ = env.step(_key, s_state, action = a_numeric(a))
            returns[idx] = r
            # Computing Delta and updating Trace
            delta = r - np.dot(weights[a, :], phi_s)
            z += phi_s
            if done:
                weights += alpha * delta * z
                break
            
            phi_s_prime = tc.encode(s_prime_obs)
            a_prime = get_a(weights, phi_s_prime, eps, key=_key)
            delta += gamma * np.dot(weights[a_prime, :], phi_s_prime)

            weights += alpha * delta * z
            z *= gamma * lam
            # Updating Variables
            s_obs = s_prime_obs
            s_state = s_prime_state
            phi_s = phi_s_prime
            a = a_prime
    
    return weights, returns

if __name__ == '__main__':
    print("Hello World!")
    # Andy Patterson's Tile Coder. TODO: Fix the bug in my tile coding function
    config = TileCoderConfig(
        dims=4,
        tilings=5,
        tiles=10,
        scale_output=False,
        input_ranges=[(-3,3)] * 4
    )
    tc = TileCoder(config)

    # Creating the environment 
    env = TBU()
    env_params = env.default_params
    key = jax.random.PRNGKey(0)
    resulting_weights, returns = sarsa_lambda(0.5, env, num_episodes=500, gamma=0.99, alpha=0.1, eps=0.01, tc=tc, key = key, env_params=env_params)
    print("Resulting Weights are :", np.sum(np.abs(resulting_weights)))

    plt.plot(returns)
    plt.xlabel("Env Step")
    plt.ylabel("Reward")
    plt.savefig("plot")