from itertools import combinations
from math import comb
import random
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch

# TODO: think about how to set random seed between action samples
# TODO: plot histogram of rewards of different n-way interactions
# TODO: consider slight perturbations (positive and negative) relative to baseline action
# TODO: explore how batch size and training frequency params affect average episode return
# TODO: explore diminishing returns of causal exploration period length

ADJUST_SEED = 829 # don't use exact same seed as main RL pipeline


def prepopulate_buffer_causal(env, rb, args) -> ReplayBuffer:
    """Runs Causal Explorer method to prepopulate replay buffer
    with experimental/controlled data.
    
    Arguments
        env: instantiated gymnasium environment
        rb: instantiated stable_baselines3 replay buffer
        args: configuration arguments dataclass
    
    Returns
        stable_baselines3 replay buffer with trajectory data
    """
    # set up random seeds and device
    random.seed(args.seed + ADJUST_SEED)
    np.random.seed(args.seed + ADJUST_SEED)
    torch.manual_seed(args.seed + ADJUST_SEED)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create combinations of column indices to unmask
    ## Testing reversing order to prioritize higher-dimensional interactions
    n_action_dims = env.action_space.shape[0]
    interaction_col_idxs = []
    interaction_level = 1
    while interaction_level <= n_action_dims:
        interaction_col_idxs += [
            x for x in combinations(range(n_action_dims), interaction_level)
        ]
        interaction_level += 1
    if args.sort_interact_high_to_low:
        interaction_col_idxs = [x for x in reversed(interaction_col_idxs)]

    # Generate experimental data
    obs, _ = env.reset(seed=(args.seed + ADJUST_SEED))
    saved_obs = 0
    buffer_cap_reached = False
    for idx, unmask_cols in enumerate(interaction_col_idxs):
        # Break loop if hard cap reached on number of saved observations
        if buffer_cap_reached:
            print(f"Reached hard cap of {args.prepopulate_buffer_hard_cap} observations")
            break
        print(f"Generating unique interaction: {idx+1} / {len(interaction_col_idxs)} " +
              f" ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        # Create action mask array
        mask_array = np.ones((n_action_dims,), dtype=bool)
        mask_array[[col_idx for col_idx in unmask_cols]] = False
        # Iterate through interaction loop
        interact_steps = 0
        interact_steps_max_reached = False
        while not (buffer_cap_reached or interact_steps_max_reached):
            # Iterate through RL environment
            obs, _ = env.reset() #TODO: consider setting seed here
            terminations = truncations = False
            while not (terminations or truncations):
                actions = env.action_space.sample()
                actions[mask_array] = 0.
                next_obs, rewards, terminations, truncations, infos = env.step(actions) #TODO: check warning
                # Add data to replay buffer
                real_next_obs = next_obs.copy() if not (terminations or truncations) else obs
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
                # DO NOT MODIFY: Crucial step (easy to overlook)
                obs = next_obs
                # Break out of RL env loop if buffer cap or max interaction steps reached
                saved_obs += 1
                interact_steps += 1
                if saved_obs >= args.prepopulate_buffer_hard_cap:
                    buffer_cap_reached = True
                    break
                if interact_steps >= args.max_steps_per_interact:
                    interact_steps_max_reached = True
                    break
    return rb


def prepopulate_buffer_random(env, rb, args, noise_scale: Optional[int] = None) -> ReplayBuffer:
    """Prepopulated replay buffer with randomly sampled actions.
    
    Arguments
        env: instantiated gymnasium environment
        rb: instantiated stable_baselines3 replay buffer
        args: configuration arguments dataclass
        noise_scale: optional argument; if not None, adds random noise
            sampled from normal distribution with stddev = noise_scale

    Returns
        stable_baselines3 replay buffer with trajectory data
    """
    # set up random seeds and device
    random.seed(args.seed + ADJUST_SEED)
    np.random.seed(args.seed + ADJUST_SEED)
    torch.manual_seed(args.seed + ADJUST_SEED)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Generate random data
    obs, _ = env.reset(seed=(args.seed + ADJUST_SEED))
    saved_obs = 0
    buffer_cap_reached = False
    while not buffer_cap_reached:
        # Iterate through RL environment
        obs, _ = env.reset() #TODO: consider setting seed here
        terminations = truncations = False
        while not (terminations or truncations):
            actions = env.action_space.sample()
            next_obs, rewards, terminations, truncations, infos = env.step(actions) #TODO: check warning
            # Add data to replay buffer
            real_next_obs = next_obs.copy() if not (terminations or truncations) else obs
            if noise_scale is not None:
                obs += np.random.randn(*obs.shape) * noise_scale
                actions += np.random.randn(*actions.shape) * noise_scale
                rewards += np.random.randn(*rewards.shape) * noise_scale
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            # DO NOT MODIFY: Crucial step (easy to overlook)
            obs = next_obs
            # Break out of RL env loop if buffer cap reached
            saved_obs += 1
            if saved_obs >= args.prepopulate_buffer_hard_cap:
                buffer_cap_reached = True
                break
    return rb
