import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import gymnasium as gym
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from causalex.config import RunArgs, ExperimentArgs
from causalex.causal_explorer import prepopulate_buffer_causal, prepopulate_buffer_random

plt.rcParams.update({"font.size": 16})


if __name__ == "__main__":

    # Set-up
    output_dir = os.path.join("runs", "time_comparisons")
    os.makedirs(output_dir, exist_ok=True)

    time_dict = {"causal":[], "random":[]}
    run_args = RunArgs()
    env_id = "Ant-v5"
    seed = 973
    n_sims = 50
    buffer_size = 1_000_000

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    # Time buffer population
    for _ in range(n_sims):
        for cx_mode in ["causal", "random"]:
            exp_args = ExperimentArgs(
                env_id=env_id,
                cx_mode=cx_mode,
                seed=seed,
            )
            start_time = time.time()
            if exp_args.cx_mode.lower() == "causal":
                rb = prepopulate_buffer_causal(env, rb, exp_args)
            elif exp_args.cx_mode.lower() == "random":
                rb = prepopulate_buffer_random(env, rb, exp_args)
            else:
                raise ValueError(
                    f"CX mode'{exp_args.cx_mode} not recognized. Double-check configuration arguments."
                )
            time_dict[cx_mode].append(time.time() - start_time)

    # Calculate metrics
    causal_avg_time = np.mean(time_dict["causal"])
    causal_sd_time = np.std(time_dict["causal"])
    random_avg_time = np.mean(time_dict["random"])
    random_sd_time = np.std(time_dict["random"])

    # Plot distributions
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    sns.histplot(
        time_dict["causal"],
        label="causal",
        ax=axes
    )
    sns.histplot(
        time_dict["random"],
        label="random",
        ax=axes
    )
    axes.tick_params(axis="x", rotation=45)
    axes.set_xlabel("Wall Clock Time (seconds)")
    axes.set_ylabel("Number of Experiments")
    axes.legend()
    fig.savefig(
        os.path.join(output_dir, f"time_comp_prepop_buffer_env_{env_id}_{n_sims}sims.png"),
        bbox_inches="tight",
    )

    # Bootstrap difference in means
    bootstrap_n_sims = 10000
    diff_in_means = []
    for _ in range(bootstrap_n_sims):
        resampled_causal = np.random.choice(
            time_dict["causal"],
            size=len(time_dict["causal"]),
            replace=True
        )
        resampled_random = np.random.choice(
            time_dict["random"],
            size=len(time_dict["random"]),
            replace=True,
        )
        diff_in_means.append(
            (np.mean(resampled_causal) - np.mean(resampled_random))
        )
    print(f"Average difference in means (causal-random): {round(np.mean(diff_in_means), 3)}")
    
    # Plot difference in means
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    sns.histplot(
        diff_in_means,
        label="causal - random",
        ax=axes
    )
    axes.tick_params(axis="x", rotation=45)
    axes.set_xlabel("Wall Clock Time (seconds)")
    axes.set_ylabel("Number of Bootstrapped Samples")
    # axes.legend()
    fig.savefig(
        os.path.join(output_dir, f"time_comp_prepop_buffer_env_{env_id}_{n_sims}sims_diff_means.png"),
        bbox_inches="tight",
    )