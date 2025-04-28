import json
import multiprocessing as mp
import os
import time

from config import RunArgs, ExperimentArgs
from A_run_all_experiments import run_all_experiments
from A_visualize_results import (
    visualize_episode_rewards,
    visualize_episode_reward_variance,
    visualize_episode_lengths,
    visualize_cumulative_rewards,
    visualize_model_losses,
)

def main():

    start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()
    run_args.setup_dirs()
    run_args.save_config(os.path.join(run_args.run_dir, "run_config.json"))

    # Load random seeds
    with open("seeds_list.json", "r") as io:
        seeds = json.load(io)[:run_args.use_n_seeds]

    # Run experiments
    run_all_experiments(run_args, seeds)

    # Create visualizations
    visualize_episode_rewards(run_args)
    visualize_episode_reward_variance(run_args)
    visualize_episode_lengths(run_args)
    visualize_cumulative_rewards(run_args)
    visualize_model_losses(run_args)

    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Run start time: {start_time} \nRun end time: {end_time}")


if __name__ == "__main__":
    main()
