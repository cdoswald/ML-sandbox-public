import json
import multiprocessing as mp
import os
import time

from causalex.config import RunArgs, ExperimentArgs
from causalex.B_train_Q_model import train_Q_model
from causalex.B_visualize_epoch_losses import visualize_epoch_losses


def main():

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()

    # Load random seeds
    with open("seeds_list.json", "r") as io:
        seeds = json.load(io)[:run_args.use_n_seeds]

    # Set experiment parameters
    run_args.test_buffer_sizes = [10_000, 100_000, 1_000_000]

    # Set training parameters
    n_epochs = 20
    n_samples_per_epoch = 5000

    # Set up loss data directory
    run_args.loss_data_dir = os.path.join("runs", "predict_rewards")
    os.makedirs(run_args.loss_data_dir, exist_ok=True)

    # Record start time
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    # Test all environments
    process_args = []
    for env_id in run_args.env_ids:

        # Test all buffer sizes
        for buffer_size in run_args.test_buffer_sizes:

            # Test cx modes
            for cx_mode in run_args.cx_modes:

                # Test all seeds
                for seed in seeds:

                    # Create experiment arguments
                    exp_args = ExperimentArgs(
                        env_id=env_id,
                        cx_mode=cx_mode,
                        buffer_size=buffer_size,
                        seed=seed,
                    )
                    exp_args.n_epochs = n_epochs
                    exp_args.n_samples_per_epoch = n_samples_per_epoch
                    exp_args.loss_data_dir = run_args.loss_data_dir

                    # Add experiment-specific arguments to processes list
                    process_args.append((exp_args))

    # Start processes
    with mp.Pool(processes=run_args.num_workers) as pool:
        pool.map(train_Q_model, process_args)

    # Visualize epoch losses
    visualize_epoch_losses(run_args)

    # Record end time and report progress
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Run start time: {start_time} \nRun end time: {end_time}")


if __name__ == "__main__":
    main()
