import multiprocessing as mp
import os

from causalex.config import RunArgs, ExperimentArgs
from causalex.A_run_single_experiment import run_single_experiment


def run_all_experiments(run_args, seeds):

    # Loop over environments
    process_args = []
    for env_id in run_args.env_ids:

        # Loop over buffer prepopulation modes
        for cx_mode in run_args.cx_modes:

            # Loop over seeds
            for seed in seeds:

                # Create experiment arguments
                exp_args = ExperimentArgs(
                    env_id=env_id,
                    cx_mode=cx_mode,
                    seed=seed,
                )
                exp_args.create_exp_dir()
                exp_args.save_config(
                    os.path.join(exp_args.exp_dir, "exp_config.json")
                )

                if run_args.debug_mode:
                    # Run sequentially
                    run_single_experiment(exp_args)
                else:
                    # Add experiment-specific arguments to processes list
                    process_args.append(exp_args)

    # Start processes
    with mp.Pool(processes=run_args.num_workers) as pool:
        pool.map(run_single_experiment, process_args)
