
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({"font.size": 16})
COLORS = {
    "causal":"tab:blue",
    "random":"tab:orange",
    "random_with_noise":"tab:green",
}
ALPHAS = {"causal": 1, "random": 1, "random_with_noise":1}
LINEWIDTH = 1.5


def visualize_epoch_losses(run_args):
    """Visualize reward prediction epoch losses averaged across experiments."""
    for env_id in run_args.env_ids:
        fig, axes = plt.subplots(1, len(run_args.test_buffer_sizes), figsize=(16,4))
        fig.subplots_adjust(wspace=0.35, hspace=0.25)
        for j, buffer_size in enumerate(run_args.test_buffer_sizes):
            for cx_mode in run_args.cx_modes:
                # Get all experiment folders
                file_pattern = f"epoch_losses_env_{env_id}_mode_{cx_mode}_buffer_{buffer_size}_*"
                path_pattern = os.path.join(run_args.loss_data_dir, file_pattern)
                paths = [f for f in glob.glob(path_pattern) if os.path.isfile(f)]
                # Load epoch losses
                epoch_losses_all = []
                for path in paths:
                    with open(path, "r") as io:
                        epoch_losses_all.append(json.load(io))
                # Compute average epoch losses across experiments
                loss_data = np.array(epoch_losses_all).T
                avg_loss_data = np.mean(loss_data, axis=1)
                # Plot average epoch losses
                sns.lineplot(
                    avg_loss_data,
                    label=cx_mode,
                    color=COLORS[cx_mode],
                    alpha=ALPHAS[cx_mode],
                    linewidth=LINEWIDTH,
                    legend=False,
                    ax=axes[j],
                )
                axes[j].set_xlabel("Epoch")
                axes[j].set_ylabel("MSE Loss")
                axes[j].set_title(f"Buffer size: {buffer_size:,}")
        axes[-1].legend()
        # fig.suptitle(f"Reward Prediction Losses ({env_id})", y=1.05)
        savefig_path = os.path.join(
            run_args.loss_data_dir, f"predict_reward_losses_env_{env_id}.png"
        )
        fig.savefig(savefig_path, bbox_inches="tight")


if __name__ == "__main__":
    from causalex.config import RunArgs
    run_args = RunArgs()
    run_args.test_buffer_sizes = [10_000, 100_000, 1_000_000]
    run_args.loss_data_dir = os.path.join("runs", "predict_rewards")
    visualize_epoch_losses(run_args)
