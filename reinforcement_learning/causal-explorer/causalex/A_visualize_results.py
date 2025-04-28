import glob
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Define constants
plt.rcParams.update({"font.size": 16})
COLORS = {"causal":"tab:blue", "random":"tab:orange", "random_with_noise":"tab:green"}
FIGSIZE = (10,4)
LINEWIDTH = 1
ALPHA = 0.2
INCLUDE_TITLES = False


def visualize_episode_rewards(run_args):
    """Visualize average episode rewards across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_rewards_all = []
            for exp_dir in exp_dirs:
                exp_rewards_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_rewards_path):
                    with h5py.File(exp_rewards_path, "r") as file:
                        exp_rewards_all.append(list(file["episode_rewards"][:]))
            # Clip metrics lists so that all seeds have same number of episodes
            min_num_episodes = min([len(exp) for exp in exp_rewards_all])
            rewards = np.array([exp[:min_num_episodes] for exp in exp_rewards_all]).T
            # Aggregate metrics across all seeds
            plot_middle = np.mean(rewards, axis=1)
            lower_bound = np.sort(rewards, axis=1)[:, 1]
            upper_bound = np.sort(rewards, axis=1)[:, -2]
            # Plot metrics
            sns.lineplot(
                x=range(min_num_episodes),
                y=plot_middle,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                x=range(min_num_episodes),
                y1=lower_bound,
                y2=upper_bound, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        # Label plot
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.legend(loc="upper left", ncol=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Episode Reward")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        if INCLUDE_TITLES:
            ax.set_title(f"Average Episode Reward by Episode ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_episode_rewards.png"),
            bbox_inches="tight",
        )

def visualize_episode_reward_variance(run_args):
    """Visualize episode reward variance across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_rewards_all = []
            for exp_dir in exp_dirs:
                exp_rewards_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_rewards_path):
                    with h5py.File(exp_rewards_path, "r") as file:
                        exp_rewards_all.append(list(file["episode_rewards"][:]))
            # Clip metrics lists so that all seeds have same number of episodes
            min_num_episodes = min([len(exp) for exp in exp_rewards_all])
            rewards = np.array([exp[:min_num_episodes] for exp in exp_rewards_all]).T
            # Aggregate metrics across all seeds
            plot_sd = np.std(rewards, axis=1)
            # Plot metrics
            sns.lineplot(
                x=range(min_num_episodes),
                y=plot_sd,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
        # Label plot
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.legend(loc="upper left", ncol=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward Std Dev")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        if INCLUDE_TITLES:
            ax.set_title(f"Episode Reward Variance by Episode ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_episode_reward_variance.png"),
            bbox_inches="tight",
        )


def visualize_episode_lengths(run_args):
    """Visualize average episode lengths across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_lengths_all = []
            for exp_dir in exp_dirs:
                exp_lengths_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_lengths_path):
                    with h5py.File(exp_lengths_path, "r") as file:
                        exp_lengths_all.append(list(file["episode_lengths"][:]))
            # Clip metrics lists so that all seeds have same number of episodes
            min_num_episodes = min([len(exp) for exp in exp_lengths_all])
            lengths = np.array([exp[:min_num_episodes] for exp in exp_lengths_all]).T
            # Aggregate metrics across all seeds
            plot_middle = np.mean(lengths, axis=1)
            lower_bound = np.sort(lengths, axis=1)[:, 1]
            upper_bound = np.sort(lengths, axis=1)[:, -2]
            # Plot metrics
            sns.lineplot(
                x=range(min_num_episodes),
                y=plot_middle,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                x=range(min_num_episodes),
                y1=lower_bound,
                y2=upper_bound,
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        # Label plot
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.legend(loc="upper left", ncol=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Episode Length")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        if INCLUDE_TITLES:
            ax.set_title(f"Average Episode Length by Episode ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_episode_lengths.png"),
            bbox_inches="tight",
        )


def visualize_cumulative_rewards(run_args):
    """Visualize cumulative rewards by global timestep across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_rewards_all = []
            for exp_dir in exp_dirs:
                exp_rewards_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_rewards_path):
                    with h5py.File(exp_rewards_path, "r") as file:
                        exp_rewards_all.append(list(file["cumulative_rewards"][:]))
            rewards = np.array(exp_rewards_all).T
            # Aggregate metrics across all seeds
            plot_middle = np.mean(rewards, axis=1)
            lower_bound = np.sort(rewards, axis=1)[:, 1]
            upper_bound = np.sort(rewards, axis=1)[:, -2]
            # Plot mean and range
            sns.lineplot(
                x=range(rewards.shape[0]),
                y=plot_middle,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                x=range(rewards.shape[0]),
                y1=lower_bound,
                y2=upper_bound, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        # Format tick labels
        xticks_step = 50000
        ax.set_xticks(range(0, rewards.shape[0] + xticks_step, xticks_step))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        # Label plot
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.legend(loc="upper left", ncol=1)
        ax.set_xlabel("Global Timestep")
        ax.set_ylabel("Cumulative Reward")
        if INCLUDE_TITLES:
            ax.set_title(f"Cumulative Reward by Timestep ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_cumul_rewards.png"),
            bbox_inches="tight",
        )


def visualize_model_losses(run_args):
    """Visualize losses for actor and critic models."""
    for env_id in run_args.env_ids:
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        fig.subplots_adjust(hspace=1)
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load loss data
            loss_data_all = {"critic1":[], "critic2":[], "actor":[], "alpha":[]}
            for exp_dir in exp_dirs:
                loss_data_path = os.path.join(exp_dir, "loss_data.h5")
                if os.path.exists(loss_data_path):
                    with h5py.File(loss_data_path, "r") as file:
                        loss_data_all["critic1"].append(list(file["critic1"][:]))
                        loss_data_all["critic2"].append(list(file["critic2"][:]))
                        loss_data_all["actor"].append(list(file["actor"][:]))
                        loss_data_all["alpha"].append(list(file["alpha"][:]))
            # Create matries of shape (n_steps, m_experiments)
            critic1_losses = np.array(loss_data_all['critic1']).T
            critic2_losses = np.array(loss_data_all['critic2']).T
            actor_losses = np.array(loss_data_all['actor']).T
            alpha_losses = np.array(loss_data_all['alpha']).T
            # Aggregate metrics across all seeds
            middle_critic1_loss = np.mean(critic1_losses, axis=1)
            middle_critic2_loss = np.mean(critic2_losses, axis=1)
            middle_actor_loss = np.mean(actor_losses, axis=1)
            middle_alpha_loss = np.mean(alpha_losses, axis=1)
            lower_critic1_loss = np.sort(critic1_losses, axis=1)[:, 1]
            lower_critic2_loss = np.sort(critic2_losses, axis=1)[:, 1]
            lower_actor_loss = np.sort(actor_losses, axis=1)[:, 1]
            lower_alpha_loss = np.sort(alpha_losses, axis=1)[:, 1]
            upper_critic1_loss = np.sort(critic1_losses, axis=1)[:, -2]
            upper_critic2_loss = np.sort(critic2_losses, axis=1)[:, -2]
            upper_actor_loss = np.sort(actor_losses, axis=1)[:, -2]
            upper_alpha_loss = np.sort(alpha_losses, axis=1)[:, -2]
            # Plot metrics
            sns.lineplot(
                x=range(len(middle_critic1_loss)),
                y=middle_critic1_loss,
                ax=axes[0],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
                legend=False,
            )
            sns.lineplot(
                x=range(len(middle_critic2_loss)),
                y=middle_critic2_loss,
                ax=axes[1],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
                legend=False,
            )
            sns.lineplot(
                x=range(len(middle_actor_loss)),
                y=middle_actor_loss,
                ax=axes[2],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
                legend=False,
            )
            sns.lineplot(
                x=range(len(middle_alpha_loss)),
                y=middle_alpha_loss,
                ax=axes[3],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
                legend=False,
            )
            axes[0].fill_between(
                x=range(len(middle_critic1_loss)),
                y1=lower_critic1_loss,
                y2=upper_critic1_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
            axes[1].fill_between(
                x=range(len(middle_critic2_loss)),
                y1=lower_critic2_loss,
                y2=upper_critic2_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
            axes[2].fill_between(
                x=range(len(middle_actor_loss)),
                y1=lower_actor_loss,
                y2=upper_actor_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
            axes[3].fill_between(
                x=range(len(middle_alpha_loss)),
                y1=lower_alpha_loss,
                y2=upper_alpha_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        for i in range(4):
            # axes[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Average Loss")
            # Format tick labels
            xticks_step = 50000
            axes[i].set_xticks(range(0, critic1_losses.shape[0] + xticks_step, xticks_step))
            axes[i].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
            axes[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        axes[0].legend(loc="upper right", ncol=2)
        axes[0].set_title("Critic 1 loss", loc="left")
        axes[1].set_title("Critic 2 loss", loc="left")
        axes[2].set_title("Actor loss", loc="left")
        axes[3].set_title("Alpha loss", loc="left")
        fig.align_ylabels()
        if INCLUDE_TITLES:
            fig.suptitle(f"Model Losses ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_losses.png"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    from causalex.config import RunArgs
    run_args = RunArgs()
    visualize_episode_rewards(run_args)
    visualize_episode_lengths(run_args)
    visualize_cumulative_rewards(run_args)
    visualize_model_losses(run_args)
    visualize_episode_reward_variance(run_args)
