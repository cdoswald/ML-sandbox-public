import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

from config import RunArgs, ExperimentArgs
from models import Actor
from utils import save_video

#TODO: update for CausalEx refactor

def eval_SAC(args, actor_path):
    """Evaluate trained SAC agent"""

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(args.env_id, render_mode="rgb_array")
    videos_dir = os.path.join(args.run_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    # env = RecordVideo(
    #     env,
    #     f"videos/{args.run_name}",
    #     episode_trigger=lambda episode_id: episode_id % 20 == 0,
    # )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # Load model
    actor = Actor(env).to(device)
    actor.load_state_dict(torch.load(actor_path, weights_only=True))

    # TRY NOT TO MODIFY: start the game
    episode_rewards = []
    episode_lengths = []
    episode_frames = []
    episode_reward = 0
    episode_length = 0
    episode_idx = 0
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.eval_timesteps):
        # ALGO LOGIC: put action logic here
        _, _, mean_action = actor.get_action(torch.Tensor(obs).to(device))
        actions = mean_action.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Render and record frame (VideoRecorder not working correctly)
        episode_frames.append(env.render().astype(np.uint8))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        episode_reward += rewards
        episode_length += 1
        if "episode" in infos:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Reset environment on termination or truncation
        if terminations or truncations:
            obs, _ = env.reset()
            print(
                f'Global step: {global_step}; '+
                f'episode reward: {round(episode_reward, 3)}; '+
                f'length = {episode_length}'
            )
            episode_reward = 0
            episode_length = 0
            # Save episode video
            if episode_idx < 10:
                save_path = os.path.join(
                    videos_dir,
                    f"env_{args.env_id}_mode_{args.cx_mode}_seed_{args.seed}_episode_{episode_idx}.mp4"
                )
                save_video(episode_frames, save_path)
            # Increment episode index and reset frames list
            episode_idx += 1
            episode_frames = []
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

    # Clean up
    env.close()


if __name__ == "__main__":

    # Record start time
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()

    # Create experiment arguments
    eval_env = "Ant-v4"
    eval_cx_mode = "causal"
    eval_seed = 523
    eval_actor_path = os.path.join(
        run_args.run_dir,
        f"env_{eval_env}_mode_{eval_cx_mode}_seed_{eval_seed}",
        "actor.pth"
    )
    exp_args = ExperimentArgs(
        env_id=eval_env,
        cx_mode=eval_cx_mode,
        seed=eval_seed,
    )
    eval_SAC(exp_args, eval_actor_path)

    # Record end time and report progress
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Eval start time: {start_time} \nEval end time: {end_time}")
