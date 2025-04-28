"""Compare success of Q-models in predicting rewards (given state and action pairs)
for causal vs. random information."""

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

from causalex.causal_explorer import prepopulate_buffer_causal, prepopulate_buffer_random
from causalex.models import SoftQNetwork
from causalex.utils import calculate_n_interactions


def train_Q_model(args):

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set up environment
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)

    # Set up models
    qf1 = SoftQNetwork(env).to(device)
    qf1_optimizer = optim.Adam(qf1.parameters(), lr=args.q_lr)

    # Prepopulate replay buffer
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    n_interact = calculate_n_interactions(env.action_space.shape[0])
    args.prepopulate_buffer_hard_cap = args.buffer_size
    args.max_steps_per_interact = args.buffer_size // n_interact
    if args.cx_mode.lower() == "causal":
        rb = prepopulate_buffer_causal(env, rb, args)
    elif args.cx_mode.lower() == "random":
        rb = prepopulate_buffer_random(env, rb, args)
    elif args.cx_mode.lower() == "random_with_noise":
        rb = prepopulate_buffer_random(env, rb, args, noise_scale=args.noise_scale)
    else:
        raise ValueError(f"Unrecognized cx_mode: {args.cx_mode}")
    empty_entries = np.sum(np.sum(rb.observations, axis=1) == 0)
    print(f"Empty entries/buffer size: {empty_entries}/{args.buffer_size}")

    # Train Q-model to predict reward conditional on state and action
    epoch_losses = []
    for epoch_i in range(args.n_epochs):
        batch_losses = []
        for _ in range(args.n_samples_per_epoch):
            data = rb.sample(args.batch_size)
            qf1_pred_rewards = qf1(data.observations, data.actions)
            qf1_loss = F.mse_loss(qf1_pred_rewards, data.rewards)
            qf1_optimizer.zero_grad()
            qf1_loss.backward()
            qf1_optimizer.step()
            batch_losses.append(qf1_loss.item())

        # Record avg epoch loss and print progress
        avg_epoch_loss = np.mean(batch_losses)
        epoch_losses.append(avg_epoch_loss)
        print(f"Completed epoch {epoch_i + 1} with avg loss {round(avg_epoch_loss, 5)}")

    # Save epoch losses to disk
    file_name = f"epoch_losses_env_{args.env_id}_mode_{args.cx_mode}_buffer_{args.buffer_size}_seed_{args.seed}.json"
    with open(os.path.join(args.loss_data_dir, file_name), "w") as io:
        json.dump(epoch_losses, io)
        print(f"Successfully saved {file_name} file")
