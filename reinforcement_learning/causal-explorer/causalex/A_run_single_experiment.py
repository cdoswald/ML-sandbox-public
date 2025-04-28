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
from causalex.models import Actor, SoftQNetwork
from causalex.utils import save_dict_to_hdf5


# Based on CleanRL SAC implementation
def run_single_experiment(args):

    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Starting experiment {args.exp_dir} on PID {os.getpid()} at {current_time}.')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    # Prepopulate replay buffer
    if args.cx_mode.lower() == "causal":
        rb = prepopulate_buffer_causal(env, rb, args)
    elif args.cx_mode.lower() == "random":
        rb = prepopulate_buffer_random(env, rb, args)
    elif args.cx_mode.lower() == "random_with_noise":
        rb = prepopulate_buffer_random(env, rb, args, noise_scale=args.noise_scale)
    else:
        raise ValueError(
            f"CX mode'{args.cx_mode} not recognized. Double-check configuration arguments."
        )

    # TRY NOT TO MODIFY: start the game
    loss_dict = {"critic1":[], "critic2":[], "actor":[], "alpha":[]}
    metrics_dict = {"cumulative_rewards":[], "episode_rewards":[], "episode_lengths":[]}
    cumulative_reward = np.float32(0)
    episode_reward = np.float32(0)
    episode_length = 0
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.train_timesteps):
        # ALGO LOGIC: put action logic here
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy() if not (terminations or truncations) else obs
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break
        rounded_reward = np.float32(round(rewards, 5))
        cumulative_reward += rounded_reward
        metrics_dict["cumulative_rewards"].append(cumulative_reward)

        episode_reward += rounded_reward
        episode_length += 1
        if "episode" in infos:
            metrics_dict["episode_rewards"].append(episode_reward)
            metrics_dict["episode_lengths"].append(episode_length)
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")

        # Reset environment on termination or truncation
        if terminations or truncations:
            obs, _ = env.reset(seed=args.seed)
            episode_reward = np.float32(0)
            episode_length = 0
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions) 
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # record losses
            loss_dict["critic1"].append(np.float32(round(qf1_loss.item(), 5)))
            loss_dict["critic2"].append(np.float32(round(qf2_loss.item(), 5)))

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                temp_actor_losses = []
                temp_alpha_losses = []
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    temp_actor_losses.append(np.float32(round(actor_loss.item(), 5)))

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        temp_alpha_losses.append(np.float32(round(alpha_loss.item(), 5)))

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                # record losses
                loss_dict["actor"] += temp_actor_losses
                loss_dict["alpha"] += temp_alpha_losses

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # Save models and metrics (periodically or after final training step)
        if (global_step % args.save_frequency == 0) or (global_step == (args.train_timesteps - 1)):
            # Models
            torch.save(actor.state_dict(), os.path.join(args.exp_dir, "actor.pth"))
            torch.save(qf1.state_dict(), os.path.join(args.exp_dir, "qf1.pth"))
            torch.save(qf2.state_dict(), os.path.join(args.exp_dir, "qf2.pth"))
            # Metrics
            metrics_path = os.path.join(args.exp_dir, "metrics.h5")
            save_dict_to_hdf5(metrics_path, metrics_dict)
            # Loss data
            loss_data_path = os.path.join(args.exp_dir, "loss_data.h5")
            save_dict_to_hdf5(loss_data_path, loss_dict)
            # Reset metric lists to reduce memory usage
            loss_dict = {"critic1":[], "critic2":[], "actor":[], "alpha":[]}
            metrics_dict = {"cumulative_rewards":[], "episode_rewards":[], "episode_lengths":[]}

    # Clean up
    env.close()
