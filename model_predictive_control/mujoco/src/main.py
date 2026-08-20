from dataclasses import dataclass
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium as gym

import mpc
from utils_video import save_video_from_frame_list


@dataclass
class Config():
    SEED: int = 9999
    TOTAL_ENV_STEPS: int = 1000

    VIDEOS_DIR: str = "videos"
    FPS: int = 30
    N_FRAMES_AFTER_FAILURE: int = 100
    
    def __post_init__(self):
        os.makedirs(self.VIDEOS_DIR, exist_ok=True)


def main(args: Config) -> None:

    env = gym.make("Ant-v5", render_mode="rgb_array")
    sim_env = gym.make("Ant-v5") # For MPC rollouts

    observation, info = env.reset(seed=args.SEED)
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)
    print("Reset info keys:", sorted(info.keys()))

    all_iteration_rewards = {}
    iteration = 0
    iter_total_reward = 0.0
    iter_frames = []

    for step in range(args.TOTAL_ENV_STEPS):

        # action = env.action_space.sample()
        qpos, qvel = mpc.get_state(env)
        action = mpc.identify_next_action(sim_env, qpos, qvel)
        observation, reward, terminated, truncated, info = env.step(action)
        iter_total_reward += reward

        frame = env.render()
        iter_frames.append(frame)

        print(
            f"iteration={iteration} "
            f"step={step + 1} "
            f"reward={reward:.3f} "
            f"total_reward={iter_total_reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated or ((step+1) == args.TOTAL_ENV_STEPS):
            for i in range(args.N_FRAMES_AFTER_FAILURE):
                frame = env.render()
                iter_frames.append(frame)

            save_video_from_frame_list(
                iter_frames,
                os.path.join(args.VIDEOS_DIR, f"output_video_iter{iteration}.mp4"),
                fps=args.FPS
            )
            iter_frames = []

            all_iteration_rewards[iteration] = iter_total_reward
            iter_total_reward = 0.0

            observation, info = env.reset(seed=args.SEED)

            print(f"Iteration #{iteration} total reward:", all_iteration_rewards[iteration])

            iteration += 1
            print(f"Starting iteration #{iteration}")

    best_iteration = max(all_iteration_rewards, key=all_iteration_rewards.get)
    best_reward = all_iteration_rewards[best_iteration]
    print(f"Best reward {best_reward} at iteration {best_iteration}")
    print(f"All rewards:\n {', '.join([f'Iter {k}:{round(float(v), 3)}' for k,v in all_iteration_rewards.items()])}")
    env.close()
    sim_env.close()


if __name__ == "__main__":
    args = Config()
    main(args)
