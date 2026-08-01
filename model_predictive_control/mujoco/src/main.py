import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium as gym


def main() -> None:

    SEED = 9999
    env = gym.make("Ant-v5", render_mode="rgb_array")

    observation, info = env.reset(seed=SEED)
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)
    print("Reset info keys:", sorted(info.keys()))

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step + 1} reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            observation, info = env.reset(seed=SEED)

    print("Final observation shape:", observation.shape)
    print("Total reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()
