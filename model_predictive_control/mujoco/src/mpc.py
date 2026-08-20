import numpy as np

import gymnasium as gym


def get_state(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot the MuJoCo physical state (qpos, qvel) of an env."""
    unwrapped = env.unwrapped
    return unwrapped.data.qpos.copy(), unwrapped.data.qvel.copy()


def set_state(env: gym.Env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Restore a previously captured MuJoCo physical state onto env."""
    env.reset()
    env.unwrapped.set_state(qpos, qvel)


def sample_trajectory_random(env: gym.Env, n_actions: int) -> list[gym.Space]:
    """Sample n_actions randomly from the environment's action space.
    
    Args:
        env (gym.Env): The environment to sample from
        n_actions (int): The number of actions to sample
    
    Returns:
        list of actions
    """
    return [env.action_space.sample() for _ in range(n_actions)]


def evaluate_trajectory(
    sim_env: gym.Env,
    policy: str,
    horizon: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Calculate the total reward obtained by following a single trajectory.

    Args:
        sim_env (gym.Env): A dedicated simulation environment used only for planning rollouts
            (must not be the real environment being stepped in the main loop)
        policy (str): The type of policy to evaluate (default: "random")
        horizon (int): The number of steps to follow the trajectory (default: 100)
        qpos (np.ndarray): Starting joint positions to roll the trajectory out from
        qvel (np.ndarray): Starting joint velocities to roll the trajectory out from

    Returns:
        tuple(np.ndarray, float): The first action of the trajectory and the total reward 
        obtained by the policy
    """
    ALLOWED_POLICIES = ["random"]
    if policy.lower() not in ALLOWED_POLICIES:
        raise NotImplementedError(f"Policy {policy} not implemented")

    # Generate trajectory
    actions = None
    if policy.lower() == "random":
        actions = sample_trajectory_random(sim_env, horizon)
    if actions is None:
        raise ValueError("No actions were generated for the trajectory")

    # Evaluate trajectory, always starting from the real env's current state
    total_reward = 0.0
    set_state(sim_env, qpos, qvel)
    for action in actions:
        observation, reward, terminated, truncated, info = sim_env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return actions[0], total_reward


def identify_next_action(
    sim_env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    policy: str = "random",
    eval_n_traj: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    """
    Identify the best action for timestep t+1 by evaluating n trajectories of horizon 
    length h. The best action is the first action of the trajectory that yields the 
    highest total reward over the horizon h.
    
    Args:
        sim_env (gym.Env): A dedicated simulation environment used only for planning rollouts
            (must not be the real environment being stepped in the main loop)
        qpos (np.ndarray): The real environment's current joint positions to plan from
        qvel (np.ndarray): The real environment's current joint velocities to plan from
        policy (str): The type of policy to evaluate (default: "random")
        eval_n_traj (int): The number of trajectories to evaluate (default: 20)
        horizon (int): The number of steps to follow the trajectory (default: 100)

    Returns:
        np.ndarray: The best action for timestep t+1, evaluated over n trajectories of horizon h
    """
    best_action = None
    best_reward = float("-inf")

    for traj_index in range(eval_n_traj):
        try:
            first_action, total_reward = evaluate_trajectory(
                sim_env, policy=policy, horizon=horizon, qpos=qpos, qvel=qvel
            )
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = first_action
        except Exception as e:
            print(f"Error evaluating trajectory {traj_index}: {e}")

    return best_action


if __name__ == "__main__":
    env = gym.make("Ant-v5", render_mode="rgb_array")
    sim_env = gym.make("Ant-v5") # For MPC rollouts
    env.reset(seed=9999)
    qpos, qvel = get_state(env)
    best_action = identify_next_action(sim_env, qpos, qvel)
    print(f"Best action for timestep t+1: {best_action}")