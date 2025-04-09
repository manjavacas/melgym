import melgym
import gymnasium as gym


def rand_control(env, n_episodes=1):
    """
    Random controller for testing purposes.

    Args:
        env (gym.Env): Gymnasium environment.
        n_episodes (int): Number of episodes to run. Default is 1.
    """
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Info: {info}")
            env.render()


if __name__ == '__main__':
    env = gym.make('pressure-v0')
    rand_control(env)
    env.close()
