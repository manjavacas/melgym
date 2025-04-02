import melgym
import gymnasium as gym

def test_melgym_env():
    """
    Test the MELGYM environment.
    """
    env = gym.make('pressure')
    
    # Test reset
    obs, info = env.reset()
    print(obs, info)
    
    # Test step
    action = env.action_space.sample()
    print(env.step(action))

test_melgym_env()