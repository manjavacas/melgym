<p align="center">
    <img src="./docs/source/_static/images/logo.png" alt="logo" width="400"/>
</p>

<p align="center">
    <a href="#"><img src="https://badgen.net/github/release/manjavacas/melgym" alt="Release"></a>
    <img src="https://img.shields.io/badge/license-GPLv3-blue" alt="License">
    <a href="#"><img src="https://badgen.net/github/contributors/manjavacas/melgym" alt="Contributors"></a>
    <a href="https://melgym.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/melgym/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://doi.org/10.5281/zenodo.13885984"><img src="https://zenodo.org/badge/621343688.svg" alt="DOI"></a>
</p>

**MELGYM** is a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based interface designed to facilitate interactive control over [MELCOR](https://melcor.sandia.gov/) 1.8.6 simulations.

Every control functionality in MELCOR is determined by Control Functions (CFs). However, the batch execution mode of MELCOR makes it difficult to interactively control and modify functions under certain user-defined conditions. Control conditions are defined a priori and sometimes requires the concatenation of several CFs that must be done in an unfriendly way.

MELGYM allows the definition of external user-defined controllers, allowing the use of reinforcement learning agents or any other custom control algorithm.

<p align="center">
    <img src="./docs/source/_static/images/mdp-simp.png" alt="mdp" width="400"/>
</p>

## ⚙️ How it works?

MELGYM leverages MELCOR's restart capabilities to modify CFs dynamically during simulation cycles. Here's how it works:

1. **On-restart control**: MELGYM discretizes simulations by modifying the MELCOR input file just before warm starts are performed.
2. **Dynamic control**: the specified CFs are updated based on the latest simulation state, allowing real-time control.
3. **Integration with RL**: MELGYM can integrate reinforcement learning agents to optimize control strategies.

This approach enables interactive control over MELCOR simulations, overcoming the limitations of its batch execution mode.

<p align="center">
    <img src="./docs/source/_static/images/mdp.png" alt="mpd-2" width="500"/>
</p>

> Check the [MELGYM documentation](https://melgym.readthedocs.io/) for more detailed information.

## 🖥️ Setting up experiments

MELGYM environments adhere to the Gymnasium interface, and can be combined with DRL libraries such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

```python
import melgym
import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make('pressure-v0')

# Training
agent = PPO('MlpPolicy', env)
agent.learn(total_timesteps=10_000)

# Evaluation
obs, _ = env.reset()
done = trunc = False

while not (done or trunc):
    env.render()
    act, _ = agent.predict(obs)
    obs, rew, done, trunc, info = env.step(act)      

env.close()
```

## 🚀 Contributing

We welcome contributions to MELGYM! Here's how you can contribute:

1. Fork this repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/new-feature
   ```

Remember to check our [contributing](./CONTRIBUTING.md) guidelines.

## 🧰 Side projects

MELGYM rely on the auxiliar toolbox [MELKIT](https://github.com/manjavacas/melkit/). Feel free to help us improving both projects!

