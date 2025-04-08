.. _examples:

Examples
********

Random Controller
=================

The following example demonstrates the use of a **random controller**, which can be used to test any MELGYM environment. This is useful for sanity checks or to ensure that the environment behaves as expected before integrating more complex controllers.

.. literalinclude:: ../_static/code/rand_controller.py

Training and Evaluation of a DRL Agent
======================================

The following script showcases how to train and evaluate a reinforcement learning agent within the MELGYM environment **pressures-v0**:

.. literalinclude:: ../_static/code/drl_controller.py

In this example, the agent learns to control the **pressure** of a specific control volume based on its current pressure values. The control action involves adjusting the **exhaust flow rate** accordingly.

Environment instantiation follows the standard procedure of any `Gymnasium`-based environment. For the agent, we use the `Stable-Baselines3` implementation of **Proximal Policy Optimization (PPO)**. The agent is initialized and trained for a user-defined number of timesteps.  
Once the training phase is complete, the model is evaluated by running it through an entire simulation episode.

.. tip:: This example provides a practical starting point for building and testing custom DRL controllers in MELGYM environments.
