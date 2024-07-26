.. _examples:

Examples
********

Manual training and testing of a DRL agent
==========================================

Here is an example script where a DRL agent is trained and evaluated in the MELGYM **branch_1** environment:

.. literalinclude:: ../_static/code/drl.py

The instantiation of the environment is similar to any inherited Gymnasium environment. On the other hand, the Stable-Baselines3 implementation of a TD3 agent is used, which is instantiated and trained during a number of user-specified timesteps. Once trained, we carry out the evaluation of the model during a complete episode.

Automatic training and testing
==============================

MELGYM facilitates the automated execution of DRL experiments through the use of a default configuration file. To do this, we use a *yaml* file like the following, which the provided `run_drl.py <../../../../run_drl.py>`_ script takes as input.

.. literalinclude:: ../_static/code/sample-config.yml

This configuration includes the tasks to be performed (*train* and/or *test*), the parameters of the environment and the algorithm used, as well as auxiliary configuration, such as the wrappers and callbacks to be used, or the paths of the generated output directories.

In this way, the execution of ``$ run_drl.py -conf config.yaml`` should be enough to run a complete DRL experiment.

.. tip:: A `Makefile <../../../../Makefile>`_ is also provided to further facilitate the rapid execution of experiments.