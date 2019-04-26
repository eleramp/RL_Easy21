#!/usr/bin/env/python
# -*- coding: utf-8 -*-
import numpy as np
"""
    The following script stores
    the configuration parameters for:
        - Monte-Carlo mc_control
        - Temporal-Difference mc_control
        - Linear Approximation control
"""

mc_params = {
    # instantiate epsilon_t for e-greedy exploration strategy
    'N0': 200,
    'r_gamma': 1,
    'n_episodes': int(10e5)
}

td_params = {
    # instantiate epsilon_t for e-greedy exploration strategy
    'lambdas': np.arange(0,1.1,0.1),
    'N0': 200,
    'r_gamma': 1,
    'n_episodes': int(1e3)
}

approx_params = {
    # instantiate epsilon_t for e-greedy exploration strategy
    'lambdas': np.arange(0,1.1,0.1),
    'epsilon': 0.05,
    'step_size': 0.01,
    'dealer_range_f': zip(range(1,8,3), range(4, 11, 3)),
    'player_range_f': zip(range(1,17,3), range(6, 22, 3)),
    'r_gamma': 1,
    'n_episodes': int(1e3)
}
