import numpy as np
import pprint
import random
import pandas as pd
import matplotlib.pyplot as plt

import connect_four
import q_learning

random_play_train_settings = {
    "width": 5,
    "height": 4,
    "alg_type": "Q",
    "learning_rate":0.9,
    "discount_factor":1,
    "episode_epsilon":0.1,
    "num_iterations": 6000000,
    "test_every": 200000,
    "test_opponents": ["random", "mcts_25", "mcts_50"],
    "policy_opponent": "random",
    "experiment_name":"Q_random_play",
    "other_notes":""
}

self_play_tuned_train_settings = {
    "width": 5,
    "height": 4,
    "alg_type": "Q",
    "learning_rate":0.9,
    "discount_factor":1,
    "episode_epsilon":0.1,
    "num_iterations": 6000000,
    "test_every": 100000,
    "test_opponents": ["random", "mcts_25", "mcts_50"],
    "policy_opponent": "random",
    "experiment_name":"Q_random_play",
    "other_notes":""
}

MC_train_settings = {
    "width": 5,
    "height": 4,
    "alg_type": "MC",
    "learning_rate":0.8,
    "discount_factor":1,
    "episode_epsilon":0.05,
    "num_iterations": 6000000,
    "test_every": 100000,
    "test_opponents": ["random", "mcts_25", "mcts_50"],
    "policy_opponent": "self_play",
    "experiment_name":"MC_lr08_ep05",
    "other_notes":""
}
# !!!!!!! REMINDER: TODO: check experiment name

train_settings = random_play_train_settings

pprint.pprint(train_settings)
q = q_learning.QLearning(train_settings['width'], train_settings['height'])
df = q.train(train_settings)