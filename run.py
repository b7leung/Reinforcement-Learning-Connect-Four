import numpy as np
import pprint
import random
import pandas as pd
import matplotlib.pyplot as plt

import connect_four
import q_learning

leftmost_train_settings = {
    "width": 5,
    "height": 4,
    "alg_type": "Q",
    "learning_rate":1,
    "discount_factor":1,
    "episode_epsilon":0.15,
    "num_iterations": 1000,
    "test_every": 10,
    "test_opponents": ["random", "leftmost"],
    "policy_opponent": "leftmost",
    "experiment_name":"leftmost_policy",
    "other_notes":""
}

random_policy_tuned_train_settings = {
    "width": 5,
    "height": 4,
    "alg_type": "Q",
    "learning_rate":0.9,
    "discount_factor":1,
    "episode_epsilon":0.15,
    "num_iterations": 20000000,
    "test_every": 50000,
    "test_opponents": ["random", "leftmost"],
    "policy_opponent": "self_play",
    "experiment_name":"5x4_tuned",
    "other_notes":""
}

self_play_tuned_train_settings = {
    "width": 5,
    "height": 4,
    "alg_type": "Q",
    "learning_rate":0.8,
    "discount_factor":1,
    "episode_epsilon":0.15,
    "num_iterations": 6000000,
    "test_every": 50000,
    "test_opponents": ["random", "leftmost", "mcts_5", "mcts_50", "mcts_250", "mcts_500"],
    #"test_opponents": ["random", "leftmost", "mcts_5", "mcts_50", "mcts_100", "mcts_250", "mcts_500", "mcts_1000"],
    "policy_opponent": "self_play",
    "experiment_name":"5x4_tuned",
    "other_notes":""
}

train_settings = self_play_tuned_train_settings

pprint.pprint(train_settings)
q = q_learning.QLearning(train_settings['width'], train_settings['height'])
df = q.train(train_settings)