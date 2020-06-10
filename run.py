import numpy as np
import pprint
import random
import pandas as pd
import matplotlib.pyplot as plt

import connect_four
import q_learning

train_settings = {
    "width": 5,
    "height": 5,
    "alg_type": "Q",
    "learning_rate":1,
    "discount_factor":0.9,
    "episode_epsilon":0.15,
    "num_iterations": 10000,
    "test_every": 1000,
    "experiment_name":"restarted",
    "other_notes":""
}

q = q_learning.QLearning(train_settings['width'], train_settings['height'])
df = q.train(train_settings)
#df = q.train("Q",1,0.9, 0.2, num_iterations=100000, test_every=10000, experiment_name = "Q_learning")