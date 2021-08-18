An AI that learns to play connect four, using reinforcement learning. For details, please refer to the [report](https://b7leung.github.io/files/Connect%20Four.pdf).


* connect_four.py: The code for the connect four game mechanics, including managing the board state, updating it with a move input, and checking for win conditions (ie, if the current board states has a vertical, horizontal, or diagonal four in a row, as well as if the board state is a tie)

* q_learning.py: The code for learning the Q function for the Connect Four AI. Contains the code for generating episodes with self-play. Also contains code for the main training loop (ie, implementing the MC-TS and Q-Learning algorithm) and testing the trained Connect Four AI.

* run.py: Used to actually start the RL experiments. 

* Plots.py: A jupyter notebook used to plot the results.

Note: For the code to be runnable, the matplotlib, numpy, pandas, and tqdm python packages are required. Also, clone the github repository from https://github.com/codebox/connect4. The code from this repo is used solely for evaluating my trained RL Connect Four agent at test time, as a black box. It is not used for the training, or anything reinforcement learning related (all the things related to RL is implemented by me).
