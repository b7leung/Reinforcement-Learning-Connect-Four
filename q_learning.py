import random
import pprint
import os
import pickle

import pandas as pd
from tqdm import tqdm

import connect_four
import ai

class QLearning:

    def __init__(self, width, height, AI_first = True, seed = 0):

        # stores tuples (game_board_state (2d tuple), input_column (int))
        # This starts off as empty, and states are created as episodes are generated
        self.Q_function = {}
        self.AI_first = AI_first
        self.width = width
        self.height = height
        random.seed(seed)
        self.experiments_folder = "./experiments"
        if not os.path.exists(self.experiments_folder):
            os.makedirs(self.experiments_folder)


    # converts a 2D array to a 2D tuple
    def array2tuple(self, array):
        return tuple(tuple(a) for a in array)


    # given a board state tuple x, get argmin_u' Q(x,u') and min_u' Q(x,u')  over all valid u'
    # if there are no entries in Q for x, none will be returned
    def minQ(self, x):
        best_control = None
        smallest_Q = None
        controls = connect_four.ConnectFour.get_valid_inputs(x)
        random.shuffle(controls)
        for u in controls:
            if (x, u) in self.Q_function:
                if smallest_Q is None or self.Q_function[(x, u)] < smallest_Q:
                    smallest_Q = self.Q_function[(x, u)]
                    best_control = u 

        return best_control, smallest_Q


    # generate episode with a epsilon-greedy policy derived from Q
    # epsilon in [0,1] is the percent chance that we pick a random control instead of the optimal
    # this also uses self-play (player 1 is AI, player 2 is AI's opponnent)
    def generate_episode(self, epsilon = 0):

        episode = []
        curr_player = 1 if self.AI_first else 2
        cf_game = connect_four.ConnectFour(self.width, self.height)

        while True:
            curr_state_tuple = self.array2tuple(cf_game.board_state)

            # add state to Q-function for all valid controls with value 0 if it's not in yet
            for possible_control in cf_game.get_valid_inputs(cf_game.board_state):
                Q_entry = (curr_state_tuple, possible_control)
                if Q_entry not in self.Q_function:
                    self.Q_function[Q_entry] = 0

            if random.uniform(0,1) < epsilon:
                move = random.choice(cf_game.get_valid_inputs(cf_game.board_state))
            else:
                move, _ = self.minQ(curr_state_tuple)

            # if it's AI's turn, update the episode
            if curr_player == 1:
                episode.append(curr_state_tuple)
                episode.append(move)

            cf_game.update_board(move, curr_player)
            #cf_game.print_board()
            game_outcome = cf_game.get_status()
            if game_outcome != 0:
                episode.append(self.array2tuple(cf_game.board_state))
                break
            
            # switch to next player
            curr_player = 2 if curr_player == 1 else 1

        return episode, game_outcome


    # test the current Q_function on a random player. returns the number of wins, losses, and ties
    # TODO: make better unexplored rate which also takes into account the control
    def test(self, iterations):
        wins = 0
        ties = 0
        losses = 0
        unexplored_rate = 0
        num_moves = 0

        for i in range(iterations):
            curr_player = 1 if self.AI_first else 2
            cf_game = connect_four.ConnectFour(self.width, self.height)

            while True:

                # AI's turn
                if curr_player == 1:
                    curr_state_tuple = self.array2tuple(cf_game.board_state)
                    move, _ = self.minQ(curr_state_tuple)
                    # if Q does not have an entry for the current state x, just pick a random move
                    if move is None:
                        unexplored_rate += 1
                        move = random.choice(cf_game.get_valid_inputs(cf_game.board_state))
                    num_moves += 1

                # random opponent's turn
                else:
                    move = random.choice(cf_game.get_valid_inputs(cf_game.board_state))

                cf_game.update_board(move, curr_player)
                game_outcome = cf_game.get_status()
                if game_outcome != 0:
                    if game_outcome == 1:
                        wins += 1
                    elif game_outcome == 2:
                        losses += 1 
                    else:
                        ties += 1
                    break

                # switch to next player
                curr_player = 2 if curr_player == 1 else 1
        
        return wins, ties, losses, unexplored_rate/num_moves


    # alg_type in {"SARSA, MC", "Q"}
    # if experiment_name not specified, results not saved
    # epsilon in [0,1] is the percent chance that we pick a random control instead of the optimal
    def train(self, alg_type, learning_rate, discount_factor, episode_epsilon, num_iterations = 1, test_every = 10000, experiment_name = ""):
        
        #outcome2reward = {1:1, 2:-1, 3:0.5}
        outcome2loss = {1:-1, 2:1, 3:-0.5}
        train_df = pd.DataFrame()

        for iteration in tqdm(range(num_iterations)):
            
            episode, game_outcome = self.generate_episode(episode_epsilon)
            end_loss = outcome2loss[game_outcome]

            # updating Q_value function from policy
            i=0
            while i < len(episode)-2:
                x = episode[i]
                u = episode[i+1]
                x_prime = episode[i+2]
                if i == len(episode)-3:
                    loss = end_loss
                else:
                    loss = 0

                # TODO: Also update states for oponnent?
                self.Q_function[(x,u)] = self.Q_function[(x,u)] + learning_rate * (loss + discount_factor*self.minQ(x)[1] - self.Q_function[(x,u)])
                i += 2

            # periodically, print win rate against random opponents, update df, update Q function
            if iteration % test_every == 0:
                wins, ties, losses, unexplored_rate = self.test(100)
                win_rate = wins / (ties + losses + wins)
                tie_rate = ties / (ties + losses + wins)
                df_log = {"iteration": iteration, "win_rate": win_rate, "tie_rate":tie_rate, "Q_function_size":len(self.Q_function), "unexplored_states_rate": unexplored_rate}
                tqdm.write(str(df_log))
                train_df = train_df.append(df_log, ignore_index = True)

                if experiment_name != "":
                    experiment_dir = os.path.join(self.experiments_folder, experiment_name)
                    if not os.path.exists(experiment_dir):
                        os.makedirs(experiment_dir)
                    train_df.to_pickle(os.path.join(experiment_dir, "training_df.p"))
                    pickle.dump(self.Q_function, open(os.path.join(experiment_dir,"Q_function.p"), "wb"))

        return train_df
