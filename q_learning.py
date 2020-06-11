import random
import pprint
import os
import pickle
import json
import sys

import pandas as pd
from tqdm import tqdm

import connect_four
import ai

sys.path.insert(1, 'MCTS/connect4/src')
from tournament import Tournament
from mcts.mcts_strategy import MctsStrategy
from game import Game
from player import Player
from board import Board

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
        self.opponents_dict = {
            "self_play": self.self_play_opponent,
            "random": self.random_opponent,
            "leftmost": self.leftmost_opponent,
            "mcts_5": self.mcts_5_opponent,
            "mcts_50": self.mcts_50_opponent,
            "mcts_100": self.mcts_100_opponent,
            "mcts_250": self.mcts_250_opponent,
            "mcts_500": self.mcts_500_opponent,
            "mcts_1000": self.mcts_1000_opponent
        }
        self.train_settings = None
        self.curr_training_iter = 0


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


    # given a board state tuple x, get argmax_u' Q(x,u') and max_u' Q(x,u')  over all valid u'
    # if there are no entries in Q for x, none will be returned
    def maxQ(self, x):
        best_control = None
        largest_Q = None
        controls = connect_four.ConnectFour.get_valid_inputs(x)
        random.shuffle(controls)
        for u in controls:
            if (x, u) in self.Q_function:
                if largest_Q is None or self.Q_function[(x, u)] > largest_Q:
                    largest_Q = self.Q_function[(x, u)]
                    best_control = u 

        return best_control, largest_Q


    # an opponent which uses the Q function as the next move
    def self_play_opponent(self, state_tuple):
        num_iterations = self.train_settings['num_iterations']
        curr_iteration = self.curr_training_iter
        #epsilon = 0.25
        # epsilon starts at 1 (all random) and moves to 0.05 as a function of the current iteration of training
        epsilon = 1-min(curr_iteration/(num_iterations/2), 1.0-0.05)

        if random.uniform(0,1) < epsilon:
            #move = random.choice(cf_game.get_valid_inputs(cf_game.board_state))
            move = random.choice(connect_four.ConnectFour.get_valid_inputs(state_tuple))
        else:
            move, _ = self.minQ(state_tuple)

        return move


    # an opponent which always just picks a random valid column as the next move
    def random_opponent (self, state_tuple):
        move = random.choice(connect_four.ConnectFour.get_valid_inputs(state_tuple))
        return move

    
    # an opponent which always pick the leftmost valid column as the next move
    def leftmost_opponent(self, state_tuple):
        move = min(connect_four.ConnectFour.get_valid_inputs(state_tuple))
        return move


    def mcts_opponent(self, state_tuple, rollout_limit):
        state_arr = [[str(t2) for t2 in t] for t in state_tuple]
        b = Board(self.width, self.height)
        b.board = state_arr[::-1]

        players = [Player('1'), Player('2')]
        game_number = 1
        g = Game(b, players, game_number)

        # fix playerid
        playerid = '2'
        mcts_s = MctsStrategy(rollout_limit)
        move = mcts_s.move(g, playerid)
        return move
    

    def mcts_5_opponent(self, state_tuple):
        return self.mcts_opponent(state_tuple, 5)
    def mcts_50_opponent(self, state_tuple):
        return self.mcts_opponent(state_tuple, 50)
    def mcts_100_opponent(self, state_tuple):
        return self.mcts_opponent(state_tuple, 100)
    def mcts_250_opponent(self, state_tuple):
        return self.mcts_opponent(state_tuple, 250)
    def mcts_500_opponent(self, state_tuple):
        return self.mcts_opponent(state_tuple, 500)
    def mcts_1000_opponent(self, state_tuple):
        return self.mcts_opponent(state_tuple, 1000)


    # generate episode with a epsilon-greedy policy derived from Q
    # epsilon in [0,1] is the percent chance that we pick a random control instead of the optimal
    # this also uses self-play (player 1 is AI, player 2 is AI's opponnent)
    def generate_episode(self, opponent, epsilon = 0):

        ai_episode = []
        opponent_episode = []
        curr_player = 1 if self.AI_first else 2
        cf_game = connect_four.ConnectFour(self.width, self.height)

        while True:

            curr_state_tuple = self.array2tuple(cf_game.board_state)

            # add state to Q-function for all valid controls with value 0 if it's not in yet
            # TODO: optimize this; this actually only needs to be outside of the AI if-statement below for self-play opponent
            for possible_control in cf_game.get_valid_inputs(cf_game.board_state):
                Q_entry = (curr_state_tuple, possible_control)
                if Q_entry not in self.Q_function:
                    self.Q_function[Q_entry] = 0

            # AI picks move using epsilon-greedy policy
            if curr_player == 1:
                if random.uniform(0,1) < epsilon:
                    move = random.choice(cf_game.get_valid_inputs(cf_game.board_state))
                else:
                    move, _ = self.minQ(curr_state_tuple)

                # update the episode
                ai_episode.append(curr_state_tuple)
                ai_episode.append(move)

            # opponent picks move using specified opponent type
            else:
                move = opponent(curr_state_tuple)
                # update the episode
                opponent_episode.append(curr_state_tuple)
                opponent_episode.append(move)


            cf_game.update_board(move, curr_player)
            game_outcome = cf_game.get_status()
            if game_outcome != 0:
                ai_episode.append(self.array2tuple(cf_game.board_state))
                opponent_episode.append(self.array2tuple(cf_game.board_state))
                break
            
            # switch to next player
            curr_player = 2 if curr_player == 1 else 1

        return ai_episode, opponent_episode, game_outcome


    # test the current Q_function on a list of opponents. Opponents list should be in form [[name, ]]
    #  returns a dict (keyed by test name) of dicts, 
    # where each dict specifies number of wins, losses, and ties for that opponent
    # also returns the unexplored rate
    # TODO: make better unexplored rate which also takes into account the control
    def test(self, iterations, opponent_names):

        test_results = {}
        unexplored_rate = 0
        num_moves = 0
        opponents = [self.opponents_dict[opponent_name] for opponent_name in opponent_names]

        for opponent_name, opponent in zip(opponent_names, opponents):
            print(opponent_name)
            wins = 0
            ties = 0
            losses = 0

            for i in tqdm(range(iterations)):
                curr_player = 1 if self.AI_first else 2
                cf_game = connect_four.ConnectFour(self.width, self.height)

                while True:

                    curr_state_tuple = self.array2tuple(cf_game.board_state)
                    # AI's turn
                    if curr_player == 1:
                        move, _ = self.minQ(curr_state_tuple)
                        # if Q does not have an entry for the current state x, just pick a random move
                        if move is None:
                            unexplored_rate += 1
                            move = random.choice(cf_game.get_valid_inputs(cf_game.board_state))
                        num_moves += 1

                    # opponent's turn
                    else:
                        move = opponent(curr_state_tuple)

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
            
                opponent_results = {"wins":wins, "ties":ties, "losses": losses}
                test_results[opponent_name] = opponent_results

        return test_results, unexplored_rate/num_moves


    # update step for loss-based Q learning
    def Q_learning_update(self, episode, game_outcome, train_settings):
        learning_rate = train_settings['learning_rate']
        discount_factor = train_settings['discount_factor']
        outcome2loss = {1:-1, 2:1, 3:-0.5}
        end_loss = outcome2loss[game_outcome]

        i=0
        while i < len(episode)-2:
            x = episode[i]
            u = episode[i+1]
            x_prime = episode[i+2]
            if i == len(episode)-3:
                loss = end_loss
                min_x_prime = 0
            else:
                loss = 0
                min_x_prime = self.minQ(x_prime)[1]
            

            # TODO: Also update states for oponnent?
            self.Q_function[(x,u)] = self.Q_function[(x,u)] + learning_rate * (loss + discount_factor*min_x_prime - self.Q_function[(x,u)])
            i += 2
    

    # in train_settings: 
    # - alg_type in {"SARSA, MC", "Q"}
    # - policy_opponent in {"self_play", "random", "leftmost"}
    # - test_opponents is a list with elements in {"self_play", "random", "leftmost"}
    # if experiment_name not specified, results not saved
    # epsilon in [0,1] is the percent chance that we pick a random control instead of the optimal
    def train(self, train_settings):
        self.train_settings = train_settings
        alg_type = train_settings['alg_type']
        learning_rate = train_settings['learning_rate']
        discount_factor = train_settings['discount_factor']
        episode_epsilon = train_settings['episode_epsilon']
        num_iterations = train_settings['num_iterations']
        test_every = train_settings['test_every']
        experiment_name = train_settings['experiment_name']
        test_opponents = train_settings['test_opponents']
        policy_opponent = self.opponents_dict[train_settings['policy_opponent']]
        

        # write settings dict to file
        experiment_dir = os.path.join(self.experiments_folder, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        with open(os.path.join(experiment_dir,'training_settings.txt'), 'w') as file:
            file.write(json.dumps(train_settings))

        train_df = pd.DataFrame()

        for iteration in tqdm(range(num_iterations)):
            self.curr_training_iter = iteration
            ai_episode, opponent_episode, game_outcome = self.generate_episode(policy_opponent, epsilon = episode_epsilon)

            if alg_type == "Q":
                self.Q_learning_update(ai_episode, game_outcome, train_settings)
                # Note: for simplicity we always also update Q value for opponent but
                # this actually only needs to be done when policy_opponent is self_play
                self.Q_learning_update(opponent_episode, game_outcome, train_settings)
            else:
                raise

            # periodically, print win rate against random opponents, update df, update Q function
            if iteration % test_every == 0 or iteration == num_iterations-1 :
                test_results, unexplored_rate = self.test(100, test_opponents)

                df_log = {"iteration": iteration, "Q_function_size":len(self.Q_function), "unexplored_states_rate": unexplored_rate}
                for opponent in test_results:
                    wins = test_results[opponent]["wins"]
                    ties = test_results[opponent]["ties"]
                    losses = test_results[opponent]["losses"]
                    df_log[opponent +"_win_rate"] = wins / (ties + losses + wins)
                tqdm.write(str(df_log))
                train_df = train_df.append(df_log, ignore_index = True)

                if experiment_name != "":
                    train_df.to_pickle(os.path.join(experiment_dir, "training_df.p"))
                    pickle.dump(self.Q_function, open(os.path.join(experiment_dir,"Q_function.p"), "wb"))

        return train_df
