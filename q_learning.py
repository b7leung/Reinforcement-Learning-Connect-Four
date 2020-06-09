import random
import pprint

import pandas as pd

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


    # converts a 2D array to a 2D tuple
    def array2tuple(self, array):
        return tuple(tuple(a) for a in array)


    # given a board state tuple x, get argmin_u' Q(x,u') and min_u' Q(x,u')  over all valid u'
    def minQ(self, x):
        best_control = None
        smallest_Q = None
        controls = list(range(self.width))
        random.shuffle(controls)
        for u in controls:
            if (x, u) in self.Q_function:
                if smallest_Q is None or self.Q_function[(x, u)] < smallest_Q:
                    smallest_Q = self.Q_function[(x, u)]
                    best_control = u 

        return best_control, smallest_Q


    # generate episode with a epsilon-greedy policy derived from Q
    # this also uses self-play (player 1 is AI, player 2 is AI's opponnent)
    def generate_episode(self, epsilon = 0):

        episode = []
        curr_player = 1 if self.AI_first else 2
        cf_game = connect_four.ConnectFour(self.width, self.height)

        while True:
            curr_state_tuple = self.array2tuple(cf_game.board_state)

            # add state to Q-function for all valid controls with value 0 if it's not in yet
            for possible_control in cf_game.get_valid_inputs():
                Q_entry = (curr_state_tuple, possible_control)
                if Q_entry not in self.Q_function:
                    self.Q_function[Q_entry] = 0

            # TODO: implement epsilon greedy
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
    def test(self, iterations):
        wins = 0
        ties = 0
        losses = 0

        for i in range(iterations):
            curr_player = 1 if self.AI_first else 2
            cf_game = connect_four.ConnectFour(self.width, self.height)

            while True:

                # AI's turn
                if curr_player == 1:
                    curr_state_tuple = self.array2tuple(cf_game.board_state)
                    move = self.minQ(curr_state_tuple)
                    # TODO: check if move is legal and in Q?

                # random opponent's turn
                else:
                    move = random.choice(cf_game.get_valid_inputs())

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
        
        return wins, ties, losses


    def train(self, learning_rate, discount_factor, num_iterations = 1, test_every = 100):
        
        #outcome2reward = {1:1, 2:-1, 3:0.5}
        outcome2loss = {1:-1, 2:1, 3:-0.5}
        train_df = pd.DataFrame()

        for iteration in range(num_iterations):

            episode, game_outcome = self.generate_episode()
            end_loss = outcome2loss[game_outcome]

            i=0
            while i < len(episode)-2:
                x = episode[i]
                u = episode[i+1]
                x_prime = episode[i+2]
                if i == len(episode)-3:
                    loss = end_loss
                else:
                    loss = 0

                self.Q_function[(x,u)] = self.Q_function[(x,u)] + learning_rate * (loss + discount_factor*self.minQ(x)[1] - self.Q_function[(x,u)])
                i += 2

            # printing win rate against random opponent
            if iteration % test_every == 0:
                wins, ties, losses = self.test(100)
                win_rate = wins / (ties + losses)
                df_log = {"iteration": iteration, "win_rate": win_rate, "Q_function_size":len(self.Q_function)}
                print(df_log)
                train_df = train_df.append(df_log, ignore_index = True)

        return train_df