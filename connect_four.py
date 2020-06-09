
import numpy as np
import pprint

class ConnectFour:

    def __init__(self, board_cols = 7, board_rows = 6):

        self.board_state = [[0 for i in range(board_cols)] for j in range(board_rows)]
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.marker_chars = {0:".", 1: "o", 2: "x"}


    def print_board(self):
        for i in range(self.board_rows):
            col_str = "|  "
            for j in range(self.board_cols):
                board_num = self.board_state[i][j]
                marker_char = self.marker_chars[board_num]
                col_str += marker_char + "  "
            print(col_str+"|")
        print("".join(["=" for c in range(self.board_cols*3+4)]))


    # player name: 1 if user, 2 if computer
    def update_board(self, column_to_place, player_name):

        row_to_place = 0
        while row_to_place < self.board_rows-1 and self.board_state[row_to_place+1][column_to_place] == 0:
            row_to_place += 1
        self.board_state[row_to_place][column_to_place] = player_name


    # returns a list of the valid columns (columns which are not full)
    def get_valid_inputs(self):
        valid_inputs = []
        for i, entry in enumerate(self.board_state[0]):
            if entry == 0:
                valid_inputs.append(i)

        return valid_inputs


    # returns 0 if no winner yet, 1 if user won, 2 if user lost (comp won), 3 if tie
    def get_status(self):

        # horizontal check
        for i in range(self.board_cols-3):
            for j in range(self.board_rows):
                if self.board_state[j][i] == 1 and self.board_state[j][i+1] == 1 and self.board_state[j][i+2] == 1 and self.board_state[j][i+3] == 1:
                    return 1
                elif self.board_state[j][i] == 2 and self.board_state[j][i+1] == 2 and self.board_state[j][i+2] == 2 and self.board_state[j][i+3] == 2:
                    return 2
        
        # vertical check
        for i in range(self.board_cols):
            for j in range(self.board_rows-3):
                if self.board_state[j][i] == 1 and self.board_state[j+1][i] == 1 and self.board_state[j+2][i] == 1 and self.board_state[j+3][i] == 1:
                    return 1
                elif self.board_state[j][i] == 2 and self.board_state[j+1][i] == 2 and self.board_state[j+2][i] == 2 and self.board_state[j+3][i] == 2:
                    return 2

        # backwards diagonal check
        for i in range(3,self.board_cols):
            for j in range(3,self.board_rows):
                if self.board_state[j][i] == 1 and self.board_state[j-1][i-1] == 1 and self.board_state[j-2][i-2] == 1 and self.board_state[j-3][i-3] == 1:
                    return 1
                elif self.board_state[j][i] == 2 and self.board_state[j-1][i-1] == 2 and self.board_state[j-2][i-2] == 2 and self.board_state[j-3][i-3] == 2:
                    return 2

        # forwards diagonal check
        for i in range(3,self.board_cols):
            for j in range(self.board_rows-3):
                if self.board_state[j][i] == 1 and self.board_state[j+1][i-1] == 1 and self.board_state[j+2][i-2] == 1 and self.board_state[j+3][i-3] == 1:
                    return 1
                elif self.board_state[j][i] == 2 and self.board_state[j+1][i-1] == 2 and self.board_state[j+2][i-2] == 2 and self.board_state[j+3][i-3] == 2:
                    return 2

        # if no spots open, game has ended as a tie
        if len(self.get_valid_inputs()) == 0:
            return 3

        return 0
