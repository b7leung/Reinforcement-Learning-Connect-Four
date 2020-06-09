
class ConnectFourAI:

    # Q_value_f is a dict
    def __init__(self, Q_value_f):
        self.Q_value_f = Q_value_f


    def make_move(self, board_state):

        try:
            move = self.Q_value_f[board_state]
        except:
            #print("Couldn't find for board state")
            move = 0

        return move