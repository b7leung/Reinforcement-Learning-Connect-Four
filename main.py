
import pprint

import connect_four
import ai

cf_game = connect_four.ConnectFour(5,5)
Q_value_f = {}
cf_ai = ai.ConnectFourAI(Q_value_f)
endgame_messages = {1:"User won game", 2:"AI won game", 3:"Game ended in a tie"}
curr_player = 1

cf_game.print_board()
while True:

    # User's Move
    if curr_player == 1:
        try:
            user_input = input("Input next column to place:  ")
            move = int(user_input)-1
        except:
            print("not recognized, try again")
            #continue

    # AI's Move
    else:
        #move = cf_ai.make_move(board_state)
        user_input = input("AI: Input next column to place:  ")
        move = int(user_input)-1


    # Update game status
    cf_game.update_board(move, curr_player)
    cf_game.print_board()
    game_status = cf_game.get_status()
    if game_status != 0:
        print(endgame_messages[game_status])
        break

    print(cf_game.get_valid_inputs())

    # switch current player
    curr_player = 2 if curr_player == 1 else 1

    

