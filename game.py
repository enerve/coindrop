'''
Created on Apr 23, 2019

@author: enerve
'''

import mechanics

import logging
import numpy as np

class Game():
    '''
    A single game of connect-4, the environment within which a single episode
    is played out.
    '''

    R_win = 1
    R_lose = -1
    R_draw = 0

    def __init__(self, player_list):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.player_list = player_list
        
    def run(self):
        B = np.zeros((6, 7), dtype=np.int32)
        h = np.zeros(7, dtype=np.int8)
        turn = 0  # player 0 vs player 1
        
        for p in self.player_list:
            p.init_game(B, h)
        
        for i in range(6*7):
            p = self.player_list[turn]
            o = self.player_list[1 - turn]
            coin = 1 if turn == 1 else -1
            
            chosen_col = p.next_move()  # assume it's a valid move
            y, x = h[chosen_col], chosen_col
            if y >= 6:
                self.logger.debug(B)
                self.logger.debug(h)

            B[y, x] = coin
            h[x] += 1

            #print("%2d: %d" %(coin, chosen_col))
            # Calculate any rewards

            if mechanics.has_won(B, coin, x, y):
                p.see_move(self.R_win, B * coin, h, moves=i)
                o.see_move(self.R_lose, B * (-coin), h, moves=i)
                p.game_over()
                o.game_over()
                #self.logger.debug("Win: %2d" % coin)
                break

            # Update everyone about the move

            p.see_move(0, B * coin, h, moves=i)
            o.see_move(0, B * (-coin), h, moves=i)
            
            turn = 1 - turn

        else:
            # Nobody won
            p.game_over()
            o.game_over()
            #self.logger.debug("Draw")
            
        #print(B)
        return B
            
