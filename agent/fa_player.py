'''
Created on Apr 30, 2019

@author: enerve
'''

import logging

from .player import Player

class FAPlayer(Player):
    '''
    A player that plays games using an existing Function Approximator
    '''

    def __init__(self,
                 fa):
        self.fa = fa

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    def prefix(self):
        pref = "faP_" + self.fa.prefix()
        return pref

    # ---------------- Single game ---------------

    def init_game(self, initial_state, initial_heights):
        self.S = initial_state
        self.total_R = 0
        
    def see_move(self, reward, new_state, h, moves=0):
        self.S = new_state
        self.total_R += reward
        self.moves = moves

    def next_move(self):
        ''' Agent's turn. Chooses the next move '''
        
        # Choose action on-policy
        return self.fa.best_action(self.S)[0]
    
    def has_won(self):
        return self.total_R > 0

    def game_performance(self):
        
        # increased significance for a win / loss the sooner it happens
        factor = (42-self.moves)/42
        
        return factor * self.total_R

