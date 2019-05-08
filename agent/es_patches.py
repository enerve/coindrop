'''
Created on 1 May 2019

@author: enerve
'''

import numpy as np
import random
from .exploration_strategy import ExplorationStrategy

class ESPatches(ExplorationStrategy):
    '''
    classdocs
    '''

    def __init__(self, config, explorate, fa):
        '''
        Constructor
        '''
        super().__init__(config, explorate, fa)
        self.num_rows = config.NUM_ROWS
        self.num_cols = config.NUM_COLUMNS

        self.N = np.zeros((3*3, 3, 3, 3, 3, 3, 3))
        
        self.recent_epsilon = 1.0
        
    def prefix(self):
        return "pat"
        
    def pick_action(self, S, moves):
        N0 = self.explorate

        m = 1.0
        for i in range(3):
            for j in range(3):
                Npatch = self.N[3*i + j]
                a, b = 2*i, 2*j
                I = tuple(S[b:(b+2), a:(a+3)].flatten() + 1)
                n = Npatch[I]
                m *= N0 / (N0 + n)
                Npatch[I] += 1
        
        epsilon = m
        
        if moves >= 6:
            f = 0.99
            self.recent_epsilon = f * self.recent_epsilon + (1-f) * epsilon 
      
        if random.random() >= epsilon:
            # Pick best
            action = self.fa.best_action(S)
        else:
            action = self.fa.random_action(S)
        
        return action