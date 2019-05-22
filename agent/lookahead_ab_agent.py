'''
Created on May 14, 2019

@author: enerve
'''

import mechanics

import logging
import random

from .explorer import Explorer

class LookaheadABAgent(Explorer):
    '''
    Recursive lookahead player, using alpha-beta pruning.
    '''

    def __init__(self, config, depth):
        ''' Initialize for a new game, and note player's start state '''
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.depth = depth

    def prefix(self):
        pref = "lookab%d" % self.depth
        return pref

    # ---------------- Single game ---------------
         
    def _choose_move(self):
        ''' Agent's turn. Chooses the next move '''

#         _, options = self._best_move2(self.depth, coin=1, return_options=True)
        _, options = self._best_move(self.depth, coin=1, alpha=-1000, beta=1000,
                                     return_options=True)
        # choose randomly from best options
        idx = random.randint(0, len(options)-1)
        A = options[idx]
        return A

    def _best_move(self, depth, coin, alpha, beta, return_options=False):
        ''' Returns score of my (coin's) best next move '''

        if depth == 0:
            # can't lookahead this deep
            return 0, None
            
        options = None
    
        if coin == 1: # Maximizing player
            best = None
            # Look at all my options and return my best move value
            options = [] if return_options else None
            for x in range(7):
                y = self.h[x]
                if y < 6:
                    self.S[y, x] = coin
                    self.h[x] += 1
                    if mechanics.has_won(self.S, coin, x, y):
                        val = 100
                    else:
                        val = 0.9 * self._best_move(depth-1, -coin, alpha/0.9, beta/0.9)[0]
                    self.h[x] -= 1
                    self.S[y, x] = 0
 
                    if best is None or val > best:
                        best = val
                        if return_options: options = [x]
                    elif val == best:
                        if return_options: options.append(x)

                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
        else:
            best = None
            for x in range(7):
                y = self.h[x]
                if y < 6:
                    self.S[y, x] = coin
                    self.h[x] += 1
                    if mechanics.has_won(self.S, coin, x, y):
                        val = -100
                    else:
                        val = 0.9 * self._best_move(depth-1, -coin, alpha/0.9, beta/0.9)[0]
                    self.h[x] -= 1
                    self.S[y, x] = 0

                    if best is None or val < best:
                        best = val
    
                    beta = min(beta, best)
                    # We don't want to mess with options collection
                    if beta < alpha:
                        break
                    elif beta <= alpha and depth < self.depth - 1:
                        break

        if best is None:
            # board is full
            best = 0

        return best, options
