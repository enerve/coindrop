'''
Created on May 21, 2019

@author: enerve
'''

import logging

from .explorer import Explorer

class FAExplorer(Explorer):
    '''
    A player that plays using the given exploration strategy, collecting
    experience data in the process.
    '''

    def __init__(self,
                 config,
                 exploration_strategy):
        
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.exploration_strategy = exploration_strategy

    def prefix(self):
        es_prefix = self.exploration_strategy.prefix()
        return "e%s" % es_prefix

    # ---------------- Single game ---------------

    def _choose_move(self):
        ''' Agent's turn. Chooses the next move '''
        # Choose action on-policy
        A = self.exploration_strategy.pick_action(self.S, self.moves)
        return A
        
    # ----------- Stats -----------

    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)

        if (ep+1)% 100 == 0:
            self.logger.debug("Recent epsilon: %f" % self.exploration_strategy.recent_epsilon)

    def report_stats(self, pref):
        super().report_stats(pref)
