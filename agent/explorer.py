'''
Created on May 15, 2019

@author: enerve
'''

import logging
import util

import numpy as np

from .player import Player

class Explorer(Player):
    '''
    Base class for a player that records its experience for future use.
    '''

    def __init__(self,
                 config):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.episodes_history = []
        self.test_episodes_history = []
        
        # stats
        self.stats_R = []
        self.sum_total_R = 0
        self.sum_played = 0

    # ---------------- Single game ---------------
    
    def prefix(self):
        pass

    def init_game(self, initial_state, initial_heights):
        super().init_game(initial_state, initial_heights)
        
        self.S = np.copy(initial_state)
        self.R = 0
        self.h = initial_heights
        self.steps_history = []
        
    def see_move(self, reward, new_state, new_heights, moves=None):
        ''' Observe the effects on this player of an action taken (possibly by
            another player.)
            reward Reward earned by this player for its last move
        '''
        super().see_move(reward, new_state, new_heights, moves)
        self.R += reward
        self.S = np.copy(new_state)

    def next_move(self):
        ''' Explorer's turn. Chooses the next move '''
        A = self._choose_move() # Implemented by subclasses

        # Record results R, S of my previous move, and the current move A.
        # (The R of the very first move is irrelevant and is later ignored.)
        self.steps_history.append((self.R, self.S, A))
        self.R = 0  # prepare to collect rewards
        
        return A

    def game_over(self):
        ''' Wrap up game  '''

        # Record results of game end
        self.steps_history.append((self.R, None, None))
        
    def save_game_for_training(self):
        self.episodes_history.append(self.steps_history)

    def save_game_for_testing(self):
        self.test_episodes_history.append(self.steps_history)

    # ----------- Stats -----------

    def collect_stats(self, ep, num_episodes):
        if (ep+1)% 100 == 0:
            self.logger.debug("  avg R: %d/%d" % (self.sum_total_R, self.sum_played))
            self.stats_R.append(self.sum_total_R / self.sum_played)
            self.sum_total_R = 0
            self.sum_played = 0
            self.live_stats()
        
        self.sum_total_R += self.total_R
        self.sum_played += 1 

    def save_stats(self, pref=""):
        util.dump(np.asarray(self.stats_R, dtype=np.float), "statsR", pref)

    def load_stats(self, subdir, pref=""):
        self.stats_R = util.load("statsR", subdir, pref).tolist()

    def report_stats(self, pref):
#         util.plot([self.stats_R],
#                   range(len(self.stats_R)),
#                   title="recent rewards",
#                   pref="rr")
        pass
        
    def live_stats(self):
        #util.plot([self.stats_R],
        #          range(len(self.stats_R)), live=True)
        pass

    def get_episodes_history(self):
        return self.episodes_history

    def get_test_episodes_history(self):
        return self.test_episodes_history

    def decimate_history(self):
        self.episodes_history = [self.episodes_history[i] for i in
                                 range(len(self.episodes_history)) 
                                 if i % 10 > 0]
        self.test_episodes_history = [self.test_episodes_history[i] for i in 
                                      range(len(self.test_episodes_history)) 
                                      if i % 10 > 0]

    def store_episode_history(self, fname):
        EH = np.asarray(self.episodes_history)
        util.dump(EH, fname, "EH")
        VEH = np.asarray(self.test_episodes_history)
        util.dump(VEH, fname, "VEH")
 
    def load_episode_history(self, fname, subdir):
        self.logger.debug("Loading episode history from %s %s", subdir, fname)
        self.episodes_history = [[(s[0], s[1], s[2]) for s in sh]
                                 for sh in util.load(fname, subdir, suffix="EH")]
        self.test_episodes_history = [[(s[0], s[1], s[2]) for s in sh] 
                                      for sh in util.load(fname, subdir, suffix="VEH")]

