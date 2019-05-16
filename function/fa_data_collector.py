'''
Created on 13 May 2019

@author: enerve
'''

import logging
import util
import numpy as np

class FADataCollector(object):
    '''
    Helps collect training/validation data rows for future FA updates
    '''


    def __init__(self, fa):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.fa = fa #TODO: not needed?

        self.reset_dataset()
        
    def reset_dataset(self):
        # Forget old steps history
        self.steps_history_state = []
        self.steps_history_action = []
        self.steps_history_target = []
        self.ireplay = None
        self.pos = self.neg = 0

    def replay_dataset(self):
        ''' Prepare to replay instead of record new data. FOR DEBUGGING ONLY. '''
        if len(self.steps_history_state) > 0:
            self.ireplay = 0
            self.pos = self.neg = 0

    def record(self, state, action, target):
        ''' Record incoming data rows '''

        if self.ireplay is not None:
            # Replaying the same datapoints but recording new targets
            # Confirm it's an exact repeat
            old_action = self.steps_history_action[self.ireplay]
            if action != old_action:
                self.logger("Got %d vs %d", action, old_action)
                old_state = self.steps_history_state[self.ireplay]
                self.logger("    %s vs %s", state, old_state)
            self.steps_history_target[self.ireplay] = target
            self.ireplay += 1
        else:
            self.steps_history_state.append(state)
            self.steps_history_action.append(action)
            self.steps_history_target.append(target)

        if target > 0:
            self.pos += 1
        else:
            self.neg += 1
        
    def store_last_dataset(self, pref=""):
        fname = "dataset_" + pref
        SHS = np.asarray(self.steps_history_state)
        SHA = np.asarray(self.steps_history_action)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHS, fname, "S")
        util.dump(SHA, fname, "A")
        util.dump(SHT, fname, "t")
 
    def load_dataset(self, subdir, pref=""):
        fname = "dataset_" + pref
        self.steps_history_state = [s for s in util.load(fname, subdir, suffix="S")]
        self.steps_history_action = [a for a in util.load(fname, subdir, suffix="A")]
        self.steps_history_target = [t for t in util.load(fname, subdir, suffix="t")]
        
    def before_update(self, pref=""):
        self.logger.debug("#pos: %d \t #neg: %d", self.pos, self.neg)

    def get_data(self):
        return self.steps_history_state, self.steps_history_action, self.steps_history_target
      