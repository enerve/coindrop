'''
Created on 13 May 2019

@author: enerve
'''

import logging
import util
import numpy as np

class FADataCollector(object):
    '''
    Helps collect training/validation data for the next FA update 
    '''


    def __init__(self, fa):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.fa = fa

        # Collectors of incoming data
        self.reset_dataset()
        
    def reset_dataset(self):
        # Forget old steps history
        self.steps_history_state = []
        self.steps_history_action = []
        self.steps_history_target = []
        self.val_steps_history_state = []
        self.val_steps_history_action = []
        self.val_steps_history_target = []
        self.ireplay = None
        self.pos = self.neg = 0

    def replay_dataset(self):
        ''' Prepare to replay instead of record new data. DEBUGGING ONLY. '''
        self.ireplay = 0
        self.pos = self.neg = 0

    def record(self, state, action, target):
        ''' Record incoming data for later training '''

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

    def record_validation(self, state, action, target):
        self.val_steps_history_state.append(state)
        self.val_steps_history_action.append(action)
        self.val_steps_history_target.append(target)
        
    def store_dataset(self, fname):
        SHS = np.asarray(self.steps_history_state)
        SHA = np.asarray(self.steps_history_action)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHS, fname, "S")
        util.dump(SHA, fname, "A")
        util.dump(SHT, fname, "t")
        vSHS = np.asarray(self.val_steps_history_state)
        vSHA = np.asarray(self.val_steps_history_action)
        vSHT = np.asarray(self.val_steps_history_target)
        util.dump(vSHS, fname, "vS")
        util.dump(vSHA, fname, "vA")
        util.dump(vSHT, fname, "vt")
 
    def load_dataset(self, fname, subdir):
        self.steps_history_state = [s for s in util.load(fname, subdir, suffix="S")]
        self.steps_history_action = [a for a in util.load(fname, subdir, suffix="A")]
        self.steps_history_target = [t for t in util.load(fname, subdir, suffix="t")]

        self.val_steps_history_state = [s for s in util.load(fname, subdir, suffix="vS")]
        self.val_steps_history_action = [a for a in util.load(fname, subdir, suffix="vA")]
        self.val_steps_history_target = [t for t in util.load(fname, subdir, suffix="vt")]
        #util.hist(self.steps_history_action, bins=7)
        
    def before_update(self, pref=""):
        self.logger.debug("#pos: %d \t #neg: %d", self.pos, self.neg)

    def get_training_data(self):
        return self.steps_history_state, self.steps_history_action, self.steps_history_target
    
    def get_validation_data(self):
        return self.val_steps_history_state, self.val_steps_history_action, self.val_steps_history_target
    
      