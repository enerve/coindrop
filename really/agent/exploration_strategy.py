'''
Created on 30 Apr 2019

@author: enerve
'''

import numpy as np
import random


class ExplorationStrategy(object):
    '''
    classdocs
    '''

    def __init__(self, config, explorate, fa):
        '''
        Constructor
        '''
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        self.fa = fa

        
    def pick_action(self, S, moves):
        pass
    
    def store_exploration_state(self, pref=""):
        pass
    
    def load_exploration_state(self, subdir, pref=""):
        pass