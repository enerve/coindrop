'''
Created on Apr 28 2019

@author: enerve
'''

import logging
import math
import torch
import util

from function import FeatureEng

class CoindropFeatureEng(FeatureEng):
    '''
    Feature engineering for Connect-4
    '''

    def __init__(self,
                 config):
        '''
        Constructor
        '''
        
        # states
        self.num_rows = config.NUM_ROWS
        self.num_cols = config.NUM_COLUMNS
        # actions
        self.num_actions = config.NUM_COLUMNS
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_inputs = self.num_rows * self.num_cols
        
#     def num_actions(self):
#         return self.num_actions
        
    def prefix(self):
        return 'FEv2_'

    def x_adjust(self, B):
        ''' Takes the input params and converts it to an input feature array
        '''

        b = torch.from_numpy(B).to(self.device)
        # Create 2 layers, one for each player, with -1 for empty grids
        empty = (b == 0).float() * -1
        x1 = (b > 0.5).float() + empty
        x2 = (b < -0.5).float() + empty
        return torch.stack([x1, x2], dim=0)

    def valid_actions_mask(self, B):
        valid_actions_mask = (B[6-1] == 0) * 1
        return torch.from_numpy(valid_actions_mask).to(self.device).float()

    def random_action(self, B):
        V = torch.rand(self.num_cols).to(self.device).float() + 1000 * self.valid_actions_mask(B)
        i = torch.argmax(V).item()
        return i
        
