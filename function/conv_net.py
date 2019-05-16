'''
Created on 2 May 2019

@author: enerve
'''

import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        #N, C, H, W = x.size() # read in N, C, H, W
        #return x.view(N, -1)
        x = x.view(-1)
        return x.unsqueeze(0)
    
