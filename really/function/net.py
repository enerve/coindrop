'''
Created on 1 Mar 2019

@author: enerve
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, layers):
        super(Net, self).__init__()
        
        self.layers = layers
        
        self.f_list = nn.ModuleList()
        for i, o in zip(layers, layers[1:]):
            self.f_list.append(nn.Linear(i, o))
        
    def prefix(self):
        return '%s' % self.layers
        
    def forward(self, x):
        f = self.f_list[0]
        z = f(x)
        
        for f in self.f_list[1:]:
            a = F.relu(z)
            z = f(a)

        return z

