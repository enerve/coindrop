'''
Created on 8 Mar 2019

@author: enerve
'''

from agent import *
from function import *
from collections import namedtuple

ALG = {
        'q' : 0,
        'sarsa' : 1,
        'qlambda' : 2,
        'sarsalambda' : 3
    }

FA = {
        'qtable' : 0,
        'poly' : 1,
        'multi' : 2,
        'nn' : 3
    }

CONFIG = namedtuple("Config",
    "NUM_ROWS, NUM_COLUMNS")

GAMMA = 0.9

def create_agent(config, alg, lam, fa):
    return create_agent_i(config, ALG[alg], lam, fa)

def create_agent_i(config, i_alg, lam, fa):
    if i_alg == 0:    
        agent = QAgent(config,
                        GAMMA, # gamma
                        fa)
    elif i_alg == 1:    
        agent = SarsaAgent(config,
                        GAMMA, # gamma
                        fa)
    elif i_alg == 2:
        agent = QLambdaAgent(config,
                        lam, #lambda
                        GAMMA, # gamma
                        fa)
    elif i_alg == 3:
        agent = SarsaLambdaAgent(config,
                        lam, #lambda
                        GAMMA, # gamma
                        fa)
    return agent
