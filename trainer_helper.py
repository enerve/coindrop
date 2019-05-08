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

def create_agent(config, alg, es, lam, fa, mimic_fa):
    return create_agent_i(config, ALG[alg], es, lam, fa, mimic_fa)

def create_agent_i(config, i_alg, es, lam, fa, mimic_fa):
    if i_alg == 0:    
        driver = QAgent(config,
                        1, # gamma
                        es, # explorate
                        fa)
#     elif i_alg == 1:    
#         driver = SarsaFADriver(config,
#                         1, # gamma
#                         es, # explorate
#                         fa,
#                         mimic_fa)
    elif i_alg == 2:
        driver = QLambdaAgent(config,
                        lam, #lambda
                        1, # gamma
                        es, # explorate
                        fa)
#     elif i_alg == 3:
#         driver = SarsaLambdaFADriver(config,
#                         lam, #lambda
#                         1, # gamma
#                         es, # explorate
#                         fa,
#                         mimic_fa)
    return driver
