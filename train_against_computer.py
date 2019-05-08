'''
Created on Apr 24, 2019

@author: enerve
'''

import logging

from epoch_trainer import EpochTrainer
from agent import *
from function import *
from coindrop_feature_eng import CoindropFeatureEng
import trainer_helper as th
import cmd_line
import log
import util

import numpy as np
import random
import torch

from game import Game

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'Coindrop'

    logger = logging.getLogger()
    log.configure_logger(logger, "Coindrop")
    logger.setLevel(logging.DEBUG)
    
    # -------------- Configure track
    
    config = th.CONFIG(
        NUM_ROWS = 6,
        NUM_COLUMNS = 7
    )

    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    coindrop_fe = CoindropFeatureEng(config)
    
    #agent_fe = RectangleFeatureEng(config)
    agent_fa = NN_FA(
                    0.0005, # alpha ... #4e-5 old alpha without batching
                    0, # regularization constant
                    512, # batch_size
                    1000, # max_iterations
                    coindrop_fe)


    es = ESPatches(config,
                  explorate=50000,
                  fa=agent_fa)
    agent = th.create_agent(config, 
                    alg = 'qlambda', 
                    es = es,
                    lam = 0.95,
                    fa=agent_fa,
                    mimic_fa=None)
    
    random_agent = RandomAgent()
    lookahead_agent1 = LookaheadAgent(1)
    lookahead_agent2 = LookaheadAgent(2)
    lookahead_agent3 = LookaheadAgent(3)

    # ------------------ Training -------------------

    opponent = lookahead_agent3
    
    trainer = EpochTrainer(agent, opponent, agent.prefix() + opponent.prefix())
    
    if False: # Train from dataset file
        #trainer.load_from_file("")
        # LA2 dir = "810178_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0003_r1e-05_b512_i100000_F_NNconvnet__"
        dir = "932109_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0003_r1e-05_b512_i30000_F_NNconvnet__"
        agent_fa.load_training_data("coindrop_mrr_opp", dir)
        agent_fa.train()
        agent_total_R = 0
        agent_wins = 0
        agent_losses = 0
        agent_sum_moves = 0
        test_runs = 1000
        logger.debug("Testing %d games", test_runs)
        fa_player = FAPlayer(agent_fa)
        for tep in range(test_runs):
            game = Game([fa_player, opponent])
            game.run()
            agent_total_R += fa_player.game_performance()
            agent_wins += 1 if fa_player.total_reward() > 0 else 0
            agent_losses += 1 if fa_player.total_reward() < 0 else 0
            agent_sum_moves += fa_player.moves
        logger.debug("#wins: %d / %d" % (agent_wins, test_runs))
        logger.debug("#loss: %d / %d" % (agent_losses, test_runs))
        logger.debug("%% score: %0.2f" % (agent_total_R/test_runs * 100))
        logger.debug("Avg #moves: %0.2f" % (agent_sum_moves/test_runs))
        agent_fa.report_stats()
    
    elif False: # Load trained model
        dir = ""
        agent_fa.load_model(dir)
    else: # Run episodes
    
        trainer.train(100, 2, 1)
        #trainer.save_to_file()
    
        agent_fa.store_training_data("coindrop")
    
    
        trainer.report_stats()

    
    agent_fa.save_model("coindrop")
    
    

if __name__ == '__main__':
    main()

