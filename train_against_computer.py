'''
Created on Apr 24, 2019

@author: enerve
'''

import logging

from epoch_trainer import EpochTrainer
from agent import *
# from function import BoundActionModel
# from function import NNBoundFA
from function import *
from coindrop_feature_eng import CoindropFeatureEng
import trainer_helper as th
import cmd_line
import log
import util

import numpy as np
import random
import torch
import time

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
    bound_action_model = BoundActionModel(config)
    
#     agent_fa = NN_FA(
#                     0.0005, # alpha ... #4e-5 old alpha without batching
#                     0, # regularization constant
#                     512, # batch_size
#                     500, # max_iterations
#                     coindrop_fe)
    agent_fa = NN_Bound_FA(
                    0.0005, # alpha ... #4e-5 old alpha without batching
                    0.5, # regularization constant
                    512, # batch_size
                    500, # max_iterations
                    bound_action_model)

    training_data_collector = FADataCollector(agent_fa)
    validation_data_collector = FADataCollector(agent_fa)

    es = ESPatches(config,
                  explorate=50000,
                  fa=agent_fa)
    agent = th.create_agent(config, 
                    alg = 'sarsalambda', 
                    es = es,
                    lam = 0.95,
                    fa=agent_fa)
    
    # ------------------ Training -------------------

    if False:
        if True: #If train model on dataset
            dir = "847998_Coindrop_DR_q_lambda_eesp_l0.95neural_bound_a0.0005_r0_b512_i500_FBAM__NNconvnetlook5__"
            training_data_collector.load_dataset(dir, "final", "t")
            validation_data_collector.load_dataset(dir, "final", "v")
            agent_fa.train(training_data_collector, validation_data_collector)
            agent_fa.report_stats()
        else: #Load model directly
            dir = "808544_Coindrop_DR_q_lambda_eesp_l0.95neural_bound_a0.0005_r0_b512_i500_FBAM__NNconvnetlook3__"
            agent_fa.load_model(dir, "v3")
            
        test_agent_fa(agent_fa, LookaheadABAgent(5))
    
    elif True: # If Run episodes
        
        opponent = LookaheadABAgent(5)
        
        trainer = EpochTrainer(agent, opponent, 
                               training_data_collector,
                               validation_data_collector,
                               agent.prefix() + opponent.prefix())
        
        if False:
            # To start training afresh 
            agent_fa.initialize_default_net()
        elif True:
            # To start training from where we last left off.
            # i.e., load episodes history, exploration state, and FA model
            dir = "937454_Coindrop_DR_sarsa_lambda_eesp_l0.95neural_bound_a0.0005_r0.5_b512_i500_FBAM__NNconvnetlook5__"
            agent.load_episode_history("agent", dir)
            es.load_exploration_state(dir)
            opponent.load_episode_history("opponent", dir)
            agent_fa.load_model(dir, "v3")
        elif False:
            # For single-epoch training/testing.
            # Load last training dataset and model, but not earlier history
            dir = "330041_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0005_r0_b512_i1000_F_NNconvnetlook3__"
            training_data_collector.load_dataset(dir, "final", "t")
            validation_data_collector.load_dataset(dir, "final", "v")
            agent_fa.load_model(dir, "v3")
    
        trainer.train(500, 30, 1)
        #trainer.save_to_file()
    
        agent.store_episode_history("agent")
        es.store_exploration_state()
        opponent.store_episode_history("opponent")
        
        training_data_collector.store_last_dataset("final_t")
        validation_data_collector.store_last_dataset("final_v")
    
        trainer.report_stats()

    
    agent_fa.save_model("v3")
    
    
def test_agent_fa(agent_fa, opponent):
    logger = logging.getLogger()
    agent_total_R = 0
    agent_wins = 0
    agent_losses = 0
    agent_sum_moves = 0
    test_runs = 100
    logger.debug("Testing %d games against %s", test_runs, opponent.prefix())
    fa_player = FAPlayer(agent_fa)
    start_time = time.clock()
    for tep in range(test_runs):
        game = Game([fa_player, opponent])
        game.run()
        agent_total_R += fa_player.game_performance()
        agent_wins += 1 if fa_player.game_performance() > 0 else 0
        agent_losses += 1 if fa_player.game_performance() < 0 else 0
        agent_sum_moves += fa_player.moves
        if (tep+1) % 100 == 0:
            logger.debug("   done %d eps in %d secs", tep+1, time.clock() - start_time)
            start_time = time.clock()
    logger.debug("#wins: %d / %d" % (agent_wins, test_runs))
    logger.debug("#loss: %d / %d" % (agent_losses, test_runs))
    logger.debug("%% score: %0.2f" % (agent_total_R/test_runs * 100))
    logger.debug("Avg #moves: %0.2f" % (agent_sum_moves/test_runs))

if __name__ == '__main__':
    main()

