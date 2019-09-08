'''
Created on Nov 6, 2018

@author: enerve
'''

import logging
import numpy as np
import time
import util
from agent import FAPlayer
from game import Game
import random

class EpochTrainer:
    ''' A class that helps train the RL agent in stages, collecting episode
        history for an epoch and then training on that data.
    '''

    def __init__(self, explorer, opponent, learner, training_data_collector,
                 validation_data_collector, tester, prefix):
        self.explorer = explorer
        self.opponent = opponent
        self.learner = learner
        self.training_data_collector = training_data_collector
        self.validation_data_collector = validation_data_collector
        self.tester = tester
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_agent_alg = prefix
        self.logger.debug("Agent: %s", util.pre_agent_alg)

        self.stat_e_1000 = []
        
        self.ep = 0

    def train(self, num_episodes_per_epoch, num_epochs, num_explorations = 1,
              debug_run_first_epoch_data_only = False):
        
        total_episodes = num_episodes_per_epoch * num_epochs * num_explorations
           
        self.logger.debug("Starting for %d episodes x %d epochs x %d expls",
                          num_episodes_per_epoch, num_epochs, num_explorations)
        start_time = time.clock()
        
        totaltime_explore = 0
        totaltime_process = 0
        totaltime_train = 0
        
        #self.end_states = {}

        ep = ep_s = self.ep
        for expl in range(num_explorations):
            for epoch in range(num_epochs):
                # In each epoch, we first collect experience, then (re)train FA
                self.logger.debug("====== Expl %d epoch %d =====", expl, epoch)
                
                best_R = -10000
                
                start_explore_time = time.clock()
                
                if not debug_run_first_epoch_data_only or epoch == 0:
                    # Run games to collect new data               
                    for ep_ in range(num_episodes_per_epoch):
                        # Create game environment for a single game
                        if random.random() < 0.5:
                            player_list = [self.explorer, self.opponent]
                        else:
                            player_list = [self.opponent, self.explorer]
                        game = Game(player_list)
                        
                        S = game.run()
    
                        if ep_ % 10 == 0:
                            # Save for validation
                            self.explorer.save_game_for_testing()
                            self.opponent.save_game_for_testing()
                        else:
                            # Use for training
                            self.explorer.save_game_for_training()
                            self.opponent.save_game_for_training()
    
                        #stS = np.array2string(S, separator='')
                        #if stS in self.end_states:
                        #    self.end_states[stS] += 1
                        #else:
                        #    self.end_states[stS] = 1
    
                        self.explorer.collect_stats(ep, total_episodes)
                        self.opponent.collect_stats(ep, total_episodes)
                
                        if util.checkpoint_reached(ep, 100):
                            self.stat_e_1000.append(ep)
                            self.logger.debug("Ep %d ", ep)
                            
                        ep += 1
                    self.training_data_collector.reset_dataset()
                    self.validation_data_collector.reset_dataset()
                else:
                    # Reuse existing data
                    self.training_data_collector.replay_dataset()
                    self.validation_data_collector.replay_dataset()

                start_process_time = time.clock()
                totaltime_explore += (start_process_time - start_explore_time)
                
                self.logger.debug("-------- processing ----------")
                self.logger.debug("Learning from agent history")
                self.learner.process(self.explorer.get_episodes_history(),
                                     self.training_data_collector,
                                     "agent train")
                self.training_data_collector.report_collected_dataset()
                #self.learner.plot_last_hists()
                self.learner.process(self.explorer.get_test_episodes_history(),
                                     self.validation_data_collector,
                                     "agent val")

#                 self.logger.debug("Learning from opponent history")
#                 self.learner.process(self.opponent.get_episodes_history(),
#                                      self.training_data_collector,
#                                      "opponent train")
#                 #self.learner.plot_last_hists()
#                 #self.learner.collect_last_hists()
#                 self.learner.process(self.opponent.get_test_episodes_history(),
#                                      self.validation_data_collector,
#                                      "opponent val")

                start_training_time = time.clock()
                totaltime_process += (start_training_time - start_process_time)

                self.logger.debug("-------- training ----------")
                self.learner.learn(self.training_data_collector,
                                   self.validation_data_collector)

                totaltime_train += (time.clock() - start_training_time)
                
                # Sacrifice some data for the sake of GPU memory
                if len(self.explorer.get_episodes_history()) >= 15000:#20000
                    self.logger.debug("Before: %d, %d",
                                      len(self.explorer.get_episodes_history()),
                                      len(self.opponent.get_episodes_history()))
                    self.explorer.decimate_history()
                    self.opponent.decimate_history()
                    self.logger.debug("After: %d, %d", 
                                      len(self.explorer.get_episodes_history()),
                                      len(self.opponent.get_episodes_history()))
                
                #self.tester.run_test(50)

                self.logger.debug("  Clock: %d seconds", time.clock() - start_time)

            #self.explorer.restart_exploration(1)

        self.logger.debug("Completed training in %0.1f minutes", (time.clock() - start_time)/60)
        self.logger.debug("   Total time for Explore: %0.1f minutes", (totaltime_explore)/60)
        self.logger.debug("   Total time for Process: %0.1f minutes", (totaltime_process)/60)
        self.logger.debug("   Total time for Train:   %0.1f minutes", (totaltime_train)/60)
    
        self.ep = ep
        
# def test_agent_fa(self, fa_player, opponent, test_runs):
#     # Test permoformance in actual games
#     for start_first in [True, False]:
#         self.logger.debug("-------- testing as %s player ----------",
#                           "first" if start_first else "second")
#         agent_sum_score = 0
#         num_wins = num_losses = 0
#         agent_sum_win_moves = 0
#         agent_sum_lose_moves = 0
#         if start_first:
#             players = [self.test_agent, self.opponent] 
#         else: 
#             players = [self.opponent, self.test_agent]
#         for tep in range(test_runs):
#             game = Game(players)
#             game.run()
#             agent_sum_score += self.test_agent.game_performance()
#             if self.test_agent.game_performance() > 0:
#                 num_wins += 1 
#                 agent_sum_win_moves += self.test_agent.moves
#             elif self.test_agent.game_performance() < 0:
#                 num_losses += 1
#                 agent_sum_lose_moves += self.test_agent.moves
#         self.logger.debug("FA agent R: %0.2f   (%d vs %d) / %d" % 
#                           (agent_sum_score, num_wins, num_losses, test_runs))                
#         if num_wins > 0:
#             self.logger.debug("Avg #moves for win: %0.2f" % (agent_sum_win_moves/num_wins))
#         if num_losses > 0:
#             self.logger.debug("Avg #moves for loss: %0.2f" % (agent_sum_lose_moves/num_losses))
# 
#     agent_total_R = 0
#     agent_wins = 0
#     agent_losses = 0
#     agent_sum_moves = 0
#     test_runs = 100
#     logger.debug("Testing %d games against %s", test_runs, opponent.prefix())
#     start_time = time.clock()
#     for tep in range(test_runs):
#         game = Game([fa_player, opponent])
#         game.run()
#         agent_total_R += fa_player.game_performance()
#         agent_wins += 1 if fa_player.game_performance() > 0 else 0
#         agent_losses += 1 if fa_player.game_performance() < 0 else 0
#         agent_sum_moves += fa_player.moves
#         if (tep+1) % 100 == 0:
#             logger.debug("   done %d eps in %d secs", tep+1, time.clock() - start_time)
#             start_time = time.clock()
#     logger.debug("#wins: %d / %d" % (agent_wins, test_runs))
#     logger.debug("#loss: %d / %d" % (agent_losses, test_runs))
#     logger.debug("%% score: %0.2f" % (agent_total_R/test_runs * 100))
#     logger.debug("Avg #moves: %0.2f" % (agent_sum_moves/test_runs))


    def load_from_file(self, subdir):
        self.learner.load_model(subdir)
        #self.load_stats(subdir)

    def save_to_file(self, pref=''):
        # save learned values to file
        self.learner.save_model(pref=pref)
        
        # save stats to file
        self.save_stats(pref=pref)        
    
    def save_stats(self, pref=""):
        self.explorer.save_stats(pref="a_" + pref)
        self.opponent.save_stats(pref="o_" + pref)
        self.learner.save_stats(pref="l_" + pref)

        self.learner.save_hists(["agent train"])#, "opponent train"])
        self.learner.write_hist_animation("agent train")
        
    def load_stats(self, subdir, pref=""):
        self.logger.debug("Loading stats...")

        self.explorer.load_stats(subdir, pref="a_" + pref)
        self.opponent.load_stats(subdir, pref="o_" + pref)
        self.learner.load_stats(subdir, pref="l_" + pref)
    
        #self.learner.load_hists(subdir)
    
    def report_stats(self, pref=""):
        self.explorer.report_stats(pref="a_" + pref)
        self.opponent.report_stats(pref="o_" + pref)
        self.learner.report_stats(pref="l_" + pref)
