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

class EpochTrainer:
    ''' A class that helps train the RL agent in stages, collecting episode
        history for an epoch and then training on that data.
    '''

    def __init__(self, agent, opponent, prefix):
        self.agent = agent
        self.opponent = opponent
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_agent_alg = prefix
        self.logger.debug("Agent: %s", util.pre_agent_alg)

        self.stat_e_1000 = []
        
        self.ep = 0

    def train(self, num_episodes_per_epoch, num_epochs, num_explorations = 1):
        
        total_episodes = num_episodes_per_epoch * num_epochs * num_explorations
           
        self.logger.debug("Starting for %d expls x %d epochs x %d episodes",
                          num_explorations, num_epochs, num_episodes_per_epoch)
        start_time = time.clock()
        
        #self.end_states = {}

        ep = ep_s = self.ep
        for expl in range(num_explorations):
            for epoch in range(num_epochs):
                # In each epoch, we first collect experience, then (re)train FA
                self.logger.debug("====== Expl %d epoch %d =====", expl, epoch)
                
                best_R = -10000
                
                for ep_ in range(num_episodes_per_epoch):
                    # Create game environment for a single game
                    game = Game([self.agent, self.opponent])
                    
                    S = game.run()

                    if ep_ % 10 == 0:
                        # Save for validation
                        self.agent.save_game_for_testing()
                        self.opponent.save_game_for_testing()
                    else:
                        # Use for training
                        self.agent.save_game_for_training()
                        self.opponent.save_game_for_training()

                    #stS = np.array2string(S, separator='')
                    #if stS in self.end_states:
                    #    self.end_states[stS] += 1
                    #else:
                    #    self.end_states[stS] = 1

                    self.agent.collect_stats(ep, total_episodes)
                    self.opponent.collect_stats(ep, total_episodes)
            
                    if util.checkpoint_reached(ep, 100):
                        self.stat_e_1000.append(ep)
                        self.logger.debug("Ep %d ", ep)
                        
                    ep += 1
                
                self.logger.debug("Learning from agent history")
                self.agent.process(self.agent.get_episodes_history())
                #self.agent.plot_last_hists()
                self.agent.process_test(self.agent.get_test_episodes_history())

                self.logger.debug("Learning from opponent history")
                self.agent.process(self.opponent.get_episodes_history())
                #self.agent.plot_last_hists()
                self.agent.collect_last_hists()
                self.agent.process_test(self.opponent.get_test_episodes_history())

                self.agent.learn()
                
                debug_first_epoch_data_only = False
                if debug_first_epoch_data_only:
                    num_episodes_per_epoch = 0
                    self.agent.fa.replay_dataset()
                else:
                    self.agent.fa.reset_dataset()
                
                # Sacrifice some data for the sake of GPU memory
                if len(self.agent.get_episodes_history()) >= 10000:
                    self.logger.debug("Before: %d, %d",
                                      len(self.agent.get_episodes_history()),
                                      len(self.opponent.get_episodes_history()))
                    self.agent.decimate_history()
                    self.opponent.decimate_history()
                    self.logger.debug("After: %d, %d", 
                                      len(self.agent.get_episodes_history()),
                                      len(self.opponent.get_episodes_history()))
                
                agent_sum_totalR = 0
                num_wins = 0
                agent_sum_win_moves = 0
                agent_sum_lose_moves = 0
                test_runs = 100
                fa_player = FAPlayer(self.agent.fa)
                for tep in range(test_runs):
                    game = Game([fa_player, self.opponent])
                    game.run()
                    agent_sum_totalR += fa_player.game_performance()
                    if fa_player.has_won():
                        num_wins += 1
                        agent_sum_win_moves += fa_player.moves
                    else:
                        agent_sum_lose_moves += fa_player.moves
                self.logger.debug("FA agent R: %0.2f / %d" % (agent_sum_totalR, test_runs))
                if num_wins > 0:
                    self.logger.debug("Avg #moves for win: %0.2f" % (agent_sum_win_moves/num_wins))
                num_losses = test_runs - num_wins
                if num_losses > 0:
                    self.logger.debug("Avg #moves for loss: %0.2f" % (agent_sum_lose_moves/num_losses))
                self.logger.debug("  Clock: %d seconds", time.clock() - start_time)

            #self.agent.restart_exploration(1)

        self.agent.store_collected_hists()

        self.logger.debug("Completed training in %0.1f minutes", (time.clock() - start_time)/60)
    
        self.ep = ep
        
    def load_from_file(self, subdir):
        self.agent.load_model(subdir)
        #self.load_stats(subdir)

    def save_to_file(self, pref=''):
        # save learned values to file
        self.agent.save_model(pref=pref)
        
        # save stats to file
        self.save_stats(pref=pref)        
    
    def save_stats(self, pref=None):
        self.agent.save_stats(pref=pref)
        
        A = np.asarray([
            self.stat_bestpath_times,
            self.stat_recent_total_R,
            self.stat_e_bp,
            self.stat_bestpath_R,
            ], dtype=np.float)
        util.dump(A, "statsA", pref)
    
        B = np.asarray(self.stat_m, dtype=np.float)
        util.dump(B, "statsB", pref)
    
        C = np.asarray(self.stat_e_1000, dtype=np.float)
        util.dump(C, "statsC", pref)
        
        # TODO: Save and load #episodes done in total
        
    def load_stats(self, subdir, pref=None):
        self.agent.load_stats(subdir, pref)
    
        self.logger.debug("Loading stats...")
        
        A = util.load("statsA", subdir, pref)
        self.stat_bestpath_times = A[0]
        self.stat_recent_total_R = A[1]
        self.stat_e_bp = A[2]
        self.stat_bestpath_R = A[3]
        
        B = util.load("statsB", subdir, pref)
        self.stat_m = B
        
        C = util.load("statsC", subdir, pref)
        self.stat_e_1000 = C
    
    def report_stats(self, pref=""):
        self.agent.report_stats(pref="a_" + pref)
        self.opponent.report_stats(pref="o_" + pref)
