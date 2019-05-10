'''
Created on Apr 29, 2019

@author: enerve
'''

import logging
import util

import numpy as np
import random

from .player import Player

class QLambdaAgent(Player):
    '''
    An agent that implements Q(λ) to explore and analyze the state-action space.
    '''

    def __init__(self,
                 config,
                 lam,
                 gamma,
                 exploration_strategy,
                 fa):
        
        self.fa = fa

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.num_columns = config.NUM_COLUMNS
        self.max_moves = config.NUM_COLUMNS * config.NUM_ROWS
        
        self.lam = lam      # lambda lookahead parameter
        self.gamma = gamma  # weight given to future rewards
        self.exploration_strategy = exploration_strategy
        
        self.eligible_mult = [(lam * gamma) ** i for i in range(self.max_moves)]        

        self.episodes_history = []
        self.test_episodes_history = []
        
        # stats
        self.stats_R = []
        self.sum_total_R = 0
        self.sum_played = 0
        self.stats_delta = []
        
        # for histograms
        self.all_currs = []
        self.all_targets = []
        self.all_deltas = []

    def prefix(self):
        es_prefix = self.exploration_strategy.prefix()
        pref = "q_lambda_e%s_l%0.2f" % (es_prefix, self.lam) + self.fa.prefix()
        return pref

    # ---------------- Single game ---------------

    def init_game(self, initial_state, initial_heights):
        self.S = initial_state
        self.R = 0
        self.total_R = 0
        self.steps_history = []
        self.moves = 0
        
    def see_move(self, reward, new_state, h, moves=None):
        ''' Observe the effects on this agent of an action taken (possibly by
            another agent.)
            reward Reward earned by this agent for its last move
        '''
        self.R += reward
        self.total_R += reward
        self.S = new_state
        if moves: self.moves = moves

    def next_move(self):
        ''' Agent's turn. Chooses the next move '''
        
        # Choose action on-policy
        A = self.exploration_strategy.pick_action(self.S, self.moves)
        # Record results R, S of my previous move, and the current move A.
        # (The R of the very first move is irrelevant and is later ignored.)
        self.steps_history.append((self.R, self.S, A))

        self.R = 0  # prepare to collect rewards

        return A

    def game_over(self):
        ''' Wrap up game  '''

        # Record results of game end
        self.steps_history.append((self.R, None, None))
        
        self.sum_total_R += self.total_R
        
    def save_game_for_training(self):
        self.episodes_history.append(self.steps_history)

    def save_game_for_testing(self):
        self.test_episodes_history.append(self.steps_history)

    # ----------- Stats -----------

    def collect_stats(self, ep, num_episodes):
        if (ep+1)% 100 == 0:
            self.logger.debug("QAgent avg R: %d/%d" % (self.sum_total_R, self.sum_played))
            self.stats_R.append(self.sum_total_R / self.sum_played)
            self.sum_total_R = 0
            self.sum_played = 0
            self.logger.debug("Recent epsilon: %f" % self.exploration_strategy.recent_epsilon)
            self.live_stats()
        
        self.sum_total_R += self.total_R
        self.sum_played += 1 

    def report_stats(self, pref):
        self.fa.report_stats(pref)
        
        util.plot([self.stats_R],
                  range(len(self.stats_R)),
                  title="recent rewards",
                  pref="rr")

        util.plot([self.stats_delta],
                  range(len(self.stats_delta)),
                  title="delta",
                  pref='delta')
        
        
    def live_stats(self):
        #self.fa.live_stats()
 
        #util.plot([self.stats_R],
        #          range(len(self.stats_R)), live=True)
        pass

    def get_episodes_history(self):
        return self.episodes_history

    def get_test_episodes_history(self):
        return self.test_episodes_history

    def decimate_history(self):
        self.episodes_history = [self.episodes_history[i] for i in
                                 range(len(self.episodes_history)) 
                                 if i % 10 > 0]
        self.test_episodes_history = [self.test_episodes_history[i] for i in 
                                      range(len(self.test_episodes_history)) 
                                      if i % 10 > 0]

    def store_episode_history(self, fname):
        EH = np.asarray(self.episodes_history)
        util.dump(EH, fname, "EH")
        VEH = np.asarray(self.test_episodes_history)
        util.dump(VEH, fname, "VEH")
 
    def load_episode_history(self, fname, subdir):
        self.episodes_history = [[(s[0], s[1], s[2]) for s in sh]
                                 for sh in util.load(fname, subdir, suffix="EH")]
        self.test_episodes_history = [[(s[0], s[1], s[2]) for s in sh] 
                                      for sh in util.load(fname, subdir, suffix="VEH")]


    # ----------- Analyze given history to extract training data -----------

    def process(self, episodes_history):
        ''' Analyzes given history to create training data from it '''
        self._process(episodes_history)

    def process_test(self, test_episodes_history):
        ''' Analyzes given history to create testing/validation data from it '''
        self._process(test_episodes_history, True)

    def _process(self, episodes_history, use_for_validation=False):
        self.deltas = []
        self.targets = []
        self.currs = []

        self.num_wins = 0
        self.num_loss = 0
        
        self.logger.debug(" Process history of %d episodes", len(episodes_history))

        for i, steps_history in enumerate(episodes_history):
            self._process_steps(steps_history, use_for_validation)

        # stats
        if not use_for_validation:
            #             npc = np.array(self.currs)
            #             self.logger.debug('  curr_val: %0.2f <%0.2f>', np.mean(npc), np.var(npc))
            #             self.logger.debug('  wins/losses (total): %d/%d (%d)', self.num_wins,
            #                               self.num_loss, len(episodes_history))
            # 
            #             npt = np.array(self.targets)
            #             self.logger.debug('  target  : %0.2f <%0.2f>', np.mean(npt), np.var(npt))

            npd = np.array(self.deltas)
            self.stats_delta.append(np.mean(np.abs(npd)))
            #self.logger.debug('  sumdelta: %0.4f', self.sum_delta)
            self.logger.debug('  delta   : %0.2f <%0.2f>', np.mean(np.abs(npd)), np.var(npd))
            #self.logger.debug('  wins/losses (total): %d/%d (%d)', self.num_wins,
            #                  self.num_loss, len(episodes_history))

    def _process_steps(self, steps_history, use_for_validation):
        ''' Observes and learns from the given episode '''
        S, A = None, None
        num_E = 0
        self.eligible_states = [None for i in range(self.max_moves)]
        self.eligible_state_target = [0 for i in range(self.max_moves)]
        for i, (R_, S_, A_) in enumerate(steps_history):
            if i > 0:
                if S_ is not None:
                    # off-policy
                    # TODO: speed this up, or parallelize it
                    max_A = self.fa.best_action(S_)
                    Q_at_max_next = self.fa.value(S_, max_A)
                else:
                    Q_at_max_next, max_A = 0, None
                
                # Learn from the reward gotten for action taken last time
                target = R_ + self.gamma * Q_at_max_next
                
#                 self.recent_target = 0.99 * self.recent_target
#                 self.recent_target += 0.01 * target 

                curr_value = self.fa.value(S, A)
                self.eligible_states[num_E] = (S, A)
                self.eligible_state_target[num_E] = curr_value
                delta = (target - curr_value)
                
                num_E += 1
                for j in range(num_E):
                    self.eligible_state_target[j] += delta * self.eligible_mult[num_E-j-1]
                if A_ != max_A:
                    # The policy diverted from Q* policy so restart eligibilities
                    # But first, flush the eligibility updates into the FA
                    self._record_eligibles(num_E, use_for_validation)
                    num_E = 0

                self.currs.append(curr_value)
                self.targets.append(target)
                self.deltas.append(delta)
                if R_ > 0: self.num_wins += 1 
                if R_ < 0: self.num_loss += 1
                
            S, A = S_, A_
            
        self._record_eligibles(num_E, use_for_validation)

    def _record_eligibles(self, num_E, use_for_validation):
        for i in range(num_E):
            S, A = self.eligible_states[i]
            target = self.eligible_state_target[i]
            if use_for_validation:
                self.fa.record_validation(S, A, target)
            else:
                self.fa.record(S, A, target)

    def plot_last_hists(self):
        npc = np.array(self.currs)
        util.hist(npc, 100, (-2, 2), "curr value", "currhist")

        npt = np.array(self.targets)
        util.hist(npt, 100, (-2, 2), "targets", "targetshist")

        npd = np.array(self.deltas)
        util.hist(npd, 100, (-2, 2), "delta", "deltahist")

    def collect_last_hists(self):
        self.all_currs.append(self.currs)
        self.all_targets.append(self.targets)
        self.all_deltas.append(self.deltas)

    def store_collected_hists(self):
        maxlen = len(self.all_currs[0]) 
        util.save_hist_animation(self.all_currs, 100, (-1.2, 1.2), maxlen, "currhist")
        util.save_hist_animation(self.all_targets, 100, (-1.2, 1.2), maxlen, "targethist")
        util.save_hist_animation(self.all_deltas, 100, (-1.2, 1.2), maxlen, "deltahist")
        AC = np.asarray(self.all_currs)
        AT = np.asarray(self.all_targets)
        AD = np.asarray(self.all_deltas)
        util.dump(AC, "currhist")
        util.dump(AT, "targethist")
        util.dump(AD, "deltahist")

    # ---------------- Train FA ---------------

    def learn(self):
        ''' Process the training data collected since last update.
        '''
        self.fa.update()

    def save_model(self, pref=""):
        self.fa.save_model(pref)

    def load_model(self, load_subdir, pref=""):
        self.fa.load_model(load_subdir, pref)
