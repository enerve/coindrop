'''
Created on May 15, 2019

@author: enerve
'''

import logging
import util

import numpy as np

from .player import Player

class SarsaLambdaAgent(Player):
    '''
    An agent that implements Sarsa(λ) to explore and analyze the state-action space.
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
        self.stats_delta = {}
        
        # for histograms
        self.all_currs = []
        self.all_targets = []
        self.all_deltas = []

    def prefix(self):
        es_prefix = self.exploration_strategy.prefix()
        pref = "sarsa_lambda_e%s_l%0.2f" % (es_prefix, self.lam) + self.fa.prefix()
        return pref

    # ---------------- Single game ---------------

    def init_game(self, initial_state, initial_heights):
        self.S = np.copy(initial_state)
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
        self.S = np.copy(new_state)
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
        
    def save_game_for_training(self):
        self.episodes_history.append(self.steps_history)

    def save_game_for_testing(self):
        self.test_episodes_history.append(self.steps_history)

    # ----------- Stats -----------

    def collect_stats(self, ep, num_episodes):
        if (ep+1)% 100 == 0:
            self.logger.debug("SarsaAgent avg R: %d/%d" % (self.sum_total_R, self.sum_played))
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

        delta_list = [deltas for name, deltas in self.stats_delta.items()]
        #deltaArr = np.asarray(delta_list)
        util.plot(delta_list,
                  range(len(delta_list[0])),
                  labels=["agent Δ", "opponent Δ"],
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


    # ----------- Analyze given history to extract training/val data -----------

    def process(self, episodes_history, data_collector, source_name):
        ''' Analyzes given history to create data rows from it '''
        self.deltas = []
        self.targets = []
        self.currs = []
        
        self.num_wins = 0
        self.num_loss = 0
        
        self.logger.debug(" Process history of %d episodes", len(episodes_history))

        for i, steps_history in enumerate(episodes_history):
            self._process_steps(steps_history, data_collector)

        # stats
        npd = np.array(self.deltas)
        if source_name not in self.stats_delta:
            self.stats_delta[source_name] = []
        self.stats_delta[source_name].append(np.mean(np.abs(npd)))
        #self.logger.debug('  sumdelta: %0.4f', self.sum_delta)
        self.logger.debug('  delta   : %0.2f <%0.2f>', np.mean(np.abs(npd)), np.var(npd))
        #self.logger.debug('  wins/losses (total): %d/%d (%d)', self.num_wins,
        #                  self.num_loss, len(episodes_history))

    def _process_steps(self, steps_history, data_collector):
        ''' Observes and learns from the given episode '''
        S, A = None, None
        Q_at_next = 0
        num_E = 0
        self.eligible_states = [None for i in range(self.max_moves)]
        self.eligible_state_target = [0 for i in range(self.max_moves)]
        for i, (R_, S_, A_) in enumerate(steps_history):
            if i > 0:
                if i > 1:
                    # Reuse fa value from the previous iteration.
                    curr_value = Q_at_next
                else:
                    # Unable to reuse. Need to calculate fa value.
                    curr_value = self.fa.value(S, A)

                if S_ is not None:
                    # TODO: speed this up, or parallelize it
                    Q_at_next = self.fa.value(S_, A_)
                else:
                    Q_at_next = 0
                
                # Learn from the reward gotten for action taken last time
                target = R_ + self.gamma * Q_at_next
                
#                 self.recent_target = 0.99 * self.recent_target
#                 self.recent_target += 0.01 * target 

                self.eligible_states[num_E] = (S, A)
                self.eligible_state_target[num_E] = curr_value
                delta = (target - curr_value)
                
                num_E += 1
                for j in range(num_E):
                    self.eligible_state_target[j] += delta * self.eligible_mult[num_E-j-1]

                self.currs.append(curr_value)
                self.targets.append(target)
                self.deltas.append(delta)
                if R_ > 0: self.num_wins += 1 
                if R_ < 0: self.num_loss += 1
                
            S, A = S_, A_
            
        self._record_eligibles(num_E, data_collector)

    def _record_eligibles(self, num_E, data_collector):
        for i in range(num_E):
            S, A = self.eligible_states[i]
            target = self.eligible_state_target[i]
            data_collector.record(S, A, target)

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
        util.save_hist_animation(self.all_currs, 100, (-1.2, 1.2), maxlen, "curr value", "currhist")
        util.save_hist_animation(self.all_targets, 100, (-1.2, 1.2), maxlen, "targets", "targethist")
        util.save_hist_animation(self.all_deltas, 100, (-1.2, 1.2), maxlen, "delta", "deltahist")
        AC = np.asarray(self.all_currs)
        AT = np.asarray(self.all_targets)
        AD = np.asarray(self.all_deltas)
        util.dump(AC, "currhist")
        util.dump(AT, "targethist")
        util.dump(AD, "deltahist")

    # ---------------- Train FA ---------------

    def learn(self, data_collector, validation_data_collector):
        ''' Process the training data collected since last update.
        '''
        self.fa.update(data_collector, validation_data_collector)

    def save_model(self, pref=""):
        self.fa.save_model(pref)

    def load_model(self, load_subdir, pref=""):
        self.fa.load_model(load_subdir, pref)
